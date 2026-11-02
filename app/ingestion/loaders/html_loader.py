"""HTML loader（BeautifulSoup + lxml）。

解析策略：**按文档顺序走一遍 DOM**，只认会产出知识的结构元素。
不转成纯文本再重新找标题 —— 那一步会丢掉「这段字是不是链接/代码」这类
只在标签上存在的信号。

刻意不做的两件事
----------------
1. 不执行 JavaScript。SPA 渲染出来的正文在 HTML 里根本不存在，
   静态解析拿到的只有骨架。这类页面请先另存为 MHTML / PDF 再入库。
2. 不递归抓取 ``<iframe>`` 与远程图片。一次导入触发几十个不受控的网络请求，
   离线环境会直接卡死，而且失败与否不可复现。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.core.errors import IngestError
from app.core.schemas import BlockType
from app.ingestion.loaders.base import (
    BaseLoader,
    LoadResult,
    OrderCounter,
    RawBlock,
    RawImage,
    clean_block_text,
    clean_inline_text,
    read_image_source,
    read_text_file,
)

__all__ = ["HtmlLoader"]

#: 整棵子树都丢掉。``nav`` / ``footer`` / ``aside`` 是站点导航与页脚，
#: 它们会在每个页面重复出现，是检索噪声的头号来源。
_DROP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "nav",
        "footer",
        "aside",
        "form",
        "button",
        "select",
        "option",
        "iframe",
    }
)

#: 这些标签的进入顺序不代表文档顺序，必须跳过它们的子节点再继续。
_SKIP_RECURSE = frozenset({"head", "title", "meta", "link", "base"})

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BLOCK_TAGS = frozenset(
    {"p", "div", "section", "article", "main", "header", "li", "dt", "dd", "blockquote", "figcaption"}
)
_CODE_TAGS = frozenset({"pre", "code"})

_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
#: 连续空行压成一个。段落内部我们已经在 clean_block_text 里处理过空白，
#: 这里只管块与块之间。
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SRC_ATTRS = ("src", "data-src", "data-original", "data-lazy-src")


class HtmlLoader(BaseLoader):
    """单个 ``.html`` / ``.htm`` 文件。"""

    suffixes = frozenset({".html", ".htm", ".xhtml"})
    kind = "html"
    mime_type = "text/html"

    def load(self, path: Path) -> LoadResult:
        self._require_file(path)
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover
            raise IngestError("解析 HTML 需要 beautifulsoup4，请安装项目依赖") from exc
        try:
            import lxml  # noqa: F401 - 只是为了确认解析器可用
        except ImportError as exc:  # pragma: no cover
            raise IngestError("解析 HTML 需要 lxml，请安装项目依赖") from exc

        text, encoding = read_text_file(path)
        try:
            soup = BeautifulSoup(text, "lxml")
        except Exception as exc:  # noqa: BLE001
            raise IngestError(
                f"HTML 解析失败：{exc}", detail={"path": str(path)}
            ) from exc

        for tag in soup.find_all(list(_DROP_TAGS)):
            tag.decompose()

        result = LoadResult(metadata={"loader": self.kind, "encoding": encoding})
        state = _WalkState(counter=OrderCounter(), base_dir=path.parent, result=result)
        _walk(soup.body or soup, state)

        result.page_count = 1
        if not result.title:
            result.title = _title_from_head(soup)
        if result.is_empty:
            result.warnings.append(
                "HTML 中没有提取到正文（页面可能由 JavaScript 渲染，"
                "请先另存为 PDF 或 MHTML 再导入）"
            )
        return result


class _WalkState:
    __slots__ = ("counter", "base_dir", "result")

    def __init__(self, *, counter: OrderCounter, base_dir: Path, result: LoadResult) -> None:
        self.counter = counter
        self.base_dir = base_dir
        self.result = result

    def emit_block(
        self,
        text: str,
        *,
        block_type: BlockType = BlockType.TEXT,
        heading_level: int = 0,
    ) -> None:
        cleaned = _normalize_block(text)
        if not cleaned:
            return
        self.result.blocks.append(
            RawBlock(
                text=cleaned,
                block_type=block_type,
                order=self.counter.next(),
                heading_level=heading_level,
                source_ref={"loader": "html"},
            )
        )

    def emit_image(self, src: str, alt: str = "") -> None:
        found = read_image_source(
            src, self.base_dir, warnings=self.result.warnings, label="HTML 中的"
        )
        if found is None:
            return
        payload, ext = found
        self.result.images.append(
            RawImage(
                data=payload,
                order=self.counter.next(),
                ext=ext,
                source_ref={"loader": "html", "src": src[:200], "alt": alt[:200]},
            )
        )


def _walk(node: Any, state: _WalkState) -> None:
    """深度优先、按文档顺序遍历。"""
    for child in node.children:
        name = getattr(child, "name", None)

        if name is None:
            continue  # NavigableString 已由父块的 get_text 收走

        if name in _DROP_TAGS:
            continue

        if name in _SKIP_RECURSE:
            continue

        if name in _HEADING_TAGS:
            heading_text = _collect_text(child, state, emit_images=False)
            state.emit_block(
                heading_text, heading_level=_HEADING_TAGS[name]
            )
            # 标题里的图片同样要收进来
            _collect_images(child, state)
            continue

        if name == "table":
            rows = _table_rows(child, state)
            if rows:
                state.result.blocks.append(
                    RawBlock(
                        text="",
                        block_type=BlockType.TABLE,
                        order=state.counter.next(),
                        table=rows,
                        source_ref={"loader": "html"},
                    )
                )
            continue

        if name in _CODE_TAGS:
            # <pre> 里的内容保持原样；<code> 单独出现时也当小块代码
            code_text = child.get_text()
            if code_text.strip():
                language = _code_language(child)
                fence = f"```{language}\n" if language else "```\n"
                state.emit_block(
                    f"{fence}{code_text.rstrip()}\n```", block_type=BlockType.CODE
                )
            continue

        if name == "img":
            _emit_img_tag(child, state)
            continue

        if name in _BLOCK_TAGS:
            text = _collect_text(child, state, emit_images=True)
            state.emit_block(text)
            continue

        # 其他标签（<span>、<a>、<b>……）只当作容器继续往下走
        _walk(child, state)


def _collect_text(node: Any, state: _WalkState, *, emit_images: bool) -> str:
    """收集一个块里的可见文本，同时把内联图片登记出去。

    用 ``get_text`` 会在 ``<br>`` 处丢失换行，于是 "第一行<br>第二行" 会粘成
    一行。这里显式遍历，遇到 ``<br>`` 补 ``\\n``。
    """
    parts: list[str] = []

    def visit(element: Any) -> None:
        for child in element.children:
            name = getattr(child, "name", None)
            if name is None:
                parts.append(str(child))
                continue
            if name in _DROP_TAGS:
                continue
            if name == "br":
                parts.append("\n")
                continue
            if name == "img":
                if emit_images:
                    _emit_img_tag(child, state)
                continue
            if name == "pre":
                # 段落里嵌套代码块：保留换行，别的照常
                parts.append(child.get_text())
                continue
            visit(child)

    visit(node)
    return "".join(parts)


def _collect_images(node: Any, state: _WalkState) -> None:
    for img in node.find_all("img"):
        _emit_img_tag(img, state)


def _emit_img_tag(tag: Any, state: _WalkState) -> None:
    src = ""
    for attribute in _SRC_ATTRS:
        value = tag.get(attribute)
        if value:
            src = value
            break
    if not src:
        return
    alt = clean_inline_text(tag.get("alt") or tag.get("title") or "")
    state.emit_image(src, alt)


def _table_rows(table: Any, state: _WalkState) -> list[list[str]]:
    """把 ``<table>`` 摊平成二维字符串数组。

    保留 ``rowspan`` / ``colspan`` 的半程语义：被跨过的格子填 ``""``，
    这样列数仍然对齐。真正合并单元格的还原需要完整的表格模型，
    对检索来说不值得 —— 空串已经能让模型看出「这一列在这一行不适用」。
    """
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue  # 嵌套表格已由外层单元格文本吸收
        cells: list[str] = []
        for cell in tr.find_all(["th", "td"], recursive=False):
            text = clean_inline_text(_collect_text(cell, state, emit_images=True))
            cells.append(text)
            try:
                colspan = int(cell.get("colspan") or 1)
            except (TypeError, ValueError):
                colspan = 1
            cells.extend([""] * max(0, colspan - 1))
        if any(cells):
            rows.append(cells)
    if not rows:
        return []
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def _code_language(node: Any) -> str:
    for attribute in ("data-lang", "data-language", "lang"):
        value = node.get(attribute)
        if value:
            return str(value).strip().split()[0]
    classes = node.get("class") or []
    for name in classes:
        text = str(name)
        for prefix in ("language-", "lang-"):
            if text.startswith(prefix):
                return text[len(prefix) :]
    return ""


def _normalize_block(text: str) -> str:
    """HTML 常见的是「一行里塞满缩进与空格」，这里先把行内空白压平。"""
    if not text:
        return ""
    flattened = "\n".join(
        _WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")
    )
    flattened = _BLANK_LINES_RE.sub("\n\n", flattened)
    return clean_block_text(flattened)


def _title_from_head(soup: Any) -> str:
    if soup.title is not None:
        text = clean_inline_text(soup.title.get_text())
        if text:
            return text

    for attribute, value in (("property", "og:title"), ("name", "twitter:title")):
        tag = soup.find("meta", attrs={attribute: value})
        if tag is not None:
            content = clean_inline_text(tag.get("content") or "")
            if content:
                return content

    for level in range(1, 7):
        tag = soup.find(f"h{level}")
        if tag is not None:
            text = clean_inline_text(tag.get_text())
            if text:
                return text
    return ""
