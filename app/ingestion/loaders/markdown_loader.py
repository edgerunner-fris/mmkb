"""Markdown loader（markdown-it-py）。

为什么用 markdown-it 而不是正则
------------------------------
``#`` 开头的行不一定是标题（代码块里就有），``|`` 分隔的行不一定是表格
（横向规则、转义、行内代码里都会出现），``![]()`` 也不一定在正文（链接文本里
可能有一对中括号）。直接正则切会在真实文档上各类误判，而 markdown-it 给的
是**已经消歧过的 token 流** —— 代价只是一个纯 Python 依赖，没有外部服务。

支持的块：ATX 标题、段落、围栏代码、GFM 表格、有序/无序列表、图片。
不支持的块：脚注、数学公式、mermaid。它们会以普通文本形式留在正文里 ——
这样至少还能被 BM25 命中，比直接丢掉强。
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

__all__ = ["MarkdownLoader"]

#: YAML front matter。只在文件最开头、且以单独一行 ``---`` 结束时才算。
#: 行尾用 ``\r?\n``：Windows 上的 Markdown 基本都是 CRLF，只认 ``\n`` 的话
#: 整段 front matter 会留在正文里，被 CommonMark 当成 setext 标题（``---``
#: 正好是二级标题的下划线），标题与第一个分块都会变成 ``title: ...``。
_FRONT_MATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(?P<body>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL
)
_HTML_IMG_RE = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*(?P<q>[\"'])(?P<src>.*?)(?P=q)", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


class MarkdownLoader(BaseLoader):
    """单个 ``.md`` / ``.markdown`` 文件。"""

    suffixes = frozenset({".md", ".markdown"})
    kind = "markdown"
    mime_type = "text/markdown"

    def load(self, path: Path) -> LoadResult:
        self._require_file(path)
        try:
            from markdown_it import MarkdownIt
        except ImportError as exc:  # pragma: no cover
            raise IngestError(
                "解析 Markdown 需要 markdown-it-py，请安装项目依赖"
            ) from exc

        text, encoding = read_text_file(path)
        front_matter, title, body_text = self._split_front_matter(text)

        parser = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
        try:
            tokens = parser.parse(body_text)
        except Exception as exc:  # noqa: BLE001 - 畸形 Markdown 不该让导入失败
            raise IngestError(
                f"Markdown 解析失败：{exc}", detail={"path": str(path)}
            ) from exc

        result = LoadResult(title=title, metadata={"loader": self.kind, "encoding": encoding})
        if front_matter:
            result.metadata["front_matter"] = front_matter

        state = _WalkState(counter=OrderCounter(), base_dir=path.parent, result=result)
        _walk(tokens, state)

        if not result.title:
            result.title = _first_heading(result)
        if result.is_empty:
            result.warnings.append("Markdown 中没有提取到任何内容")
        return result

    @staticmethod
    def _split_front_matter(text: str) -> tuple[dict[str, str], str, str]:
        """剥掉 YAML front matter，取出 ``title``。

        刻意不引入 PyYAML：这里只需要 ``key: value`` 这种最简单的形式，
        而 front matter 里的嵌套结构对检索没有价值。
        """
        match = _FRONT_MATTER_RE.match(text)
        if match is None:
            return {}, "", text

        body = match.group("body")
        meta: dict[str, str] = {}
        for line in body.splitlines():
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("'\"")
        return meta, clean_inline_text(meta.get("title", "")), text[match.end() :]


class _WalkState:
    """把 token 流摊平时需要的可变状态，集中在一个对象里。"""

    __slots__ = (
        "counter",
        "base_dir",
        "result",
        "heading_level",
        "row",
        "row_is_header",
        "table_rows",
        "header_rows",
        "list_stack",
        "pending_prefix",
        "table_mode",
    )

    def __init__(self, *, counter: OrderCounter, base_dir: Path, result: LoadResult) -> None:
        self.counter = counter
        self.base_dir = base_dir
        self.result = result
        self.heading_level = 0
        self.row: list[str] | None = None
        self.row_is_header = False
        self.table_rows: list[list[str]] = []
        self.header_rows: list[list[str]] = []
        self.list_stack: list[dict[str, Any]] = []
        self.pending_prefix = ""
        self.table_mode = False

    def emit_block(
        self,
        text: str,
        *,
        block_type: BlockType = BlockType.TEXT,
        heading_level: int = 0,
    ) -> None:
        cleaned = clean_block_text(text)
        if not cleaned:
            return
        self.result.blocks.append(
            RawBlock(
                text=cleaned,
                block_type=block_type,
                order=self.counter.next(),
                heading_level=heading_level,
                source_ref={"loader": "markdown"},
            )
        )

    def emit_image(self, src: str, *, label: str = "") -> None:
        found = read_image_source(
            src, self.base_dir, warnings=self.result.warnings, label=label
        )
        if found is None:
            return
        payload, ext = found
        self.result.images.append(
            RawImage(
                data=payload,
                order=self.counter.next(),
                ext=ext,
                source_ref={"loader": "markdown", "src": src[:200]},
            )
        )


def _walk(tokens: list[Any], state: _WalkState) -> None:
    for token in tokens:
        kind = token.type

        if kind == "heading_open":
            state.heading_level = _tag_level(token.tag)

        elif kind == "heading_close":
            state.heading_level = 0

        elif kind == "inline":
            text = _inline_text(token, state)
            if state.table_mode:
                if state.row is not None:
                    state.row.append(clean_inline_text(text))
            elif state.heading_level:
                state.emit_block(
                    text, heading_level=state.heading_level
                )
            else:
                prefix = state.pending_prefix
                state.pending_prefix = ""
                state.emit_block(f"{prefix}{text}" if prefix else text)

        elif kind == "fence":
            info = (token.info or "").strip().split()[0] if token.info else ""
            language = f"{info}\n" if info else ""
            state.emit_block(
                f"```{language}{token.content.rstrip()}\n```",
                block_type=BlockType.CODE,
            )

        elif kind == "code_block":
            state.emit_block(f"```\n{token.content.rstrip()}\n```", block_type=BlockType.CODE)

        elif kind == "table_open":
            state.table_mode = True
            state.table_rows = []

        elif kind == "table_close":
            state.table_mode = False
            rows = state.table_rows
            state.table_rows = []
            if rows:
                state.result.blocks.append(
                    RawBlock(
                        text="",
                        block_type=BlockType.TABLE,
                        order=state.counter.next(),
                        table=rows,
                        source_ref={"loader": "markdown"},
                    )
                )

        elif kind == "tr_open":
            state.row = []

        elif kind == "tr_close":
            if state.row is not None and any(cell for cell in state.row):
                state.table_rows.append(state.row)
            state.row = None

        elif kind in ("th_open", "td_open"):
            state.row_is_header = kind == "th_open"

        elif kind in ("bullet_list_open", "ordered_list_open"):
            state.list_stack.append(
                {
                    "ordered": kind == "ordered_list_open",
                    "index": _starting_index(token),
                }
            )

        elif kind in ("bullet_list_close", "ordered_list_close"):
            if state.list_stack:
                state.list_stack.pop()

        elif kind == "list_item_open":
            if state.list_stack:
                current = state.list_stack[-1]
                if current["ordered"]:
                    current["index"] += 1
                    state.pending_prefix = f"{current['index']}. "
                else:
                    state.pending_prefix = "- "

        elif kind == "html_block":
            for match in _HTML_IMG_RE.finditer(token.content or ""):
                state.emit_image(match.group("src"), label="HTML 块中的")
            stripped = _TAG_RE.sub("", token.content or "")
            state.emit_block(stripped)

        elif kind == "html_inline":
            for match in _HTML_IMG_RE.finditer(token.content or ""):
                state.emit_image(match.group("src"), label="HTML 行内的")


def _inline_text(token: Any, state: _WalkState) -> str:
    """把 inline token 的子节点还原成纯文本，顺带取出其中的图片。"""
    if not token.children:
        return token.content or ""

    parts: list[str] = []
    for child in token.children:
        child_type = child.type
        if child_type == "text":
            parts.append(child.content)
        elif child_type == "code_inline":
            parts.append(child.content)
        elif child_type in ("softbreak", "hardbreak"):
            parts.append("\n")
        elif child_type == "image":
            src = _attr(child, "src")
            alt = child.content or ""
            if src:
                state.emit_image(src, label="正文中的")
            if alt.strip():
                # 保留 alt 作为文字线索：图片本身进不了文本检索，
                # 但作者写在 alt 里的描述是最廉价、最准确的语义信息。
                parts.append(f"[图片: {clean_inline_text(alt)}]")
        elif child_type == "html_inline":
            for match in _HTML_IMG_RE.finditer(child.content or ""):
                state.emit_image(match.group("src"), label="HTML 行内的")
            parts.append(_TAG_RE.sub("", child.content or ""))
        elif child_type == "emoji":
            parts.append(child.content)
    return "".join(parts)


def _attr(token: Any, name: str) -> str:
    """兼容 markdown-it-py 2.x（attrs 是 list[tuple]）与 3.x（attrs 是 dict）。"""
    attrs = getattr(token, "attrs", None)
    if not attrs:
        return ""
    if isinstance(attrs, dict):
        return str(attrs.get(name) or "")
    for key, value in attrs:
        if key == name:
            return str(value or "")
    return ""


def _tag_level(tag: str) -> int:
    if tag and tag[0] == "h" and tag[1:].isdigit():
        return max(1, min(6, int(tag[1:])))
    return 1


def _starting_index(token: Any) -> int:
    raw = _attr(token, "start")
    try:
        return int(raw) - 1 if raw else 0
    except ValueError:
        return 0


def _first_heading(result: LoadResult) -> str:
    for block in result.blocks:
        if block.is_heading:
            return block.text
    return ""
