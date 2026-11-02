"""纯文本、CSV/TSV、以及源码/配置文件。

三种形态走同一条入口，因为**判断依据是内容与扩展名，而不是用户的选择**：

* ``.txt .text .log .rst`` → 按空行分段，同时尝试识别「第一章 / 1.2 / 一、」这类标题；
* ``.csv .tsv``           → 用 :mod:`csv` 解析成二维数组，交给表格序列化；
* 源码与配置（``.py .json .yaml ...``） → 整份作为**代码块**，保留缩进。

为什么认标题要让位给「宁可漏判」
--------------------------------
纯文本里没有标题语义。``1.2 部署步骤`` 既可能是小节标题，也可能是
「1.2 秒后重试」这种正文开头。这里只在**短行 + 独占一行 + 不与上一行连句**
时才认定为标题；漏判的后果只是一个标题进了正文，错判的后果则是把半句话
劈成新的小节，后者对 ``heading_path`` 的污染严重得多。
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from app.core.errors import IngestError
from app.core.schemas import BlockType
from app.ingestion.loaders.base import (
    BaseLoader,
    LoadResult,
    OrderCounter,
    RawBlock,
    clean_block_text,
    clean_inline_text,
    read_text_file,
)

__all__ = ["TextLoader"]

#: 代码类扩展名 → 围栏语言标记。没有语言标记的围栏在很多 Markdown 渲染器里
#: 不会高亮，更重要的是：切分器要靠它区分「代码」与「散文」。
_CODE_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".sql": "sql",
    ".sh": "bash",
    ".ps1": "powershell",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".xml": "xml",
}

#: 散文类扩展名（走分段 + 标题识别）。
_PROSE_SUFFIXES = frozenset({".txt", ".text", ".log", ".rst"})
_TABLE_SUFFIXES = frozenset({".csv", ".tsv"})

#: 标题行长度上限。超过这个长度的一行几乎不可能是标题。
_MAX_HEADING_CHARS = 60

#: ``第X章`` / ``第X节`` / ``第X部分``
_CN_CHAPTER_RE = re.compile(r"^第[0-9一二三四五六七八九十百千零〇两]+[章节部分篇]\s*\S")
#: ``1.`` / ``1.2`` / ``1.2.3`` 后跟空白
_NUMBERED_RE = re.compile(r"^\d+(?:\.\d+){0,4}[、.]?\s+\S")
#: ``一、`` / ``（一）`` / ``(一)``
_CN_LIST_RE = re.compile(r"^(?:[一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[）)])\s*\S")

#: reST 的标题下划线：符号与层级按出现顺序绑定，不是固定映射。
_RST_UNDERLINE_RE = re.compile(r"^([=\-~^*#\"'`+:_.])\1{2,}$")

#: CSV 单文件行数上限。与 XLSX 保持一致的理由：再大的表是数据导出，不是知识。
_MAX_TABLE_ROWS = 5000
_MAX_TABLE_COLS = 80


class TextLoader(BaseLoader):
    """``.txt`` / ``.csv`` / ``.py`` 等文本类文件。"""

    suffixes = frozenset(_PROSE_SUFFIXES | _TABLE_SUFFIXES | set(_CODE_LANGUAGES))
    kind = "text"
    mime_type = "text/plain"

    def load(self, path: Path) -> LoadResult:
        self._require_file(path)
        suffix = path.suffix.lower()

        result = LoadResult(metadata={"loader": self.kind})

        if suffix in _TABLE_SUFFIXES:
            text, encoding = read_text_file(path, max_bytes=64 << 20)
            result.metadata["encoding"] = encoding
            self._load_table(text, suffix, result)
        elif suffix in _CODE_LANGUAGES:
            text, encoding = read_text_file(path, max_bytes=32 << 20)
            result.metadata["encoding"] = encoding
            result.metadata["language"] = _CODE_LANGUAGES[suffix]
            self._load_code(text, _CODE_LANGUAGES[suffix], result)
        else:
            text, encoding = read_text_file(path, max_bytes=32 << 20)
            result.metadata["encoding"] = encoding
            rst = suffix == ".rst"
            self._load_prose(text, result, rst=rst, counter=OrderCounter())

        result.title = result.title or _fallback_title(path, result)
        if result.is_empty:
            result.warnings.append(f"{path.name} 中没有提取到任何内容")
        return result

    # -- 表格 -----------------------------------------------------------------------

    def _load_table(self, text: str, suffix: str, result: LoadResult) -> None:
        if not text.strip():
            return

        delimiter = "\t" if suffix == ".tsv" else _sniff_delimiter(text)
        rows: list[list[str]] = []
        truncated = False
        try:
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            for index, raw_row in enumerate(reader):
                if index >= _MAX_TABLE_ROWS:
                    truncated = True
                    break
                cells = [clean_inline_text(cell) for cell in raw_row[:_MAX_TABLE_COLS]]
                while cells and not cells[-1]:
                    cells.pop()
                if any(cells):
                    rows.append(cells)
        except csv.Error as exc:
            raise IngestError(f"CSV 解析失败：{exc}", detail={"delimiter": delimiter}) from exc

        if truncated:
            result.warnings.append(f"超过 {_MAX_TABLE_ROWS} 行，已截断")
        if not rows:
            return

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        result.blocks.append(
            RawBlock(
                text="",
                block_type=BlockType.TABLE,
                order=0,
                table=rows,
                source_ref={"loader": self.kind, "delimiter": delimiter},
            )
        )
        result.metadata["delimiter"] = delimiter
        result.metadata["rows"] = len(rows)

    # -- 源码 -----------------------------------------------------------------------

    def _load_code(self, text: str, language: str, result: LoadResult) -> None:
        body = text.strip("\n")
        if not body.strip():
            return
        result.blocks.append(
            RawBlock(
                text=f"```{language}\n{body}\n```",
                block_type=BlockType.CODE,
                order=0,
                source_ref={"loader": self.kind, "language": language},
            )
        )

    # -- 散文 -----------------------------------------------------------------------

    def _load_prose(
        self, text: str, result: LoadResult, *, rst: bool, counter: OrderCounter
    ) -> None:
        paragraphs = _split_paragraphs(text)
        # reST 的层级由下划线符号的**首次出现顺序**决定，而不是符号本身，
        # 所以这里要在一份文档内累积，不能每个段落各算各的。
        rst_levels: dict[str, int] = {}

        for raw in paragraphs:
            lines = [line.strip() for line in raw.split("\n") if line.strip()]
            if not lines:
                continue

            # reST：标题行 + 紧随其后的全符号下划行
            if rst and len(lines) >= 2 and _RST_UNDERLINE_RE.match(lines[1]):
                char = lines[1][0]
                level = rst_levels.setdefault(char, len(rst_levels) + 1)
                title_text = clean_inline_text(lines[0])
                if title_text:
                    result.blocks.append(
                        RawBlock(
                            text=title_text,
                            order=counter.next(),
                            heading_level=min(level, 6),
                            source_ref={"loader": self.kind},
                        )
                    )
                if result.title is None:
                    result.title = title_text
                remainder = lines[2:]
                if remainder:
                    result.blocks.append(
                        RawBlock(
                            text=clean_block_text("\n".join(remainder)),
                            order=counter.next(),
                            source_ref={"loader": self.kind},
                        )
                    )
                continue

            # 单行短句 + 形似编号 → 标题
            if len(lines) == 1:
                level = _heading_level(lines[0])
                if level:
                    result.blocks.append(
                        RawBlock(
                            text=lines[0],
                            order=counter.next(),
                            heading_level=level,
                            source_ref={"loader": self.kind},
                        )
                    )
                    if result.title is None:
                        result.title = lines[0]
                    continue

            body = clean_block_text("\n".join(lines))
            if body:
                result.blocks.append(
                    RawBlock(
                        text=body,
                        order=counter.next(),
                        source_ref={"loader": self.kind},
                    )
                )


def _split_paragraphs(text: str) -> list[str]:
    """按空行切段。

    ``\\n\\s*\\n`` 比 ``\\n\\n`` 稳：Windows 上常见「空行里带几个空格」，
    严格匹配两个换行会把整篇文档粘成一段。
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", normalized)
    return [part for part in parts if part.strip()]


def _heading_level(line: str) -> int:
    """行内标题的层级判断。返回 0 表示「不是标题」。

    行尾带句末标点的一律不算 —— 那说明这是一句话，不是标题。
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return 0
    if stripped[-1] in "。！？.!?；;，,：:":
        return 0
    if _CN_CHAPTER_RE.match(stripped):
        return 1
    if _CN_LIST_RE.match(stripped):
        return 3
    if _NUMBERED_RE.match(stripped):
        # 「3.14 是圆周率」这种句子也会命中前缀，靠「不含句末标点 + 足够短」
        # 挡掉大部分；再按段数给层级，让 1 / 1.2 / 1.2.3 落成三级。
        depth = stripped.split()[0].rstrip("、.").count(".")
        return min(depth + 1, 4)
    return 0


def _sniff_delimiter(text: str) -> str:
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # Sniffer 在两三行的短表上经常失败，而短表恰恰是最常见的手写配置表
        return ","


def _fallback_title(path: Path, result: LoadResult) -> str:
    for block in result.blocks:
        if block.is_heading:
            return block.text
    if result.blocks and result.blocks[0].block_type is BlockType.TEXT:
        # 首段第一行常是文档标题（很多纯文本没有编号）
        first_line = result.blocks[0].text.split("\n", 1)[0]
        if len(first_line) <= 80:
            return clean_inline_text(first_line)
    return path.stem
