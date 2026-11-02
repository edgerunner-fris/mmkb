"""表格序列化。

所有格式（PDF 的 ``find_tables``、DOCX 的 ``Table``、XLSX 的工作表、
HTML 的 ``<table>``、Markdown 的 GFM 表）最终都归一成
``list[list[str]]``，再在这里变成一段**可以原样喂给模型**的 Markdown。

为什么是 Markdown 而不是 CSV 或 JSON
------------------------------------
* CSV：分隔符与单元格内容会打架，中文文档里的逗号、分号遍地都是；
* JSON：每个单元格都要重复一遍列名，token 成本翻倍以上；
* Markdown：列名只出现一次，模型对它的理解又足够好，还能被 BM25
  按「列名 + 单元格」这种组合命中。
"""

from __future__ import annotations

from app.core.tokens import DEFAULT_ENCODING, count_tokens

__all__ = [
    "split_table_markdown",
    "table_header_markdown",
    "to_markdown",
]

#: 单元格内容的硬上限（字符）。极长的单元格（例如一整段文本被塞进一个格子）
#: 会把整行撑爆，导致无论怎么按行切都超预算。
_MAX_CELL_CHARS = 500


def to_markdown(rows: list[list[str]]) -> str:
    """二维数组 → Markdown 表格。

    * 第一行当表头（GFM 语法要求必须有表头，没有的话渲染器会把整个表当普通文本）；
    * 参差不齐的行按最长行补空串，否则列会错位；
    * 单元格里的 ``|`` 转义成 ``\\|``，换行转成 ``<br>`` —— 这两条是 Markdown
      表格里唯一会破坏结构的字符。
    """
    normalized = _normalize_rows(rows)
    if not normalized:
        return ""

    header = normalized[0]
    body = normalized[1:]
    lines = [_row_line(header), _separator_line(len(header))]
    lines.extend(_row_line(row) for row in body)
    return "\n".join(lines)


def table_header_markdown(rows: list[list[str]]) -> str:
    """只要表头两行（列名 + 分隔线）。切分时用于给后续片段补表头。"""
    normalized = _normalize_rows(rows)
    if not normalized:
        return ""
    return "\n".join([_row_line(normalized[0]), _separator_line(len(normalized[0]))])


def split_table_markdown(
    rows: list[list[str]],
    *,
    max_tokens: int,
    repeat_header: bool = True,
    encoding: str = DEFAULT_ENCODING,
) -> list[str]:
    """把大表按行切成若干段 Markdown，每段不超过 ``max_tokens``。

    **按行切而不是按字符切**：表格的语义单位是行，从中间截断一行会造出
    半条记录 —— 它看起来仍然像数据，模型却会把它当成另一条记录来引用。

    ``repeat_header=True`` 时每段都补上表头。多花的那点 token 换来的，
    是「第 700 行的那个 3.5 到底是准确率还是召回率」这个问题的答案：
    没有表头的片段里，列的含义已经丢了。
    """
    normalized = _normalize_rows(rows)
    if not normalized:
        return []

    header_rows = normalized[:1]
    body_rows = normalized[1:]
    header_md = table_header_markdown(header_rows)
    header_tokens = count_tokens(header_md, encoding)

    if not body_rows:
        return [header_md]

    # 单行就超预算的极端情况：先压单元格，再截断整行。
    budget = max(max_tokens - header_tokens, max_tokens // 4)
    limited = [_shrink_row(row, budget // 2, encoding) for row in body_rows]

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for row in limited:
        line = _row_line(row)
        line_tokens = count_tokens(line, encoding) + 1  # +1 近似补上换行边界

        if current and current_tokens + line_tokens > budget:
            pieces.append(_assemble(header_md, current, include_header=repeat_header))
            current = []
            current_tokens = 0

        current.append(line)
        current_tokens += line_tokens

    if current:
        pieces.append(_assemble(header_md, current, include_header=repeat_header))

    # 第一段永远带表头（哪怕 repeat_header=False），否则它连「是个表」都不成立
    if not repeat_header and pieces:
        pieces[0] = _assemble(header_md, pieces[0].split("\n"), include_header=True)
    return pieces


def _assemble(header_md: str, lines: list[str], *, include_header: bool) -> str:
    if include_header:
        return "\n".join([header_md, *lines])
    return "\n".join(lines)


def _normalize_rows(rows: list[list[str]]) -> list[list[str]]:
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [_cell_text(cell) for cell in row]
        while cells and not cells[-1]:
            cells.pop()
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return []
    width = max(len(row) for row in cleaned)
    return [row + [""] * (width - len(row)) for row in cleaned]


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > _MAX_CELL_CHARS:
        text = text[:_MAX_CELL_CHARS] + "…"
    # 管道符必须先转义，否则会被当成列分隔；换行不能直接留在单元格里
    return text.replace("|", "\\|").replace("\n", "<br>")


def _row_line(row: list[str]) -> str:
    return "| " + " | ".join(row) + " |"


def _separator_line(width: int) -> str:
    return "| " + " | ".join(["---"] * width) + " |"


def _shrink_row(row: list[str], max_tokens: int, encoding: str) -> list[str]:
    """把一行压到预算内：保持列数不变，等比例截断各单元格。"""
    line = _row_line(row)
    if count_tokens(line, encoding) <= max_tokens:
        return row
    if not row:
        return row

    # 每列至少留 8 个字符，避免压成全是省略号
    per_cell = max(8, (max_tokens * 4) // max(len(row), 1))
    shrunk = [
        (cell[:per_cell] + "…") if len(cell) > per_cell else cell for cell in row
    ]
    return shrunk
