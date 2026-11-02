"""切分：``ParsedDocument`` → ``Chunk`` 列表。

切分质量直接决定检索上限 —— 一个把标题和正文切散的块，再好的 reranker
也救不回来。这里的几条规则都是围绕这一点定的：

**1. 标题是不可跨越的边界。**
   两个相邻段落即使都只有 80 token 也不跨小节合并：合并之后这段文字
   同时属于两个主题，``heading_path`` 只能写一个，另一个就丢了。

**2. 表格与代码默认整体保留。**
   ``| 准确率 | 0.91 |`` 从中间截断后仍然长得像数据，模型会把它当成另一条
   记录来引用。只有超过 ``TABLE_MAX_TOKENS`` 时才按行切，且每段重附表头。

**3. 块文本前面带章节路径。**
   ``"第三章 架构 > 3.2 检索"`` 既让 embedding 知道语境，也让 BM25 能在
   正文没提章节名时仍然命中。代价是每个块多花十几个 token。

**4. 块 ID 由 ``(doc_id, ordinal)`` 派生，而不是随机生成。**
   重跑导入时同一份文档得到同一批 ID，向量库的 upsert 才是真幂等；
   随机 ID 会让每次重跑都留下一批再也删不掉的孤儿向量。

**5. 图片 caption 也是 Chunk。**
   它进同一个文本向量库与 BM25 语料，因此图片能被「文字」检索命中；
   图片本身另有多模态向量，用于「以文搜图 / 以图搜图」。两条路互不替代。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.config import ChunkingSettings
from app.core.ids import stable_id
from app.core.schemas import BlockType, Chunk, ImageAsset
from app.core.tokens import count_tokens
from app.ingestion.parser import Block, ParsedDocument
from app.ingestion.table_serializer import split_table_markdown, to_markdown

__all__ = ["ChunkOutcome", "chunk_document"]

#: 句子边界。保留分隔符本身（``(?<=...)`` 零宽断言），否则中文句子会丢掉句号。
_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*")


@dataclass(slots=True)
class ChunkOutcome:
    chunks: list[Chunk] = field(default_factory=list)
    #: ``block.order`` → ``chunk_id``。图片正是靠它把 ``text_chunk_id`` 补上。
    block_to_chunk: dict[int, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _Unit:
    """切分的最小单位：一段不可再分的文本 + 它来自哪个块。"""

    text: str
    tokens: int
    block: Block
    kind: BlockType


@dataclass(slots=True)
class _Piece:
    """打包结果：若干单位 + 从上一块继承的重叠前缀。"""

    units: list[_Unit]
    overlap: str = ""

    @property
    def tokens(self) -> int:
        return sum(unit.tokens for unit in self.units)

    @property
    def primary_type(self) -> BlockType:
        """一个块只能有一个 ``block_type``，混合内容按「信息量最大的」定。

        表格 > 代码 > 文本：前端拿到类型后决定怎么渲染，把带表格的块标成
        ``text`` 会让表格在引用里退化成一堆竖线。
        """
        kinds = {unit.kind for unit in self.units}
        if BlockType.TABLE in kinds:
            return BlockType.TABLE
        if BlockType.CODE in kinds:
            return BlockType.CODE
        return BlockType.TEXT

    def body(self) -> str:
        parts: list[str] = []
        if self.overlap:
            parts.append(self.overlap)
        previous: _Unit | None = None
        for unit in self.units:
            if previous is not None:
                # 同一块内部的换行是原文的换行（列表项、软换行），
                # 保留单换行；跨块才用空行分段。
                same_block = unit.block is previous.block
                both_text = unit.kind is BlockType.TEXT and previous.kind is BlockType.TEXT
                parts.append("\n" if same_block and both_text else "\n\n")
            parts.append(unit.text)
            previous = unit
        return "".join(parts)


def chunk_document(
    doc: ParsedDocument,
    *,
    settings: ChunkingSettings,
    keep_image_ids: set[str] | None = None,
    caption_images: Sequence[ImageAsset] = (),
) -> ChunkOutcome:
    """切分一份文档。

    ``keep_image_ids``：经过过滤/去重后**真正保留**的图片 ID。块上的
    ``image_ids`` 会被限制在这个集合内 —— 否则引用里会出现一张已经被丢弃、
    点开就 404 的图。

    ``caption_images``：需要额外生成 caption 分块的图片（通常就是所有保留图）。
    """
    encoding = settings.encoding
    outcome = ChunkOutcome()
    ordinal = 0
    dropped = 0

    for section in _group_sections(doc.blocks):
        heading_path = section[0].heading_path
        units: list[_Unit] = []
        for block in section:
            units.extend(_units_for_block(block, settings, encoding))
        if not units:
            continue

        pieces = _merge_tail(_pack(units, settings, encoding), settings)
        for piece in pieces:
            body = piece.body()
            if not body.strip():
                continue
            # 极短碎片（页码、残句、单个标签）只会在召回里占据名额
            if (
                len(body) < settings.min_chunk_chars
                and piece.primary_type is BlockType.TEXT
                and not any(unit.block.image_ids for unit in piece.units)
            ):
                dropped += 1
                continue

            breadcrumb = " > ".join(heading_path)
            text = f"{breadcrumb}\n\n{body}" if breadcrumb else body
            image_ids = _collect_image_ids(piece, keep_image_ids)
            pages = [unit.block.page for unit in piece.units if unit.block.page is not None]

            chunk = Chunk(
                chunk_id=stable_id("chunk", doc.doc_id, str(ordinal)),
                kb_id=doc.kb_id,
                doc_id=doc.doc_id,
                ordinal=ordinal,
                text=text,
                token_count=count_tokens(text, encoding),
                block_type=piece.primary_type,
                heading_path=list(heading_path),
                page_from=min(pages) if pages else None,
                page_to=max(pages) if pages else None,
                image_ids=image_ids,
                source_ref={
                    "synthetic": False,
                    "block_orders": sorted({unit.block.order for unit in piece.units}),
                    "block_ids": sorted({unit.block.block_id for unit in piece.units}),
                    "block_types": sorted({str(unit.kind) for unit in piece.units}),
                },
            )
            outcome.chunks.append(chunk)
            for unit in piece.units:
                outcome.block_to_chunk[unit.block.order] = chunk.chunk_id
            ordinal += 1

    if dropped:
        outcome.warnings.append(f"丢弃了 {dropped} 个过短的碎片（< {settings.min_chunk_chars} 字符）")

    for image in caption_images:
        text = (image.caption_text or "").strip()
        if not text:
            continue
        chunk = Chunk(
            chunk_id=stable_id("chunk", doc.doc_id, str(ordinal)),
            kb_id=doc.kb_id,
            doc_id=doc.doc_id,
            ordinal=ordinal,
            text=text,
            token_count=count_tokens(text, encoding),
            block_type=BlockType.IMAGE_CAPTION,
            heading_path=list(image.heading_path),
            page_from=image.page,
            page_to=image.page,
            image_ids=[image.image_id],
            source_ref={
                "synthetic": True,
                "image_id": image.image_id,
                "caption_source": str(image.caption_source),
            },
        )
        outcome.chunks.append(chunk)
        ordinal += 1

    _link_neighbors(outcome.chunks)
    return outcome


# --------------------------------------------------------------------------------------
# 分组与单位
# --------------------------------------------------------------------------------------


def _group_sections(blocks: list[Block]) -> list[list[Block]]:
    """按 ``heading_path`` 的连续变化切段。

    只合并**相邻且路径相同**的块。路径相同但不连续（例如两章之间回到父章节
    又进来）属于两个不同的小节，不能合并 —— 合并会让中间那段顺序错乱。
    """
    sections: list[list[Block]] = []
    for block in blocks:
        if sections and sections[-1][-1].heading_path == block.heading_path:
            sections[-1].append(block)
        else:
            sections.append([block])
    return sections


def _units_for_block(
    block: Block, settings: ChunkingSettings, encoding: str
) -> list[_Unit]:
    if block.block_type is BlockType.TABLE and block.table:
        markdown = block.text or to_markdown(block.table)
        limit = settings.table_max_tokens if settings.keep_table_atomic else settings.max_tokens
        if count_tokens(markdown, encoding) <= limit:
            return [_Unit(markdown, count_tokens(markdown, encoding), block, BlockType.TABLE)]
        pieces = split_table_markdown(
            block.table,
            max_tokens=limit,
            repeat_header=settings.table_repeat_header,
            encoding=encoding,
        )
        return [
            _Unit(piece, count_tokens(piece, encoding), block, BlockType.TABLE)
            for piece in pieces
            if piece.strip()
        ]

    if block.block_type is BlockType.CODE:
        return _code_units(block, settings.max_tokens, encoding)

    return _text_units(block, settings.max_tokens, encoding)


def _text_units(block: Block, max_tokens: int, encoding: str) -> list[_Unit]:
    """文本块 → 行/句级单位。

    先按行切（保留作者的分段），单行超预算再按句切，单句仍超预算才硬切字符。
    三级降级里前两级都尊重原文结构，只有最后一级是「实在没办法」。
    """
    units: list[_Unit] = []
    for line in block.text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = count_tokens(stripped, encoding)
        if tokens <= max_tokens:
            units.append(_Unit(stripped, tokens, block, BlockType.TEXT))
            continue

        for sentence in _SENTENCE_RE.split(stripped):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = count_tokens(sentence, encoding)
            if sentence_tokens <= max_tokens:
                units.append(_Unit(sentence, sentence_tokens, block, BlockType.TEXT))
                continue
            for piece in _hard_split(sentence, max_tokens, encoding):
                units.append(
                    _Unit(piece, count_tokens(piece, encoding), block, BlockType.TEXT)
                )
    return units


def _code_units(block: Block, max_tokens: int, encoding: str) -> list[_Unit]:
    """代码块 → 带围栏的若干段。围栏必须在每段上重新闭合。"""
    if not block.text.strip():
        return []
    lines = block.text.split("\n")
    fence = "```"
    if lines and lines[0].startswith("```"):
        fence = lines[0]
        if lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
    elif lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    budget = max(16, max_tokens - count_tokens(fence + "\n```", encoding))

    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        line_tokens = count_tokens(line + "\n", encoding)
        if current and current_tokens + line_tokens > budget:
            groups.append(current)
            current = []
            current_tokens = 0
        if not current and line_tokens > budget:
            for piece in _hard_split(line, budget, encoding):
                groups.append([piece])
            continue
        current.append(line)
        current_tokens += line_tokens
    if current:
        groups.append(current)

    units: list[_Unit] = []
    for group in groups:
        body = f"{fence}\n" + "\n".join(group) + "\n```"
        units.append(_Unit(body, count_tokens(body, encoding), block, BlockType.CODE))
    return units


def _hard_split(text: str, max_tokens: int, encoding: str) -> list[str]:
    """按 token 上限硬切。二分出尽量长的前缀，避免逐字符累加。"""
    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        low, high = start + 1, length
        best = start + 1
        while low <= high:
            mid = (low + high) // 2
            if count_tokens(text[start:mid], encoding) <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        pieces.append(text[start:best])
        start = best
    return pieces


# --------------------------------------------------------------------------------------
# 打包
# --------------------------------------------------------------------------------------


def _pack(units: list[_Unit], settings: ChunkingSettings, encoding: str) -> list[_Piece]:
    pieces: list[_Piece] = []
    current: list[_Unit] = []
    current_tokens = 0
    current_overlap = ""

    def flush() -> None:
        nonlocal current, current_tokens, current_overlap
        if not current:
            return
        pieces.append(_Piece(current, current_overlap))
        current = []
        current_tokens = 0
        current_overlap = ""

    for unit in units:
        # 超预算的原子单位（大表格）独占一块。它已经无法再切 ——
        # 再切就会把一个表格拆成两条互不相干的记录。
        if unit.tokens > settings.max_tokens:
            flush()
            pieces.append(_Piece([unit]))
            continue

        if current and current_tokens + unit.tokens > settings.max_tokens:
            flush()

        if not current and pieces and settings.overlap_tokens > 0:
            previous = pieces[-1]
            if previous.units and previous.units[-1].kind is BlockType.TEXT and unit.kind is BlockType.TEXT:
                candidate = _tail_tokens(previous.body(), settings.overlap_tokens, encoding)
                candidate_tokens = count_tokens(candidate, encoding) if candidate else 0
                if candidate_tokens and candidate_tokens + unit.tokens <= settings.max_tokens:
                    current_overlap = candidate
                    current_tokens = candidate_tokens

        current.append(unit)
        current_tokens += unit.tokens

    flush()
    return pieces


def _tail_tokens(text: str, max_tokens: int, encoding: str) -> str:
    """取文本尾部不超过 ``max_tokens`` 的一段，优先从换行处开始。"""
    if not text or max_tokens <= 0:
        return ""
    if count_tokens(text, encoding) <= max_tokens:
        return text

    low, high = 0, len(text)
    best = len(text)
    # 找最小的 start，使 text[start:] 不超过预算
    while low <= high:
        mid = (low + high) // 2
        if count_tokens(text[mid:], encoding) <= max_tokens:
            best = mid
            high = mid - 1
        else:
            low = mid + 1

    tail = text[best:]
    newline = tail.find("\n")
    if 0 <= newline < len(tail) // 3:
        tail = tail[newline + 1 :]
    return tail.strip()


def _merge_tail(pieces: list[_Piece], settings: ChunkingSettings) -> list[_Piece]:
    """把最后一小段并回前一块。

    ``min_tokens`` 不是硬性下限 —— 贪婪打包本身就会尽量装满，唯一可能
    偏小的是每节的最后一块。为它单独放宽预算（不超过 ``max_tokens``）是
    划算的：一个 60 token 的孤块在 top-k 里几乎从不被选中，却占着一个名额。
    """
    if not settings.merge_small_siblings or len(pieces) < 2:
        return pieces
    last = pieces[-1]
    if last.tokens >= settings.min_tokens:
        return pieces

    previous = pieces[-2]
    # 合并时丢弃 last 的重叠前缀：它本来就是 previous 的尾巴，留着会重复
    merged_tokens = previous.tokens + last.tokens
    if merged_tokens > settings.max_tokens:
        return pieces

    merged = _Piece(previous.units + last.units, previous.overlap)
    return [*pieces[:-2], merged]


def _collect_image_ids(piece: _Piece, keep: set[str] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for unit in piece.units:
        for image_id in unit.block.image_ids:
            if image_id in seen:
                continue
            if keep is not None and image_id not in keep:
                continue
            seen.add(image_id)
            ordered.append(image_id)
    return ordered


def _link_neighbors(chunks: list[Chunk]) -> None:
    """补上前后指针（跳过 caption 分块）。

    caption 分块在序上排在所有正文之后，与正文没有阅读顺序上的相邻关系。
    把它们串进链表会让「展开相邻块」跳出章节。
    """
    body = [chunk for chunk in chunks if chunk.block_type is not BlockType.IMAGE_CAPTION]
    for index, chunk in enumerate(body):
        chunk.prev_chunk_id = body[index - 1].chunk_id if index > 0 else None
        chunk.next_chunk_id = body[index + 1].chunk_id if index + 1 < len(body) else None
