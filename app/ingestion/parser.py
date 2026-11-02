"""解析结果归一：``LoadResult`` → ``ParsedDocument``。

这一步做三件事，都是**所有格式共有的**，所以不能留给各自的 loader：

1. **标题栈**：把 ``heading_level`` 流还原成每个块的 ``heading_path``
   （``["第三章 系统设计", "3.2 检索模块"]``）。它是检索时的过滤条件、
   重排时的上下文补充、以及前端引用的定位信息。
2. **表格序列化**：``table: list[list[str]]`` → Markdown，块类型固定为
   :attr:`BlockType.TABLE`。之后所有格式的表格在切分器眼里长得一模一样。
3. **图片锚定**：块与图片共享同一个 ``order`` 计数器（见 loader 基类的说明），
   因此「这张图上面最近的一段正文」是可以确定的。锚点决定了两件事：
   图片的 ``heading_path``（它属于哪一节），以及送给视觉模型的
   ``context_text``（这张图在讲什么）。

图片 ID 在这里就定下来
----------------------
``image_id = sha1(kb_id + ":" + sha256)`` 的前 12 位。**必须带上 kb_id**：
``image_id`` 是数据库主键，只按内容哈希的话，同一张图导入两个知识库时
后一个会覆盖前一个 —— 表现是「A 库的图在 B 库里被引用，删掉 B 库的文档
导致 A 库的图 404」。这个错误不会在写入时报，只会在某天图片突然裂开。
同一文档内的重复图片按 sha256 去重（只保留首次出现）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import BlockType
from app.ingestion.loaders.base import LoadResult, RawBlock, RawImage
from app.ingestion.table_serializer import to_markdown

__all__ = [
    "Block",
    "ParsedDocument",
    "ParsedImage",
    "image_id_for",
    "parse_document",
]

#: 送给视觉模型的上下文长度。取「约 200–300 个汉字」的量级：
#: 再短说不清图在讲什么，再长会把整页正文塞进 prompt，成本翻倍而信息量不变。
DEFAULT_CONTEXT_CHARS = 800


@dataclass(slots=True)
class Block:
    """一段正文（文本/代码/表格），已绑定标题路径。

    ``block_id`` 只在单份文档内唯一（``b{order}``），用于图片锚定，
    **不落库**。真正落库的 ID 由切分器按 ``doc_id + 序号`` 派生。
    """

    block_id: str
    order: int
    ordinal: int
    text: str
    block_type: BlockType
    heading_path: list[str] = field(default_factory=list)
    page: int | None = None
    source_ref: dict[str, Any] = field(default_factory=dict)
    image_ids: list[str] = field(default_factory=list)
    #: 表格的原始二维数组。``text`` 已经是 Markdown，但切分大表时需要按行重排，
    #: 从 Markdown 反解析回来既易错又多一次转换，所以原样留一份。
    table: list[list[str]] | None = None

    @property
    def heading_text(self) -> str:
        return " > ".join(self.heading_path)


@dataclass(slots=True)
class ParsedImage:
    """一张已定位的图片。``data`` 仍留在内存里，由后续步骤决定落盘还是丢弃。"""

    image_id: str
    order: int
    data: bytes
    ext: str
    mime_type: str
    sha256: str
    page: int | None = None
    bbox: list[float] | None = None
    heading_path: list[str] = field(default_factory=list)
    anchor_block_id: str | None = None
    anchor_order: int | None = None
    context_text: str = ""
    source_ref: dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass(slots=True)
class ParsedDocument:
    doc_id: str
    kb_id: str
    title: str
    blocks: list[Block]
    images: list[ParsedImage]
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        return bool(self.blocks or self.images)

    @property
    def text_chars(self) -> int:
        return sum(len(block.text) for block in self.blocks)

    @property
    def heading_paths(self) -> list[list[str]]:
        return [block.heading_path for block in self.blocks if block.heading_path]


def image_id_for(kb_id: str, sha256: str) -> str:
    """图片主键：``kb_id`` + 内容哈希。见模块 docstring 的事故说明。"""
    return stable_id("image", kb_id, sha256)


def parse_document(
    result: LoadResult,
    *,
    doc_id: str,
    kb_id: str,
    fallback_title: str = "",
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> ParsedDocument:
    """把 loader 的原始输出整理成一份可直接切分的文档。"""
    blocks: list[Block] = []
    images: list[ParsedImage] = []
    warnings = list(result.warnings)

    heading_stack: list[tuple[int, str]] = []
    anchor: Block | None = None
    seen_image_hashes: set[str] = set()
    duplicates = 0

    for _order, kind, payload in _merged_order(result.blocks, result.images):
        if kind == "block":
            raw: RawBlock = payload  # type: ignore[assignment]

            if raw.is_heading:
                level = raw.heading_level
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, raw.text))
                # 标题切换后清空锚点：新小节开头的图片不该继承上一节的正文
                # 当上下文 —— 那会把图牢牢配错主题，且后面每一步都无从发现。
                anchor = None
                continue

            text = _block_text(raw)
            if not text:
                continue

            heading_path = [text_ for _, text_ in heading_stack]
            block = Block(
                block_id=f"b{raw.order}",
                order=raw.order,
                ordinal=len(blocks),
                text=text,
                block_type=raw.block_type,
                heading_path=heading_path,
                page=raw.page,
                source_ref=dict(raw.source_ref),
                table=[list(row) for row in raw.table] if raw.table else None,
            )
            blocks.append(block)
            anchor = block
            continue

        raw_image: RawImage = payload  # type: ignore[assignment]
        digest = hashlib.sha256(raw_image.data).hexdigest()
        if digest in seen_image_hashes:
            duplicates += 1
            continue
        seen_image_hashes.add(digest)

        heading_path = [text_ for _, text_ in heading_stack]
        parsed = ParsedImage(
            image_id=image_id_for(kb_id, digest),
            order=raw_image.order,
            data=raw_image.data,
            ext=raw_image.ext,
            mime_type=raw_image.mime_type,
            sha256=digest,
            page=raw_image.page,
            bbox=list(raw_image.bbox) if raw_image.bbox else None,
            heading_path=heading_path,
            anchor_block_id=anchor.block_id if anchor else None,
            anchor_order=anchor.order if anchor else None,
            context_text=_tail(anchor.text, context_chars) if anchor else "",
            source_ref=dict(raw_image.source_ref),
        )
        images.append(parsed)

        if anchor is not None:
            # 反向关联：切分器据此把图片 ID 写进所在分块，检索命中分块时
            # 前端能直接拿到可展示的图。
            anchor.image_ids.append(parsed.image_id)

    if duplicates:
        warnings.append(f"文档内有 {duplicates} 张重复图片，已按内容哈希去重")

    title = (result.title or "").strip() or fallback_title
    if not title:
        title = _first_heading_text(blocks) or doc_id

    parsed_doc = ParsedDocument(
        doc_id=doc_id,
        kb_id=kb_id,
        title=title,
        blocks=blocks,
        images=images,
        page_count=result.page_count,
        warnings=warnings,
        metadata=dict(result.metadata),
    )
    if not parsed_doc.has_content:
        parsed_doc.warnings.append("解析后没有任何可入库的正文或图片")
    return parsed_doc


def _merged_order(
    blocks: list[RawBlock], images: list[RawImage]
) -> list[tuple[int, str, RawBlock | RawImage]]:
    """块与图片按共享的 order 交错排序。

    ``order`` 相同时块在前：这只可能发生在 loader 没按契约取号时，
    此时「文字在上、图片在下」是更安全的猜测。
    """
    items: list[tuple[int, str, RawBlock | RawImage]] = [
        (block.order, "block", block) for block in blocks
    ]
    items.extend((image.order, "image", image) for image in images)
    items.sort(key=lambda item: (item[0], 0 if item[1] == "block" else 1))
    return items


def _block_text(raw: RawBlock) -> str:
    if raw.block_type is BlockType.TABLE or raw.table is not None:
        return to_markdown(raw.table or [])
    return raw.text.strip()


def _tail(text: str, limit: int) -> str:
    """取尾部 ``limit`` 个字符，尽量从句首开始。

    直接 ``text[-limit:]`` 会把「因此，」这种半句留成开头，视觉模型
    会把它当成一个转折关系的后半句去解读整张图。往前找最近的换行/句号，
    最多多让出 20% 的长度。
    """
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text

    start = len(text) - limit
    floor = max(0, start - limit // 5)
    for index in range(start, floor, -1):
        if text[index - 1] in "\n。！？.!?；;":
            return text[index:].strip()
    return text[start:].strip()


def _first_heading_text(blocks: list[Block]) -> str:
    for block in blocks:
        if block.heading_path:
            return block.heading_path[0]
    return ""
