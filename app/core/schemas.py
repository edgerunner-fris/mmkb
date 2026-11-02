"""领域模型。

六个核心结构（KnowledgeBase / Document / Chunk / ImageAsset / RetrievedItem / Citation）
加上问答链路的信封结构（RoundTrace / AnswerResult）与任务、会话。

约定
----
* 时间为 **UTC aware** ``datetime``；存储层负责序列化。
* ``heading_path`` 是从根到叶的标题链，例如 ``["第三章 系统设计", "3.2 检索模块"]``。
  它同时充当：检索时的元数据过滤器、重排时补足上下文的文本、以及引用回显的定位信息。
* 所有 ``*_metadata()`` 方法产出 **扁平标量字典**，因为 Chroma 的 metadata
  只接受 str/int/float/bool（列表需序列化）。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "AnswerResult",
    "BlockType",
    "CaptionSource",
    "Chunk",
    "Citation",
    "DebugRound",
    "DegradedReason",
    "DocStatus",
    "Document",
    "FilteredReason",
    "ImageAsset",
    "ImageCaption",
    "ImageType",
    "ItemType",
    "JudgeVerdict",
    "KnowledgeBase",
    "Message",
    "RetrievalFilters",
    "RetrievedItem",
    "Route",
    "Session",
    "Task",
    "TaskStatus",
    "utcnow",
]


def utcnow() -> datetime:
    """统一的时间源。全项目只在这里取「现在」，便于测试替换。"""
    return datetime.now(UTC)


# --------------------------------------------------------------------------------------
# 枚举
# --------------------------------------------------------------------------------------


class BlockType(StrEnum):
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    IMAGE_CAPTION = "image_caption"


class DocStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    CAPTIONING = "captioning"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    DONE = "done"
    FAILED = "failed"
    SKIPPED_DUPLICATE = "skipped_duplicate"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ImageStatus(StrEnum):
    EXTRACTED = "extracted"
    FILTERED_OUT = "filtered_out"
    DONE = "done"
    FAILED = "failed"


class FilteredReason(StrEnum):
    SMALL = "small"
    BLANK = "blank"
    DUPLICATE = "duplicate"
    DECORATIVE = "decorative"
    OVER_QUOTA = "over_quota"


class CaptionSource(StrEnum):
    VLM = "vlm"
    CONTEXT_ONLY = "context_only"
    NONE = "none"


class ImageType(StrEnum):
    CHART = "chart"
    SCREENSHOT = "screenshot"
    DIAGRAM = "diagram"
    PHOTO = "photo"
    FORMULA = "formula"
    TABLE_IMAGE = "table_image"
    LOGO = "logo"
    OTHER = "other"


class ItemType(StrEnum):
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    IMAGE = "image"


class Route(StrEnum):
    BM25 = "bm25"
    DENSE_TEXT = "dense_text"
    DENSE_IMAGE = "dense_image"
    HYBRID = "hybrid"


class DegradedReason(StrEnum):
    IMAGE_EMBED_UNAVAILABLE = "image_embed_unavailable"
    TEXT_EMBED_UNAVAILABLE = "text_embed_unavailable"
    VISION_UNAVAILABLE = "vision_unavailable"
    RERANK_UNAVAILABLE = "rerank_unavailable"
    JUDGE_UNAVAILABLE = "judge_unavailable"
    BM25_UNAVAILABLE = "bm25_unavailable"
    LLM_UNAVAILABLE = "llm_unavailable"
    REASONING_EFFORT_REJECTED = "reasoning_effort_rejected"
    #: 有资料、有答案，但答案一个 ``[n]`` 都没标。不是「没有引用可用」，
    #: 而是「模型没有遵守引用约束」—— 两者对使用者的含义完全不同。
    CITATION_MISSING = "citation_missing"


_BLOCK_TO_ITEM: dict[BlockType, ItemType] = {
    BlockType.TEXT: ItemType.TEXT,
    BlockType.CODE: ItemType.CODE,
    BlockType.TABLE: ItemType.TABLE,
    BlockType.IMAGE_CAPTION: ItemType.IMAGE,
}

_ITEM_TO_BLOCK: dict[ItemType, BlockType] = {v: k for k, v in _BLOCK_TO_ITEM.items()}


# --------------------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------------------


def _flatten(value: Any) -> str | int | float | bool | None:
    """把任意可 JSON 化的值压成 Chroma metadata 能接受的样子。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        # 必须先转 UTC 再 isoformat：Chroma 不支持日期类型，时间过滤靠的是
        # ISO 字符串的**字典序与时间序一致**这个性质。带 +08:00 的字符串
        # 字典序会排到 +00:00 前面去，过滤结果就静默错了。
        return value.astimezone(UTC).isoformat()
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _flatten_dict(source: dict[str, Any]) -> dict[str, str | int | float | bool]:
    out: dict[str, str | int | float | bool] = {}
    for key, value in source.items():
        flat = _flatten(value)
        if flat is not None:
            out[key] = flat
    return out


# --------------------------------------------------------------------------------------
# 1. KnowledgeBase
# --------------------------------------------------------------------------------------


class KnowledgeBase(BaseModel):
    """一个知识库 = 两个 Chroma collection + 一份 SQLite 命名空间。

    ``text_dim`` / ``image_dim`` / 各模型名都是**建库时的快照**。它们的作用是检测
    「换了 embedding 模型」这种事故：模型变了，向量空间就变了，旧向量与新查询
    不再可比。检索前会拿快照和当前配置比对，不一致就明确报错而不是静默返回垃圾结果。
    """

    kb_id: str
    name: str = Field(min_length=1, max_length=64)
    description: str = ""
    status: Literal["active", "deleting"] = "active"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    text_collection: str
    image_collection: str

    text_dim: int
    image_dim: int
    llm_model: str
    text_embed_model: str
    image_embed_model: str
    rerank_model: str

    doc_count: int = 0
    chunk_count: int = 0
    image_count: int = 0

    @property
    def is_active(self) -> bool:
        return self.status == "active"


# --------------------------------------------------------------------------------------
# 2. Document
# --------------------------------------------------------------------------------------


class Document(BaseModel):
    """一份被导入的原始文件。``(kb_id, doc_hash)`` 唯一 —— 幂等导入的根据。"""

    doc_id: str
    kb_id: str
    source_path: str
    source_name: str
    mime_type: str
    doc_hash: str
    size_bytes: int = 0
    page_count: int | None = None
    title: str | None = None

    status: DocStatus = DocStatus.QUEUED
    task_id: str | None = None
    error: str | None = None

    chunk_count: int = 0
    image_count: int = 0
    skipped_image_count: int = 0

    extra: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    indexed_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (DocStatus.DONE, DocStatus.FAILED, DocStatus.SKIPPED_DUPLICATE)


# --------------------------------------------------------------------------------------
# 3. Chunk
# --------------------------------------------------------------------------------------


class Chunk(BaseModel):
    """文本块。

    图片 caption **也**是一个 Chunk（``block_type=image_caption``），只是同时
    带 ``image_ids``。这样它天然进入文本向量库和 BM25 语料，无需第二套索引 ——
    这是「图片既能被文本检索命中、也能被跨模态检索命中」的实现基础。
    """

    chunk_id: str
    kb_id: str
    doc_id: str
    ordinal: int

    text: str
    token_count: int
    block_type: BlockType = BlockType.TEXT

    heading_path: list[str] = Field(default_factory=list)
    page_from: int | None = None
    page_to: int | None = None
    image_ids: list[str] = Field(default_factory=list)

    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None

    source_ref: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def item_type(self) -> ItemType:
        return _BLOCK_TO_ITEM[self.block_type]

    @property
    def heading_text(self) -> str:
        return " > ".join(self.heading_path)

    def to_metadata(self) -> dict[str, str | int | float | bool]:
        """Chroma metadata。``text`` 不入 metadata（Chroma 的 document 字段存它）。"""
        return _flatten_dict(
            {
                "kb_id": self.kb_id,
                "doc_id": self.doc_id,
                "chunk_id": self.chunk_id,
                "ordinal": self.ordinal,
                "block_type": str(self.block_type),
                "heading_path": self.heading_path,
                "heading_text": self.heading_text,
                "page_from": self.page_from,
                "page_to": self.page_to,
                "image_ids": self.image_ids,
                "token_count": self.token_count,
                "created_at": self.created_at,
                "source_ref": self.source_ref,
            }
        )


# --------------------------------------------------------------------------------------
# 4. ImageAsset
# --------------------------------------------------------------------------------------


class ImageCaption(BaseModel):
    """VLM 结构化输出。字段设计直接服务于后续检索与重排：

    * ``summary`` / ``keywords`` → 语义召回
    * ``ocr_text`` → 图内文字（截图、公式、图表标签）的精确匹配
    * ``image_type`` → 过滤与展示
    * ``relation_to_section`` → 让重排知道这张图和所在章节的关系
    """

    image_type: ImageType = ImageType.OTHER
    summary: str = ""
    ocr_text: str = ""
    relation_to_section: str = ""
    keywords: list[str] = Field(default_factory=list)
    caption_model: str = ""
    caption_latency_ms: int = 0


class ImageAsset(BaseModel):
    """一张图片的完整生命周期记录：抽取 → 过滤 → caption → 双向量化 → 落盘。"""

    image_id: str
    kb_id: str
    doc_id: str | None = None
    page: int | None = None
    bbox: list[float] | None = None
    order_in_doc: int = 0

    file_path: str = ""
    url: str = ""
    mime_type: str = "image/png"
    width: int = 0
    height: int = 0
    size_bytes: int = 0

    sha256: str = ""
    phash: str = ""
    status: ImageStatus = ImageStatus.EXTRACTED
    filtered_reason: FilteredReason | None = None

    caption: ImageCaption | None = None
    caption_text: str = ""
    caption_source: CaptionSource = CaptionSource.NONE
    anchor_chunk_id: str | None = None
    context_text: str = ""
    #: 图片所在章节的标题链。重排与展示都要用它，所以必须**随图片持久化**，
    #: 不能只存在于入库时的临时对象里 —— 检索阶段拿不到解析上下文。
    heading_path: list[str] = Field(default_factory=list)

    text_chunk_id: str | None = None
    image_vector_id: str | None = None

    created_at: datetime = Field(default_factory=utcnow)

    @property
    def is_usable(self) -> bool:
        """能否参与检索：过滤掉的、caption 失败的都不算。"""
        return self.status == ImageStatus.DONE and bool(self.caption_text)

    def to_metadata(self) -> dict[str, str | int | float | bool]:
        """图像 collection 的 metadata（以文搜图/以图搜图时用于过滤与回显）。"""
        return _flatten_dict(
            {
                "kb_id": self.kb_id,
                "doc_id": self.doc_id,
                "image_id": self.image_id,
                "page": self.page,
                "order_in_doc": self.order_in_doc,
                "url": self.url,
                "width": self.width,
                "height": self.height,
                "image_type": str(self.caption.image_type) if self.caption else None,
                "caption": self.caption_text,
                "heading_path": self.heading_path,
                "heading_text": " > ".join(self.heading_path),
                "created_at": self.created_at,
            }
        )


def build_caption_text(
    caption: ImageCaption,
    heading_path: list[str],
) -> str:
    """caption → 参与文本向量与重排的文本。

    这个拼接格式是**检索质量的关键**：章节路径提供上下文（「这张图在讲什么主题」），
    summary 提供语义，OCR 文本提供精确匹配能力。重排器只吃文本，
    所以这段文本的质量直接决定图片候选能不能被排上来。
    """
    parts: list[str] = []
    if heading_path:
        parts.append(" > ".join(heading_path))
    if caption.summary:
        # OTHER 是「没认出来」的兜底类型，把它印成标签只会给重排器添噪声
        type_label = (
            f"[{caption.image_type}] "
            if caption.image_type and caption.image_type != ImageType.OTHER
            else ""
        )
        parts.append(f"{type_label}{caption.summary}")
    if caption.ocr_text:
        parts.append(f"图中文字: {caption.ocr_text}")
    if caption.keywords:
        parts.append(f"关键词: {', '.join(caption.keywords)}")
    return " | ".join(parts)


# --------------------------------------------------------------------------------------
# 5. RetrievedItem
# --------------------------------------------------------------------------------------


class RetrievedItem(BaseModel):
    """三路召回汇合后的统一中间结构。

    保留 ``route_ranks`` / ``route_scores`` 是为了让 ``debug_trace`` 能完整还原
    「某条结果是被哪一路、以第几名捞上来的」—— 排查召回问题时这是唯一有用的信息。
    """

    item_id: str
    item_type: ItemType = ItemType.TEXT
    kb_id: str
    doc_id: str

    chunk_id: str | None = None
    image_id: str | None = None

    text: str = ""
    score: float = 0.0
    route: Route = Route.HYBRID
    route_scores: dict[str, float] = Field(default_factory=dict)
    route_ranks: dict[str, int] = Field(default_factory=dict)
    rrf_score: float = 0.0
    rerank_score: float | None = None

    heading_path: list[str] = Field(default_factory=list)
    page: int | None = None
    source_name: str = ""
    image_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        return self.item_type == ItemType.IMAGE

    @property
    def heading_text(self) -> str:
        return " > ".join(self.heading_path)


# --------------------------------------------------------------------------------------
# 6. Citation
# --------------------------------------------------------------------------------------


class Citation(BaseModel):
    """回填给前端的引用。``index`` 与答案正文里的 ``[n]`` 一一对应。"""

    index: int = Field(ge=1)
    item_id: str
    item_type: ItemType = ItemType.TEXT
    doc_id: str = ""
    doc_name: str = ""
    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    snippet: str = ""
    image_url: str | None = None
    image_id: str | None = None
    score: float = 0.0
    rerank_score: float | None = None

    @property
    def is_image(self) -> bool:
        return self.item_type == ItemType.IMAGE


# --------------------------------------------------------------------------------------
# Agent / 问答信封
# --------------------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """LLM 对「检索结果与问题的相关性」的判定。"""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    threshold: float = 0.0
    passed: bool = False
    latency_ms: int = 0


class DebugRound(BaseModel):
    """一轮检索的完整快照。``AnswerResult.debug_trace`` 是它的列表。"""

    round: int = 1
    query: str = ""
    rewrites: list[str] = Field(default_factory=list)
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    fused_unique: int = 0
    reranked: int = 0
    judge: JudgeVerdict | None = None
    timings_ms: dict[str, int] = Field(default_factory=dict)
    degraded_reasons: list[DegradedReason] = Field(default_factory=list)


class AnswerResult(BaseModel):
    """``POST /api/chat`` 的响应体。"""

    answer_markdown: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    used_rounds: int = 1
    latency_ms: int = 0
    degraded: bool = False
    degraded_reasons: list[DegradedReason] = Field(default_factory=list)
    debug_trace: list[DebugRound] = Field(default_factory=list)
    session_id: str = ""
    kb_id: str = ""


# --------------------------------------------------------------------------------------
# 任务 / 会话
# --------------------------------------------------------------------------------------


class Task(BaseModel):
    """异步导入任务的进度记录。前端轮询这个。"""

    task_id: str
    kb_id: str
    doc_id: str | None = None
    kind: Literal["ingest", "eval", "delete"] = "ingest"
    status: TaskStatus = TaskStatus.QUEUED
    stage: str = "queued"
    progress_current: int = 0
    progress_total: int = 0
    stage_timings: dict[str, int] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None

    @property
    def progress_ratio(self) -> float:
        if self.progress_total <= 0:
            return 0.0
        return min(1.0, self.progress_current / self.progress_total)


class Session(BaseModel):
    session_id: str
    kb_id: str
    created_at: datetime = Field(default_factory=utcnow)


class Message(BaseModel):
    message_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] | None = None
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------------------
# 检索过滤器
# --------------------------------------------------------------------------------------


class RetrievalFilters(BaseModel):
    """检索过滤条件，在存储层生效（不是召回后再筛）。"""

    block_types: list[BlockType] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    image_types: list[ImageType] = Field(default_factory=list)
    created_after: datetime | None = None
    created_before: datetime | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.block_types,
                self.doc_ids,
                self.image_types,
                self.created_after,
                self.created_before,
            ]
        )

    def to_chroma_where(
        self, *, space: Literal["text", "image"] = "text"
    ) -> dict[str, Any] | None:
        """转成 Chroma 的 ``where`` 子句；无过滤条件时返回 ``None``。

        时间用 ISO 字符串比较 —— Chroma 不支持日期类型，而 ISO-8601 的
        字典序与时间序一致，所以字符串比较是正确且可索引的。

        ``space`` 只影响 ``block_types``：它描述的是**文本分块**的形态
        （正文/代码/表格/caption），而图像 collection 里根本没有这个字段。
        把 ``block_type`` 条件原样带进图像库，Chroma 不会报错，只会
        静默返回空结果 —— 一个「过滤条件没生效」伪装成的「没搜到」。
        图像侧真正有意义的过滤字段是 ``image_types``。
        """
        clauses: list[dict[str, Any]] = []
        if self.block_types and space == "text":
            clauses.append({"block_type": {"$in": [str(b) for b in self.block_types]}})
        if self.doc_ids:
            clauses.append({"doc_id": {"$in": list(self.doc_ids)}})
        # 同理，``image_type`` 只存在于图像 collection 的 metadata 里；
        # 文本 collection 的分块没有这个字段，带进去会得到空结果。
        if self.image_types and space == "image":
            clauses.append({"image_type": {"$in": [str(t) for t in self.image_types]}})
        # 反过来，按图片类型过滤时，文本空间里的 caption 分块没有任何字段
        # 能表达这个条件。放它过去等于让一部分图片**绕过**用户设置的过滤，
        # 所以这里把它们排除掉 —— 图片结果交给图像空间按类型精确召回。
        # 用户显式点名要 image_caption 时以 block_types 为准。
        if (
            self.image_types
            and space == "text"
            and str(BlockType.IMAGE_CAPTION)
            not in {str(block) for block in self.block_types}
        ):
            clauses.append({"block_type": {"$ne": str(BlockType.IMAGE_CAPTION)}})
        if self.created_after is not None:
            clauses.append({"created_at": {"$gte": self.created_after.isoformat()}})
        if self.created_before is not None:
            clauses.append({"created_at": {"$lte": self.created_before.isoformat()}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
