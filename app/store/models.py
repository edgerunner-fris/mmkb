"""SQLAlchemy 2.x ORM 表定义。

这里只描述**元数据**（知识库 / 文档 / 分块 / 图片 / 任务 / 会话），
向量本体在 ``vectorstore_*``，稀疏索引在 ``bm25store``，图片字节在 ``blobstore``。

两个容易踩的坑，都在本文件里显式处理掉了
------------------------------------------------

1. **时区**。SQLite 没有原生时间类型，SQLAlchemy 的 ``DateTime`` 落到磁盘上
   就是一串 ``"2026-10-16 02:37:32.123456"`` —— tzinfo 直接丢失。
   读回来是 naive datetime，一旦有人拿它和 ``datetime.now(UTC)`` 相减就会
   ``TypeError``，或者更糟：被当成本地时间，整体偏移 8 小时。
   :class:`UTCDateTime` 在写时统一转成「带时区的 UTC → 去掉 tzinfo 的 UTC」，
   读时补回 ``tzinfo=UTC``，让进出这一层的时间**永远是 aware 的**。

2. **JSON 字段的就地修改**。``.extra["x"] = 1`` 这种写法 SQLAlchemy 是看不见的，
   因为纯 ``JSON`` 类型不做变更跟踪，flush 时不会写回。
   这里用 :class:`MutableDict` / :class:`MutableList` 包一层，
   让就地改也能被感知 —— 否则会出现「改了没保存，而且不报错」这种最难查的 bug。

关系一律 ``lazy="raise"``：异步 session 里隐式懒加载本来就会炸
（``MissingGreenlet``），不如让它炸在我们自己的、看得懂的错误上。
级联删除靠数据库的 ``ON DELETE CASCADE``（``passive_deletes=True``），
不在 Python 侧逐条加载子行。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.schemas import (
    BlockType,
    CaptionSource,
    DocStatus,
    FilteredReason,
    ImageStatus,
    TaskStatus,
    utcnow,
)

__all__ = [
    "Base",
    "ChunkRow",
    "DocumentRow",
    "ImageRow",
    "KnowledgeBaseRow",
    "MessageRow",
    "SessionRow",
    "TaskRow",
    "UTCDateTime",
]

#: 统一的约束命名规则。现在看起来多余，但等到要写迁移脚本时，
#: 「那个唯一约束到底叫什么」这个问题会在每个数据库上有不同答案。
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# --------------------------------------------------------------------------------------
# 自定义列类型
# --------------------------------------------------------------------------------------


class UTCDateTime(TypeDecorator[datetime]):
    """永远是 aware-UTC 的 datetime。

    写：aware → 转 UTC → 去掉 tzinfo 存盘（naive UTC）。
    读：补回 ``tzinfo=UTC``。

    传入 naive datetime **直接报错**而不是猜。把它当 UTC 会让「本地时间被误存」
    这类 bug 静默通过；把它当本地时间又会让同一份代码在不同时区的机器上行为不同。
    唯一安全的做法是拒绝：调用方要么用 :func:`app.core.schemas.utcnow`，
    要么显式 ``dt.replace(tzinfo=UTC)`` 说明自己的意图。
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"UTCDateTime 只接受 datetime，收到 {type(value).__name__}")
        if value.tzinfo is None:
            raise ValueError(
                "拒绝写入不带时区的 datetime：naive 时间在不同机器上含义不同。"
                "请使用 app.core.schemas.utcnow()，或显式 replace(tzinfo=UTC)"
            )
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(
        self, value: datetime | None, dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


#: 可变的 JSON 对象 / 数组列。见模块 docstring 第 2 点。
JsonDict = MutableDict.as_mutable(JSON)
JsonList = MutableList.as_mutable(JSON)


# --------------------------------------------------------------------------------------
# 1. 知识库
# --------------------------------------------------------------------------------------


class KnowledgeBaseRow(Base):
    """一个知识库 = 两个向量 collection（``{kb_id}__text`` / ``{kb_id}__image``）
    + 这个命名空间下的全部元数据行。

    ``text_dim`` / ``image_dim`` / 各模型名是**建库时的快照**。
    它们存在的唯一目的是检测「换了 embedding 模型」这种事故 —— 模型一变，
    向量空间就变了，库里已有的向量与新查询不再可比，而余弦相似度照样能算出一个数。
    比对不上时必须明确报错，不能静默返回垃圾结果。
    """

    __tablename__ = "knowledge_bases"

    kb_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")

    text_collection: Mapped[str] = mapped_column(String(128), nullable=False)
    image_collection: Mapped[str] = mapped_column(String(128), nullable=False)

    text_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    image_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    text_embed_model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    image_embed_model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    rerank_model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    #: 图像融合模式。与 image_embed_model 一起构成「图像空间指纹」——
    #: enable_fusion 改了，同一个模型的向量也不再可比。
    image_enable_fusion: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    doc_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    documents: Mapped[list[DocumentRow]] = relationship(
        back_populates="kb",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    def __repr__(self) -> str:
        return f"<KnowledgeBaseRow {self.kb_id} {self.name!r} docs={self.doc_count}>"


# --------------------------------------------------------------------------------------
# 2. 文档
# --------------------------------------------------------------------------------------


class DocumentRow(Base):
    """一份被导入的原始文件。

    ``(kb_id, doc_hash)`` 唯一 —— 这是**幂等导入**的根据：
    同一份文件重复上传，第二次直接返回既有 doc_id 并标记 ``skipped_duplicate``，
    而不是产生一堆重复向量把检索结果稀释掉。
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("kb_id", "doc_hash", name="uq_documents_kb_id_doc_hash"),
        Index("ix_documents_kb_id_status", "kb_id", "status"),
        Index("ix_documents_kb_id_created_at", "kb_id", "created_at"),
    )

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False
    )

    source_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_name: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    doc_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)

    status: Mapped[str] = mapped_column(String(24), nullable=False, default=DocStatus.QUEUED)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    extra: Mapped[dict[str, Any]] = mapped_column(JsonDict, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )
    indexed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    kb: Mapped[KnowledgeBaseRow] = relationship(back_populates="documents", lazy="raise")
    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )
    images: Mapped[list[ImageRow]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    def __repr__(self) -> str:
        return f"<DocumentRow {self.doc_id} {self.source_name!r} {self.status}>"


# --------------------------------------------------------------------------------------
# 3. 分块
# --------------------------------------------------------------------------------------


class ChunkRow(Base):
    """文本块。

    图片 caption **也**存在这张表里（``block_type='image_caption'``），
    只是同时带 ``image_ids``。这样它天然进入文本向量库与 BM25 语料，
    不需要第二套索引 —— 这是「图片既能被文本检索命中、也能被跨模态检索命中」
    的实现基础，不是顺手为之。

    ``prev_chunk_id`` / ``next_chunk_id`` 支撑检索后的邻居扩展
    （``RETRIEVAL_NEIGHBOR_EXPAND``）：命中一个块时把前后文一起喂给生成模型，
    能显著缓解「答案被切在分块边界上」的问题。
    """

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_kb_id_doc_id_ordinal", "kb_id", "doc_id", "ordinal"),
        Index("ix_chunks_doc_id", "doc_id"),
    )

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False
    )
    doc_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    block_type: Mapped[str] = mapped_column(
        String(24), nullable=False, default=BlockType.TEXT
    )

    heading_path: Mapped[list[str]] = mapped_column(JsonList, nullable=False, default=list)
    page_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_ids: Mapped[list[str]] = mapped_column(JsonList, nullable=False, default=list)

    prev_chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    next_chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    source_ref: Mapped[dict[str, Any]] = mapped_column(JsonDict, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    document: Mapped[DocumentRow] = relationship(back_populates="chunks", lazy="raise")

    def __repr__(self) -> str:
        return f"<ChunkRow {self.chunk_id} #{self.ordinal} {self.block_type} {len(self.text)}字>"


# --------------------------------------------------------------------------------------
# 4. 图片
# --------------------------------------------------------------------------------------


class ImageRow(Base):
    """一张图片的完整生命周期：抽取 → 过滤 → caption → 双向量化 → 落盘。

    ``caption`` 存的是 :class:`app.core.schemas.ImageCaption` 的 ``model_dump()``。
    刻意不把 summary / ocr_text / keywords 拆成独立列：caption 的字段结构
    会随 prompt 演进，拆列意味着每次调整 prompt 都要改表结构；
    而这里从不需要按 caption 内部字段做 SQL 查询（检索走向量与 BM25）。
    """

    __tablename__ = "images"
    __table_args__ = (
        Index("ix_images_kb_id_phash", "kb_id", "phash"),
        Index("ix_images_kb_id_sha256", "kb_id", "sha256"),
        Index("ix_images_doc_id", "doc_id"),
        Index("ix_images_doc_id_order_in_doc", "doc_id", "order_in_doc"),
    )

    image_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False
    )
    doc_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=True
    )
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox: Mapped[list[float] | None] = mapped_column(JsonList, nullable=True)
    order_in_doc: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    file_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False, default="image/png")
    width: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    phash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default=ImageStatus.EXTRACTED
    )
    filtered_reason: Mapped[str | None] = mapped_column(String(24), nullable=True)

    caption: Mapped[dict[str, Any] | None] = mapped_column(JsonDict, nullable=True)
    caption_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    caption_source: Mapped[str] = mapped_column(
        String(24), nullable=False, default=CaptionSource.NONE
    )
    anchor_chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    context_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    #: caption 落在文本向量库里的那条 chunk（``block_type='image_caption'``）
    text_chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: 图片多模态向量在 ``{kb}__image`` 里的 id
    image_vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    document: Mapped[DocumentRow | None] = relationship(back_populates="images", lazy="raise")

    def __repr__(self) -> str:
        return f"<ImageRow {self.image_id} {self.width}x{self.height} {self.status}>"


# --------------------------------------------------------------------------------------
# 5. 任务
# --------------------------------------------------------------------------------------


class TaskRow(Base):
    """异步导入任务的进度记录。前端轮询这张表。

    ``stage_timings`` 让「这次导入为什么慢」这个问题在事后可回答，
    也让 ``/api/health`` 能给出各阶段的 p50 参考值。
    """

    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_kb_id_created_at", "kb_id", "created_at"),
        Index("ix_tasks_status", "status"),
    )

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(String(64), nullable=False)
    doc_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="ingest")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=TaskStatus.QUEUED
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")

    progress_current: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stage_timings: Mapped[dict[str, Any]] = mapped_column(
        JsonDict, nullable=False, default=dict
    )
    result: Mapped[dict[str, Any]] = mapped_column(JsonDict, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<TaskRow {self.task_id} {self.kind} {self.status} "
            f"{self.progress_current}/{self.progress_total}>"
        )


# --------------------------------------------------------------------------------------
# 6. 会话与消息
# --------------------------------------------------------------------------------------


class SessionRow(Base):
    """一次对话。多轮追问依赖它提供 history。"""

    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_kb_id_created_at", "kb_id", "created_at"),)

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    messages: Mapped[list[MessageRow]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    def __repr__(self) -> str:
        return f"<SessionRow {self.session_id} kb={self.kb_id}>"


class MessageRow(Base):
    """一问一答。

    ``citations`` 随消息一起存：前端刷新页面后要能原样回显引用卡片，
    而不是重新检索一遍（重新检索可能因为索引变动而给出不同的引用，
    那会让用户觉得「刚才看到的那个来源怎么不见了」）。
    """

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_session_id_created_at", "session_id", "created_at"),)

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JsonList, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    session: Mapped[SessionRow] = relationship(back_populates="messages", lazy="raise")

    def __repr__(self) -> str:
        return f"<MessageRow {self.message_id} {self.role} {len(self.content)}字>"


# --------------------------------------------------------------------------------------
# 取值合法性
# --------------------------------------------------------------------------------------

#: 各枚举列允许的取值。SQLite 不校验字符串内容，所以由 docstore 在写入前检查 ——
#: 少了这道检查，「status 拼错一个字母」会安静地写进去，
#: 直到某天发现有一批文档永远卡在 ``embeddingg`` 状态。
ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "knowledge_bases.status": frozenset({"active", "deleting"}),
    "documents.status": frozenset(str(s) for s in DocStatus),
    "chunks.block_type": frozenset(str(b) for b in BlockType),
    "images.status": frozenset(str(s) for s in ImageStatus),
    "images.filtered_reason": frozenset(str(r) for r in FilteredReason),
    "images.caption_source": frozenset(str(s) for s in CaptionSource),
    "tasks.status": frozenset(str(s) for s in TaskStatus),
    "tasks.kind": frozenset({"ingest", "eval", "delete"}),
    "messages.role": frozenset({"user", "assistant"}),
}


def check_enum(column: str, value: str | None) -> None:
    """校验枚举列的取值。``None`` 永远合法（表示「没有」）。"""
    if value is None:
        return
    allowed = ALLOWED_VALUES.get(column)
    if allowed is not None and value not in allowed:
        raise ValueError(
            f"{column}={value!r} 不是合法取值，可选：{sorted(allowed)}"
        )
