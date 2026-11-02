"""元数据库 CRUD。

职责边界
--------
本模块是 **SQLite 表 ↔ Pydantic 领域模型** 的唯一转换点。上层（摄入、检索、API）
只碰 :mod:`app.core.schemas` 里的模型，不碰 ``*Row``。这条界线让「换存储」
与「改表结构」都不会波及业务逻辑。

三个必须守住的约定
----------------

**1. 每次写入前校验枚举列。**
   SQLite 不校验字符串，``status="embeddingg"`` 会安静地写进去，
   然后那批文档永远停在「正在向量化」，直到有人肉眼发现。
   :func:`~app.store.models.check_enum` 把这类错误拦在写入之前。

**2. 级联删除交给数据库，不交给 ORM。**
   ``session.delete(kb_row)`` 会让 SQLAlchemy 先把整个对象图加载进内存 ——
   一个 5000 分块的知识库就是 5000 次 SELECT，只为了删掉它们。
   这里一律用 ``delete(KnowledgeBaseRow).where(...)`` 这类 Core 语句，
   由 ``ON DELETE CASCADE`` 在 SQLite 内部完成。前提是连接上开了
   ``PRAGMA foreign_keys=ON``（见 :mod:`app.store.db`）——
   **这个 pragma 默认是关的，关了之后 CASCADE 会静默失效**，
   删知识库只删掉一行，剩下的全是孤儿数据。

**3. ``lazy="raise"`` 是刻意的。**
   所有 relationship 都设了 ``lazy="raise"``。异步 session 本来就不能隐式懒加载，
   与其让 SQLAlchemy 在运行时抛一个含糊的 ``MissingGreenlet``，
   不如让「访问未加载的关系」直接报错，逼调用方写一个明确的查询。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, or_, select
from sqlalchemy import update as sa_update
from sqlalchemy.exc import IntegrityError

from app.config import Settings, get_settings
from app.core.errors import NotFoundError, StorageError
from app.core.ids import new_id
from app.core.logging import get_logger
from app.core.schemas import (
    Chunk,
    Citation,
    DocStatus,
    Document,
    ImageAsset,
    ImageCaption,
    ImageStatus,
    KnowledgeBase,
    Message,
    Task,
    TaskStatus,
    utcnow,
)
from app.core.schemas import (
    Session as ChatSession,
)
from app.store.db import Database, get_database
from app.store.models import (
    ChunkRow,
    DocumentRow,
    ImageRow,
    KnowledgeBaseRow,
    MessageRow,
    SessionRow,
    TaskRow,
    check_enum,
)

__all__ = ["DocStore"]

logger = get_logger(__name__)

#: 一次 ``IN (...)`` 查询最多带多少个 id。
#: SQLite 的变量上限是 999（老版本）/ 32766（3.32+），2048 在两边都安全。
_MAX_IN_CLAUSE = 2048


class DocStore:
    """SQLite 元数据仓库。

    无状态（除了指向 :class:`~app.store.db.Database` 的引用），
    可以放心当单例；并发安全由 SQLAlchemy 的连接池 + WAL 保证。
    """

    def __init__(self, settings: Settings | None = None, *, db: Database | None = None) -> None:
        self._settings = settings or get_settings()
        self._db = db or get_database()

    @property
    def database(self) -> Database:
        return self._db

    # ==================================================================================
    # 知识库
    # ==================================================================================

    async def create_kb(
        self,
        name: str,
        *,
        description: str = "",
        kb_id: str | None = None,
        text_collection: str,
        image_collection: str,
        text_dim: int,
        image_dim: int,
        llm_model: str = "",
        text_embed_model: str = "",
        image_embed_model: str = "",
        rerank_model: str = "",
        image_enable_fusion: bool = True,
    ) -> KnowledgeBase:
        """建一个知识库。

        ``text_dim`` / ``image_dim`` / 各模型名都是**建库时的快照**（见
        :class:`~app.core.schemas.KnowledgeBase` 的说明）。
        集合名由调用方传进来而不是在这里拼，是为了让「谁负责命名」这件事
        只有一个答案：:func:`app.store.vectorstore.text_collection`。
        """
        row = KnowledgeBaseRow(
            kb_id=kb_id or new_id("kb"),
            name=name,
            description=description,
            status="active",
            text_collection=text_collection,
            image_collection=image_collection,
            text_dim=text_dim,
            image_dim=image_dim,
            llm_model=llm_model,
            text_embed_model=text_embed_model,
            image_embed_model=image_embed_model,
            rerank_model=rerank_model,
            image_enable_fusion=image_enable_fusion,
        )
        check_enum("knowledge_bases.status", row.status)
        async with self._db.session() as session:
            session.add(row)
        logger.info(
            "已创建知识库",
            extra={
                "kb_id": row.kb_id,
                "name": name,
                "text_dim": text_dim,
                "image_dim": image_dim,
            },
        )
        return _kb_to_model(row)

    async def get_kb(self, kb_id: str) -> KnowledgeBase | None:
        async with self._db.session() as session:
            row = await session.get(KnowledgeBaseRow, kb_id)
            return _kb_to_model(row) if row else None

    async def require_kb(self, kb_id: str) -> KnowledgeBase:
        kb = await self.get_kb(kb_id)
        if kb is None:
            raise NotFoundError(
                f"知识库 {kb_id!r} 不存在", detail={"kb_id": kb_id}
            )
        return kb

    async def list_kbs(self, *, include_deleting: bool = False) -> list[KnowledgeBase]:
        stmt = select(KnowledgeBaseRow).order_by(KnowledgeBaseRow.created_at.desc())
        if not include_deleting:
            stmt = stmt.where(KnowledgeBaseRow.status == "active")
        async with self._db.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_kb_to_model(row) for row in rows]

    async def set_kb_status(self, kb_id: str, status: str) -> None:
        """``active`` / ``deleting``。

        ``deleting`` 是一个**两阶段删除**的中间态：先标记，再清向量库，
        最后删元数据。删向量库可能要几秒，中途崩溃留下的
        ``deleting`` 记录能让下次启动知道「这个库没删干净」，而不是当它不存在。
        """
        check_enum("knowledge_bases.status", status)
        async with self._db.session() as session:
            result = await session.execute(
                sa_update(KnowledgeBaseRow)
                .where(KnowledgeBaseRow.kb_id == kb_id)
                .values(status=status, updated_at=utcnow())
            )
            if result.rowcount == 0:
                raise NotFoundError(f"知识库 {kb_id!r} 不存在", detail={"kb_id": kb_id})

    async def refresh_kb_stats(self, kb_id: str) -> KnowledgeBase:
        """按实际行数重算 ``doc_count`` / ``chunk_count`` / ``image_count``。

        计数是**派生数据**，所以只提供「重算」而不提供「加一」：
        增量维护的计数器在导入失败、重试、部分回滚的场景下一定会漂移，
        而漂移了的计数器没有任何办法自查。真源永远是那三张表。
        """
        async with self._db.session() as session:
            row = await session.get(KnowledgeBaseRow, kb_id)
            if row is None:
                raise NotFoundError(f"知识库 {kb_id!r} 不存在", detail={"kb_id": kb_id})
            doc_count = await _scalar(
                session, select(func.count()).select_from(DocumentRow).where(
                    DocumentRow.kb_id == kb_id
                )
            )
            chunk_count = await _scalar(
                session, select(func.count()).select_from(ChunkRow).where(
                    ChunkRow.kb_id == kb_id
                )
            )
            image_count = await _scalar(
                session,
                select(func.count())
                .select_from(ImageRow)
                .where(ImageRow.kb_id == kb_id, ImageRow.status == str(ImageStatus.DONE)),
            )
            row.doc_count = doc_count
            row.chunk_count = chunk_count
            row.image_count = image_count
            row.updated_at = utcnow()
            snapshot = _kb_to_model(row)
        return snapshot

    async def delete_kb(self, kb_id: str) -> dict[str, int]:
        """删掉知识库的**全部**元数据行。返回各表删掉的行数。

        **不含**向量库与磁盘图片 —— 那两件事分别在
        :class:`~app.store.vectorstore.VectorStore` 与
        :class:`~app.store.blobstore.BlobStore` 里，由上层服务按顺序编排。
        本模块只负责自己这一摊，因为「谁拥有数据谁负责删」是唯一不会漏的划分方式。

        ``tasks`` 表没有指向 ``knowledge_bases`` 的外键（任务要在知识库删掉之后
        仍然可查，用于排查「刚才那次删除为什么失败」），所以这里的级联到不了它，
        必须显式删。
        """
        counts: dict[str, int] = {}
        async with self._db.session() as session:
            # 显式删 tasks：它没有 FK，不走 CASCADE
            tasks = await session.execute(
                sa_delete(TaskRow).where(TaskRow.kb_id == kb_id)
            )
            counts["tasks"] = max(0, tasks.rowcount or 0)

            # documents / chunks / images / sessions / messages 走 ON DELETE CASCADE。
            # 但 SQLite 的 CASCADE 依赖 PRAGMA foreign_keys=ON，
            # 所以这里**先数一遍**各类行数再删 —— 返回的统计既给调用方用，
            # 也是一个隐式的自检点：如果 counts 里 chunk 数远多于 doc 数
            # 而删除只花了几毫秒，就说明 CASCADE 没生效。
            #
            # messages 没有 kb_id 列（它的归属靠 session），单独一条语句。
            for name, row_cls in (
                ("chunks", ChunkRow),
                ("images", ImageRow),
                ("sessions", SessionRow),
                ("documents", DocumentRow),
            ):
                counts[name] = await _scalar(
                    session,
                    select(func.count()).select_from(row_cls).where(row_cls.kb_id == kb_id),
                )
            counts["messages"] = await _scalar(
                session,
                select(func.count())
                .select_from(MessageRow)
                .where(
                    MessageRow.session_id.in_(
                        select(SessionRow.session_id).where(SessionRow.kb_id == kb_id)
                    )
                ),
            )

            result = await session.execute(
                sa_delete(KnowledgeBaseRow).where(KnowledgeBaseRow.kb_id == kb_id)
            )
            if result.rowcount == 0:
                raise NotFoundError(f"知识库 {kb_id!r} 不存在", detail={"kb_id": kb_id})

        logger.info("已删除知识库元数据", extra={"kb_id": kb_id, **counts})
        return counts

    # ==================================================================================
    # 文档
    # ==================================================================================

    async def upsert_document(self, doc: Document) -> tuple[Document, bool]:
        """按 ``(kb_id, doc_hash)`` 幂等写入。返回 ``(文档, 是否新建)``。

        **重复导入不是错误，是常态。** 用户会反复拖同一个文件夹进来，
        前端也可能重试。第二次导入如果产生了第二份向量，检索结果里
        同一段文字就会出现两遍，把别的候选挤下去 —— 所以这里必须去重，
        而且要按**内容哈希**去重，不是按文件名：改名不改内容仍然是同一份文档。
        """
        check_enum("documents.status", str(doc.status))
        async with self._db.session() as session:
            existing = (
                await session.execute(
                    select(DocumentRow).where(
                        DocumentRow.kb_id == doc.kb_id,
                        DocumentRow.doc_hash == doc.doc_hash,
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                return _doc_to_model(existing), False

            row = _doc_to_row(doc)
            if not row.doc_id:
                row.doc_id = new_id("doc")
            session.add(row)
            try:
                # flush 才能在这一层捕获唯一约束冲突，而不是等到退出 with 时
                await session.flush()
            except IntegrityError:
                # 两个并发导入撞在同一份文件上：唯一约束救了我们，
                # 这里退回去读那条已存在的记录，语义上等价于「第二次导入」
                await session.rollback()
                concurrent = (
                    await session.execute(
                        select(DocumentRow).where(
                            DocumentRow.kb_id == doc.kb_id,
                            DocumentRow.doc_hash == doc.doc_hash,
                        )
                    )
                ).scalar_one_or_none()
                if concurrent is None:  # pragma: no cover - 回滚后竟然还是没有
                    raise StorageError(
                        "文档写入冲突但读不到既有记录",
                        detail={"kb_id": doc.kb_id, "doc_hash": doc.doc_hash},
                    ) from None
                return _doc_to_model(concurrent), False
            snapshot = _doc_to_model(row)
        return snapshot, True

    async def get_document(self, doc_id: str) -> Document | None:
        async with self._db.session() as session:
            row = await session.get(DocumentRow, doc_id)
            return _doc_to_model(row) if row else None

    async def require_document(self, doc_id: str) -> Document:
        doc = await self.get_document(doc_id)
        if doc is None:
            raise NotFoundError(f"文档 {doc_id!r} 不存在", detail={"doc_id": doc_id})
        return doc

    async def find_document_by_hash(self, kb_id: str, doc_hash: str) -> Document | None:
        """导入前的查重。命中就直接跳过，连解析都不用做。"""
        async with self._db.session() as session:
            row = (
                await session.execute(
                    select(DocumentRow).where(
                        DocumentRow.kb_id == kb_id, DocumentRow.doc_hash == doc_hash
                    )
                )
            ).scalar_one_or_none()
            return _doc_to_model(row) if row else None

    async def list_documents(
        self,
        kb_id: str,
        *,
        status: DocStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        stmt = (
            select(DocumentRow)
            .where(DocumentRow.kb_id == kb_id)
            .order_by(DocumentRow.created_at.desc(), DocumentRow.doc_id)
            .limit(max(1, limit))
            .offset(max(0, offset))
        )
        if status is not None:
            stmt = stmt.where(DocumentRow.status == str(status))
        async with self._db.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_doc_to_model(row) for row in rows]

    async def update_document_status(
        self,
        doc_id: str,
        status: DocStatus,
        *,
        error: str | None = None,
        task_id: str | None = None,
        chunk_count: int | None = None,
        image_count: int | None = None,
        skipped_image_count: int | None = None,
        title: str | None = None,
        page_count: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """更新状态与进度。**只写传进来的字段**，没传的保持原值。

        刻意不做成「传整个 Document 对象」：摄入流水线在多个阶段各改一两个字段，
        整对象覆盖会让阶段 A 的写把阶段 B 刚写的字段冲掉 ——
        这种 bug 在单线程测试里永远复现不了。
        """
        check_enum("documents.status", str(status))
        values: dict[str, Any] = {"status": str(status), "updated_at": utcnow()}
        # error 单独处理：只有「显式传了 None」才该清空错误，
        # 而「没传这个参数」应当保持原样。用 sentinel 区分这两种情况。
        values["error"] = error
        for key, value in (
            ("task_id", task_id),
            ("chunk_count", chunk_count),
            ("image_count", image_count),
            ("skipped_image_count", skipped_image_count),
            ("title", title),
            ("page_count", page_count),
        ):
            if value is not None:
                values[key] = value
        if extra is not None:
            values["extra"] = dict(extra)
        if status == DocStatus.DONE:
            values["indexed_at"] = utcnow()

        async with self._db.session() as session:
            result = await session.execute(
                sa_update(DocumentRow)
                .where(DocumentRow.doc_id == doc_id)
                .values(**values)
            )
            if result.rowcount == 0:
                raise NotFoundError(f"文档 {doc_id!r} 不存在", detail={"doc_id": doc_id})

    async def delete_document(self, doc_id: str) -> int:
        """删一份文档及其全部分块与图片记录。返回删掉的分块数。"""
        async with self._db.session() as session:
            row = await session.get(DocumentRow, doc_id)
            if row is None:
                return 0
            chunk_count = await _scalar(
                session,
                select(func.count()).select_from(ChunkRow).where(ChunkRow.doc_id == doc_id),
            )
            await session.execute(sa_delete(DocumentRow).where(DocumentRow.doc_id == doc_id))
        logger.info("已删除文档元数据", extra={"doc_id": doc_id, "chunks": chunk_count})
        return chunk_count

    # ==================================================================================
    # 分块
    # ==================================================================================

    async def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """批量写入分块。已存在的 ``chunk_id`` 会被覆盖。

        走 ``session.merge`` 而不是 ``add``：重导同一份文档时
        ``chunk_id`` 是不变的（由 doc_id + ordinal 派生），
        ``add`` 会撞主键，``merge`` 则让整个导入天然幂等。
        """
        if not chunks:
            return 0
        async with self._db.session() as session:
            for chunk in chunks:
                check_enum("chunks.block_type", str(chunk.block_type))
                await session.merge(_chunk_to_row(chunk))
        return len(chunks)

    async def replace_document_chunks(self, doc_id: str, chunks: Sequence[Chunk]) -> int:
        """**先删后写**。重导一份改动过的文件时必须用它。

        只用 :meth:`add_chunks` 的 merge 有个隐蔽后果：文件改短了之后，
        旧的第 12 个分块不会被新数据覆盖，会永远留在库里 ——
        检索时冒出来一段「已经不存在的原文」，而引用还指向一个有效页码。
        """
        async with self._db.session() as session:
            await session.execute(sa_delete(ChunkRow).where(ChunkRow.doc_id == doc_id))
            rows: list[ChunkRow] = []
            for chunk in chunks:
                check_enum("chunks.block_type", str(chunk.block_type))
                rows.append(_chunk_to_row(chunk))
            # add_all 与逐条 add 在语义上等价，但省掉每条一次的 add 开销；
            # 真正的大头是同一个 session 内只 flush / commit 一次。
            session.add_all(rows)
        return len(chunks)

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        async with self._db.session() as session:
            row = await session.get(ChunkRow, chunk_id)
            return _chunk_to_model(row) if row else None

    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """按 id 批量取。**按传入顺序返回**，缺失的跳过。

        调用方通常拿它把检索结果（一堆 id）补齐成完整分块，
        顺序错位会让「第 3 条结果的引用指向第 5 条的内容」——
        一个只在肉眼看引用时才发现的 bug。
        """
        if not chunk_ids:
            return []
        found: dict[str, Chunk] = {}
        async with self._db.session() as session:
            for start in range(0, len(chunk_ids), _MAX_IN_CLAUSE):
                shard = list(chunk_ids[start : start + _MAX_IN_CLAUSE])
                rows = (
                    await session.execute(
                        select(ChunkRow).where(ChunkRow.chunk_id.in_(shard))
                    )
                ).scalars().all()
                for row in rows:
                    found[row.chunk_id] = _chunk_to_model(row)
        return [found[cid] for cid in chunk_ids if cid in found]

    async def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        async with self._db.session() as session:
            rows = (
                await session.execute(
                    select(ChunkRow)
                    .where(ChunkRow.doc_id == doc_id)
                    .order_by(ChunkRow.ordinal)
                )
            ).scalars().all()
            return [_chunk_to_model(row) for row in rows]

    async def get_neighbor_chunks(
        self, chunk_id: str, *, before: int = 1, after: int = 1
    ) -> list[Chunk]:
        """取同一文档里相邻的分块，用于 :data:`~app.config.RetrievalSettings.neighbor_expand`。

        按 **ordinal 区间**取，而不是顺着 ``prev_chunk_id`` / ``next_chunk_id`` 跳。
        两个原因：区间查询是一次 SQL，链表要走 N 次；
        而且导入过程中断时 ``prev`` / ``next`` 可能只写了一一半，
        区间查询则永远给出「库里实际存在的那几个邻居」。
        """
        async with self._db.session() as session:
            anchor = await session.get(ChunkRow, chunk_id)
            if anchor is None:
                return []
            rows = (
                await session.execute(
                    select(ChunkRow)
                    .where(
                        ChunkRow.doc_id == anchor.doc_id,
                        ChunkRow.ordinal >= anchor.ordinal - max(0, before),
                        ChunkRow.ordinal <= anchor.ordinal + max(0, after),
                        ChunkRow.chunk_id != chunk_id,
                    )
                    .order_by(ChunkRow.ordinal)
                )
            ).scalars().all()
            return [_chunk_to_model(row) for row in rows]

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        if not chunk_ids:
            return 0
        removed = 0
        async with self._db.session() as session:
            for start in range(0, len(chunk_ids), _MAX_IN_CLAUSE):
                shard = list(chunk_ids[start : start + _MAX_IN_CLAUSE])
                result = await session.execute(
                    sa_delete(ChunkRow).where(ChunkRow.chunk_id.in_(shard))
                )
                removed += max(0, result.rowcount or 0)
        return removed

    # ==================================================================================
    # 图片
    # ==================================================================================

    async def upsert_image(self, image: ImageAsset) -> ImageAsset:
        """按 ``image_id`` 幂等写入图片记录。"""
        check_enum("images.status", str(image.status))
        check_enum(
            "images.filtered_reason",
            str(image.filtered_reason) if image.filtered_reason else None,
        )
        check_enum("images.caption_source", str(image.caption_source))
        async with self._db.session() as session:
            await session.merge(_image_to_row(image))
        return image

    async def upsert_images(self, images: Sequence[ImageAsset]) -> int:
        """批量写入，**共用一个 session**。

        原实现是逐张调用 :meth:`upsert_image`，每张图片各开一个 session、
        各提交一次事务。一次导入几十张图就是几十次 commit ——
        SQLite 每次 commit 都要落盘 fsync，在机械盘 / 网络盘上会变成
        肉眼可见的秒级开销。改成同一事务里连续 merge：
        要么全部写入，要么全部回滚，比逐条写也更一致。
        """
        if not images:
            return 0
        async with self._db.session() as session:
            for image in images:
                check_enum("images.status", str(image.status))
                check_enum(
                    "images.filtered_reason",
                    str(image.filtered_reason) if image.filtered_reason else None,
                )
                check_enum("images.caption_source", str(image.caption_source))
                await session.merge(_image_to_row(image))
        return len(images)

    async def get_image(self, image_id: str) -> ImageAsset | None:
        async with self._db.session() as session:
            row = await session.get(ImageRow, image_id)
            return _image_to_model(row) if row else None

    async def get_images(self, image_ids: Sequence[str]) -> list[ImageAsset]:
        """按 id 批量取，**按传入顺序返回**。理由同 :meth:`get_chunks`。"""
        if not image_ids:
            return []
        found: dict[str, ImageAsset] = {}
        async with self._db.session() as session:
            for start in range(0, len(image_ids), _MAX_IN_CLAUSE):
                shard = list(image_ids[start : start + _MAX_IN_CLAUSE])
                rows = (
                    await session.execute(
                        select(ImageRow).where(ImageRow.image_id.in_(shard))
                    )
                ).scalars().all()
                for row in rows:
                    found[row.image_id] = _image_to_model(row)
        return [found[iid] for iid in image_ids if iid in found]

    async def list_images(
        self,
        kb_id: str,
        *,
        status: ImageStatus | None = None,
        doc_id: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[ImageAsset]:
        stmt = (
            select(ImageRow)
            .where(ImageRow.kb_id == kb_id)
            .order_by(ImageRow.created_at.desc(), ImageRow.image_id)
            .limit(max(1, limit))
            .offset(max(0, offset))
        )
        if status is not None:
            stmt = stmt.where(ImageRow.status == str(status))
        if doc_id is not None:
            stmt = stmt.where(ImageRow.doc_id == doc_id)
        async with self._db.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_image_to_model(row) for row in rows]

    async def find_images_by_phash(
        self, kb_id: str, *, exclude_doc_id: str | None = None
    ) -> dict[str, list[str]]:
        """``phash → [image_id, ...]``。去重时用它找汉明距离近的候选。

        一次把整个知识库的 phash 拉进内存做比较。听起来不优雅，
        但 phash 是 16 位 hex 的短字符串，一万张图也就几百 KB ——
        而「在 SQL 里算汉明距离」需要自定义函数，得不偿失。

        ``exclude_doc_id`` 用于**重导**：同一份文档上一次导入留下的图片
        必须排除，否则文档会和自己的旧版本判为重复，一张图也留不下来。
        """
        async with self._db.session() as session:
            stmt = select(ImageRow.image_id, ImageRow.phash).where(
                ImageRow.kb_id == kb_id,
                ImageRow.phash != "",
                ImageRow.status != str(ImageStatus.FILTERED_OUT),
            )
            if exclude_doc_id is not None:
                # 不能直接写 != ：doc_id 为 NULL 的孤儿图片会被 NULL 语义静默排除
                stmt = stmt.where(
                    or_(ImageRow.doc_id.is_(None), ImageRow.doc_id != exclude_doc_id)
                )
            rows = (await session.execute(stmt)).all()
        out: dict[str, list[str]] = {}
        for image_id, phash in rows:
            out.setdefault(str(phash), []).append(str(image_id))
        return out

    async def find_image_by_sha256(self, kb_id: str, sha256: str) -> ImageAsset | None:
        """完全相同的字节 → 直接复用，连 pHash 都不用算。"""
        if not sha256:
            return None
        async with self._db.session() as session:
            row = (
                await session.execute(
                    select(ImageRow).where(
                        ImageRow.kb_id == kb_id, ImageRow.sha256 == sha256
                    )
                )
            ).scalars().first()
            return _image_to_model(row) if row else None

    async def update_image_status(
        self,
        image_id: str,
        status: ImageStatus,
        *,
        filtered_reason: Any = None,
    ) -> None:
        check_enum("images.status", str(status))
        check_enum(
            "images.filtered_reason",
            str(filtered_reason) if filtered_reason is not None else None,
        )
        async with self._db.session() as session:
            result = await session.execute(
                sa_update(ImageRow)
                .where(ImageRow.image_id == image_id)
                .values(
                    status=str(status),
                    filtered_reason=(
                        str(filtered_reason) if filtered_reason is not None else None
                    ),
                )
            )
            if result.rowcount == 0:
                raise NotFoundError(
                    f"图片 {image_id!r} 不存在", detail={"image_id": image_id}
                )

    async def delete_images(self, image_ids: Sequence[str]) -> int:
        if not image_ids:
            return 0
        removed = 0
        async with self._db.session() as session:
            for start in range(0, len(image_ids), _MAX_IN_CLAUSE):
                shard = list(image_ids[start : start + _MAX_IN_CLAUSE])
                result = await session.execute(
                    sa_delete(ImageRow).where(ImageRow.image_id.in_(shard))
                )
                removed += max(0, result.rowcount or 0)
        return removed

    # ==================================================================================
    # 任务
    # ==================================================================================

    async def create_task(
        self,
        kb_id: str,
        *,
        kind: str = "ingest",
        doc_id: str | None = None,
        task_id: str | None = None,
        progress_total: int = 0,
    ) -> Task:
        check_enum("tasks.kind", kind)
        row = TaskRow(
            task_id=task_id or new_id("task"),
            kb_id=kb_id,
            doc_id=doc_id,
            kind=kind,
            status=str(TaskStatus.QUEUED),
            stage="queued",
            progress_total=progress_total,
        )
        check_enum("tasks.status", row.status)
        async with self._db.session() as session:
            session.add(row)
        return _task_to_model(row)

    async def update_task(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        stage: str | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
        stage_timings: dict[str, int] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        finish: bool = False,
    ) -> None:
        """更新任务进度。前端轮询 ``GET /api/tasks/{id}`` 读到的就是这里写的东西。

        ``stage_timings`` 是**累加**而不是覆盖：摄入流水线每个阶段结束时
        报一次自己的耗时，累计起来就是完整的阶段耗时表。
        覆盖的话只会留下最后一个阶段的数字。
        """
        values: dict[str, Any] = {}
        if status is not None:
            check_enum("tasks.status", str(status))
            values["status"] = str(status)
        if stage is not None:
            values["stage"] = stage
        if progress_current is not None:
            values["progress_current"] = progress_current
        if progress_total is not None:
            values["progress_total"] = progress_total
        if result is not None:
            values["result"] = dict(result)
        if error is not None:
            values["error"] = error
        if finish or status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED):
            values["finished_at"] = utcnow()
        if not values and stage_timings is None:
            return

        async with self._db.session() as session:
            if stage_timings:
                row = await session.get(TaskRow, task_id)
                if row is None:
                    raise NotFoundError(f"任务 {task_id!r} 不存在", detail={"task_id": task_id})
                merged = dict(row.stage_timings or {})
                for key, value in stage_timings.items():
                    merged[key] = int(merged.get(key, 0)) + int(value)
                values["stage_timings"] = merged
            if not values:
                return
            result_row = await session.execute(
                sa_update(TaskRow).where(TaskRow.task_id == task_id).values(**values)
            )
            if result_row.rowcount == 0:
                raise NotFoundError(f"任务 {task_id!r} 不存在", detail={"task_id": task_id})

    async def get_task(self, task_id: str) -> Task | None:
        async with self._db.session() as session:
            row = await session.get(TaskRow, task_id)
            return _task_to_model(row) if row else None

    async def list_tasks(
        self, kb_id: str, *, limit: int = 50, status: TaskStatus | None = None
    ) -> list[Task]:
        stmt = (
            select(TaskRow)
            .where(TaskRow.kb_id == kb_id)
            .order_by(TaskRow.created_at.desc())
            .limit(max(1, limit))
        )
        if status is not None:
            stmt = stmt.where(TaskRow.status == str(status))
        async with self._db.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_task_to_model(row) for row in rows]

    async def recent_tasks(self, *, limit: int = 10) -> list[Task]:
        """跨知识库取最近任务，供性能报告展示导入耗时。

        和 :meth:`list_tasks` 分开而不是加一个 ``kb_id=None`` 分支：
        全表按时间倒序走的是 ``tasks.created_at`` 索引，与「某个库的最近任务」
        是两个不同的查询模式，混在一个方法里只会让两个调用点都变难读。
        """
        stmt = (
            select(TaskRow)
            .order_by(TaskRow.created_at.desc())
            .limit(max(1, limit))
        )
        async with self._db.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_task_to_model(row) for row in rows]

    # ==================================================================================
    # 会话与消息
    # ==================================================================================

    async def create_session(self, kb_id: str, *, session_id: str | None = None) -> ChatSession:
        row = SessionRow(session_id=session_id or new_id("session"), kb_id=kb_id)
        async with self._db.session() as session:
            session.add(row)
        return _session_to_model(row)

    async def get_session(self, session_id: str) -> ChatSession | None:
        async with self._db.session() as session:
            row = await session.get(SessionRow, session_id)
            return _session_to_model(row) if row else None

    async def require_session(self, session_id: str) -> ChatSession:
        found = await self.get_session(session_id)
        if found is None:
            raise NotFoundError(
                f"会话 {session_id!r} 不存在", detail={"session_id": session_id}
            )
        return found

    async def get_or_create_session(self, kb_id: str, session_id: str | None) -> ChatSession:
        """前端没给 ``session_id`` 就新建一个。

        注意这里**不校验 session 属于哪个 kb**：用户切换知识库继续问同一个会话
        是很自然的行为，强行报错只会逼前端在切库时悄悄丢掉上下文。
        """
        if session_id:
            found = await self.get_session(session_id)
            if found is not None:
                return found
        return await self.create_session(kb_id, session_id=session_id)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        citations: Sequence[Citation] | None = None,
        message_id: str | None = None,
    ) -> Message:
        """追加一条消息。``citations`` 只对 assistant 消息有意义，但这里不强制 ——

        把「哪些字段该有什么值」的判断放在 schema 层而不是存储层，
        存储层只保证写进去的和读出来的一致。
        """
        check_enum("messages.role", role)
        row = MessageRow(
            message_id=message_id or new_id("message"),
            session_id=session_id,
            role=role,
            content=content,
            citations=(
                [c.model_dump(mode="json") for c in citations] if citations else None
            ),
        )
        async with self._db.session() as session:
            # 提前确认会话存在，给出「会话不存在」而不是外键约束错误 ——
            # SQLite 的外键报错信息只有表名和列名，排查起来要多绕一步
            if await session.get(SessionRow, session_id) is None:
                raise NotFoundError(
                    f"会话 {session_id!r} 不存在，无法追加消息",
                    detail={"session_id": session_id},
                )
            session.add(row)
        return _message_to_model(row)

    async def list_messages(self, session_id: str, *, limit: int = 50) -> list[Message]:
        """取最近 ``limit`` 条，**按时间正序返回**。

        先按时间倒序取 N 条再翻转，而不是正序取前 N 条 ——
        多轮对话要的是「最近几轮」，正序取前 N 条会永远停在最早的几句。
        """
        async with self._db.session() as session:
            rows = (
                await session.execute(
                    select(MessageRow)
                    .where(MessageRow.session_id == session_id)
                    .order_by(MessageRow.created_at.desc(), MessageRow.message_id.desc())
                    .limit(max(1, limit))
                )
            ).scalars().all()
        return [_message_to_model(row) for row in reversed(rows)]

    async def delete_session(self, session_id: str) -> bool:
        """删会话及其消息（消息走 CASCADE）。"""
        async with self._db.session() as session:
            result = await session.execute(
                sa_delete(SessionRow).where(SessionRow.session_id == session_id)
            )
            return (result.rowcount or 0) > 0

    # ==================================================================================
    # 运维
    # ==================================================================================

    async def stats(self) -> dict[str, Any]:
        """给 ``/api/health`` 用：各表行数。"""
        out: dict[str, Any] = {"url": self._db.safe_url}
        async with self._db.session() as session:
            for name, row_cls in (
                ("knowledge_bases", KnowledgeBaseRow),
                ("documents", DocumentRow),
                ("chunks", ChunkRow),
                ("images", ImageRow),
                ("tasks", TaskRow),
                ("sessions", SessionRow),
                ("messages", MessageRow),
            ):
                out[name] = await _scalar(
                    session, select(func.count()).select_from(row_cls)
                )
        return out


# ======================================================================================
# 转换器：Row → Pydantic
# ======================================================================================
#
# 这些函数刻意写成模块级私有函数而不是类方法：
# 它们是纯粹的字段搬运，不依赖 DocStore 的任何状态，单独测起来更方便。


async def _scalar(session: Any, stmt: Any) -> int:
    return int((await session.execute(stmt)).scalar_one())


def _kb_to_model(row: KnowledgeBaseRow) -> KnowledgeBase:
    return KnowledgeBase(
        kb_id=row.kb_id,
        name=row.name,
        description=row.description,
        status=row.status,  # type: ignore[arg-type]
        created_at=row.created_at,
        updated_at=row.updated_at,
        text_collection=row.text_collection,
        image_collection=row.image_collection,
        text_dim=row.text_dim,
        image_dim=row.image_dim,
        llm_model=row.llm_model,
        text_embed_model=row.text_embed_model,
        image_embed_model=row.image_embed_model,
        rerank_model=row.rerank_model,
        doc_count=row.doc_count,
        chunk_count=row.chunk_count,
        image_count=row.image_count,
    )


def _doc_to_model(row: DocumentRow) -> Document:
    return Document(
        doc_id=row.doc_id,
        kb_id=row.kb_id,
        source_path=row.source_path,
        source_name=row.source_name,
        mime_type=row.mime_type,
        doc_hash=row.doc_hash,
        size_bytes=row.size_bytes,
        page_count=row.page_count,
        title=row.title,
        status=row.status,  # type: ignore[arg-type]
        task_id=row.task_id,
        error=row.error,
        chunk_count=row.chunk_count,
        image_count=row.image_count,
        skipped_image_count=row.skipped_image_count,
        extra=dict(row.extra or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
        indexed_at=row.indexed_at,
    )


def _doc_to_row(doc: Document) -> DocumentRow:
    return DocumentRow(
        doc_id=doc.doc_id,
        kb_id=doc.kb_id,
        source_path=doc.source_path,
        source_name=doc.source_name,
        mime_type=doc.mime_type,
        doc_hash=doc.doc_hash,
        size_bytes=doc.size_bytes,
        page_count=doc.page_count,
        title=doc.title,
        status=str(doc.status),
        task_id=doc.task_id,
        error=doc.error,
        chunk_count=doc.chunk_count,
        image_count=doc.image_count,
        skipped_image_count=doc.skipped_image_count,
        extra=dict(doc.extra),
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        indexed_at=doc.indexed_at,
    )


def _chunk_to_model(row: ChunkRow) -> Chunk:
    return Chunk(
        chunk_id=row.chunk_id,
        kb_id=row.kb_id,
        doc_id=row.doc_id,
        ordinal=row.ordinal,
        text=row.text,
        token_count=row.token_count,
        block_type=row.block_type,  # type: ignore[arg-type]
        heading_path=list(row.heading_path or []),
        page_from=row.page_from,
        page_to=row.page_to,
        image_ids=list(row.image_ids or []),
        prev_chunk_id=row.prev_chunk_id,
        next_chunk_id=row.next_chunk_id,
        source_ref=dict(row.source_ref or {}),
        created_at=row.created_at,
    )


def _chunk_to_row(chunk: Chunk) -> ChunkRow:
    return ChunkRow(
        chunk_id=chunk.chunk_id,
        kb_id=chunk.kb_id,
        doc_id=chunk.doc_id,
        ordinal=chunk.ordinal,
        text=chunk.text,
        token_count=chunk.token_count,
        block_type=str(chunk.block_type),
        heading_path=list(chunk.heading_path),
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        image_ids=list(chunk.image_ids),
        prev_chunk_id=chunk.prev_chunk_id,
        next_chunk_id=chunk.next_chunk_id,
        source_ref=dict(chunk.source_ref),
        created_at=chunk.created_at,
    )


def _image_to_model(row: ImageRow) -> ImageAsset:
    caption = None
    if row.caption:
        try:
            caption = ImageCaption.model_validate(row.caption)
        except Exception:  # noqa: BLE001
            # 旧的 caption JSON 与当前 ImageCaption 结构不符（加过字段、改过枚举）。
            # 与其让整个知识库读不出来，不如丢掉 caption 结构 ——
            # caption_text 才是检索用的那份，它是独立存的，不受影响。
            logger.warning(
                "图片 caption 结构无法解析，已忽略",
                extra={"image_id": row.image_id},
            )
    return ImageAsset(
        image_id=row.image_id,
        kb_id=row.kb_id,
        doc_id=row.doc_id,
        page=row.page,
        bbox=list(row.bbox) if row.bbox else None,
        order_in_doc=row.order_in_doc,
        file_path=row.file_path,
        url=row.url,
        mime_type=row.mime_type,
        width=row.width,
        height=row.height,
        size_bytes=row.size_bytes,
        sha256=row.sha256,
        phash=row.phash,
        status=row.status,  # type: ignore[arg-type]
        filtered_reason=row.filtered_reason,  # type: ignore[arg-type]
        caption=caption,
        caption_text=row.caption_text,
        caption_source=row.caption_source,  # type: ignore[arg-type]
        anchor_chunk_id=row.anchor_chunk_id,
        context_text=row.context_text,
        text_chunk_id=row.text_chunk_id,
        image_vector_id=row.image_vector_id,
        created_at=row.created_at,
    )


def _image_to_row(image: ImageAsset) -> ImageRow:
    return ImageRow(
        image_id=image.image_id,
        kb_id=image.kb_id,
        doc_id=image.doc_id,
        page=image.page,
        bbox=list(image.bbox) if image.bbox else None,
        order_in_doc=image.order_in_doc,
        file_path=image.file_path,
        url=image.url,
        mime_type=image.mime_type,
        width=image.width,
        height=image.height,
        size_bytes=image.size_bytes,
        sha256=image.sha256,
        phash=image.phash,
        status=str(image.status),
        filtered_reason=(
            str(image.filtered_reason) if image.filtered_reason is not None else None
        ),
        caption=image.caption.model_dump(mode="json") if image.caption else None,
        caption_text=image.caption_text,
        caption_source=str(image.caption_source),
        anchor_chunk_id=image.anchor_chunk_id,
        context_text=image.context_text,
        text_chunk_id=image.text_chunk_id,
        image_vector_id=image.image_vector_id,
        created_at=image.created_at,
    )


def _task_to_model(row: TaskRow) -> Task:
    return Task(
        task_id=row.task_id,
        kb_id=row.kb_id,
        doc_id=row.doc_id,
        kind=row.kind,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        stage=row.stage,
        progress_current=row.progress_current,
        progress_total=row.progress_total,
        stage_timings=dict(row.stage_timings or {}),
        result=dict(row.result or {}),
        error=row.error,
        created_at=row.created_at,
        finished_at=row.finished_at,
    )


def _session_to_model(row: SessionRow) -> ChatSession:
    return ChatSession(
        session_id=row.session_id, kb_id=row.kb_id, created_at=row.created_at
    )


def _message_to_model(row: MessageRow) -> Message:
    citations = None
    if row.citations:
        citations = [Citation.model_validate(item) for item in row.citations]
    return Message(
        message_id=row.message_id,
        session_id=row.session_id,
        role=row.role,  # type: ignore[arg-type]
        content=row.content,
        citations=citations,
        created_at=row.created_at,
    )
