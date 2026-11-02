"""存储层测试。

全部在临时目录里跑，**不碰 data/ 下的真实数据，也不需要任何 API Key**，
所以可以直接进 CI。

覆盖的四类问题，共同点是「错了不会立刻被发现」：

1. **级联删除** —— ``PRAGMA foreign_keys`` 默认是 OFF 的，
   忘了开的话删知识库只删掉一行，剩下的分块变成永远查不到的孤儿数据，
   磁盘还一直涨；
2. **幂等 upsert** —— 重复导入必须收敛到同一份数据，
   否则同一段文字在检索结果里出现两次，把别的候选挤下去；
3. **向量空间隔离（铁律 3）** —— 两个 collection 的维度恰好相同时，
   光靠维度检查发现不了写错了空间；
4. **时间戳时区** —— ``datetime`` 存进 SQLite 再读出来丢了 tzinfo，
   会让 ``created_after`` 这类过滤静默地少算 8 小时。
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.config import (
    BlobSettings,
    BM25Settings,
    DocStoreSettings,
    PathSettings,
    Settings,
    VectorStoreSettings,
)
from app.core.errors import NotFoundError, StorageError
from app.core.schemas import (
    BlockType,
    Chunk,
    DocStatus,
    Document,
    ImageAsset,
    ImageStatus,
    TaskStatus,
    utcnow,
)
from app.store.blobstore import BlobStore
from app.store.bm25store import BM25Doc, BM25Store, tokenize
from app.store.db import Database
from app.store.docstore import DocStore
from app.store.models import UTCDateTime
from app.store.vectorstore import (
    VectorRecord,
    VectorSpaceMismatchError,
    image_collection,
    text_collection,
)
from app.store.vectorstore_chroma import ChromaVectorStore

# --------------------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------------------


def _settings(tmp_path: Path, **vectorstore: Any) -> Settings:
    """一套完全指向 tmp_path 的配置。不读 .env，不依赖真实 API Key。"""
    return Settings(
        paths=PathSettings(data_dir=tmp_path),
        docstore=DocStoreSettings(sqlite_path=tmp_path / "test.db"),
        blob=BlobSettings(images_dir=tmp_path / "images"),
        bm25=BM25Settings(index_dir=tmp_path / "bm25", persist=True),
        vectorstore=VectorStoreSettings(chroma_dir=tmp_path / "chroma", **vectorstore),
    )


@pytest.fixture
async def db(tmp_path: Path) -> Any:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}", is_sqlite=True)
    await database.init_schema()
    yield database
    await database.dispose()


@pytest.fixture
async def store(db: Database, tmp_path: Path) -> DocStore:
    return DocStore(_settings(tmp_path), db=db)


@pytest.fixture
async def kb(store: DocStore) -> Any:
    return await store.create_kb(
        "测试知识库",
        kb_id="kb_teststore01",
        text_collection=text_collection("kb_teststore01"),
        image_collection=image_collection("kb_teststore01"),
        text_dim=4,
        image_dim=8,
        llm_model="deepseek-flash",
        text_embed_model="text-embedding-v4",
        image_embed_model="qwen3-vl-embedding",
    )


def _chunk(kb_id: str, doc_id: str, ordinal: int, text: str = "内容") -> Chunk:
    # chunk_id 必须由**整个** doc_id 派生，不能只取后几位 ——
    # 截断会让 doc_xxx000001 与 doc_yyy000001 生成同一个 id，
    # 后写的把先写的顶掉，症状是「某个文档凭空少了几个分块」
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8]
    return Chunk(
        chunk_id=f"ck_{digest}{ordinal:04d}",
        kb_id=kb_id,
        doc_id=doc_id,
        ordinal=ordinal,
        text=f"{text} #{ordinal}",
        token_count=len(text),
        heading_path=["第一章", f"1.{ordinal}"],
    )


# --------------------------------------------------------------------------------------
# 1. UTCDateTime 类型
# --------------------------------------------------------------------------------------


def test_utc_datetime_rejects_naive_values() -> None:
    """naive datetime 必须炸。

    放进去的话，SQLite 会原样存下这个「没有时区」的时间，
    而读出来时我们会给它补上 UTC —— 于是北京时间 10:00 变成了 UTC 10:00，
    整整差 8 小时，且没有任何报错。
    """
    column = UTCDateTime()
    with pytest.raises(ValueError, match="naive"):
        column.process_bind_param(datetime(2026, 1, 1, 10, 0), None)  # noqa: DTZ001


def test_utc_datetime_roundtrip_preserves_instant() -> None:
    """带 +08:00 的时间存进去，读出来应当是**同一时刻**的 UTC。"""
    column = UTCDateTime()
    original = datetime(2026, 1, 1, 18, 0, tzinfo=timezone(timedelta(hours=8)))
    stored = column.process_bind_param(original, None)
    restored = column.process_result_value(stored, None)

    assert restored.tzinfo is UTC
    assert restored == original
    assert restored.hour == 10  # 18:00+08:00 == 10:00Z


# --------------------------------------------------------------------------------------
# 2. 知识库 CRUD
# --------------------------------------------------------------------------------------


async def test_create_and_read_kb_roundtrip(store: DocStore, kb: Any) -> None:
    loaded = await store.require_kb(kb.kb_id)
    assert loaded.name == "测试知识库"
    assert loaded.text_dim == 4 and loaded.image_dim == 8
    assert loaded.text_collection == "kb_teststore01__text"
    assert loaded.image_collection == "kb_teststore01__image"
    assert loaded.status == "active"
    assert loaded.created_at.tzinfo is UTC


async def test_require_kb_raises_not_found(store: DocStore) -> None:
    with pytest.raises(NotFoundError):
        await store.require_kb("kb_doesnotexist")


async def test_refresh_kb_stats_counts_real_rows(store: DocStore, kb: Any) -> None:
    doc, _ = await store.upsert_document(
        Document(
            doc_id="doc_stat000001",
            kb_id=kb.kb_id,
            source_path="/tmp/a.md",
            source_name="a.md",
            mime_type="text/markdown",
            doc_hash="hash-a",
        )
    )
    await store.add_chunks([_chunk(kb.kb_id, doc.doc_id, i) for i in range(3)])
    await store.upsert_image(
        ImageAsset(
            image_id="img_stat000001",
            kb_id=kb.kb_id,
            doc_id=doc.doc_id,
            status=ImageStatus.DONE,
            caption_text="一张图",
        )
    )

    refreshed = await store.refresh_kb_stats(kb.kb_id)
    assert refreshed.doc_count == 1
    assert refreshed.chunk_count == 3
    assert refreshed.image_count == 1


async def test_delete_kb_cascades_to_children(store: DocStore, db: Database, kb: Any) -> None:
    """删知识库必须把分块/图片/会话/消息一并带走。

    这是 ``PRAGMA foreign_keys=ON`` 是否真的生效的**唯一**验证点：
    pragma 关掉时这条 SQL 照样执行成功，只是什么都不级联。
    """
    doc, _ = await store.upsert_document(
        Document(
            doc_id="doc_casc000001",
            kb_id=kb.kb_id,
            source_path="/tmp/b.md",
            source_name="b.md",
            mime_type="text/markdown",
            doc_hash="hash-b",
        )
    )
    chunks = [_chunk(kb.kb_id, doc.doc_id, i) for i in range(5)]
    await store.add_chunks(chunks)
    await store.upsert_image(
        ImageAsset(image_id="img_casc000001", kb_id=kb.kb_id, doc_id=doc.doc_id)
    )
    session = await store.create_session(kb.kb_id)
    await store.add_message(session.session_id, "user", "你好")
    await store.create_task(kb.kb_id)
    # 另建一个知识库，用来证明删除**没有**误伤别人
    other = await store.create_kb(
        "别的库",
        kb_id="kb_other00001",
        text_collection=text_collection("kb_other00001"),
        image_collection=image_collection("kb_other00001"),
        text_dim=4,
        image_dim=8,
    )
    other_doc, _ = await store.upsert_document(
        Document(
            doc_id="doc_surv000001",
            kb_id=other.kb_id,
            source_path="/tmp/c.md",
            source_name="c.md",
            mime_type="text/markdown",
            doc_hash="hash-c",
        )
    )
    await store.add_chunks([_chunk(other.kb_id, other_doc.doc_id, 0)])

    counts = await store.delete_kb(kb.kb_id)

    assert counts["chunks"] == 5
    assert counts["images"] == 1
    assert counts["sessions"] == 1
    assert counts["messages"] == 1
    assert counts["documents"] == 1
    assert counts["tasks"] == 1  # tasks 没有外键，靠显式 DELETE

    assert await store.get_kb(kb.kb_id) is None
    assert await store.get_chunk(chunks[0].chunk_id) is None
    # 邻居完好无损
    assert await store.get_kb(other.kb_id) is not None
    assert len(await store.get_document_chunks(other_doc.doc_id)) == 1

    # 直接数一遍表，确认没有孤儿行 —— counts 是我们自己算的，不能自证。
    # 注意要**按 kb_id 数**：另一个知识库的数据必须原样留着，
    # 无条件数全表会把「删对了」和「删多了」两种情况都判成失败。
    async with db.session() as session:
        from sqlalchemy import func, select

        from app.store.models import ChunkRow, ImageRow, SessionRow, TaskRow

        for row_cls in (ChunkRow, ImageRow, SessionRow, TaskRow):
            remaining = (
                await session.execute(
                    select(func.count()).select_from(row_cls).where(row_cls.kb_id == kb.kb_id)
                )
            ).scalar_one()
            assert remaining == 0, f"{row_cls.__name__} 在被删的知识库里留下了孤儿行"

        survivor_chunks = (
            await session.execute(
                select(func.count()).select_from(ChunkRow).where(ChunkRow.kb_id == other.kb_id)
            )
        ).scalar_one()
        assert survivor_chunks == 1  # 邻居的分块一根都没少


# --------------------------------------------------------------------------------------
# 3. 文档：幂等 upsert
# --------------------------------------------------------------------------------------


async def test_upsert_document_is_idempotent_on_hash(store: DocStore, kb: Any) -> None:
    """同一个 ``doc_hash`` 导入两次，必须收敛到同一个 doc_id。"""
    payload = {
        "kb_id": kb.kb_id,
        "source_path": "/tmp/x.md",
        "source_name": "x.md",
        "mime_type": "text/markdown",
        "doc_hash": "same-hash",
    }
    first, created_first = await store.upsert_document(
        Document(doc_id="doc_first00001", **payload)
    )
    second, created_second = await store.upsert_document(
        Document(doc_id="doc_second0001", **payload)
    )

    assert created_first is True
    assert created_second is False
    # 第二次返回的是**既有**记录，而不是新写进去的那条
    assert second.doc_id == first.doc_id == "doc_first00001"


async def test_find_document_by_hash_finds_existing(store: DocStore, kb: Any) -> None:
    await store.upsert_document(
        Document(
            doc_id="doc_find000001",
            kb_id=kb.kb_id,
            source_path="/tmp/y.md",
            source_name="y.md",
            mime_type="text/markdown",
            doc_hash="find-me",
        )
    )
    assert (await store.find_document_by_hash(kb.kb_id, "find-me")) is not None
    assert (await store.find_document_by_hash(kb.kb_id, "nope")) is None


async def test_replace_document_chunks_removes_stale_tail(store: DocStore, kb: Any) -> None:
    """文件改短后重导，多余的老分块必须消失。

    只靠 merge 的话第 3 个分块会留下来，检索时冒出一段「已经不存在的原文」，
    引用还指向一个有效页码 —— 没有任何迹象表明它是过期的。
    """
    doc, _ = await store.upsert_document(
        Document(
            doc_id="doc_trunc00001",
            kb_id=kb.kb_id,
            source_path="/tmp/z.md",
            source_name="z.md",
            mime_type="text/markdown",
            doc_hash="trunc",
        )
    )
    await store.replace_document_chunks(
        doc.doc_id, [_chunk(kb.kb_id, doc.doc_id, i) for i in range(5)]
    )
    assert len(await store.get_document_chunks(doc.doc_id)) == 5

    await store.replace_document_chunks(
        doc.doc_id, [_chunk(kb.kb_id, doc.doc_id, i) for i in range(2)]
    )
    remaining = await store.get_document_chunks(doc.doc_id)
    assert [c.ordinal for c in remaining] == [0, 1]


async def test_update_document_status_only_touches_given_fields(
    store: DocStore, kb: Any
) -> None:
    doc, _ = await store.upsert_document(
        Document(
            doc_id="doc_patch00001",
            kb_id=kb.kb_id,
            source_path="/tmp/p.md",
            source_name="p.md",
            mime_type="text/markdown",
            doc_hash="patch",
            title="原标题",
        )
    )
    await store.update_document_status(doc.doc_id, DocStatus.EMBEDDING, chunk_count=7)
    await store.update_document_status(doc.doc_id, DocStatus.DONE)

    loaded = await store.require_document(doc.doc_id)
    assert loaded.status == DocStatus.DONE
    assert loaded.chunk_count == 7  # 第一次写的，第二次没传就不该被清掉
    assert loaded.title == "原标题"
    assert loaded.indexed_at is not None  # DONE 时自动盖时间戳


async def test_update_document_status_rejects_unknown_status(
    store: DocStore, kb: Any
) -> None:
    """枚举列必须校验 —— 否则 ``embeddingg`` 会安静地写进去。"""
    with pytest.raises(ValueError, match="不是合法取值"):
        await store.update_document_status("doc_whatever", "embeddingg")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# 4. 分块：顺序与邻居
# --------------------------------------------------------------------------------------


async def test_get_chunks_returns_in_requested_order(store: DocStore, kb: Any) -> None:
    """``get_chunks`` 必须按传入顺序返回。

    顺序错位会让「第 3 条检索结果的引用指向第 5 条的内容」，
    而两段文字的分数都是对的 —— 只有肉眼看引用时才发现。
    """
    doc, _ = await store.upsert_document(
        Document(
            doc_id="doc_order00001",
            kb_id=kb.kb_id,
            source_path="/tmp/o.md",
            source_name="o.md",
            mime_type="text/markdown",
            doc_hash="order",
        )
    )
    chunks = [_chunk(kb.kb_id, doc.doc_id, i) for i in range(4)]
    await store.add_chunks(chunks)

    requested = [chunks[3].chunk_id, chunks[0].chunk_id, chunks[2].chunk_id]
    found = await store.get_chunks(requested)
    assert [c.chunk_id for c in found] == requested
    # 不存在的 id 被跳过，但其余顺序不变
    assert [c.chunk_id for c in await store.get_chunks([*requested, "ck_nope"])] == requested


async def test_neighbor_expansion_stays_inside_the_document(
    store: DocStore, kb: Any
) -> None:
    doc_a, _ = await store.upsert_document(
        Document(
            doc_id="doc_neigh00001",
            kb_id=kb.kb_id,
            source_path="/tmp/n1.md",
            source_name="n1.md",
            mime_type="text/markdown",
            doc_hash="n1",
        )
    )
    doc_b, _ = await store.upsert_document(
        Document(
            doc_id="doc_neigh00002",
            kb_id=kb.kb_id,
            source_path="/tmp/n2.md",
            source_name="n2.md",
            mime_type="text/markdown",
            doc_hash="n2",
        )
    )
    a_chunks = [_chunk(kb.kb_id, doc_a.doc_id, i) for i in range(3)]
    await store.add_chunks(a_chunks)
    await store.add_chunks([_chunk(kb.kb_id, doc_b.doc_id, 1)])

    # ordinal 是各文档内部独立编号的，所以「同文档」这个条件必须显式带上，
    # 否则会取到另一份文档里 ordinal 相近的分块
    neighbors = await store.get_neighbor_chunks(a_chunks[1].chunk_id, before=1, after=1)
    assert [c.ordinal for c in neighbors] == [0, 2]
    assert {c.doc_id for c in neighbors} == {doc_a.doc_id}


# --------------------------------------------------------------------------------------
# 5. 会话与消息
# --------------------------------------------------------------------------------------


async def test_list_messages_returns_recent_in_chronological_order(
    store: DocStore, kb: Any
) -> None:
    """取「最近 N 条」且正序返回。正序取前 N 条会永远停在最早的几句。"""
    session = await store.create_session(kb.kb_id)
    for i in range(5):
        await store.add_message(session.session_id, "user", f"第 {i} 句")

    recent = await store.list_messages(session.session_id, limit=2)
    assert [m.content for m in recent] == ["第 3 句", "第 4 句"]
    # 时间必须单调不减
    assert recent[0].created_at <= recent[1].created_at


async def test_add_message_requires_existing_session(store: DocStore) -> None:
    """会话不存在时给 NotFoundError，而不是 SQLite 的外键约束错误。"""
    with pytest.raises(NotFoundError, match="会话"):
        await store.add_message("sess_missing", "user", "你好")


async def test_message_citations_roundtrip(store: DocStore, kb: Any) -> None:
    from app.core.schemas import Citation, ItemType

    session = await store.create_session(kb.kb_id)
    await store.add_message(
        session.session_id,
        "assistant",
        "答案见 [1]",
        citations=[
            Citation(
                index=1,
                item_id="ck_abc",
                item_type=ItemType.IMAGE,
                doc_name="论文.pdf",
                page=3,
                image_url="/static/images/kb_x/img_1.png",
            )
        ],
    )
    loaded = (await store.list_messages(session.session_id))[0]
    assert loaded.citations is not None
    assert loaded.citations[0].is_image
    assert loaded.citations[0].image_url == "/static/images/kb_x/img_1.png"


# --------------------------------------------------------------------------------------
# 6. 任务
# --------------------------------------------------------------------------------------


async def test_task_stage_timings_accumulate(store: DocStore, kb: Any) -> None:
    """阶段耗时是累加的。覆盖的话只会留下最后一个阶段的数字。"""
    task = await store.create_task(kb.kb_id, progress_total=10)
    await store.update_task(task.task_id, stage="parsing", stage_timings={"parse": 120})
    await store.update_task(task.task_id, stage="embedding", stage_timings={"embed": 340})
    await store.update_task(task.task_id, stage_timings={"parse": 30})

    loaded = await store.get_task(task.task_id)
    assert loaded is not None
    assert loaded.stage_timings == {"parse": 150, "embed": 340}
    assert loaded.stage == "embedding"


async def test_task_finish_sets_finished_at(store: DocStore, kb: Any) -> None:
    task = await store.create_task(kb.kb_id)
    assert (await store.get_task(task.task_id)).finished_at is None  # type: ignore[union-attr]

    await store.update_task(task.task_id, status=TaskStatus.DONE)
    loaded = await store.get_task(task.task_id)
    assert loaded is not None
    assert loaded.finished_at is not None
    assert loaded.finished_at.tzinfo is UTC


# --------------------------------------------------------------------------------------
# 7. 向量库（Chroma）
# --------------------------------------------------------------------------------------


@pytest.fixture
def chroma(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(_settings(tmp_path))


def _vec(values: list[float], dim: int) -> list[float]:
    return (values + [0.0] * dim)[:dim]


async def test_chroma_upsert_query_and_delete(chroma: ChromaVectorStore) -> None:
    name = text_collection("kb_vec00000001")
    await chroma.ensure_collection(
        name, dim=4, space="text", embed_model="text-embedding-v4"
    )
    await chroma.upsert(
        name,
        [
            VectorRecord(
                id="ck_1",
                vector=[1.0, 0.0, 0.0, 0.0],
                document="第一段",
                metadata={"chunk_id": "ck_1", "doc_id": "doc_1"},
            ),
            VectorRecord(
                id="ck_2",
                vector=[0.0, 1.0, 0.0, 0.0],
                document="第二段",
                metadata={"chunk_id": "ck_2", "doc_id": "doc_2"},
            ),
        ],
        space="text",
    )

    hits = await chroma.query(name, [1.0, 0.0, 0.0, 0.0], top_k=2, space="text")
    assert [h.id for h in hits] == ["ck_1", "ck_2"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[0].document == "第一段"
    assert hits[0].chunk_id == "ck_1"

    # where 过滤
    filtered = await chroma.query(
        name, [1.0, 0.0, 0.0, 0.0], top_k=2, where={"doc_id": "doc_2"}
    )
    assert [h.id for h in filtered] == ["ck_2"]

    assert await chroma.count(name) == 2
    removed = await chroma.delete(name, ids=["ck_1"])
    assert removed == 1
    assert await chroma.count(name) == 1


async def test_chroma_upsert_is_idempotent_by_id(chroma: ChromaVectorStore) -> None:
    """同 id 重写是覆盖而不是新增 —— 导入重跑不能把向量翻倍。"""
    name = text_collection("kb_vec00000002")
    await chroma.ensure_collection(name, dim=4, space="text")
    record = VectorRecord(id="ck_same", vector=[1.0, 0.0, 0.0, 0.0], document="v1")
    await chroma.upsert(name, [record], space="text")
    await chroma.upsert(
        name,
        [VectorRecord(id="ck_same", vector=[0.0, 1.0, 0.0, 0.0], document="v2")],
        space="text",
    )

    assert await chroma.count(name) == 1
    fetched = await chroma.get(name, ["ck_same"])
    assert fetched[0].document == "v2"
    assert fetched[0].vector == [0.0, 1.0, 0.0, 0.0]


async def test_two_spaces_live_in_separate_collections(chroma: ChromaVectorStore) -> None:
    """铁律 3：文本向量与图像向量在两个 collection 里，互不可见。"""
    kb_id = "kb_vec00000003"
    t_name, i_name = text_collection(kb_id), image_collection(kb_id)
    # 故意让两个空间**维度相同**：这样维度检查发现不了写错空间，
    # 唯一能拦住它的是 space 守卫
    await chroma.ensure_collection(t_name, dim=4, space="text")
    await chroma.ensure_collection(i_name, dim=4, space="image")

    await chroma.upsert(
        t_name,
        [VectorRecord(id="ck_t", vector=[1.0, 0.0, 0.0, 0.0], document="文本")],
        space="text",
    )
    await chroma.upsert(
        i_name,
        [VectorRecord(id="img_i", vector=[1.0, 0.0, 0.0, 0.0], document="图片 caption")],
        space="image",
    )

    text_hits = await chroma.query(t_name, [1.0, 0.0, 0.0, 0.0], space="text")
    image_hits = await chroma.query(i_name, [1.0, 0.0, 0.0, 0.0], space="image")
    assert [h.id for h in text_hits] == ["ck_t"]
    assert [h.id for h in image_hits] == ["img_i"]


async def test_writing_to_the_wrong_space_is_rejected(chroma: ChromaVectorStore) -> None:
    """把图像空间的向量写进文本 collection 必须抛错，不能静默接受。"""
    name = text_collection("kb_vec00000004")
    await chroma.ensure_collection(name, dim=4, space="text")
    with pytest.raises(VectorSpaceMismatchError):
        await chroma.upsert(
            name,
            [VectorRecord(id="img_bad", vector=[1.0, 0.0, 0.0, 0.0])],
            space="image",
        )


async def test_dimension_mismatch_is_rejected(chroma: ChromaVectorStore) -> None:
    """collection 是 4 维，写 8 维向量必须炸（且不能自动重建）。"""
    from app.core.errors import VectorDimensionMismatchError

    name = text_collection("kb_vec00000005")
    await chroma.ensure_collection(name, dim=4, space="text")
    with pytest.raises(VectorDimensionMismatchError):
        await chroma.upsert(
            name,
            [VectorRecord(id="ck_bad", vector=[1.0] * 8)],
            space="text",
        )
    # 而且**没有**把原有 collection 悄悄改掉
    info = await chroma.get_collection_info(name)
    assert info is not None and info.dim == 4


async def test_ensure_collection_reports_existing_dim_conflict(
    chroma: ChromaVectorStore,
) -> None:
    from app.core.errors import VectorDimensionMismatchError

    name = text_collection("kb_vec00000006")
    await chroma.ensure_collection(name, dim=4, space="text")
    with pytest.raises(VectorDimensionMismatchError):
        await chroma.ensure_collection(name, dim=8, space="text")


async def test_delete_collection_is_idempotent(chroma: ChromaVectorStore) -> None:
    name = text_collection("kb_vec00000007")
    await chroma.ensure_collection(name, dim=4, space="text")
    assert await chroma.delete_collection(name) is True
    assert await chroma.delete_collection(name) is False


async def test_list_collections_filters_by_suffix(chroma: ChromaVectorStore) -> None:
    await chroma.ensure_collection(text_collection("kb_lst00000001"), dim=4, space="text")
    await chroma.ensure_collection(image_collection("kb_lst00000001"), dim=8, space="image")
    # 一个不属于我们命名规则的 collection：它自己建的话得有 embedding function，
    # 这里直接让底层客户端建一个空 collection
    chroma._client.get_or_create_collection("someones_else_table", embedding_function=None)

    ours = await chroma.list_collections(only_mmkb=True)
    assert ours == ["kb_lst00000001__image", "kb_lst00000001__text"]
    assert "someones_else_table" in await chroma.list_collections(only_mmkb=False)


async def test_query_on_missing_collection_returns_empty(chroma: ChromaVectorStore) -> None:
    """查一个不存在的 collection 返回空，而不是抛错。

    检索层在知识库还没索引任何内容时就会走到这条路 ——
    抛错会让「刚建好、还没导入文件」的知识库一提问就 500。
    """
    assert await chroma.query(text_collection("kb_nope0000001"), [1.0, 0.0, 0.0, 0.0]) == []


# --------------------------------------------------------------------------------------
# 8. BM25
# --------------------------------------------------------------------------------------


def test_tokenize_auto_splits_chinese_and_english() -> None:
    tokens = tokenize("检索模块 uses BM25 for ranking", tokenizer="auto")
    # jieba 把中文切成词，正则把英文转小写 —— 两种都要有
    assert "检索" in tokens
    assert "bm25" in tokens
    assert "ranking" in tokens


def test_tokenize_keeps_technical_tokens_whole() -> None:
    """``gpt-4`` ``bge-m3`` 这类词不能被切碎，否则再也匹配不上。"""
    tokens = tokenize("用 bge-m3 与 gpt-4 对比", tokenizer="auto")
    assert "bge-m3" in tokens
    assert "gpt-4" in tokens


def test_tokenize_regex_mode_falls_back_to_single_chars() -> None:
    """``regex`` 模式下汉字逐字切 —— 这是 A/B 对比用的档位，行为要可预期。"""
    tokens = tokenize("检索模块", tokenizer="regex")
    assert tokens == ["检", "索", "模", "块"]


def test_tokenize_drops_punctuation_and_stopwords() -> None:
    tokens = tokenize(
        "检索，模块。的 the a", tokenizer="auto", stopwords=frozenset({"的"}), min_len=1
    )
    assert "，" not in tokens
    assert "。" not in tokens
    assert "的" not in tokens
    assert "the" in tokens


async def test_bm25_ranks_matching_document_first(tmp_path: Path) -> None:
    index = BM25Store(_settings(tmp_path))
    await index.ensure_kb("kb_bm000000001")
    await index.add_documents(
        "kb_bm000000001",
        [
            BM25Doc(id="ck_1", text="检索模块负责三路召回与 RRF 融合", payload={"doc_id": "d1"}),
            BM25Doc(id="ck_2", text="图片压缩使用 JPEG 质量参数", payload={"doc_id": "d1"}),
            BM25Doc(id="ck_3", text="重排模型只接受纯文本输入", payload={"doc_id": "d2"}),
        ],
    )

    hits = await index.search("kb_bm000000001", "三路召回融合", top_k=3)
    assert hits[0].id == "ck_1"
    assert hits[0].rank == 1
    assert hits[0].score > 0
    assert hits[0].payload["doc_id"] == "d1"


async def test_bm25_filters_out_zero_scores(tmp_path: Path) -> None:
    """没命中的文档必须**不出现**在结果里，而不是以 0 分垫底。

    它们一旦进入 RRF，就会以「第 300 名」的身份贡献一份融合分数，
    把真正相关的候选挤下去。
    """
    index = BM25Store(_settings(tmp_path))
    kb_id = "kb_bm000000002"
    await index.ensure_kb(kb_id)
    await index.add_documents(
        kb_id,
        [
            BM25Doc(id="ck_hit", text="向量检索与稀疏检索的融合"),
            BM25Doc(id="ck_miss", text="完全不相干的一段话，讲的是园艺和烘焙"),
        ],
    )
    hits = await index.search(kb_id, "向量检索融合", top_k=10)
    assert [h.id for h in hits] == ["ck_hit"]
    assert all(h.score > 0 for h in hits)


async def test_bm25_empty_query_and_empty_index_are_safe(tmp_path: Path) -> None:
    index = BM25Store(_settings(tmp_path))
    assert await index.search("kb_bm000000003", "任何") == []
    await index.ensure_kb("kb_bm000000003")
    assert await index.search("kb_bm000000003", "任何") == []
    assert await index.search("kb_bm000000003", "") == []


async def test_bm25_where_filter_does_not_change_scores(tmp_path: Path) -> None:
    """``where`` 只筛结果，不改变分数 —— 先过滤语料会改 idf，让分数不可比。"""
    index = BM25Store(_settings(tmp_path))
    kb_id = "kb_bm000000004"
    await index.ensure_kb(kb_id)
    await index.add_documents(
        kb_id,
        [
            BM25Doc(id="ck_a", text="向量检索", payload={"doc_id": "d1"}),
            BM25Doc(id="ck_b", text="向量检索的评估", payload={"doc_id": "d2"}),
        ],
    )
    unfiltered = {h.id: h.score for h in await index.search(kb_id, "向量检索", top_k=10)}
    filtered = await index.search(kb_id, "向量检索", top_k=10, where={"doc_id": "d2"})

    assert [h.id for h in filtered] == ["ck_b"]
    assert filtered[0].score == pytest.approx(unfiltered["ck_b"])


async def test_bm25_persist_and_reload_roundtrip(tmp_path: Path) -> None:
    """落盘 + 重新加载后，检索结果必须与内存里一致。"""
    settings = _settings(tmp_path)
    kb_id = "kb_bm000000005"
    docs = [
        BM25Doc(id="ck_1", text="多模态向量与文本向量不在同一个空间", payload={"doc_id": "d1"}),
        BM25Doc(id="ck_2", text="重排器读不懂图片，只能看 caption", payload={"doc_id": "d2"}),
    ]

    first = BM25Store(settings)
    await first.ensure_kb(kb_id)
    await first.add_documents(kb_id, docs)
    expected = [(h.id, h.score) for h in await first.search(kb_id, "文本向量空间", top_k=5)]
    path = await first.persist(kb_id)
    assert path is not None and path.is_file()

    # 全新的实例，只能从磁盘加载
    second = BM25Store(settings)
    assert await second.ensure_kb(kb_id) is True
    assert await second.count(kb_id) == 2
    reloaded = [(h.id, h.score) for h in await second.search(kb_id, "文本向量空间", top_k=5)]
    assert reloaded == pytest.approx(expected)


async def test_bm25_retokenizes_when_tokenizer_changes(tmp_path: Path) -> None:
    """换了分词器之后，存下来的 tokens 必须作废重切。

    索引文件里之所以还要存 ``text``，就是为了这一刻 ——
    只存 tokens 的话，改配置之后只能全量重建，而重建需要回数据库捞原文。
    """
    auto = BM25Store(_settings(tmp_path))
    kb_id = "kb_bm000000006"
    await auto.ensure_kb(kb_id)
    await auto.add_documents(kb_id, [BM25Doc(id="ck_1", text="检索模块")])
    await auto.persist(kb_id)

    regex_settings = _settings(tmp_path)
    regex_settings.bm25.tokenizer = "regex"
    regex = BM25Store(regex_settings)
    assert await regex.ensure_kb(kb_id) is True
    hits = await regex.search(kb_id, "检索模块", top_k=5)
    # 逐字切之后，"检索模块" 这个查询也是逐字的，仍然能命中
    assert hits and hits[0].id == "ck_1"


async def test_bm25_reloads_without_index_dir(tmp_path: Path) -> None:
    """没有索引文件时 ``ensure_kb`` 返回 False 而不是抛错 —— 调用方据此重建。"""
    index = BM25Store(_settings(tmp_path))
    assert await index.ensure_kb("kb_bm000000007") is False


async def test_bm25_persist_disabled_writes_nothing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.bm25.persist = False
    index = BM25Store(settings)
    kb_id = "kb_bm000000008"
    await index.ensure_kb(kb_id)
    await index.add_documents(kb_id, [BM25Doc(id="ck_1", text="任意内容")])
    assert await index.persist(kb_id) is None
    assert not list((tmp_path / "bm25").glob("*.json")) if (tmp_path / "bm25").exists() else True


async def test_bm25_remove_by_doc_drops_all_its_chunks(tmp_path: Path) -> None:
    index = BM25Store(_settings(tmp_path))
    kb_id = "kb_bm000000009"
    await index.ensure_kb(kb_id)
    await index.add_documents(
        kb_id,
        [
            BM25Doc(id="ck_a1", text="甲的第一段", payload={"doc_id": "d1"}),
            BM25Doc(id="ck_a2", text="甲的第二段", payload={"doc_id": "d1"}),
            BM25Doc(id="ck_b1", text="乙的第一段", payload={"doc_id": "d2"}),
        ],
    )
    assert await index.remove_documents_by_doc(kb_id, "d1") == 2
    assert await index.count(kb_id) == 1
    assert [h.id for h in await index.search(kb_id, "第一段", top_k=5)] == ["ck_b1"]


async def test_bm25_drop_kb_removes_memory_and_file(tmp_path: Path) -> None:
    index = BM25Store(_settings(tmp_path))
    kb_id = "kb_bm000000010"
    await index.ensure_kb(kb_id)
    await index.add_documents(kb_id, [BM25Doc(id="ck_1", text="随便什么内容")])
    path = await index.persist(kb_id)
    assert path is not None and path.is_file()

    assert await index.drop_kb(kb_id) is True
    assert not path.exists()
    assert await index.count(kb_id) == 0
    assert await index.drop_kb(kb_id) is False


async def test_bm25_documents_are_isolated_per_kb(tmp_path: Path) -> None:
    """两个知识库的索引必须互不可见 —— 否则 A 库的答案会引用 B 库的内容。"""
    index = BM25Store(_settings(tmp_path))
    await index.add_documents("kb_iso00000001", [BM25Doc(id="ck_a", text="甲的检索模块")])
    await index.add_documents("kb_iso00000002", [BM25Doc(id="ck_b", text="乙的检索模块")])

    assert [h.id for h in await index.search("kb_iso00000001", "检索模块")] == ["ck_a"]
    assert [h.id for h in await index.search("kb_iso00000002", "检索模块")] == ["ck_b"]


def test_bm25_doc_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="text 为空"):
        BM25Doc(id="ck_x", text="   ")


def test_bm25_doc_rejects_non_scalar_payload() -> None:
    with pytest.raises(TypeError, match="payload"):
        BM25Doc(id="ck_x", text="内容", payload={"tags": ["a", "b"]})  # type: ignore[dict-item]


# --------------------------------------------------------------------------------------
# 9. BlobStore
# --------------------------------------------------------------------------------------


@pytest.fixture
def blobs(tmp_path: Path) -> BlobStore:
    return BlobStore(_settings(tmp_path))


def test_blob_put_writes_file_and_returns_url(blobs: BlobStore) -> None:
    path, url = blobs.put("kb_blob0000001", "img_abc000001", b"\x89PNG-fake", mime_type="image/png")
    assert path.is_file()
    assert path.read_bytes() == b"\x89PNG-fake"
    assert url == "/static/images/kb_blob0000001/img_abc000001.png"
    # URL 与磁盘路径必须严格同构，否则前端加载不出来
    assert url.endswith(f"/{path.parent.name}/{path.name}")


def test_blob_put_dedupes_extension(blobs: BlobStore) -> None:
    """``image_id`` 自带扩展名时不能拼成 ``x.png.png``。"""
    path, url = blobs.put("kb_blob0000002", "img_abc000002.png", b"data")
    assert path.name == "img_abc000002.png"
    assert url.endswith("img_abc000002.png")


def test_blob_put_rejects_empty_bytes(blobs: BlobStore) -> None:
    with pytest.raises(StorageError, match="为空"):
        blobs.put("kb_blob0000003", "img_abc000003", b"")


def test_blob_resolves_unknown_extension(blobs: BlobStore) -> None:
    """上游说的 mime 与实际扩展名不符时，仍要能找到文件。"""
    blobs.put("kb_blob0000004", "img_abc000004", b"jpg-bytes", mime_type="image/jpeg")
    assert blobs.exists("kb_blob0000004", "img_abc000004", "jpg")
    assert blobs.read("kb_blob0000004", "img_abc000004") == b"jpg-bytes"


@pytest.mark.parametrize(
    "bad",
    [
        "../escape",
        "..",
        "a/b",
        "a\\b",
        ".hidden",
        "",
        "x" * 200,
    ],
)
def test_blob_rejects_path_traversal(blobs: BlobStore, bad: str) -> None:
    """``kb_id`` 直接来自请求，必须是这一个地方拦住 ``../``。

    这是「只要有一个调用点忘了校验就全线失守」的那类漏洞，
    所以放在最底层的路径构造函数里，而不是靠每个路由自己记得检查。
    """
    with pytest.raises(StorageError):
        blobs.path_for(bad, "img_x")
    with pytest.raises(StorageError):
        blobs.url_for(bad, "img_x")
    with pytest.raises(StorageError):
        blobs.put(bad, "img_x", b"data")


def test_blob_rejects_bad_image_id(blobs: BlobStore) -> None:
    with pytest.raises(StorageError):
        blobs.path_for("kb_ok00000001", "../../etc/passwd")


def test_blob_delete_kb_removes_everything(blobs: BlobStore) -> None:
    kb_id = "kb_blob0000005"
    for i in range(3):
        blobs.put(kb_id, f"img_del{i:06d}", b"x")
    assert blobs.stats()["file_count"] == 3

    assert blobs.delete_kb(kb_id) == 3
    assert blobs.stats()["file_count"] == 0
    assert blobs.delete_kb(kb_id) == 0  # 幂等


def test_blob_overwrite_is_atomic_and_leaves_no_tmp(blobs: BlobStore) -> None:
    kb_id = "kb_blob0000006"
    blobs.put(kb_id, "img_atomic0001", b"v1")
    blobs.put(kb_id, "img_atomic0001", b"v2")

    assert blobs.read(kb_id, "img_atomic0001") == b"v2"
    assert list(blobs.root.rglob("*.tmp")) == []


def test_blob_cleanup_tmp_removes_leftovers(blobs: BlobStore) -> None:
    """模拟上次崩溃留下的半截文件。"""
    kb_id = "kb_blob0000007"
    directory = blobs.ensure_root() / kb_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "img_half000001.png.tmp").write_bytes(b"partial")

    assert blobs.cleanup_tmp() == 1
    assert not (directory / "img_half000001.png.tmp").exists()


def test_blob_read_missing_raises_storage_error(blobs: BlobStore) -> None:
    with pytest.raises(StorageError, match="不存在"):
        blobs.read("kb_blob0000008", "img_missing0001")


# --------------------------------------------------------------------------------------
# 10. pgvector 的 where 翻译（离线，不连数据库）
# --------------------------------------------------------------------------------------


def test_pg_where_translation_orders_params_with_placeholders() -> None:
    """占位符顺序必须与参数顺序严格一致。

    pgvector 后端里，参数列表 = [SELECT 的向量, ...where 参数, ORDER BY 的向量, LIMIT]。
    where 翻译一旦少给或多给参数，整条 SQL 会错位 ——
    症状是「过滤条件莫名其妙地不起作用」，而不是报错。
    """
    pytest.importorskip("psycopg", reason="未安装 [pg] extra")

    from app.store.vectorstore_pg import _where_to_sql

    clause, params = _where_to_sql(
        {"$and": [{"doc_id": {"$in": ["d1", "d2"]}}, {"block_type": "table"}]}
    )
    rendered = clause.as_string(None)
    assert params == ["d1", "d2", "table"]
    # 三个 %s 按顺序出现
    assert rendered.count("%s") == 3
    assert rendered.count("metadata->>") == 2
    assert "IN" in rendered


def test_pg_where_translation_casts_numeric_for_range_ops() -> None:
    """数字比较必须套 ``::numeric``。

    jsonb 里取出来的是 text，``'10' < '9'`` 为真而 ``10 < 9`` 为假 ——
    不套转换的话「按页码/时间范围过滤」会静默地给出错误结果。
    """
    pytest.importorskip("psycopg", reason="未安装 [pg] extra")
    from app.store.vectorstore_pg import _where_to_sql

    _, params = _where_to_sql({"page": {"$gte": 10}})
    clause, _ = _where_to_sql({"page": {"$gte": 10}})
    assert params == [10]
    assert "::numeric" in clause.as_string(None)

    # 字符串不该被转成数字
    text_clause, _ = _where_to_sql({"doc_id": {"$eq": "d1"}})
    assert "::numeric" not in text_clause.as_string(None)


def test_pg_where_rejects_unsupported_operator() -> None:
    pytest.importorskip("psycopg", reason="未安装 [pg] extra")
    from app.store.vectorstore_pg import _where_to_sql

    with pytest.raises(StorageError, match="不支持"):
        _where_to_sql({"page": {"$regex": "^1"}})


def test_pg_where_empty_in_list_is_not_a_syntax_error() -> None:
    """``IN ()`` 在 SQL 里是语法错误。空集合必须特殊处理成常量真假值。"""
    pytest.importorskip("psycopg", reason="未安装 [pg] extra")
    from app.store.vectorstore_pg import _where_to_sql

    clause, params = _where_to_sql({"doc_id": {"$in": []}})
    assert params == []
    assert clause.as_string(None).strip() == "FALSE"

    clause, params = _where_to_sql({"doc_id": {"$nin": []}})
    assert params == []
    assert clause.as_string(None).strip() == "TRUE"


def test_pg_vector_literal_format() -> None:
    pytest.importorskip("psycopg", reason="未安装 [pg] extra")
    from app.store.vectorstore_pg import _parse_vector_literal, _vector_literal

    literal = _vector_literal([1.0, 0.5, -2.0])
    assert literal == "[1.0,0.5,-2.0]"
    assert _parse_vector_literal(literal) == [1.0, 0.5, -2.0]
    assert _parse_vector_literal("[]") == []


# --------------------------------------------------------------------------------------
# 11. store.factory 的惰性导入
# --------------------------------------------------------------------------------------


def test_factory_builds_chroma_by_default(tmp_path: Path) -> None:
    from app.store.factory import build_vectorstore

    store = build_vectorstore(_settings(tmp_path))
    assert store.backend == "chroma"


def test_factory_reports_missing_pg_extra_clearly(tmp_path: Path) -> None:
    """``VECTORSTORE_BACKEND=pg`` 但没装驱动时，报错必须说人话。"""
    from app.config import ConfigurationError
    from app.store.factory import build_vectorstore

    settings = _settings(tmp_path, backend="pg", pg_dsn="postgresql://u:p@localhost/db")
    try:
        store = build_vectorstore(settings)
    except ConfigurationError as exc:
        # 报错文本要能让用户直接照抄命令，而不是自己去猜装哪个包
        assert "pip install -e" in str(exc)
        assert ".[pg]" in str(exc)
    except ImportError as exc:  # pragma: no cover - 装了 extra 才会走到
        pytest.skip(f"已安装 [pg] extra，走的是真实构造路径：{exc}")
    else:
        assert store.backend == "pg"


def test_get_vectorstore_is_a_zero_arg_cached_singleton() -> None:
    """``get_*`` 不能接收 settings 参数。

    ``lru_cache`` 要求参数可哈希，而 pydantic 的 ``BaseSettings`` 默认不可哈希 ——
    写成 ``get_vectorstore(settings)`` 会在**第一次调用**就抛
    ``TypeError: unhashable type: 'Settings'``，而那时已经在生产环境启动路径上了。
    这条测试把「接口形状」本身钉住。
    """
    import inspect

    from app.store import factory

    for name in ("get_vectorstore", "get_blobstore", "get_bm25store"):
        signature = inspect.signature(getattr(factory, name))
        assert not signature.parameters, f"{name} 应当是零参数"
    # 想换配置必须走 build_*
    assert "settings" in inspect.signature(factory.build_vectorstore).parameters


async def test_reset_store_caches_is_safe_without_config() -> None:
    """没配置好也要能安全关停 —— 不然「启动失败」会连带「关停也异常」。"""
    from app.store import factory

    factory.reset_vectorstore_cache()
    await factory.reset_store_caches()


# --------------------------------------------------------------------------------------
# 12. 并发写入
# --------------------------------------------------------------------------------------


async def test_concurrent_chunk_writes_do_not_deadlock(store: DocStore, kb: Any) -> None:
    """WAL + busy_timeout 是否配好，只能靠并发写来验证。

    没配的话 SQLite 会立刻抛 ``database is locked``，
    而不是等一会儿重试 —— 导入大文件时几乎必然踩到。
    """
    doc, _ = await store.upsert_document(
        Document(
            doc_id="doc_conc000001",
            kb_id=kb.kb_id,
            source_path="/tmp/c.md",
            source_name="c.md",
            mime_type="text/markdown",
            doc_hash="conc",
        )
    )

    async def write_batch(start: int) -> None:
        await store.add_chunks([_chunk(kb.kb_id, doc.doc_id, start + i) for i in range(5)])

    await asyncio.gather(*(write_batch(i * 5) for i in range(4)))
    assert len(await store.get_document_chunks(doc.doc_id)) == 20


async def test_stats_reports_all_tables(store: DocStore, kb: Any) -> None:
    stats = await store.stats()
    assert stats["knowledge_bases"] == 1
    assert set(stats) >= {
        "knowledge_bases",
        "documents",
        "chunks",
        "images",
        "tasks",
        "sessions",
        "messages",
        "url",
    }
    assert "***" in stats["url"] or "password" not in stats["url"]


def test_docstatus_values_are_strings() -> None:
    """枚举列存的是字符串而不是序号 —— 直接读数据库时能看懂。"""
    assert str(DocStatus.SKIPPED_DUPLICATE) == "skipped_duplicate"
    assert str(BlockType.IMAGE_CAPTION) == "image_caption"


def test_utcnow_is_timezone_aware() -> None:
    assert utcnow().tzinfo is UTC
