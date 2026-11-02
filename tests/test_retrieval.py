"""检索层测试。

全部离线：使用真实的 Chroma / SQLite / BM25（都指到 ``tmp_path``），
只把三个外部模型换成确定性假实现。这样验证的是检索链路本身 ——
融合公式、去重身份、降级路径、快照校验 —— 而不是任何厂商接口。

覆盖的问题，共同点是「错了不会报错，只会让结果悄悄变差」：

1. **同一张图的两种身份** —— caption 分块与图像向量必须合并成一条，
   否则同一张图会在结果里出现两次并互相挤占名次；
2. **RRF 的数值** —— 按公式手算出期望分数，而不是只看「排第一的是它」；
3. **快照不一致** —— 换过 embedding 模型/维度之后继续检索必须报错，
   而不是返回一批看似正常的垃圾排序；
4. **降级路径** —— 某一路失败、重排失败、改名失败，都只能影响自己；
5. **空库** —— ``--no-embed`` 导入的库要能正常用 BM25 检索，
   而不是因为另外两路没数据就整体失败。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import (
    AgentSettings,
    BlobSettings,
    BM25Settings,
    DocStoreSettings,
    ImageEmbeddingSettings,
    PathSettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    TextEmbeddingSettings,
    VectorStoreSettings,
)
from app.core.errors import RetrievalError
from app.core.schemas import (
    BlockType,
    Chunk,
    DegradedReason,
    Document,
    ImageAsset,
    ImageStatus,
    ImageType,
    ItemType,
    RetrievalFilters,
    RetrievedItem,
    Route,
)
from app.providers.base import (
    ChatDelta,
    ChatResult,
    EmbeddingProvider,
    EmbeddingResult,
    LLMProvider,
    RerankHit,
    RerankProvider,
    RerankResult,
)
from app.providers.base import (
    ChatMessage as ProviderChatMessage,
)
from app.retrieval.base import (
    canonical_identity,
    guard_kb_snapshot,
    route_weight_map,
)
from app.retrieval.fusion import RankedList, reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.rerank import Reranker
from app.store.bm25store import BM25Doc, BM25Store
from app.store.db import Database
from app.store.docstore import DocStore
from app.store.vectorstore import VectorRecord, image_collection, text_collection
from app.store.vectorstore_chroma import ChromaVectorStore

TEXT_MODEL = "fake-text"
IMAGE_MODEL = "fake-image"
KB_ID = "kb_000000000001"
DOC_ID = "doc_000000000001"
IMG_ID = "img_000000000001"


# --------------------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------------------


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "paths": PathSettings(data_dir=tmp_path),
        "docstore": DocStoreSettings(sqlite_path=tmp_path / "test.db"),
        "blob": BlobSettings(images_dir=tmp_path / "images"),
        "bm25": BM25Settings(index_dir=tmp_path / "bm25", persist=False),
        "vectorstore": VectorStoreSettings(chroma_dir=tmp_path / "chroma"),
        "text_embed": TextEmbeddingSettings(model=TEXT_MODEL, dim=8),
        "image_embed": ImageEmbeddingSettings(model=IMAGE_MODEL, dim=16),
        "retrieval": RetrievalSettings(
            enable_rerank=False,
            top_k_per_route=10,
            final_top_k=10,
            image_query_top_k=10,
            rrf_k=60,
        ),
        "rerank": RerankSettings(enabled=False),
        "agent": AgentSettings(query_rewrite_enabled=True, query_rewrite_n=3),
    }
    values.update(overrides)
    return Settings(**values)


def _vec(text: str, dim: int) -> list[float]:
    """确定性向量。同一段文字永远得到同一个向量，不同文字几乎不会撞。"""
    values = [0.0] * dim
    for index, char in enumerate(text):
        values[index % dim] += (ord(char) % 11) + 1.0
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


class FakeEmbedding(EmbeddingProvider):
    """只服务一个空间的假 provider，能力边界与真实 provider 一致。"""

    name = "fake"

    def __init__(self, *, space: str, dim: int, model: str) -> None:
        self.space = space
        self.dim = dim
        self.model = model
        #: 记录每次调用的文本，便于断言 query 走的是哪条编码路径
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def supports_text(self) -> bool:
        return self.space == "text"

    @property
    def supports_image(self) -> bool:
        return self.space == "image"

    @property
    def supports_fused(self) -> bool:
        return self.space == "image"

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        if not self.supports_text:
            raise self._unsupported("embed_texts", "多模态 provider 不编码纯文本")
        self.calls.append(("texts", tuple(texts)))
        return EmbeddingResult(
            vectors=[_vec(text, self.dim) for text in texts],
            model=self.model,
            dim=self.dim,
        )

    async def embed_images(self, images: Sequence[Any]) -> EmbeddingResult:
        if not self.supports_image:
            raise self._unsupported("embed_images", "文本 provider 不编码图片")
        vector = [1.0] + [0.0] * (self.dim - 1)
        return EmbeddingResult(
            vectors=[list(vector) for _ in images], model=self.model, dim=self.dim
        )

    async def embed_fused(self, texts: Sequence[str]) -> EmbeddingResult:
        if not self.supports_fused:
            raise self._unsupported("embed_fused", "纯文本 provider 不进入融合空间")
        self.calls.append(("fused", tuple(texts)))
        vector = [1.0] + [0.0] * (self.dim - 1)
        return EmbeddingResult(
            vectors=[list(vector) for _ in texts], model=self.model, dim=self.dim
        )


class FakeRerank(RerankProvider):
    """可编程的重排器：可以按规则打分，也可以直接抛异常。"""

    name = "fake-rerank"
    model = "fake-rerank-model"

    def __init__(self, *, boom: bool = False, image_first: bool = True) -> None:
        self.boom = boom
        self.image_first = image_first
        self.documents: list[str] = []

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> RerankResult:
        if self.boom:
            raise RuntimeError("reranker 服务不可用")
        self.documents = list(documents)
        scored: list[RerankHit] = []
        for index, document in enumerate(documents):
            if self.image_first and document.startswith("[图片]"):
                score = 0.99
            else:
                # 越靠后的文档分数越低，保持输入顺序
                score = 0.5 - index * 0.01
            scored.append(RerankHit(index=index, score=score))
        scored.sort(key=lambda hit: (-hit.score, hit.index))
        limit = top_n or len(scored)
        return RerankResult(hits=scored[:limit], model=self.model)


class FakeLLM(LLMProvider):
    name = "fake-llm"
    model = "fake-llm-model"

    def __init__(self, *, reply: str = "", boom: bool = False) -> None:
        self.reply = reply
        self.boom = boom
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[ProviderChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls += 1
        if self.boom:
            raise RuntimeError("LLM 不可用")
        return ChatResult(text=self.reply, model=self.model)

    async def chat_stream(
        self,
        messages: Sequence[ProviderChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatDelta]:
        yield ChatDelta(text=self.reply, done=True)


@asynccontextmanager
async def _env(
    tmp_path: Path,
    *,
    text: bool = True,
    image: bool = True,
    rerank: RerankProvider | None = None,
    llm: LLMProvider | None = None,
    retrieval: RetrievalSettings | None = None,
):
    """一套隔离的存储 + 检索器。所有对象都指向 tmp_path。"""
    settings = _settings(
        tmp_path, **({"retrieval": retrieval} if retrieval is not None else {})
    )
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}", is_sqlite=True)
    await database.init_schema()

    docstore = DocStore(settings, db=database)
    vectorstore = ChromaVectorStore(settings)
    bm25store = BM25Store(settings)

    kb = await docstore.create_kb(
        "retrieval-demo",
        kb_id=KB_ID,
        text_collection=text_collection(KB_ID),
        image_collection=image_collection(KB_ID),
        text_dim=8,
        image_dim=16,
        text_embed_model=TEXT_MODEL,
        image_embed_model=IMAGE_MODEL,
    )
    await vectorstore.ensure_collection(
        kb.text_collection, dim=8, space="text", embed_model=TEXT_MODEL
    )
    await vectorstore.ensure_collection(
        kb.image_collection, dim=16, space="image", embed_model=IMAGE_MODEL
    )

    text_embedding = (
        FakeEmbedding(space="text", dim=8, model=TEXT_MODEL) if text else None
    )
    image_embedding = (
        FakeEmbedding(space="image", dim=16, model=IMAGE_MODEL) if image else None
    )
    retriever = HybridRetriever(
        settings,
        docstore=docstore,
        vectorstore=vectorstore,
        bm25store=bm25store,
        text_embedding=text_embedding,
        image_embedding=image_embedding,
        rerank=rerank,
        llm=llm,
    )

    env = SimpleNamespace(
        settings=settings,
        database=database,
        docstore=docstore,
        vectorstore=vectorstore,
        bm25store=bm25store,
        kb=kb,
        text_embedding=text_embedding,
        image_embedding=image_embedding,
        retriever=retriever,
    )
    try:
        yield env
    finally:
        await vectorstore.aclose()
        await database.dispose()


async def _add_doc(env: SimpleNamespace) -> None:
    await env.docstore.upsert_document(
        Document(
            doc_id=DOC_ID,
            kb_id=KB_ID,
            source_path="demo.md",
            source_name="demo.md",
            mime_type="text/markdown",
            doc_hash="hash-demo-1",
        )
    )


def _chunk(chunk_id: str, ordinal: int, text: str, **kwargs: Any) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        ordinal=ordinal,
        text=text,
        token_count=max(1, len(text)),
        **kwargs,
    )


async def _index_chunk(env: SimpleNamespace, chunk: Chunk) -> None:
    """把一个分块同时写进文本向量库与 BM25。"""
    await env.vectorstore.upsert(
        env.kb.text_collection,
        [
            VectorRecord(
                id=chunk.chunk_id,
                vector=_vec(chunk.text, 8),
                document=chunk.text,
                metadata=chunk.to_metadata(),
            )
        ],
        space="text",
    )
    await env.bm25store.add_documents(
        KB_ID, [BM25Doc(id=chunk.chunk_id, text=chunk.text, payload=chunk.to_metadata())]
    )


def _image_asset(caption_text: str, *, url: str = "/images/demo.png") -> ImageAsset:
    return ImageAsset(
        image_id=IMG_ID,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        url=url,
        status=ImageStatus.DONE,
        caption_text=caption_text,
        heading_path=["架构", "检索层"],
    )


async def _index_image(env: SimpleNamespace, asset: ImageAsset) -> None:
    await env.docstore.upsert_image(asset)
    await env.vectorstore.upsert(
        env.kb.image_collection,
        [
            VectorRecord(
                id=asset.image_id,
                vector=[1.0] + [0.0] * 15,
                document=asset.caption_text,
                metadata=asset.to_metadata(),
            )
        ],
        space="image",
    )


def _item(item_id: str, *, text: str = "") -> RetrievedItem:
    return RetrievedItem(
        item_id=item_id,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        text=text,
    )


# --------------------------------------------------------------------------------------
# 1. 身份归一
# --------------------------------------------------------------------------------------


class TestCanonicalIdentity:
    def test_caption_chunk_becomes_image(self) -> None:
        metadata = {
            "block_type": str(BlockType.IMAGE_CAPTION),
            "image_ids": json.dumps([IMG_ID]),
        }
        assert canonical_identity(metadata, "ck_000000000001") == (IMG_ID, IMG_ID)

    def test_plain_chunk_keeps_chunk_id(self) -> None:
        metadata = {"block_type": str(BlockType.TEXT), "image_ids": "[]"}
        assert canonical_identity(metadata, "ck_000000000002") == (
            "ck_000000000002",
            None,
        )

    def test_broken_image_ids_does_not_crash(self) -> None:
        metadata = {"block_type": str(BlockType.IMAGE_CAPTION), "image_ids": "not-json"}
        # 解析失败时退化成普通分块，而不是让整次检索挂掉
        item_id, image_id = canonical_identity(metadata, "ck_000000000003")
        assert item_id == "ck_000000000003"
        assert image_id is None


# --------------------------------------------------------------------------------------
# 2. RRF
# --------------------------------------------------------------------------------------


class TestRRF:
    def test_formula_matches_hand_computation(self) -> None:
        a, b, c = _item("a"), _item("b"), _item("c")
        lists = [
            RankedList("bm25", Route.BM25, [a, b], 1.0),
            RankedList("dense_text", Route.DENSE_TEXT, [b, c], 1.0),
        ]
        fused = reciprocal_rank_fusion(lists, k=60)

        # b 在两路分别是第 2、第 1；a、c 各只在一路第 1 / 第 2
        assert fused[0].item_id == "b"
        assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
        assert fused[0].route_ranks == {"bm25": 2, "dense_text": 1}
        assert fused[0].route is Route.HYBRID

        by_id = {item.item_id: item for item in fused}
        assert by_id["a"].rrf_score == pytest.approx(1 / 61)
        assert by_id["c"].rrf_score == pytest.approx(1 / 62)
        assert by_id["a"].route is Route.BM25
        assert by_id["c"].route is Route.DENSE_TEXT
        assert [item.item_id for item in fused] == ["b", "a", "c"]

    def test_weights_scale_contributions(self) -> None:
        a = _item("a")
        one = reciprocal_rank_fusion([RankedList("x", Route.BM25, [a], 1.0)], k=60)
        two = reciprocal_rank_fusion([RankedList("x", Route.BM25, [a], 2.0)], k=60)
        assert two[0].rrf_score == pytest.approx(one[0].rrf_score * 2)

    def test_zero_weight_list_is_ignored(self) -> None:
        fused = reciprocal_rank_fusion(
            [RankedList("x", Route.BM25, [_item("a")], 0.0)], k=60
        )
        assert fused == []

    def test_tie_break_is_deterministic(self) -> None:
        # 两条同分条目必须按 item_id 排序，否则同一查询两次运行可能不���
        a, b = _item("ck_a"), _item("ck_b")
        fused = reciprocal_rank_fusion(
            [
                RankedList("l1", Route.BM25, [a, b], 1.0),
                RankedList("l2", Route.DENSE_TEXT, [b, a], 1.0),
            ],
            k=60,
        )
        assert [item.item_id for item in fused] == ["ck_a", "ck_b"]

    def test_duplicate_in_one_list_counted_once(self) -> None:
        a = _item("a")
        fused = reciprocal_rank_fusion(
            [RankedList("l1", Route.BM25, [a, a], 1.0)], k=60
        )
        assert len(fused) == 1
        assert fused[0].rrf_score == pytest.approx(1 / 61)
        assert fused[0].route_ranks == {"l1": 1}

    def test_top_k_truncates(self) -> None:
        items = [_item(f"x{i}") for i in range(5)]
        fused = reciprocal_rank_fusion(
            [RankedList("l", Route.BM25, items, 1.0)], k=60, top_k=2
        )
        assert [item.item_id for item in fused] == ["x0", "x1"]


# --------------------------------------------------------------------------------------
# 3. 快照校验
# --------------------------------------------------------------------------------------


class TestSnapshotGuard:
    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises(self, tmp_path: Path) -> None:
        async with _env(tmp_path) as env:
            provider = FakeEmbedding(space="text", dim=99, model=TEXT_MODEL)
            with pytest.raises(RetrievalError) as info:
                guard_kb_snapshot(env.kb, provider, space="text")
            assert "维度" in info.value.message

    @pytest.mark.asyncio
    async def test_model_mismatch_raises(self, tmp_path: Path) -> None:
        async with _env(tmp_path) as env:
            provider = FakeEmbedding(space="text", dim=8, model="other-model")
            with pytest.raises(RetrievalError) as info:
                guard_kb_snapshot(env.kb, provider, space="text")
            assert "模型" in info.value.message

    @pytest.mark.asyncio
    async def test_space_mismatch_raises(self, tmp_path: Path) -> None:
        async with _env(tmp_path) as env:
            # 维度恰好都是 16，光看维度发现不了 —— 这正是铁律 3 要拦的情况
            provider = FakeEmbedding(space="text", dim=16, model=IMAGE_MODEL)
            with pytest.raises(RetrievalError) as info:
                guard_kb_snapshot(env.kb, provider, space="image")
            assert "空间" in info.value.message

    @pytest.mark.asyncio
    async def test_route_weight_map_rejects_wrong_length(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, retrieval=RetrievalSettings(route_weights="1,1"))
        with pytest.raises(RetrievalError):
            route_weight_map(settings)


# --------------------------------------------------------------------------------------
# 4. 端到端融合
# --------------------------------------------------------------------------------------


CAPTION_TEXT = "架构 > 检索层 | [架构图] 三路召回与 RRF 融合的流程"


async def _seed_mixed_kb(env: SimpleNamespace) -> None:
    """一个正文分块 + 一个 caption 分块 + 一条图像向量。"""
    await _add_doc(env)
    plain = _chunk("ck_000000000001", 0, "RRF 融合原理：按名次加权求和。")
    caption = _chunk(
        "ck_000000000002",
        1,
        CAPTION_TEXT,
        block_type=BlockType.IMAGE_CAPTION,
        image_ids=[IMG_ID],
        heading_path=["架构", "检索层"],
    )
    tail = _chunk("ck_000000000003", 2, "邻居分块：重排只接受文本输入。")
    await env.docstore.add_chunks([plain, caption, tail])
    for chunk in (plain, caption, tail):
        await _index_chunk(env, chunk)

    asset = _image_asset(CAPTION_TEXT)
    await _index_image(env, asset)


class TestHybrid:
    @pytest.mark.asyncio
    async def test_caption_and_image_merge_into_one_item(self, tmp_path: Path) -> None:
        async with _env(tmp_path) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search(
                env.kb, "三路召回", use_rerank=False
            )

            images = [item for item in outcome.items if item.image_id == IMG_ID]
            assert len(images) == 1, "同一张图被两路召回时必须合并成一条"
            image = images[0]
            assert image.item_type is ItemType.IMAGE
            assert image.chunk_id is None
            assert "dense_text" in image.route_scores
            assert "dense_image" in image.route_scores
            assert image.route is Route.HYBRID
            # 文本空间命中的 caption 分块没有 URL，回填阶段从 SQLite 补上
            assert image.image_url == "/images/demo.png"

    @pytest.mark.asyncio
    async def test_source_name_is_hydrated(self, tmp_path: Path) -> None:
        async with _env(tmp_path) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search(env.kb, "RRF", use_rerank=False)
            assert outcome.items
            assert all(item.source_name == "demo.md" for item in outcome.items)

    @pytest.mark.asyncio
    async def test_dense_routes_skip_on_empty_collections(self, tmp_path: Path) -> None:
        """``--no-embed`` 的库：文本/图像 collection 都是空的，只跑 BM25。"""
        async with _env(tmp_path) as env:
            await _add_doc(env)
            chunk = _chunk("ck_000000000010", 0, "只有 BM25 索引的离线知识库。")
            await env.docstore.add_chunks([chunk])
            await env.bm25store.add_documents(
                KB_ID,
                [BM25Doc(id=chunk.chunk_id, text=chunk.text, payload=chunk.to_metadata())],
            )

            outcome = await env.retriever.search(env.kb, "离线知识库", use_rerank=False)
            assert [item.item_id for item in outcome.items] == [chunk.chunk_id]

            skipped = {
                str(result.route): result
                for result in outcome.route_results
                if result.skipped
            }
            assert "dense_text" in skipped
            assert "dense_image" in skipped
            assert "为空" in skipped["dense_text"].note
            # 「库是空的」是正常状态，不是降级
            assert outcome.degraded_reasons == []

    @pytest.mark.asyncio
    async def test_missing_image_provider_degrades(self, tmp_path: Path) -> None:
        async with _env(tmp_path, image=False) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search(env.kb, "RRF", use_rerank=False)
            assert DegradedReason.IMAGE_EMBED_UNAVAILABLE in outcome.degraded_reasons
            assert outcome.items  # 其余两路照常返回

    @pytest.mark.asyncio
    async def test_route_filter_limits_execution(self, tmp_path: Path) -> None:
        async with _env(tmp_path) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search(
                env.kb, "RRF", routes=[Route.BM25], use_rerank=False
            )
            executed = {
                str(result.route)
                for result in outcome.route_results
                if not result.skipped
            }
            assert executed == {"bm25"}
            assert all(
                set(item.route_ranks) <= {"bm25"} for item in outcome.items
            )

    @pytest.mark.asyncio
    async def test_unknown_route_raises(self, tmp_path: Path) -> None:
        async with _env(tmp_path) as env:
            with pytest.raises(RetrievalError):
                await env.retriever.search(env.kb, "x", routes=["nope"])

    @pytest.mark.asyncio
    async def test_multi_query_labels_and_weights(self, tmp_path: Path) -> None:
        async with _env(tmp_path) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search_multi(
                env.kb,
                ["三路召回", "RRF 融合"],
                original_query="三路召回",
                routes=[Route.BM25],
                use_rerank=False,
            )
            labels = {label for item in outcome.items for label in item.route_ranks}
            assert "bm25#1" in labels
            assert "bm25#2" in labels

            # 同一张图的 caption 在两个查询里都被 BM25 命中：
            # 原问题权重 1.0，改写权重 0.7，分数必须正好是这两项之和
            image = next(item for item in outcome.items if item.image_id == IMG_ID)
            rank_original = image.route_ranks["bm25#1"]
            rank_rewrite = image.route_ranks["bm25#2"]
            assert image.rrf_score == pytest.approx(
                1.0 / (60 + rank_original) + 0.7 / (60 + rank_rewrite)
            )


# --------------------------------------------------------------------------------------
# 5. 重排
# --------------------------------------------------------------------------------------


class TestRerank:
    @pytest.mark.asyncio
    async def test_image_document_uses_caption_template(self) -> None:
        settings = _settings(Path("."))
        reranker = Reranker(FakeRerank(), settings)
        item = RetrievedItem(
            item_id=IMG_ID,
            item_type=ItemType.IMAGE,
            kb_id=KB_ID,
            doc_id=DOC_ID,
            image_id=IMG_ID,
            text=CAPTION_TEXT,
            heading_path=["架构", "检索层"],
        )
        document = reranker.build_document(item)
        # caption 文本本身以标题路径开头，拼模板前要剥掉，避免标题出现两遍
        assert document == "[图片] 架构 > 检索层 | [架构图] 三路召回与 RRF 融合的流程"

    @pytest.mark.asyncio
    async def test_rerank_puts_image_first(self, tmp_path: Path) -> None:
        reranker = FakeRerank(image_first=True)
        async with _env(tmp_path, rerank=reranker) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search(env.kb, "三路召回", use_rerank=True)
            assert outcome.reranked is True
            assert outcome.items[0].image_id == IMG_ID
            assert outcome.items[0].rerank_score is not None
            # 图片候选送给 reranker 的是拼好的文本，不是图片本身
            assert any(doc.startswith("[图片]") for doc in reranker.documents)

    @pytest.mark.asyncio
    async def test_rerank_failure_falls_back_to_rrf(self, tmp_path: Path) -> None:
        async with _env(tmp_path, rerank=FakeRerank(boom=True)) as env:
            await _seed_mixed_kb(env)
            fallback = await env.retriever.search(
                env.kb, "三路召回", use_rerank=True
            )
            assert fallback.reranked is False
            assert DegradedReason.RERANK_UNAVAILABLE in fallback.degraded_reasons
            # 排序与纯 RRF 完全一致 —— 重排失败不该改变结果集
            plain = await env.retriever.search(env.kb, "三路召回", use_rerank=False)
            assert [item.item_id for item in fallback.items] == [
                item.item_id for item in plain.items
            ]

    @pytest.mark.asyncio
    async def test_rerank_disabled_by_config_adds_no_degradation(
        self, tmp_path: Path
    ) -> None:
        async with _env(tmp_path, rerank=FakeRerank()) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search(
                env.kb, "三路召回", use_rerank=False
            )
            assert outcome.reranked is False
            assert DegradedReason.RERANK_UNAVAILABLE not in outcome.degraded_reasons


# --------------------------------------------------------------------------------------
# 6. 邻居扩展与过滤
# --------------------------------------------------------------------------------------


class TestPostProcessing:
    @pytest.mark.asyncio
    async def test_neighbor_expansion_attaches_text(self, tmp_path: Path) -> None:
        retrieval = RetrievalSettings(
            enable_rerank=False,
            top_k_per_route=10,
            final_top_k=10,
            neighbor_expand=1,
        )
        async with _env(tmp_path, retrieval=retrieval) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search(
                env.kb, "邻居分块", routes=[Route.BM25], use_rerank=False
            )
            target = next(
                item for item in outcome.items if "邻居分块" in item.text
            )
            expanded = target.metadata.get("expanded_text", "")
            assert "RRF 融合原理" in expanded or CAPTION_TEXT in expanded
            assert target.metadata.get("expanded_chunk_ids")

    @pytest.mark.asyncio
    async def test_block_type_filter_applies_to_text_space_only(
        self, tmp_path: Path
    ) -> None:
        """``block_types`` 是文本分块的过滤器，不会连带把图片结果筛掉。

        图片该不该出现由 ``image_types`` 决定 —— 两个维度分开，
        用户想「只看表格」时仍然能同时看到相关的图。
        """
        async with _env(tmp_path) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search(
                env.kb,
                "三路召回",
                filters=RetrievalFilters(block_types=[BlockType.TABLE]),
                use_rerank=False,
            )
            text_routes = [
                result
                for result in outcome.route_results
                if result.route in (Route.BM25, Route.DENSE_TEXT) and not result.skipped
            ]
            assert text_routes
            assert all(result.count == 0 for result in text_routes)
            # 图像空间没有 block_type 字段，这一路不受影响
            assert [item.image_id for item in outcome.items] == [IMG_ID]

    @pytest.mark.asyncio
    async def test_image_type_filter_does_not_break_text_route(
        self, tmp_path: Path
    ) -> None:
        """``image_types`` 只对图像 collection 生效。

        如果把它原样带进文本 collection，Chroma 不报错、只返回空结果 ——
        一个「过滤条件写错了」伪装成「没搜到」。
        """
        async with _env(tmp_path) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search(
                env.kb,
                "RRF",
                filters=RetrievalFilters(image_types=[ImageType.CHART]),
                use_rerank=False,
            )
            # 图像这一路被过滤到空；文本/BM25 路仍正常返回，
            # 但文本空间里的 caption 分块也必须被排除 —— 否则图片会绕过过滤
            image_route = next(
                result
                for result in outcome.route_results
                if result.route is Route.DENSE_IMAGE and not result.skipped
            )
            assert image_route.count == 0
            assert outcome.items
            assert all(item.image_id != IMG_ID for item in outcome.items)


# --------------------------------------------------------------------------------------
# 7. 查询改写
# --------------------------------------------------------------------------------------


class TestQueryRewrite:
    @pytest.mark.asyncio
    async def test_original_query_always_first(self) -> None:
        settings = _settings(Path("."))
        llm = FakeLLM(reply='{"queries": ["RRF 融合 原理", "倒数排名融合"]}')
        result = await QueryRewriter(llm, settings).rewrite("RRF 是怎么融合的？")
        assert result.queries[0] == "RRF 是怎么融合的？"
        assert result.queries[1:] == ["RRF 融合 原理", "倒数排名融合"]
        assert result.ok

    @pytest.mark.asyncio
    async def test_fenced_json_is_parsed(self) -> None:
        settings = _settings(Path("."))
        llm = FakeLLM(reply='```json\n{"queries": ["a", "b"]}\n```')
        result = await QueryRewriter(llm, settings).rewrite("q")
        assert result.queries == ["q", "a", "b"]

    @pytest.mark.asyncio
    async def test_duplicates_and_long_variants_dropped(self) -> None:
        settings = _settings(Path("."))
        llm = FakeLLM(
            reply=json.dumps(
                {"queries": ["q", "x", "x", "y" * 400, "z"]}, ensure_ascii=False
            )
        )
        result = await QueryRewriter(llm, settings).rewrite("q", n=5)
        assert result.queries == ["q", "x", "z"]

    @pytest.mark.asyncio
    async def test_failure_degrades_to_original(self) -> None:
        settings = _settings(Path("."))
        result = await QueryRewriter(FakeLLM(boom=True), settings).rewrite("q")
        assert result.queries == ["q"]
        assert result.degraded_reason is DegradedReason.LLM_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_search_with_rewrite_merges_variants(self, tmp_path: Path) -> None:
        llm = FakeLLM(reply='{"queries": ["BM25 索引", "稀疏检索"]}')
        async with _env(tmp_path, llm=llm) as env:
            await _seed_mixed_kb(env)
            outcome = await env.retriever.search(
                env.kb, "离线检索", rewrite=True, use_rerank=False
            )
            assert outcome.queries == ["离线检索", "BM25 索引", "稀疏检索"]
            assert llm.calls == 1
