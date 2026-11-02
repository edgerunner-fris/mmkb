"""铁律 2：以文搜图。

query 是一段纯文本，但**不能**用文本 embedding 编码 —— 它必须送进
``qwen3-vl-embedding`` 的原生端点，以 text-only 输入、``enable_fusion=true``
编码，然后检索 ``{kb_id}__image``。

以文搜图的 query 必须走多模态融合空间，不能用 text-embedding-v4 编码。

为什么这条注释放在这里而不是只写在文档里：这是整个项目最容易犯、也最难发现的
错误。两个空间如果维度恰好相同（例如都配成 1024），拿文本向量去查图像库
不会抛任何异常，只会返回一批「看起来排好了、其实完全随机」的结果。
真正的错误要等到人工看 demo 时才会暴露。

因此本模块做了三件事：
1. 只调用 ``embed_fused``（文本 provider 调用它会抛
   ``ProviderUnsupportedOperationError``）；
2. ``VectorStore.query(..., space="image")`` 在存储层再校验一次 collection 空间；
3. :func:`~app.retrieval.base.guard_kb_snapshot` 校验 provider 与知识库快照。
"""

from __future__ import annotations

from app.config import Settings
from app.core.cache import TTLCache, cache_key
from app.core.errors import ProviderResponseError, RetrievalError
from app.core.perf import get_metrics
from app.core.schemas import DegradedReason, KnowledgeBase, RetrievalFilters, Route
from app.providers.base import EmbeddingProvider
from app.retrieval.base import (
    RouteResult,
    Stopwatch,
    guard_kb_snapshot,
    item_from_image_hit,
)
from app.store.vectorstore import VectorStore

__all__ = ["DenseImageRetriever"]


class DenseImageRetriever:
    """``{kb}__image`` 的跨模态召回。"""

    route = Route.DENSE_IMAGE
    degraded_reason = DegradedReason.IMAGE_EMBED_UNAVAILABLE

    def __init__(
        self,
        *,
        embedding: EmbeddingProvider,
        vectorstore: VectorStore,
        settings: Settings,
    ) -> None:
        self.embedding = embedding
        self.vectorstore = vectorstore
        self._settings = settings
        cache_cfg = settings.cache
        #: 融合空间的 query 缓存。键里必须带 ``enable_fusion``：
        #: 同一段文本在 fusion 开/关两种模式下会得到不同的向量，
        #: 它们属于不同子空间，混用等于悄悄破坏铁律 3。
        self._query_cache: TTLCache[tuple[float, ...]] = TTLCache(
            "query_embedding.image",
            ttl_s=cache_cfg.query_embedding_ttl_s,
            max_entries=cache_cfg.query_embedding_max_entries,
            enabled=cache_cfg.enabled,
        )

    def _cache_key(self, query: str) -> str:
        return cache_key(
            "image",
            self.embedding.name,
            self.embedding.model,
            self.embedding.dim,
            self.embedding.space,
            self._settings.image_embed.enable_fusion,
            query,
        )

    def cache_stats(self) -> dict[str, object]:
        return {"query_embedding": self._query_cache.stats().to_dict()}

    def clear_cache(self) -> None:
        """清空缓存并归零命中率统计（``POST /api/performance/cache/clear``）。"""
        self._query_cache.clear()
        self._query_cache.reset_stats()

    async def probe(self, kb: KnowledgeBase) -> tuple[bool, str]:
        if not self.embedding.supports_fused:
            raise RetrievalError(
                f"{self.embedding.name}（{self.embedding.model}）不支持融合空间编码，"
                "无法用于以文搜图。",
                detail={"provider": self.embedding.name, "space": self.embedding.space},
            )
        count = await self.vectorstore.count(kb.image_collection)
        if count <= 0:
            return False, f"{kb.image_collection} 为空（没有可检索的图片向量）"
        guard_kb_snapshot(kb, self.embedding, space="image")
        return True, ""

    async def search(
        self,
        kb: KnowledgeBase,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> RouteResult:
        started = Stopwatch()
        if not query.strip():
            return RouteResult(route=self.route, skipped=True, note="空 query")

        # 以文搜图的 query 必须走多模态融合空间，不能用 text-embedding-v4 编码。
        # enable_fusion / 模型 / 维度必须与图片入库时完全一致，否则 query 与图片
        # 落在两个不同的子空间里，余弦相似度没有意义。
        key = self._cache_key(query)
        cached = self._query_cache.get(key)
        if cached is not None:
            vector = list(cached)
            get_metrics().incr("cache.query_embedding.hit")
        else:
            get_metrics().incr("cache.query_embedding.miss")
            result = await self.embedding.embed_fused([query])
            vector = result.vectors[0] if result.vectors else []
            if vector:
                self._query_cache.set(key, tuple(vector))
        if not vector:
            # 同 dense_text：单路故障，可降级
            raise ProviderResponseError(
                "多模态 embedding 返回了空向量，无法检索。",
                provider=self.embedding.name,
                model=self.embedding.model,
            )

        where = None if filters is None or filters.is_empty() else filters.to_chroma_where(
            space="image"
        )
        # 图像空间不需要「每路 20 条」那么多：图片结果最终占用的是答案里的
        # 视觉引用位，条数过多只会把文本证据挤出去。默认上限 10。
        limit = min(top_k, self._settings.retrieval.image_query_top_k)
        hits = await self.vectorstore.query(
            kb.image_collection,
            vector,
            top_k=limit,
            where=where,
            space="image",
        )
        items = [
            item_from_image_hit(hit, kb_id=kb.kb_id, route=self.route, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]
        return RouteResult(route=self.route, items=items, latency_ms=started.ms())
