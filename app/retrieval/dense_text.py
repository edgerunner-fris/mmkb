"""铁律 1：文本查文本。

query 用文本 embedding 模型编码，检索 ``{kb_id}__text``。

这个 collection 里放两类向量：

* 文档分块向量（正文 / 代码 / 表格）；
* 图片 caption 向量 —— caption 本身是一段文本，所以图片天然能被文本检索命中。

**不要在这里用多模态模型编码 query。** 多模态向量属于另一个空间，
只该用于 :mod:`app.retrieval.dense_image`。两个 provider 的 ``space`` 属性
不同，混用会在 :func:`~app.retrieval.base.guard_kb_snapshot` 里被拦下。
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
    item_from_vector_hit,
)
from app.store.vectorstore import VectorStore

__all__ = ["DenseTextRetriever"]


class DenseTextRetriever:
    """``{kb}__text`` 的向量召回。"""

    route = Route.DENSE_TEXT
    #: 这一路失败时对外报告的降级原因
    degraded_reason = DegradedReason.TEXT_EMBED_UNAVAILABLE

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
        #: query → 向量。键里带上 provider/model/dim/space：
        #: 换模型或换维度后旧向量绝不能复用（那正是铁律 3 要防的混空间）。
        self._query_cache: TTLCache[tuple[float, ...]] = TTLCache(
            "query_embedding.text",
            ttl_s=cache_cfg.query_embedding_ttl_s,
            max_entries=cache_cfg.query_embedding_max_entries,
            enabled=cache_cfg.enabled,
        )

    def _cache_key(self, query: str) -> str:
        return cache_key(
            "text",
            self.embedding.name,
            self.embedding.model,
            self.embedding.dim,
            self.embedding.space,
            query,
        )

    def cache_stats(self) -> dict[str, object]:
        return {"query_embedding": self._query_cache.stats().to_dict()}

    def clear_cache(self) -> None:
        """清空缓存并归零命中率统计（``POST /api/performance/cache/clear``）。"""
        self._query_cache.clear()
        self._query_cache.reset_stats()

    async def probe(self, kb: KnowledgeBase) -> tuple[bool, str]:
        """跑之前先判断「这一路有没有意义」。

        先查 count 再编码，是为了**不在空库上花钱**：``--no-embed`` 导入的
        知识库在文本 collection 里一条向量都没有，此时发一次 embedding 请求
        既浪费配额又一无所得。快照校验放在 count 之后同理 ——
        空的 collection 不存在「新旧向量不可比」的问题。
        """
        if not self.embedding.supports_text:
            raise RetrievalError(
                f"{self.embedding.name}（{self.embedding.model}）不能编码文本，"
                "无法用于 dense_text 路。",
                detail={"provider": self.embedding.name, "space": self.embedding.space},
            )
        count = await self.vectorstore.count(kb.text_collection)
        if count <= 0:
            return False, f"{kb.text_collection} 为空（导入时可能用了 --no-embed）"
        guard_kb_snapshot(kb, self.embedding, space="text")
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

        # 查询向量必须是文本模型产出的、属于 text 空间的向量（铁律 1）
        key = self._cache_key(query)
        cached = self._query_cache.get(key)
        if cached is not None:
            vector = list(cached)
            get_metrics().incr("cache.query_embedding.hit")
        else:
            get_metrics().incr("cache.query_embedding.miss")
            result = await self.embedding.embed_texts([query])
            vector = result.vectors[0] if result.vectors else []
            if vector:
                self._query_cache.set(key, tuple(vector))
        if not vector:
            # 用 ProviderResponseError 而不是 RetrievalError：这是**可降级**的
            # 单路故障，其余几路仍然有效。RetrievalError 留给「快照不一致」
            # 这类继续检索只会得到垃圾排序的问题。
            raise ProviderResponseError(
                "文本 embedding 返回了空向量，无法检索。",
                provider=self.embedding.name,
                model=self.embedding.model,
            )

        where = None if filters is None or filters.is_empty() else filters.to_chroma_where(
            space="text"
        )
        hits = await self.vectorstore.query(
            kb.text_collection,
            vector,
            top_k=top_k,
            where=where,
            # 铁律 3 的运行时兜底：collection 若是 image 空间，这里直接报错
            space="text",
        )
        items = [
            item_from_vector_hit(hit, kb_id=kb.kb_id, route=self.route, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]
        return RouteResult(route=self.route, items=items, latency_ms=started.ms())
