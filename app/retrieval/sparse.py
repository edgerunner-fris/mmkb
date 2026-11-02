"""BM25 稀疏召回。

BM25 与向量召回是**互补**而不是备份：它对专有名词、型号、错误码、
API 名称这类「语义模型容易糊掉、字面必须一致」的查询特别有效，
而这些恰好是技术文档问答里最常见的一类问题。

中文必须走 jieba 分词（见 :mod:`app.store.bm25store`），
否则整句会被正则切成一个 token，这一路直接失效。

**过滤条件与向量库不完全等价。** BM25 的 ``where`` 在打分之后求值
（见 ``BM25Store.search`` 的说明），而且只支持等值/成员判断。
时间范围在这条路上无法表达，:func:`_bm25_where` 会把它去掉并在日志里说明 ——
静默丢掉过滤条件比报错更糟：用户会以为筛选生效了，只是没搜到。
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.core.logging import get_logger
from app.core.schemas import (
    BlockType,
    DegradedReason,
    KnowledgeBase,
    RetrievalFilters,
    Route,
)
from app.retrieval.base import RouteResult, Stopwatch, item_from_bm25_hit
from app.store.bm25store import BM25Store

__all__ = ["SparseRetriever"]

logger = get_logger(__name__)


def _bm25_where(filters: RetrievalFilters | None) -> dict[str, Any] | None:
    """把检索过滤条件收窄到 BM25 能表达的形态。

    只保留 ``doc_id`` 与 ``block_type``：它们是分块 metadata 里的枚举型字段，
    ``_match_payload`` 能精确求值。``image_type`` 是图片的字段，
    时间范围需要 ``$gte``/``$lte``，BM25 侧都不支持。
    """
    if filters is None or filters.is_empty():
        return None
    clauses: list[dict[str, Any]] = []
    if filters.block_types:
        clauses.append({"block_type": {"$in": [str(b) for b in filters.block_types]}})
    if filters.doc_ids:
        clauses.append({"doc_id": {"$in": list(filters.doc_ids)}})
    # 与 ``RetrievalFilters.to_chroma_where(space="text")`` 保持同一语义：
    # 按图片类型过滤时，文本空间里的 caption 分块无法表达该条件，排除掉
    if (
        filters.image_types
        and str(BlockType.IMAGE_CAPTION)
        not in {str(block) for block in filters.block_types}
    ):
        clauses.append({"block_type": {"$ne": str(BlockType.IMAGE_CAPTION)}})
    if (filters.created_after or filters.created_before) and clauses:
        logger.debug("BM25 路不支持时间范围过滤，本次检索将忽略 created_* 条件")
    if not clauses:
        if filters.created_after or filters.created_before or filters.image_types:
            logger.debug("过滤条件在 BM25 路上没有可表达的字段，这一路将不带过滤")
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class SparseRetriever:
    """内存 BM25 索引上的召回。"""

    route = Route.BM25
    degraded_reason = DegradedReason.BM25_UNAVAILABLE

    def __init__(self, *, bm25store: BM25Store, settings: Settings) -> None:
        self.bm25store = bm25store
        self._settings = settings

    async def probe(self, kb: KnowledgeBase) -> tuple[bool, str]:
        await self.bm25store.ensure_kb(kb.kb_id)
        count = await self.bm25store.count(kb.kb_id)
        if count <= 0:
            return False, "BM25 索引为空（没有已导入的分块）"
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

        hits = await self.bm25store.search(
            kb.kb_id, query, top_k=top_k, where=_bm25_where(filters)
        )
        # BM25Hit.rank 已经是 1 起的名次，直接用；不要按列表下标重排 ——
        # search() 只返回正分结果，两者在「有 0 分候选」时不是一回事
        items = [
            item_from_bm25_hit(hit, kb_id=kb.kb_id, route=self.route, rank=hit.rank)
            for hit in hits
        ]
        return RouteResult(route=self.route, items=items, latency_ms=started.ms())
