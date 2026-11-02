"""混合检索编排：三路并行召回 → RRF 融合 → 重排 → 邻居扩展 → 回填。

这个模块只做**编排**。每一路的实现、融合公式、重排封装都在各自的模块里，
这里负责的是「只有把它们串起来才会出现」的问题：

1. **哪几路该跑** —— 配置开关、provider 是否可用、权重是否为 0、
   collection 是否有数据。四种原因分开记录，前端才能给出有意义的提示
   （「没配 Key」和「库是空的」对用户的下一步动作完全不同）。
2. **失败隔离** —— 某一路网络故障只让那一路为空，其余照常；
   但「知识库快照与当前配置不一致」必须让整次检索失败（见
   :func:`~app.retrieval.base.guard_kb_snapshot`）。
3. **多查询融合** —— 改写出的每个 query 都独立跑每一路，
   所有 ``(query, route)`` 列表一起进 RRF。
4. **后处理顺序** —— 融合 → 重排 → 邻居扩展 → 回填展示字段。
   这个顺序不能换：重排前扩展邻居会让 reranker 看到正文之外的内容，
   回填必须在扩展之后，否则扩展出来的内容没有文件名。

并发模型
--------
所有 ``(query, route)`` 任务一次性 ``gather``。三路之间没有依赖，
串行执行会让延迟变成三者之和；而对一次问答来说，检索延迟几乎全部
花在「等外部 API」上，并发是这里最有效的一处优化。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.core.errors import RetrievalError
from app.core.logging import get_logger
from app.core.perf import get_metrics
from app.core.schemas import (
    DegradedReason,
    KnowledgeBase,
    RetrievalFilters,
    RetrievedItem,
    Route,
)
from app.providers.base import EmbeddingProvider, LLMProvider, RerankProvider
from app.retrieval.base import (
    ROUTE_WEIGHT_ORDER,
    RouteResult,
    Stopwatch,
    hydrate_items,
    route_weight_map,
)
from app.retrieval.dense_image import DenseImageRetriever
from app.retrieval.dense_text import DenseTextRetriever
from app.retrieval.fusion import RankedList, reciprocal_rank_fusion
from app.retrieval.query_rewrite import QueryRewriter, QueryRewriteResult
from app.retrieval.rerank import Reranker
from app.retrieval.sparse import SparseRetriever
from app.store.bm25store import BM25Store
from app.store.docstore import DocStore
from app.store.vectorstore import VectorStore

__all__ = ["HybridRetriever", "SearchOutcome"]

logger = get_logger(__name__)

#: 改写 query 的权重。原问题 1.0，每条改写 0.7。
#:
#: 为什么不相等：改写是「提高召回的手段」，不是「用户想找的东西」。
#: 某条改写跑偏时，低权重让它在 RRF 里只能给条目加一点分，
#: 不足以把原问题没命中、但语义沾边的噪声顶到前面。
REWRITE_WEIGHT = 0.7


# --------------------------------------------------------------------------------------
# 结果
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class SearchOutcome:
    """一次检索的完整快照。

    它同时服务三个消费者：答案生成（``items``）、前端调试面板
    （``route_results`` / ``timings``）、评估脚本（``degraded_reasons``）。
    把「每一路为什么没出结果」保留下来，比只返回一个条目列表有用得多。
    """

    query: str
    items: list[RetrievedItem] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    route_results: list[RouteResult] = field(default_factory=list)
    degraded_reasons: list[DegradedReason] = field(default_factory=list)
    timings: dict[str, int] = field(default_factory=dict)
    reranked: bool = False
    filters: RetrievalFilters | None = None
    #: 融合后的去重候选数（重排 / 截断之前）。``debug_trace`` 用它回答
    #: 「三路一共捞上来多少条，最后为什么只剩这么几条」。
    fused_count: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.items

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "queries": list(self.queries),
            "count": len(self.items),
            "items": [item.model_dump(mode="json") for item in self.items],
            "routes": [
                {
                    "route": str(result.route),
                    "count": result.count,
                    "latency_ms": result.latency_ms,
                    "skipped": result.skipped,
                    "note": result.note,
                    "error": result.error,
                    "degraded_reason": (
                        str(result.degraded_reason) if result.degraded_reason else None
                    ),
                }
                for result in self.route_results
            ],
            "degraded_reasons": [str(reason) for reason in self.degraded_reasons],
            "timings": dict(self.timings),
            "reranked": self.reranked,
            "fused_count": self.fused_count,
        }


@dataclass(slots=True)
class _RouteSpec:
    route: Route
    retriever: SparseRetriever | DenseTextRetriever | DenseImageRetriever


# --------------------------------------------------------------------------------------
# 编排
# --------------------------------------------------------------------------------------


class HybridRetriever:
    """三路混合检索。

    依赖全部通过构造函数注入，不读全局单例 —— 测试里可以只给一个假 BM25
    库和假 provider，跑一次完整的三路融合而不碰网络。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        docstore: DocStore,
        vectorstore: VectorStore,
        bm25store: BM25Store,
        text_embedding: EmbeddingProvider | None = None,
        image_embedding: EmbeddingProvider | None = None,
        rerank: RerankProvider | None = None,
        llm: LLMProvider | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.docstore = docstore
        self.vectorstore = vectorstore
        self.bm25store = bm25store

        self._retrievers: dict[Route, _RouteSpec] = {
            Route.BM25: _RouteSpec(
                Route.BM25, SparseRetriever(bm25store=bm25store, settings=self._settings)
            )
        }
        if text_embedding is not None:
            self._retrievers[Route.DENSE_TEXT] = _RouteSpec(
                Route.DENSE_TEXT,
                DenseTextRetriever(
                    embedding=text_embedding,
                    vectorstore=vectorstore,
                    settings=self._settings,
                ),
            )
        if image_embedding is not None:
            self._retrievers[Route.DENSE_IMAGE] = _RouteSpec(
                Route.DENSE_IMAGE,
                DenseImageRetriever(
                    embedding=image_embedding,
                    vectorstore=vectorstore,
                    settings=self._settings,
                ),
            )

        self._reranker = Reranker(rerank, self._settings) if rerank is not None else None
        self._rewriter = rewriter or (
            QueryRewriter(llm, self._settings) if llm is not None else None
        )

    # ------------------------------------------------------------------ 对外接口

    async def search(
        self,
        kb: KnowledgeBase,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        routes: Iterable[Route | str] | None = None,
        use_rerank: bool | None = None,
        rewrite: bool = False,
        expand_neighbors: bool | None = None,
    ) -> SearchOutcome:
        """单查询检索。取一条最常用的路径，复杂场景用 :meth:`search_multi`。"""
        queries = [query]
        rewrite_result: QueryRewriteResult | None = None
        if rewrite and self._rewriter is not None and query.strip():
            rewrite_result = await self._rewriter.rewrite(query)
            queries = rewrite_result.queries

        outcome = await self.search_multi(
            kb,
            queries,
            original_query=query,
            top_k=top_k,
            filters=filters,
            routes=routes,
            use_rerank=use_rerank,
            expand_neighbors=expand_neighbors,
        )
        if rewrite_result is not None and rewrite_result.degraded_reason is not None:
            _append_reason(outcome.degraded_reasons, rewrite_result.degraded_reason)
        return outcome

    async def search_multi(
        self,
        kb: KnowledgeBase,
        queries: list[str],
        *,
        original_query: str | None = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        routes: Iterable[Route | str] | None = None,
        use_rerank: bool | None = None,
        expand_neighbors: bool | None = None,
    ) -> SearchOutcome:
        """多查询检索。``queries[0]`` 必须是原问题（权重 1.0）。"""
        total = Stopwatch()
        cleaned = _dedupe_queries(queries)
        if not cleaned:
            raise RetrievalError("查询为空，无法检索。")

        cfg = self._settings.retrieval
        final_top_k = top_k if top_k is not None else cfg.final_top_k
        weights = route_weight_map(self._settings)
        allowed = _normalize_routes(routes)

        # 1. 规划：哪几路能跑
        plan_started = Stopwatch()
        active, route_results = await self._plan(kb, allowed, weights)
        plan_ms = plan_started.ms()
        degraded: list[DegradedReason] = [
            result.degraded_reason
            for result in route_results
            if result.degraded_reason is not None
        ]

        # 2. 并发召回：每个 (query, route) 组合一个任务
        retrieve_started = Stopwatch()
        executed: list[RouteResult] = []
        if active:
            tasks = [
                self._run_one(
                    spec, kb, query, query_index, len(cleaned), filters, weights
                )
                for query_index, query in enumerate(cleaned)
                for spec in active
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):  # pragma: no cover - 防御
                    logger.error("检索任务异常逃逸", exc_info=result)
                    continue
                executed.append(result)
                route_results.append(result)
                if result.degraded_reason is not None:
                    _append_reason(degraded, result.degraded_reason)
        retrieve_ms = retrieve_started.ms()

        # 3. 融合
        ranked_lists = _collect_ranked(executed)
        fuse_started = Stopwatch()
        fused = reciprocal_rank_fusion(ranked_lists, k=cfg.rrf_k)
        fuse_ms = fuse_started.ms()
        timings: dict[str, int] = {
            "plan": plan_ms,
            "retrieve": retrieve_ms,
            "fuse": fuse_ms,
        }

        # 4. 重排。用**原问题**打分，而不是某条改写：
        #    最终答案是为原问题服务的，改写只是召回手段。
        reranked = False
        want_rerank = cfg.enable_rerank if use_rerank is None else use_rerank
        if want_rerank and self._reranker is not None and fused:
            rerank_outcome = await self._reranker.apply(
                original_query or cleaned[0],
                fused,
                top_n=final_top_k,
                input_k=cfg.rerank_input_k,
            )
            items = rerank_outcome.items
            reranked = rerank_outcome.applied
            if rerank_outcome.degraded_reason is not None:
                _append_reason(degraded, rerank_outcome.degraded_reason)
            timings["rerank"] = rerank_outcome.latency_ms
        else:
            if want_rerank and self._reranker is None:
                # 「配置里开了重排，但没配 Key」—— 这是一次真实降级，
                # 必须让调用方知道排序是纯 RRF 的
                _append_reason(degraded, DegradedReason.RERANK_UNAVAILABLE)
            items = fused[:final_top_k]
            timings["rerank"] = 0

        # 5. 邻居扩展
        expand_started = Stopwatch()
        do_expand = (
            cfg.neighbor_expand > 0 if expand_neighbors is None else expand_neighbors
        )
        if do_expand and cfg.neighbor_expand > 0 and items:
            await self._expand_neighbors(items, cfg.neighbor_expand)
        timings["expand"] = expand_started.ms()

        # 6. 回填展示字段
        hydrate_started = Stopwatch()
        await hydrate_items(items, self.docstore)
        timings["hydrate"] = hydrate_started.ms()
        timings["total"] = total.ms()

        metrics = get_metrics()
        metrics.observe("retrieval.total", timings["total"])
        metrics.observe("retrieval.plan", plan_ms)
        metrics.observe("retrieval.retrieve", retrieve_ms)
        metrics.observe("retrieval.fuse", fuse_ms)
        metrics.observe("retrieval.expand", timings["expand"])
        metrics.observe("retrieval.hydrate", timings["hydrate"])
        if want_rerank and fused:
            metrics.observe("retrieval.rerank", timings.get("rerank", 0))
        metrics.incr("retrieval.requests")
        metrics.incr("retrieval.queries", len(cleaned))
        metrics.incr("retrieval.routes_active", len(active))
        if reranked:
            metrics.incr("retrieval.reranked")
        if not items:
            metrics.incr("retrieval.empty")
        for reason in degraded:
            metrics.incr(f"retrieval.degraded.{reason}")

        logger.info(
            "检索完成",
            extra={
                "kb_id": kb.kb_id,
                "queries": len(cleaned),
                "routes": sorted({str(result.route) for result in route_results}),
                "items": len(items),
                "reranked": reranked,
                "degraded": [str(reason) for reason in degraded],
                "total_ms": timings["total"],
            },
        )

        return SearchOutcome(
            query=original_query or cleaned[0],
            items=items,
            queries=cleaned,
            route_results=route_results,
            degraded_reasons=degraded,
            timings=timings,
            reranked=reranked,
            filters=filters,
            fused_count=len(fused),
        )

    # ------------------------------------------------------------------ 可观测性

    def cache_stats(self) -> dict[str, Any]:
        """收集这条链路上所有缓存的命中率，供性能报告使用。

        用 ``getattr`` 而不是给 ABC 加方法：缓存是可选优化，
        任何自定义 store / rerank provider 都不应该为了「能报命中率」而改接口。
        """
        out: dict[str, Any] = {}
        for spec in self._retrievers.values():
            stats = getattr(spec.retriever, "cache_stats", None)
            if callable(stats):
                out[str(spec.route)] = stats()
        rerank_stats = getattr(self._reranker, "cache_stats", None)
        if callable(rerank_stats):
            out["rerank"] = rerank_stats()
        vector_stats = getattr(self.vectorstore, "cache_stats", None)
        if callable(vector_stats):
            out["vectorstore"] = vector_stats()
        return out

    def clear_caches(self) -> list[str]:
        """清空这条链路上的所有进程内缓存，返回实际清理过的部件名。

        同样用 ``getattr``：缓存是检索器的可选优化，自定义实现没有它也能工作。
        """
        cleared: list[str] = []
        targets: list[tuple[str, Any]] = [
            (str(spec.route), spec.retriever) for spec in self._retrievers.values()
        ]
        targets.append(("rerank", self._reranker))
        targets.append(("vectorstore", self.vectorstore))
        for name, target in targets:
            clear = getattr(target, "clear_cache", None)
            if not callable(clear):
                continue
            try:
                clear()
            except Exception:  # noqa: BLE001 - 清缓存失败只影响收益，不影响可用性
                logger.warning("清理缓存失败", extra={"component": name}, exc_info=True)
                continue
            cleared.append(name)
        return cleared

    # ------------------------------------------------------------------ 规划

    async def _plan(
        self,
        kb: KnowledgeBase,
        allowed: set[Route] | None,
        weights: dict[Route, float],
    ) -> tuple[list[_RouteSpec], list[RouteResult]]:
        """决定哪几路真的执行，并为每一路写一条可解释的 ``RouteResult``。"""
        cfg = self._settings.retrieval
        switches = {
            Route.BM25: cfg.enable_bm25,
            Route.DENSE_TEXT: cfg.enable_dense_text,
            Route.DENSE_IMAGE: cfg.enable_dense_image,
        }
        missing_provider = {
            Route.DENSE_TEXT: (
                DegradedReason.TEXT_EMBED_UNAVAILABLE,
                "TEXT_EMBED_API_KEY 未配置或文本向量 provider 不可用",
            ),
            Route.DENSE_IMAGE: (
                DegradedReason.IMAGE_EMBED_UNAVAILABLE,
                "IMAGE_EMBED_API_KEY 未配置或未启用多模态向量",
            ),
        }

        active: list[_RouteSpec] = []
        by_route: dict[Route, RouteResult] = {}
        candidates: list[_RouteSpec] = []

        # 第一遍只做**不碰 IO** 的判断（开关、权重、provider 是否存在）。
        # 这些条件决定了哪些路值得探测，本身不需要 await。
        for route in ROUTE_WEIGHT_ORDER:
            if allowed is not None and route not in allowed:
                by_route[route] = RouteResult(
                    route=route, skipped=True, note="本次请求未启用该路"
                )
                continue
            if not switches[route]:
                by_route[route] = RouteResult(
                    route=route, skipped=True, note="配置中已关闭该路"
                )
                continue
            if weights.get(route, 0.0) <= 0:
                by_route[route] = RouteResult(
                    route=route, skipped=True, note="权重为 0"
                )
                continue

            spec = self._retrievers.get(route)
            if spec is None:
                reason, note = missing_provider.get(
                    route, (DegradedReason.BM25_UNAVAILABLE, "该路不可用")
                )
                by_route[route] = RouteResult(
                    route=route,
                    skipped=True,
                    note=note,
                    degraded_reason=reason,
                )
                continue
            candidates.append(spec)

        # 第二遍并发探测。``probe`` 会读 collection 条数 / BM25 索引状态，
        # 本质上是几次本地元数据查询；串行时它们的延迟相加。
        # 顺序在装配阶段按 ROUTE_WEIGHT_ORDER 还原，不依赖完成顺序。
        probes: list[Any] = []
        if candidates:
            probes = await asyncio.gather(
                *(spec.retriever.probe(kb) for spec in candidates),
                return_exceptions=True,
            )

        for spec, probe in zip(candidates, probes, strict=True):
            route = spec.route
            if isinstance(probe, RetrievalError):
                # 快照不一致是**不可降级**的：继续检索只会得到无意义的排序
                raise probe
            if isinstance(probe, BaseException):
                logger.warning(
                    "检索路探测失败，本次跳过",
                    extra={"route": str(route), "error": str(probe)[:200]},
                )
                by_route[route] = RouteResult(
                    route=route,
                    skipped=True,
                    error=str(probe)[:500],
                    degraded_reason=spec.retriever.degraded_reason,
                )
                continue
            ok, note = probe
            if not ok:
                by_route[route] = RouteResult(route=route, skipped=True, note=note)
                continue
            active.append(spec)

        results = [by_route[route] for route in ROUTE_WEIGHT_ORDER if route in by_route]
        return active, results

    # ------------------------------------------------------------------ 执行

    async def _run_one(
        self,
        spec: _RouteSpec,
        kb: KnowledgeBase,
        query: str,
        query_index: int,
        query_count: int,
        filters: RetrievalFilters | None,
        weights: dict[Route, float],
    ) -> RouteResult:
        """跑 ``(query, route)`` 这一个组合。任何异常都变成 ``RouteResult.error``。"""
        label = (
            str(spec.route)
            if query_count == 1
            else f"{spec.route}#{query_index + 1}"
        )
        base_weight = weights.get(spec.route, 1.0)
        weight = base_weight if query_index == 0 else base_weight * REWRITE_WEIGHT
        started = Stopwatch()
        metrics = get_metrics()
        try:
            result = await spec.retriever.search(
                kb, query, top_k=self._settings.retrieval.top_k_per_route, filters=filters
            )
        except Exception as exc:  # noqa: BLE001 - 单路失败只能影响这一路
            elapsed = started.ms()
            metrics.observe(f"retrieval.route.{spec.route}", elapsed, error=True)
            metrics.incr(f"retrieval.route_errors.{spec.route}")
            logger.warning(
                "检索路失败，本次跳过该路结果",
                extra={
                    "route": str(spec.route),
                    "query_index": query_index,
                    "error": str(exc)[:200],
                },
            )
            return RouteResult(
                route=spec.route,
                latency_ms=elapsed,
                error=str(exc)[:500],
                degraded_reason=spec.retriever.degraded_reason,
                label=label,
                weight=weight,
            )
        result.latency_ms = result.latency_ms or started.ms()
        metrics.observe(f"retrieval.route.{spec.route}", result.latency_ms)
        metrics.incr(f"retrieval.route_calls.{spec.route}")
        result.label = label
        result.weight = weight
        return result

    # ------------------------------------------------------------------ 后处理

    async def _expand_neighbors(self, items: list[RetrievedItem], window: int) -> None:
        """把每条结果的相邻分块取回来，挂在 ``metadata["expanded_text"]``。

        为什么是挂 metadata 而不是新增条目：邻居不是「另一条召回结果」，
        它只是这条结果的上下文延长。若作为独立条目进入重排，
        同一段正文会出现多次，最终上下文里一半是重复内容。

        只对文本分块做 —— 图片本来就没有相邻分块的概念。
        """
        targets = [
            item for item in items if item.chunk_id and not item.is_image
        ]
        if not targets:
            return
        try:
            neighbours = await asyncio.gather(
                *(
                    self.docstore.get_neighbor_chunks(
                        item.chunk_id or "", before=window, after=window
                    )
                    for item in targets
                ),
                return_exceptions=True,
            )
        except Exception:  # noqa: BLE001 - 扩展失败不影响主结果
            logger.warning("邻居扩展失败", exc_info=True)
            return

        for item, chunks in zip(targets, neighbours, strict=True):
            if isinstance(chunks, BaseException) or not chunks:
                continue
            item.metadata["expanded_text"] = "\n\n".join(chunk.text for chunk in chunks)
            item.metadata["expanded_chunk_ids"] = [chunk.chunk_id for chunk in chunks]


# --------------------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------------------


def _dedupe_queries(queries: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in queries:
        text = (raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_routes(routes: Iterable[Route | str] | None) -> set[Route] | None:
    if routes is None:
        return None
    allowed: set[Route] = set()
    for raw in routes:
        try:
            route = raw if isinstance(raw, Route) else Route(str(raw))
        except ValueError as exc:
            raise RetrievalError(
                f"未知的检索路 {raw!r}，可选：bm25、dense_text、dense_image"
            ) from exc
        if route is Route.HYBRID:
            raise RetrievalError("hybrid 是融合后的结果标记，不能作为检索路指定。")
        allowed.add(route)
    return allowed or None


def _append_reason(reasons: list[DegradedReason], reason: DegradedReason) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _collect_ranked(results: list[RouteResult]) -> list[RankedList]:
    """把任务结果整理成 ``RankedList``。

    ``_run_one`` 已经把 ``label`` / ``weight`` 写进 ``RouteResult``，
    这里只做格式转换。空列表直接丢弃 —— 一条没命中的路对 RRF 没有贡献。
    """
    ranked: list[RankedList] = []
    for result in results:
        if not result.items or result.skipped or result.error:
            continue
        ranked.append(
            RankedList(
                label=result.label or str(result.route),
                route=result.route,
                items=result.items,
                weight=result.weight,
            )
        )
    return ranked
