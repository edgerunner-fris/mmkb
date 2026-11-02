"""重排：用交叉编码器给融合后的候选重新打分。

**已知限制，必须在代码里写清楚**
--------------------------------
``gte-rerank`` / ``bge-reranker`` 这类模型**只接受纯文本，无法理解图像**。
图片候选因此只能用「caption 文本 + 章节路径」拼出来的一段文字参与重排，
模板是 ``RERANK_IMAGE_DOC_TEMPLATE``（默认 ``[图片] {heading} | {caption}``）。

这个取舍的实际后果：

* caption 写得好的图 → 重排能正确抬高它；
* 图里有信息但 caption 没写出来（例如复杂架构图、表格截图）→ 会被低估，
  即使它的多模态向量与 query 非常接近。

这是当前所有纯文本 reranker 的共性限制，不是本项目的实现缺陷。
它带来的精度损失由 Batch 8 的 A/B 消融（``hybrid_rrf`` vs
``hybrid_rrf_rerank``）量化，README 的「已知限制」一节也单独列出。

失败一律降级回 RRF 顺序，并记 ``RERANK_UNAVAILABLE`` ——
重排是「让好结果更靠前」，不是「让结果存在」，它挂了不该让整次检索挂掉。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.core.cache import TTLCache, cache_key
from app.core.logging import get_logger
from app.core.perf import get_metrics
from app.core.schemas import DegradedReason, RetrievedItem
from app.providers.base import RerankProvider
from app.retrieval.base import Stopwatch

__all__ = ["RerankOutcome", "Reranker"]

logger = get_logger(__name__)

_FALLBACK_IMAGE_TEMPLATE = "[图片] {heading} | {caption}"


@dataclass(slots=True)
class RerankOutcome:
    """重排结果。``applied=False`` 表示这次用的是 RRF 顺序。"""

    items: list[RetrievedItem]
    applied: bool = False
    latency_ms: int = 0
    degraded_reason: DegradedReason | None = None
    error: str = ""


class Reranker:
    """重排 provider 的薄包装：负责拼文本、截断、排序与失败降级。

    业务代码不直接调 ``provider.rerank``，因为「图片用 caption 拼文本」
    这件事必须只有一处实现 —— 散落在多个调用点时，总有一处会忘了拼章节路径，
    而那种降级是静默的：结果照样返回，只是排序变差。
    """

    def __init__(self, provider: RerankProvider, settings: Settings) -> None:
        self.provider = provider
        self._settings = settings
        cache_cfg = settings.cache
        #: (query, documents) → 原始打分。只缓存**成功的**打分；
        #: 失败与空结果不缓存 —— 那是一次瞬时故障，下一次请求应该重新尝试。
        self._cache: TTLCache[tuple[tuple[int, float], ...]] = TTLCache(
            "rerank.scores",
            ttl_s=cache_cfg.rerank_ttl_s,
            max_entries=cache_cfg.rerank_max_entries,
            enabled=cache_cfg.enabled,
        )

    def cache_stats(self) -> dict[str, object]:
        return {"scores": self._cache.stats().to_dict()}

    def clear_cache(self) -> None:
        """清空缓存并归零命中率统计（``POST /api/performance/cache/clear``）。"""
        self._cache.clear()
        self._cache.reset_stats()

    def _cache_key(self, query: str, documents: list[str]) -> str:
        # 文档数量与每一篇的内容都进键：任何一篇变了，重排结果就不再可复用。
        # 不能只拿 query 做键 —— 同一个问题在不同知识库上的候选集不同。
        return cache_key(
            self.provider.name,
            self.provider.model,
            query,
            len(documents),
            *documents,
        )

    # ------------------------------------------------------------------ 拼文本

    def build_document(self, item: RetrievedItem) -> str:
        """候选 → 送给 reranker 的文本。

        * 文本类：直接用分块正文。分块在切分阶段已经带了标题路径前缀，
          再拼一次只会重复。
        * 图片类：按模板拼「章节路径 + caption」。caption 本身也以章节路径开头
          （见 ``build_caption_text``），所以先把它剥掉，避免标题出现两遍 ——
          重复的标题会稀释 caption 主体的权重，对重排分数有实际影响。
        """
        heading = item.heading_text or "（无章节）"
        if not item.is_image:
            return item.text.strip() or heading

        caption = item.text.strip()
        if item.heading_path and caption.startswith(item.heading_text):
            # 先 strip 再剥分隔符：caption 里标题与实际内容之间是 " | "，
            # 不先去掉空白的话 lstrip("|") 匹配不到第一个字符
            caption = caption[len(item.heading_text) :].strip().lstrip("|").strip()
        caption = caption or (item.metadata.get("caption") or "").strip()
        if not caption:
            # caption 为空时至少把章节路径给出去：总比一个空字符串强，
            # 空文档在多数 reranker 上会得到无意义的分数
            return f"[图片] {heading}"
        template = self._settings.rerank.image_doc_template or _FALLBACK_IMAGE_TEMPLATE
        try:
            return template.format(heading=heading, caption=caption)
        except (KeyError, IndexError, ValueError):
            # 模板是用户配置项，写错不该让检索挂掉
            logger.warning(
                "RERANK_IMAGE_DOC_TEMPLATE 无法用 {heading}/{caption} 格式化，"
                "本次回退到内置模板",
                extra={"template": template},
            )
            return _FALLBACK_IMAGE_TEMPLATE.format(heading=heading, caption=caption)

    # ------------------------------------------------------------------ 重排

    async def apply(
        self,
        query: str,
        items: list[RetrievedItem],
        *,
        top_n: int,
        input_k: int | None = None,
    ) -> RerankOutcome:
        """对 ``items`` 重排，返回最多 ``top_n`` 条。

        ``input_k`` 是送进模型的候选上限（``RETRIEVAL_RERANK_INPUT_K``）：
        rerank 的延迟与 token 成本都随候选数线性增长，而排在第 50 名之后的
        候选本来也几乎不可能进最终 top_k。
        """
        if not items:
            return RerankOutcome(items=[])
        limit = input_k if input_k is not None and input_k > 0 else len(items)
        candidates = items[:limit]

        started = Stopwatch()
        documents = [self.build_document(item) for item in candidates]
        metrics = get_metrics()

        key = self._cache_key(query, documents)
        cached = self._cache.get(key)
        if cached is not None:
            # 命中时**不调用** provider，但依旧走同一段排序 / 截断 / 阈值逻辑，
            # 保证缓存命中与未命中产出的 items 完全一致（除了 latency）。
            metrics.incr("cache.rerank.hit")
            latency = started.ms()
            metrics.observe("rerank.total", latency)
            return self._finalize(candidates, list(cached), top_n, latency_ms=latency)

        metrics.incr("cache.rerank.miss")
        metrics.incr("rerank.calls")
        try:
            result = await self.provider.rerank(query, documents, top_n=len(documents))
        except Exception as exc:  # noqa: BLE001 - 重排可降级，任何失败都不该中断检索
            latency = started.ms()
            metrics.observe("rerank.total", latency, error=True)
            logger.warning(
                "重排失败，回退到 RRF 顺序",
                extra={"provider": self.provider.name, "error": str(exc)[:200]},
            )
            return RerankOutcome(
                items=items[:top_n],
                applied=False,
                latency_ms=latency,
                degraded_reason=DegradedReason.RERANK_UNAVAILABLE,
                error=str(exc)[:500],
            )

        if not result.hits:
            latency = started.ms()
            metrics.observe("rerank.total", latency, error=True)
            logger.warning("重排返回空结果，回退到 RRF 顺序")
            return RerankOutcome(
                items=items[:top_n],
                applied=False,
                latency_ms=latency,
                degraded_reason=DegradedReason.RERANK_UNAVAILABLE,
                error="reranker 返回空结果",
            )

        pairs = tuple((int(hit.index), float(hit.score)) for hit in result.hits)
        # 只缓存成功且非空的结果；下标合法性在 _finalize 里统一校验，
        # 因此缓存里存的是 provider 原始输出，而不是我们「修过」的版本。
        self._cache.set(key, pairs)

        latency = started.ms()
        metrics.observe("rerank.total", latency)
        metrics.observe("rerank.provider", latency)
        metrics.incr("rerank.documents_scored", len(documents))
        return self._finalize(candidates, list(pairs), top_n, latency_ms=latency)

    def _finalize(
        self,
        candidates: list[RetrievedItem],
        hits: list[tuple[int, float]],
        top_n: int,
        *,
        latency_ms: int = 0,
    ) -> RerankOutcome:
        """把 ``(下标, 分数)`` 落到候选上：校验、阈值、排序、补尾。"""
        scored: list[tuple[int, float, RetrievedItem]] = []
        seen: set[int] = set()
        for index, score in hits:
            if not 0 <= index < len(candidates) or index in seen:
                # 下标越界或重复说明 provider 实现有问题；跳过这一条而不是
                # 用错误的下标去索引，后者会取到完全无关的文档
                logger.warning(
                    "重排返回了非法下标，已忽略",
                    extra={"index": index, "count": len(candidates)},
                )
                continue
            seen.add(index)
            item = candidates[index].model_copy(deep=True)
            item.rerank_score = float(score)
            item.score = float(score)
            scored.append((index, float(score), item))

        threshold = self._settings.retrieval.min_score_after_rerank
        scored = [entry for entry in scored if entry[1] >= threshold]
        # 同分时按重排前的顺序（RRF 名次）兜底，保证可复现
        scored.sort(key=lambda entry: (-entry[1], entry[0]))
        ordered = [entry[2] for entry in scored]

        # provider 没打分的候选（某些实现只返回 top_n）保留在末尾，
        # 而不是直接丢掉 —— 丢掉会让最终的上下文在「刚好够」时意外变少
        for index, item in enumerate(candidates):
            if index not in seen:
                tail = item.model_copy(deep=True)
                tail.rerank_score = None
                if threshold <= 0.0:
                    ordered.append(tail)

        return RerankOutcome(
            items=ordered[:top_n],
            applied=True,
            latency_ms=latency_ms,
        )
