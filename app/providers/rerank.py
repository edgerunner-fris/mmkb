"""重排 provider。

⚠️ 已知限制，必须正面说明
================================================================
``gte-rerank-v2`` / ``BAAI/bge-reranker-v2-m3`` 这类交叉编码器
**只接受纯文本输入，无法理解图像**。它们没有视觉侧，喂图片字节进去没有意义。

所以在本项目的检索链路里：

* 文本候选 → 直接把 chunk 正文送进重排；
* **图片候选 → 送进的是 caption + 章节路径拼出来的文本**
  （见 :func:`app.core.schemas.build_caption_text`，模板可配 ``RERANK_IMAGE_DOC_TEMPLATE``）。

由此产生一个**明确的、可量化的精度损失**：一张视觉上和问题高度相关、
但 VLM 没能把关键信息写进 caption 的图，在重排阶段会被低估甚至挤出候选。
这不是实现缺陷，而是当前所有纯文本 reranker 的共性限制。

Batch 8 的 A/B 消融（``hybrid_rrf`` vs ``hybrid_rrf_rerank``）会用
Recall@k / MRR 把这部分损失量化出来 —— 如果重排对含图问题的提升明显小于
纯文本问题，那就是这个限制在起作用。

两个厂商的差异只在 URL 与 JSON 字段名上，传输与分片逻辑完全共用。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

import httpx

from app.config import RerankSettings
from app.core.errors import (
    ProviderBadRequestError,
    ProviderResponseError,
)
from app.providers.base import (
    RateLimiter,
    RerankHit,
    RerankProvider,
    RerankResult,
    run_with_retry,
)
from app.providers.http_util import build_async_client, post_json, truncate_body

__all__ = ["DashScopeRerank", "HttpRerankProvider", "SiliconFlowRerank"]

logger = logging.getLogger(__name__)


class HttpRerankProvider(RerankProvider):
    """分片 → 并发请求 → 合并 → 全局取 top_n。

    分片的必要性：rerank 接口对单次请求的文档条数有上限（``RERANK_MAX_DOCS_PER_REQUEST``），
    而融合之后候选可能有上百条。

    **合并必须在全局做**：不能每片各取 top_n 再拼起来 —— 那样会漏掉
    「在这一片里排第 5、但在全局能排进前 3」的文档。所以每片都向服务端要全量结果
    （``top_n = 片长``），本地合并后再统一截断。
    """

    def __init__(self, settings: RerankSettings) -> None:
        self.name = settings.resolved.provider
        self.model = settings.resolved_model
        self._url = settings.endpoint
        self._max_docs = max(1, settings.max_docs_per_request)
        self._max_chars = max(64, settings.max_chars_per_doc)
        self._max_attempts = max(1, settings.max_retries + 1)
        self._image_doc_template = settings.image_doc_template
        self._limiter = RateLimiter(rpm=settings.rpm_limit, name=f"{self.name}:rerank")
        self._client: httpx.AsyncClient = build_async_client(
            timeout_s=settings.timeout_s,
            api_key=settings.resolved.api_key.get_secret_value(),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- 子类实现 ------------------------------------------------------------------

    def build_payload(
        self, query: str, documents: list[str], top_n: int
    ) -> dict[str, Any]:
        raise NotImplementedError

    def parse_response(
        self, body: dict[str, Any], *, expected: int
    ) -> list[RerankHit]:
        raise NotImplementedError

    # -- 主流程 --------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> RerankResult:
        if not documents:
            return RerankResult(model=self.model)

        prepared = [self._prepare_document(text) for text in documents]
        shards = [
            prepared[start : start + self._max_docs]
            for start in range(0, len(prepared), self._max_docs)
        ]

        started = time.perf_counter()
        responses = await asyncio.gather(
            *(
                self._rerank_shard(query, shard, offset)
                for offset, shard in _offsets(shards)
            )
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        merged: list[RerankHit] = []
        attempts = 1
        for hits, shard_attempts in responses:
            merged.extend(hits)
            attempts = max(attempts, shard_attempts)

        # 全局排序。同分时用 index 兜底，保证结果稳定可复现 ——
        # 评测集跑两次得到不同排序会让人怀疑实现有问题。
        merged.sort(key=lambda hit: (-hit.score, hit.index))
        limit = top_n if top_n is not None and top_n > 0 else len(merged)
        return RerankResult(
            hits=merged[:limit], model=self.model, latency_ms=latency_ms, attempts=attempts
        )

    def _prepare_document(self, text: str) -> str:
        """按 ``RERANK_MAX_CHARS_PER_DOC`` 截断。

        超长文档不会提升重排质量，只会拖慢请求并可能触发服务端的长度限制。
        """
        stripped = text.strip()
        if len(stripped) <= self._max_chars:
            return stripped
        return stripped[: self._max_chars]

    async def _rerank_shard(
        self, query: str, shard: list[str], offset: int
    ) -> tuple[list[RerankHit], int]:
        payload = self.build_payload(query, shard, top_n=len(shard))
        result, attempts = await run_with_retry(
            f"rerank[{offset}:{offset + len(shard)}]",
            lambda: post_json(
                self._client, self._url, payload, provider=self.name, model=self.model
            ),
            max_attempts=self._max_attempts,
            limiter=self._limiter,
            provider=self.name,
            model=self.model,
        )
        local_hits = self.parse_response(result, expected=len(shard))

        # 把分片内的局部下标平移成原始序列的全局下标
        return (
            [RerankHit(index=offset + hit.index, score=hit.score) for hit in local_hits],
            attempts,
        )

    @property
    def image_doc_template(self) -> str:
        """图片候选参与重排时使用的文本模板。

        重排器只吃文本，所以模板里的 ``{heading}`` / ``{caption}``
        就是图片对重排器而言的**全部信息**。
        """
        return self._image_doc_template


# --------------------------------------------------------------------------------------
# DashScope
# --------------------------------------------------------------------------------------


class DashScopeRerank(HttpRerankProvider):
    """``gte-rerank-v2`` 原生端点。

    请求体形态与 OpenAI 不同，是 DashScope 自己的嵌套结构::

        {"model": ..., "input": {"query": ..., "documents": [...]},
         "parameters": {"return_documents": false, "top_n": N}}

    响应取 ``output.results[*].{index, relevance_score}``。
    """

    def build_payload(
        self, query: str, documents: list[str], top_n: int
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {
                # 文档本来就在调用方手里，让服务端再回传一遍纯属浪费带宽，
                # 而且顺序一旦和解码方式有关就会引入不一致。
                "return_documents": False,
                "top_n": top_n,
            },
        }

    def parse_response(self, body: dict[str, Any], *, expected: int) -> list[RerankHit]:
        _raise_on_native_error(body, provider=self.name, model=self.model)
        output = body.get("output")
        if not isinstance(output, dict):
            raise ProviderResponseError(
                f"{self.name} 的 rerank 响应里没有 output 对象",
                provider=self.name,
                model=self.model,
                detail={"keys": sorted(body.keys())},
            )
        return _parse_results(
            output.get("results"), expected=expected, provider=self.name, model=self.model
        )


# --------------------------------------------------------------------------------------
# SiliconFlow
# --------------------------------------------------------------------------------------


class SiliconFlowRerank(HttpRerankProvider):
    """``BAAI/bge-reranker-v2-m3``，``POST /rerank``，扁平请求体::

        {"model": ..., "query": ..., "documents": [...], "top_n": N}

    响应取顶层 ``results[*].{index, relevance_score}``。
    """

    def build_payload(
        self, query: str, documents: list[str], top_n: int
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }

    def parse_response(self, body: dict[str, Any], *, expected: int) -> list[RerankHit]:
        _raise_on_flat_error(body, provider=self.name, model=self.model)
        return _parse_results(
            body.get("results"), expected=expected, provider=self.name, model=self.model
        )


# --------------------------------------------------------------------------------------
# 解析工具
# --------------------------------------------------------------------------------------


def _offsets(shards: list[list[str]]) -> list[tuple[int, list[str]]]:
    out: list[tuple[int, list[str]]] = []
    offset = 0
    for shard in shards:
        out.append((offset, shard))
        offset += len(shard)
    return out


def _parse_results(
    results: Any, *, expected: int, provider: str, model: str
) -> list[RerankHit]:
    if not isinstance(results, list):
        raise ProviderResponseError(
            f"{provider} 的 rerank 响应里没有 results 数组",
            provider=provider,
            model=model,
        )
    if len(results) > expected:
        raise ProviderResponseError(
            f"{provider} 返回的重排条数超过请求条数（请求 {expected}，返回 {len(results)}）",
            provider=provider,
            model=model,
        )

    hits: list[RerankHit] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if not isinstance(index, int) or score is None:
            raise ProviderResponseError(
                f"{provider} 的 rerank 结果缺少 index 或 relevance_score",
                provider=provider,
                model=model,
                detail={"keys": sorted(item.keys())},
            )
        hits.append(RerankHit(index=index, score=float(score)))
    return hits


def _raise_on_native_error(body: dict[str, Any], *, provider: str, model: str) -> None:
    """DashScope 偶尔以 HTTP 200 + ``{"code":..., "message":...}`` 报错。"""
    if body.get("output") is not None:
        return
    code = str(body.get("code") or "")
    message = str(body.get("message") or "")
    if not code and not message:
        return
    raise ProviderBadRequestError(
        f"{provider} rerank 返回错误 {code}：{message}".strip(),
        provider=provider,
        model=model,
        detail={"code": code, "body": truncate_body(str(body))},
    )


def _raise_on_flat_error(body: dict[str, Any], *, provider: str, model: str) -> None:
    error = body.get("error")
    if not error or "results" in body:
        return
    message = error.get("message") if isinstance(error, dict) else str(error)
    raise ProviderBadRequestError(
        f"{provider} rerank 返回错误：{message}",
        provider=provider,
        model=model,
        detail={"body": truncate_body(str(body))},
    )
