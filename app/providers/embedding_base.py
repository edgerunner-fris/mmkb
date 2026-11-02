"""HTTP embedding provider 的公共基类与响应解析。

DashScope（兼容模式）与 SiliconFlow 的文本 embedding 都遵循 OpenAI 的
``POST /embeddings`` 协议，响应形状完全一致：``{"data":[{"embedding":[...],"index":n}]}``。
差异只在 base_url、模型名，以及**要不要发 ``dimensions`` 参数**：

* DashScope ``text-embedding-v4`` 有 2048/1536/1024/768/512/256 多档维度，
  必须显式指定，否则拿到默认值会与配置悄悄对不上；
* SiliconFlow ``BAAI/bge-m3`` 维度固定 1024，发 ``dimensions`` 反而可能被拒。

这个开关由子类的 ``_send_dimensions`` 控制，一个布尔值说清差异。
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Sequence
from typing import Any

import httpx

from app.core.errors import (
    DimensionMismatchError,
    ProviderBadRequestError,
    ProviderResponseError,
)
from app.providers.base import (
    EmbeddingProvider,
    EmbeddingResult,
    PreparedImage,
    RateLimiter,
    run_with_retry,
)
from app.providers.http_util import build_async_client, post_json, truncate_body

__all__ = [
    "HttpEmbeddingBase",
    "l2_normalize",
    "order_by_index",
    "parse_openai_embeddings",
    "raise_on_openai_error",
    "with_offsets",
]

logger = logging.getLogger(__name__)


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """L2 归一化。

    Chroma 的 cosine 空间本身对模长不敏感，但归一化之后向量点积就等于余弦相似度，
    分数可解释、可跨集合比较，也方便在 debug_trace 里直接看数。代价可以忽略。
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return list(vector)
    return [value / norm for value in vector]


def with_offsets(shards: Sequence[Sequence[Any]]) -> list[tuple[int, Sequence[Any]]]:
    """给分片标上起始下标，仅用于日志里定位是第几批出的问题。"""
    out: list[tuple[int, Sequence[Any]]] = []
    offset = 0
    for shard in shards:
        out.append((offset, shard))
        offset += len(shard)
    return out


def order_by_index(
    items: list[dict[str, Any]], expected: int, *, provider: str, model: str
) -> list[list[float]]:
    """按响应里的 ``index`` 还原顺序。

    批量接口不保证返回顺序，但每条都带 ``index``。若某条缺 ``index``，
    退回按数组原顺序 —— 在「一次提交一条」的部署下这是唯一可行的解释。
    """
    if len(items) != expected:
        raise ProviderResponseError(
            f"{provider} 返回的条目数不符（期望 {expected}，实际 {len(items)}）",
            provider=provider,
            model=model,
            detail={"preview": truncate_body(str(items))},
        )

    indexes = [item.get("index") for item in items]
    ordered = (
        sorted(items, key=lambda item: int(item["index"]))
        if all(isinstance(value, int) for value in indexes)
        else items
    )

    vectors: list[list[float]] = []
    for item in ordered:
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise ProviderResponseError(
                f"{provider} 返回的条目里没有 embedding 数组",
                provider=provider,
                model=model,
                detail={"keys": sorted(item.keys())},
            )
        vectors.append([float(value) for value in embedding])
    return vectors


def parse_openai_embeddings(
    body: dict[str, Any], *, provider: str, model: str, expected: int
) -> list[list[float]]:
    """解析 OpenAI 兼容的 ``/embeddings`` 响应。"""
    data = body.get("data")
    if not isinstance(data, list):
        raise ProviderResponseError(
            f"{provider} 的 /embeddings 响应里没有 data 数组",
            provider=provider,
            model=model,
            detail={"keys": sorted(body.keys())},
        )
    return order_by_index(data, expected, provider=provider, model=model)


def raise_on_openai_error(body: dict[str, Any], *, provider: str, model: str) -> None:
    """OpenAI 兼容端点偶尔会以 HTTP 200 + ``{"error": ...}`` 的形式报错。"""
    error = body.get("error")
    if not error or "data" in body:
        return
    message = error.get("message") if isinstance(error, dict) else str(error)
    raise ProviderBadRequestError(
        f"{provider} 返回了错误：{message}",
        provider=provider,
        model=model,
        detail={"body": truncate_body(str(body))},
    )


class HttpEmbeddingBase(EmbeddingProvider):
    """HTTP 传输、限流、维度断言的公共部分。

    维度断言放在基类里而不是各子类里，因为它是**唯一一处必须无条件的检查**：
    配置写 1024、服务端返回 2560 这种事故一旦放过去，整个 collection 的相似度
    计算都会是错的，而且要到检索结果明显不对劲时才会被发现。
    """

    #: 是否在请求体里发 ``dimensions``。维度固定的模型必须为 ``False``。
    _send_dimensions = True
    #: 单条文本的字符上限，超过则保护性截断
    _truncate_chars = 6000

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        model: str,
        dim: int,
        dim_scope: str,
        batch_size: int,
        max_concurrency: int,
        rpm_limit: int,
        max_retries: int,
        timeout_s: float,
        normalize: bool,
        base_url: str,
    ) -> None:
        self.name = name
        self.model = model
        self.dim = dim
        self._dim_scope = dim_scope
        self._batch_size = max(1, batch_size)
        self._normalize = normalize
        self._max_attempts = max(1, max_retries + 1)
        self._base_url = base_url.rstrip("/")
        self._limiter = RateLimiter(
            max_concurrency=max_concurrency, rpm=rpm_limit, name=f"{name}:embedding"
        )
        self._client: httpx.AsyncClient = build_async_client(
            timeout_s=timeout_s, api_key=api_key
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- 传输 ----------------------------------------------------------------------

    async def post(
        self, url: str, payload: dict[str, Any], op: str
    ) -> tuple[dict[str, Any], int]:
        return await run_with_retry(
            op,
            lambda: post_json(self._client, url, payload, provider=self.name, model=self.model),
            max_attempts=self._max_attempts,
            limiter=self._limiter,
            provider=self.name,
            model=self.model,
        )

    # -- 结果整理 ------------------------------------------------------------------

    def finalize(
        self, vectors: list[list[float]], *, count: int, latency_ms: int, attempts: int
    ) -> EmbeddingResult:
        if len(vectors) != count:
            raise ProviderResponseError(
                f"{self.name} 返回的向量条数与请求不一致（请求 {count} 条，返回 {len(vectors)} 条）",
                provider=self.name,
                model=self.model,
                detail={"batch_size": self._batch_size},
            )
        for position, vector in enumerate(vectors):
            if not vector:
                raise ProviderResponseError(
                    f"{self.name} 返回了空向量（第 {position} 条）",
                    provider=self.name,
                    model=self.model,
                )
            if len(vector) != self.dim:
                raise DimensionMismatchError(
                    scope=self._dim_scope,
                    expected=self.dim,
                    actual=len(vector),
                    provider=self.name,
                    model=self.model,
                )
        if self._normalize:
            vectors = [l2_normalize(vector) for vector in vectors]
        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            dim=self.dim,
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def normalize_inputs(self, texts: Sequence[str]) -> list[str]:
        """按字符数保护性截断。

        只是防止单条超长文本被服务端直接拒绝；真正的 token 预算由切分阶段
        （``CHUNK_*``）负责，这里不重复做 token 级计算。
        """
        limit = self._truncate_chars
        return [text[:limit] if len(text) > limit else text for text in texts]

    async def _gather_shards(self, coros: list[Any]) -> tuple[list[list[float]], int, int]:
        """并发跑所有分片，返回 ``(向量列表, 实际最大尝试次数, 耗时毫秒)``。"""
        started = time.perf_counter()
        results = await asyncio.gather(*coros)
        latency_ms = int((time.perf_counter() - started) * 1000)

        vectors: list[list[float]] = []
        attempts = 1
        for part, part_attempts in results:
            vectors.extend(part)
            attempts = max(attempts, part_attempts)
        return vectors, attempts, latency_ms

    # -- 默认不支持的操作 ----------------------------------------------------------

    async def embed_images(self, images: Sequence[bytes | PreparedImage]) -> EmbeddingResult:
        raise self._unsupported(
            "embed_images",
            "文本 embedding 模型不能生成图片向量。图片向量必须由 "
            "IMAGE_EMBED_PROVIDER（qwen3-vl-embedding）生成后写入 {kb_id}__image。",
        )

    async def embed_fused(self, texts: Sequence[str]) -> EmbeddingResult:
        raise self._unsupported(
            "embed_fused",
            "以文搜图的 query 必须走多模态融合空间。用文本 embedding 模型的 query "
            "向量去查图像库是本项目明令禁止的 —— 两者不是同一个向量空间，"
            "直接算余弦相似度是错的。",
        )
