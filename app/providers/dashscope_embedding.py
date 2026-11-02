"""DashScope 向量化适配。

两个**互不相通**的向量空间，因此是两个 provider 实例
================================================================

.. code-block:: text

    文本空间  DashScopeTextEmbedding   model=text-embedding-v4    dim=1024
              → 写入 {kb_id}__text     文档块向量 + 图片 caption 向量

    图像空间  DashScopeImageEmbedding  model=qwen3-vl-embedding   dim=2560
              → 写入 {kb_id}__image    qwen3-vl-embedding + enable_fusion=true

★ 铁律（本项目的硬约束，违反它得到的相似度没有任何意义）
----------------------------------------------------------------

* **铁律 1 文本查文本**：query 文本用 ``text-embedding-v4`` 编码 →
  检索 ``{kb_id}__text``。走 :meth:`DashScopeTextEmbedding.embed_texts`。
* **铁律 2 以文搜图**：query 文本必须送进 ``qwen3-vl-embedding`` 多模态端点
  （text-only 输入），且 ``enable_fusion`` 与图片入库时**完全一致**，
  得到 query 向量后再检索 ``{kb_id}__image``。
  走 :meth:`DashScopeImageEmbedding.embed_fused`。
* **铁律 3 禁止混用**：不得用 ``text-embedding-v4`` 的向量查图像库，
  也不得用 ``qwen3-vl-embedding`` 的向量查纯文本库。
  两者不是同一个向量空间，直接算余弦相似度是错的。

铁律 3 由类型系统兜底：两个类各自只实现自己那一半方法，
调错方向会抛 :class:`ProviderUnsupportedOperationError`。

⚠️ 为什么多模态端点必须走原生 HTTP
----------------------------------------------------------------
DashScope 的 **OpenAI 兼容模式只支持文本 embedding**。多模态 embedding
（``/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding``）
没有 OpenAI 协议形态，用 ``openai`` SDK 去调它一定会失败 ——
这是本项目已经踩过的坑，所以这里直接用 ``httpx`` 发原生 JSON。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from app.config import ImageEmbeddingSettings, TextEmbeddingSettings
from app.core.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderUnsupportedOperationError,
)
from app.providers.base import (
    EmbeddingResult,
    PreparedImage,
    chunked,
    prepare_image_for_api,
)
from app.providers.embedding_base import (
    HttpEmbeddingBase,
    order_by_index,
    parse_openai_embeddings,
    raise_on_openai_error,
    with_offsets,
)
from app.providers.http_util import truncate_body

__all__ = ["DashScopeImageEmbedding", "DashScopeTextEmbedding"]

logger = logging.getLogger(__name__)

#: multimodal-embedding 原生端点 contents 数组里各类条目的键名。
#: 抽成常量：万一官方调整字段名，只有一个地方要改。
_CONTENT_KEY_IMAGE = "image"
_CONTENT_KEY_TEXT = "text"

#: DashScope 原生错误码 → 处理类别。没匹配上的按 400 处理。
_NATIVE_ERROR_KIND: dict[str, str] = {
    "InvalidApiKey": "auth",
    "ModelNotExist": "bad_request",
    "InvalidParameter": "bad_request",
    "InvalidFile": "bad_request",
    "Throttling": "rate_limit",
    "Throttling.RateQuota": "rate_limit",
    "Throttling.AllocationQuota": "rate_limit",
}


# --------------------------------------------------------------------------------------
# 文本空间
# --------------------------------------------------------------------------------------


class DashScopeTextEmbedding(HttpEmbeddingBase):
    """``text-embedding-v4``（OpenAI 兼容模式）。

    负责文本空间的全部职责：
      * 文档块向量（正文 / 代码 / 表格 / 图片 caption）
      * 纯文本 query 向量（**铁律 1**）

    ``embed_images`` / ``embed_fused`` 继承基类的「不支持」实现 ——
    text-embedding-v4 产出的向量属于文本空间，写进图像库或用来查图像库都是错的。
    """

    space = "text"
    _send_dimensions = True

    def __init__(self, settings: TextEmbeddingSettings) -> None:
        super().__init__(
            name="dashscope",
            api_key=settings.resolved.api_key.get_secret_value(),
            model=settings.resolved_model,
            dim=settings.dim,
            dim_scope="TEXT_EMBED_DIM",
            batch_size=settings.batch_size,
            max_concurrency=settings.max_concurrency,
            rpm_limit=settings.rpm_limit,
            max_retries=settings.max_retries,
            timeout_s=settings.timeout_s,
            normalize=settings.normalize,
            base_url=settings.resolved_base_url,
        )
        self._truncate_chars = settings.truncate_chars
        self._url = f"{self._base_url}/embeddings"

    @property
    def supports_text(self) -> bool:
        return True

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(model=self.model, dim=self.dim)

        prepared = self.normalize_inputs(texts)
        vectors, attempts, latency_ms = await self._gather_shards(
            [
                self._embed_shard(shard, offset)
                for offset, shard in with_offsets(list(chunked(prepared, self._batch_size)))
            ]
        )
        return self.finalize(
            vectors, count=len(prepared), latency_ms=latency_ms, attempts=attempts
        )

    async def _embed_shard(
        self, shard: Sequence[str], offset: int
    ) -> tuple[list[list[float]], int]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": list(shard),
            "encoding_format": "float",
            # 维度必须显式指定：text-embedding-v4 有 2048/1536/1024/768/512/256
            # 多个档位，不指定就拿默认值，会与 .env 里配的维度悄悄对不上。
            "dimensions": self.dim,
        }
        body, attempts = await self.post(
            self._url, payload, op=f"embed_texts[{offset}:{offset + len(shard)}]"
        )
        raise_on_openai_error(body, provider=self.name, model=self.model)
        vectors = parse_openai_embeddings(
            body, provider=self.name, model=self.model, expected=len(shard)
        )
        return vectors, attempts


# --------------------------------------------------------------------------------------
# 图像（多模态融合）空间
# --------------------------------------------------------------------------------------


class DashScopeImageEmbedding(HttpEmbeddingBase):
    """``qwen3-vl-embedding``（**原生 HTTP 端点**）。

    ``enable_fusion=True`` 时图片与文本被映射到同一个融合空间，因此可以：
      * 用图片向量检索图片（以图搜图）
      * 用文本向量检索图片（以文搜图，**铁律 2**）

    ⚠️ 入库与查询的 ``enable_fusion`` 必须一模一样。改这个开关等于换了一个
       向量空间，已有的 ``{kb_id}__image`` 全部作废，必须重建索引。
    """

    space = "image"
    _send_dimensions = False  # 走 parameters.dimension，不是 OpenAI 的 dimensions

    def __init__(self, settings: ImageEmbeddingSettings) -> None:
        super().__init__(
            name="dashscope",
            api_key=settings.resolved.api_key.get_secret_value(),
            model=settings.resolved.model,
            dim=settings.dim,
            dim_scope="IMAGE_EMBED_DIM",
            batch_size=settings.batch_size,
            max_concurrency=settings.max_concurrency,
            rpm_limit=settings.rpm_limit,
            max_retries=settings.max_retries,
            timeout_s=settings.timeout_s,
            normalize=settings.normalize,
            base_url=settings.resolved.base_url,
        )
        self._enable_fusion = settings.enable_fusion
        self._max_image_side = settings.max_image_side
        self._url = settings.endpoint
        if self._enable_fusion and self._batch_size > 1:
            logger.warning(
                "IMAGE_EMBED_ENABLE_FUSION=true 时，multimodal 端点会把整批 contents "
                "融合成 1 个向量；已强制逐条请求。请把 IMAGE_EMBED_BATCH_SIZE 设为 1 "
                "以消除本警告（当前配置为 %s）",
                self._batch_size,
            )

    @property
    def supports_image(self) -> bool:
        return True

    @property
    def supports_fused(self) -> bool:
        return True

    @property
    def enable_fusion(self) -> bool:
        return self._enable_fusion

    def prepare(self, raw: bytes) -> PreparedImage:
        """把原图转成可上行、且能与 VLM 共用的 :class:`PreparedImage`。"""
        return prepare_image_for_api(raw, max_side=self._max_image_side)

    async def embed_images(self, images: Sequence[bytes | PreparedImage]) -> EmbeddingResult:
        """图片入库。与 :meth:`embed_fused` 共用同一套 ``parameters``。"""
        if not images:
            return EmbeddingResult(model=self.model, dim=self.dim)
        prepared = [
            item if isinstance(item, PreparedImage) else self.prepare(item) for item in images
        ]
        return await self._embed_contents(
            [{_CONTENT_KEY_IMAGE: item.data_uri} for item in prepared],
            count=len(prepared),
            op="embed_images",
        )

    async def embed_fused(self, texts: Sequence[str]) -> EmbeddingResult:
        """**铁律 2：以文搜图的 query 必须走多模态融合空间，不能用 text-embedding-v4 编码。**

        这里发出的 ``enable_fusion`` 与 :meth:`embed_images` 取自**同一个字段**
        （``self._enable_fusion``），因此 query 向量与库里的图片向量落在同一空间。
        任何「文本走文本模型、图片走多模态模型」的偷懒写法都会让以文搜图失效，
        而且不会报错 —— 只会静默返回一堆无关图片，这正是最难排查的失败模式。
        """
        if not texts:
            return EmbeddingResult(model=self.model, dim=self.dim)
        return await self._embed_contents(
            [{_CONTENT_KEY_TEXT: text} for text in texts],
            count=len(texts),
            op="embed_fused",
        )

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        raise ProviderUnsupportedOperationError(
            "dashscope(qwen3-vl-embedding，多模态空间)不支持 embed_texts。"
            "多模态空间的向量不能写进 {kb_id}__text；文本向量请用 "
            "TEXT_EMBED_PROVIDER（text-embedding-v4）。",
            provider=self.name,
            model=self.model,
            detail={"operation": "embed_texts", "space": self.space},
        )

    # -- 内部 ----------------------------------------------------------------------

    @property
    def _effective_batch_size(self) -> int:
        """单次请求实际能放的 contents 条数。

        ⚠️ ``enable_fusion=true`` 时，端点会把一次请求里的**所有** contents
        融合成一个向量（实测 4 条输入只返回 1 条 embedding）。若继续按
        ``IMAGE_EMBED_BATCH_SIZE`` 批量发送，轻则触发「条目数不符」，
        重则把多张图/多段文本共用一个向量写进库 —— 检索会静默失效。
        因此融合模式下必须逐条请求；只有 ``enable_fusion=false`` 才允许批量。
        """
        return 1 if self._enable_fusion else self._batch_size

    async def _embed_contents(
        self, contents: list[dict[str, str]], *, count: int, op: str
    ) -> EmbeddingResult:
        shard_size = self._effective_batch_size
        vectors, attempts, latency_ms = await self._gather_shards(
            [
                self._embed_shard(shard, offset, op=op)
                for offset, shard in with_offsets(list(chunked(contents, shard_size)))
            ]
        )
        return self.finalize(vectors, count=count, latency_ms=latency_ms, attempts=attempts)

    async def _embed_shard(
        self, shard: Sequence[dict[str, str]], offset: int, *, op: str
    ) -> tuple[list[list[float]], int]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": {"contents": list(shard)},
            "parameters": {
                # 维度必须与 .env 的 IMAGE_EMBED_DIM 一致；
                # 不一致会在 finalize() 里被 DimensionMismatchError 拦下。
                "dimension": self.dim,
                # ⚠️ 这个开关决定「图片与文本是否落在同一空间」。
                #    入库与查询必须取同一个值，否则以文搜图会静默失效 ——
                #    不报错，只是召回结果全是噪声。
                "enable_fusion": self._enable_fusion,
            },
        }
        body, attempts = await self.post(
            self._url, payload, op=f"{op}[{offset}:{offset + len(shard)}]"
        )
        raise_on_native_error(body, provider=self.name, model=self.model)
        vectors = parse_native_embeddings(
            body, provider=self.name, model=self.model, expected=len(shard)
        )
        return vectors, attempts


# --------------------------------------------------------------------------------------
# 原生端点响应解析
# --------------------------------------------------------------------------------------


def parse_native_embeddings(
    body: dict[str, Any], *, provider: str, model: str, expected: int
) -> list[list[float]]:
    """解析 DashScope 原生 multimodal-embedding 响应：``output.embeddings[*].embedding``。"""
    output = body.get("output")
    if not isinstance(output, dict):
        raise ProviderResponseError(
            f"{provider} 的多模态响应里没有 output 对象",
            provider=provider,
            model=model,
            detail={"keys": sorted(body.keys())},
        )
    items = output.get("embeddings")
    if not isinstance(items, list):
        raise ProviderResponseError(
            f"{provider} 的多模态响应里没有 output.embeddings 数组",
            provider=provider,
            model=model,
            detail={"output_keys": sorted(output.keys())},
        )
    return order_by_index(items, expected, provider=provider, model=model)


def raise_on_native_error(body: dict[str, Any], *, provider: str, model: str) -> None:
    """DashScope 原生端点在部分错误下返回 **HTTP 200 + 错误体**。

    这种情况 ``raise_for_status`` 抓不到，必须自己检查。
    判据：没有 ``output``，却有 ``code`` 或 ``message``。
    """
    if body.get("output") is not None:
        return
    code = str(body.get("code") or "")
    message = str(body.get("message") or body.get("msg") or "")
    if not code and not message:
        return

    text = f"{provider} 原生端点返回错误 {code}：{message}".strip()
    detail = {"code": code, "body": truncate_body(str(body))}
    kind = _NATIVE_ERROR_KIND.get(code, "bad_request")
    if kind == "auth":
        raise ProviderAuthError(text, provider=provider, model=model, detail=detail)
    if kind == "rate_limit":
        raise ProviderRateLimitError(text, provider=provider, model=model, detail=detail)
    raise ProviderBadRequestError(text, provider=provider, model=model, detail=detail)
