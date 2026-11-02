"""SiliconFlow 文本向量适配（``BAAI/bge-m3``）。

作为 DashScope ``text-embedding-v4`` 的**替代文本 embedding**，
用 ``TEXT_EMBED_PROVIDER=siliconflow`` 切换。

与 DashScope 的两点差异：

1. endpoint 是标准 OpenAI 兼容的 ``{base_url}/embeddings``；
2. **不发 ``dimensions``** —— ``bge-m3`` 维度固定 1024，多数兼容实现遇到
   不认识的参数会直接 400，而不是忽略。配置里的 ``TEXT_EMBED_DIM`` 仍需填对，
   返回维度不符会被 :meth:`~app.providers.embedding_base.HttpEmbeddingBase.finalize` 拦下。

⚠️ ``bge-m3`` 与 ``text-embedding-v4`` 是**两个不同的向量空间**。
   换 provider 等于换空间，已有 ``{kb_id}__text`` 全部作废，必须重建索引。
   本项目不做「混合两家的向量」这种事。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.config import TextEmbeddingSettings
from app.core.errors import ProviderUnsupportedOperationError
from app.providers.base import EmbeddingResult, PreparedImage, chunked
from app.providers.embedding_base import (
    HttpEmbeddingBase,
    parse_openai_embeddings,
    raise_on_openai_error,
    with_offsets,
)

__all__ = ["SiliconFlowTextEmbedding"]

logger = logging.getLogger(__name__)


class SiliconFlowTextEmbedding(HttpEmbeddingBase):
    """``BAAI/bge-m3``。只服务文本空间。"""

    space = "text"
    _send_dimensions = False

    def __init__(self, settings: TextEmbeddingSettings) -> None:
        super().__init__(
            name="siliconflow",
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
        payload: dict[str, object] = {
            "model": self.model,
            "input": list(shard),
            "encoding_format": "float",
        }
        body, attempts = await self.post(
            self._url, payload, op=f"embed_texts[{offset}:{offset + len(shard)}]"
        )
        raise_on_openai_error(body, provider=self.name, model=self.model)
        vectors = parse_openai_embeddings(
            body, provider=self.name, model=self.model, expected=len(shard)
        )
        return vectors, attempts

    async def embed_images(self, images: Sequence[bytes | PreparedImage]) -> EmbeddingResult:
        raise ProviderUnsupportedOperationError(
            "siliconflow(BAAI/bge-m3，文本空间)不支持 embed_images。"
            "图片向量必须由 IMAGE_EMBED_PROVIDER（qwen3-vl-embedding）生成，"
            "且当前只有 dashscope 实现了该多模态端点。",
            provider=self.name,
            model=self.model,
            detail={"operation": "embed_images", "space": self.space},
        )
