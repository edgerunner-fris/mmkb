"""模型 provider 抽象层。

业务代码只依赖本包的 ABC 与 ``ProviderBundle``，永远不 import 厂商 SDK：

.. code-block:: python

    from app.providers import ChatMessage, get_providers

    bundle = get_providers()
    result = await bundle.llm.chat([ChatMessage.user("你好")])

需要具体 provider 时走 :func:`app.providers.factory.build_*`，
不要在业务模块里直接 new。
"""

from __future__ import annotations

from app.providers.base import (
    ChatDelta,
    ChatMessage,
    ChatResult,
    EmbeddingProvider,
    EmbeddingResult,
    ImagePart,
    LLMProvider,
    PreparedImage,
    RerankHit,
    RerankProvider,
    RerankResult,
    TextPart,
    chunked,
    prepare_image_for_api,
    retry_policy,
)
from app.providers.factory import (
    CheckResult,
    ProviderBundle,
    VerifyReport,
    build_image_embedding,
    build_llm,
    build_rerank,
    build_text_embedding,
    close_providers,
    get_providers,
    provider_matrix,
    verify_providers,
)

__all__ = [
    "ChatDelta",
    "ChatMessage",
    "ChatResult",
    "CheckResult",
    "EmbeddingProvider",
    "EmbeddingResult",
    "ImagePart",
    "LLMProvider",
    "PreparedImage",
    "ProviderBundle",
    "RerankHit",
    "RerankProvider",
    "RerankResult",
    "TextPart",
    "VerifyReport",
    "build_image_embedding",
    "build_llm",
    "build_rerank",
    "build_text_embedding",
    "chunked",
    "close_providers",
    "get_providers",
    "prepare_image_for_api",
    "provider_matrix",
    "retry_policy",
    "verify_providers",
]
