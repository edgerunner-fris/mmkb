"""provider 层离线测试。

全部用 ``httpx.MockTransport`` 伪造上游，**不发任何真实网络请求**，
所以可以放进 CI（见 Batch 8 的 GitHub Actions）。

重点覆盖四类最容易出错、且错了不会立刻暴露的行为：

1. **向量空间隔离**（铁律 3）—— 调错方向必须抛异常，而不是返回一个
   「维度看起来对、其实来自另一个空间」的向量；
2. **维度断言** —— 配置与服务端不一致时必须炸，不能静默写进向量库；
3. **分片与重试** —— 批量接口的条数限制、瞬时故障的退避重试；
4. **协议细节** —— ``extra_body`` 是否真的进了请求体、原生端点的
   contents 结构、200 状态码下的错误体。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.config import (
    ConfigurationError,
    ImageEmbeddingSettings,
    LLMSettings,
    RerankSettings,
    TextEmbeddingSettings,
)
from app.core.errors import (
    DimensionMismatchError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderUnsupportedOperationError,
    is_retryable,
)
from app.core.schemas import DegradedReason
from app.providers.base import (
    ChatMessage,
    RateLimiter,
    chunked,
    prepare_image_for_api,
    run_with_retry,
)
from app.providers.dashscope_embedding import (
    DashScopeImageEmbedding,
    DashScopeTextEmbedding,
)
from app.providers.deepseek_llm import DeepSeekLLM
from app.providers.embedding_base import l2_normalize, order_by_index
from app.providers.rerank import SiliconFlowRerank
from app.providers.siliconflow_embedding import SiliconFlowTextEmbedding

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------------------
# 脚手架
# --------------------------------------------------------------------------------------


def _fake_client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _install(provider: Any, handler: Handler) -> list[httpx.Request]:
    """把 provider 内部的 httpx 客户端换成 MockTransport 版本，并记录请求。

    直接替换 ``_client`` 是白盒做法，但比在生产代码里为测试开一个
    ``transport`` 注入口要干净 —— 测试可以知道内部结构，生产代码不必为此变形。
    """
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    provider._client = _fake_client(recording)
    return seen


def _embedding_response(count: int, dim: int, *, reverse: bool = False) -> dict[str, Any]:
    indexes = list(range(count))
    if reverse:
        indexes = indexes[::-1]
    return {
        "data": [
            {"index": index, "embedding": [float(index + 1)] * dim} for index in indexes
        ],
        "model": "fake",
        "usage": {"total_tokens": 3},
    }


def _echo_embedding_handler(dim: int, *, reverse: bool = False) -> Handler:
    """按请求里的条数返回对应数量的向量 —— 分片测试的 mock 必须跟着请求走。"""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200, json=_embedding_response(len(body["input"]), dim, reverse=reverse)
        )

    return handler


def _native_response(count: int, dim: int) -> dict[str, Any]:
    return {
        "output": {
            "embeddings": [
                {"index": index, "embedding": [0.5] * dim} for index in range(count)
            ]
        },
        "usage": {"input_tokens": 5},
        "request_id": "req-1",
    }


def _text_settings(**overrides: Any) -> TextEmbeddingSettings:
    base: dict[str, Any] = {
        "provider": "dashscope",
        "api_key": "test-key",
        "dim": 4,
        "batch_size": 2,
        "max_retries": 0,
        "normalize": False,
        "timeout_s": 5.0,
    }
    base.update(overrides)
    return TextEmbeddingSettings(**base)


def _image_settings(**overrides: Any) -> ImageEmbeddingSettings:
    base: dict[str, Any] = {
        "provider": "dashscope",
        "api_key": "test-key",
        "dim": 8,
        "batch_size": 1,
        "max_retries": 0,
        "normalize": False,
        "timeout_s": 5.0,
        "enable_fusion": True,
        "max_image_side": 64,
    }
    base.update(overrides)
    return ImageEmbeddingSettings(**base)


def _png(size: tuple[int, int] = (400, 300), mode: str = "RGB") -> bytes:
    import io

    from PIL import Image

    image = Image.new(mode, size, (200, 30, 30) if mode == "RGB" else (200, 30, 30, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------------------
# 1. 纯函数：分片、归一化、可重试判据
# --------------------------------------------------------------------------------------


def test_chunked_splits_evenly_and_keeps_remainder() -> None:
    assert [list(part) for part in chunked(list(range(5)), 2)] == [[0, 1], [2, 3], [4]]
    assert [list(part) for part in chunked([], 2)] == []
    # size <= 0 表示不分片
    assert [list(part) for part in chunked([1, 2, 3], 0)] == [[1, 2, 3]]


def test_l2_normalize_produces_unit_vector_and_handles_zero() -> None:
    normed = l2_normalize([3.0, 4.0])
    assert normed == pytest.approx([0.6, 0.8])
    assert sum(value * value for value in normed) == pytest.approx(1.0)
    # 零向量不能除出 NaN
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_order_by_index_restores_server_side_shuffle() -> None:
    items = [
        {"index": 2, "embedding": [2.0]},
        {"index": 0, "embedding": [0.0]},
        {"index": 1, "embedding": [1.0]},
    ]
    assert order_by_index(items, 3, provider="p", model="m") == [[0.0], [1.0], [2.0]]


def test_order_by_index_rejects_count_mismatch() -> None:
    from app.core.errors import ProviderResponseError

    with pytest.raises(ProviderResponseError, match="条目数不符"):
        order_by_index([{"index": 0, "embedding": [1.0]}], 3, provider="p", model="m")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ProviderUnavailableError("x"), True),
        (ProviderRateLimitError("x"), True),
        (ProviderBadRequestError("x"), False),
        (ProviderAuthError("x"), False),
        (DimensionMismatchError(scope="s", expected=1, actual=2), False),
        (ValueError("nope"), False),
    ],
)
def test_is_retryable_only_retries_transient_failures(exc: Exception, expected: bool) -> None:
    assert is_retryable(exc) is expected


def test_is_retryable_understands_httpx_status_codes() -> None:
    request = httpx.Request("POST", "https://example.com")
    too_many = httpx.HTTPStatusError(
        "429", request=request, response=httpx.Response(429, request=request)
    )
    bad_request = httpx.HTTPStatusError(
        "400", request=request, response=httpx.Response(400, request=request)
    )
    assert is_retryable(too_many) is True
    assert is_retryable(bad_request) is False


# --------------------------------------------------------------------------------------
# 2. 重试与限流
# --------------------------------------------------------------------------------------


async def test_run_with_retry_retries_transient_then_succeeds() -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderUnavailableError("上游抖动")
        return "ok"

    result, attempts = await run_with_retry(
        "flaky", flaky, max_attempts=5, min_wait=0.001, max_wait=0.005
    )
    assert result == "ok"
    assert attempts == 3
    assert calls["n"] == 3


async def test_run_with_retry_does_not_retry_client_errors() -> None:
    calls = {"n": 0}

    async def bad() -> str:
        calls["n"] += 1
        raise ProviderBadRequestError("参数错了")

    with pytest.raises(ProviderBadRequestError):
        await run_with_retry("bad", bad, max_attempts=5, min_wait=0.001, max_wait=0.005)
    assert calls["n"] == 1, "400 类错误重试没有意义，必须只尝试一次"


async def test_run_with_retry_reraises_after_exhausting_attempts() -> None:
    calls = {"n": 0}

    async def always_down() -> str:
        calls["n"] += 1
        raise ProviderUnavailableError("一直挂")

    with pytest.raises(ProviderUnavailableError):
        await run_with_retry(
            "down", always_down, max_attempts=3, min_wait=0.001, max_wait=0.005
        )
    assert calls["n"] == 3


async def test_rate_limiter_caps_concurrency() -> None:
    import asyncio

    limiter = RateLimiter(max_concurrency=2, name="test")
    live = 0
    peak = 0

    async def worker() -> None:
        nonlocal live, peak
        async with limiter:
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1

    await asyncio.gather(*(worker() for _ in range(8)))
    assert peak <= 2


# --------------------------------------------------------------------------------------
# 3. 向量空间隔离（铁律 3）
# --------------------------------------------------------------------------------------


async def test_text_provider_refuses_to_touch_the_image_space() -> None:
    provider = DashScopeTextEmbedding(_text_settings())

    with pytest.raises(ProviderUnsupportedOperationError, match="embed_fused"):
        await provider.embed_fused(["以文搜图的 query"])
    with pytest.raises(ProviderUnsupportedOperationError, match="embed_images"):
        await provider.embed_images([_png()])


async def test_image_provider_refuses_to_pollute_the_text_space() -> None:
    provider = DashScopeImageEmbedding(_image_settings())

    with pytest.raises(ProviderUnsupportedOperationError, match="embed_texts"):
        await provider.embed_texts(["这段文本不该进多模态空间"])


def test_capability_flags_match_the_iron_rules() -> None:
    text_provider = DashScopeTextEmbedding(_text_settings())
    image_provider = DashScopeImageEmbedding(_image_settings())

    assert (text_provider.supports_text, text_provider.supports_image) == (True, False)
    assert (text_provider.supports_fused, text_provider.space) == (False, "text")
    assert (image_provider.supports_image, image_provider.supports_fused) == (True, True)
    assert (image_provider.supports_text, image_provider.space) == (False, "image")


# --------------------------------------------------------------------------------------
# 4. 文本空间：请求形状、分片、维度断言
# --------------------------------------------------------------------------------------


async def test_text_provider_batches_and_sends_dimensions() -> None:
    provider = DashScopeTextEmbedding(_text_settings(batch_size=2))
    seen = _install(provider, _echo_embedding_handler(4))

    result = await provider.embed_texts(["a", "b", "c", "d", "e"])

    assert len(seen) == 3, "5 条文本按 batch_size=2 应切成 3 批"
    assert len(result.vectors) == 5
    first = json.loads(seen[0].content)
    assert first["model"] == provider.model
    assert first["input"] == ["a", "b"]
    assert first["dimensions"] == 4
    assert seen[0].url.path.endswith("/embeddings")


async def test_text_provider_restores_order_from_server_side_shuffle() -> None:
    provider = DashScopeTextEmbedding(_text_settings(batch_size=3))
    _install(provider, _echo_embedding_handler(4, reverse=True))

    result = await provider.embed_texts(["a", "b", "c"])

    assert [vector[0] for vector in result.vectors] == [1.0, 2.0, 3.0]


async def test_text_provider_raises_on_dimension_mismatch() -> None:
    """配置 4 维、服务端返回 8 维 —— 必须炸，不能静默入库。"""
    provider = DashScopeTextEmbedding(_text_settings(dim=4))
    _install(provider, lambda request: httpx.Response(200, json=_embedding_response(1, 8)))

    with pytest.raises(DimensionMismatchError) as excinfo:
        await provider.embed_texts(["hello"])

    assert excinfo.value.expected == 4
    assert excinfo.value.actual == 8
    assert excinfo.value.scope == "TEXT_EMBED_DIM"


async def test_text_provider_detects_http_200_error_body() -> None:
    provider = DashScopeTextEmbedding(_text_settings())
    _install(
        provider,
        lambda request: httpx.Response(200, json={"error": {"message": "invalid api key"}}),
    )

    with pytest.raises(ProviderBadRequestError, match="invalid api key"):
        await provider.embed_texts(["hello"])


async def test_text_provider_surfaces_reason_when_key_is_model_restricted() -> None:
    """DashScope 把「Key 被限制模型访问」塞进 HTTP 400，原因必须出现在异常里。

    真实报错体::

        {"error": {"message": "Access denied by API-Key restrictions.",
                   "type": "AccessDenied", "code": "AccessDenied"}}

    如果只报「拒绝了请求（400）」，排查方向会被带偏到请求体、维度参数上，
    而真正要改的是 Key 的模型访问范围。
    """
    provider = DashScopeTextEmbedding(_text_settings())
    seen = _install(
        provider,
        lambda request: httpx.Response(
            400,
            json={
                "error": {
                    "message": "Access denied by API-Key restrictions.",
                    "type": "AccessDenied",
                    "code": "AccessDenied",
                }
            },
        ),
    )

    with pytest.raises(ProviderAuthError) as excinfo:
        await provider.embed_texts(["hello"])

    message = excinfo.value.message
    assert "Access denied by API-Key restrictions." in message
    assert "AccessDenied" in message
    assert excinfo.value.detail["status"] == 400
    assert "API-Key restrictions" in excinfo.value.detail["body"]
    assert len(seen) == 1, "鉴权类错误重试没有意义"


async def test_text_provider_normalizes_when_configured() -> None:
    provider = DashScopeTextEmbedding(_text_settings(dim=2, normalize=True))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})

    _install(provider, handler)
    result = await provider.embed_texts(["x"])
    assert result.vectors[0] == pytest.approx([0.6, 0.8])


async def test_text_provider_empty_input_makes_no_request() -> None:
    provider = DashScopeTextEmbedding(_text_settings())
    seen = _install(provider, lambda request: httpx.Response(500))

    result = await provider.embed_texts([])
    assert result.vectors == []
    assert seen == []


async def test_siliconflow_does_not_send_dimensions() -> None:
    """bge-m3 维度固定，发 dimensions 可能被 400 拒绝。"""
    provider = SiliconFlowTextEmbedding(
        _text_settings(provider="siliconflow", batch_size=2, max_retries=0)
    )
    seen = _install(provider, _echo_embedding_handler(4))

    await provider.embed_texts(["a", "b"])

    body = json.loads(seen[0].content)
    assert "dimensions" not in body
    assert provider.model == "BAAI/bge-m3"


# --------------------------------------------------------------------------------------
# 5. 图像空间：原生端点、融合参数、错误体
# --------------------------------------------------------------------------------------


async def test_image_provider_hits_the_native_multimodal_endpoint() -> None:
    """融合模式下即使 batch_size>1 也必须逐条请求。

    实测：enable_fusion=true 时，一次请求里的多条 contents 会被融成 1 个向量。
    若按 batch_size=2 发送，会得到「期望 2、实际 1」的响应错误；就算不报错，
    两张图共用一个向量也是错的。
    """
    provider = DashScopeImageEmbedding(_image_settings(batch_size=2))
    seen = _install(
        provider,
        lambda request: httpx.Response(200, json=_native_response(1, 8)),
    )

    await provider.embed_images([_png(), _png(size=(100, 100))])

    assert len(seen) == 2, "融合模式必须每张图单独发一次请求"
    for request in seen:
        assert (
            request.url.path
            == "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
        )
        body = json.loads(request.content)
        assert body["model"] == "qwen3-vl-embedding"
        assert len(body["input"]["contents"]) == 1
        # 每条 content 用 image 键，值是 data URI
        assert "image" in body["input"]["contents"][0]
        assert body["input"]["contents"][0]["image"].startswith("data:image/jpeg;base64,")
        assert body["parameters"] == {"dimension": 8, "enable_fusion": True}


async def test_image_provider_batches_only_when_fusion_is_disabled() -> None:
    """enable_fusion=false 才按 batch_size 合并请求（并与返回条数一一对应）。"""
    provider = DashScopeImageEmbedding(
        _image_settings(enable_fusion=False, batch_size=2)
    )
    seen = _install(
        provider,
        lambda request: httpx.Response(200, json=_native_response(2, 8)),
    )

    result = await provider.embed_images([_png(), _png(size=(100, 100))])

    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert len(body["input"]["contents"]) == 2
    assert body["parameters"]["enable_fusion"] is False
    assert len(result.vectors) == 2


async def test_image_provider_query_uses_the_same_fusion_switch() -> None:
    """铁律 2：以文搜图的 query 与图片入库必须落在同一参数下。"""
    provider = DashScopeImageEmbedding(_image_settings(batch_size=1))
    seen = _install(provider, lambda request: httpx.Response(200, json=_native_response(1, 8)))

    await provider.embed_fused(["这张图说明了什么？"])

    body = json.loads(seen[0].content)
    assert body["input"]["contents"] == [{"text": "这张图说明了什么？"}]
    assert body["parameters"]["enable_fusion"] is True
    assert body["parameters"]["dimension"] == 8


async def test_image_provider_propagates_enable_fusion_false() -> None:
    provider = DashScopeImageEmbedding(_image_settings(enable_fusion=False, batch_size=2))
    seen = _install(provider, lambda request: httpx.Response(200, json=_native_response(1, 8)))

    await provider.embed_fused(["query"])

    assert json.loads(seen[0].content)["parameters"]["enable_fusion"] is False
    assert provider.enable_fusion is False


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("InvalidApiKey", ProviderAuthError),
        ("Throttling.RateQuota", ProviderRateLimitError),
        ("InvalidParameter", ProviderBadRequestError),
    ],
)
async def test_image_provider_maps_native_error_codes(
    code: str, expected: type[Exception]
) -> None:
    """DashScope 原生端点会返回 HTTP 200 + 错误体，raise_for_status 抓不到。"""
    provider = DashScopeImageEmbedding(_image_settings())
    _install(
        provider,
        lambda request: httpx.Response(200, json={"code": code, "message": "出错了"}),
    )

    with pytest.raises(expected):
        await provider.embed_fused(["query"])


async def test_image_provider_dimension_mismatch_mentions_image_scope() -> None:
    provider = DashScopeImageEmbedding(_image_settings(dim=8))
    _install(provider, lambda request: httpx.Response(200, json=_native_response(1, 16)))

    with pytest.raises(DimensionMismatchError) as excinfo:
        await provider.embed_fused(["query"])
    assert excinfo.value.scope == "IMAGE_EMBED_DIM"


# --------------------------------------------------------------------------------------
# 6. 图片预处理
# --------------------------------------------------------------------------------------


def test_prepare_image_resizes_and_emits_jpeg_data_uri() -> None:
    prepared = prepare_image_for_api(_png(size=(4000, 2000)), max_side=512)

    assert prepared.data_uri.startswith("data:image/jpeg;base64,")
    assert max(prepared.width, prepared.height) == 512
    assert prepared.width / prepared.height == pytest.approx(2.0, rel=1e-2)
    assert prepared.byte_size < prepared.source_bytes
    assert prepared.b64 and "," not in prepared.b64


def test_prepare_image_flattens_alpha_onto_white() -> None:
    """带 alpha 的 PNG 必须先合成到白底，否则透明区域会变黑。"""
    prepared = prepare_image_for_api(_png(size=(32, 32), mode="RGBA"), max_side=64)

    import base64
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(prepared.b64))).convert("RGB")
    assert image.mode == "RGB"
    # 中心像素应当偏白（原图 alpha=128 叠在白底上），而不是偏黑
    assert sum(image.getpixel((16, 16))) > 300


def test_prepare_image_rejects_empty_bytes() -> None:
    with pytest.raises(ValueError, match="为空"):
        prepare_image_for_api(b"")


# --------------------------------------------------------------------------------------
# 7. DeepSeek 对话：reasoning_effort 的发送与自动摘除
# --------------------------------------------------------------------------------------


def _chat_completion(content: str = "正常") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _install_openai(provider: DeepSeekLLM, handler: Handler) -> list[httpx.Request]:
    """把 DeepSeek provider 的 AsyncOpenAI 换成走 MockTransport 的实例。"""
    from openai import AsyncOpenAI

    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    provider._client = AsyncOpenAI(
        api_key="test-key",
        base_url=provider._settings.base_url,
        http_client=_fake_client(recording),
        max_retries=0,
    )
    return seen


def _llm_settings(**overrides: Any) -> LLMSettings:
    base: dict[str, Any] = {
        "provider": "deepseek",
        "api_key": "test-key",
        "model": "deepseek-flash",
        "reasoning_effort": "high",
        "send_reasoning_effort": True,
        "max_retries": 0,
        "timeout_s": 5.0,
        "stream_timeout_s": 5.0,
    }
    base.update(overrides)
    return LLMSettings(**base)


async def test_deepseek_sends_reasoning_effort_via_extra_body() -> None:
    provider = DeepSeekLLM(_llm_settings())
    seen = _install_openai(provider, lambda request: httpx.Response(200, json=_chat_completion()))

    result = await provider.chat([ChatMessage.user("你好")])

    body = json.loads(seen[0].content)
    # openai SDK 会把 extra_body 合并进请求体顶层，而不是包一层 extra_body 键
    assert body["reasoning_effort"] == "high"
    assert "extra_body" not in body
    assert result.text == "正常"
    assert result.degraded is False


async def test_deepseek_omits_reasoning_effort_when_disabled() -> None:
    provider = DeepSeekLLM(_llm_settings(reasoning_effort="none"))
    seen = _install_openai(provider, lambda request: httpx.Response(200, json=_chat_completion()))

    await provider.chat([ChatMessage.user("你好")])

    assert "reasoning_effort" not in json.loads(seen[0].content)
    assert provider.model == "deepseek-flash"


async def test_deepseek_strips_reasoning_effort_when_server_rejects_it() -> None:
    """服务端 400 拒绝该参数时，摘掉它重试一次并标记降级。"""
    provider = DeepSeekLLM(_llm_settings())

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "reasoning_effort" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unrecognized request argument supplied: reasoning_effort",
                        "type": "invalid_request_error",
                    }
                },
            )
        return httpx.Response(200, json=_chat_completion("降级后正常"))

    seen = _install_openai(provider, handler)
    result = await provider.chat([ChatMessage.user("你好")])

    assert len(seen) == 2
    assert "reasoning_effort" in json.loads(seen[0].content)
    assert "reasoning_effort" not in json.loads(seen[1].content)
    assert result.text == "降级后正常"
    assert result.degraded is True
    assert str(DegradedReason.REASONING_EFFORT_REJECTED) in result.degraded_reasons


async def test_deepseek_stops_sending_reasoning_effort_after_one_rejection() -> None:
    """被拒过一次之后不再发送 —— 每次请求都先失败一次太浪费。"""
    provider = DeepSeekLLM(_llm_settings())

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "reasoning_effort" in body:
            return httpx.Response(
                400, json={"error": {"message": "reasoning_effort is not supported"}}
            )
        return httpx.Response(200, json=_chat_completion())

    seen = _install_openai(provider, handler)
    await provider.chat([ChatMessage.user("第一次")])
    await provider.chat([ChatMessage.user("第二次")])

    assert len(seen) == 3, "第一次两发（失败+重试），第二次只发一次"
    assert "reasoning_effort" not in json.loads(seen[2].content)
    assert provider.degraded_reasons == [str(DegradedReason.REASONING_EFFORT_REJECTED)]


async def test_deepseek_maps_auth_error_without_retrying() -> None:
    provider = DeepSeekLLM(_llm_settings())
    seen = _install_openai(
        provider, lambda request: httpx.Response(401, json={"error": {"message": "bad key"}})
    )

    with pytest.raises(ProviderAuthError):
        await provider.chat([ChatMessage.user("你好")])
    assert len(seen) == 1


async def test_deepseek_vision_chat_sends_image_parts() -> None:
    provider = DeepSeekLLM(_llm_settings())
    seen = _install_openai(provider, lambda request: httpx.Response(200, json=_chat_completion("一张图")))

    prepared = prepare_image_for_api(_png(), max_side=64)
    result = await provider.vision_chat("描述这张图", [prepared])

    body = json.loads(seen[0].content)
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[1] == {"type": "text", "text": "描述这张图"}
    assert result.text == "一张图"


async def test_deepseek_raises_on_empty_choices() -> None:
    from app.core.errors import ProviderResponseError

    provider = DeepSeekLLM(_llm_settings())
    _install_openai(provider, lambda request: httpx.Response(200, json={"choices": []}))

    with pytest.raises(ProviderResponseError, match="choices"):
        await provider.chat([ChatMessage.user("你好")])


# --------------------------------------------------------------------------------------
# 8. 重排：分片与全局合并
# --------------------------------------------------------------------------------------


def _rerank_settings(**overrides: Any) -> RerankSettings:
    base: dict[str, Any] = {
        "provider": "siliconflow",
        "api_key": "test-key",
        "max_docs_per_request": 2,
        "max_chars_per_doc": 100,
        "max_retries": 0,
        "timeout_s": 5.0,
    }
    base.update(overrides)
    return RerankSettings(**base)


async def test_rerank_merges_shards_globally_not_per_shard() -> None:
    """每片各取 top_n 再拼会漏结果，必须在全局排序后再截断。"""
    provider = SiliconFlowRerank(_rerank_settings(max_docs_per_request=2))
    shard_scores = [
        [0.1, 0.2],   # 文档 0、1
        [0.9, 0.3],   # 文档 2、3
    ]
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        scores = shard_scores[counter["n"]]
        counter["n"] += 1
        assert len(body["documents"]) == 2
        assert body["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": index, "relevance_score": score}
                    for index, score in enumerate(scores)
                ]
            },
        )

    _install(provider, handler)
    result = await provider.rerank("查询", ["d0", "d1", "d2", "d3"], top_n=2)

    assert [hit.index for hit in result.hits] == [2, 3]
    assert [hit.score for hit in result.hits] == pytest.approx([0.9, 0.3])


async def test_rerank_truncates_long_documents() -> None:
    # RERANK_MAX_CHARS_PER_DOC 的下界是 64（见 RerankSettings）
    provider = SiliconFlowRerank(_rerank_settings(max_chars_per_doc=64))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert len(body["documents"][0]) == 64
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    _install(provider, handler)
    await provider.rerank("查询", ["x" * 500])


async def test_rerank_empty_documents_makes_no_request() -> None:
    provider = SiliconFlowRerank(_rerank_settings())
    seen = _install(provider, lambda request: httpx.Response(500))

    result = await provider.rerank("查询", [])
    assert result.hits == []
    assert seen == []


async def test_rerank_dashscope_uses_nested_input_shape() -> None:
    from app.providers.rerank import DashScopeRerank

    provider = DashScopeRerank(_rerank_settings(provider="dashscope"))
    seen = _install(
        provider,
        lambda request: httpx.Response(
            200,
            json={"output": {"results": [{"index": 0, "relevance_score": 0.7}]}},
        ),
    )

    await provider.rerank("查询", ["doc"], top_n=1)

    body = json.loads(seen[0].content)
    assert body["input"]["query"] == "查询"
    assert body["input"]["documents"] == ["doc"]
    assert body["parameters"]["return_documents"] is False
    assert seen[0].url.path.endswith("/api/v1/services/rerank/text-rerank/text-rerank")


def test_rerank_image_template_is_configurable() -> None:
    provider = SiliconFlowRerank(_rerank_settings(image_doc_template="[图] {heading} :: {caption}"))
    rendered = provider.image_doc_template.format(heading="第三章", caption="一张流程图")
    assert rendered == "[图] 第三章 :: 一张流程图"


# --------------------------------------------------------------------------------------
# 9. 装配与自检
# --------------------------------------------------------------------------------------


def _native_echo_handler(dim: int) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        count = len(body["input"]["contents"])
        return httpx.Response(200, json=_native_response(count, dim))

    return handler


def _build_bundle(**overrides: Any) -> Any:
    from app.providers.factory import ProviderBundle

    llm = DeepSeekLLM(_llm_settings())
    _install_openai(llm, lambda request: httpx.Response(200, json=_chat_completion()))

    text = DashScopeTextEmbedding(_text_settings(dim=4, batch_size=8))
    _install(text, _echo_embedding_handler(4))

    image = DashScopeImageEmbedding(_image_settings(dim=8, batch_size=1, max_image_side=64))
    _install(image, _native_echo_handler(8))

    rerank = SiliconFlowRerank(_rerank_settings())
    _install(
        rerank,
        lambda request: httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.91},
                    {"index": 1, "relevance_score": 0.02},
                ]
            },
        ),
    )

    bundle = ProviderBundle(
        llm=llm, text_embedding=text, image_embedding=image, rerank=rerank
    )
    for key, value in overrides.items():
        setattr(bundle, key, value)
    return bundle


async def test_verify_providers_checks_all_four_capabilities() -> None:
    from app.providers import factory

    report = await factory.verify_providers(_build_bundle())

    assert [check.name for check in report.checks] == [
        "LLM 对话",
        "文本向量",
        "多模态向量",
        "重排",
    ]
    assert report.ok
    assert report.failures == []
    # 多模态那一项必须同时验证「以文搜图」和「图片入库」两条路径
    image_check = next(c for c in report.checks if c.name == "多模态向量")
    assert "enable_fusion=True" in image_check.detail


async def test_verify_providers_treats_missing_optional_capability_as_skip() -> None:
    from app.providers import factory

    bundle = _build_bundle(image_embedding=None, rerank=None)
    report = await factory.verify_providers(bundle)

    skipped = [check for check in report.checks if check.skipped]
    assert {check.name for check in skipped} == {"多模态向量", "重排"}
    assert report.ok is True, "可选能力缺失是降级，不是失败"
    assert set(bundle.degraded_reasons) == {
        DegradedReason.IMAGE_EMBED_UNAVAILABLE,
        DegradedReason.RERANK_UNAVAILABLE,
    }
    assert bundle.capabilities == {
        "llm": True,
        "dense_text": True,
        "dense_image": False,
        "rerank": False,
    }


async def test_verify_providers_reports_dimension_mismatch_as_failure() -> None:
    from app.providers import factory

    text = DashScopeTextEmbedding(_text_settings(dim=4, batch_size=8))
    _install(text, _echo_embedding_handler(16))  # 服务端返回 16 维，配置写 4
    bundle = _build_bundle(text_embedding=text)

    report = await factory.verify_providers(bundle)

    assert report.ok is False
    failure = next(c for c in report.failures if c.name == "文本向量")
    assert "dimension_mismatch" in failure.detail


async def test_verify_providers_isolates_one_broken_provider() -> None:
    """一个 provider 挂了，其余项仍要给出结论，而不是整份报告只剩一行错。"""
    from app.providers import factory

    broken = DashScopeTextEmbedding(_text_settings(dim=4, batch_size=8))
    _install(broken, lambda request: httpx.Response(503, text="upstream down"))
    bundle = _build_bundle(text_embedding=broken)

    report = await factory.verify_providers(bundle)

    assert report.ok is False
    assert len(report.checks) == 4
    assert [c.name for c in report.checks if c.ok and not c.skipped] == [
        "LLM 对话",
        "多模态向量",
        "重排",
    ]


async def test_verify_providers_exposes_upstream_body_for_failure_diagnosis() -> None:
    """自检失败必须带上上游原文，而不是只有一句「拒绝了请求（400）」。"""
    from app.providers import factory

    broken = DashScopeTextEmbedding(_text_settings(dim=4, batch_size=8))
    _install(
        broken,
        lambda request: httpx.Response(
            400,
            json={
                "error": {
                    "message": "Access denied by API-Key restrictions.",
                    "code": "AccessDenied",
                }
            },
        ),
    )
    bundle = _build_bundle(text_embedding=broken)

    report = await factory.verify_providers(bundle)

    failure = next(c for c in report.failures if c.name == "文本向量")
    assert "Access denied by API-Key restrictions." in failure.detail
    assert failure.extra["detail"]["status"] == 400
    assert "API-Key restrictions" in failure.extra["detail"]["body"]


async def test_bundle_aclose_survives_none_providers() -> None:
    bundle = _build_bundle(image_embedding=None, rerank=None)
    await bundle.aclose()  # 不应抛异常


def test_optional_providers_degrade_when_keys_are_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺 Key 的可选能力返回 None（降级），必需能力直接报错。"""
    from app.config import get_settings, reset_settings_cache
    from app.providers import factory

    for name in ("LLM_API_KEY", "TEXT_EMBED_API_KEY", "IMAGE_EMBED_API_KEY", "RERANK_API_KEY"):
        monkeypatch.setenv(name, "")
    reset_settings_cache()
    factory.get_providers.cache_clear()
    try:
        settings = get_settings()
        assert factory.build_image_embedding(settings) is None
        assert factory.build_rerank(settings) is None
        with pytest.raises(ConfigurationError) as excinfo:
            factory.get_providers()
    finally:
        reset_settings_cache()
        factory.get_providers.cache_clear()

    problems = "\n".join(excinfo.value.problems)
    # 一次性列全所有缺 Key 的必需 provider，而不是跑一次补一个
    assert "LLM_API_KEY" in problems
    assert "TEXT_EMBED_API_KEY" in problems
