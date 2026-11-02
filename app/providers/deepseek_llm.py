"""DeepSeek 对话模型适配（OpenAI 兼容协议）。

⚠️ 本文件是**适配器**，是项目里少数允许 import 厂商 SDK 的地方之一。
   业务代码永远只 import :mod:`app.providers.base` 里的 ABC。

要点
----
* ``deepseek-flash`` 原生支持图片输入，走标准 ``image_url`` content part，
  所以 ``vision_chat`` 直接用基类默认实现即可。
* ``reasoning_effort`` 是 DeepSeek 的**扩展参数**（取值 ``low/high/max``，
  与 OpenAI 官方的 ``low/medium/high`` 不同），通过 ``extra_body`` 发送，
  不占用 SDK 的类型化参数位。
* 该参数可能被服务端拒绝（模型不支持 / 权限不足），此时**摘掉它重试一次**，
  并在结果上标记 ``degraded``。不能因为一个可选的优化项就让整个问答挂掉。
  拒绝过一次之后，本进程内不再发送它 —— 每次请求都先失败一次太浪费。
* 客户端 ``max_retries=0``：重试完全交给 tenacity，走统一的
  :func:`app.core.errors.is_retryable` 判据。两层重试叠在一起会让延迟成倍放大
  且难以观测。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, TypeVar

import openai
from openai import AsyncOpenAI

from app.config import LLMSettings
from app.core.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.core.schemas import DegradedReason
from app.providers.base import (
    ChatDelta,
    ChatMessage,
    ChatResult,
    LLMProvider,
    RateLimiter,
    run_with_retry,
)

__all__ = ["DeepSeekLLM"]

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: 服务端拒绝 reasoning_effort 时，错误文案里通常包含这些片段之一
_REASONING_REJECT_HINTS = ("reasoning_effort", "reasoning effort", "reasoning_effort is not")

#: 触发「摘掉 reasoning_effort 重试一次」的 HTTP 状态码
_REASONING_REJECT_STATUS = (400, 422)


class DeepSeekLLM(LLMProvider):
    """基于 ``openai.AsyncOpenAI`` 的 DeepSeek 客户端。"""

    name = "deepseek"

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self.model = settings.model
        self._max_attempts = max(1, settings.max_retries + 1)
        self._limiter = RateLimiter(
            max_concurrency=settings.max_concurrency,
            rpm=settings.rpm_limit,
            name=f"{self.name}:llm",
        )
        # 服务端明确拒绝过 reasoning_effort 之后，本进程内不再发送
        self._reasoning_disabled_by_server = False
        self._client = AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout=float(settings.timeout_s),
            max_retries=0,
        )

    # -- 生命周期 ------------------------------------------------------------------

    async def aclose(self) -> None:
        await self._client.close()

    @property
    def degraded_reasons(self) -> list[str]:
        if self._reasoning_disabled_by_server:
            return [str(DegradedReason.REASONING_EFFORT_REJECTED)]
        return []

    # -- 请求构造 ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
        extra: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if stream:
            payload["stream"] = True

        extra_body: dict[str, Any] = {}
        effort = self._settings.effective_reasoning_effort
        if effort is not None and not self._reasoning_disabled_by_server:
            extra_body["reasoning_effort"] = effort
        if extra:
            # extra 是调用方补充 provider 私有参数的逃生通道
            extra_body.update(extra)
        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    @staticmethod
    def _strip_reasoning_effort(payload: dict[str, Any]) -> dict[str, Any]:
        """返回一份不带头 reasoning_effort 的 payload（不改原对象）。"""
        body = dict(payload.get("extra_body") or {})
        body.pop("reasoning_effort", None)
        updated = dict(payload)
        if body:
            updated["extra_body"] = body
        else:
            updated.pop("extra_body", None)
        return updated

    def _is_reasoning_rejection(self, exc: BaseException) -> bool:
        if not isinstance(exc, openai.APIStatusError):
            return False
        if exc.status_code not in _REASONING_REJECT_STATUS:
            return False
        text = str(exc).lower()
        return any(hint in text for hint in _REASONING_REJECT_HINTS)

    def _can_strip_reasoning(self, payload: dict[str, Any]) -> bool:
        if self._reasoning_disabled_by_server:
            return False
        body = payload.get("extra_body") or {}
        return "reasoning_effort" in body

    async def _create_once(
        self, payload: dict[str, Any], *, timeout_s: float | None = None
    ) -> tuple[Any, bool]:
        """发一次请求，返回 ``(响应, 是否因摘掉 reasoning_effort 而降级)``。

        **所有**异常路径都在这里翻译成本项目的异常层级 —— 重试判据
        :func:`app.core.errors.is_retryable` 只认自家异常的 ``retryable`` 标志，
        openai SDK 的异常类型不在它的兜底名单里，漏翻译就等于不重试。
        """
        kwargs: dict[str, Any] = {} if timeout_s is None else {"timeout": float(timeout_s)}
        try:
            return await self._client.chat.completions.create(**kwargs, **payload), False
        except Exception as raw:
            if not (self._can_strip_reasoning(payload) and self._is_reasoning_rejection(raw)):
                raise _translate(raw, provider=self.name, model=self.model) from raw

        self._reasoning_disabled_by_server = True
        logger.warning(
            "服务端拒绝 reasoning_effort，本次摘掉该参数重试，并在本进程内停用它",
            extra={
                "provider": self.name,
                "model": self.model,
                "effort": self._settings.effective_reasoning_effort,
            },
        )
        try:
            return (
                await self._client.chat.completions.create(
                    **kwargs, **self._strip_reasoning_effort(payload)
                ),
                True,
            )
        except Exception as raw:
            raise _translate(raw, provider=self.name, model=self.model) from raw

    # -- 重试外壳 ------------------------------------------------------------------

    async def _with_retry(self, op: str, call: Callable[[], Awaitable[T]]) -> tuple[T, int]:
        """限流 + 重试。返回 ``(结果, 实际尝试次数)``。"""
        return await run_with_retry(
            op,
            call,
            max_attempts=self._max_attempts,
            limiter=self._limiter,
            provider=self.name,
            model=self.model,
        )

    # -- 一次性对话 ----------------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        resolved_temperature = (
            self._settings.temperature if temperature is None else temperature
        )
        resolved_max_tokens = self._settings.max_tokens if max_tokens is None else max_tokens
        payload = self._build_payload(
            messages,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            response_format=response_format,
            extra=extra,
            stream=False,
        )

        (completion, stripped_reasoning), attempts = await self._with_retry(
            "chat", lambda: self._create_once(payload)
        )
        return self._to_result(completion, attempts=attempts, stripped=stripped_reasoning)

    def _to_result(self, completion: Any, *, attempts: int, stripped: bool) -> ChatResult:
        choices = getattr(completion, "choices", None)
        if not choices:
            raise ProviderResponseError(
                "对话响应里没有 choices",
                provider=self.name,
                model=self.model,
                detail={"response_id": getattr(completion, "id", "")},
            )
        message = getattr(choices[0], "message", None)
        if message is None:
            raise ProviderResponseError(
                "对话响应里没有 message",
                provider=self.name,
                model=self.model,
            )

        usage = getattr(completion, "usage", None)
        reasons: list[str] = []
        if stripped or self._reasoning_disabled_by_server:
            reasons.append(str(DegradedReason.REASONING_EFFORT_REJECTED))

        return ChatResult(
            text=getattr(message, "content", "") or "",
            reasoning_text=_extra_field(message, "reasoning_content"),
            model=getattr(completion, "model", "") or self.model,
            finish_reason=getattr(choices[0], "finish_reason", "") or "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            attempts=attempts,
            degraded=bool(reasons),
            degraded_reasons=reasons,
            extra={"response_id": getattr(completion, "id", "")},
        )

    # -- 流式对话 ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatDelta]:
        payload = self._build_payload(
            messages,
            temperature=self._settings.temperature if temperature is None else temperature,
            max_tokens=self._settings.max_tokens if max_tokens is None else max_tokens,
            response_format=None,
            extra=extra,
            stream=True,
        )

        # 只有「建立连接」这一步进重试外壳。一旦开始吐字节，中途失败再重试
        # 会让前端收到两段拼不起来的答案 —— 流式场景下，宁可报错让用户重问。
        (stream, _stripped), _attempts = await self._with_retry(
            "chat_stream",
            lambda: self._create_once(payload, timeout_s=self._settings.stream_timeout_s),
        )

        try:
            async for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                text = getattr(delta, "content", "") or ""
                reasoning = _extra_field(delta, "reasoning_content")
                finish = getattr(choices[0], "finish_reason", "") or ""
                if text or reasoning or finish:
                    yield ChatDelta(
                        text=text,
                        reasoning_text=reasoning,
                        done=bool(finish),
                        finish_reason=finish,
                    )
        except Exception as exc:
            raise _translate(exc, provider=self.name, model=self.model) from exc
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001 - 关闭失败不该覆盖真正的异常
                    logger.debug("关闭流式响应失败", exc_info=True)

        yield ChatDelta(done=True)

    # -- 结构化输出辅助 ------------------------------------------------------------

    async def json_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], ChatResult]:
        """要求模型返回 JSON 对象并解析。

        用于 caption、查询改写、相关性判定这些「输出必须可机读」的场合。
        解析失败抛 :class:`ProviderResponseError`，由调用方决定是降级还是重试 ——
        这里不做「正则抠 JSON」的兜底，那只会把问题往后推。
        """
        result = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = result.text.strip()
        if text.startswith("```"):
            # 少数情况下即使要求了 json_object，模型仍会包一层代码块围栏
            text = _strip_code_fence(text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                "模型返回的内容不是合法 JSON",
                provider=self.name,
                model=self.model,
                detail={"raw_preview": result.text[:500], "error": str(exc)},
            ) from exc
        if not isinstance(parsed, dict):
            raise ProviderResponseError(
                "模型返回的 JSON 不是对象",
                provider=self.name,
                model=self.model,
                detail={"type": type(parsed).__name__},
            )
        return parsed, result


# --------------------------------------------------------------------------------------
# 内部工具
# --------------------------------------------------------------------------------------


def _extra_field(obj: Any, key: str) -> str:
    """读取 OpenAI 兼容响应里的非标准字段。

    ``reasoning_content`` 不在 OpenAI 官方 schema 里，openai SDK 的模型配置为
    ``extra="allow"``，所以它以「额外字段」的形式存在。不同 SDK 版本对额外字段的
    暴露方式可能是属性也可能是 ``model_extra``，两条路都试一遍。
    """
    value = getattr(obj, key, None)
    if value is None:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(key)
    return value if isinstance(value, str) else ""


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _translate(exc: BaseException, *, provider: str, model: str) -> ProviderError:
    """把 openai SDK 的异常翻译成本项目的异常层级。

    已经在 :class:`ProviderError` 里的异常原样返回 —— 重试策略依赖
    ``retryable`` 标志，再包一层会把它冲掉。
    """
    if isinstance(exc, ProviderError):
        return exc

    kwargs = {"provider": provider, "model": model}
    detail = {"upstream": type(exc).__name__, "message": str(exc)[:500]}

    if isinstance(exc, openai.APITimeoutError):
        return ProviderTimeoutError(f"请求 {provider} 超时", detail=detail, **kwargs)
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitError(f"{provider} 触发限流（429）", detail=detail, **kwargs)
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return ProviderAuthError(
            f"{provider} 鉴权失败，请检查 API Key 与模型权限", detail=detail, **kwargs
        )
    if isinstance(exc, openai.BadRequestError):
        return ProviderBadRequestError(f"{provider} 拒绝了请求（400）", detail=detail, **kwargs)
    if isinstance(exc, openai.NotFoundError):
        return ProviderBadRequestError(
            f"{provider} 返回 404：模型名或路径不存在", detail=detail, **kwargs
        )
    if isinstance(exc, openai.APIConnectionError):
        return ProviderUnavailableError(f"无法连接 {provider}", detail=detail, **kwargs)
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", 0) or 0
        if status >= 500:
            return ProviderUnavailableError(
                f"{provider} 服务端错误（{status}）", detail=detail, **kwargs
            )
        return ProviderBadRequestError(
            f"{provider} 返回 {status}", detail=detail, **kwargs
        )
    return ProviderUnavailableError(
        f"调用 {provider} 失败：{type(exc).__name__}", detail=detail, **kwargs
    )
