"""provider 层的 HTTP 公共设施。

DashScope 的原生端点（多模态 embedding、text-rerank）与 SiliconFlow 的
``/rerank`` 都是「自己拼 JSON、自己发 POST」，没有官方 SDK 可用。
把这三处的传输、错误翻译、响应体诊断信息集中在这里，避免同一段
「status_code → 异常类型」的映射写三遍还写得不一样。

**厂商差异只体现在 URL 与 JSON 字段名上**，传输层完全一致。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.core.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

__all__ = [
    "build_async_client",
    "post_json",
    "translate_http_error",
    "truncate_body",
    "upstream_error",
]

logger = logging.getLogger(__name__)

#: 错误详情里回显的上游响应体长度上限。
#: 国产 API 的错误信息往往只有中文一句话，500 字符足够看清原因，
#: 又不至于把整篇 HTML 错误页塞进日志。
_BODY_PREVIEW = 500

#: 语义上属于「Key 无权访问该模型」的错误码。
#: 百炼会把这类错误塞进 HTTP 400（例如 ``AccessDenied`` +
#: "Access denied by API-Key restrictions."），如果按普通 400 处理，
#: 用户会误以为是请求体格式或维度参数写错了，白白排查很久。
_AUTH_ERROR_CODES = {
    "AccessDenied",
    "AccessDenied.APIKey",
    "InvalidApiKey",
    "Unauthorized",
    "AuthenticationError",
}


def truncate_body(text: str, limit: int = _BODY_PREVIEW) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…(truncated)"


def upstream_error(response: httpx.Response) -> tuple[str, str]:
    """从上游错误响应里提取 ``(message, code)``。

    两种常见形态都认：

    * OpenAI 兼容：``{"error": {"message": ..., "code": ...}}``
    * DashScope 原生：``{"code": ..., "message": ...}``

    取不到就返回空串。调用方不应因为解析失败而改变错误分类 ——
    这里只是给错误消息补上「为什么被拒」，不影响重试与状态码映射。
    """
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return "", ""
    if not isinstance(body, dict):
        return "", ""

    error = body.get("error")
    message = ""
    code = ""
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("msg") or "")
        code = str(error.get("code") or error.get("type") or "")
    elif isinstance(error, str):
        message = error
    if not message:
        message = str(body.get("message") or body.get("msg") or "")
    if not code:
        code = str(body.get("code") or "")

    message = " ".join(message.split())
    if len(message) > 200:
        message = message[:199] + "…"
    if message and code and code not in message:
        message = f"{message}（{code}）"
    return message, code


def build_async_client(
    *,
    timeout_s: float,
    api_key: str,
    extra_headers: dict[str, str] | None = None,
    connect_timeout_s: float = 15.0,
) -> httpx.AsyncClient:
    """构造一个带鉴权头的 ``httpx.AsyncClient``。

    连接超时单独设一个较短的值：DNS 解析失败或网络不通时，
    让请求快速失败并进入重试，而不是干等满整个业务超时。
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
    )


def translate_http_error(
    exc: BaseException,
    *,
    provider: str,
    model: str,
    url: str = "",
) -> ProviderError:
    """把 httpx 异常翻译成本项目的异常层级。

    已经是 :class:`ProviderError` 的原样返回 —— 重试策略依赖 ``retryable`` 标志，
    再包一层会把它冲掉。
    """
    if isinstance(exc, ProviderError):
        return exc

    kwargs = {"provider": provider, "model": model}
    base_detail: dict[str, Any] = {"upstream": type(exc).__name__, "url": url}

    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeoutError(
            f"请求 {provider} 超时（{url or 'unknown'}）",
            detail={**base_detail, "message": str(exc)[:200]},
            **kwargs,
        )
    if isinstance(exc, httpx.TransportError):
        return ProviderUnavailableError(
            f"无法连接 {provider}：{type(exc).__name__}",
            detail={**base_detail, "message": str(exc)[:200]},
            **kwargs,
        )

    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        status = response.status_code
        detail = {**base_detail, "status": status, "body": truncate_body(response.text)}
        reason, error_code = upstream_error(response)
        reason_text = f"：{reason}" if reason else ""
        if status == 429:
            return ProviderRateLimitError(
                f"{provider} 触发限流（429）{reason_text}", detail=detail, **kwargs
            )
        if status in (401, 403) or error_code in _AUTH_ERROR_CODES:
            return ProviderAuthError(
                f"{provider} 鉴权失败（{status}）{reason_text}。"
                "请检查 API Key 是否有效、是否被限制了模型访问范围，"
                "以及该模型是否已开通",
                detail=detail,
                **kwargs,
            )
        if status == 404:
            return ProviderBadRequestError(
                f"{provider} 返回 404：端点或模型名不存在{reason_text}",
                detail=detail,
                **kwargs,
            )
        if status >= 500:
            return ProviderUnavailableError(
                f"{provider} 服务端错误（{status}）{reason_text}", detail=detail, **kwargs
            )
        return ProviderBadRequestError(
            f"{provider} 拒绝了请求（{status}）{reason_text}", detail=detail, **kwargs
        )

    return ProviderUnavailableError(
        f"调用 {provider} 失败：{type(exc).__name__}",
        detail={**base_detail, "message": str(exc)[:200]},
        **kwargs,
    )


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> dict[str, Any]:
    """POST 一个 JSON 并返回解析后的对象。

    异常一律翻译成本项目层级；HTTP 200 但 body 不是合法 JSON 也会抛
    :class:`ProviderResponseError` —— 那通常意味着打到了错误的地址
    （拿到了一个 HTML 页面），静默返回空结果会让人排查很久。
    """
    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()
    except Exception as exc:
        raise translate_http_error(exc, provider=provider, model=model, url=url) from exc

    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProviderResponseError(
            f"{provider} 返回的不是 JSON，可能请求打到了错误的地址",
            provider=provider,
            model=model,
            detail={"url": url, "body": truncate_body(response.text)},
        ) from exc

    if not isinstance(body, dict):
        raise ProviderResponseError(
            f"{provider} 返回的 JSON 顶层不是对象",
            provider=provider,
            model=model,
            detail={"url": url, "type": type(body).__name__},
        )
    return body
