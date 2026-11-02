"""统一异常层级。

约定
----
* 每个异常带 ``code``（机器可读的稳定标识）与 ``http_status``（API 层直接映射）。
* ``retryable`` 决定 provider 层是否值得重试 —— 这是重试策略的唯一判据来源，
  避免在 tenacity 的 ``retry_if_exception_type`` 里散落一堆厂商异常类型。
* ``detail`` 用于携带结构化上下文（如 :class:`DimensionMismatchError` 的期望/实际维度），
  它会进入错误响应体的 ``detail`` 字段与结构化日志。
"""

from __future__ import annotations

from typing import Any

from app.config import ConfigurationError as _ConfigurationError

__all__ = [
    "ConfigurationError",
    "ConflictError",
    "DimensionMismatchError",
    "IngestError",
    "MMKBError",
    "NotFoundError",
    "ProviderAuthError",
    "ProviderBadRequestError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "ProviderUnsupportedOperationError",
    "RetrievalError",
    "StorageError",
    "UnsupportedFileTypeError",
    "ValidationError",
    "VectorDimensionMismatchError",
    "is_retryable",
]


class MMKBError(Exception):
    """所有业务异常的基类。"""

    code: str = "internal_error"
    http_status: int = 500
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = detail or {}
        if retryable is not None:
            self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "detail": self.detail,
            }
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --------------------------------------------------------------------------------------
# 客户端错误
# --------------------------------------------------------------------------------------


class ValidationError(MMKBError):
    code = "validation_error"
    http_status = 422


class NotFoundError(MMKBError):
    code = "not_found"
    http_status = 404


class ConflictError(MMKBError):
    """幂等冲突：例如同名知识库、重复导入。"""

    code = "conflict"
    http_status = 409


class UnsupportedFileTypeError(MMKBError):
    code = "unsupported_file_type"
    http_status = 415


# --------------------------------------------------------------------------------------
# provider 错误
# --------------------------------------------------------------------------------------


class ProviderError(MMKBError):
    """所有外部模型调用失败的基类。"""

    code = "provider_error"
    http_status = 502

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        detail: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        merged = dict(detail or {})
        if provider:
            merged.setdefault("provider", provider)
        if model:
            merged.setdefault("model", model)
        super().__init__(message, detail=merged, retryable=retryable)
        self.provider = provider
        self.model = model


class ProviderTimeoutError(ProviderError):
    code = "provider_timeout"
    http_status = 504
    retryable = True


class ProviderRateLimitError(ProviderError):
    code = "provider_rate_limited"
    http_status = 429
    retryable = True


class ProviderUnavailableError(ProviderError):
    """5xx / 连接失败 / DNS 失败等瞬时故障。"""

    code = "provider_unavailable"
    http_status = 502
    retryable = True


class ProviderAuthError(ProviderError):
    """401 / 403。重试没有意义。"""

    code = "provider_auth_error"
    http_status = 502
    retryable = False


class ProviderBadRequestError(ProviderError):
    """400。请求本身有问题，重试同样会失败。"""

    code = "provider_bad_request"
    http_status = 502
    retryable = False


class ProviderResponseError(ProviderError):
    """响应体结构不符合预期 —— 通常意味着 API 契约变了。"""

    code = "provider_response_error"
    http_status = 502
    retryable = False


class ProviderUnsupportedOperationError(ProviderError):
    """在错误的向量空间上调用了不支持的操作。

    这个异常存在的唯一目的，是让「铁律 3：禁止混用向量空间」在运行期变成
    一个响亮的报错，而不是一个静默的、维度恰好对得上却毫无意义的余弦相似度。

    例：拿文本 embedding provider 去 ``embed_fused``（那是多模态融合空间的操作），
    或拿多模态 provider 去 ``embed_texts``（那会污染 ``{kb}__text``）。
    """

    code = "provider_unsupported_operation"
    http_status = 500
    retryable = False


class DimensionMismatchError(ProviderError):
    """配置的向量维度与实际返回的维度不一致。

    这是启动期最关键的一道断言：维度错了，整个向量库的相似度计算都是错的，
    而且往往要到检索结果明显不合理时才会被发现。宁可启动就炸。
    """

    code = "dimension_mismatch"
    http_status = 500
    retryable = False

    def __init__(
        self,
        *,
        scope: str,
        expected: int,
        actual: int,
        provider: str = "",
        model: str = "",
    ) -> None:
        super().__init__(
            f"{scope} 维度不一致：配置 expected={expected}，实际返回 actual={actual}。"
            f"请修正 .env 中的 {scope} 维度配置，或换回与该维度匹配的模型。",
            provider=provider,
            model=model,
            detail={
                "scope": scope,
                "expected": expected,
                "actual": actual,
                "hint": "把 .env 里对应的 *_DIM 改成 actual，或把模型换回与 expected 匹配的那个",
            },
        )
        self.scope = scope
        self.expected = expected
        self.actual = actual


# --------------------------------------------------------------------------------------
# 摄入错误
# --------------------------------------------------------------------------------------


class IngestError(MMKBError):
    code = "ingest_error"
    http_status = 500


# --------------------------------------------------------------------------------------
# 检索错误
# --------------------------------------------------------------------------------------


class RetrievalError(MMKBError):
    """检索链路的**不可降级**失败。

    典型场景是知识库快照与当前配置不一致（换过 embedding 模型/维度）：
    此时新旧向量已经不在同一个空间里，继续检索只会得到看似正常、
    实则毫无意义的排序。这类问题必须显式报错，而不是像「rerank 没配 Key」
    那样降级成 RRF 排序。

    单路召回的网络故障**不**用这个异常 —— 那属于可降级失败，
    由检索层包成 ``RouteResult.error`` 并记录 ``degraded_reasons``。
    """

    code = "retrieval_error"
    http_status = 500
    retryable = False


# --------------------------------------------------------------------------------------
# 存储错误
# --------------------------------------------------------------------------------------


class StorageError(MMKBError):
    """存储层失败的基类（向量库 / 元数据库 / 图片落盘）。"""

    code = "storage_error"
    http_status = 500


class VectorDimensionMismatchError(StorageError):
    """要写进向量库的向量，维度与 collection 建立时记录的不一致。

    与 :class:`DimensionMismatchError` 的区别值得说清楚，因为两者的报错长得很像
    但排查方向完全不同：

    * :class:`DimensionMismatchError` —— 「**provider 返回的**维度 vs `.env` 配置」，
      发生在调用外部模型时。修法：改 `.env` 的 ``*_DIM``。
    * 本异常 —— 「**要写进库的**向量 vs 库里已有的向量空间」，发生在存储层。
      修法：查是不是换了 embedding 模型，或者写错了 collection。

    两者都会让余弦相似度变成无意义的数字，而且都不会自己报错。
    """

    code = "vector_dimension_mismatch"
    http_status = 500
    retryable = False

    def __init__(
        self,
        *,
        collection: str,
        expected: int,
        actual: int,
        hint: str = "",
    ) -> None:
        detail: dict[str, Any] = {
            "collection": collection,
            "expected": expected,
            "actual": actual,
        }
        if hint:
            detail["hint"] = hint
        super().__init__(
            f"向量维度与 collection {collection!r} 不符："
            f"该 collection 是 {expected} 维，收到 {actual} 维的向量。"
            + (f"{hint}" if hint else ""),
            detail=detail,
        )
        self.collection = collection
        self.expected = expected
        self.actual = actual


# --------------------------------------------------------------------------------------
# 再导出
# --------------------------------------------------------------------------------------

#: config 层的配置错误。定义在 config.py 里是为了让 config 保持零内部依赖；
#: 这里再导出一次，使业务代码可以只 import app.core.errors 一个地方。
ConfigurationError = _ConfigurationError


def is_retryable(exc: BaseException) -> bool:
    """provider 重试策略的唯一判据。

    对本项目自定义的 :class:`ProviderError` 直接看 ``retryable`` 标志；
    对 httpx / openai 等第三方异常按类型兜底判断。
    """
    if isinstance(exc, MMKBError):
        return exc.retryable

    try:
        import httpx
    except ImportError:  # pragma: no cover
        return False

    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return False
