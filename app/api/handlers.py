"""异常 → HTTP 响应。

目标只有一个：**客户端拿到的永远是可解析的 JSON**，结构统一为::

    {"error": {"code": "...", "message": "...", "detail": {...}}}

这个结构由 :meth:`app.core.errors.MMKBError.to_dict` 定义，流式事件里的
``error`` 事件也用同一份结构 —— 前端只需要写一次错误处理。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import ConfigurationError
from app.core.errors import MMKBError
from app.core.logging import current_context, get_logger

__all__ = ["install_exception_handlers"]

logger = get_logger(__name__)


def _body(code: str, message: str, detail: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        payload["detail"] = detail
    return {"error": payload}


def _clean_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """去掉 Pydantic 错误里不可 JSON 序列化的 ``ctx.error`` 对象。"""
    cleaned: list[dict[str, Any]] = []
    for raw in errors:
        item = dict(raw)
        ctx = item.get("ctx")
        if isinstance(ctx, dict):
            item["ctx"] = {
                key: str(value) if isinstance(value, BaseException) else value
                for key, value in ctx.items()
            }
        cleaned.append(item)
    return cleaned


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(MMKBError)
    async def _mmkb_error(_request: Request, exc: MMKBError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(ConfigurationError)
    async def _config_error(_request: Request, exc: ConfigurationError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=_body(
                "configuration_error",
                "所需能力未配置，接口暂不可用",
                {
                    "problems": exc.problems,
                    "hint": "在 .env 中补齐配置后重启服务；GET /api/health 可查看完整问题列表",
                },
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_body(
                "validation_error",
                "请求参数不合法",
                {"errors": _clean_validation_errors(exc.errors())},
            ),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "请求处理出现未预期异常",
            extra={"method": request.method, "path": request.url.path},
        )
        return JSONResponse(
            status_code=500,
            content=_body(
                "internal_error",
                "服务器内部错误",
                {"request_id": current_context().get("request_id")},
            ),
        )
