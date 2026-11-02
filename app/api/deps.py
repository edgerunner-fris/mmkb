"""FastAPI 依赖。

只做一件事：从 ``request.app.state`` 取容器。容器由 ``lifespan`` 放进
state —— 这样测试里换一个容器（临时目录、假 provider）只需要换一个
``create_app(settings)`` 参数，不用 monkeypatch 任何全局单例。
"""

from __future__ import annotations

from fastapi import Request

from app.services.container import AppContainer

__all__ = ["get_container", "get_settings"]


def get_container(request: Request) -> AppContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - 只有绕过 lifespan 的调用会走到
        raise RuntimeError(
            "应用容器尚未初始化：请通过 app.main:create_app() 并运行 lifespan"
        )
    return container


def get_settings(request: Request):
    return get_container(request).settings
