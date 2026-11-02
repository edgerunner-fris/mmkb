"""FastAPI 应用装配。

启动策略
--------
``APP_ENV=prod``：配置不全直接拒绝启动（``Settings.require_ready``）。
``APP_ENV=dev``（默认）：缺 Key 只降级 —— 进程照常起，``/api/health``
如实列出缺什么，``/api/chat`` 返回 503。这不是「放松要求」，而是
把「配置问题」与「代码问题」分开：没有 Key 的人也应该能打开 ``/docs``
看到完整接口定义。

静态图片
--------
``BlobStore`` 生成的 URL 是 ``{STATIC_URL_PREFIX}/images/{kb}/{image_id}.{ext}``，
磁盘路径是 ``{IMAGES_DIR}/{kb}/{image_id}.{ext}``。所以挂载点是
``{prefix}/images`` 而不是 ``{prefix}`` —— 挂错一级会把
``data/images/images/...`` 当成真实路径，表现为「图片 404 但文件明明在」。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import routes_chat, routes_health, routes_ingest, routes_kb, routes_perf
from app.api.handlers import install_exception_handlers
from app.config import Settings, collect_config_problems, get_settings
from app.core.logging import (
    bind_context,
    clear_context,
    get_logger,
    new_request_id,
    setup_logging,
)
from app.core.perf import get_metrics
from app.services.container import AppContainer

__all__ = ["create_app", "app"]

logger = get_logger(__name__)

ContainerFactory = Callable[[Settings], AppContainer]

_DESCRIPTION = """\
通用多模态知识库问答系统。

**检索链路**：BM25 + 文本向量 + 多模态向量三路召回 → RRF 融合 → 重排 → 带引用生成。

三条向量空间铁律：

1. 文本查文本：query 用文本 embedding 编码，检索 `{kb_id}__text`；
2. 以文搜图：query 文本送进多模态融合端点，检索 `{kb_id}__image`；
3. 禁止混用：两个空间的余弦相似度没有意义。
"""


def create_app(
    settings: Settings | None = None,
    *,
    container_factory: ContainerFactory | None = None,
) -> FastAPI:
    """应用工厂。

    ``container_factory`` 是给测试/演示用的注入点：换一个容器就能用临时
    目录与假 provider 跑完整 HTTP 链路，不需要 monkeypatch 全局单例。
    """
    settings = settings or get_settings()
    setup_logging(settings.app.log_level, json_output=settings.app.log_json)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if settings.app.is_prod:
            # 生产：宁可起不来，也不要带着半个能力接流量
            settings.require_ready()

        factory = container_factory or AppContainer
        container = factory(settings)
        application.state.container = container
        try:
            await container.startup()
        except Exception:
            await container.shutdown()
            raise

        errors, warnings = collect_config_problems(settings)
        if errors:
            logger.warning(
                "配置不完整，服务以降级模式启动",
                extra={
                    "env": settings.app.env,
                    "errors": errors,
                    "hint": "GET /api/health 查看详情；APP_ENV=prod 时该配置会拒绝启动",
                },
            )
        elif warnings:
            logger.info("配置存在非致命告警", extra={"warnings": warnings})

        try:
            yield
        finally:
            await container.shutdown()
            application.state.container = None

    app = FastAPI(
        title="mmkb · 通用多模态知识库问答",
        version="0.1.0",
        description=_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    install_exception_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origin_list,
        allow_credentials="*" not in settings.app.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _request_context(request: Request, call_next: Any):
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        bind_context(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # 异常请求同样要计入延迟，但必须标成 error ——
            # 只统计成功请求的 p95 会让「接口一挂，延迟看起来更好了」。
            get_metrics().observe(
                "http.request", (time.perf_counter() - started) * 1000, error=True
            )
            clear_context()
            raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Request-ID"] = request_id

        # 用**路由模板**（``/api/kbs/{kb_id}/documents``）而不是真实路径，
        # 否则每个 kb_id 都会成为一个独立指标，报告里瞬间出现上千行。
        metrics = get_metrics()
        metrics.observe("http.request", elapsed_ms, error=response.status_code >= 500)
        metrics.incr("http.requests")
        metrics.incr(f"http.status.{response.status_code}")
        route = request.scope.get("route")
        template = getattr(route, "path", None) or request.url.path
        if template != request.url.path or not route:
            # 未匹配到路由（404 / 静态文件）统一归到 ``http.route.unmatched``，
            # 免得报告被任意扫描路径刷满
            template = "unmatched"
        metrics.observe(f"http.route.{template}", elapsed_ms)

        level = "debug" if request.url.path in {"/api/health", "/api/health/ready"} else "info"
        getattr(logger, level)(
            "http request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": elapsed_ms,
            },
        )
        clear_context()
        return response

    # 目录由 lifespan 的 paths.ensure() 创建；check_dir=False 让应用对象的
    # 构造不产生文件系统副作用（测试会 import 本模块），目录一出现即可服务
    app.mount(
        f"{settings.app.static_url_prefix.rstrip('/')}/images",
        StaticFiles(directory=settings.images_dir, check_dir=False),
        name="images",
    )

    api_router = APIRouter()
    api_router.include_router(routes_health.router)
    api_router.include_router(routes_kb.router)
    api_router.include_router(routes_ingest.router)
    api_router.include_router(routes_chat.router)
    api_router.include_router(routes_perf.router)
    app.include_router(api_router, prefix=settings.app.api_prefix)

    @app.get("/", include_in_schema=False)
    async def index() -> dict[str, str]:
        return {
            "service": "mmkb",
            "docs": "/docs",
            "health": f"{settings.app.api_prefix}/health",
        }

    return app


app = create_app()
