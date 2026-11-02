"""健康检查路由。

两个端点语义不同：

* ``GET /api/health`` —— 存活 + 自检报告，**永远 200**；
* ``GET /api/health/ready`` —— 就绪探针，``not_ready`` 时 503。

把两者分开，是因为「进程活着但没配 Key」在开发机上必须能打开页面排查，
在编排系统里则不应该被派流量。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse

from app.api.deps import get_container
from app.services.container import AppContainer

__all__ = ["router"]

router = APIRouter(tags=["health"])


@router.get("/health", summary="健康与配置自检（永远 200）")
async def health(
    container: AppContainer = Depends(get_container),
) -> JSONResponse:
    return JSONResponse(status_code=200, content=await container.health.snapshot())


@router.get("/health/ready", summary="就绪探针（未就绪返回 503）")
async def ready(
    response: Response,
    container: AppContainer = Depends(get_container),
) -> JSONResponse:
    snapshot: dict[str, Any] = await container.health.snapshot()
    status = 200 if snapshot.get("ready") else 503
    response.status_code = status
    return JSONResponse(status_code=status, content=snapshot)
