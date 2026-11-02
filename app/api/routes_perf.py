"""性能报告路由。

路由函数只做三件事：取容器、调用服务、选择响应类型。排序、占比、Markdown
渲染都在 :class:`~app.services.perf_service.PerformanceService` 里 ——
同一条链路如果 JSON 快照与 Markdown 报告各写一份逻辑，迟早会出现
「页面显示 120ms、报告写着 80ms」这种无法解释的差异。

两个读端点分开是刻意的：JSON 快照包含原始指标与缓存明细，适合前端画图；
Markdown 报告适合直接贴进 README / 面试材料。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from app.api.deps import get_container
from app.services.container import AppContainer

__all__ = ["router"]

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("", summary="性能快照（进程内指标 / 缓存 / 存储规模）")
async def performance(
    container: AppContainer = Depends(get_container),
) -> JSONResponse:
    return JSONResponse(content=await container.perf.snapshot())


@router.get(
    "/report",
    summary="性能报告（Markdown，可直接保存/粘贴）",
    response_class=Response,
)
async def performance_report(
    container: AppContainer = Depends(get_container),
) -> Response:
    snapshot = await container.perf.snapshot()
    markdown = container.perf.render_markdown(snapshot)
    return Response(content=markdown, media_type="text/markdown")


@router.post("/reset", summary="清零进程内指标（样本与计数器）")
async def reset_performance(
    container: AppContainer = Depends(get_container),
) -> JSONResponse:
    container.perf.reset_metrics()
    return JSONResponse(
        content={
            "ok": True,
            "message": "指标已清零；知识库、缓存与任务记录不受影响。",
        }
    )


@router.post("/cache/clear", summary="清空检索链路缓存")
async def clear_cache(
    container: AppContainer = Depends(get_container),
) -> JSONResponse:
    cleared = container.perf.clear_caches()
    return JSONResponse(
        content={
            "ok": True,
            "cleared": cleared,
            "message": "已清空 query 向量、重排分数与 collection 元信息缓存。",
        }
    )
