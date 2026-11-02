"""知识库管理路由。函数只做协议转换，业务在 ``KnowledgeBaseService``。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_container
from app.api.dto import CreateKBRequest
from app.core.schemas import KnowledgeBase
from app.services.container import AppContainer

__all__ = ["router"]

router = APIRouter(tags=["knowledge-base"])


@router.get("/kbs", response_model=list[KnowledgeBase], summary="列出知识库")
async def list_kbs(
    include_deleting: bool = Query(default=False, description="包含正在删除的库"),
    container: AppContainer = Depends(get_container),
) -> list[KnowledgeBase]:
    return await container.kbs.list_kbs(include_deleting=include_deleting)


@router.post(
    "/kbs",
    response_model=KnowledgeBase,
    status_code=201,
    summary="新建知识库",
)
async def create_kb(
    payload: CreateKBRequest,
    container: AppContainer = Depends(get_container),
) -> KnowledgeBase:
    return await container.kbs.create_kb(payload.name, payload.description)


@router.get("/kbs/{kb_id}", response_model=KnowledgeBase, summary="知识库详情")
async def get_kb(
    kb_id: str,
    container: AppContainer = Depends(get_container),
) -> KnowledgeBase:
    return await container.kbs.get_kb(kb_id)


@router.delete("/kbs/{kb_id}", summary="删除知识库（两阶段，可续做）")
async def delete_kb(
    kb_id: str,
    container: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return await container.kbs.delete_kb(kb_id)
