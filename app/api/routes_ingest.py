"""导入与文档路由。

上传接口返回 **202 + Task**：真正的解析/向量化在后台跑，前端拿
``task_id`` 轮询 ``GET /api/tasks/{task_id}``。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from app.api.deps import get_container
from app.core.schemas import DocStatus, Document, Task
from app.services.container import AppContainer
from app.services.ingest_service import IncomingFile

__all__ = ["router"]

router = APIRouter(tags=["ingest"])


@router.post(
    "/kbs/{kb_id}/ingest",
    response_model=Task,
    status_code=202,
    summary="上传文件并创建后台导入任务",
)
async def ingest_files(
    kb_id: str,
    files: list[UploadFile] = File(..., description="可一次上传多个文件"),
    embed: bool = Form(default=True, description="是否生成向量（关闭则只写 BM25）"),
    caption: bool = Form(default=True, description="是否让视觉模型生成图片描述"),
    reindex: bool = Form(default=False, description="为 True 时强制重跑已完成的文档"),
    container: AppContainer = Depends(get_container),
) -> Task:
    service = container.ingest
    incoming: list[IncomingFile] = []
    try:
        for upload in files:
            # 先看扩展名再读字节：一个 200 MB 的 .mp4 不值得先传进内存
            service.validate_name(upload.filename or "")
            data = await service.read_upload(upload)
            incoming.append(IncomingFile(upload.filename or "upload", data))
    finally:
        for upload in files:
            await upload.close()

    return await service.submit(
        kb_id,
        incoming,
        embed=embed,
        caption=caption,
        reindex=reindex,
    )


@router.get("/tasks/{task_id}", response_model=Task, summary="查询导入任务进度")
async def get_task(
    task_id: str,
    container: AppContainer = Depends(get_container),
) -> Task:
    return await container.ingest.get_task(task_id)


@router.get(
    "/kbs/{kb_id}/tasks",
    response_model=list[Task],
    summary="列出知识库的导入任务",
)
async def list_tasks(
    kb_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    container: AppContainer = Depends(get_container),
) -> list[Task]:
    return await container.ingest.list_tasks(kb_id, limit=limit)


@router.get(
    "/kbs/{kb_id}/documents",
    response_model=list[Document],
    summary="列出知识库的文档",
)
async def list_documents(
    kb_id: str,
    status: DocStatus | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    container: AppContainer = Depends(get_container),
) -> list[Document]:
    return await container.ingest.list_documents(
        kb_id, status=status, limit=limit, offset=offset
    )


@router.get("/documents/{doc_id}", response_model=Document, summary="文档详情")
async def get_document(
    doc_id: str,
    container: AppContainer = Depends(get_container),
) -> Document:
    return await container.ingest.get_document(doc_id)
