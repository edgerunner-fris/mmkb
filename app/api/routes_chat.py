"""问答、会话与检索路由。

``POST /api/chat/stream`` 是 SSE。它在返回 ``StreamingResponse`` **之前**
先做一次预检（知识库存在、LLM 可用），因为响应头一旦发出，就再也无法
把错误改成 404/503 的 JSON 了。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import get_container
from app.api.dto import ChatRequest, SearchRequest
from app.api.sse import with_heartbeat
from app.core.schemas import AnswerResult, Message
from app.services.container import AppContainer

__all__ = ["router"]

router = APIRouter(tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # nginx 默认会缓冲代理响应，SSE 会变成「等模型全部生成完才一次性吐出来」
    "X-Accel-Buffering": "no",
}


@router.post("/chat", response_model=AnswerResult, summary="知识库问答（非流式）")
async def chat(
    payload: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> AnswerResult:
    return await container.chat.ask(
        payload.kb_id,
        payload.question,
        session_id=payload.session_id,
        top_k=payload.top_k,
        use_rerank=payload.use_rerank,
        rewrite=payload.rewrite,
        debug=payload.debug,
    )


@router.post("/chat/stream", summary="知识库问答（SSE 流式）")
async def chat_stream(
    payload: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    await container.chat.preflight(payload.kb_id)
    events = container.chat.stream(
        payload.kb_id,
        payload.question,
        session_id=payload.session_id,
        top_k=payload.top_k,
        use_rerank=payload.use_rerank,
        rewrite=payload.rewrite,
        debug=payload.debug,
    )
    return StreamingResponse(
        with_heartbeat(
            events,
            interval_s=container.settings.generation.stream_heartbeat_s,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[Message],
    summary="会话历史消息",
)
async def list_messages(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    container: AppContainer = Depends(get_container),
) -> list[Message]:
    return await container.chat.messages(session_id, limit=limit)


@router.delete("/sessions/{session_id}", status_code=204, summary="删除会话")
async def delete_session(
    session_id: str,
    container: AppContainer = Depends(get_container),
) -> Response:
    await container.chat.delete_session(session_id)
    return Response(status_code=204)


@router.post(
    "/search/image",
    summary="以图搜图（多模态向量空间，独立于文本检索）",
)
async def search_image(
    file: UploadFile = File(..., description="查询图片"),
    kb_id: str = Form(..., description="知识库 ID"),
    top_k: int = Form(default=10, ge=1, le=100),
    container: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    """图片直接编码进多模态空间，**不经过** text-embedding（铁律 3）。

    没有文本 query，因此不使用 reranker —— 它只接受纯文本。
    """
    try:
        data = await container.ingest.read_upload(file)
    finally:
        await file.close()
    return await container.search.image_search(kb_id, data, top_k=top_k)


@router.post(
    "/kbs/{kb_id}/search",
    summary="原始混合检索（含每路诊断信息）",
)
async def search(
    kb_id: str,
    payload: SearchRequest,
    container: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return await container.search.search(
        kb_id,
        payload.query,
        routes=list(payload.routes) if payload.routes else None,
        top_k=payload.top_k,
        use_rerank=payload.use_rerank,
        rewrite=payload.rewrite,
        filters=payload.filters,
    )
