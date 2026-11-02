"""HTTP 层测试。

全部离线：没有任何 API Key，也不访问外网。这里验证的不是「模型答得好不好」，
而是**接线的正确性** —— 这些错误在本地很难靠肉眼发现：

1. 没配 Key 时应用仍能启动，``/api/health`` 如实报告 ``not_ready``，
   而不是启动崩溃或假装健康；
2. 错误必须有稳定的 JSON 信封（``{"error": {"code", "message"}}``），
   SSE 与普通 HTTP 共用同一套 code；
3. 上传 → 后台任务 → 轮询 → 文档列表 → BM25 检索 这条链路能走通，
   即使完全不构造 embedding provider；
4. SSE 帧格式与心跳：心跳必须用「不取消生成器」的方式实现，
   否则一次 ping 就会把模型输出整条流打断；
5. 流式问答在 ``done`` 之后才落 assistant 消息，且历史里能看到两条消息。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.sse import PING, encode_event, with_heartbeat
from app.config import (
    AppSettings,
    BlobSettings,
    BM25Settings,
    DocStoreSettings,
    ImageEmbeddingSettings,
    LLMSettings,
    PathSettings,
    RerankSettings,
    Settings,
    TextEmbeddingSettings,
    VectorStoreSettings,
    VisionSettings,
)
from app.generation.answer import AnswerEvent
from app.main import create_app

# --------------------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------------------


def _settings(tmp_path: Path) -> Settings:
    """指向 tmp_path 且所有 Key 为空。显式传空值可覆盖本机 .env 里的真实 Key。"""
    return Settings(
        app=AppSettings(env="dev", log_level="WARNING"),
        llm=LLMSettings(api_key=SecretStr("")),
        vision=VisionSettings(enabled=False),
        text_embed=TextEmbeddingSettings(api_key=SecretStr("")),
        image_embed=ImageEmbeddingSettings(api_key=SecretStr("")),
        rerank=RerankSettings(api_key=SecretStr("")),
        paths=PathSettings(data_dir=tmp_path),
        docstore=DocStoreSettings(sqlite_path=tmp_path / "test.db"),
        blob=BlobSettings(images_dir=tmp_path / "images"),
        bm25=BM25Settings(index_dir=tmp_path / "bm25", persist=False),
        vectorstore=VectorStoreSettings(chroma_dir=tmp_path / "chroma"),
    )


@pytest.fixture
def client(tmp_path: Path) -> Any:
    """with 语句保证 lifespan 跑完 —— 容器与后台任务都依赖它。"""
    app = create_app(_settings(tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def _create_kb(client: TestClient, name: str = "测试知识库") -> str:
    response = client.post("/api/kbs", json={"name": name, "description": "离线接口测试"})
    assert response.status_code == 201, response.text
    return response.json()["kb_id"]


def _wait_task(client: TestClient, task_id: str, timeout_s: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.get(f"/api/tasks/{task_id}")
        assert response.status_code == 200, response.text
        task = response.json()
        if task["status"] in {"done", "failed", "cancelled"}:
            return task
        time.sleep(0.05)
    raise AssertionError(f"任务 {task_id} 在 {timeout_s}s 内没有结束")


# --------------------------------------------------------------------------------------
# 健康检查
# --------------------------------------------------------------------------------------


def test_health_reports_not_ready_without_keys(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200, "健康端点永远 200，否则运维无法区分崩溃与未配置"

    body = response.json()
    assert body["service"] == "mmkb"
    assert body["status"] == "not_ready"
    assert body["ready"] is False
    assert body["capabilities"]["llm"] is False
    assert body["capabilities"]["dense_text"] is False
    # BM25 不依赖外部服务，没有 Key 也应该可用
    assert body["capabilities"]["bm25"] is True
    assert any("LLM_API_KEY" in error for error in body["config"]["errors"])

    ready = client.get("/api/health/ready")
    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"


# --------------------------------------------------------------------------------------
# 知识库 CRUD
# --------------------------------------------------------------------------------------


def test_kb_lifecycle_and_duplicate_name(client: TestClient) -> None:
    kb_id = _create_kb(client)

    listed = client.get("/api/kbs").json()
    assert [kb["kb_id"] for kb in listed] == [kb_id]

    fetched = client.get(f"/api/kbs/{kb_id}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "测试知识库"

    duplicate = client.post("/api/kbs", json={"name": "测试知识库"})
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "conflict"

    deleted = client.delete(f"/api/kbs/{kb_id}")
    assert deleted.status_code == 200
    assert deleted.json()["kb_id"] == kb_id

    assert client.get(f"/api/kbs/{kb_id}").status_code == 404
    assert client.get("/api/kbs").json() == []


def test_unknown_kb_returns_json_error(client: TestClient) -> None:
    response = client.get("/api/kbs/kb_000000000000")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert "kb_000000000000" in body["error"]["detail"]["kb_id"]


# --------------------------------------------------------------------------------------
# 上传 → 后台任务 → BM25 检索
# --------------------------------------------------------------------------------------


def test_ingest_then_bm25_search_without_embedding_keys(client: TestClient) -> None:
    kb_id = _create_kb(client, "上传测试")

    response = client.post(
        f"/api/kbs/{kb_id}/ingest",
        files={
            "files": (
                "note.txt",
                "多模态知识库支持图文混合检索，图片会生成描述并进入文本向量空间。".encode(),
                "text/plain",
            )
        },
        data={"embed": "false", "caption": "false", "reindex": "false"},
    )
    assert response.status_code == 202, response.text
    task_id = response.json()["task_id"]

    task = _wait_task(client, task_id)
    assert task["status"] == "done", task
    assert task["result"]["files"] == 1

    documents = client.get(f"/api/kbs/{kb_id}/documents").json()
    assert len(documents) == 1
    assert documents[0]["status"] == "done"
    assert documents[0]["chunk_count"] >= 1

    search = client.post(
        f"/api/kbs/{kb_id}/search",
        json={"query": "图文混合检索", "routes": ["bm25"], "top_k": 5},
    )
    assert search.status_code == 200, search.text
    outcome = search.json()
    assert outcome["count"] >= 1
    assert outcome["items"][0]["source_name"].endswith("note.txt")
    assert not outcome["routes"][0]["error"]

    detail = client.get(f"/api/documents/{documents[0]['doc_id']}")
    assert detail.status_code == 200
    assert detail.json()["doc_id"] == documents[0]["doc_id"]


def test_unsupported_upload_type_is_415(client: TestClient) -> None:
    kb_id = _create_kb(client, "类型校验")

    response = client.post(
        f"/api/kbs/{kb_id}/ingest",
        files={"files": ("movie.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")},
        data={"embed": "false", "caption": "false"},
    )
    assert response.status_code == 415
    body = response.json()
    assert body["error"]["code"] == "unsupported_file_type"
    assert body["error"]["detail"]["filename"] == "movie.mp4"


# --------------------------------------------------------------------------------------
# 问答降级
# --------------------------------------------------------------------------------------


def test_chat_without_llm_returns_configuration_error(client: TestClient) -> None:
    kb_id = _create_kb(client, "问答降级")

    response = client.post(
        "/api/chat", json={"kb_id": kb_id, "question": "你好"}
    )
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "configuration_error"
    assert body["error"]["detail"]["problems"]

    # 流式接口必须在发出 SSE 响应头之前就把配置问题挡掉
    stream = client.post(
        "/api/chat/stream", json={"kb_id": kb_id, "question": "你好"}
    )
    assert stream.status_code == 503
    assert stream.headers["content-type"].startswith("application/json")


# --------------------------------------------------------------------------------------
# SSE 帧与心跳
# --------------------------------------------------------------------------------------


def test_encode_event_is_valid_sse_frame() -> None:
    frame = encode_event(AnswerEvent("token", text="你好"))
    assert frame.startswith("event: token\ndata: ")
    assert frame.endswith("\n\n")

    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload == {"type": "token", "text": "你好"}

    done = encode_event(AnswerEvent("done", data={"confidence": 0.8}))
    assert json.loads(done.split("data: ", 1)[1].strip())["data"]["confidence"] == 0.8


async def test_heartbeat_does_not_cancel_the_generator() -> None:
    """生成器先睡一会儿再吐事件；期间应收到 ping，且最终事件必须还在。"""

    async def slow_events() -> AsyncIterator[AnswerEvent]:
        await asyncio.sleep(0.12)
        yield AnswerEvent("done", data={"ok": True})

    frames = [frame async for frame in with_heartbeat(slow_events(), interval_s=0.03)]

    assert frames[0] == PING
    assert any("event: done" in frame for frame in frames)
    # ping 不能把后续内容吃掉，也不能重复输出 done
    assert sum("event: done" in frame for frame in frames) == 1


# --------------------------------------------------------------------------------------
# 流式问答的落库顺序
# --------------------------------------------------------------------------------------


class _FakeAgent:
    """只实现 ChatService 用到的 answer_stream，验证 HTTP + 落库接线。"""

    def __init__(self) -> None:
        self.sessions: list[str | None] = []

    async def answer_stream(
        self,
        kb: Any,
        question: str,
        *,
        history: Any = None,
        top_k: Any = None,
        use_rerank: Any = None,
        rewrite: Any = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AnswerEvent]:
        self.sessions.append(session_id)
        yield AnswerEvent("meta", data={"session_id": session_id, "question": question})
        yield AnswerEvent("token", text="你好")
        yield AnswerEvent("token", text="，世界")
        yield AnswerEvent("citations", data={"citations": []})
        yield AnswerEvent("done", data={"confidence": 0.5})


def test_stream_chat_persists_messages_after_done(client: TestClient) -> None:
    kb_id = _create_kb(client, "流式测试")
    container = client.app.state.container
    agent = _FakeAgent()
    container.require_agent = lambda: agent

    response = client.post(
        "/api/chat/stream",
        json={"kb_id": kb_id, "question": "你好吗"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: token" in body
    assert "event: done" in body
    assert "event: error" not in body

    assert agent.sessions and agent.sessions[0], "ChatService 应创建并传入 session_id"
    session_id = agent.sessions[0]
    messages = client.get(f"/api/sessions/{session_id}/messages").json()
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "你好吗"),
        ("assistant", "你好，世界"),
    ]

    deleted = client.delete(f"/api/sessions/{session_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/sessions/{session_id}/messages").status_code == 404


# --------------------------------------------------------------------------------------
# 参数校验
# --------------------------------------------------------------------------------------


def test_invalid_payload_returns_422_with_clean_detail(client: TestClient) -> None:
    response = client.post("/api/kbs", json={"name": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["detail"]["errors"]
