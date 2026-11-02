"""mmkb 的 Streamlit 前端（单文件）。

启动::

    streamlit run frontend/streamlit_app.py

后端地址通过 ``MMKB_API_BASE`` 指定，默认 ``http://127.0.0.1:8000``。
Docker Compose 里把它设成 ``http://api:8000``。

这个文件只做三件事：调用 HTTP API、渲染、维护会话级 UI 状态。
**不 import app 包里的任何模块** —— 前端可以在没有项目依赖的环境里单独跑，
也避免有人顺手把业务逻辑塞进来。

三个页面：

* 问答 —— SSE 流式输出，引用可展开；图片引用直接渲染原图；
* 知识库 —— 上传（202 + 任务轮询）、文档列表、任务进度；
* 检索诊断 —— 绕过生成，直接看三路召回各自捞了什么、为什么被跳过；
* 性能 —— 进程内延迟分位、阶段拆解、缓存命中率与最近导入耗时。

错误统一处理后端信封 ``{"error": {"code", "message", "detail"}}``。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

API_BASE = os.getenv("MMKB_API_BASE", "http://127.0.0.1:8000").rstrip("/")
#: 浏览器能访问到的后端地址。Docker 里 API_BASE 是 `http://api:8000`（容器名，
#: 只有前端容器能解析），而图片 URL 由**浏览器**加载，必须用宿主机地址。
PUBLIC_BASE = os.getenv("MMKB_PUBLIC_BASE", API_BASE).rstrip("/")
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=120.0, pool=5.0)
# 流式读**不设总时限**：长回答与模型思考可能持续几分钟；
# 连接阶段仍然限时，避免后端没起来时页面一直转圈。
STREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=60.0, pool=5.0)

UPLOAD_TYPES = [
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "xlsm",
    "md",
    "markdown",
    "txt",
    "log",
    "rst",
    "csv",
    "tsv",
    "html",
    "htm",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "webp",
    "tif",
    "tiff",
    "py",
    "js",
    "ts",
    "java",
    "go",
    "rs",
    "c",
    "cpp",
    "h",
    "hpp",
    "sql",
    "sh",
    "json",
    "yaml",
    "yml",
    "toml",
]

STATUS_BADGE = {
    "ok": ("🟢", "正常"),
    "degraded": ("🟡", "降级运行"),
    "not_ready": ("🔴", "未就绪"),
}


# --------------------------------------------------------------------------------------
# HTTP 客户端
# --------------------------------------------------------------------------------------


class ApiError(RuntimeError):
    """后端不可用，或后端返回了错误信封。"""

    def __init__(self, code: str, message: str, status: int = 0, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail

    def __str__(self) -> str:
        prefix = f"[{self.code}]" if self.code else "[error]"
        return f"{prefix} {self.message}"


def _parse_error(response: httpx.Response) -> ApiError:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - 代理返回 HTML 时也要能显示
        return ApiError(
            "http_error",
            f"HTTP {response.status_code}: {response.text[:300]}",
            response.status_code,
        )
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return ApiError("http_error", f"HTTP {response.status_code}", response.status_code)
    return ApiError(
        str(error.get("code") or "http_error"),
        str(error.get("message") or "请求失败"),
        response.status_code,
        error.get("detail"),
    )


def _request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{API_BASE}{path}"
    # 大文件导入需要更长的 read/write 超时，调用方可以覆盖默认值；
    # 不能再额外传 timeout=，否则 httpx.request() 会收到重复关键字参数。
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    try:
        response = httpx.request(method, url, **kwargs)
    except httpx.HTTPError as exc:
        raise ApiError(
            "network_error",
            f"无法连接后端 {API_BASE}：{type(exc).__name__}: {exc}",
        ) from exc
    if response.status_code >= 400:
        raise _parse_error(response)
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def absolute_url(url: str | None) -> str | None:
    """后端返回的图片是 ``/static/images/...``，前端要补上浏览器可达的主机。"""
    if not url:
        return None
    if url.startswith(("http://", "https://", "data:")):
        return url
    return f"{PUBLIC_BASE}/{url.lstrip('/')}"


def stream_chat(payload: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """POST ``/api/chat/stream``，逐帧产出 ``(事件名, 负载)``。

    SSE frame 的格式是 ``event: <name>\\ndata: <json>\\n\\n``；
    以 ``:`` 开头的是心跳注释，直接忽略。这里用同步 httpx：
    Streamlit 的脚本本身跑在同步线程里，异步客户端反而要额外管理事件循环。
    """
    url = f"{API_BASE}/api/chat/stream"
    try:
        with httpx.Client(timeout=STREAM_TIMEOUT) as client:
            with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise _parse_error(response)

                event_name = ""
                data_lines: list[str] = []
                for line in response.iter_lines():
                    line = line.rstrip("\r")
                    if not line:
                        if event_name and data_lines:
                            raw = "\n".join(data_lines)
                            try:
                                parsed = json.loads(raw)
                            except json.JSONDecodeError as exc:
                                raise ApiError(
                                    "bad_stream",
                                    f"SSE 数据不是合法 JSON：{raw[:200]}",
                                ) from exc
                            yield event_name, parsed
                        event_name = ""
                        data_lines = []
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].strip())
    except httpx.HTTPError as exc:
        raise ApiError(
            "network_error",
            f"流式连接失败：{type(exc).__name__}: {exc}",
        ) from exc


# --------------------------------------------------------------------------------------
# 轻量读取封装：失败时在页面上显示，不打断整个脚本
# --------------------------------------------------------------------------------------


def fetch_health() -> dict[str, Any] | None:
    try:
        return _request("GET", "/api/health")
    except ApiError:
        return None


def fetch_kbs() -> list[dict[str, Any]]:
    try:
        return _request("GET", "/api/kbs") or []
    except ApiError as exc:
        st.sidebar.error(str(exc))
        return []


def fetch_documents(kb_id: str) -> list[dict[str, Any]]:
    try:
        return _request("GET", f"/api/kbs/{kb_id}/documents") or []
    except ApiError as exc:
        st.error(str(exc))
        return []


def fetch_tasks(kb_id: str) -> list[dict[str, Any]]:
    try:
        return _request("GET", f"/api/kbs/{kb_id}/tasks") or []
    except ApiError as exc:
        st.error(str(exc))
        return []


def _request_text(path: str) -> str:
    """取纯文本响应（Markdown 报告）。与 :func:`_request` 分开，避免 JSON 解析。"""
    url = f"{API_BASE}{path}"
    try:
        response = httpx.get(url, timeout=REQUEST_TIMEOUT)
    except httpx.HTTPError as exc:
        raise ApiError(
            "network_error",
            f"无法连接后端 {API_BASE}：{type(exc).__name__}: {exc}",
        ) from exc
    if response.status_code >= 400:
        raise _parse_error(response)
    return response.text


def fetch_performance() -> dict[str, Any] | None:
    try:
        return _request("GET", "/api/performance")
    except ApiError as exc:
        st.error(f"读取性能快照失败：{exc}")
        return None


def _fmt_ms(value: Any) -> str:
    try:
        return f"{float(value):,.0f} ms"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------------------
# 渲染辅助
# --------------------------------------------------------------------------------------


def render_health(snapshot: dict[str, Any] | None) -> None:
    if snapshot is None:
        st.sidebar.error("🔴 无法连接后端")
        st.sidebar.caption(f"API：`{API_BASE}`")
        st.sidebar.caption("先启动：`uvicorn app.main:app --port 8000`")
        return

    status = str(snapshot.get("status") or "unknown")
    icon, label = STATUS_BADGE.get(status, ("⚪", status))
    st.sidebar.markdown(f"### {icon} {label}")
    st.sidebar.caption(
        f"API `{API_BASE}` · v{snapshot.get('version', '?')} · "
        f"env `{snapshot.get('env', '?')}`"
    )
    capabilities = snapshot.get("capabilities") or {}
    labels = {
        "llm": "对话生成",
        "dense_text": "文本向量",
        "dense_image": "图像向量",
        "rerank": "重排",
        "vision": "图片理解",
        "bm25": "BM25",
    }
    columns = st.sidebar.columns(2)
    for index, (key, text) in enumerate(labels.items()):
        ok = bool(capabilities.get(key))
        columns[index % 2].caption(f"{'✅' if ok else '❌'} {text}")

    problems = (snapshot.get("config") or {}).get("errors") or []
    if problems:
        with st.sidebar.expander(f"配置问题（{len(problems)}）", expanded=False):
            for problem in problems:
                st.warning(str(problem))
    provider_errors = snapshot.get("provider_errors") or []
    if provider_errors:
        with st.sidebar.expander(f"Provider 告警（{len(provider_errors)}）"):
            for problem in provider_errors:
                st.caption(str(problem))


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    st.markdown("**引用来源**")
    for citation in citations:
        index = citation.get("index", "?")
        name = citation.get("doc_name") or citation.get("doc_id") or citation.get("item_id")
        title = f"[{index}] {name}"
        if citation.get("page"):
            title += f" · 第 {citation['page']} 页"
        heading = " > ".join(citation.get("heading_path") or [])
        if heading:
            title += f" · {heading}"

        if citation.get("item_type") == "image" and citation.get("image_url"):
            url = absolute_url(citation.get("image_url"))
            if url:
                st.image(url, caption=title, width=480)
            else:
                st.caption(title)
            if citation.get("snippet"):
                st.caption(str(citation["snippet"])[:400])
            continue

        with st.expander(title):
            st.write(citation.get("snippet") or "（无摘要）")
            score = citation.get("score")
            rerank_score = citation.get("rerank_score")
            detail = f"item_id=`{citation.get('item_id')}`"
            if score is not None:
                detail += f" · score={float(score):.4f}"
            if rerank_score is not None:
                detail += f" · rerank={float(rerank_score):.4f}"
            st.caption(detail)


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message.get("content") or "（空）")
        if message.get("role") == "assistant":
            render_citations(message.get("citations") or [])
            meta_bits: list[str] = []
            if message.get("confidence") is not None:
                meta_bits.append(f"置信度 {float(message['confidence']):.2f}")
            if message.get("used_rounds"):
                meta_bits.append(f"检索 {message['used_rounds']} 轮")
            if message.get("latency_ms"):
                meta_bits.append(f"{message['latency_ms']} ms")
            degraded = message.get("degraded") or []
            if degraded:
                meta_bits.append("降级：" + ", ".join(str(item) for item in degraded))
            if meta_bits:
                st.caption(" · ".join(meta_bits))


def severity_icon(status: str) -> str:
    return {
        "done": "✅",
        "failed": "❌",
        "cancelled": "⏹️",
        "running": "⏳",
        "queued": "🕒",
    }.get(status, "•")


# --------------------------------------------------------------------------------------
# 页面
# --------------------------------------------------------------------------------------


def page_chat(kb: dict[str, Any], settings: dict[str, Any]) -> None:
    kb_id = kb["kb_id"]
    st.subheader(f"问答 · {kb['name']}")

    controls = st.columns([1.2, 1.2, 3])
    with controls[0]:
        if st.button("新建会话", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()
    with controls[1]:
        st.caption(f"session：`{st.session_state.session_id or '未开始'}`")
    with controls[2]:
        if st.session_state.last_meta:
            meta = st.session_state.last_meta
            st.caption(
                f"意图 `{meta.get('intent', '?')}` · 改写查询 "
                + (", ".join(meta.get("queries") or []) or "—")
            )

    if not st.session_state.messages:
        st.info(
            "回答会标注引用编号 `[n]`，下方「引用来源」里可以展开原文；"
            "图片引用会直接渲染原图。"
        )

    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input("请输入问题…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message(st.session_state.messages[-1])

    payload: dict[str, Any] = {
        "kb_id": kb_id,
        "question": prompt,
        "session_id": st.session_state.session_id,
        "top_k": settings["top_k"],
        "use_rerank": settings["use_rerank"],
        "rewrite": settings["rewrite"],
        "debug": True,
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()

        answer = ""
        reasoning = ""
        citations: list[dict[str, Any]] = []
        done_data: dict[str, Any] = {}
        error: ApiError | None = None

        try:
            for event_name, event in stream_chat(payload):
                data = event.get("data") or {}
                if event_name == "meta":
                    st.session_state.last_meta = data
                elif event_name == "token":
                    answer += str(event.get("text") or "")
                    placeholder.markdown(answer + "▌")
                elif event_name == "reasoning":
                    reasoning += str(event.get("text") or "")
                elif event_name == "citations":
                    citations = list(data.get("citations") or [])
                elif event_name == "done":
                    done_data = data
                elif event_name == "error":
                    error = ApiError(
                        str(data.get("code") or "stream_error"),
                        str(data.get("message") or "问答失败"),
                        detail=data.get("detail"),
                    )
        except ApiError as exc:
            error = exc

        if error is not None:
            placeholder.empty()
            st.error(str(error))
            if error.detail:
                st.caption(str(error.detail)[:500])
            return

        if not answer:
            placeholder.warning("模型没有返回内容。")
            return
        placeholder.markdown(answer)
        if reasoning.strip():
            with st.expander("推理过程（模型思考）", expanded=False):
                st.markdown(reasoning)
        render_citations(citations)

        session_id = done_data.get("session_id")
        if session_id:
            st.session_state.session_id = session_id
        degraded = done_data.get("degraded_reasons") or []
        footer: list[str] = []
        if done_data.get("confidence") is not None:
            footer.append(f"置信度 {float(done_data['confidence']):.2f}")
        if done_data.get("used_rounds"):
            footer.append(f"检索 {done_data['used_rounds']} 轮")
        if done_data.get("latency_ms"):
            footer.append(f"{done_data['latency_ms']} ms")
        if degraded:
            footer.append("降级：" + ", ".join(str(item) for item in degraded))
        if footer:
            st.caption(" · ".join(footer))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "confidence": done_data.get("confidence"),
                "used_rounds": done_data.get("used_rounds"),
                "latency_ms": done_data.get("latency_ms"),
                "degraded": degraded,
            }
        )
        st.session_state.last_done = done_data


def page_knowledge_base(kb: dict[str, Any]) -> None:
    kb_id = kb["kb_id"]
    st.subheader(f"知识库 · {kb['name']}")
    st.caption(
        f"`{kb_id}` · 文档 {kb.get('doc_count', 0)} · "
        f"分块 {kb.get('chunk_count', 0)} · 图片 {kb.get('image_count', 0)}"
    )

    upload_tab, documents_tab, tasks_tab, danger_tab = st.tabs(
        ["上传", "文档", "导入任务", "危险操作"]
    )

    with upload_tab:
        files = st.file_uploader(
            "选择文件（可多选）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="支持 PDF / Office / Markdown / 文本 / 代码 / 图片；单文件大小上限见后端配置。",
        )
        options = st.columns(3)
        embed = options[0].checkbox("生成向量", value=True)
        caption = options[1].checkbox("图片生成描述", value=True)
        reindex = options[2].checkbox("强制重建已完成文档", value=False)
        if st.button("开始导入", type="primary", disabled=not files):
            multipart = [
                (
                    "files",
                    (
                        upload.name,
                        upload.getvalue(),
                        upload.type or "application/octet-stream",
                    ),
                )
                for upload in files or []
            ]
            try:
                task = _request(
                    "POST",
                    f"/api/kbs/{kb_id}/ingest",
                    files=multipart,
                    data={
                        "embed": str(embed).lower(),
                        "caption": str(caption).lower(),
                        "reindex": str(reindex).lower(),
                    },
                    timeout=httpx.Timeout(connect=5.0, read=600.0, write=600.0, pool=5.0),
                )
            except ApiError as exc:
                st.error(str(exc))
                if exc.detail:
                    st.json(exc.detail)
            else:
                st.success(f"已创建导入任务 `{task['task_id']}`，状态 `{task['status']}`")
                st.caption("任务在服务端后台执行；可以离开本页，稍后到「导入任务」查看。")
        st.divider()
        st.caption(
            "图片会被单独抽出来：VLM 生成 caption（进入文本索引）"
            "+ 多模态向量（进入图像索引）。没有配置视觉模型时 caption 会降级为上下文文本。"
        )

    with documents_tab:
        if st.button("刷新文档列表"):
            st.rerun()
        documents = fetch_documents(kb_id)
        if not documents:
            st.info("还没有文档。先在上传页导入文件。")
        else:
            rows = [
                {
                    "状态": f"{severity_icon(doc.get('status', ''))} {doc.get('status')}",
                    "文件": doc.get("source_name") or doc.get("source_path"),
                    "类型": doc.get("mime_type", ""),
                    "分块": doc.get("chunk_count", 0),
                    "图片": doc.get("image_count", 0),
                    "过滤图片": doc.get("skipped_image_count", 0),
                    "大小": f"{(doc.get('size_bytes') or 0) / 1024:.1f} KB",
                    "doc_id": doc.get("doc_id", ""),
                }
                for doc in documents
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            failed = [doc for doc in documents if doc.get("status") == "failed"]
            for doc in failed:
                st.error(f"{doc.get('source_name')}：{doc.get('error') or '导入失败'}")

    with tasks_tab:
        header = st.columns([1, 3])
        if header[0].button("刷新任务", use_container_width=True):
            st.rerun()
        header[1].caption("任务进度不会自动刷新，点左侧按钮拉取最新状态。")
        tasks = fetch_tasks(kb_id)
        if not tasks:
            st.info("还没有导入任务。")
        else:
            for task in tasks:
                status = str(task.get("status", ""))
                total = int(task.get("progress_total") or 0)
                current = int(task.get("progress_current") or 0)
                title = (
                    f"{severity_icon(status)} `{task.get('task_id')}` · {status} · "
                    f"{task.get('stage', '')}"
                )
                with st.expander(title, expanded=status in {"queued", "running", "failed"}):
                    if total > 0:
                        st.progress(min(1.0, current / total), text=f"{current}/{total}")
                    st.caption(
                        f"创建 {task.get('created_at')} · 结束 {task.get('finished_at') or '—'}"
                    )
                    if task.get("error"):
                        st.error(str(task["error"]))
                    if task.get("stage_timings"):
                        st.json(task["stage_timings"])
                    if task.get("result"):
                        st.json(task["result"])

    with danger_tab:
        st.warning(
            "删除会清空该知识库的两个向量 collection、BM25 索引、图片文件与元数据。"
            "服务端是两阶段删除，中途失败会在下次启动时续做。"
        )
        confirm = st.checkbox("我确认要删除整个知识库", value=False)
        if st.button("删除知识库", type="primary", disabled=not confirm):
            try:
                result = _request("DELETE", f"/api/kbs/{kb_id}")
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.session_state.kb_id = None
                st.session_state.session_id = None
                st.session_state.messages = []
                st.success(f"已删除：{result}")
                st.rerun()


def page_performance() -> None:
    """性能报告页。

    数据来自后端**当前进程**的内存指标，重启即清零；页面上所有表都由
    ``GET /api/performance`` 的同一份快照渲染，避免多处口径不一致。
    """
    st.subheader("⚡ 性能报告")
    st.caption(
        "延迟为进程内指标（重启清零）。p50/p95/p99 采用最近秩百分位，"
        "样本来自真实问答、检索与导入调用。"
    )

    controls = st.columns([1, 1, 1, 3])
    if controls[0].button("刷新", use_container_width=True):
        st.rerun()
    if controls[1].button("清零指标", use_container_width=True):
        try:
            _request("POST", "/api/performance/reset")
            st.toast("指标已清零")
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))
    if controls[2].button("清空缓存", use_container_width=True):
        try:
            result = _request("POST", "/api/performance/cache/clear") or {}
            st.toast(f"已清理：{', '.join(result.get('cleared') or []) or '无缓存'}")
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))

    data = fetch_performance()
    if data is None:
        return

    summary = data.get("summary") or {}
    qa = summary.get("qa") or {}
    retrieval = summary.get("retrieval") or {}
    caches = (data.get("caches") or {}).get("totals") or {}
    ingest = data.get("ingest") or {}

    cards = st.columns(4)
    cards[0].metric(
        "端到端问答 p95",
        _fmt_ms(qa.get("p95_ms")),
        help=f"样本 {qa.get('count', 0)} 次",
    )
    cards[1].metric(
        "检索链路 p95",
        _fmt_ms(retrieval.get("p95_ms")),
        help=f"样本 {retrieval.get('count', 0)} 次",
    )
    cards[2].metric(
        "缓存命中率",
        f"{float(caches.get('hit_rate') or 0) * 100:.1f}%",
        help=f"{caches.get('hits', 0)} / {caches.get('lookups', 0)} 次查询命中",
    )
    cards[3].metric(
        "运行中导入任务",
        ingest.get("running_tasks", 0),
        help=f"进程已运行 {data.get('uptime_s', 0)} 秒",
    )

    findings = data.get("findings") or []
    if findings:
        st.markdown("#### 自动判读")
        for item in findings:
            st.markdown(f"- {item}")

    st.markdown("#### 各链路耗时总览")
    overview = [
        {
            "链路": row.get("label", key),
            "样本数": row.get("count", 0),
            "p50 (ms)": round(float(row.get("p50_ms") or 0), 1),
            "p95 (ms)": round(float(row.get("p95_ms") or 0), 1),
            "p99 (ms)": round(float(row.get("p99_ms") or 0), 1),
            "最大 (ms)": round(float(row.get("max_ms") or 0), 1),
            "错误率": f"{float(row.get('error_rate') or 0) * 100:.1f}%",
        }
        for key, row in summary.items()
        if row.get("count")
    ]
    if overview:
        st.dataframe(overview, use_container_width=True, hide_index=True)
    else:
        st.info("还没有耗时样本。先导入文档、问几个问题，再回到这里刷新。")

    breakdowns = data.get("breakdowns") or {}
    stages = breakdowns.get("retrieval_stages") or []
    stage_rows = [
        {
            "阶段": row.get("label", row.get("key")),
            "样本数": row.get("count", 0),
            "平均 (ms)": round(float(row.get("mean_ms") or 0), 1),
            "p95 (ms)": round(float(row.get("p95_ms") or 0), 1),
            "占平均总耗时": (
                f"{row['share_pct']}%" if row.get("share_pct") is not None else "—"
            ),
        }
        for row in stages
    ]
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### 检索阶段拆解")
        st.dataframe(stage_rows, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### 平均耗时（ms）")
        chart = [
            {"阶段": row.get("label", row.get("key")), "平均耗时 (ms)": row.get("mean_ms", 0)}
            for row in stages
            if row.get("count")
        ]
        if chart:
            st.bar_chart(chart, x="阶段", y="平均耗时 (ms)", height=280)
        else:
            st.info("暂无检索样本")

    st.markdown("#### 三路召回")
    route_rows = [
        {
            "召回路": row.get("label", row.get("route")),
            "成功调用": row.get("calls", 0),
            "失败调用": row.get("route_errors", 0),
            "p50 (ms)": round(float(row.get("p50_ms") or 0), 1),
            "p95 (ms)": round(float(row.get("p95_ms") or 0), 1),
        }
        for row in breakdowns.get("routes") or []
    ]
    st.dataframe(route_rows, use_container_width=True, hide_index=True)

    st.markdown("#### 缓存命中率")
    cache_rows = [
        {
            "缓存": row.get("name"),
            "命中": row.get("hits", 0),
            "查询": row.get("lookups", 0),
            "命中率": f"{float(row.get('hit_rate') or 0) * 100:.1f}%",
            "条目": row.get("size", 0),
            "容量": row.get("max_entries", 0),
            "TTL (s)": row.get("ttl_s", 0),
        }
        for row in (data.get("caches") or {}).get("rows") or []
    ]
    if cache_rows:
        st.dataframe(cache_rows, use_container_width=True, hide_index=True)
    else:
        st.info("缓存已启用，但还没有查询经过它。")

    ingest_stages = breakdowns.get("ingest_stages") or []
    st.markdown("#### 导入阶段")
    ingest_rows = [
        {
            "阶段": row.get("label", row.get("key")),
            "样本数": row.get("count", 0),
            "平均 (ms)": round(float(row.get("mean_ms") or 0), 1),
            "p95 (ms)": round(float(row.get("p95_ms") or 0), 1),
            "占单文件平均耗时": (
                f"{row['share_pct']}%" if row.get("share_pct") is not None else "—"
            ),
        }
        for row in ingest_stages
    ]
    st.dataframe(ingest_rows, use_container_width=True, hide_index=True)

    st.markdown("#### 接口延迟")
    http_rows = [
        {
            "路由模板": row.get("route"),
            "样本数": row.get("count", 0),
            "p50 (ms)": round(float(row.get("p50_ms") or 0), 1),
            "p95 (ms)": round(float(row.get("p95_ms") or 0), 1),
            "错误率": f"{float(row.get('error_rate') or 0) * 100:.1f}%",
        }
        for row in breakdowns.get("http_routes") or []
        if "status" not in row
    ]
    st.dataframe(http_rows, use_container_width=True, hide_index=True)

    recent = ingest.get("recent_tasks") or []
    if recent:
        st.markdown("#### 最近导入任务")
        st.dataframe(
            [
                {
                    "任务": task.get("task_id"),
                    "状态": task.get("status"),
                    "文件数": task.get("files", 0),
                    "累计耗时 (ms)": task.get("elapsed_ms"),
                    "阶段耗时 (ms)": ", ".join(
                        f"{key}={value}"
                        for key, value in (task.get("stage_timings") or {}).items()
                    )
                    or "—",
                }
                for task in recent
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Markdown 报告（可下载 / 粘贴）"):
        try:
            report = _request_text("/api/performance/report")
        except ApiError as exc:
            st.error(f"生成 Markdown 报告失败：{exc}")
        else:
            st.download_button(
                "下载 report.md",
                data=report,
                file_name="mmkb_performance.md",
                mime="text/markdown",
            )
            st.code(report, language="markdown")

    with st.expander("当前配置"):
        st.json(data.get("config") or {})


def page_diagnostics(kb: dict[str, Any]) -> None:
    kb_id = kb["kb_id"]
    st.subheader("检索诊断")
    st.caption(
        "文本查询走 `/api/kbs/{kb_id}/search`，以图搜图走 `/api/search/image`。"
        "两条路径都不经过生成，每一路为什么被跳过、各花了多少毫秒都在这里。"
    )

    mode = st.radio(
        "检索方式",
        ["文本查询", "以图搜图"],
        horizontal=True,
        help="以图搜图直接用图片编码进多模态空间，没有文本 query，因此不参与重排。",
    )

    outcome: dict[str, Any] | None = None
    if mode == "文本查询":
        outcome = _diagnose_text_search(kb_id)
    else:
        outcome = _diagnose_image_search(kb_id)

    if outcome is not None:
        _render_search_outcome(outcome)


def _diagnose_text_search(kb_id: str) -> dict[str, Any] | None:
    query = st.text_input("检索词", placeholder="例如：三路召回如何融合？")
    columns = st.columns(4)
    routes = columns[0].multiselect(
        "召回路",
        options=["bm25", "dense_text", "dense_image"],
        default=["bm25", "dense_text", "dense_image"],
        help="留空等于按服务端配置跑全部启用的路。",
    )
    top_k = columns[1].number_input("top_k", min_value=1, max_value=50, value=10)
    rerank_mode = columns[2].selectbox("重排", ["按配置", "开启", "关闭"], index=0)
    rewrite = columns[3].checkbox("查询改写", value=False)

    if not st.button("执行检索", type="primary", disabled=not query.strip()):
        return None

    payload: dict[str, Any] = {
        "query": query.strip(),
        "routes": routes or None,
        "top_k": int(top_k),
        "use_rerank": {"按配置": None, "开启": True, "关闭": False}[rerank_mode],
        "rewrite": rewrite,
    }
    try:
        with st.spinner("检索中…"):
            return _request("POST", f"/api/kbs/{kb_id}/search", json=payload)
    except ApiError as exc:
        st.error(str(exc))
        if exc.detail:
            st.json(exc.detail)
        return None


def _diagnose_image_search(kb_id: str) -> dict[str, Any] | None:
    uploaded = st.file_uploader(
        "查询图片",
        type=["png", "jpg", "jpeg", "webp", "bmp", "gif"],
        help="图片直接进多模态向量空间，不会经过文本 embedding。",
    )
    top_k = st.number_input("top_k", min_value=1, max_value=50, value=10, key="image_top_k")

    if not st.button(
        "执行以图搜图",
        type="primary",
        disabled=uploaded is None,
    ):
        return None

    if uploaded is None:
        return None
    try:
        with st.spinner("编码查询图片并检索…"):
            return _request(
                "POST",
                "/api/search/image",
                files={
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    )
                },
                data={"kb_id": kb_id, "top_k": str(int(top_k))},
            )
    except ApiError as exc:
        st.error(str(exc))
        if exc.detail:
            st.json(exc.detail)
        return None


def _render_search_outcome(outcome: dict[str, Any]) -> None:
    summary = st.columns(4)
    summary[0].metric("候选总数", outcome.get("fused_count", 0))
    summary[1].metric("返回条目", outcome.get("count", 0))
    summary[2].metric("已重排", "是" if outcome.get("reranked") else "否")
    summary[3].metric("查询数", len(outcome.get("queries") or []))

    if outcome.get("queries"):
        st.caption("实际执行的查询：" + " ｜ ".join(outcome["queries"]))
    degraded = outcome.get("degraded_reasons") or []
    if degraded:
        st.warning("降级：" + ", ".join(str(reason) for reason in degraded))

    st.markdown("**每一路的情况**")
    route_rows = []
    for route in outcome.get("routes") or []:
        route_rows.append(
            {
                "路": route.get("route"),
                "条数": route.get("count"),
                "耗时(ms)": route.get("latency_ms"),
                "跳过": "是" if route.get("skipped") else "否",
                "说明": route.get("note") or "",
                "错误": route.get("error") or "",
                "降级": route.get("degraded_reason") or "",
            }
        )
    st.dataframe(route_rows, use_container_width=True, hide_index=True)

    items = outcome.get("items") or []
    st.markdown(f"**结果（{len(items)} 条）**")
    if not items:
        st.info("没有召回结果。先确认已导入文档，且对应 provider 可用。")
        return

    item_rows = [
        {
            "排名": index,
            "类型": item.get("item_type"),
            "分数": round(float(item.get("score") or 0.0), 4),
            "RRF": round(float(item.get("rrf_score") or 0.0), 4),
            "重排": item.get("rerank_score"),
            "来源路": item.get("route"),
            "文档": item.get("source_name") or item.get("doc_id"),
            "章节": " > ".join(item.get("heading_path") or []),
            "摘要": (item.get("text") or "")[:80].replace("\n", " "),
        }
        for index, item in enumerate(items, start=1)
    ]
    st.dataframe(item_rows, use_container_width=True, hide_index=True)

    for index, item in enumerate(items, start=1):
        heading = " > ".join(item.get("heading_path") or [])
        label = f"#{index} {item.get('item_type')} · {item.get('source_name') or item.get('doc_id')}"
        if heading:
            label += f" · {heading}"
        with st.expander(label):
            st.caption(
                f"item_id=`{item.get('item_id')}` · route=`{item.get('route')}` · "
                f"score={float(item.get('score') or 0):.4f} · "
                f"rrf={float(item.get('rrf_score') or 0):.4f}"
            )
            if item.get("item_type") == "image" and item.get("image_url"):
                url = absolute_url(item.get("image_url"))
                if url:
                    st.image(url, width=480)
            st.write(item.get("text") or "（无文本）")
            if item.get("route_ranks"):
                st.caption("各路原始名次：" + json.dumps(item["route_ranks"], ensure_ascii=False))


# --------------------------------------------------------------------------------------
# 侧边栏
# --------------------------------------------------------------------------------------


def render_sidebar(
    kbs: list[dict[str, Any]], snapshot: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    st.sidebar.title("mmkb")
    st.sidebar.caption("通用多模态知识库问答")

    render_health(snapshot)

    st.sidebar.divider()
    st.sidebar.subheader("知识库")

    if kbs:
        by_id = {kb["kb_id"]: kb for kb in kbs}
        ids = list(by_id)
        current = st.session_state.get("kb_id")
        index = ids.index(current) if current in by_id else 0
        chosen = st.sidebar.selectbox(
            "当前知识库",
            options=ids,
            index=index,
            format_func=lambda kb_id: (
                f"{by_id[kb_id]['name']} · {kb_id[:8]}"
                + ("  (deleting)" if by_id[kb_id].get("status") == "deleting" else "")
            ),
            key="kb_picker",
        )
        if chosen != st.session_state.get("kb_id"):
            st.session_state.kb_id = chosen
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.last_meta = {}
            st.session_state.last_done = {}
    else:
        st.sidebar.info("还没有知识库。先创建一个。")
        st.session_state.kb_id = None

    with st.sidebar.expander("新建知识库"):
        name = st.text_input("名称", key="new_kb_name", max_chars=64)
        description = st.text_area("描述", key="new_kb_desc", max_chars=500)
        if st.button("创建", key="create_kb", disabled=not name.strip()):
            try:
                created = _request(
                    "POST",
                    "/api/kbs",
                    json={"name": name.strip(), "description": description.strip()},
                )
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.session_state.kb_id = created["kb_id"]
                st.session_state.session_id = None
                st.session_state.messages = []
                st.success(f"已创建 `{created['kb_id']}`")
                st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("问答参数")
    top_k = st.sidebar.slider("检索条数 top_k", min_value=1, max_value=30, value=10)
    rerank_mode = st.sidebar.selectbox("重排", ["按配置", "开启", "关闭"], index=0)
    rewrite = st.sidebar.checkbox("查询改写", value=False)
    settings = {
        "top_k": int(top_k),
        "use_rerank": {"按配置": None, "开启": True, "关闭": False}[rerank_mode],
        "rewrite": bool(rewrite),
    }

    if snapshot:
        with st.sidebar.expander("服务详情"):
            st.json(
                {
                    "status": snapshot.get("status"),
                    "capabilities": snapshot.get("capabilities"),
                    "stores": {
                        name: {
                            key: value
                            for key, value in (store or {}).items()
                            if key in {"ok", "count", "path", "error", "backend"}
                        }
                        for name, store in (snapshot.get("stores") or {}).items()
                    },
                    "tasks": snapshot.get("tasks"),
                }
            )

    st.sidebar.caption(
        "前端只调用 HTTP API；后端地址由环境变量 `MMKB_API_BASE` 指定。"
    )
    selected = next((kb for kb in kbs if kb["kb_id"] == st.session_state.get("kb_id")), None)
    return selected, settings


def init_state() -> None:
    defaults: dict[str, Any] = {
        "kb_id": None,
        "session_id": None,
        "messages": [],
        "last_meta": {},
        "last_done": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(
        page_title="mmkb · 多模态知识库问答",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()

    st.title("📚 mmkb")
    st.caption(
        "文档 + 图片双模态入库 · 文本向量 / 多模态向量 / BM25 三路混合检索 · "
        "RRF 融合 · 重排 · 可追溯引用"
    )

    snapshot = fetch_health()
    if snapshot is None:
        st.error(
            f"连接不上后端 `{API_BASE}`。\n\n"
            "先在项目根目录启动服务：\n\n"
            "```bash\nuvicorn app.main:app --host 127.0.0.1 --port 8000\n```"
        )
        st.stop()

    kbs = fetch_kbs()
    kb, settings = render_sidebar(kbs, snapshot)

    if kb is None:
        st.info("先在左侧新建一个知识库，或在「知识库」页导入示例文档。")
        st.stop()
    if kb.get("status") == "deleting":
        st.error("这个知识库正在删除，暂时不能使用。")
        st.stop()

    chat_tab, kb_tab, diagnostics_tab, performance_tab = st.tabs(
        ["💬 问答", "📄 知识库", "🔍 检索诊断", "⚡ 性能"]
    )
    with chat_tab:
        page_chat(kb, settings)
    with kb_tab:
        page_knowledge_base(kb)
    with diagnostics_tab:
        page_diagnostics(kb)
    with performance_tab:
        page_performance()


if __name__ == "__main__":
    main()
