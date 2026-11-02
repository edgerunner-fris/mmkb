"""性能层测试：指标登记表、进程内缓存、报告端点。

这里刻意不去 mock ``time.monotonic`` 之外的东西：性能相关的 bug 往往
不是「算错了」，而是「缓存把不该复用的东西复用了」「重置后计数器还留着」
「报告口径与页面不一致」。这些都能用确定性的小样本直接验证。

HTTP 部分沿用 ``test_api`` 的思路：临时目录 + 空 Key，完全不碰网络。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

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
from app.core.cache import TTLCache, cache_key
from app.core.perf import MetricsRegistry, get_metrics, reset_metrics
from app.main import create_app
from app.services.perf_service import PerformanceService

# --------------------------------------------------------------------------------------
# 指标登记表
# --------------------------------------------------------------------------------------


def test_percentiles_use_nearest_rank() -> None:
    registry = MetricsRegistry()
    for value in range(1, 101):
        registry.observe("demo", value)

    summary = registry.summary("demo")
    assert summary is not None
    assert summary.count == 100
    # 最近秩：第 50 个样本是 50，第 95 个是 95，不做插值
    assert summary.p50_ms == 50
    assert summary.p95_ms == 95
    assert summary.p99_ms == 99
    assert summary.min_ms == 1
    assert summary.max_ms == 100
    assert summary.mean_ms == pytest.approx(50.5)


def test_error_samples_and_counters_are_separate() -> None:
    registry = MetricsRegistry()
    registry.observe("call", 10)
    registry.observe("call", 30, error=True)
    registry.incr("requests", 3)

    summary = registry.summary("call")
    assert summary is not None
    assert summary.count == 2
    assert summary.errors == 1
    assert summary.error_rate == pytest.approx(0.5)

    snapshot = registry.snapshot()
    assert snapshot["counters"] == {"requests": 3}
    assert snapshot["metrics"]["call"]["errors"] == 1


def test_window_keeps_only_recent_samples_but_count_keeps_growing() -> None:
    registry = MetricsRegistry(max_samples=3)
    for value in (1, 2, 3, 4, 5):
        registry.observe("demo", value)

    summary = registry.summary("demo")
    assert summary is not None
    # 计数是累计值；分位数只来自最近 3 个样本（3/4/5）
    assert summary.count == 5
    assert summary.min_ms == 3
    assert summary.max_ms == 5
    assert summary.p50_ms == 4


def test_reset_clears_samples_and_counters() -> None:
    registry = MetricsRegistry()
    registry.observe("demo", 12)
    registry.incr("demo_count")
    registry.reset()

    snapshot = registry.snapshot()
    assert snapshot["metrics"] == {}
    assert snapshot["counters"] == {}


# --------------------------------------------------------------------------------------
# TTL + LRU 缓存
# --------------------------------------------------------------------------------------


def test_cache_hit_and_miss_are_counted() -> None:
    cache: TTLCache[str] = TTLCache("demo", ttl_s=60, max_entries=4)
    cache.set("a", "value")

    assert cache.get("a") == "value"
    assert cache.get("missing") is None
    stats = cache.stats()
    assert (stats.hits, stats.misses, stats.lookups) == (1, 1, 2)
    assert stats.hit_rate == pytest.approx(0.5)


def test_cache_expires_and_counts_expiration() -> None:
    cache: TTLCache[str] = TTLCache("demo", ttl_s=0.01)
    cache.set("a", "value")
    time.sleep(0.03)

    assert cache.get("a") is None
    stats = cache.stats()
    assert stats.expirations == 1
    assert stats.size == 0


def test_cache_evicts_least_recently_used() -> None:
    cache: TTLCache[str] = TTLCache("demo", ttl_s=60, max_entries=2)
    cache.set("a", "1")
    cache.set("b", "2")
    assert cache.get("a") == "1"  # a 变成最近使用
    cache.set("c", "3")

    assert cache.get("b") is None
    assert cache.get("a") == "1"
    assert cache.get("c") == "3"
    assert cache.stats().evictions == 1


def test_disabled_cache_never_returns_values() -> None:
    cache: TTLCache[str] = TTLCache("demo", ttl_s=60, enabled=False)
    cache.set("a", "value")

    assert cache.get("a") is None
    # 关闭时连 miss 都不统计：否则报告里会出现一个永远 0% 的假缓存
    assert cache.stats().lookups == 0


def test_cache_key_is_length_prefixed() -> None:
    # 用分隔符拼接时 ("ab","c") 与 ("a","bc") 会撞键；长度前缀避免这一点
    assert cache_key("ab", "c") != cache_key("a", "bc")
    assert cache_key("ab", "c") == cache_key("ab", "c")


def test_prefix_invalidation_and_clear() -> None:
    cache: TTLCache[str] = TTLCache("demo", ttl_s=60)
    cache.set("ns:1", "a")
    cache.set("ns:2", "b")
    cache.set("other", "c")

    assert cache.invalidate_prefix("ns:") == 2
    assert cache.get("ns:1") is None
    assert cache.get("other") == "c"

    cache.clear()
    assert cache.stats().size == 0


# --------------------------------------------------------------------------------------
# 报告服务（纯渲染）
# --------------------------------------------------------------------------------------


def _fake_container(tmp_path: Path) -> Any:
    """只包含 PerformanceService 会读的字段，避免为了渲染报告启动整个应用。"""

    class _DocStore:
        async def stats(self) -> dict[str, Any]:
            return {"url": "sqlite:///:memory:", "documents": 2, "chunks": 7}

        async def list_kbs(self, *, include_deleting: bool = False) -> list[Any]:
            return [
                type(
                    "KB",
                    (),
                    {
                        "kb_id": "kb_1",
                        "name": "演示库",
                        "status": "active",
                        "doc_count": 2,
                        "chunk_count": 7,
                        "image_count": 1,
                        "text_collection": "kb_1__text",
                        "image_collection": "kb_1__image",
                    },
                )()
            ]

        async def recent_tasks(self, *, limit: int = 10) -> list[Any]:
            return []

    class _VectorStore:
        async def stats(self) -> dict[str, Any]:
            return {"backend": "chroma", "kb_1__text": {"count": 7}}

    class _Retriever:
        def cache_stats(self) -> dict[str, Any]:
            return {
                "dense_text": {
                    "query_embedding": {
                        "hits": 3,
                        "misses": 1,
                        "lookups": 4,
                        "hit_rate": 0.75,
                        "size": 1,
                        "max_entries": 8,
                        "ttl_s": 60.0,
                    }
                }
            }

        def clear_caches(self) -> list[str]:
            return ["dense_text"]

    class _Ingest:
        active_task_count = 0

    class _BM25:
        def stats(self) -> dict[str, Any]:
            return {"docs": 7}

    class _Blob:
        def stats(self) -> dict[str, Any]:
            return {"images": 1}

    settings = type(
        "Settings",
        (),
        {
            "retrieval": type(
                "R",
                (),
                {
                    "enable_bm25": True,
                    "enable_dense_text": True,
                    "enable_dense_image": True,
                    "enable_rerank": False,
                    "top_k_per_route": 20,
                    "rrf_k": 60,
                    "rerank_input_k": 50,
                    "final_top_k": 8,
                    "neighbor_expand": 0,
                },
            )(),
            "ingest": type(
                "I",
                (),
                {"max_concurrent_docs": 1, "image_inspect_concurrency": 4, "max_file_mb": 200},
            )(),
            "cache": type(
                "C",
                (),
                {
                    "enabled": True,
                    "query_embedding_ttl_s": 600.0,
                    "query_embedding_max_entries": 512,
                    "rerank_ttl_s": 300.0,
                    "rerank_max_entries": 256,
                    "collection_meta_ttl_s": 30.0,
                },
            )(),
        },
    )()

    return type(
        "Container",
        (),
        {
            "docstore": _DocStore(),
            "vectorstore": _VectorStore(),
            "bm25store": _BM25(),
            "blobstore": _Blob(),
            "ingest": _Ingest(),
            "retriever": _Retriever(),
            "provider_matrix": {},
            "settings": settings,
        },
    )()


@pytest.mark.asyncio
async def test_snapshot_and_markdown_report(tmp_path: Path) -> None:
    reset_metrics()
    get_metrics().observe("retrieval.total", 120)
    get_metrics().observe("retrieval.retrieve", 80)
    get_metrics().observe("retrieval.fuse", 1)
    get_metrics().incr("retrieval.requests")
    get_metrics().incr("retrieval.empty")

    service = PerformanceService(_fake_container(tmp_path))
    snapshot = await service.snapshot()

    assert snapshot["summary"]["retrieval"]["count"] == 1
    assert snapshot["caches"]["totals"]["hit_rate"] == pytest.approx(0.75)
    assert snapshot["knowledge_bases"][0]["kb_id"] == "kb_1"
    assert snapshot["stores"]["database"]["documents"] == 2
    # 空结果比例应当被自动判读出来
    assert any("空结果" in item for item in snapshot["findings"])

    markdown = service.render_markdown(snapshot)
    assert markdown.startswith("# mmkb 性能报告")
    assert "检索链路拆解" in markdown
    assert "缓存命中率" in markdown
    assert "dense_text.query_embedding" in markdown

    # 报告与 JSON 快照来自同一个 snapshot，不应出现两份口径
    assert snapshot["breakdowns"]["retrieval_stages"][1]["count"] == 1


def test_clear_caches_returns_component_names(tmp_path: Path) -> None:
    service = PerformanceService(_fake_container(tmp_path))
    assert service.clear_caches() == ["dense_text"]


# --------------------------------------------------------------------------------------
# HTTP 端点
# --------------------------------------------------------------------------------------


def _settings(tmp_path: Path) -> Settings:
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
    app = create_app(_settings(tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def test_performance_endpoints_are_wired(client: TestClient) -> None:
    reset_metrics()
    response = client.get("/api/performance")
    assert response.status_code == 200, response.text
    body = response.json()
    for key in ("metrics", "counters", "summary", "breakdowns", "caches", "findings"):
        assert key in body
    assert "markdown" not in body, "JSON 快照不应内嵌整份 Markdown，否则响应翻倍"

    report = client.get("/api/performance/report")
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/markdown")
    assert "# mmkb 性能报告" in report.text

    cleared = client.post("/api/performance/cache/clear")
    assert cleared.status_code == 200
    assert cleared.json()["ok"] is True
    assert "vectorstore" in cleared.json()["cleared"]

    reset = client.post("/api/performance/reset")
    assert reset.status_code == 200
    assert reset.json()["ok"] is True


def test_ingest_and_search_emit_metrics_and_stage_timings(client: TestClient) -> None:
    reset_metrics()
    kb = client.post("/api/kbs", json={"name": "性能测试"}).json()
    kb_id = kb["kb_id"]

    response = client.post(
        f"/api/kbs/{kb_id}/ingest",
        files={
            "files": (
                "perf.txt",
                "性能报告需要能看到解析、分块、向量化与落库各花了多少时间。".encode(),
                "text/plain",
            )
        },
        data={"embed": "false", "caption": "false"},
    )
    assert response.status_code == 202, response.text
    task_id = response.json()["task_id"]

    deadline = time.monotonic() + 15
    task: dict[str, Any] = {}
    while time.monotonic() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in {"done", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert task.get("status") == "done", task

    stage_timings = task["result"].get("stage_timings") or {}
    assert task["stage_timings"] == stage_timings
    assert {"parse", "chunk", "commit"} <= set(stage_timings)
    assert task["result"]["elapsed_ms"] >= 0

    search = client.post(
        f"/api/kbs/{kb_id}/search",
        json={"query": "性能报告", "routes": ["bm25"], "top_k": 3},
    )
    assert search.status_code == 200, search.text

    snapshot = client.get("/api/performance").json()
    assert "ingest.file.total" in snapshot["metrics"]
    assert "ingest.stage.parse" in snapshot["metrics"]
    assert "retrieval.total" in snapshot["metrics"]
    assert "bm25.search" in snapshot["metrics"]
    assert snapshot["counters"].get("retrieval.requests", 0) >= 1
    assert snapshot["counters"].get("ingest.files.done", 0) >= 1

    stages = {row["key"]: row for row in snapshot["breakdowns"]["ingest_stages"]}
    assert stages["parse"]["count"] >= 1
    assert snapshot["ingest"]["recent_tasks"], "最近任务表应能看到刚结束的导入"
