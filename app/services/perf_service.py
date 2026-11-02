"""性能报告服务：把进程内指标、缓存命中率与存储规模拼成一份可读的报告。

为什么报告要由服务层生成
------------------------
路由函数只负责「把请求交给谁」。把表格渲染、指标挑选、TOP 慢查询排序写进
``routes_perf.py``，会让同一条链路出现两份逻辑：JSON 快照一份、Markdown
报告再一份，两者的字段迟早会对不上。这里统一在 :meth:`PerformanceService.snapshot`
里构造**一份**结构化快照，Markdown 只是它的一个渲染视图。

指标口径
--------
* 耗时单位全部是毫秒；``count`` 是样本数，不是请求数（一次检索里
  ``retrieval.retrieve`` 只有一条样本，但它内部可能并发跑了 9 个任务）。
* ``retrieval.total`` 是墙钟时间；各阶段之和与之接近，但**不应**把阶段
  耗时相加当作总耗时对外宣称 —— 并发召回让「三路之和」远大于实际等待。
* 导入阶段耗时按文件累加，任务总耗时是墙钟时间；两者比值就是并发收益。
  报告里同时给出，避免把「CPU 总工作量」误读成「用户等了这么久」。
* 进程重启即清零。这是刻意的：冷启动样本混进热路径会让 p95 永远难看。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger
from app.core.perf import get_metrics, reset_metrics

if TYPE_CHECKING:
    from app.services.container import AppContainer

__all__ = ["PerformanceService"]

logger = get_logger(__name__)

#: 总览表里的链路。顺序就是用户排查问题的顺序：先看端到端，再往下钻。
_SUMMARY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("http", "HTTP 请求", "http.request"),
    ("qa", "端到端问答", "qa.total"),
    ("retrieval", "检索链路", "retrieval.total"),
    ("rerank", "重排（含缓存命中）", "rerank.total"),
    ("generation", "答案生成", "generation.answer"),
    ("ingest_file", "单文件导入", "ingest.file.total"),
    ("ingest_task", "批量导入任务", "ingest.task.total"),
    ("bm25", "BM25 检索", "bm25.search"),
    ("vector", "向量库查询", "vectorstore.query"),
)

#: 检索阶段。``retrieve`` 内部已经并发，因此它是「三路召回的实际等待」。
_RETRIEVAL_STAGES: tuple[tuple[str, str], ...] = (
    ("plan", "规划（路由探测）"),
    ("retrieve", "三路召回（并发）"),
    ("fuse", "RRF 融合"),
    ("rerank", "重排"),
    ("expand", "邻居扩展"),
    ("hydrate", "回填展示字段"),
)

#: 单文件导入阶段。按流水线顺序排列，与 ``FileOutcome.stage_timings`` 一致。
_INGEST_STAGES: tuple[tuple[str, str], ...] = (
    ("hash", "哈希 / 去重"),
    ("parse", "解析"),
    ("caption", "图片描述（VLM）"),
    ("chunk", "分块"),
    ("embed", "向量化（文本 + 图像）"),
    ("commit", "写入存储"),
    ("stats", "知识库统计"),
)

_ROUTES: tuple[tuple[str, str], ...] = (
    ("bm25", "BM25"),
    ("dense_text", "文本向量"),
    ("dense_image", "多模态向量"),
)


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------


async def _safe(awaitable: Any, fallback: Any, label: str) -> Any:
    """读统计失败不该让性能页面 500 —— 它本身就是用来排查故障的。"""
    try:
        return await awaitable
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s失败", label, extra={"error": str(exc)[:200]})
        return fallback


def _fmt_ms(value: float | int | None, *, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _cell(value: Any) -> str:
    """Markdown 表格单元格：管道符与换行会破坏表格结构。"""
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_暂无数据_"
    lines = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def _metric_rows(
    metrics: dict[str, Any],
    specs: tuple[tuple[str, str, str], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, label, name in specs:
        item = metrics.get(name)
        if item is None:
            rows.append({"key": key, "label": label, "metric": name, "count": 0})
            continue
        rows.append({"key": key, "label": label, "metric": name, **item})
    return rows


def _stage_rows(
    metrics: dict[str, Any],
    prefix: str,
    stages: tuple[tuple[str, str], ...],
    denominator: str,
) -> list[dict[str, Any]]:
    """把阶段耗时整理成可直接画图的表。

    占比用 ``mean_ms / 总耗时 mean``：p95 之间不可相加（每个阶段的
    95 分位不一定发生在同一次请求里），而均值近似可加，作为占比更诚实。
    """
    total_mean = float((metrics.get(denominator) or {}).get("mean_ms") or 0.0)
    rows: list[dict[str, Any]] = []
    for key, label in stages:
        item = metrics.get(f"{prefix}{key}")
        if item is None:
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "metric": f"{prefix}{key}",
                    "count": 0,
                    "mean_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "share_pct": None,
                }
            )
            continue
        mean_ms = float(item.get("mean_ms") or 0.0)
        rows.append(
            {
                "key": key,
                "label": label,
                "metric": f"{prefix}{key}",
                "count": item.get("count", 0),
                "mean_ms": mean_ms,
                "p50_ms": item.get("p50_ms", 0.0),
                "p95_ms": item.get("p95_ms", 0.0),
                "p99_ms": item.get("p99_ms", 0.0),
                "share_pct": (
                    round(mean_ms / total_mean * 100, 1) if total_mean > 0 else None
                ),
            }
        )
    return rows


def _flatten_caches(caches: dict[str, Any]) -> list[dict[str, Any]]:
    """把嵌套的 ``cache_stats()`` 压平成一张表。

    只保留带 ``hits`` 的字典 —— ``collection_handles`` 这类计数没有命中率，
    塞进缓存命中率表只会制造一个空行。
    """
    rows: list[dict[str, Any]] = []

    def walk(path: str, value: Any) -> None:
        if not isinstance(value, dict):
            return
        if "hits" in value and "lookups" in value:
            rows.append({"name": path, **value})
            return
        for key, child in value.items():
            walk(f"{path}.{key}" if path else str(key), child)

    for key, value in caches.items():
        walk(str(key), value)
    return rows


def _cache_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(int(row.get("hits") or 0) for row in rows)
    lookups = sum(int(row.get("lookups") or 0) for row in rows)
    return {
        "hits": hits,
        "lookups": lookups,
        "hit_rate": round(hits / lookups, 4) if lookups else 0.0,
        "entries": sum(int(row.get("size") or 0) for row in rows),
    }


# --------------------------------------------------------------------------------------
# 服务
# --------------------------------------------------------------------------------------


class PerformanceService:
    """性能快照 + Markdown 报告 + 指标/缓存重置。"""

    def __init__(self, container: AppContainer) -> None:
        self._c = container

    # ------------------------------------------------------------------ 快照

    async def snapshot(self) -> dict[str, Any]:
        registry = get_metrics()
        raw = registry.snapshot()
        metrics: dict[str, Any] = raw["metrics"]
        counters: dict[str, Any] = raw["counters"]

        database, vector_stats, kbs, recent_tasks = await asyncio.gather(
            _safe(self._c.docstore.stats(), {"error": "无法读取数据库统计"}, "读取数据库统计"),
            _safe(
                self._c.vectorstore.stats(),
                {"error": "无法读取向量库统计"},
                "读取向量库统计",
            ),
            _safe(
                self._c.docstore.list_kbs(include_deleting=True),
                [],
                "读取知识库列表",
            ),
            _safe(
                self._c.docstore.recent_tasks(limit=10),
                [],
                "读取最近任务",
            ),
        )

        try:
            caches = self._c.retriever.cache_stats()
        except Exception as exc:  # noqa: BLE001 - 缓存统计失败不影响其余报告
            logger.warning("读取缓存统计失败", extra={"error": str(exc)[:200]})
            caches = {"error": f"{type(exc).__name__}: {exc}"}
        cache_rows = _flatten_caches(caches)
        cache_totals = _cache_totals(cache_rows)

        retrieval_stages = _stage_rows(
            metrics, "retrieval.", _RETRIEVAL_STAGES, "retrieval.total"
        )
        ingest_stages = _stage_rows(
            metrics, "ingest.stage.", _INGEST_STAGES, "ingest.file.total"
        )
        route_rows = self._route_rows(metrics, counters)
        http_rows = self._http_rows(metrics, counters)

        summary_rows = _metric_rows(metrics, _SUMMARY_ROWS)
        summary = {row["key"]: row for row in summary_rows}

        snapshot: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "started_at": raw["started_at"],
            "uptime_s": raw["uptime_s"],
            "window_samples": raw["window_samples"],
            "metrics": metrics,
            "counters": counters,
            "summary": summary,
            "breakdowns": {
                "retrieval_stages": retrieval_stages,
                "ingest_stages": ingest_stages,
                "routes": route_rows,
                "http_routes": http_rows,
            },
            "caches": {
                "raw": caches,
                "rows": cache_rows,
                "totals": cache_totals,
            },
            "stores": {
                "database": database,
                "vectorstore": vector_stats,
                "bm25": self._bm25_stats(),
                "blobstore": self._blob_stats(),
            },
            "knowledge_bases": [
                {
                    "kb_id": kb.kb_id,
                    "name": kb.name,
                    "status": kb.status,
                    "doc_count": kb.doc_count,
                    "chunk_count": kb.chunk_count,
                    "image_count": kb.image_count,
                    "text_collection": kb.text_collection,
                    "image_collection": kb.image_collection,
                }
                for kb in kbs
            ],
            "ingest": {
                "running_tasks": self._c.ingest.active_task_count,
                "recent_tasks": [self._task_row(task) for task in recent_tasks],
            },
            "providers": self._c.provider_matrix,
            "config": self._config_snapshot(),
        }
        snapshot["findings"] = self._findings(snapshot)
        return snapshot

    # ------------------------------------------------------------------ 重置

    def reset_metrics(self) -> None:
        reset_metrics()

    def clear_caches(self) -> list[str]:
        """清空检索链路缓存。返回实际清理过的部件名，便于前端展示。"""
        clear = getattr(self._c.retriever, "clear_caches", None)
        if not callable(clear):
            return []
        cleared = clear()
        return [str(item) for item in cleared]

    # ------------------------------------------------------------------ 取数

    def _route_rows(
        self, metrics: dict[str, Any], counters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for route, label in _ROUTES:
            item = metrics.get(f"retrieval.route.{route}") or {}
            rows.append(
                {
                    "route": route,
                    "label": label,
                    "count": item.get("count", 0),
                    "errors": item.get("errors", 0),
                    "p50_ms": item.get("p50_ms", 0.0),
                    "p95_ms": item.get("p95_ms", 0.0),
                    "calls": counters.get(f"retrieval.route_calls.{route}", 0),
                    "route_errors": counters.get(f"retrieval.route_errors.{route}", 0),
                }
            )
        return rows

    def _http_rows(
        self, metrics: dict[str, Any], counters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        prefix = "http.route."
        rows = [
            {
                "route": name[len(prefix) :],
                **item,
            }
            for name, item in metrics.items()
            if name.startswith(prefix)
        ]
        rows.sort(key=lambda row: float(row.get("p95_ms") or 0.0), reverse=True)
        status = {
            name[len("http.status.") :]: value
            for name, value in counters.items()
            if name.startswith("http.status.")
        }
        if status:
            rows.append({"route": "（状态码分布）", "status": status})
        return rows

    def _bm25_stats(self) -> dict[str, Any]:
        try:
            return self._c.bm25store.stats()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _blob_stats(self) -> dict[str, Any]:
        try:
            return self._c.blobstore.stats()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _task_row(task: Any) -> dict[str, Any]:
        result = task.result or {}
        return {
            "task_id": task.task_id,
            "kb_id": task.kb_id,
            "status": str(task.status),
            "files": result.get("files", task.progress_total),
            "elapsed_ms": result.get("elapsed_ms"),
            "stage_timings": dict(task.stage_timings or {}),
            "created_at": task.created_at.isoformat(),
        }

    def _config_snapshot(self) -> dict[str, Any]:
        settings = self._c.settings
        return {
            "retrieval": {
                "enable_bm25": settings.retrieval.enable_bm25,
                "enable_dense_text": settings.retrieval.enable_dense_text,
                "enable_dense_image": settings.retrieval.enable_dense_image,
                "enable_rerank": settings.retrieval.enable_rerank,
                "top_k_per_route": settings.retrieval.top_k_per_route,
                "rrf_k": settings.retrieval.rrf_k,
                "rerank_input_k": settings.retrieval.rerank_input_k,
                "final_top_k": settings.retrieval.final_top_k,
                "neighbor_expand": settings.retrieval.neighbor_expand,
            },
            "ingest": {
                "max_concurrent_docs": settings.ingest.max_concurrent_docs,
                "image_inspect_concurrency": settings.ingest.image_inspect_concurrency,
                "max_file_mb": settings.ingest.max_file_mb,
            },
            "cache": {
                "enabled": settings.cache.enabled,
                "query_embedding_ttl_s": settings.cache.query_embedding_ttl_s,
                "query_embedding_max_entries": settings.cache.query_embedding_max_entries,
                "rerank_ttl_s": settings.cache.rerank_ttl_s,
                "rerank_max_entries": settings.cache.rerank_max_entries,
                "collection_meta_ttl_s": settings.cache.collection_meta_ttl_s,
            },
        }

    # ------------------------------------------------------------------ 结论

    def _findings(self, snapshot: dict[str, Any]) -> list[str]:
        """把数字翻译成人话。只陈述能从指标直接推出的事实。"""
        metrics = snapshot["metrics"]
        counters = snapshot["counters"]
        findings: list[str] = []

        if not metrics:
            return [
                "当前进程还没有耗时样本。先导入一份文档、问几个问题，再刷新报告。",
            ]

        stages = [row for row in snapshot["breakdowns"]["retrieval_stages"] if row["count"]]
        if stages:
            slowest = max(stages, key=lambda row: float(row.get("p95_ms") or 0.0))
            if slowest.get("p95_ms"):
                findings.append(
                    f"检索链路最慢的阶段是「{slowest['label']}」："
                    f"p95 {_fmt_ms(slowest['p95_ms'])} ms，占平均总耗时的 "
                    f"{slowest['share_pct'] if slowest['share_pct'] is not None else '—'}%。"
                )

        totals = snapshot["caches"]["totals"]
        if totals["lookups"]:
            findings.append(
                f"缓存整体命中率 {_fmt_pct(totals['hit_rate'])}"
                f"（{totals['hits']}/{totals['lookups']} 次查询命中）。"
                "重复提问、多轮追问与评估集回放都会抬高这个数字。"
            )
        elif snapshot["config"]["cache"]["enabled"]:
            findings.append("缓存已启用，但还没有查询经过它。")

        requests = int(counters.get("retrieval.requests", 0) or 0)
        empty = int(counters.get("retrieval.empty", 0) or 0)
        if requests and empty:
            findings.append(
                f"有 {empty}/{requests} 次检索返回空结果"
                f"（{empty / requests * 100:.1f}%）。请检查知识库是否已向量化，"
                "以及对应语言的检索路是否开启。"
            )

        image_failures = int(counters.get("ingest.embed.image_failures", 0) or 0)
        if image_failures:
            findings.append(
                f"有 {image_failures} 次图像向量化失败并以文本向量继续导入。"
                "这些图片将无法被「以图搜图 / 以文搜图」命中，"
                "日志里能看到具体原因。"
            )

        errors = [
            (name, item["errors"])
            for name, item in metrics.items()
            if item.get("errors")
        ]
        if errors:
            detail = "、".join(f"{name}×{count}" for name, count in errors[:5])
            findings.append(f"存在失败样本：{detail}。失败请求也计入延迟，但会被标记为 error。")

        if not findings:
            findings.append("当前没有发现明显的性能问题，继续积累样本即可。")
        return findings

    # ------------------------------------------------------------------ Markdown

    def render_markdown(self, snapshot: dict[str, Any]) -> str:
        """把快照渲染成一份可以直接贴进 README / 面试材料的报告。"""
        metrics = snapshot.get("metrics", {})
        counters = snapshot.get("counters", {})
        summary = snapshot.get("summary", {})
        breakdowns = snapshot.get("breakdowns", {})
        caches = snapshot.get("caches", {})
        config = snapshot.get("config", {})

        lines: list[str] = [
            "# mmkb 性能报告",
            "",
            f"- 生成时间：{snapshot.get('generated_at')}",
            f"- 进程启动：{snapshot.get('started_at')}",
            f"- 已运行：{snapshot.get('uptime_s', 0)} 秒",
            f"- 统计窗口：每个指标最近 {snapshot.get('window_samples', 0)} 个样本",
            "",
            "## 1. 总览",
            "",
        ]

        overview: list[list[Any]] = []
        for row in summary.values():
            count = int(row.get("count") or 0)
            if not count:
                continue
            overview.append(
                [
                    row["label"],
                    count,
                    _fmt_ms(row.get("p50_ms")),
                    _fmt_ms(row.get("p95_ms")),
                    _fmt_ms(row.get("p99_ms")),
                    _fmt_ms(row.get("max_ms")),
                    _fmt_pct(row.get("error_rate")),
                ]
            )
        lines.append(
            _table(
                ["链路", "样本数", "p50 (ms)", "p95 (ms)", "p99 (ms)", "最大 (ms)", "错误率"],
                overview,
            )
        )
        lines += ["", "**自动判读**", ""]
        lines += [f"- {item}" for item in snapshot.get("findings", [])]
        lines += ["", "## 2. 检索链路拆解", ""]

        stage_rows = breakdowns.get("retrieval_stages", [])
        lines.append(
            _table(
                ["阶段", "样本数", "平均 (ms)", "p50 (ms)", "p95 (ms)", "占平均总耗时"],
                [
                    [
                        row["label"],
                        row["count"],
                        _fmt_ms(row.get("mean_ms")),
                        _fmt_ms(row.get("p50_ms")),
                        _fmt_ms(row.get("p95_ms")),
                        f"{row['share_pct']}%" if row.get("share_pct") is not None else "—",
                    ]
                    for row in stage_rows
                ],
            )
        )
        total = metrics.get("retrieval.total") or {}
        if total:
            lines += [
                "",
                f"检索总耗时（墙钟）：平均 {_fmt_ms(total.get('mean_ms'))} ms，"
                f"p95 {_fmt_ms(total.get('p95_ms'))} ms。"
                "「三路召回」内部并发执行，因此不要把各阶段 p95 相加当作总耗时。",
            ]

        lines += ["", "## 3. 三路召回", ""]
        route_rows = breakdowns.get("routes", [])
        lines.append(
            _table(
                ["召回路", "成功调用", "失败调用", "p50 (ms)", "p95 (ms)"],
                [
                    [
                        row["label"],
                        row["calls"],
                        row["route_errors"],
                        _fmt_ms(row.get("p50_ms")),
                        _fmt_ms(row.get("p95_ms")),
                    ]
                    for row in route_rows
                ],
            )
        )
        lines += [
            "",
            "说明：BM25 是进程内内存索引；文本向量与多模态向量各走外部 API，"
            "二者不在同一个向量空间，禁止混用（见 README「三条铁律」）。",
            "",
        ]

        lines += ["## 4. 缓存命中率", ""]
        cache_rows = caches.get("rows", [])
        lines.append(
            _table(
                ["缓存", "命中", "查询", "命中率", "条目", "容量", "TTL (s)"],
                [
                    [
                        row["name"],
                        row["hits"],
                        row["lookups"],
                        _fmt_pct(row.get("hit_rate")),
                        row.get("size", 0),
                        row.get("max_entries", 0),
                        row.get("ttl_s", 0),
                    ]
                    for row in cache_rows
                ],
            )
        )
        cache_totals = caches.get("totals") or {}
        if cache_totals.get("lookups"):
            lines += [
                "",
                f"整体命中率 {_fmt_pct(cache_totals.get('hit_rate'))}"
                f"（{cache_totals.get('hits')}/{cache_totals.get('lookups')}）。"
                "缓存键包含 provider / model / dim / space / fusion 开关，"
                "因此换模型后不会错误复用旧向量。",
            ]
        lines += [
            "",
            "缓存只影响延迟，不影响正确性：``CACHE_ENABLED=false`` 后所有路径"
            "与未加缓存时完全一致；写入时主动失效，TTL 作为跨进程兜底。",
            "",
        ]

        ingest_rows = breakdowns.get("ingest_stages", [])
        lines += ["## 5. 导入阶段拆解", ""]
        lines.append(
            _table(
                ["阶段", "样本数", "平均 (ms)", "p50 (ms)", "p95 (ms)", "占单文件平均耗时"],
                [
                    [
                        row["label"],
                        row["count"],
                        _fmt_ms(row.get("mean_ms")),
                        _fmt_ms(row.get("p50_ms")),
                        _fmt_ms(row.get("p95_ms")),
                        f"{row['share_pct']}%" if row.get("share_pct") is not None else "—",
                    ]
                    for row in ingest_rows
                ],
            )
        )
        file_total = metrics.get("ingest.file.total") or {}
        if file_total:
            lines += [
                "",
                f"单文件导入平均 {_fmt_ms(file_total.get('mean_ms'))} ms，"
                f"p95 {_fmt_ms(file_total.get('p95_ms'))} ms。",
                "文本向量与图像向量在 ``embed`` 阶段并发执行；"
                "图片检查按 ``INGEST_IMAGE_INSPECT_CONCURRENCY`` 并发，"
                "caption 仍受 provider 并发上限约束。",
            ]
        lines += [
            "",
            "导入计数："
            f"文件 {counters.get('ingest.files', 0)}，"
            f"完成 {counters.get('ingest.files.done', 0)}，"
            f"重复跳过 {counters.get('ingest.files.duplicate', 0)}，"
            f"失败 {counters.get('ingest.files.failed', 0)}；"
            f"分块 {counters.get('ingest.chunks', 0)}，"
            f"图片 {counters.get('ingest.images', 0)}，"
            f"文本向量 {counters.get('ingest.text_vectors', 0)}，"
            f"图像向量 {counters.get('ingest.image_vectors', 0)}。",
            "",
        ]

        lines += ["## 6. HTTP 接口", ""]
        http_rows = [
            row for row in breakdowns.get("http_routes", []) if "status" not in row
        ]
        lines.append(
            _table(
                ["路由模板", "样本数", "p50 (ms)", "p95 (ms)", "p99 (ms)", "错误率"],
                [
                    [
                        row["route"],
                        row.get("count", 0),
                        _fmt_ms(row.get("p50_ms")),
                        _fmt_ms(row.get("p95_ms")),
                        _fmt_ms(row.get("p99_ms")),
                        _fmt_pct(row.get("error_rate")),
                    ]
                    for row in http_rows
                ],
            )
        )
        statuses = [
            row["status"]
            for row in breakdowns.get("http_routes", [])
            if "status" in row
        ]
        if statuses:
            detail = "，".join(f"{code}: {count}" for code, count in statuses[0].items())
            lines += ["", f"状态码分布：{detail}。"]
        lines += [""]

        kbs = snapshot.get("knowledge_bases", [])
        lines += ["## 7. 数据规模", ""]
        lines.append(
            _table(
                ["知识库", "文档", "分块", "图片", "文本 collection", "图像 collection"],
                [
                    [
                        kb["name"],
                        kb["doc_count"],
                        kb["chunk_count"],
                        kb["image_count"],
                        kb["text_collection"],
                        kb["image_collection"],
                    ]
                    for kb in kbs
                ],
            )
        )
        db = (snapshot.get("stores") or {}).get("database") or {}
        table_counts = {
            key: value
            for key, value in db.items()
            if key not in {"url", "error"} and isinstance(value, int)
        }
        if table_counts:
            lines += [
                "",
                "SQLite 行数："
                + "，".join(f"{key} {value}" for key, value in table_counts.items())
                + "。",
            ]
        lines += [""]

        recent = (snapshot.get("ingest") or {}).get("recent_tasks", [])
        lines += ["## 8. 最近导入任务", ""]
        lines.append(
            _table(
                ["任务", "知识库", "状态", "文件数", "累计耗时 (ms)", "阶段耗时 (ms)"],
                [
                    [
                        task["task_id"],
                        task["kb_id"],
                        task["status"],
                        task["files"],
                        _fmt_ms(task.get("elapsed_ms"), digits=0)
                        if task.get("elapsed_ms") is not None
                        else "—",
                        ", ".join(
                            f"{key}={value}"
                            for key, value in (task.get("stage_timings") or {}).items()
                        )
                        or "—",
                    ]
                    for task in recent
                ],
            )
        )
        lines += [
            "",
            "累计耗时是各文件耗时之和（并发导入的总工作量），"
            "可能大于任务的墙钟耗时。",
            "",
        ]

        lines += ["## 9. 当前配置", ""]
        retrieval_cfg = config.get("retrieval", {})
        cache_cfg = config.get("cache", {})
        ingest_cfg = config.get("ingest", {})
        lines += [
            f"- 检索：BM25={retrieval_cfg.get('enable_bm25')}，"
            f"文本向量={retrieval_cfg.get('enable_dense_text')}，"
            f"多模态向量={retrieval_cfg.get('enable_dense_image')}，"
            f"重排={retrieval_cfg.get('enable_rerank')}",
            f"- 召回：每路 {retrieval_cfg.get('top_k_per_route')} 条，"
            f"RRF k={retrieval_cfg.get('rrf_k')}，"
            f"重排输入 {retrieval_cfg.get('rerank_input_k')} 条，"
            f"最终 {retrieval_cfg.get('final_top_k')} 条，"
            f"邻居扩展 ±{retrieval_cfg.get('neighbor_expand')}",
            f"- 缓存：enabled={cache_cfg.get('enabled')}，"
            f"query TTL={cache_cfg.get('query_embedding_ttl_s')}s，"
            f"rerank TTL={cache_cfg.get('rerank_ttl_s')}s，"
            f"collection 元信息 TTL={cache_cfg.get('collection_meta_ttl_s')}s",
            f"- 导入：文档并发 {ingest_cfg.get('max_concurrent_docs')}，"
            f"图片检查并发 {ingest_cfg.get('image_inspect_concurrency')}，"
            f"单文件上限 {ingest_cfg.get('max_file_mb')} MB",
            "",
            "---",
            "",
            "指标是**进程内**的，服务重启即清零。"
            "需要跨进程、跨时间的趋势时，再把同一份快照定期写入时序库；"
            "当前 demo 不引入 Prometheus / Grafana，保持零外部依赖。",
            "",
        ]
        return "\n".join(lines)
