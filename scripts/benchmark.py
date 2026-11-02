#!/usr/bin/env python
"""检索与导入性能基准：一次跑完，直接出 Markdown + JSON。

为什么单独做一个脚本
--------------------
``GET /api/performance`` 回答的是「这个进程**到现在**有多快」，适合演示；
但面试/写报告时还需要一组**可复现**的数字：固定 query 集、固定重复次数、
明确区分冷缓存与热缓存。这个脚本就是那组数字的来源。

三条口径
--------
1. **冷 / 热分开**：``--cold`` 在每次测量前清空缓存，测的是 provider 与
   检索本身；不加时缓存会生效，随后重复同一 query 的延迟接近 0。
   两种数字都真实，但不能混在一张表里比较。
2. **墙钟 vs 阶段**：``total_ms`` 是整次检索的墙钟时间，阶段耗时用于解释
   它花在了哪里。三路召回并发执行，阶段分位**不能相加**。
3. **不做 ground truth**：这个脚本只测速度，不生成任何「准确率」。
   召回质量由 ``app.eval.run_eval`` 在人工标注的数据集上计算。

用法::

    # 只看检索，热缓存
    python scripts/benchmark.py --kb demo --queries "三路召回如何融合？" "架构图" --repeat 5

    # 冷缓存：每次测量前清空 query 向量 / 重排 / collection 元信息缓存
    python scripts/benchmark.py --kb demo --queries-file eval/queries.txt --repeat 3 --cold

    # 顺带测一次导入（会自动建一个临时知识库，跑完删除）
    python scripts/benchmark.py --kb demo --queries "RRF" --ingest data/raw/demo.pdf

退出码：``0`` 完成 · ``1`` 有检索/导入失败 · ``2`` 用法或配置问题 · ``130`` 中断。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import ConfigurationError, Settings, get_settings  # noqa: E402
from app.core.errors import MMKBError  # noqa: E402
from app.core.ids import is_valid_id  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.schemas import KnowledgeBase, Route  # noqa: E402
from app.ingestion.pipeline import FileOutcome, IngestOptions  # noqa: E402
from app.services.container import AppContainer  # noqa: E402

__all__ = ["main"]

_ROUTE_LABELS = {
    "bm25": "BM25",
    "dense_text": "文本向量",
    "dense_image": "多模态向量",
}

_STAGE_LABELS = {
    "plan": "规划（路由探测）",
    "retrieve": "三路召回（并发）",
    "fuse": "RRF 融合",
    "rerank": "重排",
    "expand": "邻居扩展",
    "hydrate": "回填展示字段",
    "total": "总计",
}

_INGEST_STAGE_LABELS = {
    "hash": "哈希 / 去重",
    "parse": "解析",
    "caption": "图片描述（VLM）",
    "chunk": "分块",
    "embed": "向量化（文本 + 图像）",
    "commit": "写入存储",
    "stats": "知识库统计",
}


# --------------------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------------------


def _percentile(ordered: list[float], percent: float) -> float:
    """最近秩百分位，与 ``app.core.perf`` / 评测报告保持同一口径。"""
    if not ordered:
        return 0.0
    rank = math.ceil(percent / 100.0 * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "p50_ms": round(_percentile(ordered, 50), 2),
        "p95_ms": round(_percentile(ordered, 95), 2),
        "p99_ms": round(_percentile(ordered, 99), 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}"


def _order_by(labels: dict[str, str], present: list[str]) -> list[str]:
    """按语义顺序排列（规划 → 召回 → 融合 → …），未知键追加在后。

    ``dict`` 的插入顺序来自样本到达顺序，不同查询可能不同；直接 ``sorted``
    又会把「阶段」按拼音排成看似随机的顺序。报告是给人读的，顺序要固定。
    """
    known = [name for name in labels if name in present]
    return known + [name for name in present if name not in labels]


# --------------------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------------------


def _parse_routes(raw: str) -> list[Route] | None:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        return None
    # hybrid 是融合后的结果标记，不能作为召回路传入（与 HybridRetriever 的校验一致）
    allowed = {"bm25", "dense_text", "dense_image"}
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"未知的召回路：{unknown}；可选 {sorted(allowed)}")
    return [Route(name) for name in names]


def _load_queries(args: argparse.Namespace) -> list[str]:
    queries: list[str] = list(args.queries or [])
    if args.queries_file:
        path = Path(args.queries_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"查询文件不存在：{path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and payload.get("query"):
                    text = str(payload["query"])
            queries.append(text)
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        query = query.strip()
        if query and query not in seen:
            seen.add(query)
            unique.append(query)
    return unique


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="检索 / 导入性能基准（输出 Markdown + JSON，不计算准确率）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python scripts/benchmark.py --kb demo --queries "三路召回如何融合？" "架构图" --repeat 5\n'
            "  python scripts/benchmark.py --kb demo --queries-file eval/queries.txt --repeat 3 --cold\n"
            "  python scripts/benchmark.py --kb demo --queries RRF --ingest data/raw/demo.pdf\n\n"
            "冷缓存与热缓存不能混为一谈：--cold 每次测量前清空缓存，"
            "测的是 provider 与检索本身；不加时重复 query 会命中缓存。\n"
            "准确率请用 app.eval.run_eval（需要人工标注的 ground truth）。"
        ),
    )
    parser.add_argument("--kb", help="知识库名称或 kb_ 开头的 ID（--ingest-only 时不需要）")
    parser.add_argument("--queries", nargs="+", default=[], help="一个或多个查询词")
    parser.add_argument("--queries-file", default="", help="每行一个查询；支持 JSONL 的 query 字段")
    parser.add_argument("--top-k", type=int, default=None, help="最终返回条数")
    parser.add_argument("--routes", default="", help="只跑指定路：bm25,dense_text,dense_image")
    parser.add_argument("--no-rerank", action="store_true", help="关闭重排（消融对照）")
    parser.add_argument("--repeat", type=int, default=3, help="每个查询的测量次数（默认 3）")
    parser.add_argument("--warmup", type=int, default=1, help="测量前的预热次数（默认 1，不计入）")
    parser.add_argument(
        "--cold",
        action="store_true",
        help="每次测量前清空检索缓存，测冷路径延迟",
    )
    parser.add_argument(
        "--ingest",
        default="",
        help="额外测一次导入；会自动创建临时知识库并在结束时删除",
    )
    parser.add_argument("--no-embed", action="store_true", help="导入基准跳过向量化")
    parser.add_argument("--no-caption", action="store_true", help="导入基准跳过图片 caption")
    parser.add_argument("--ingest-repeat", type=int, default=1, help="导入重复次数（默认 1）")
    parser.add_argument("--keep-kb", action="store_true", help="导入基准结束后保留临时知识库")
    parser.add_argument("--ingest-only", action="store_true", help="只跑导入基准，不跑检索")
    parser.add_argument("--output", default="data/bench", help="报告输出目录（默认 data/bench）")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印 INFO 日志")
    return parser


# --------------------------------------------------------------------------------------
# 检索基准
# --------------------------------------------------------------------------------------


async def _resolve_kb(container: AppContainer, ref: str) -> KnowledgeBase:
    if is_valid_id(ref, "kb"):
        kb = await container.docstore.get_kb(ref)
        if kb is not None:
            return kb
    for kb in await container.docstore.list_kbs():
        if kb.name == ref:
            return kb
    raise MMKBError(f"知识库不存在：{ref}")


async def _one_search(
    container: AppContainer,
    kb: KnowledgeBase,
    query: str,
    args: argparse.Namespace,
    routes: list[Route] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    outcome = await container.retriever.search(
        kb,
        query,
        top_k=args.top_k,
        routes=routes,
        use_rerank=False if args.no_rerank else None,
        rewrite=False,
        expand_neighbors=False,
    )
    wall_ms = (time.perf_counter() - started) * 1000
    return {
        "query": query,
        "total_ms": round(wall_ms, 2),
        "stages": dict(outcome.timings),
        "routes": {
            str(result.route): int(result.latency_ms)
            for result in outcome.route_results
            if not result.skipped
        },
        "items": len(outcome.items),
        "fused_count": outcome.fused_count,
        "degraded": [str(reason) for reason in outcome.degraded_reasons],
    }


async def _bench_retrieval(
    container: AppContainer,
    kb: KnowledgeBase,
    queries: list[str],
    args: argparse.Namespace,
    routes: list[Route] | None,
) -> dict[str, Any]:
    warmup = max(0, args.warmup)
    repeat = max(1, args.repeat)
    samples: list[dict[str, Any]] = []

    for query in queries:
        for _ in range(warmup):
            await _one_search(container, kb, query, args, routes)
        for _ in range(repeat):
            if args.cold:
                container.retriever.clear_caches()
            samples.append(await _one_search(container, kb, query, args, routes))

    totals = [float(sample["total_ms"]) for sample in samples]
    stage_values: dict[str, list[float]] = {}
    route_values: dict[str, list[float]] = {}
    for sample in samples:
        for name, value in sample["stages"].items():
            stage_values.setdefault(name, []).append(float(value))
        for name, value in sample["routes"].items():
            route_values.setdefault(name, []).append(float(value))

    per_query: list[dict[str, Any]] = []
    for query in queries:
        group = [sample for sample in samples if sample["query"] == query]
        per_query.append(
            {
                "query": query,
                "runs": len(group),
                "latency": _stats([float(sample["total_ms"]) for sample in group]),
                "items": [sample["items"] for sample in group],
                "degraded": sorted({reason for sample in group for reason in sample["degraded"]}),
            }
        )

    errors = sum(1 for sample in samples if sample["degraded"])
    return {
        "queries": len(queries),
        "repeat": repeat,
        "warmup": warmup,
        "cold": bool(args.cold),
        "total": _stats(totals),
        "stages": {
            name: _stats(stage_values[name])
            for name in _order_by(_STAGE_LABELS, list(stage_values))
        },
        "routes": {
            name: _stats(route_values[name])
            for name in _order_by(_ROUTE_LABELS, list(route_values))
        },
        "per_query": per_query,
        "degraded_runs": errors,
        "cache_stats": container.retriever.cache_stats(),
        "samples": samples,
    }


# --------------------------------------------------------------------------------------
# 导入基准
# --------------------------------------------------------------------------------------


def _outcome_row(outcome: FileOutcome) -> dict[str, Any]:
    return {
        "file": outcome.path.name,
        "status": str(outcome.status),
        "ok": outcome.ok,
        "elapsed_ms": outcome.elapsed_ms,
        "chunks": outcome.chunks,
        "images": outcome.images,
        "text_vectors": outcome.text_vectors,
        "image_vectors": outcome.image_vectors,
        "stage_timings": dict(outcome.stage_timings),
        "degraded": list(outcome.degraded_reasons),
        "warnings": list(outcome.warnings[:10]),
        "error": outcome.error,
    }


async def _bench_ingest(
    container: AppContainer, args: argparse.Namespace
) -> dict[str, Any] | None:
    path = Path(args.ingest).expanduser()  # noqa: ASYNC240 - 只读一次本机路径
    if not path.is_file():
        raise FileNotFoundError(f"待导入文件不存在：{path}")

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    kb = await container.kbs.create_kb(f"bench-{stamp}", "benchmark.py 临时知识库")
    runs: list[dict[str, Any]] = []
    try:
        for index in range(max(1, args.ingest_repeat)):
            outcome = await container.pipeline.ingest_file(
                path,
                kb,
                options=IngestOptions(
                    embed=not args.no_embed,
                    caption=not args.no_caption,
                    # 同一文件重复导入会命中去重；reindex 让第二次真正重跑
                    reindex=index > 0,
                ),
            )
            runs.append(_outcome_row(outcome))
            if not outcome.ok:
                break
    finally:
        if not args.keep_kb:
            try:
                await container.kbs.delete_kb(kb.kb_id)
            except Exception as exc:  # noqa: BLE001 - 清理失败不该覆盖基准结果
                print(f"警告：临时知识库删除失败：{exc}", file=sys.stderr)

    ok_runs = [row for row in runs if row["ok"]]
    stage_values: dict[str, list[float]] = {}
    for row in ok_runs:
        for name, value in row["stage_timings"].items():
            stage_values.setdefault(name, []).append(float(value))
    return {
        "file": str(path),
        "kb_id": kb.kb_id,
        "kept": bool(args.keep_kb),
        "runs": runs,
        "total": _stats([float(row["elapsed_ms"]) for row in ok_runs]),
        "stages": {
            name: _stats(stage_values[name])
            for name in _order_by(_INGEST_STAGE_LABELS, list(stage_values))
        },
        "all_ok": len(ok_runs) == len(runs) and bool(runs),
    }


# --------------------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_无数据_"
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def _latency_row(label: str, stats: dict[str, Any]) -> list[Any]:
    return [
        label,
        stats.get("count", 0),
        _fmt(stats.get("p50_ms")),
        _fmt(stats.get("p95_ms")),
        _fmt(stats.get("p99_ms")),
        _fmt(stats.get("mean_ms")),
        _fmt(stats.get("max_ms")),
    ]


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    retrieval = report.get("retrieval")
    ingest = report.get("ingest")
    lines: list[str] = [
        "# mmkb 性能基准",
        "",
        f"- 生成时间：{metadata['generated_at']}",
        f"- 知识库：{metadata.get('kb_name') or '—'} (`{metadata.get('kb_id') or '—'}`)",
        (
            f"- 检索配置：routes={metadata['routes']}，rerank={metadata['rerank']}，"
            f"top_k={metadata['top_k'] or '配置默认'}"
            if retrieval
            else "- 检索配置：未运行（--ingest-only）"
        ),
        (
            f"- 缓存：{'每次测量前清空（冷路径）' if metadata['cold'] else '启用（热路径）'}，"
            f"repeat={metadata['repeat']}，warmup={metadata['warmup']}"
            if retrieval
            else "- 缓存：与导入无关"
        ),
        f"- Provider：{json.dumps(metadata['providers'], ensure_ascii=False)}",
        "",
    ]

    if retrieval:
        lines += ["## 1. 检索总延迟", ""]
        lines.append(
            _table(
                ["指标", "样本数", "p50 (ms)", "p95 (ms)", "p99 (ms)", "平均 (ms)", "最大 (ms)"],
                [_latency_row("全部检索", retrieval["total"])],
            )
        )
        lines += ["", "## 2. 阶段拆解", ""]
        lines.append(
            _table(
                ["阶段", "样本数", "p50 (ms)", "p95 (ms)", "p99 (ms)", "平均 (ms)", "最大 (ms)"],
                [
                    _latency_row(_STAGE_LABELS.get(name, name), stats)
                    for name, stats in retrieval["stages"].items()
                ],
            )
        )
        lines += [
            "",
            "> 三路召回并发执行，阶段耗时不等于墙钟时间的加总；"
            "`total` 行才是真实等待。",
            "",
            "## 3. 分路召回延迟",
            "",
        ]
        lines.append(
            _table(
                ["召回路", "样本数", "p50 (ms)", "p95 (ms)", "p99 (ms)", "平均 (ms)", "最大 (ms)"],
                [
                    _latency_row(_ROUTE_LABELS.get(name, name), stats)
                    for name, stats in retrieval["routes"].items()
                ],
            )
        )
        lines += ["", "## 4. 每次查询明细", ""]
        lines.append(
            _table(
                ["查询", "次数", "p50 (ms)", "p95 (ms)", "返回条目", "降级"],
                [
                    [
                        row["query"],
                        row["runs"],
                        _fmt(row["latency"].get("p50_ms")),
                        _fmt(row["latency"].get("p95_ms")),
                        "/".join(str(count) for count in row["items"]),
                        ", ".join(row["degraded"]) or "—",
                    ]
                    for row in retrieval["per_query"]
                ],
            )
        )
        lines += ["", "## 5. 缓存命中率", ""]
        cache_rows: list[list[Any]] = []

        def add_cache_row(name: str, value: dict[str, Any]) -> None:
            # 一次都没查过的缓存不占一行：报告里应只有真实发生过的命中
            if not value.get("lookups") and not value.get("size"):
                return
            cache_rows.append(
                [
                    name,
                    value.get("hits", 0),
                    value.get("lookups", 0),
                    f"{float(value.get('hit_rate') or 0) * 100:.1f}%",
                    value.get("size", 0),
                ]
            )

        for component, value in retrieval["cache_stats"].items():
            if isinstance(value, dict) and "hits" in value:
                add_cache_row(component, value)
            elif isinstance(value, dict):
                for name, child in value.items():
                    if isinstance(child, dict) and "hits" in child:
                        add_cache_row(f"{component}.{name}", child)
        lines.append(_table(["缓存", "命中", "查询", "命中率", "条目"], cache_rows))
        lines += [""]

    if ingest:
        ingest_no = 6 if retrieval else 1
        lines += [f"## {ingest_no}. 导入基准", ""]
        lines.append(
            _table(
                ["文件", "状态", "耗时 (ms)", "分块", "图片", "文本向量", "图像向量", "错误"],
                [
                    [
                        row["file"],
                        row["status"],
                        _fmt(row["elapsed_ms"], digits=0),
                        row["chunks"],
                        row["images"],
                        row["text_vectors"],
                        row["image_vectors"],
                        row["error"] or "—",
                    ]
                    for row in ingest["runs"]
                ],
            )
        )
        if ingest["stages"]:
            lines += ["", "阶段耗时：", ""]
            lines.append(
                _table(
                    ["阶段", "样本数", "平均 (ms)", "p95 (ms)"],
                    [
                        [
                            _INGEST_STAGE_LABELS.get(name, name),
                            stats.get("count", 0),
                            _fmt(stats.get("mean_ms")),
                            _fmt(stats.get("p95_ms")),
                        ]
                        for name, stats in ingest["stages"].items()
                    ],
                )
            )
        lines += [
            "",
            f"> 临时知识库 `{ingest['kb_id']}` "
            + ("已保留（--keep-kb）。" if ingest["kept"] else "已删除。"),
            "",
        ]

    # 只跑导入时没有第 1–5 节，编号从 1 开始，避免报告里出现「空章节 1–5」
    ingest_no = 6 if retrieval else 1
    notes_no = ingest_no + 1 if ingest else ingest_no
    lines += [
        f"## {notes_no}. 解读与限制",
        "",
        "- 这是**速度**基准，不是准确率基准；召回质量请用 `app.eval.run_eval`。",
        "- 延迟受网络与 provider 限流影响。跨机器对比时，先确认 provider、模型、维度一致。",
        "- 冷路径包含 query 编码（外部 HTTP）；热路径命中进程内缓存，延迟接近纯检索。",
        "- 导入耗时按文件统计；批量导入时文件并发会提高吞吐，但单文件延迟不一定下降。",
        "- 重排模型只接受纯文本，图片候选按 caption + 章节路径参与重排；"
        "`--no-rerank` 的差值包含这部分已知取舍。",
        "",
    ]
    return "\n".join(lines)


def _write_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "benchmark.md"
    json_path = output_dir / "benchmark.json"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return markdown_path, json_path


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


def _metadata(
    settings: Settings,
    kb: KnowledgeBase | None,
    args: argparse.Namespace,
    cache_stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "kb_id": kb.kb_id if kb else None,
        "kb_name": kb.name if kb else None,
        "routes": args.routes or "全部启用路",
        "rerank": not args.no_rerank,
        "top_k": args.top_k,
        "repeat": max(1, args.repeat),
        "warmup": max(0, args.warmup),
        "cold": bool(args.cold),
        "providers": {
            "llm": f"{settings.llm.provider}/{settings.llm.model}",
            "text_embedding": f"{settings.text_embed.provider}/{settings.text_embed.resolved_model}",
            "image_embedding": f"{settings.image_embed.provider}/{settings.image_embed.model}",
            "rerank": f"{settings.rerank.provider}/{settings.rerank.resolved_model}",
        },
        "retrieval": {
            "enable_bm25": settings.retrieval.enable_bm25,
            "enable_dense_text": settings.retrieval.enable_dense_text,
            "enable_dense_image": settings.retrieval.enable_dense_image,
            "enable_rerank": settings.retrieval.enable_rerank,
            "top_k_per_route": settings.retrieval.top_k_per_route,
            "rrf_k": settings.retrieval.rrf_k,
            "rerank_input_k": settings.retrieval.rerank_input_k,
            "final_top_k": settings.retrieval.final_top_k,
        },
        "ingest": {
            "max_concurrent_docs": settings.ingest.max_concurrent_docs,
            "image_inspect_concurrency": settings.ingest.image_inspect_concurrency,
        },
        "cache": {
            "enabled": settings.cache.enabled,
            "query_embedding_ttl_s": settings.cache.query_embedding_ttl_s,
            "rerank_ttl_s": settings.cache.rerank_ttl_s,
            "collection_meta_ttl_s": settings.cache.collection_meta_ttl_s,
        },
        "cache_stats": cache_stats,
    }


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    queries = _load_queries(args)
    if not args.ingest_only and not queries:
        raise ValueError("至少提供一个查询：--queries 或 --queries-file")
    if not args.ingest_only and not args.kb:
        raise ValueError("检索基准需要 --kb（知识库名称或 ID）")
    if args.ingest_only and not args.ingest:
        raise ValueError("--ingest-only 需要同时提供 --ingest <文件>")
    # 参数错误在构造 provider / 打开数据库之前就报出来
    routes = _parse_routes(args.routes)

    container = AppContainer(settings)
    # 不跑启动恢复：基准可能和正在运行的 API 共用数据库，
    # 把对方的 running 任务标成失败不是「恢复」。
    await container.startup(recover=False)
    exit_code = 0
    try:
        kb = None
        if not args.ingest_only:
            kb = await _resolve_kb(container, args.kb)
            print(
                f"检索基准：kb={kb.name} ({kb.kb_id}) queries={len(queries)} "
                f"repeat={max(1, args.repeat)} cold={args.cold}",
                file=sys.stderr,
            )
        retrieval = (
            await _bench_retrieval(container, kb, queries, args, routes)
            if kb is not None
            else None
        )
        ingest = await _bench_ingest(container, args) if args.ingest else None

        cache_stats = retrieval["cache_stats"] if retrieval else {}
        report = {
            "metadata": _metadata(settings, kb, args, cache_stats),
            "retrieval": retrieval,
            "ingest": ingest,
        }
        output_dir = Path(args.output).expanduser()  # noqa: ASYNC240 - 只读一次本机路径
        markdown_path, json_path = _write_reports(report, output_dir)

        if retrieval:
            total = retrieval["total"]
            print(
                f"检索延迟：p50={_fmt(total.get('p50_ms'))}ms "
                f"p95={_fmt(total.get('p95_ms'))}ms "
                f"（{total.get('count', 0)} 次）"
            )
        if ingest:
            total = ingest["total"]
            print(
                f"导入耗时：p50={_fmt(total.get('p50_ms'))}ms "
                f"p95={_fmt(total.get('p95_ms'))}ms "
                f"（{total.get('count', 0)} 次）"
            )
            if not ingest["all_ok"]:
                exit_code = 1
        print(f"Markdown: {markdown_path}")
        print(f"JSON:     {json_path}")
        return exit_code
    finally:
        await container.shutdown()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(level="INFO" if args.verbose else "WARNING", stream=sys.stderr)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except (ConfigurationError, MMKBError, ValueError, FileNotFoundError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
