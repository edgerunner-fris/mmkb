"""基准脚本的纯函数测试。

这里不跑真实检索：``scripts/benchmark.py`` 的价值一半在测量、一半在**口径**
（最近秩分位、冷热缓存、阶段不可相加）。口径错了，测出来的数字再多也没用，
所以把统计与渲染单独钉住。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from app.core.schemas import Route
from scripts.benchmark import (
    _build_parser,
    _load_queries,
    _order_by,
    _parse_routes,
    _percentile,
    _stats,
    _write_reports,
    render_markdown,
)


def test_percentile_uses_nearest_rank() -> None:
    ordered = [float(value) for value in range(1, 101)]
    assert _percentile(ordered, 50) == 50.0
    assert _percentile(ordered, 95) == 95.0
    assert _percentile(ordered, 99) == 99.0
    assert _percentile([], 95) == 0.0


def test_stats_shape_and_empty_values() -> None:
    stats = _stats([120, 140, 160])
    assert stats["count"] == 3
    assert stats["p50_ms"] == 140.0
    assert stats["mean_ms"] == 140.0

    empty = _stats([])
    assert empty["count"] == 0
    assert empty["p50_ms"] is None


def test_parse_routes_rejects_hybrid_and_unknown() -> None:
    assert _parse_routes("") is None
    assert _parse_routes("bm25,dense_text") == [Route.BM25, Route.DENSE_TEXT]
    with pytest.raises(ValueError, match="hybrid"):
        _parse_routes("hybrid")
    with pytest.raises(ValueError, match="未知"):
        _parse_routes("graph")


def test_load_queries_dedupes_and_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "queries.txt"
    path.write_text(
        "# 注释行\n三路召回\n\n架构图\n三路召回\n"
        '{"query": "RRF 的 k 值"}\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(queries=["重复问题"], queries_file=str(path))
    assert _load_queries(args) == ["重复问题", "三路召回", "架构图", "RRF 的 k 值"]


def _report() -> dict[str, object]:
    latency = _stats([120, 140])
    return {
        "metadata": {
            "generated_at": "2026-10-20T00:00:00+00:00",
            "kb_id": "kb_demo",
            "kb_name": "演示库",
            "routes": "全部启用路",
            "rerank": True,
            "top_k": None,
            "repeat": 2,
            "warmup": 1,
            "cold": False,
            "providers": {"text_embedding": "dashscope/text-embedding-v4"},
        },
        "retrieval": {
            "queries": 1,
            "repeat": 2,
            "warmup": 1,
            "cold": False,
            "total": latency,
            "stages": {"plan": latency, "retrieve": latency, "total": latency},
            "routes": {"bm25": latency, "dense_text": latency},
            "per_query": [
                {
                    "query": "三路召回",
                    "runs": 2,
                    "latency": latency,
                    "items": [3, 3],
                    "degraded": [],
                }
            ],
            "degraded_runs": 0,
            "cache_stats": {
                "dense_text": {
                    "query_embedding": {
                        "hits": 3,
                        "lookups": 4,
                        "hit_rate": 0.75,
                        "size": 1,
                    }
                },
                "vectorstore": {
                    "collection_meta": {
                        "hits": 1,
                        "lookups": 2,
                        "hit_rate": 0.5,
                        "size": 1,
                    },
                    # 非命中率统计不应被渲染成一行缓存
                    "collection_handles": 2,
                },
            },
            "samples": [],
        },
        "ingest": None,
    }


def test_render_markdown_contains_all_sections() -> None:
    text = render_markdown(_report())

    assert text.startswith("# mmkb 性能基准")
    for heading in ("## 1. 检索总延迟", "## 2. 阶段拆解", "## 3. 分路召回延迟",
                    "## 4. 每次查询明细", "## 5. 缓存命中率"):
        assert heading in text
    # 导入基准没跑时不渲染空章节
    assert "## 6. 导入基准" not in text
    assert "## 6. 解读与限制" in text

    assert "| 全部检索 | 2 | 120.0 | 140.0 |" in text
    assert "dense_text.query_embedding" in text
    assert "75.0%" in text
    assert "collection_handles" not in text
    assert "阶段耗时不等于墙钟时间" in text


def test_render_markdown_includes_ingest_when_present() -> None:
    report = _report()
    report["ingest"] = {
        "file": "demo.pdf",
        "kb_id": "kb_bench",
        "kept": False,
        "runs": [
            {
                "file": "demo.pdf",
                "status": "done",
                "elapsed_ms": 1200,
                "chunks": 12,
                "images": 2,
                "text_vectors": 12,
                "image_vectors": 2,
                "error": None,
            }
        ],
        "total": _stats([1200]),
        "stages": {"parse": _stats([300])},
        "all_ok": True,
    }
    text = render_markdown(report)
    assert "## 6. 导入基准" in text
    assert "已删除" in text
    # 阶段按中文语义标签渲染，而不是原始键
    assert "| 解析 | 1 |" in text


def test_write_reports_writes_markdown_and_json(tmp_path: Path) -> None:
    report = _report()
    markdown_path, json_path = _write_reports(report, tmp_path / "bench")

    assert markdown_path.read_text(encoding="utf-8").startswith("# mmkb 性能基准")
    decoded = json.loads(json_path.read_text(encoding="utf-8"))
    assert decoded["retrieval"]["total"]["p95_ms"] == 140.0


def test_parser_has_expected_defaults() -> None:
    args = _build_parser().parse_args([])
    assert args.repeat == 3
    assert args.warmup == 1
    assert args.cold is False
    assert args.output == "data/bench"


def test_stage_order_follows_pipeline_not_alphabet() -> None:
    labels = {"plan": "规划", "retrieve": "召回", "fuse": "融合"}
    assert _order_by(labels, ["fuse", "plan", "unknown", "retrieve"]) == [
        "plan",
        "retrieve",
        "fuse",
        "unknown",
    ]
