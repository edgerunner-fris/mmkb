"""评测结果结构与产物渲染。

产物有三个，各有用途：

* ``report.md`` —— 人看的结论，直接贴进 README / 面试材料；
* ``report.json`` —— 机器可读的汇总，便于 CI 比较两次提交；
* ``cases.jsonl`` —— 逐题明细，用来回答「到底是哪道题掉了分」。

另附 matplotlib 柱状图。图表用英文标签，避免容器里没有中文字体时出现方块；
Markdown 报告里仍然用中文表头。
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.eval.ablation import AblationPreset

__all__ = [
    "CaseResult",
    "EvalReport",
    "GroupResult",
    "format_metric",
    "render_markdown",
    "write_outputs",
]

TASKS = ("text", "text_to_image", "image_to_image")

TASK_LABELS = {
    "text": "文本检索",
    "text_to_image": "以文搜图",
    "image_to_image": "以图搜图",
}


@dataclass(slots=True)
class CaseResult:
    """一道题在一个消融组里的完整结果。"""

    case_id: str
    task: str
    query: str
    ranked: list[str] = field(default_factory=list)
    relevant: list[str] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)
    generation: dict[str, float | None] = field(default_factory=dict)
    answer: str = ""
    item_count: int = 0
    latency_ms: int = 0
    degraded_reasons: list[str] = field(default_factory=list)
    skipped_reason: str = ""
    notes: str = ""

    @property
    def scored(self) -> bool:
        return not self.skipped_reason and any(
            value is not None for value in self.metrics.values()
        )


@dataclass(slots=True)
class GroupResult:
    preset: AblationPreset
    cases: list[CaseResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ragas: dict[str, float] = field(default_factory=dict)
    #: 运行期生成样本，只供可选的 RAGAS 适配层使用，不写进 JSON 报告
    ragas_samples: list[dict[str, Any]] = field(default_factory=list)

    def cases_for(self, task: str) -> list[CaseResult]:
        return [case for case in self.cases if case.task == task]

    def metric(self, name: str, task: str | None = None) -> float | None:
        values = [
            case.metrics.get(name)
            for case in self.cases
            if (task is None or case.task == task)
        ]
        return _mean(values)

    def generation_metric(self, name: str) -> float | None:
        return _mean([case.generation.get(name) for case in self.cases])

    def scored_count(self, task: str | None = None) -> int:
        return sum(
            1
            for case in self.cases
            if (task is None or case.task == task) and case.scored
        )

    def degraded_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            for reason in case.degraded_reasons:
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def latencies(self, task: str | None = None) -> list[int]:
        """检索延迟样本。

        只统计实际执行过检索的用例（``latency_ms > 0``）：被跳过、因缺少
        召回路而未执行的用例延迟为 0，混进 p50 会把整组延迟拉低，
        让「这一组更快」变成一个由跳过造成的假象。
        """
        return [
            case.latency_ms
            for case in self.cases
            if (task is None or case.task == task) and case.latency_ms > 0
        ]

    def latency(self, task: str | None = None) -> dict[str, float | int | None]:
        return _latency_stats(self.latencies(task))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.preset.name,
            "routes": self.preset.route_labels,
            "use_rerank": self.preset.use_rerank,
            "description": self.preset.description,
            "scored_cases": self.scored_count(),
            "metrics": {
                task: {
                    name: self.metric(name, task)
                    for name in _metric_names(self.cases_for(task))
                }
                for task in TASKS
            },
            "generation": {
                name: self.generation_metric(name)
                for name in ("faithfulness", "answer_relevancy", "context_precision")
            },
            "latency": {
                **{task: self.latency(task) for task in TASKS},
                "all": self.latency(),
            },
            "ragas": dict(self.ragas),
            "degraded": self.degraded_counts(),
            "errors": list(self.errors),
        }


@dataclass(slots=True)
class EvalReport:
    dataset_path: str
    kb_id: str
    kb_name: str
    top_k: int
    recall_ks: list[int]
    mrr_k: int
    ndcg_k: int
    total_cases: int
    labeled_cases: int
    unlabeled_cases: int
    groups: list[GroupResult] = field(default_factory=list)
    generation_group: str = ""
    generation_skipped_reason: str = ""
    ragas_error: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def group(self, name: str) -> GroupResult | None:
        return next((group for group in self.groups if group.preset.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "kb_id": self.kb_id,
            "kb_name": self.kb_name,
            "generated_at": self.generated_at,
            "top_k": self.top_k,
            "recall_ks": list(self.recall_ks),
            "mrr_k": self.mrr_k,
            "ndcg_k": self.ndcg_k,
            "counts": {
                "total": self.total_cases,
                "labeled": self.labeled_cases,
                "unlabeled": self.unlabeled_cases,
            },
            "generation_group": self.generation_group,
            "generation_skipped_reason": self.generation_skipped_reason,
            "ragas_error": self.ragas_error,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "duration_ms": self.duration_ms,
            "groups": [group.to_dict() for group in self.groups],
        }


# --------------------------------------------------------------------------------------
# 计算辅助
# --------------------------------------------------------------------------------------


def _mean(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return statistics.fmean(kept) if kept else None


def _percentile(ordered: list[int], percent: float) -> float:
    """最近秩百分位，与 ``app.core.perf`` 保持同一口径。

    两个模块各写一半实现不划算，但让评测脚本 import Web 进程的指标登记表
    更不划算（评测是一个独立 CLI，不该拉起 FastAPI 依赖链）。
    """
    if not ordered:
        return 0.0
    rank = math.ceil(percent / 100.0 * len(ordered)) - 1
    return float(ordered[max(0, min(rank, len(ordered) - 1))])


def _latency_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 1),
        "p50_ms": round(_percentile(ordered, 50), 1),
        "p95_ms": round(_percentile(ordered, 95), 1),
        "max_ms": float(ordered[-1]),
    }


def format_latency(stats: dict[str, Any], key: str = "p50_ms") -> str:
    value = stats.get(key)
    return "—" if value is None else f"{float(value):.0f}"


def _metric_names(cases: list[CaseResult]) -> list[str]:
    """保持指标列顺序稳定：recall@5、recall@10、mrr@10、ndcg@10、context_precision。"""
    order: list[str] = []
    for case in cases:
        for name in case.metrics:
            if name not in order:
                order.append(name)
    return sorted(order, key=_metric_sort_key)


def _metric_sort_key(name: str) -> tuple[int, int, str]:
    if name.startswith("recall@"):
        return (0, int(name.split("@", 1)[1]), name)
    if name.startswith("mrr@"):
        return (1, int(name.split("@", 1)[1]), name)
    if name.startswith("ndcg@"):
        return (2, int(name.split("@", 1)[1]), name)
    return (3, 0, name)


def format_metric(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def render_markdown(report: EvalReport) -> str:
    lines: list[str] = []
    add = lines.append

    add("# mmkb 检索与生成评估报告")
    add("")
    add(
        f"> 生成时间：{report.generated_at} ｜ 知识库：**{report.kb_name}** "
        f"(`{report.kb_id}`) ｜ 数据集：`{report.dataset_path}`"
    )
    add("")
    add(
        f"> **只统计人工标注用例**：{report.labeled_cases} / {report.total_cases} 条计入；"
        f"{report.unlabeled_cases} 条未标注已跳过。"
    )
    if report.unlabeled_cases:
        add(">")
        add(
            "> ⚠️ 未标注条目不计分，也不会被当成 0 分。请先在 JSONL 中完成标注"
            "（`labeled=true` 并填写 `item_ids` / `doc_ids` / `image_ids`）。"
        )
    add("")

    add("## 1. 消融配置")
    add("")
    add("| 组 | 召回路 | 重排 | 说明 |")
    add("|---|---|---|---|")
    for group in report.groups:
        add(
            f"| `{group.preset.name}` | {group.preset.route_text} | "
            f"{'是' if group.preset.use_rerank else '否'} | {group.preset.description} |"
        )
    add("")

    add("## 2. 文本检索")
    add("")
    add("| 组 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | 计分用例 |")
    add("|---|---:|---:|---:|---:|---:|")
    for group in report.groups:
        add(
            f"| `{group.preset.name}` | {format_metric(group.metric('recall@5', 'text'))} | "
            f"{format_metric(group.metric('recall@10', 'text'))} | "
            f"{format_metric(group.metric(f'mrr@{report.mrr_k}', 'text'))} | "
            f"{format_metric(group.metric(f'ndcg@{report.ndcg_k}', 'text'))} | "
            f"{group.scored_count('text')} |"
        )
    add("")

    add("## 3. 跨模态检索")
    add("")
    add("| 组 | 以文搜图 Recall@5 | 以图搜图 Recall@5 | 以文搜图用例 | 以图搜图用例 |")
    add("|---|---:|---:|---:|---:|")
    for group in report.groups:
        add(
            f"| `{group.preset.name}` | "
            f"{format_metric(group.metric('recall@5', 'text_to_image'))} | "
            f"{format_metric(group.metric('recall@5', 'image_to_image'))} | "
            f"{group.scored_count('text_to_image')} | "
            f"{group.scored_count('image_to_image')} |"
        )
    add("")
    add(
        "> 不含 `dense_image` 的组不会执行以图搜图；`—` 表示该组没有可计算的用例，"
        "不是 0 分。"
    )
    add("")

    add("## 4. 生成质量")
    add("")
    add(f"评估组：`{report.generation_group or '（未指定）'}`")
    add("")
    if report.generation_skipped_reason:
        add(f"> 跳过原因：{report.generation_skipped_reason}")
        add("")
    else:
        generation_group = report.group(report.generation_group)
        if generation_group is None:
            add("> 未找到指定评估组，生成指标不可用。")
            add("")
        else:
            add("| 指标 | 分数 | 计算方式 |")
            add("|---|---:|---|")
            add(
                f"| Faithfulness | {format_metric(generation_group.generation_metric('faithfulness'))} "
                "| LLM judge：答案是否都能在检索资料中找到依据 |"
            )
            add(
                f"| Answer Relevancy | "
                f"{format_metric(generation_group.generation_metric('answer_relevancy'))} "
                "| LLM judge：回答是否直接解决问题 |"
            )
            add(
                f"| Context Precision | "
                f"{format_metric(generation_group.generation_metric('context_precision'))} "
                "| 人工标注 + 检索排序，确定性计算 |"
            )
            add("")
            if generation_group.ragas:
                add(
                    "> Faithfulness / Answer Relevancy 由 RAGAS 计算："
                    + ", ".join(
                        f"{name}={value:.4f}" for name, value in generation_group.ragas.items()
                    )
                )
                add("")
            elif report.ragas_error:
                add(f"> RAGAS 未启用或失败，已回退到自实现 judge：{report.ragas_error}")
                add("")
            add(
                "> `—` 表示该指标未评估（缺少 LLM Key、judge 调用失败或参考答案为空）。"
                "未评估不会被写成 0。"
            )
            add("")

    add("## 5. 检索延迟")
    add("")
    add(
        "> 单位为毫秒，只统计实际执行过检索的用例。延迟与召回质量必须一起看："
        "多开一路、开启重排都会提高延迟，消融报告的价值就在于把收益和代价放在同一张表里。"
    )
    add("")
    add(
        "| 组 | 文本 p50 | 文本 p95 | 以文搜图 p50 | 以文搜图 p95 | "
        "以图搜图 p50 | 以图搜图 p95 | 全部 p50 | 全部 p95 | 样本数 |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group in report.groups:
        text = group.latency("text")
        text_image = group.latency("text_to_image")
        image_image = group.latency("image_to_image")
        overall = group.latency()
        add(
            f"| `{group.preset.name}` | {format_latency(text)} | {format_latency(text, 'p95_ms')} | "
            f"{format_latency(text_image)} | {format_latency(text_image, 'p95_ms')} | "
            f"{format_latency(image_image)} | {format_latency(image_image, 'p95_ms')} | "
            f"{format_latency(overall)} | {format_latency(overall, 'p95_ms')} | "
            f"{overall['count']} |"
        )
    add("")

    add("## 6. 运行信息")
    add("")
    add(f"- 耗时：{report.duration_ms / 1000:.1f} s")
    add(f"- 检索截断：top_k = {report.top_k}（Recall/MRR/nDCG 以该深度计算）")
    add(
        f"- 指标参数：Recall@{','.join(str(k) for k in report.recall_ks)}、"
        f"MRR@{report.mrr_k}、nDCG@{report.ndcg_k}"
    )
    add("")

    degraded_lines: list[str] = []
    for group in report.groups:
        counts = group.degraded_counts()
        if counts:
            detail = ", ".join(f"`{name}`×{count}" for name, count in counts.items())
            degraded_lines.append(f"- `{group.preset.name}`：{detail}")
    if degraded_lines:
        add("**降级统计**")
        add("")
        lines.extend(degraded_lines)
        add("")

    if report.warnings:
        add("**数据告警**")
        add("")
        for warning in report.warnings:
            add(f"- {warning}")
        add("")

    if report.errors:
        add("**运行错误**")
        add("")
        for error in report.errors:
            add(f"- {error}")
        add("")

    add("**复现命令**")
    add("")
    add("```powershell")
    add(
        f".\\.venv\\Scripts\\python.exe -m app.eval.run_eval "
        f"--dataset {report.dataset_path} --kb {report.kb_id} "
        f"--ablation {','.join(group.preset.name for group in report.groups)} "
        f"--top-k {report.top_k} --output data/eval"
    )
    add("```")
    add("")

    add("## 7. 标注说明")
    add("")
    add("- `text`：至少填写 `item_ids`（分块 ID）或 `doc_ids`（文档 ID）；")
    add("- `text_to_image`：填写 `image_ids`（图片 ID），跨模态指标只认这个；")
    add("- `image_to_image`：填写 `image_ids`，并提供存在的 `query_image`；")
    add("- `ground_truth.answer` 可选，用于 Answer Relevancy 的参考；")
    add("- 未标注的条目请保持 `labeled=false`，不要用空数组冒充「无相关结果」。")
    add("")
    add("> 已知限制：reranker 只接受纯文本、无法理解图像。图片候选在重排时使用 "
        "`caption + 章节路径` 拼成的文本，因此 `hybrid_rrf_rerank` 与 `hybrid_rrf` "
        "的差值**包含**这一取舍带来的损失，解读时应注明。")
    add("")

    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 产物写盘
# --------------------------------------------------------------------------------------


def write_outputs(
    report: EvalReport,
    output_dir: Path,
    *,
    make_chart: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    markdown_path = output_dir / "report.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    written.append(markdown_path)

    json_path = output_dir / "report.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(json_path)

    cases_path = output_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for group in report.groups:
            for case in group.cases:
                payload = asdict(case)
                payload["group"] = group.preset.name
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    written.append(cases_path)

    if make_chart:
        written.extend(_write_charts(report, output_dir))
    return written


def _write_charts(report: EvalReport, output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:  # noqa: BLE001 - 图表是锦上添花，缺依赖不该让评测失败
        return []

    written: list[Path] = []
    names = [group.preset.name for group in report.groups]

    text_series = [
        ("Recall@5", "recall@5"),
        ("Recall@10", "recall@10"),
        (f"MRR@{report.mrr_k}", f"mrr@{report.mrr_k}"),
        (f"nDCG@{report.ndcg_k}", f"ndcg@{report.ndcg_k}"),
    ]
    text_values = [
        [group.metric(metric, "text") for group in report.groups]
        for _, metric in text_series
    ]
    if any(value is not None for series in text_values for value in series):
        path = output_dir / "ablation_text.png"
        _grouped_bars(
            plt,
            np,
            names,
            [label for label, _ in text_series],
            text_values,
            title="Text retrieval ablation",
            ylabel="score",
            path=path,
        )
        written.append(path)

    cross_values = [
        [group.metric("recall@5", "text_to_image") for group in report.groups],
        [group.metric("recall@5", "image_to_image") for group in report.groups],
    ]
    if any(value is not None for series in cross_values for value in series):
        path = output_dir / "ablation_cross_modal.png"
        _grouped_bars(
            plt,
            np,
            names,
            ["Text-to-image Recall@5", "Image-to-image Recall@5"],
            cross_values,
            title="Cross-modal retrieval ablation",
            ylabel="recall@5",
            path=path,
        )
        written.append(path)

    return written


def _grouped_bars(
    plt: Any,
    np: Any,
    group_names: list[str],
    series_labels: list[str],
    series_values: list[list[float | None]],
    *,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    x = np.arange(len(group_names))
    width = 0.8 / max(len(series_labels), 1)

    figure, axis = plt.subplots(figsize=(max(7.5, len(group_names) * 1.6), 4.8))
    for index, (label, values) in enumerate(zip(series_labels, series_values, strict=True)):
        heights = [0.0 if value is None else value for value in values]
        offset = (index - (len(series_labels) - 1) / 2) * width
        bars = axis.bar(x + offset, heights, width, label=label)
        for bar, value in zip(bars, values, strict=True):
            if value is None:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.01,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
    axis.set_xticks(x)
    axis.set_xticklabels(group_names, rotation=15, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
