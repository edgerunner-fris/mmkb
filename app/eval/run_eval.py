"""评估 CLI：``python -m app.eval.run_eval --kb <id|name> --ablation all``。

执行顺序
--------
1. 读数据集（只保留人工标注过的题）；
2. 对每组消融跑完全部题目的检索，计算确定性指标；
3. 只对**一个**指定组跑生成，避免五组各生成一遍把费用放大五倍；
4. 输出 ``report.md`` / ``report.json`` / ``cases.jsonl`` / 柱状图。

以图搜图
--------
检索层没有「图片作为 query」的公开入口（线上以文搜图才是主场景），
但评测需要量化图搜图能力，因此这里直接用多模态 provider 编码查询图片、
查 ``{kb_id}__image``。它仍然遵守铁律 2/3：用的是图像空间 provider、
``space="image"``，并先做知识库快照校验。

退出码：``0`` 完成 · ``1`` 运行失败 · ``2`` 用法/配置/数据集问题 · ``130`` 中断。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import ConfigurationError, Settings, get_settings
from app.core.errors import MMKBError, NotFoundError, RetrievalError
from app.core.logging import get_logger, setup_logging
from app.core.schemas import KnowledgeBase, RetrievedItem, Route
from app.eval.ablation import AblationPreset, resolve_presets
from app.eval.dataset import (
    DatasetError,
    EvalCase,
    EvalDataset,
    load_dataset,
    resolve_query_image,
)
from app.eval.metrics import (
    GenerationJudge,
    JudgeScore,
    retrieval_metrics,
    try_ragas,
)
from app.eval.report import (
    CaseResult,
    EvalReport,
    GroupResult,
    format_metric,
    write_outputs,
)
from app.generation.answer import AnswerGenerator
from app.retrieval.base import guard_kb_snapshot, item_from_image_hit
from app.services.container import AppContainer

__all__ = ["EvalOptions", "EvalRunner", "main"]

logger = get_logger(__name__)

Progress = Callable[[str], None]
_CASE_CONTEXT_CHARS = 2000
_JUDGE_CONTEXT_ITEMS = 8


@dataclass(slots=True)
class EvalOptions:
    top_k: int = 10
    skip_generation: bool = False
    generation_group: str = "hybrid_rrf_rerank"
    use_ragas: bool = False
    max_questions: int = 0
    make_chart: bool = True


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _unit_key(item: RetrievedItem, case: EvalCase) -> str:
    """把一条检索结果映射成「评测单元」。

    * 图片任务：只认 ``image_id``，文本分块命中了 caption 也不算找到图；
    * 文本任务：如果标注了文档级相关，同一文档的多个分块折叠成一个单元，
      否则用分块 ID。折叠必须发生在算分前，否则一个相关文档会占满前 10 名，
      Recall 看起来很高，实际只找到了 1 篇文档。
    """
    if case.task in {"text_to_image", "image_to_image"}:
        return item.image_id or item.item_id
    if item.doc_id and item.doc_id in case.ground_truth.doc_ids:
        return item.doc_id
    return item.item_id


def _context_text(item: RetrievedItem, limit: int = _CASE_CONTEXT_CHARS) -> str:
    heading = " > ".join(item.heading_path)
    text = item.text.strip()
    if heading:
        text = f"{heading}\n{text}"
    return text[:limit]


class EvalRunner:
    """把「数据集 × 消融组」展开成结果。不读全局单例，容器由外部注入。"""

    def __init__(
        self,
        container: AppContainer,
        kb: KnowledgeBase,
        dataset: EvalDataset,
        settings: Settings | None = None,
        *,
        options: EvalOptions | None = None,
    ) -> None:
        self.container = container
        self.kb = kb
        self.dataset = dataset
        self.settings = settings or container.settings
        self.options = options or EvalOptions()
        cfg = self.settings.eval
        self.recall_ks = cfg.recall_k_list
        self.mrr_k = cfg.mrr_k
        self.ndcg_k = cfg.ndcg_k
        # Recall@10 需要至少 10 条候选；用户把 top_k 调小只会让指标失真
        self.top_k = max(
            self.options.top_k,
            max(self.recall_ks, default=0),
            self.mrr_k,
            self.ndcg_k,
        )

        self._cases = dataset.limit(self.options.max_questions).labeled
        self._items: dict[tuple[str, str], list[RetrievedItem]] = {}
        self._judge = (
            GenerationJudge(container.llm, self.settings)
            if container.llm is not None
            else None
        )
        self._generator = (
            AnswerGenerator(container.llm, self.settings)
            if container.llm is not None
            else None
        )

    # ------------------------------------------------------------------ 主流程

    async def run(
        self,
        presets: Sequence[AblationPreset],
        *,
        progress: Progress | None = None,
    ) -> EvalReport:
        started = time.perf_counter()
        notify = progress or (lambda message: logger.info("%s", message))
        errors: list[str] = []

        if not self._cases:
            raise DatasetError(
                "数据集中没有已标注用例（labeled=true 且 ground_truth 非空）。"
                "请先人工标注；空标注不会被当成 0 分参与评估。"
            )

        groups: list[GroupResult] = []
        for preset in presets:
            notify(f"[{preset.name}] 开始：{len(self._cases)} 条用例")
            group = await self._run_group(preset, notify)
            groups.append(group)
            if group.errors:
                errors.extend(f"[{preset.name}] {error}" for error in group.errors)

        generation_skipped = ""
        if self.options.skip_generation:
            generation_skipped = "显式跳过（--skip-generation）"
        elif self._generator is None or self._judge is None:
            generation_skipped = "LLM_API_KEY 未配置或生成 provider 不可用"
        elif self.options.generation_group not in {group.preset.name for group in groups}:
            generation_skipped = (
                f"生成评估组 {self.options.generation_group!r} 不在本次消融组中"
            )
        else:
            target = next(
                group
                for group in groups
                if group.preset.name == self.options.generation_group
            )
            notify(f"[{target.preset.name}] 开始生成质量评估")
            await self._run_generation(target, notify)

        ragas_error = ""
        if self.options.use_ragas and not self.options.skip_generation:
            target = next(
                (
                    group
                    for group in groups
                    if group.preset.name == self.options.generation_group
                ),
                None,
            )
            samples = target.ragas_samples if target else []
            if samples:
                scores, ragas_error = await try_ragas(samples, self.settings)
                if scores and target is not None:
                    target.ragas = scores
            elif target is not None:
                ragas_error = "没有可用于 RAGAS 的生成样本"
            if ragas_error:
                logger.warning("RAGAS 未生效，保留自实现指标", extra={"error": ragas_error})

        report = EvalReport(
            dataset_path=str(self.dataset.path),
            kb_id=self.kb.kb_id,
            kb_name=self.kb.name,
            top_k=self.top_k,
            recall_ks=list(self.recall_ks),
            mrr_k=self.mrr_k,
            ndcg_k=self.ndcg_k,
            total_cases=len(self.dataset.cases),
            labeled_cases=len(self.dataset.labeled),
            unlabeled_cases=len(self.dataset.unlabeled),
            groups=groups,
            generation_group=self.options.generation_group,
            generation_skipped_reason=generation_skipped,
            ragas_error=ragas_error,
            warnings=list(self.dataset.warnings),
            errors=errors,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        notify(
            f"完成：{len(groups)} 组，{len(self._cases)} 条已标注用例，"
            f"{report.duration_ms / 1000:.1f}s"
        )
        return report

    # ------------------------------------------------------------------ 单组

    async def _run_group(self, preset: AblationPreset, notify: Progress) -> GroupResult:
        group = GroupResult(preset=preset)
        total = len(self._cases)
        for index, case in enumerate(self._cases, start=1):
            notify(f"[{preset.name}] {index}/{total} {case.id} {case.task}")
            try:
                result, items = await self._run_case(case, preset)
            except RetrievalError:
                # 快照不一致、provider 空间不符、权重配置错误这类问题对整组
                # 每一条用例都一模一样。继续跑只会重复输出同一条错误，还会
                # 让「配置错了」看起来像「检索效果差」；直接中止整次评测。
                raise
            except Exception as exc:  # noqa: BLE001 - 单题失败不该中断整组
                logger.exception(
                    "评测用例失败", extra={"group": preset.name, "case_id": case.id}
                )
                message = f"{case.id}: {type(exc).__name__}: {exc}"[:500]
                group.errors.append(message)
                result = CaseResult(
                    case_id=case.id,
                    task=case.task,
                    query=case.query,
                    skipped_reason=message,
                )
                items = []
            group.cases.append(result)
            self._items[(preset.name, case.id)] = items
        return group

    async def _run_case(
        self, case: EvalCase, preset: AblationPreset
    ) -> tuple[CaseResult, list[RetrievedItem]]:
        relevant = sorted(case.relevant_keys)
        if not relevant:
            return (
                CaseResult(
                    case_id=case.id,
                    task=case.task,
                    query=case.query,
                    skipped_reason="标注中没有与任务类型匹配的相关 ID",
                ),
                [],
            )

        if case.task == "image_to_image":
            if not preset.supports_image_query:
                return (
                    CaseResult(
                        case_id=case.id,
                        task=case.task,
                        query=case.query,
                        relevant=relevant,
                        skipped_reason="该消融组不含多模态向量路，无法以图搜图",
                    ),
                    [],
                )
            items, notes, latency_ms = await self._image_search(case)
            return (
                self._build_result(
                    case,
                    items,
                    relevant,
                    latency_ms=latency_ms,
                    notes=notes,
                ),
                items,
            )

        started = time.perf_counter()
        outcome = await self.container.retriever.search(
            self.kb,
            case.query,
            top_k=self.top_k,
            routes=preset.routes,
            use_rerank=preset.use_rerank,
            expand_neighbors=False,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        notes = ""
        if preset.use_rerank and not outcome.reranked:
            notes = "本组配置了重排，但本次未实际生效（provider 缺失或调用失败）"
        result = self._build_result(
            case,
            outcome.items,
            relevant,
            latency_ms=latency_ms,
            degraded=[str(reason) for reason in outcome.degraded_reasons],
            notes=notes,
        )
        return result, list(outcome.items)

    async def _image_search(
        self, case: EvalCase
    ) -> tuple[list[RetrievedItem], str, int]:
        provider = self.container.image_embedding
        if provider is None:
            return [], "多模态向量 provider 不可用（IMAGE_EMBED_API_KEY 未配置）", 0
        path = resolve_query_image(case, self.dataset.path)
        if path is None:
            return [], f"query_image 不存在：{case.query_image}", 0

        guard_kb_snapshot(self.kb, provider, space="image")
        if await self.container.vectorstore.count(self.kb.image_collection) <= 0:
            return [], f"{self.kb.image_collection} 为空，没有可检索的图片向量", 0

        started = time.perf_counter()
        image_bytes = await asyncio.to_thread(path.read_bytes)
        embedding = await provider.embed_images([image_bytes])
        if not embedding.vectors:
            return [], "多模态 embedding 返回空向量", int((time.perf_counter() - started) * 1000)

        hits = await self.container.vectorstore.query(
            self.kb.image_collection,
            embedding.vectors[0],
            top_k=self.top_k,
            space="image",
        )
        items = [
            item_from_image_hit(hit, kb_id=self.kb.kb_id, route=Route.DENSE_IMAGE, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]
        return (
            items,
            "以图搜图没有文本 query，重排器无法参与；本组结果按向量相似度排序",
            int((time.perf_counter() - started) * 1000),
        )

    def _build_result(
        self,
        case: EvalCase,
        items: Sequence[RetrievedItem],
        relevant: list[str],
        *,
        latency_ms: int,
        degraded: list[str] | None = None,
        notes: str = "",
    ) -> CaseResult:
        ranked = _dedupe(_unit_key(item, case) for item in items)
        metrics = retrieval_metrics(
            ranked,
            relevant,
            recall_ks=self.recall_ks,
            mrr_k=self.mrr_k,
            ndcg_k=self.ndcg_k,
        )
        return CaseResult(
            case_id=case.id,
            task=case.task,
            query=case.query,
            ranked=ranked,
            relevant=relevant,
            metrics=metrics,
            item_count=len(items),
            latency_ms=latency_ms,
            degraded_reasons=degraded or [],
            notes=notes,
        )

    # ------------------------------------------------------------------ 生成

    async def _run_generation(self, group: GroupResult, notify: Progress) -> None:
        assert self._generator is not None and self._judge is not None
        samples: list[dict[str, Any]] = []
        total = len(group.cases)
        for index, result in enumerate(group.cases, start=1):
            case = self._case(result.case_id)
            if case.task == "image_to_image":
                result.generation = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": result.metrics.get("context_precision"),
                }
                continue
            if result.skipped_reason:
                continue

            notify(f"[生成] {index}/{total} {case.id}")
            items = self._items.get((group.preset.name, case.id), [])
            try:
                outcome = await self._generator.answer(
                    case.query,
                    items,
                    reranked=group.preset.use_rerank,
                )
            except Exception as exc:  # noqa: BLE001 - 单题生成失败不中断评测
                message = f"{case.id}: 生成失败 {type(exc).__name__}: {exc}"[:500]
                group.errors.append(message)
                result.generation = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": result.metrics.get("context_precision"),
                }
                continue

            contexts = [_context_text(item) for item in items[:_JUDGE_CONTEXT_ITEMS]]
            faith: JudgeScore = await self._judge.faithfulness(
                case.query, outcome.answer_markdown, contexts
            )
            relevancy: JudgeScore = await self._judge.answer_relevancy(
                case.query,
                outcome.answer_markdown,
                reference=case.ground_truth.answer,
            )
            for score in (faith, relevancy):
                if score.error:
                    group.errors.append(f"{case.id}: judge — {score.error}")

            result.answer = outcome.answer_markdown
            result.generation = {
                "faithfulness": faith.score,
                "answer_relevancy": relevancy.score,
                "context_precision": result.metrics.get("context_precision"),
            }
            samples.append(
                {
                    "question": case.query,
                    "answer": outcome.answer_markdown,
                    "contexts": contexts,
                    "reference": case.ground_truth.answer,
                }
            )

        # 挂在 dataclass 字段上而不是靠 setattr：GroupResult 用了 slots，
        # 动态属性会直接抛 AttributeError。该字段不会进入 JSON 报告。
        group.ragas_samples = samples

    def _case(self, case_id: str) -> EvalCase:
        for case in self._cases:
            if case.id == case_id:
                return case
        raise KeyError(case_id)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


async def _resolve_kb(container: AppContainer, key: str) -> KnowledgeBase:
    """支持传 kb_id 或人工可读的名字。重名在建库时已被拒绝。"""
    kb = await container.docstore.get_kb(key)
    if kb is not None:
        return kb
    for candidate in await container.docstore.list_kbs(include_deleting=False):
        if candidate.name == key:
            return candidate
    raise NotFoundError(
        f"知识库 {key!r} 不存在（既不是 kb_id 也不是名称）",
        detail={"kb": key},
    )


def _parse_args(argv: Sequence[str] | None, settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.run_eval",
        description="五组 A/B 消融 + 多模态检索指标评估",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="评测集 JSONL；默认读取 EVAL_DATASET_PATH 或 <DATA_DIR>/eval/eval_set.jsonl",
    )
    parser.add_argument("--kb", required=True, help="知识库 ID 或名称")
    parser.add_argument(
        "--ablation",
        default=None,
        help="消融组，逗号分隔；默认 all（五组）",
    )
    parser.add_argument("--top-k", type=int, default=10, help="检索深度，默认 10")
    parser.add_argument(
        "--generation-group",
        default="hybrid_rrf_rerank",
        help="只对这一组跑生成质量评估，默认 hybrid_rrf_rerank",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="只算检索指标，不调用生成与 LLM judge",
    )
    parser.add_argument(
        "--use-ragas",
        action="store_true",
        help="尝试用 RAGAS 计算 faithfulness / answer_relevancy，失败自动回退",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=settings.eval.max_questions,
        help="最多评估多少条已标注用例，0 = 全部",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="产物目录；默认 EVAL_OUTPUT_DIR 或 <DATA_DIR>/eval",
    )
    parser.add_argument("--no-chart", action="store_true", help="不生成 matplotlib 图")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    dataset_path = (args.dataset or settings.eval_dataset_path).resolve()
    dataset = load_dataset(dataset_path)
    if not dataset.labeled:
        print(
            "[错误] 数据集中没有已标注用例。\n"
            "代码不会替你生成 ground truth。请复制示例并人工标注：\n"
            "  PowerShell:\n"
            "    New-Item -ItemType Directory -Force data/eval | Out-Null\n"
            "    Copy-Item eval/samples/eval_set.example.jsonl data/eval/eval_set.jsonl\n"
            "  macOS / Linux:\n"
            "    mkdir -p data/eval\n"
            "    cp eval/samples/eval_set.example.jsonl data/eval/eval_set.jsonl\n"
            "然后把 labeled 改为 true，并填写 item_ids / doc_ids / image_ids。",
            file=sys.stderr,
        )
        return 2

    presets = resolve_presets(args.ablation or settings.eval.ablation_presets)
    options = EvalOptions(
        top_k=max(1, args.top_k),
        skip_generation=args.skip_generation,
        generation_group=args.generation_group,
        use_ragas=args.use_ragas or settings.eval.use_ragas,
        max_questions=max(0, args.max_questions),
        make_chart=not args.no_chart,
    )

    container = AppContainer(settings)
    await container.startup()
    try:
        kb = await _resolve_kb(container, args.kb)
        runner = EvalRunner(container, kb, dataset, settings, options=options)
        report = await runner.run(presets)
        output_dir = (args.output or settings.eval_output_dir).resolve()
        written = write_outputs(report, output_dir, make_chart=options.make_chart)
    finally:
        await container.shutdown()

    print("")
    print("评估完成")
    print(f"  数据集      {dataset_path}")
    print(f"  知识库      {kb.name} ({kb.kb_id})")
    print(
        f"  用例        {report.labeled_cases} 条已标注 / {report.total_cases} 条总计"
        f"（{report.unlabeled_cases} 条未标注已跳过）"
    )
    print("")
    print(f"  {'组':<20} {'文本 Recall@5':>16} {'文搜图 Recall@5':>18} {'图搜图 Recall@5':>18}")
    for group in report.groups:
        print(
            f"  {group.preset.name:<20} "
            f"{format_metric(group.metric('recall@5', 'text')):>16} "
            f"{format_metric(group.metric('recall@5', 'text_to_image')):>18} "
            f"{format_metric(group.metric('recall@5', 'image_to_image')):>18}"
        )
    print("")
    print("  产物：")
    for path in written:
        print(f"    {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = get_settings()
        setup_logging(settings.app.log_level, json_output=False)
        args = _parse_args(argv, settings)
        return asyncio.run(_run(args, settings))
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130
    except (DatasetError, ConfigurationError, NotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except MMKBError as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI 最后一道兜底
        logger.exception("评估失败")
        print(f"[失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
