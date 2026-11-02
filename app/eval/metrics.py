"""检索与生成指标。

检索指标是**纯函数**：不碰网络、不碰数据库，因此可以精确单测。
所有函数都对 ranked 列表去重并保持首次出现顺序 —— 同一张图可能被三路
各召回一次，身份归一后仍可能因为邻居扩展重复出现，去重必须发生在算分之前。

生成指标分两类：

* ``context_precision`` —— 由人工标注与检索排序直接算出，确定性；
* ``faithfulness`` / ``answer_relevancy`` —— 必须由 LLM judge 打分。
  没有 LLM、judge 输出无法解析、或调用失败时返回 ``None``，
  **不填 0、不猜**：报告里显示「未评估」，比一个假的平均数诚实。

可选的 RAGAS 适配层只负责 faithfulness / answer_relevancy。RAGAS 0.2 的
``evaluate`` API 与版本绑定较紧，且需要 LangChain 包装器；因此这里做了
「导入/调用失败就回退自实现 judge」的防御，并把失败原因写进报告。
默认 ``EVAL_USE_RAGAS=false``，不会走到这条路径。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from app.agent.prompts import extract_json_object
from app.config import Settings
from app.core.logging import get_logger
from app.providers.base import ChatMessage, LLMProvider

__all__ = [
    "GenerationJudge",
    "JudgeScore",
    "average",
    "context_precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "retrieval_metrics",
    "try_ragas",
]

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# 确定性检索指标
# --------------------------------------------------------------------------------------


def _unique(ranked: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in ranked:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def recall_at_k(
    ranked: Sequence[str], relevant: Collection[str], k: int
) -> float | None:
    """Recall@k = 命中数 / 标注相关数。

    标注为空时返回 ``None`` —— 这不是 0 分，而是「这道题没有可算的分」。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if not relevant:
        return None
    hits = sum(1 for key in _unique(ranked)[:k] if key in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float | None:
    """MRR@k：第一个相关结果的名次倒数。没有命中记 0。"""
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if not relevant:
        return None
    for rank, key in enumerate(_unique(ranked)[:k], start=1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked: Sequence[str], relevant: Collection[str], k: int
) -> float | None:
    """二值相关性的 nDCG@k。

    IDCG 用 ``min(len(relevant), k)`` 个理想位置计算；这样即使标注了 20 条
    相关文档、但只看前 10 名，也不会得出一个永远达不到 1.0 的分数。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if not relevant:
        return None

    dcg = 0.0
    for rank, key in enumerate(_unique(ranked)[:k], start=1):
        if key in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def context_precision_at_k(
    ranked: Sequence[str], relevant: Collection[str], k: int
) -> float | None:
    """排序感知的 Context Precision（AP 形式，RAGAS 同名指标的确定性版本）。

    ``sum(precision@i × rel(i)) / min(|relevant|, k)``。
    它同时惩罚「没召回」和「把相关内容排得很后」。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if not relevant:
        return None

    hits = 0
    score = 0.0
    for rank, key in enumerate(_unique(ranked)[:k], start=1):
        if key in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def retrieval_metrics(
    ranked: Sequence[str],
    relevant: Collection[str],
    *,
    recall_ks: Sequence[int],
    mrr_k: int,
    ndcg_k: int,
) -> dict[str, float | None]:
    """一次算齐报告需要的全部指标。key 直接用于报告列名。"""
    metrics: dict[str, float | None] = {}
    for k in sorted(set(recall_ks)):
        metrics[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
    metrics[f"mrr@{mrr_k}"] = mrr_at_k(ranked, relevant, mrr_k)
    metrics[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranked, relevant, ndcg_k)
    metrics["context_precision"] = context_precision_at_k(ranked, relevant, mrr_k)
    return metrics


def average(values: Sequence[float | None]) -> float | None:
    """忽略 ``None`` 求平均；全为 ``None`` 时返回 ``None``。"""
    kept = [value for value in values if value is not None]
    return sum(kept) / len(kept) if kept else None


# --------------------------------------------------------------------------------------
# LLM judge
# --------------------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
你是严格的事实核查员。给定「检索资料」和「回答」，判断回答中的事实性结论
是否都能在资料中找到依据。不要因为回答里出现了资料之外的常识就一律给 0，
但如果回答编造了具体数字、名称、结论，必须扣分。

只输出 JSON，不要 Markdown 代码块：
{"score": 0.0, "reason": "一句话说明扣分点"}
score 取值 0~1：1 = 全部有依据，0 = 大面积编造。
"""

_RELEVANCY_PROMPT = """\
你是检索问答的评审。给定「用户问题」和「回答」，判断回答是否直接解决了问题。
答非所问、只复述资料、拒答但本可回答，都应扣分。

只输出 JSON，不要 Markdown 代码块：
{"score": 0.0, "reason": "一句话说明"}
score 取值 0~1。
"""


@dataclass(slots=True)
class JudgeScore:
    score: float | None = None
    reason: str = ""
    error: str = ""


class GenerationJudge:
    """用 LLM 给生成结果打分。失败只影响这两个指标，不影响检索指标。"""

    def __init__(self, llm: LLMProvider, settings: Settings) -> None:
        self.llm = llm
        self._settings = settings

    async def faithfulness(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> JudgeScore:
        if not answer.strip():
            return JudgeScore(score=None, error="答案为空，无法评估")
        context_text = _join_contexts(contexts)
        user = (
            f"<question>{question}</question>\n\n"
            f"<contexts>\n{context_text}\n</contexts>\n\n"
            f"<answer>\n{answer}\n</answer>"
        )
        return await self._ask(_FAITHFULNESS_PROMPT, user)

    async def answer_relevancy(
        self, question: str, answer: str, reference: str = ""
    ) -> JudgeScore:
        if not answer.strip():
            return JudgeScore(score=None, error="答案为空，无法评估")
        reference_block = (
            f"\n\n<reference>\n{reference}\n</reference>" if reference.strip() else ""
        )
        user = (
            f"<question>{question}</question>\n\n"
            f"<answer>\n{answer}\n</answer>{reference_block}"
        )
        return await self._ask(_RELEVANCY_PROMPT, user)

    async def _ask(self, system: str, user: str) -> JudgeScore:
        try:
            result = await self.llm.chat(
                [ChatMessage.system(system), ChatMessage.user(user)],
                temperature=0.0,
                max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001 - judge 失败不能中断评测
            logger.warning("生成指标 judge 调用失败", extra={"error": str(exc)[:200]})
            return JudgeScore(score=None, error=f"{type(exc).__name__}: {exc}"[:300])

        payload = extract_json_object(result.text or "")
        score = _coerce_score(payload.get("score") if payload else None)
        if score is None:
            score = _coerce_score(_first_number(result.text or ""))
        if score is None:
            return JudgeScore(
                score=None,
                error=f"judge 输出无法解析：{(result.text or '')[:200]}",
            )
        reason = str(payload.get("reason", ""))[:300] if payload else ""
        return JudgeScore(score=score, reason=reason)


def _join_contexts(contexts: Sequence[str], limit_chars: int = 12000) -> str:
    parts: list[str] = []
    used = 0
    for index, text in enumerate(contexts, start=1):
        block = f"[{index}] {text.strip()}"
        if not block:
            continue
        if used + len(block) > limit_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts) if parts else "（无检索资料）"


def _first_number(text: str) -> float | None:
    import re

    match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    return float(match.group(0)) if match else None


def _coerce_score(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return max(0.0, min(1.0, value))


# --------------------------------------------------------------------------------------
# 可选 RAGAS
# --------------------------------------------------------------------------------------


async def try_ragas(
    samples: Sequence[dict[str, Any]],
    settings: Settings,
) -> tuple[dict[str, float] | None, str]:
    """尝试用 RAGAS 计算 faithfulness / answer_relevancy。

    返回 ``(scores, error)``。scores 为 ``None`` 表示不可用，调用方应回退到
    :class:`GenerationJudge`。**不把异常抛给主流程**：评测的可复现性优先于
    某个可选依赖的版本兼容性。

    ``samples`` 每项至少包含 ``question`` / ``answer`` / ``contexts``，
    可选 ``reference``。
    """
    if not samples:
        return None, "没有可用于 RAGAS 的样本"

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, faithfulness
    except Exception as exc:  # noqa: BLE001 - 可选依赖，任何导入失败都回退
        return None, f"RAGAS 不可用（{type(exc).__name__}: {exc}）"

    if not settings.llm.has_api_key or not settings.text_embed.has_api_key:
        return None, "RAGAS 需要 LLM_API_KEY 与 TEXT_EMBED_API_KEY"

    try:
        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.llm.model,
                api_key=settings.llm.api_key.get_secret_value(),
                base_url=settings.llm.endpoint,
                temperature=0.0,
                timeout=settings.llm.timeout_s,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=settings.text_embed.resolved_model,
                api_key=settings.text_embed.resolved.api_key.get_secret_value(),
                base_url=settings.text_embed.resolved_base_url,
                # DashScope / SiliconFlow 不是 OpenAI，必须关掉 tiktoken 长度校验
                check_embedding_ctx_length=False,
            )
        )
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=sample["question"],
                    response=sample["answer"],
                    retrieved_contexts=list(sample.get("contexts", [])),
                    reference=sample.get("reference") or None,
                )
                for sample in samples
            ]
        )
        result = await asyncio.to_thread(
            evaluate,
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )
        frame = result.to_pandas()
        scores: dict[str, float] = {}
        for name in ("faithfulness", "answer_relevancy"):
            if name in frame.columns:
                value = frame[name].dropna()
                if len(value):
                    scores[name] = float(value.mean())
        if not scores:
            return None, "RAGAS 返回结果中没有 faithfulness / answer_relevancy"
        return scores, ""
    except Exception as exc:  # noqa: BLE001 - 版本差异/网络失败都回退
        logger.warning("RAGAS 评估失败，回退自实现指标", extra={"error": str(exc)[:300]})
        return None, f"{type(exc).__name__}: {exc}"[:500]
