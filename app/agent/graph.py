"""问答主链路：路由 → 检索 → 相关性判定 → （最多两轮）→ 生成。

为什么不用 LangGraph
-------------------
这条流程是**确定性**的：节点固定、分支固定、最多两轮。用图框架表达它，
换来的是一层需要理解的抽象与一个新的依赖；而这里真正需要的是
「每一轮为什么执行 / 为什么停止」可解释、可测试。一个普通的异步函数
配上显式的 ``for round_no in range(1, max_rounds + 1)`` 已经足够，
``debug_trace`` 也更容易完整记录。

循环的停止条件（按优先级）
--------------------------
1. 判定达标（``score >= relevance_threshold``）→ 用它；
2. 判定不可用（``judge_unavailable``）→ 不再重检，直接用本轮结果；
3. 没有新的改写查询可用 → 再检也是同一个 query，纯浪费；
4. 达到 ``AGENT_MAX_ROUNDS``。

**取「历次轮次中判定分数最高的一轮」**，而不是简单用最后一轮：
第二轮换了 query，可能反而更差；用最后一轮会让一次失败的改写
把本来能回答的资料丢掉。

降级
----
生成模型不可用 → 返回资料原文 + 完整引用（``llm_unavailable``）；
判定模型不可用 → 跳过第二轮（``judge_unavailable``）；
检索某一路失败 → 由检索层隔离，本轮照常。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.agent.prompts import JUDGE_SYSTEM_PROMPT, extract_json_object
from app.agent.router import Intent, Router, RouterDecision, StandaloneQuery
from app.agent.tools import SearchTools, ToolCall
from app.config import Settings, get_settings
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.core.perf import get_metrics
from app.core.schemas import (
    AnswerResult,
    DebugRound,
    DegradedReason,
    JudgeVerdict,
    KnowledgeBase,
    RetrievedItem,
)
from app.generation.answer import AnswerEvent, AnswerGenerator, GenerationOutcome
from app.providers.base import ChatMessage, LLMProvider
from app.retrieval.base import Stopwatch
from app.retrieval.hybrid import HybridRetriever, SearchOutcome

__all__ = ["AnswerAgent", "Judge", "JudgeOutcome"]

logger = get_logger(__name__)

History = Sequence[tuple[str, str]] | None


def add_reason(reasons: list[DegradedReason], reason: DegradedReason | None) -> None:
    if reason is not None and reason not in reasons:
        reasons.append(reason)


@dataclass(slots=True)
class JudgeOutcome:
    """判定结果 + 失败信息。

    ``verdict`` 始终存在（即使失败），是为了让调用方不需要到处判空；
    但「失败」必须通过 ``degraded_reason`` 显式表达 ——
    score=0.0 既可能是「真的无关」，也可能是「judge 挂了」，
    两者对后续决策的含义完全不同。
    """

    verdict: JudgeVerdict
    degraded_reason: DegradedReason | None = None
    error: str = ""


def _candidate_text(item: RetrievedItem, limit: int = 300) -> str:
    text = " ".join(item.text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    header = f"{item.item_type} | {item.heading_text or '（无章节）'}"
    return f"{header}\n{text}"


class Judge:
    """LLM 对「这批资料能否回答问题」打 0–1 分。"""

    def __init__(self, llm: LLMProvider, settings: Settings) -> None:
        self.llm = llm
        self._settings = settings

    async def score(
        self, question: str, items: list[RetrievedItem]
    ) -> JudgeOutcome:
        cfg = self._settings.agent
        started = Stopwatch()
        threshold = cfg.relevance_threshold

        if not items:
            return JudgeOutcome(
                JudgeVerdict(
                    score=0.0,
                    reason="没有候选资料",
                    threshold=threshold,
                    passed=False,
                    latency_ms=started.ms(),
                )
            )

        listing = "\n\n".join(
            f"[{index}] {_candidate_text(item)}"
            for index, item in enumerate(items[: cfg.judge_max_context_items], start=1)
        )
        try:
            result = await self.llm.chat(
                [
                    ChatMessage.system(JUDGE_SYSTEM_PROMPT),
                    ChatMessage.user(f"问题：{question}\n\n候选资料：\n{listing}"),
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001 - 判定失败必须降级而不是中断问答
            logger.warning("相关性判定失败，跳过重检", extra={"error": str(exc)[:200]})
            outcome = JudgeOutcome(
                JudgeVerdict(
                    score=0.0,
                    reason="judge 不可用",
                    threshold=threshold,
                    passed=False,
                    latency_ms=started.ms(),
                ),
                degraded_reason=DegradedReason.JUDGE_UNAVAILABLE,
                error=str(exc)[:500],
            )
            get_metrics().observe("qa.judge", outcome.verdict.latency_ms, error=True)
            return outcome

        data = extract_json_object(result.text or "")
        try:
            score = float(data["score"]) if data else None
        except (KeyError, TypeError, ValueError):
            score = None
        if score is None:
            logger.warning(
                "相关性判定返回无法解析的内容",
                extra={"raw": (result.text or "")[:200]},
            )
            return JudgeOutcome(
                JudgeVerdict(
                    score=0.0,
                    reason="judge 返回无法解析",
                    threshold=threshold,
                    passed=False,
                    latency_ms=started.ms(),
                ),
                degraded_reason=DegradedReason.JUDGE_UNAVAILABLE,
                error="无法解析 judge 输出",
            )

        score = max(0.0, min(1.0, score))
        verdict = JudgeVerdict(
            score=score,
            reason=str(data.get("reason", ""))[:300],
            threshold=threshold,
            passed=score >= threshold,
            latency_ms=started.ms(),
        )
        get_metrics().observe("qa.judge", verdict.latency_ms)
        return JudgeOutcome(verdict)


@dataclass(slots=True)
class _Prepared:
    decision: RouterDecision
    standalone: StandaloneQuery
    outcome: SearchOutcome
    judge: JudgeOutcome | None
    rounds: list[DebugRound] = field(default_factory=list)
    degraded: list[DegradedReason] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    judge_score: float | None = None


class AnswerAgent:
    """问答编排器。依赖全部注入，测试里不需要网络与真实存储。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        llm: LLMProvider | None = None,
        router: Router | None = None,
        judge: Judge | None = None,
        tools: SearchTools | None = None,
        judge_enabled: bool | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.generator = generator
        self.router = router or Router(llm, self._settings)
        enabled = (
            self._settings.agent.judge_enabled
            if judge_enabled is None
            else judge_enabled
        )
        if judge is not None:
            self.judge: Judge | None = judge
        elif llm is not None and enabled:
            self.judge = Judge(llm, self._settings)
        else:
            self.judge = None
        self.tools = tools or SearchTools(retriever)

    # ------------------------------------------------------------------ 非流式

    async def answer(
        self,
        kb: KnowledgeBase,
        question: str,
        *,
        history: History = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        rewrite: bool | None = None,
        session_id: str = "",
    ) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValidationError("问题不能为空")

        started = Stopwatch()
        decision = await self.router.route(question)

        if self._is_smalltalk(decision):
            generation = await self.generator.answer_smalltalk(
                question, history=history
            )
            return self._smalltalk_result(
                kb, generation, session_id=session_id, latency_ms=started.ms()
            )

        standalone = await self.router.standalone_query(question, history)
        if standalone.rewritten:
            # 压缩后的独立问题才是真正要检索的东西，重新路由一次
            decision = await self.router.route(standalone.text)

        prepared = await self._retrieve(
            kb,
            question,
            decision,
            standalone,
            top_k=top_k,
            use_rerank=use_rerank,
            rewrite=rewrite,
        )

        generation = await self.generator.answer(
            question,
            prepared.outcome.items,
            history=history,
            judge_score=prepared.judge_score,
            reranked=prepared.outcome.reranked,
        )
        reasons = list(prepared.degraded)
        for reason in generation.degraded_reasons:
            add_reason(reasons, reason)

        logger.info(
            "问答完成",
            extra={
                "kb_id": kb.kb_id,
                "intent": str(decision.intent),
                "rounds": len(prepared.rounds),
                "items": len(prepared.outcome.items),
                "citations": len(generation.citations),
                "degraded": [str(reason) for reason in reasons],
                "total_ms": started.ms(),
            },
        )
        result = AnswerResult(
            answer_markdown=generation.answer_markdown,
            citations=generation.citations,
            confidence=generation.confidence,
            used_rounds=len(prepared.rounds),
            latency_ms=started.ms(),
            degraded=bool(reasons),
            degraded_reasons=reasons,
            debug_trace=prepared.rounds,
            session_id=session_id,
            kb_id=kb.kb_id,
        )
        get_metrics().observe("qa.total", result.latency_ms, error=False)
        get_metrics().incr("qa.requests")
        get_metrics().incr("qa.rounds", len(prepared.rounds))
        return result

    # ------------------------------------------------------------------ 流式

    async def answer_stream(
        self,
        kb: KnowledgeBase,
        question: str,
        *,
        history: History = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        rewrite: bool | None = None,
        session_id: str = "",
    ) -> AsyncIterator[AnswerEvent]:
        """逐事件产出 :class:`AnswerEvent`。

        检索与判定必须在第一个 token 之前完成（生成需要最终 context），
        所以 ``meta`` 事件先告诉调用方这次路由到了哪个意图、跑了几轮；
        之后才是真正的流式生成。
        """
        question = question.strip()
        if not question:
            raise ValidationError("问题不能为空")

        started = Stopwatch()
        decision = await self.router.route(question)

        if self._is_smalltalk(decision):
            yield self._meta_event(decision, used_rounds=0, prepared=None)
            final: dict[str, Any] = {}
            async for event in self.generator.answer_smalltalk_stream(
                question, history=history
            ):
                if event.type == "done":
                    final = event.data
                    continue
                yield event
            latency_ms = started.ms()
            get_metrics().observe("qa.total", latency_ms)
            get_metrics().incr("qa.smalltalk")
            yield AnswerEvent(
                "done",
                data={
                    "kb_id": kb.kb_id,
                    "session_id": session_id,
                    "intent": str(decision.intent),
                    "used_rounds": 0,
                    "confidence": final.get("confidence", 0.0),
                    "citation_count": final.get("citation_count", 0),
                    "degraded_reasons": final.get("degraded_reasons", []),
                    "latency_ms": latency_ms,
                    "debug_trace": [],
                },
            )
            return

        standalone = await self.router.standalone_query(question, history)
        if standalone.rewritten:
            decision = await self.router.route(standalone.text)

        prepared = await self._retrieve(
            kb,
            question,
            decision,
            standalone,
            top_k=top_k,
            use_rerank=use_rerank,
            rewrite=rewrite,
        )
        yield self._meta_event(decision, used_rounds=len(prepared.rounds), prepared=prepared)

        final = {}
        async for event in self.generator.answer_stream(
            question,
            prepared.outcome.items,
            history=history,
            judge_score=prepared.judge_score,
            reranked=prepared.outcome.reranked,
        ):
            if event.type == "done":
                final = event.data
                continue
            yield event

        reasons = list(prepared.degraded)
        for raw in final.get("degraded_reasons", []):
            try:
                add_reason(reasons, DegradedReason(raw))
            except ValueError:  # pragma: no cover - provider 返回了未知字符串
                logger.warning("未知的降级原因", extra={"reason": raw})

        latency_ms = started.ms()
        get_metrics().observe("qa.total", latency_ms)
        get_metrics().incr("qa.requests")
        get_metrics().incr("qa.rounds", len(prepared.rounds))
        yield AnswerEvent(
            "done",
            data={
                "kb_id": kb.kb_id,
                "session_id": session_id,
                "intent": str(decision.intent),
                "used_rounds": len(prepared.rounds),
                "confidence": final.get("confidence", 0.0),
                "citation_count": final.get("citation_count", 0),
                "degraded_reasons": [str(reason) for reason in reasons],
                "latency_ms": latency_ms,
                "debug_trace": [
                    debug.model_dump(mode="json") for debug in prepared.rounds
                ],
            },
        )

    # ------------------------------------------------------------------ 检索循环

    async def _retrieve(
        self,
        kb: KnowledgeBase,
        question: str,
        decision: RouterDecision,
        standalone: StandaloneQuery,
        *,
        top_k: int | None,
        use_rerank: bool | None,
        rewrite: bool | None,
    ) -> _Prepared:
        cfg = self._settings.agent
        want_rewrite = cfg.query_rewrite_enabled if rewrite is None else rewrite
        degraded: list[DegradedReason] = []
        add_reason(degraded, standalone.degraded_reason)

        current = standalone.text
        used_queries = {current}
        rounds: list[DebugRound] = []
        calls: list[ToolCall] = []
        best_outcome: SearchOutcome | None = None
        best_judge: JudgeOutcome | None = None
        best_score: float | None = None

        for round_no in range(1, cfg.max_rounds + 1):
            outcome, call = await self.tools.run(
                decision.intent,
                kb,
                current,
                top_k=top_k,
                use_rerank=use_rerank,
                # 只有第一轮做多查询改写：第二轮的 query 本身就是第一轮
                # 改写出来的，再改写一次只会把它越推越远
                rewrite=want_rewrite and round_no == 1,
            )
            calls.append(call)
            for reason in outcome.degraded_reasons:
                add_reason(degraded, reason)

            judge_outcome: JudgeOutcome | None = None
            if self.judge is not None:
                judge_outcome = await self.judge.score(question, outcome.items)
                add_reason(degraded, judge_outcome.degraded_reason)

            score = judge_outcome.verdict.score if judge_outcome else None
            if best_outcome is None or (
                score is not None and (best_score is None or score > best_score)
            ):
                best_outcome = outcome
                best_judge = judge_outcome
                best_score = score

            rounds.append(
                self._debug_round(round_no, current, outcome, judge_outcome)
            )
            logger.info(
                "agent 检索轮次",
                extra={
                    "round": round_no,
                    "intent": str(decision.intent),
                    "tool": str(call.name),
                    "query": current,
                    "items": call.item_count,
                    "judge_score": score,
                    "judge_passed": (
                        judge_outcome.verdict.passed if judge_outcome else None
                    ),
                    "tool_ms": call.latency_ms,
                },
            )

            if judge_outcome is None:
                break
            if judge_outcome.degraded_reason is not None:
                # judge 挂了就不能判断「换 query 会不会更好」，再跑一轮是赌博
                break
            if judge_outcome.verdict.passed:
                break
            if round_no >= cfg.max_rounds:
                break

            next_query = next(
                (q for q in outcome.queries[1:] if q and q not in used_queries), None
            )
            if next_query is None:
                logger.info(
                    "没有新的改写查询可用于重检，提前结束",
                    extra={"round": round_no, "queries": outcome.queries},
                )
                break
            used_queries.add(next_query)
            current = next_query

        assert best_outcome is not None  # max_rounds >= 1，循环至少执行一次
        return _Prepared(
            decision=decision,
            standalone=standalone,
            outcome=best_outcome,
            judge=best_judge,
            rounds=rounds,
            degraded=degraded,
            tool_calls=calls,
            judge_score=best_score,
        )

    def _debug_round(
        self,
        round_no: int,
        query: str,
        outcome: SearchOutcome,
        judge_outcome: JudgeOutcome | None,
    ) -> DebugRound:
        timings = dict(outcome.timings)
        timings["judge"] = (
            judge_outcome.verdict.latency_ms if judge_outcome is not None else 0
        )
        reasons = list(outcome.degraded_reasons)
        if judge_outcome is not None:
            add_reason(reasons, judge_outcome.degraded_reason)
        return DebugRound(
            round=round_no,
            query=query,
            rewrites=list(outcome.queries[1:]),
            candidate_counts={
                str(result.route): result.count for result in outcome.route_results
            },
            fused_unique=outcome.fused_count,
            reranked=len(outcome.items),
            judge=judge_outcome.verdict if judge_outcome is not None else None,
            timings_ms=timings,
            degraded_reasons=reasons,
        )

    # ------------------------------------------------------------------ 小工具

    def _is_smalltalk(self, decision: RouterDecision) -> bool:
        return (
            decision.intent is Intent.SMALLTALK
            and self._settings.agent.smalltalk_skip_retrieval
        )

    def _smalltalk_result(
        self,
        kb: KnowledgeBase,
        generation: GenerationOutcome,
        *,
        session_id: str,
        latency_ms: int,
    ) -> AnswerResult:
        return AnswerResult(
            answer_markdown=generation.answer_markdown,
            citations=[],
            confidence=generation.confidence,
            used_rounds=0,
            latency_ms=latency_ms,
            degraded=bool(generation.degraded_reasons),
            degraded_reasons=generation.degraded_reasons,
            debug_trace=[],
            session_id=session_id,
            kb_id=kb.kb_id,
        )

    def _meta_event(
        self,
        decision: RouterDecision,
        *,
        used_rounds: int,
        prepared: _Prepared | None,
    ) -> AnswerEvent:
        data: dict[str, Any] = {
            "intent": str(decision.intent),
            "intent_source": decision.source,
            "used_rounds": used_rounds,
            "standalone_query": prepared.standalone.text if prepared else "",
            "queries": list(prepared.outcome.queries) if prepared else [],
            "items": len(prepared.outcome.items) if prepared else 0,
            "tool_calls": [
                {
                    "name": str(call.name),
                    "query": call.query,
                    "routes": list(call.routes),
                    "item_count": call.item_count,
                    "latency_ms": call.latency_ms,
                }
                for call in (prepared.tool_calls if prepared else [])
            ],
            "degraded_reasons": (
                [str(reason) for reason in prepared.degraded] if prepared else []
            ),
            "debug_trace": (
                [debug.model_dump(mode="json") for debug in prepared.rounds]
                if prepared
                else []
            ),
        }
        return AnswerEvent("meta", data=data)
