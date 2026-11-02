"""答案生成：检索结果 → 带可溯源引用的回答。

引用是**反解析**出来的，不是让模型自报的
------------------------------------------
设计稿里模型输出 ``{answer_markdown, citations[], confidence}``。
本实现改为输出纯 Markdown，再由代码扫描正文中的 ``[n]`` 生成 citations。
原因有三：

1. **两者不可能不一致。** 如果模型返回的 ``citations`` 列表与正文里的 ``[n]``
   对不上（多报、少报、编号错位），前端要么显示一条正文根本没引用的来源，
   要么显示一个点不开的 ``[n]``。``[n]`` 是用户唯一看得见的事实，
   以它为准才不会有这类界面级的谎言。
2. **流式可用。** JSON 在生成到一半时无法可靠解析；纯文本可以边收边发。
   SSE 与一次性回答因此共用同一套提示词与解析逻辑。
3. **降级路径可测。** 引用完全由确定性代码产生，单测不需要模拟模型行为。

置信度不是让模型自评（模型对「自己编没编」的判断并不可靠），而是由
相关性打分 + 引用覆盖率计算，公式见 :func:`compute_confidence`。

降级
----
生成模型不可用时，按设计稿的降级矩阵返回检索到的资料原文与完整引用列表，
并标记 ``llm_unavailable`` —— 有原文可读，比一句「服务不可用」有用。
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from app.config import Settings
from app.core.logging import get_logger
from app.core.perf import get_metrics
from app.core.schemas import Citation, DegradedReason, RetrievedItem
from app.generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    REFUSAL_MARKER,
    REFUSAL_TEXT,
    SMALLTALK_FALLBACK_TEXT,
    SMALLTALK_SYSTEM_PROMPT,
    answer_user_prompt,
    context_block,
)
from app.providers.base import ChatMessage, LLMProvider
from app.retrieval.base import Stopwatch

__all__ = [
    "AnswerEvent",
    "AnswerGenerator",
    "GenerationOutcome",
    "compute_confidence",
    "parse_citations",
]

logger = get_logger(__name__)

#: ``[n]``，n 为 1–999。只接受这种最朴素的写法：模型很擅长在别的括号样式上
#: 即兴发挥，放宽解析会让「[图1]」「【1】」这类文本被误当成引用。
_CITATION_RE = re.compile(r"\[(\d{1,3})\]")

#: 围栏代码块与行内代码。解析引用前先剔除 —— 代码里的 ``arr[1]``、``x[2]``
#: 是下标，不是引用；不剔除就会凭空多出一条「来源」。
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

#: 送给模型前，每条资料的正文最多占多少字符由调用方预算控制；
#: 这里只限制 citation 摘要长度（配置项 GEN_SNIPPET_CHARS）。
_ELLIPSIS = "…"


@dataclass(slots=True)
class GenerationOutcome:
    answer_markdown: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    degraded_reasons: list[DegradedReason] = field(default_factory=list)
    latency_ms: int = 0


@dataclass(slots=True)
class AnswerEvent:
    """流式事件。Batch 7 的 SSE 层只负责把它序列化，不做业务判断。"""

    type: Literal["meta", "token", "reasoning", "citations", "done", "error"]
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.text:
            payload["text"] = self.text
        if self.data:
            payload["data"] = self.data
        return payload


# --------------------------------------------------------------------------------------
# 文本处理
# --------------------------------------------------------------------------------------


def _strip_code(text: str) -> str:
    """去掉围栏代码块与行内代码，避免把数组下标当成引用。"""
    return _INLINE_CODE_RE.sub(" ", _FENCED_CODE_RE.sub(" ", text))


def _strip_heading_prefix(item: RetrievedItem) -> str:
    """去掉正文开头的标题路径。

    分块在切分阶段已经带了标题前缀，而 prompt 的元数据行里又有章节，
    两处重复会白占上下文预算、也稀释正文权重。
    """
    text = item.text.strip()
    heading = item.heading_text
    if heading and text.startswith(heading):
        text = text[len(heading) :].strip().lstrip("|").strip()
    return text


def _snippet(item: RetrievedItem, limit: int) -> str:
    text = " ".join(_strip_heading_prefix(item).split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + _ELLIPSIS


def _add_reason(reasons: list[DegradedReason], reason: DegradedReason | None) -> None:
    if reason is not None and reason not in reasons:
        reasons.append(reason)


def is_refusal(text: str) -> bool:
    return REFUSAL_MARKER in text


def build_context(
    items: list[RetrievedItem],
    *,
    max_chars: int,
    max_items: int,
) -> tuple[list[RetrievedItem], str]:
    """按字符预算选取资料并编号，返回 ``(入选条目, context 文本)``。

    只保留**完整参与编号**的条目：被预算截掉的条目不进入 citations，
    因此不会出现「答案引用了 [9]，但引用列表里没有 9」。
    最后一条放不下时截断它的正文而不是丢弃 —— 半条资料通常仍比没有有用。
    """
    selected = items[:max_items]
    kept: list[RetrievedItem] = []
    blocks: list[str] = []
    used = 0

    for index, item in enumerate(selected, start=1):
        body = _strip_heading_prefix(item) or item.text.strip()
        # 先量出「这条完整放进来」的长度，再决定是否截断：
        # 预算计算的单位必须与真正喂给模型的文本完全一致
        full_block = context_block(index, item, body)
        remaining = max_chars - used - len(full_block) + len(body)
        if remaining <= 40:
            if kept:
                break
            # 第一条就超预算（CONTEXT_MAX_CHARS 配得极小）：至少给出一段正文，
            # 否则整个回答会退化成「知识库中未找到相关内容」，与检索结果矛盾
            remaining = 200
        if len(body) > remaining:
            body = body[:remaining].rstrip() + _ELLIPSIS
        block = context_block(index, item, body)
        if kept and used + len(block) > max_chars:
            break
        kept.append(item)
        blocks.append(block)
        used += len(block) + 2

    return kept, "\n\n".join(blocks)


def parse_citations(
    answer: str,
    items: list[RetrievedItem],
    *,
    snippet_chars: int,
    max_citations: int,
) -> list[Citation]:
    """从答案正文的 ``[n]`` 反解析引用列表。

    * 顺序按**首次出现**，与用户阅读顺序一致；
    * 越界编号（``[9]`` 但只有 5 条资料）直接忽略，不报错 ——
      模型多写一个编号不该让整次回答失败；
    * 重复引用只产生一条 citation。
    """
    seen: set[int] = set()
    ordered: list[int] = []
    for match in _CITATION_RE.finditer(_strip_code(answer)):
        index = int(match.group(1))
        if 1 <= index <= len(items) and index not in seen:
            seen.add(index)
            ordered.append(index)
            if len(ordered) >= max_citations:
                break

    citations: list[Citation] = []
    for index in ordered:
        item = items[index - 1]
        citations.append(
            Citation(
                index=index,
                item_id=item.item_id,
                item_type=item.item_type,
                doc_id=item.doc_id,
                doc_name=item.source_name,
                page=item.page,
                heading_path=list(item.heading_path),
                snippet=_snippet(item, snippet_chars),
                image_url=item.image_url,
                image_id=item.image_id,
                score=(
                    item.rerank_score
                    if item.rerank_score is not None
                    else item.rrf_score
                ),
                rerank_score=item.rerank_score,
            )
        )
    return citations


def citations_for_all(
    items: list[RetrievedItem],
    *,
    snippet_chars: int,
    max_citations: int,
) -> list[Citation]:
    """给降级答案用：把入选资料按顺序全部作为引用返回。"""
    numbered_answer = " ".join(f"[{i}]" for i in range(1, len(items) + 1))
    return parse_citations(
        numbered_answer,
        items,
        snippet_chars=snippet_chars,
        max_citations=max_citations,
    )


def compute_confidence(
    *,
    judge_score: float | None,
    cited: int,
    available: int,
    reranked: bool,
) -> float:
    """置信度 = 0.6×相关性 + 0.3×引用覆盖率 + 0.1×排序质量。

    各分量的理由：

    * **相关性**来自 judge 对「资料能否回答问题」的打分，没有 judge 时取 0.6
      （既不奖励也不惩罚）；
    * **引用覆盖率** = 被引用条数 / min(可用条数, 4)，封顶 1。只引用一条的
      长答案，通常意味着其余资料没被用上，可信度理应更低；
    * **排序质量**：重排生效才给满 0.1，纯 RRF 时最高只能到 0.9。

    这不是概率，而是排序用的启发式分值；前端只展示，不做阈值判断。
    """
    if available <= 0:
        return 0.0
    grounding = 0.6 if judge_score is None else max(0.0, min(1.0, judge_score))
    coverage = min(1.0, cited / max(1, min(available, 4)))
    score = 0.6 * grounding + 0.3 * coverage + (0.1 if reranked else 0.0)
    return round(max(0.0, min(1.0, score)), 3)


# --------------------------------------------------------------------------------------
# 生成器
# --------------------------------------------------------------------------------------


History = Sequence[tuple[str, str]]


class AnswerGenerator:
    """把回答生成拆成「准备 prompt」与「后处理」两半，流式与非流式共用。"""

    def __init__(self, llm: LLMProvider, settings: Settings) -> None:
        self.llm = llm
        self._settings = settings

    # ------------------------------------------------------------------ 准备

    def _prepare_context(
        self, items: list[RetrievedItem]
    ) -> tuple[list[RetrievedItem], str]:
        cfg = self._settings.generation
        return build_context(
            items,
            max_chars=self._settings.retrieval.context_max_chars,
            max_items=cfg.max_citations,
        )

    def _history_messages(self, history: History | None) -> list[ChatMessage]:
        if not history:
            return []
        turns = self._settings.agent.history_max_turns
        recent = list(history)[-turns * 2 :] if turns else []
        messages: list[ChatMessage] = []
        for role, content in recent:
            text = content.strip()
            if not text:
                continue
            if role == "assistant":
                messages.append(ChatMessage.assistant(text))
            else:
                messages.append(ChatMessage.user(text))
        return messages

    def _answer_messages(
        self, question: str, context: str, history: History | None
    ) -> list[ChatMessage]:
        messages = [ChatMessage.system(ANSWER_SYSTEM_PROMPT)]
        messages.extend(self._history_messages(history))
        messages.append(ChatMessage.user(answer_user_prompt(question, context)))
        return messages

    def _smalltalk_messages(
        self, question: str, history: History | None
    ) -> list[ChatMessage]:
        messages = [ChatMessage.system(SMALLTALK_SYSTEM_PROMPT)]
        messages.extend(self._history_messages(history))
        messages.append(ChatMessage.user(question))
        return messages

    # ------------------------------------------------------------------ 后处理

    def _finalize(
        self,
        text: str,
        items: list[RetrievedItem],
        *,
        judge_score: float | None,
        reranked: bool,
        latency_ms: int,
        reasons: list[DegradedReason] | None = None,
    ) -> GenerationOutcome:
        cfg = self._settings.generation
        degraded = list(reasons or [])
        refusal = is_refusal(text)

        citations = parse_citations(
            text,
            items,
            snippet_chars=cfg.snippet_chars,
            max_citations=cfg.max_citations,
        )
        if cfg.require_citations and items and not citations and not refusal:
            # 答案给了结论却一个来源都没标。不能伪造引用，只能如实标记，
            # 并把置信度按「零覆盖率」计算 —— 这正是 REVIEW 时最该看见的信号
            _add_reason(degraded, DegradedReason.CITATION_MISSING)

        confidence = (
            0.0
            if refusal or not items
            else compute_confidence(
                judge_score=judge_score,
                cited=len(citations),
                available=len(items),
                reranked=reranked,
            )
        )
        return GenerationOutcome(
            answer_markdown=text.strip(),
            citations=citations,
            confidence=confidence,
            degraded_reasons=degraded,
            latency_ms=latency_ms,
        )

    def _refusal(self) -> GenerationOutcome:
        return GenerationOutcome(answer_markdown=REFUSAL_TEXT)

    def _fallback(
        self,
        items: list[RetrievedItem],
        *,
        latency_ms: int,
        error: str,
        partial: str = "",
    ) -> GenerationOutcome:
        cfg = self._settings.generation
        lines: list[str] = []
        if partial.strip():
            lines.extend([partial.strip(), ""])
        lines.append("生成模型暂时不可用，以下是检索到的资料原文：")
        lines.append("")
        for index, item in enumerate(items, start=1):
            title = item.source_name or item.doc_id
            if item.heading_path:
                title += " · " + " > ".join(item.heading_path)
            lines.append(f"### [{index}] {title}")
            lines.append(_snippet(item, max(cfg.snippet_chars, 200)))
            lines.append("")
        return GenerationOutcome(
            answer_markdown="\n".join(lines).strip(),
            citations=citations_for_all(
                items,
                snippet_chars=cfg.snippet_chars,
                max_citations=cfg.max_citations,
            ),
            confidence=0.0,
            degraded_reasons=[DegradedReason.LLM_UNAVAILABLE],
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------ 非流式

    async def answer(
        self,
        question: str,
        items: list[RetrievedItem],
        *,
        history: History | None = None,
        judge_score: float | None = None,
        reranked: bool = False,
    ) -> GenerationOutcome:
        started = Stopwatch()
        cfg = self._settings.generation

        if not items and cfg.refuse_when_empty:
            # 没有资料时不调模型：让模型在空 context 上「尽量回答」，
            # 正是幻觉率最高的用法
            return self._refusal()

        kept, context = self._prepare_context(items)
        if not kept and cfg.refuse_when_empty:
            return self._refusal()

        messages = self._answer_messages(question, context, history)
        try:
            result = await self.llm.chat(
                messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            text = (result.text or "").strip()
            if not text:
                raise ValueError("模型返回空内容")
        except Exception as exc:  # noqa: BLE001 - 生成失败必须降级而不是 500
            logger.warning(
                "答案生成失败，降级为资料原文",
                extra={"provider": self.llm.name, "error": str(exc)[:200]},
            )
            outcome = self._fallback(
                kept, latency_ms=started.ms(), error=str(exc)[:500]
            )
            get_metrics().observe("generation.answer", outcome.latency_ms, error=True)
            return outcome

        outcome = self._finalize(
            text,
            kept,
            judge_score=judge_score,
            reranked=reranked,
            latency_ms=started.ms(),
        )
        get_metrics().observe("generation.answer", outcome.latency_ms)
        get_metrics().incr("generation.answers")
        return outcome

    # ------------------------------------------------------------------ 流式

    async def answer_stream(
        self,
        question: str,
        items: list[RetrievedItem],
        *,
        history: History | None = None,
        judge_score: float | None = None,
        reranked: bool = False,
    ) -> AsyncIterator[AnswerEvent]:
        """先发 token，结束时发一条 ``citations``，最后 ``done``。

        流到一半失败时：已经发出的内容保留（用户已经看到了），
        在其后追加降级说明；一个字都没发出去才整体回退为资料原文。
        """
        started = Stopwatch()
        cfg = self._settings.generation

        if not items and cfg.refuse_when_empty:
            yield AnswerEvent("token", text=REFUSAL_TEXT)
            yield self._citations_event(self._refusal())
            yield self._done_event(self._refusal())
            return

        kept, context = self._prepare_context(items)
        if not kept and cfg.refuse_when_empty:
            outcome = self._refusal()
            yield AnswerEvent("token", text=outcome.answer_markdown)
            yield self._citations_event(outcome)
            yield self._done_event(outcome)
            return

        messages = self._answer_messages(question, context, history)
        chunks: list[str] = []
        error = ""
        try:
            async for delta in self.llm.chat_stream(
                messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            ):
                if delta.reasoning_text:
                    yield AnswerEvent("reasoning", text=delta.reasoning_text)
                if delta.text:
                    chunks.append(delta.text)
                    yield AnswerEvent("token", text=delta.text)
        except Exception as exc:  # noqa: BLE001 - 流中断不能把连接直接掐掉
            error = str(exc)[:500]
            logger.warning(
                "流式生成中断",
                extra={"provider": self.llm.name, "error": error[:200]},
            )

        full = "".join(chunks).strip()
        if full:
            outcome = self._finalize(
                full,
                kept,
                judge_score=judge_score,
                reranked=reranked,
                latency_ms=started.ms(),
                reasons=[DegradedReason.LLM_UNAVAILABLE] if error else None,
            )
            get_metrics().observe("generation.answer", outcome.latency_ms)
            get_metrics().incr("generation.answers")
            yield self._citations_event(outcome)
            yield self._done_event(outcome)
            return

        outcome = self._fallback(kept, latency_ms=started.ms(), error=error or "空回复")
        get_metrics().observe("generation.answer", outcome.latency_ms, error=True)
        yield AnswerEvent("token", text=outcome.answer_markdown)
        yield self._citations_event(outcome)
        yield self._done_event(outcome)

    # ------------------------------------------------------------------ 寒暄

    async def answer_smalltalk(
        self, question: str, *, history: History | None = None
    ) -> GenerationOutcome:
        started = Stopwatch()
        cfg = self._settings.generation
        messages = self._smalltalk_messages(question, history)
        try:
            result = await self.llm.chat(
                messages, temperature=cfg.temperature, max_tokens=512
            )
            text = (result.text or "").strip()
            if not text:
                raise ValueError("模型返回空内容")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "寒暄生成失败", extra={"provider": self.llm.name, "error": str(exc)[:200]}
            )
            return GenerationOutcome(
                answer_markdown=SMALLTALK_FALLBACK_TEXT,
                degraded_reasons=[DegradedReason.LLM_UNAVAILABLE],
                latency_ms=started.ms(),
            )
        return GenerationOutcome(
            answer_markdown=text,
            confidence=0.0,
            latency_ms=started.ms(),
        )

    async def answer_smalltalk_stream(
        self, question: str, *, history: History | None = None
    ) -> AsyncIterator[AnswerEvent]:
        started = Stopwatch()
        cfg = self._settings.generation
        messages = self._smalltalk_messages(question, history)
        chunks: list[str] = []
        error = ""
        try:
            async for delta in self.llm.chat_stream(
                messages, temperature=cfg.temperature, max_tokens=512
            ):
                if delta.reasoning_text:
                    yield AnswerEvent("reasoning", text=delta.reasoning_text)
                if delta.text:
                    chunks.append(delta.text)
                    yield AnswerEvent("token", text=delta.text)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)[:500]
            logger.warning(
                "流式寒暄生成失败",
                extra={"provider": self.llm.name, "error": error[:200]},
            )

        text = "".join(chunks).strip()
        if not text:
            text = SMALLTALK_FALLBACK_TEXT
            yield AnswerEvent("token", text=text)
        outcome = GenerationOutcome(
            answer_markdown=text,
            confidence=0.0,
            degraded_reasons=[DegradedReason.LLM_UNAVAILABLE] if error else [],
            latency_ms=started.ms(),
        )
        yield self._citations_event(outcome)
        yield self._done_event(outcome)

    # ------------------------------------------------------------------ 事件

    @staticmethod
    def _citations_event(outcome: GenerationOutcome) -> AnswerEvent:
        return AnswerEvent(
            "citations",
            data={
                "citations": [
                    citation.model_dump(mode="json") for citation in outcome.citations
                ],
                "confidence": outcome.confidence,
                "degraded_reasons": [
                    str(reason) for reason in outcome.degraded_reasons
                ],
            },
        )

    @staticmethod
    def _done_event(outcome: GenerationOutcome) -> AnswerEvent:
        return AnswerEvent(
            "done",
            data={
                "confidence": outcome.confidence,
                "citation_count": len(outcome.citations),
                "degraded_reasons": [
                    str(reason) for reason in outcome.degraded_reasons
                ],
                "latency_ms": outcome.latency_ms,
            },
        )
