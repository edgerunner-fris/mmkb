"""生成层 + Agent 层测试。

全部离线：不碰任何厂商 API，不碰向量库。检索器与 LLM 都是可编程假实现，
因此可以精确控制「第几轮返回什么」，验证的是**决策逻辑**：

1. 引用必须从答案正文的 ``[n]`` 反解析 —— 正文与引用列表不允许不一致；
2. 代码块里的 ``arr[1]`` 不是引用；
3. 上下文预算截断后编号仍然连续，不会出现指向不存在资料的 ``[n]``；
4. 生成失败必须降级为资料原文，而不是抛异常；
5. 相关性判定不达标才重检，最多两轮，且取「历次最好的一轮」；
6. 寒暄不检索、以图搜图只跑图像 + BM25；
7. 每个降级原因都要出现在最终 ``degraded_reasons`` 里。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agent.graph import AnswerAgent, Judge
from app.agent.router import Intent, Router
from app.agent.tools import SearchTools, ToolName
from app.config import (
    AgentSettings,
    BlobSettings,
    BM25Settings,
    DocStoreSettings,
    GenerationSettings,
    ImageEmbeddingSettings,
    PathSettings,
    RetrievalSettings,
    Settings,
    TextEmbeddingSettings,
    VectorStoreSettings,
)
from app.core.schemas import (
    DegradedReason,
    ItemType,
    KnowledgeBase,
    RetrievedItem,
    Route,
)
from app.generation.answer import (
    AnswerGenerator,
    build_context,
    compute_confidence,
    parse_citations,
)
from app.generation.prompts import REFUSAL_MARKER
from app.providers.base import (
    ChatDelta,
    ChatMessage,
    ChatResult,
    LLMProvider,
)
from app.retrieval.base import RouteResult
from app.retrieval.hybrid import SearchOutcome

KB_ID = "kb_000000000001"
DOC_ID = "doc_000000000001"


# --------------------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------------------


def _settings(
    tmp_path: Path,
    *,
    agent: AgentSettings | None = None,
    generation: GenerationSettings | None = None,
    context_max_chars: int = 4000,
) -> Settings:
    return Settings(
        paths=PathSettings(data_dir=tmp_path),
        docstore=DocStoreSettings(sqlite_path=tmp_path / "test.db"),
        blob=BlobSettings(images_dir=tmp_path / "images"),
        bm25=BM25Settings(index_dir=tmp_path / "bm25", persist=False),
        vectorstore=VectorStoreSettings(chroma_dir=tmp_path / "chroma"),
        text_embed=TextEmbeddingSettings(model="fake-text", dim=8),
        image_embed=ImageEmbeddingSettings(model="fake-image", dim=16),
        retrieval=RetrievalSettings(
            enable_rerank=False,
            final_top_k=8,
            context_max_chars=context_max_chars,
        ),
        agent=agent
        or AgentSettings(
            max_rounds=2,
            relevance_threshold=0.6,
            judge_enabled=True,
            query_rewrite_enabled=False,
            history_compress_enabled=True,
            history_compress_trigger_chars=100,
        ),
        generation=generation
        or GenerationSettings(temperature=0.0, max_tokens=512, snippet_chars=60),
    )


def _kb() -> KnowledgeBase:
    return KnowledgeBase(
        kb_id=KB_ID,
        name="demo",
        text_collection=f"{KB_ID}__text",
        image_collection=f"{KB_ID}__image",
        text_dim=8,
        image_dim=16,
        llm_model="fake-llm-model",
        text_embed_model="fake-text",
        image_embed_model="fake-image",
        rerank_model="fake-rerank",
    )


def _item(
    index: int,
    *,
    text: str | None = None,
    item_type: ItemType = ItemType.TEXT,
    image: bool = False,
    heading: Sequence[str] = ("第一章 系统设计", "1.1 检索"),
    source: str = "design.md",
) -> RetrievedItem:
    if image:
        return RetrievedItem(
            item_id=f"img_{index:012d}",
            item_type=ItemType.IMAGE,
            kb_id=KB_ID,
            doc_id=DOC_ID,
            image_id=f"img_{index:012d}",
            text=text
            or "第一章 系统设计 > 1.1 检索 | [diagram] 三路召回架构图 | 关键词: RRF, BM25",
            rrf_score=0.02,
            heading_path=list(heading),
            page=4,
            source_name=source,
            image_url=f"/static/images/{KB_ID}/img_{index}.png",
        )
    return RetrievedItem(
        item_id=f"ck_{index:012d}",
        item_type=item_type,
        kb_id=KB_ID,
        doc_id=DOC_ID,
        chunk_id=f"ck_{index:012d}",
        text=text or "第一章 系统设计 > 1.1 检索 三路召回之后用 RRF 融合，k 取 60。",
        rrf_score=0.02,
        heading_path=list(heading),
        page=3,
        source_name=source,
    )


def _outcome(
    items: list[RetrievedItem],
    *,
    queries: Sequence[str] = ("问题",),
    reranked: bool = True,
) -> SearchOutcome:
    return SearchOutcome(
        query=queries[0],
        items=list(items),
        queries=list(queries),
        route_results=[
            RouteResult(
                route=Route.BM25,
                items=list(items),
                latency_ms=5,
                label="bm25",
            )
        ],
        timings={"retrieve": 5, "rerank": 0, "expand": 0, "hydrate": 1, "total": 6},
        reranked=reranked,
        fused_count=len(items),
    )


class FakeLLM(LLMProvider):
    """按 system prompt 分派的假模型。

    整条问答链路里同一个 provider 会被路由 / 判定 / 生成各调用一次，
    按「第几次调用」脚本化会让测试与实现顺序强绑定；按提示词分派更稳。
    """

    name = "fake-llm"
    model = "fake-llm-model"
    answer_marker = "知识库问答助手。回答时必须遵守以下规则"
    smalltalk_marker = "正在回应用户的寒暄"
    router_marker = "意图分类器"
    judge_marker = "相关性评审"
    compress_marker = "多轮对话压缩"

    def __init__(
        self,
        *,
        answers: Sequence[str] = ("根据资料，三路结果用 RRF 融合 [1]。",),
        judges: Sequence[str] = ('{"score": 0.9, "reason": "直接回答"}',),
        router: str = '{"intent": "knowledge", "confidence": 0.8, "reason": "x"}',
        compress: str = "混合检索如何融合三路结果",
        smalltalk: str = "你好，我是知识库助手。",
        boom_kind: str | None = None,
        stream_boom: bool = False,
    ) -> None:
        self.answers = list(answers)
        self.judges = list(judges)
        self.router_reply = router
        self.compress_reply = compress
        self.smalltalk_reply = smalltalk
        self.boom_kind = boom_kind
        self.stream_boom = stream_boom
        self.calls: list[str] = []
        self.messages: list[list[ChatMessage]] = []

    # -- 分派 ------------------------------------------------------------------

    @staticmethod
    def _system(messages: Sequence[ChatMessage]) -> str:
        if not messages:
            return ""
        content = messages[0].content
        return content if isinstance(content, str) else ""

    def _kind(self, messages: Sequence[ChatMessage]) -> str:
        system = self._system(messages)
        if self.judge_marker in system:
            return "judge"
        if self.router_marker in system:
            return "router"
        if self.compress_marker in system:
            return "compress"
        if self.smalltalk_marker in system:
            return "smalltalk"
        return "answer"

    def _reply(self, kind: str) -> str:
        if kind == "judge":
            if len(self.judges) > 1:
                return self.judges.pop(0)
            return self.judges[0]
        if kind == "router":
            return self.router_reply
        if kind == "compress":
            return self.compress_reply
        if kind == "smalltalk":
            return self.smalltalk_reply
        if len(self.answers) > 1:
            return self.answers.pop(0)
        return self.answers[0]

    def _maybe_boom(self, kind: str) -> None:
        if self.boom_kind in (kind, "all"):
            raise RuntimeError(f"{kind} 不可用")

    # -- LLMProvider -----------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        kind = self._kind(messages)
        self.calls.append(kind)
        self.messages.append(list(messages))
        self._maybe_boom(kind)
        return ChatResult(text=self._reply(kind), model=self.model)

    async def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatDelta]:
        kind = self._kind(messages)
        self.calls.append(kind)
        self.messages.append(list(messages))
        if self.stream_boom or self.boom_kind in (kind, "all"):
            raise RuntimeError(f"{kind} 流不可用")
        reply = self._reply(kind)
        step = max(1, len(reply) // 4)
        for start in range(0, len(reply), step):
            yield ChatDelta(text=reply[start : start + step])
        yield ChatDelta(text="", done=True)


class FakeRetriever:
    """按调用次序返回预置 SearchOutcome 的检索器。"""

    def __init__(self, outcomes: Sequence[SearchOutcome]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def search(self, kb: KnowledgeBase, query: str, **kwargs: Any) -> SearchOutcome:
        self.calls.append({"kb": kb, "query": query, **kwargs})
        index = min(len(self.calls) - 1, len(self.outcomes) - 1)
        return self.outcomes[index]


def _agent(
    settings: Settings,
    llm: FakeLLM,
    retriever: FakeRetriever,
    *,
    judge_enabled: bool | None = None,
) -> AnswerAgent:
    return AnswerAgent(
        settings,
        retriever=retriever,  # type: ignore[arg-type] - 鸭子类型，测试只用到 search
        generator=AnswerGenerator(llm, settings),
        llm=llm,
        judge_enabled=judge_enabled,
    )


# --------------------------------------------------------------------------------------
# 上下文拼装
# --------------------------------------------------------------------------------------


class TestContext:
    def test_numbers_items_in_order(self, tmp_path: Path) -> None:
        items = [_item(1), _item(2)]
        kept, context = build_context(items, max_chars=4000, max_items=8)
        assert kept == items
        assert context.startswith("[1] 类型：正文")
        assert "\n\n[2] 类型：正文" in context

    def test_strips_duplicated_heading(self, tmp_path: Path) -> None:
        _, context = build_context([_item(1)], max_chars=4000, max_items=8)
        # 正文开头的标题被剥掉，只保留元数据行里的章节，避免同一标题出现两遍
        assert context.count("第一章 系统设计 > 1.1 检索") == 1
        assert "三路召回之后用 RRF 融合" in context

    def test_budget_truncation_keeps_indices_contiguous(self) -> None:
        items = [_item(i, text="内容" * 200) for i in range(1, 5)]
        kept, context = build_context(items, max_chars=600, max_items=8)
        assert 1 <= len(kept) < len(items)
        # 未入选的条目不编号，因此不会出现指向不存在资料的 [n]
        assert f"[{len(kept)}]" in context
        assert f"[{len(kept) + 1}]" not in context

    def test_respects_max_items(self) -> None:
        items = [_item(i) for i in range(1, 6)]
        kept, _ = build_context(items, max_chars=4000, max_items=3)
        assert len(kept) == 3

    def test_first_item_always_included_even_if_over_budget(self) -> None:
        items = [_item(1, text="长" * 5000)]
        kept, context = build_context(items, max_chars=100, max_items=8)
        assert len(kept) == 1
        assert "[1]" in context


# --------------------------------------------------------------------------------------
# 引用反解析
# --------------------------------------------------------------------------------------


class TestCitationParsing:
    def test_orders_by_first_appearance_and_dedupes(self) -> None:
        items = [_item(1), _item(2), _item(3)]
        citations = parse_citations(
            "先看 [2]，再看 [1]，[2] 也适用。",
            items,
            snippet_chars=60,
            max_citations=8,
        )
        assert [c.index for c in citations] == [2, 1]
        assert citations[0].item_id == items[1].item_id

    def test_ignores_out_of_range(self) -> None:
        citations = parse_citations(
            "只有 [1] 和 [9]。", [_item(1)], snippet_chars=60, max_citations=8
        )
        assert [c.index for c in citations] == [1]

    def test_ignores_indices_inside_code(self) -> None:
        answer = "代码里 `arr[1]` 是下标。\n```python\nprint(x[2])\n```\n真正引用 [3]。"
        items = [_item(1), _item(2), _item(3)]
        citations = parse_citations(answer, items, snippet_chars=60, max_citations=8)
        assert [c.index for c in citations] == [3]

    def test_citation_fields_include_image_url(self) -> None:
        items = [_item(1), _item(2, image=True)]
        citations = parse_citations(
            "见图 [2]。", items, snippet_chars=60, max_citations=8
        )
        citation = citations[0]
        assert citation.item_type is ItemType.IMAGE
        assert citation.image_url == items[1].image_url
        assert citation.image_id == items[1].image_id
        assert citation.doc_name == "design.md"
        assert citation.page == 4
        # 摘要里不应再出现章节标题前缀
        assert citation.snippet.startswith("[diagram]")

    def test_snippet_is_truncated(self) -> None:
        item = _item(1, text="第一章 系统设计 > 1.1 检索 " + "长" * 500)
        citations = parse_citations(
            "[1]", [item], snippet_chars=30, max_citations=8
        )
        assert len(citations[0].snippet) <= 30
        assert citations[0].snippet.endswith("…")


# --------------------------------------------------------------------------------------
# 置信度
# --------------------------------------------------------------------------------------


class TestConfidence:
    def test_formula(self) -> None:
        # 0.6*0.8 + 0.3*(2/4) + 0.1 = 0.73
        assert compute_confidence(
            judge_score=0.8, cited=2, available=4, reranked=True
        ) == pytest.approx(0.73)

    def test_without_judge_or_rerank(self) -> None:
        assert compute_confidence(
            judge_score=None, cited=1, available=4, reranked=False
        ) == pytest.approx(0.6 * 0.6 + 0.3 * 0.25)

    def test_zero_without_items(self) -> None:
        assert compute_confidence(
            judge_score=1.0, cited=0, available=0, reranked=True
        ) == 0.0


# --------------------------------------------------------------------------------------
# 生成器
# --------------------------------------------------------------------------------------


class TestAnswerGenerator:
    async def test_answer_parses_citations(self, tmp_path: Path) -> None:
        llm = FakeLLM(answers=("融合公式是 RRF [1][2]。",))
        generator = AnswerGenerator(llm, _settings(tmp_path))
        outcome = await generator.answer("如何融合？", [_item(1), _item(2)])
        assert outcome.answer_markdown == "融合公式是 RRF [1][2]。"
        assert [c.index for c in outcome.citations] == [1, 2]
        assert outcome.confidence > 0
        assert outcome.degraded_reasons == []

    async def test_prompt_contains_numbered_context(self, tmp_path: Path) -> None:
        llm = FakeLLM(answers=("答案 [1]",))
        generator = AnswerGenerator(llm, _settings(tmp_path))
        await generator.answer("如何融合？", [_item(1)], history=[("user", "上一问")])
        prompt = llm.messages[0][-1].content
        assert isinstance(prompt, str)
        assert "<context>" in prompt
        assert "[1] 类型：正文" in prompt
        assert "如何融合？" in prompt
        # 历史作为独立消息插在 system 与当前问题之间
        assert len(llm.messages[0]) == 3

    async def test_refusal_when_no_items_does_not_call_llm(self, tmp_path: Path) -> None:
        llm = FakeLLM()
        generator = AnswerGenerator(llm, _settings(tmp_path))
        outcome = await generator.answer("库里有吗？", [])
        assert REFUSAL_MARKER in outcome.answer_markdown
        assert outcome.citations == []
        assert outcome.confidence == 0.0
        assert llm.calls == []

    async def test_llm_failure_falls_back_to_excerpts(self, tmp_path: Path) -> None:
        llm = FakeLLM(boom_kind="answer")
        generator = AnswerGenerator(llm, _settings(tmp_path))
        outcome = await generator.answer("如何融合？", [_item(1), _item(2, image=True)])
        assert "生成模型暂时不可用" in outcome.answer_markdown
        assert [c.index for c in outcome.citations] == [1, 2]
        assert DegradedReason.LLM_UNAVAILABLE in outcome.degraded_reasons
        assert outcome.confidence == 0.0

    async def test_missing_citations_are_flagged(self, tmp_path: Path) -> None:
        llm = FakeLLM(answers=("这是一个没有任何来源标注的答案。",))
        generator = AnswerGenerator(llm, _settings(tmp_path))
        outcome = await generator.answer("如何融合？", [_item(1)])
        assert outcome.citations == []
        assert DegradedReason.CITATION_MISSING in outcome.degraded_reasons

    async def test_history_respects_max_turns(self, tmp_path: Path) -> None:
        settings = _settings(
            tmp_path,
            agent=AgentSettings(history_max_turns=1, history_compress_enabled=False),
        )
        llm = FakeLLM(answers=("答案 [1]",))
        generator = AnswerGenerator(llm, settings)
        history = [
            ("user", "很久以前的问题"),
            ("assistant", "很久以前的回答"),
            ("user", "最近的问题"),
            ("assistant", "最近的回答"),
        ]
        await generator.answer("继续", [_item(1)], history=history)
        contents = [m.content for m in llm.messages[0]]
        assert "很久以前的问题" not in contents
        assert "最近的问题" in contents


class TestAnswerStream:
    async def test_emits_tokens_then_citations_then_done(self, tmp_path: Path) -> None:
        llm = FakeLLM(answers=("结论见 [1]。",))
        generator = AnswerGenerator(llm, _settings(tmp_path))
        events = [
            event
            async for event in generator.answer_stream("问题", [_item(1)])
        ]
        assert events[0].type == "token"
        assert "".join(e.text for e in events if e.type == "token") == "结论见 [1]。"
        types = [event.type for event in events]
        assert types[-2:] == ["citations", "done"]
        citations_event = next(e for e in events if e.type == "citations")
        assert citations_event.data["citations"][0]["index"] == 1
        assert citations_event.data["confidence"] > 0

    async def test_stream_failure_falls_back(self, tmp_path: Path) -> None:
        llm = FakeLLM(stream_boom=True)
        generator = AnswerGenerator(llm, _settings(tmp_path))
        events = [event async for event in generator.answer_stream("问题", [_item(1)])]
        text = "".join(e.text for e in events if e.type == "token")
        assert "生成模型暂时不可用" in text
        done = events[-1]
        assert done.type == "done"
        assert "llm_unavailable" in done.data["degraded_reasons"]

    async def test_refusal_stream(self, tmp_path: Path) -> None:
        llm = FakeLLM()
        generator = AnswerGenerator(llm, _settings(tmp_path))
        events = [event async for event in generator.answer_stream("问题", [])]
        assert REFUSAL_MARKER in "".join(e.text for e in events if e.type == "token")
        assert events[-1].data["citation_count"] == 0
        assert llm.calls == []


# --------------------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------------------


class TestRouter:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("你好", Intent.SMALLTALK),
            ("谢谢你！", Intent.SMALLTALK),
            ("这张图里画的是什么", Intent.IMAGE_LOOKUP),
            ("帮我找一张架构图", Intent.IMAGE_LOOKUP),
            ("给我一段 Python 代码", Intent.CODING),
            ("这个接口报错了怎么办", Intent.CODING),
            ("混合检索是如何融合三路结果的？", Intent.KNOWLEDGE),
        ],
    )
    async def test_rules(
        self, tmp_path: Path, query: str, expected: Intent
    ) -> None:
        llm = FakeLLM()
        decision = await Router(llm, _settings(tmp_path)).route(query)
        assert decision.intent is expected
        assert decision.source == "rule" or expected is Intent.KNOWLEDGE
        if expected is not Intent.KNOWLEDGE:
            assert llm.calls == []  # 规则命中时零 LLM 调用

    async def test_smalltalk_requires_full_match(self, tmp_path: Path) -> None:
        decision = await Router(None, _settings(tmp_path)).route(
            "你好，请介绍一下混合检索的融合方式"
        )
        assert decision.intent is Intent.KNOWLEDGE

    async def test_uncertain_query_uses_llm(self, tmp_path: Path) -> None:
        llm = FakeLLM(router='{"intent": "image_lookup", "confidence": 0.7, "reason": "问图"}')
        decision = await Router(llm, _settings(tmp_path)).route("图标是什么意思")
        assert decision.intent is Intent.IMAGE_LOOKUP
        assert decision.source == "llm"
        assert llm.calls == ["router"]

    async def test_llm_router_failure_falls_back(self, tmp_path: Path) -> None:
        llm = FakeLLM(boom_kind="router")
        decision = await Router(llm, _settings(tmp_path)).route("这个图标是什么意思")
        assert decision.intent is Intent.KNOWLEDGE
        assert decision.source == "default"

    def test_image_lookup_routes(self) -> None:
        from app.agent.router import RouterDecision

        decision = RouterDecision(Intent.IMAGE_LOOKUP)
        assert decision.routes == [Route.DENSE_IMAGE, Route.BM25]
        assert RouterDecision(Intent.KNOWLEDGE).routes is None

    async def test_short_history_is_not_compressed(self, tmp_path: Path) -> None:
        llm = FakeLLM()
        router = Router(llm, _settings(tmp_path))
        result = await router.standalone_query("它怎么实现？", [("user", "短")])
        assert result.text == "它怎么实现？"
        assert result.rewritten is False
        assert llm.calls == []

    async def test_long_history_is_compressed(self, tmp_path: Path) -> None:
        llm = FakeLLM(compress="混合检索如何融合三路结果")
        router = Router(llm, _settings(tmp_path))
        history = [("user", "长" * 200), ("assistant", "回答" * 50)]
        result = await router.standalone_query("它怎么实现？", history)
        assert result.rewritten is True
        assert result.text == "混合检索如何融合三路结果"
        assert llm.calls == ["compress"]

    async def test_compression_failure_degrades(self, tmp_path: Path) -> None:
        llm = FakeLLM(boom_kind="compress")
        router = Router(llm, _settings(tmp_path))
        result = await router.standalone_query(
            "它怎么实现？", [("user", "长" * 200)]
        )
        assert result.text == "它怎么实现？"
        assert result.degraded_reason is DegradedReason.LLM_UNAVAILABLE


# --------------------------------------------------------------------------------------
# 相关性判定
# --------------------------------------------------------------------------------------


class TestJudge:
    async def test_passes_above_threshold(self, tmp_path: Path) -> None:
        llm = FakeLLM(judges=('{"score": 0.82, "reason": "直接回答"}',))
        outcome = await Judge(llm, _settings(tmp_path)).score("问题", [_item(1)])
        assert outcome.verdict.score == pytest.approx(0.82)
        assert outcome.verdict.passed is True
        assert outcome.degraded_reason is None

    async def test_fails_below_threshold(self, tmp_path: Path) -> None:
        llm = FakeLLM(judges=('{"score": 0.2, "reason": "不相关"}',))
        outcome = await Judge(llm, _settings(tmp_path)).score("问题", [_item(1)])
        assert outcome.verdict.passed is False

    async def test_unparsable_output_degrades(self, tmp_path: Path) -> None:
        llm = FakeLLM(judges=("我觉得还行",))
        outcome = await Judge(llm, _settings(tmp_path)).score("问题", [_item(1)])
        assert outcome.degraded_reason is DegradedReason.JUDGE_UNAVAILABLE
        assert outcome.verdict.passed is False

    async def test_no_items_skips_llm(self, tmp_path: Path) -> None:
        llm = FakeLLM()
        outcome = await Judge(llm, _settings(tmp_path)).score("问题", [])
        assert llm.calls == []
        assert outcome.verdict.score == 0.0
        assert outcome.degraded_reason is None


# --------------------------------------------------------------------------------------
# Agent 循环
# --------------------------------------------------------------------------------------


class TestAgent:
    async def test_single_round_when_judge_passes(self, tmp_path: Path) -> None:
        llm = FakeLLM(answers=("RRF 融合 [1]。",), judges=('{"score": 0.9}',))
        retriever = FakeRetriever([_outcome([_item(1)])])
        result = await _agent(_settings(tmp_path), llm, retriever).answer(_kb(), "如何融合？")
        assert result.used_rounds == 1
        assert len(retriever.calls) == 1
        assert [c.index for c in result.citations] == [1]
        assert result.confidence > 0
        assert result.debug_trace[0].judge is not None
        assert result.debug_trace[0].fused_unique == 1

    async def test_second_round_uses_rewritten_query(self, tmp_path: Path) -> None:
        settings = _settings(
            tmp_path,
            agent=AgentSettings(
                max_rounds=2,
                relevance_threshold=0.6,
                judge_enabled=True,
                query_rewrite_enabled=False,
            ),
        )
        llm = FakeLLM(
            answers=("第二轮答案 [1]。",),
            judges=('{"score": 0.1}', '{"score": 0.9}'),
        )
        retriever = FakeRetriever(
            [
                _outcome([_item(1)], queries=("如何融合？", "RRF 融合公式")),
                _outcome([_item(2)], queries=("RRF 融合公式",)),
            ]
        )
        result = await _agent(settings, llm, retriever).answer(_kb(), "如何融合？")
        assert result.used_rounds == 2
        assert retriever.calls[1]["query"] == "RRF 融合公式"
        assert [c.index for c in result.citations] == [1]
        assert result.citations[0].item_id == _item(2).item_id

    async def test_keeps_best_round_when_retry_is_worse(self, tmp_path: Path) -> None:
        llm = FakeLLM(judges=('{"score": 0.5}', '{"score": 0.2}'))
        retriever = FakeRetriever(
            [
                _outcome([_item(1)], queries=("q", "alt")),
                _outcome([_item(2)], queries=("alt",)),
            ]
        )
        result = await _agent(_settings(tmp_path), llm, retriever).answer(_kb(), "q")
        assert result.used_rounds == 2
        assert result.citations[0].item_id == _item(1).item_id

    async def test_no_new_query_stops_early(self, tmp_path: Path) -> None:
        llm = FakeLLM(judges=('{"score": 0.1}',))
        retriever = FakeRetriever([_outcome([_item(1)], queries=("q",))])
        result = await _agent(_settings(tmp_path), llm, retriever).answer(_kb(), "q")
        assert result.used_rounds == 1
        assert len(retriever.calls) == 1

    async def test_judge_failure_stops_after_first_round(self, tmp_path: Path) -> None:
        llm = FakeLLM(boom_kind="judge")
        retriever = FakeRetriever(
            [_outcome([_item(1)], queries=("q", "alt")), _outcome([_item(2)])]
        )
        result = await _agent(_settings(tmp_path), llm, retriever).answer(_kb(), "q")
        assert result.used_rounds == 1
        assert DegradedReason.JUDGE_UNAVAILABLE in result.degraded_reasons
        assert result.degraded is True

    async def test_smalltalk_skips_retrieval(self, tmp_path: Path) -> None:
        llm = FakeLLM()
        retriever = FakeRetriever([_outcome([_item(1)])])
        result = await _agent(_settings(tmp_path), llm, retriever).answer(_kb(), "你好")
        assert retriever.calls == []
        assert result.used_rounds == 0
        assert result.citations == []
        assert result.debug_trace == []
        assert llm.calls == ["smalltalk"]

    async def test_image_lookup_uses_image_routes(self, tmp_path: Path) -> None:
        llm = FakeLLM(answers=("架构图见 [1]。",))
        retriever = FakeRetriever([_outcome([_item(1, image=True)])])
        result = await _agent(_settings(tmp_path), llm, retriever).answer(
            _kb(), "帮我找一张架构图"
        )
        assert retriever.calls[0]["routes"] == [Route.DENSE_IMAGE, Route.BM25]
        assert result.citations[0].image_url is not None

    async def test_generation_failure_returns_excerpts(self, tmp_path: Path) -> None:
        llm = FakeLLM(boom_kind="answer")
        retriever = FakeRetriever([_outcome([_item(1)])])
        result = await _agent(_settings(tmp_path), llm, retriever).answer(_kb(), "q")
        assert "生成模型暂时不可用" in result.answer_markdown
        assert DegradedReason.LLM_UNAVAILABLE in result.degraded_reasons
        assert result.citations

    async def test_judge_disabled_runs_one_round(self, tmp_path: Path) -> None:
        llm = FakeLLM()
        retriever = FakeRetriever([_outcome([_item(1)], queries=("q", "alt"))])
        result = await _agent(
            _settings(tmp_path), llm, retriever, judge_enabled=False
        ).answer(_kb(), "q")
        assert result.used_rounds == 1
        assert "judge" not in llm.calls
        assert result.debug_trace[0].judge is None

    async def test_debug_trace_records_counts_and_timings(
        self, tmp_path: Path
    ) -> None:
        llm = FakeLLM()
        retriever = FakeRetriever([_outcome([_item(1)])])
        result = await _agent(_settings(tmp_path), llm, retriever).answer(_kb(), "q")
        trace = result.debug_trace[0]
        assert trace.round == 1
        assert trace.candidate_counts == {"bm25": 1}
        assert trace.timings_ms["judge"] >= 0
        assert trace.reranked == 1
        assert trace.degraded_reasons == []


class TestAgentStream:
    async def test_stream_meta_tokens_done(self, tmp_path: Path) -> None:
        llm = FakeLLM(answers=("答案 [1]。",))
        retriever = FakeRetriever([_outcome([_item(1)])])
        agent = _agent(_settings(tmp_path), llm, retriever)
        events = [event async for event in agent.answer_stream(_kb(), "如何融合？")]

        assert events[0].type == "meta"
        assert events[0].data["intent"] == "knowledge"
        assert events[0].data["used_rounds"] == 1
        assert "".join(e.text for e in events if e.type == "token") == "答案 [1]。"
        assert [e.type for e in events][-2:] == ["citations", "done"]
        final = events[-1].data
        assert final["kb_id"] == KB_ID
        assert final["citation_count"] == 1
        assert final["used_rounds"] == 1
        # generator 自己的 done 被 graph 的汇总事件替换，不重复
        assert sum(1 for e in events if e.type == "done") == 1

    async def test_stream_smalltalk_meta(self, tmp_path: Path) -> None:
        llm = FakeLLM()
        retriever = FakeRetriever([_outcome([_item(1)])])
        agent = _agent(_settings(tmp_path), llm, retriever)
        events = [event async for event in agent.answer_stream(_kb(), "你好")]
        assert events[0].data["intent"] == "smalltalk"
        assert retriever.calls == []
        assert events[-1].data["used_rounds"] == 0


# --------------------------------------------------------------------------------------
# 工具映射
# --------------------------------------------------------------------------------------


class TestTools:
    def test_intent_to_tool(self) -> None:
        assert SearchTools.tool_for(Intent.KNOWLEDGE) == (
            ToolName.SEARCH_KNOWLEDGE,
            None,
        )
        assert SearchTools.tool_for(Intent.CODING)[0] is ToolName.SEARCH_KNOWLEDGE
        assert SearchTools.tool_for(Intent.IMAGE_LOOKUP)[0] is ToolName.SEARCH_IMAGES
        assert SearchTools.tool_for(Intent.SMALLTALK)[0] is ToolName.SEARCH_KNOWLEDGE

    async def test_run_forwards_arguments(self, tmp_path: Path) -> None:
        retriever = FakeRetriever([_outcome([_item(1)])])
        tools = SearchTools(retriever)  # type: ignore[arg-type]
        outcome, call = await tools.run(
            Intent.IMAGE_LOOKUP,
            _kb(),
            "找图",
            top_k=3,
            use_rerank=False,
            rewrite=True,
        )
        assert retriever.calls[0]["top_k"] == 3
        assert retriever.calls[0]["use_rerank"] is False
        assert retriever.calls[0]["rewrite"] is True
        assert call.name is ToolName.SEARCH_IMAGES
        assert call.routes == ["dense_image", "bm25"]
        assert outcome is not None
