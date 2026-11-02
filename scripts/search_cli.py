#!/usr/bin/env python
"""命令行检索：把一个问题送进三路混合检索，打印融合与重排结果。

这是 :class:`~app.retrieval.hybrid.HybridRetriever` 的薄壳 ——
参数解析与展示在这里，业务逻辑全在检索层。

它是验证「检索质量」最快的工具：不需要起服务、不需要前端，
``--debug`` 会打印每一路的状态、耗时与每条结果被哪几路以第几名捞上来。
排查「答案里怎么没有那张图」这类问题时，这些信息比最终排序本身有用。

用法::

    # 离线可验证：--no-embed 导入的知识库只有 BM25 一路有数据，
    # 另外两路会显示「collection 为空」并被跳过
    python scripts/search_cli.py --kb demo --q "混合检索是如何融合三路结果的？" --debug

    # 只用某几路做消融
    python scripts/search_cli.py --kb demo --q "架构图" --routes dense_image

    # 开启 LLM 多查询改写（需要 LLM Key）
    python scripts/search_cli.py --kb demo --q "它怎么保证不丢图？" --rewrite --debug

    # 机器可读输出（日志走 stderr，stdout 只有 JSON）
    python scripts/search_cli.py --kb demo --q "RRF" --json

    # 走完整问答链路：路由 → 检索 → 相关性判定 → 带引用的答案（需要 LLM Key）
    python scripts/search_cli.py --kb demo --q "混合检索如何融合？" --ask --debug

退出码::

    0  检索完成（没有结果也算完成）
    1  检索链路不可降级失败（知识库快照与配置不一致等）
    2  用法错误：知识库不存在、参数非法、缺少必需的 Key
    130 用户中断
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.agent.graph import AnswerAgent  # noqa: E402
from app.config import ConfigurationError, Settings, get_settings  # noqa: E402
from app.core.console import enable_ansi  # noqa: E402
from app.core.errors import MMKBError, NotFoundError  # noqa: E402
from app.core.ids import is_valid_id  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.schemas import (  # noqa: E402
    AnswerResult,
    Citation,
    DebugRound,
    ItemType,
    KnowledgeBase,
    Route,
)
from app.generation.answer import AnswerGenerator  # noqa: E402
from app.providers.base import (  # noqa: E402
    EmbeddingProvider,
    LLMProvider,
    RerankProvider,
)
from app.providers.factory import (  # noqa: E402
    build_image_embedding,
    build_llm,
    build_rerank,
    build_text_embedding,
)
from app.retrieval.hybrid import HybridRetriever, SearchOutcome  # noqa: E402
from app.store.db import get_database  # noqa: E402
from app.store.docstore import DocStore  # noqa: E402
from app.store.factory import get_bm25store, get_vectorstore  # noqa: E402

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[90m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"

_TYPE_LABEL = {
    ItemType.TEXT: "正文",
    ItemType.CODE: "代码",
    ItemType.TABLE: "表格",
    ItemType.IMAGE: "图片",
}


# --------------------------------------------------------------------------------------
# 输出工具
# --------------------------------------------------------------------------------------


def _force_utf8_stdio() -> None:
    """被管道重定向时强制 UTF-8。理由与 ``ingest_cli.py`` 相同。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if enable_ansi() else text


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, *, align: str = "left") -> str:
    gap = max(0, width - _width(text))
    return (" " * gap + text) if align == "right" else (text + " " * gap)


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if _width(text) <= limit:
        return text
    out = ""
    used = 0
    for ch in text:
        step = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used + step > limit - 1:
            break
        out += ch
        used += step
    return out + "…"


def _render_routes(outcome: SearchOutcome) -> str:
    """每一路的状态。这是 --debug 的核心输出。"""
    headers = ["路", "条数", "耗时", "状态"]
    rows: list[list[str]] = []
    for result in outcome.route_results:
        if result.error:
            status = _color("失败：" + _shorten(result.error, 60), _RED)
        elif result.skipped:
            status = _color("跳过：" + (result.note or "-"), _DIM)
        else:
            status = _color("正常", _GREEN)
            if result.degraded_reason is not None:
                status += _color(f"（降级 {result.degraded_reason}）", _YELLOW)
        rows.append(
            [
                str(result.route),
                str(result.count),
                f"{result.latency_ms}ms",
                status,
            ]
        )

    if not rows:
        return _color("（没有任何检索路被执行）", _DIM)

    widths = [max(_width(headers[i]), *(_width(r[i]) for r in rows)) for i in range(4)]
    lines = [
        _color("  ".join(_pad(h, widths[i]) for i, h in enumerate(headers)), _BOLD),
        _color("  ".join("─" * w for w in widths), _DIM),
    ]
    lines.extend(
        "  ".join(
            _pad(cell, widths[i], align="right" if i in (1, 2) else "left")
            for i, cell in enumerate(row)
        )
        for row in rows
    )
    return "\n".join(lines)


def _render_items(outcome: SearchOutcome, *, debug: bool) -> str:
    if not outcome.items:
        return _color("没有检索到任何结果。", _YELLOW)

    headers = ["#", "类型", "路线", "RRF", "重排", "来源", "章节", "内容"]
    rows: list[list[str]] = []
    for index, item in enumerate(outcome.items, start=1):
        rerank_text = (
            f"{item.rerank_score:.4f}" if item.rerank_score is not None else "-"
        )
        source = item.source_name or (item.image_id or "") or "-"
        # 名次来源放在「路线」列而不是标题列：它本身就是路线信息，
        # 且该列宽度随内容自动扩展，不会被标题列的固定宽度截掉
        route_cell = str(item.route)
        if debug and item.route_ranks:
            route_cell = " ".join(
                f"{label}:{rank}" for label, rank in sorted(item.route_ranks.items())
            )
        rows.append(
            [
                str(index),
                _TYPE_LABEL.get(item.item_type, str(item.item_type)),
                route_cell,
                f"{item.rrf_score:.5f}",
                rerank_text,
                _shorten(source, 24),
                _shorten(item.heading_text or "-", 30),
                _shorten(item.text, 64),
            ]
        )

    widths = [max(_width(headers[i]), *(_width(r[i]) for r in rows)) for i in range(8)]
    numeric = {0, 3, 4}
    lines = [
        _color(
            "  ".join(_pad(h, widths[i], align="right" if i in numeric else "left")
                      for i, h in enumerate(headers)),
            _BOLD,
        ),
        _color("  ".join("─" * w for w in widths), _DIM),
    ]
    for row in rows:
        lines.append(
            "  ".join(
                _pad(cell, widths[i], align="right" if i in numeric else "left")
                for i, cell in enumerate(row)
            )
        )
    return "\n".join(lines)


def _render_citations(citations: list[Citation]) -> str:
    headers = ["#", "类型", "来源", "章节", "页", "摘要"]
    rows: list[list[str]] = []
    for citation in citations:
        rows.append(
            [
                f"[{citation.index}]",
                _TYPE_LABEL.get(citation.item_type, str(citation.item_type)),
                _shorten(citation.doc_name or "-", 22),
                _shorten(" > ".join(citation.heading_path) or "-", 26),
                str(citation.page) if citation.page is not None else "-",
                _shorten(citation.snippet, 52),
            ]
        )
    widths = [max(_width(headers[i]), *(_width(r[i]) for r in rows)) for i in range(6)]
    numeric = {0, 4}
    lines = [
        _color(
            "  ".join(
                _pad(h, widths[i], align="right" if i in numeric else "left")
                for i, h in enumerate(headers)
            ),
            _BOLD,
        ),
        _color("  ".join("─" * w for w in widths), _DIM),
    ]
    for row in rows:
        lines.append(
            "  ".join(
                _pad(cell, widths[i], align="right" if i in numeric else "left")
                for i, cell in enumerate(row)
            )
        )
    return "\n".join(lines)


def _render_round(round_trace: DebugRound) -> str:
    judge = round_trace.judge
    if judge is None:
        verdict = _color("未判定", _DIM)
    else:
        state = "通过" if judge.passed else "未通过"
        color = _GREEN if judge.passed else _YELLOW
        verdict = _color(f"{judge.score:.2f}（阈值 {judge.threshold:.2f}，{state}）", color)

    counts = " ".join(
        f"{name}={count}" for name, count in sorted(round_trace.candidate_counts.items())
    )
    timings = " ".join(
        f"{name}={ms}ms" for name, ms in round_trace.timings_ms.items()
    )
    lines = [f"轮次 {round_trace.round}  查询：{_shorten(round_trace.query, 60)}"]
    if round_trace.rewrites:
        lines.append(
            "  改写：" + "、".join(_shorten(q, 34) for q in round_trace.rewrites)
        )
    lines.append(
        f"  候选：{counts or '-'} | 融合 {round_trace.fused_unique} | "
        f"重排 {round_trace.reranked} | 判定 {verdict}"
    )
    lines.append("  耗时：" + (timings or "-"))
    if round_trace.degraded_reasons:
        lines.append(
            "  降级："
            + _color("、".join(str(r) for r in round_trace.degraded_reasons), _YELLOW)
        )
    return "\n".join(lines)


def _render_answer(result: AnswerResult, *, debug: bool) -> str:
    lines = [_color("答案", _BOLD), result.answer_markdown, ""]

    if result.citations:
        lines.append(_color("引用", _BOLD))
        lines.append(_render_citations(result.citations))
        images = [c for c in result.citations if c.image_url]
        if images:
            lines.append("")
            lines.append(
                _color("图片引用：", _DIM)
                + "  ".join(f"[{c.index}] {c.image_url}" for c in images)
            )
        lines.append("")

    summary = (
        f"置信度 {result.confidence:.2f} · 轮次 {result.used_rounds} · "
        f"耗时 {result.latency_ms}ms"
    )
    lines.append(_color(summary, _DIM))

    if result.degraded_reasons:
        lines.append(
            _color("降级：", _YELLOW)
            + "、".join(str(reason) for reason in result.degraded_reasons)
        )

    if debug:
        if result.debug_trace:
            lines.append("")
            lines.append(_color("检索轮次", _BOLD))
            lines.extend(_render_round(trace) for trace in result.debug_trace)
        else:
            lines.append("")
            lines.append(_color("本次没有执行检索（寒暄意图）。", _DIM))
    return "\n".join(lines)


def _render_outcome(outcome: SearchOutcome, *, debug: bool) -> str:
    lines: list[str] = []
    if debug:
        lines.append(_color("检索路", _BOLD))
        lines.append(_render_routes(outcome))
        lines.append("")

    lines.append(_render_items(outcome, debug=debug))
    lines.append("")

    summary = f"共 {len(outcome.items)} 条结果"
    if outcome.queries and len(outcome.queries) > 1:
        summary += f"；改写查询 {len(outcome.queries)} 条：" + "、".join(
            _shorten(q, 24) for q in outcome.queries
        )
    summary += "；重排 " + ("已启用" if outcome.reranked else "未启用")
    lines.append(_color(summary, _DIM))

    if outcome.degraded_reasons:
        lines.append(
            _color("降级：", _YELLOW)
            + "、".join(str(reason) for reason in outcome.degraded_reasons)
        )
    if debug:
        timings = "  ".join(f"{name}={ms}ms" for name, ms in outcome.timings.items())
        lines.append(_color("耗时：" + timings, _DIM))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


async def _resolve_kb(docstore: DocStore, name_or_id: str) -> KnowledgeBase:
    """按 ID 或名称取知识库。

    与 ``IngestionPipeline.ensure_kb`` 的区别：这里**不建库**。
    检索一个不存在的知识库应该报错，而不是悄悄建一个空库然后返回 0 条结果 ——
    后者会让「kb 名字打错了」看起来像「知识库里没内容」。
    """
    if is_valid_id(name_or_id, "kb"):
        kb = await docstore.get_kb(name_or_id)
        if kb is not None:
            return kb
        raise NotFoundError(
            f"知识库 {name_or_id!r} 不存在", detail={"kb_id": name_or_id}
        )

    for kb in await docstore.list_kbs():
        if kb.name == name_or_id:
            return kb
    available = [kb.name for kb in await docstore.list_kbs()]
    raise NotFoundError(
        f"知识库 {name_or_id!r} 不存在。已存在：{available or '（无）'}。"
        "请先运行 scripts/ingest_cli.py 导入数据。",
        detail={"requested": name_or_id, "available": available},
    )


def _build_providers(
    settings: Settings, args: argparse.Namespace
) -> tuple[
    EmbeddingProvider | None,
    EmbeddingProvider | None,
    RerankProvider | None,
    LLMProvider | None,
    list[str],
]:
    """按需构造 provider。

    刻意不用 ``get_providers()``：那个 bundle 要求 LLM Key 必须存在。
    而「只想验证 BM25 + 文本向量」是完全合理的用法，
    不该被一句「LLM_API_KEY 未配置」拦在门外。
    """
    warnings: list[str] = []

    text_embedding: EmbeddingProvider | None = None
    try:
        text_embedding = build_text_embedding(settings)
    except ConfigurationError as exc:
        warnings.append("文本向量不可用：" + "；".join(exc.problems))

    image_embedding: EmbeddingProvider | None = None
    try:
        image_embedding = build_image_embedding(settings)
    except ConfigurationError as exc:
        warnings.append("多模态向量不可用：" + "；".join(exc.problems))

    rerank_provider: RerankProvider | None = None
    if not args.no_rerank:
        try:
            rerank_provider = build_rerank(settings)
        except ConfigurationError as exc:
            warnings.append("重排不可用：" + "；".join(exc.problems))

    llm: LLMProvider | None = None
    if args.ask or args.rewrite:
        try:
            llm = build_llm(settings)
        except ConfigurationError as exc:
            label = "问答生成不可用" if args.ask else "查询改写不可用（本次只检索原问题）"
            warnings.append(label + "：" + "；".join(exc.problems))

    return text_embedding, image_embedding, rerank_provider, llm, warnings


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    database = get_database()
    await database.init_schema()
    docstore = DocStore(settings)
    vectorstore = get_vectorstore()
    bm25store = get_bm25store()

    text_embedding, image_embedding, rerank_provider, llm, warnings = _build_providers(
        settings, args
    )

    retriever = HybridRetriever(
        settings,
        docstore=docstore,
        vectorstore=vectorstore,
        bm25store=bm25store,
        text_embedding=text_embedding,
        image_embedding=image_embedding,
        rerank=rerank_provider,
        llm=llm,
    )

    try:
        return await _execute(args, settings, docstore, retriever, llm, warnings)
    finally:
        for provider in (text_embedding, image_embedding, rerank_provider, llm):
            if provider is not None:
                try:
                    await provider.aclose()
                except Exception:  # noqa: BLE001 - 关闭失败不影响退出码
                    pass
        await vectorstore.aclose()
        await database.dispose()


async def _execute(
    args: argparse.Namespace,
    settings: Settings,
    docstore: DocStore,
    retriever: HybridRetriever,
    llm: LLMProvider | None,
    warnings: list[str],
) -> int:
    """解析知识库 → 按模式执行 → 渲染输出。资源释放由 ``_run`` 负责。"""
    routes: list[Route] | None = None
    if args.routes:
        routes = []
        for raw in args.routes.split(","):
            name = raw.strip()
            if not name:
                continue
            try:
                route = Route(name)
            except ValueError as exc:
                raise ConfigurationError(
                    [f"--routes 里有未知的路 {name!r}（可选：bm25,dense_text,dense_image）"]
                ) from exc
            if route is Route.HYBRID:
                raise ConfigurationError(["--routes 不能包含 hybrid"])
            routes.append(route)

    kb = await _resolve_kb(docstore, args.kb)
    header = (
        _color(f"知识库 {kb.name}", _BOLD)
        + _color(f"  {kb.kb_id}", _DIM)
        + _color(f"  查询：{args.q}", _CYAN)
    )

    if args.ask:
        if llm is None:
            raise ConfigurationError(
                ["--ask 需要可用的 LLM：请在 .env 配置 LLM_API_KEY 后重试"]
            )
        if routes is not None:
            raise ConfigurationError(
                ["--routes 只用于纯检索模式；问答模式下检索路由由 agent 根据意图决定"]
            )

        agent = AnswerAgent(
            settings,
            retriever=retriever,
            generator=AnswerGenerator(llm, settings),
            llm=llm,
            judge_enabled=False if args.no_judge else None,
        )
        result = await agent.answer(
            kb,
            args.q,
            top_k=args.top_k,
            use_rerank=False if args.no_rerank else None,
            rewrite=True if args.rewrite else None,
        )
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 0

        print(header)
        for warning in warnings:
            print(_color("提示：" + warning, _YELLOW))
        print()
        print(_render_answer(result, debug=args.debug))
        return 0

    outcome = await retriever.search(
        kb,
        args.q,
        top_k=args.top_k,
        routes=routes,
        use_rerank=False if args.no_rerank else None,
        rewrite=args.rewrite,
    )

    if args.json:
        payload = outcome.to_dict()
        payload["kb"] = {
            "kb_id": kb.kb_id,
            "name": kb.name,
            "text_collection": kb.text_collection,
            "image_collection": kb.image_collection,
        }
        payload["warnings"] = warnings
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(header)
    for warning in warnings:
        print(_color("提示：" + warning, _YELLOW))
    print()
    print(_render_outcome(outcome, debug=args.debug))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search_cli.py",
        description="在知识库上执行三路混合检索（BM25 + 文本向量 + 多模态向量 + RRF + 重排）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "离线验证（只需要已导入的知识库）：\n"
            '  python scripts/search_cli.py --kb demo --q "混合检索是如何融合三路结果的？" --debug\n\n'
            "只看以文搜图：\n"
            '  python scripts/search_cli.py --kb demo --q "架构图" --routes dense_image\n\n'
            "完整问答（需要 LLM Key）：\n"
            '  python scripts/search_cli.py --kb demo --q "混合检索如何融合？" --ask --debug\n\n'
            "注意：重排模型只接受纯文本，图片候选是用 caption + 章节路径参与重排的。\n"
            "这是已知取舍，详见 app/retrieval/rerank.py 与 README。"
        ),
    )
    parser.add_argument("--kb", required=True, help="知识库名称或 kb_ 开头的 ID")
    parser.add_argument("--q", required=True, help="检索问题")
    parser.add_argument(
        "--ask",
        action="store_true",
        help="走完整问答链路（意图路由 → 检索 → 相关性判定 → 带引用的答案），需要 LLM Key",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="问答模式下跳过相关性判定（只检索一轮，消融用）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="最终返回条数（默认取 RETRIEVAL_FINAL_TOP_K）",
    )
    parser.add_argument(
        "--routes",
        default="",
        help="只启用指定检索路，逗号分隔：bm25,dense_text,dense_image（默认全部）",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="跳过重排，直接用 RRF 分数排序（消融实验用）",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="用 LLM 做多查询改写（需要 LLM Key；失败自动退回原问题）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印每一路的状态、耗时，以及每条结果的名次来源",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON（日志走 stderr）")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印 INFO 级别日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _force_utf8_stdio()
    setup_logging(level="INFO" if args.verbose else "WARNING", stream=sys.stderr)

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except ConfigurationError as exc:
        print(_color("配置错误：", _RED) + "；".join(exc.problems), file=sys.stderr)
        return 2
    except NotFoundError as exc:
        print(_color("未找到：", _RED) + exc.message, file=sys.stderr)
        return 2
    except MMKBError as exc:
        print(
            _color(f"检索失败（{exc.code}）：", _RED) + exc.message,
            file=sys.stderr,
        )
        detail: Any = getattr(exc, "detail", None)
        if detail:
            print(json.dumps(detail, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
