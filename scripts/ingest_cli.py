#!/usr/bin/env python
"""命令行导入：把一个目录或若干文件灌进知识库。

它是 :class:`~app.ingestion.pipeline.IngestionPipeline` 的薄壳 ——
所有业务逻辑都在流水线里，这里只负责参数解析与结果展示。

**没有 API Key 也能跑**：``--no-embed`` 会跳过全部向量化，只写 SQLite 与
BM25。这条路径覆盖了「解析 → 图片过滤与去重 → 切分 → 落库」的全部逻辑，
是验证摄入层最省钱、最快的方式。

用法::

    # 离线验证：不调用任何外部 API
    python scripts/ingest_cli.py --kb demo --path data/raw --no-embed

    # 完整导入（需要 .env 里配好三把 Key）
    python scripts/ingest_cli.py --kb demo --path data/raw/sample.pdf

    # 只看解析与切分结果，不写任何数据
    python scripts/ingest_cli.py --kb demo --path data/raw --dry-run

    # 机器可读输出（日志走 stderr，stdout 只有 JSON）
    python scripts/ingest_cli.py --kb demo --path data/raw --json

退出码::

    0  全部成功（重复跳过也算成功）
    1  至少一个文件失败
    2  用法错误：路径不存在、没有可导入的文件、或 --strict 下有降级
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

from app.config import get_settings  # noqa: E402
from app.core.console import enable_ansi  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.schemas import DocStatus  # noqa: E402
from app.ingestion.pipeline import (  # noqa: E402
    FileOutcome,
    IngestionPipeline,
    IngestOptions,
    ProgressCallback,
    collect_files,
)
from app.store.db import get_database  # noqa: E402

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[90m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"

_STATUS_STYLE: dict[DocStatus, tuple[str, str]] = {
    DocStatus.DONE: (_GREEN, "完成"),
    DocStatus.SKIPPED_DUPLICATE: (_DIM, "跳过·重复"),
    DocStatus.FAILED: (_RED, "失败"),
    DocStatus.QUEUED: (_DIM, "预览"),
}


# --------------------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------------------


def _force_utf8_stdio() -> None:
    """被管道重定向时强制 UTF-8 输出。

    Windows 下 ``sys.stdout`` 连终端时走 Unicode 控制台 API，中文本来就正常；
    但重定向到文件或管道时会退回 ``cp936``，另一端按 UTF-8 解码就是乱码。
    终端里不改（``reconfigure`` 到非 UTF-8 会在控制台上出问题）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if enable_ansi() else text


def _width(text: str) -> int:
    """终端显示宽度。中日韩字符占两列，``len`` 会让表格歪掉。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, *, align: str = "left") -> str:
    gap = max(0, width - _width(text))
    return (" " * gap + text) if align == "right" else (text + " " * gap)


def _shorten(text: str, limit: int) -> str:
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


def _render_table(outcomes: list[FileOutcome]) -> str:
    headers = ["文件", "状态", "分块", "图片", "向量(文/图)", "耗时"]
    rows: list[list[str]] = []
    for outcome in outcomes:
        status_code, status_label = _STATUS_STYLE.get(
            outcome.status, (_DIM, str(outcome.status))
        )
        rows.append(
            [
                _shorten(outcome.path.name, 42),
                _color(status_label, status_code),
                str(outcome.chunks),
                f"{outcome.images}"
                + (
                    f" ({outcome.images_filtered} 过滤)"
                    if outcome.images_filtered
                    else ""
                ),
                f"{outcome.text_vectors}/{outcome.image_vectors}",
                f"{outcome.elapsed_ms / 1000:.2f}s",
            ]
        )

    widths = [
        max(_width(headers[i]), *(_width(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    numeric = {2, 3, 4, 5}

    def line(cells: list[str], *, header: bool = False) -> str:
        padded = [
            _pad(cell, widths[i], align="right" if i in numeric else "left")
            for i, cell in enumerate(cells)
        ]
        text = "  ".join(padded)
        return _color(text, _BOLD) if header else text

    parts = [line(headers, header=True)]
    parts.append(_color("  ".join("─" * width for width in widths), _DIM))
    parts.extend(line(row) for row in rows)
    return "\n".join(parts)


def _render_summary(outcomes: list[FileOutcome], skipped: list[Path]) -> str:
    done = sum(1 for item in outcomes if item.status is DocStatus.DONE)
    dup = sum(1 for item in outcomes if item.status is DocStatus.SKIPPED_DUPLICATE)
    failed = [item for item in outcomes if item.status is DocStatus.FAILED]

    # 重复跳过也计入总量：它代表「库里确实有这些内容」，
    # 而不是「这次什么都没导入」。只统计 DONE 会让重跑看起来像清空了知识库。
    counted = (DocStatus.DONE, DocStatus.SKIPPED_DUPLICATE)
    chunks = sum(item.chunks for item in outcomes if item.status in counted)
    images = sum(item.images for item in outcomes if item.status in counted)
    tokens = sum(item.tokens for item in outcomes if item.status in counted)

    lines = [
        "",
        f"共 {len(outcomes)} 个文件："
        f"{_color(f'{done} 成功', _GREEN)}，"
        f"{dup} 跳过（重复），"
        f"{_color(f'{len(failed)} 失败', _RED) if failed else '0 失败'}；"
        f"分块 {chunks}，图片 {images}，token {tokens}",
    ]

    degraded = sorted({reason for item in outcomes for reason in item.degraded_reasons})
    if degraded:
        lines.append(_color("降级：", _YELLOW) + "、".join(degraded))

    reasons = sorted({path.suffix or "（无扩展名）" for path in skipped})
    if skipped:
        lines.append(
            f"{_color('跳过的不支持文件：', _DIM)}{len(skipped)} 个"
            f"（{'、'.join(reasons)}）"
        )

    for item in failed:
        lines.append(_color(f"✗ {item.path.name}", _RED) + f"  {item.error}")

    warnings = [(item, warning) for item in outcomes for warning in item.warnings]
    if warnings:
        lines.append("")
        lines.append(_color("警告：", _YELLOW))
        for item, warning in warnings[:20]:
            lines.append(f"  · {item.path.name}: {warning}")
        if len(warnings) > 20:
            lines.append(f"  … 还有 {len(warnings) - 20} 条")

    return "\n".join(lines)


def _render_kb(kb: Any) -> str:
    lines = [
        _color(f"知识库 {kb.name}", _BOLD) + _color(f"  {kb.kb_id}", _DIM),
        f"  文本空间  {kb.text_collection}  dim={kb.text_dim}",
        f"  图像空间  {kb.image_collection}  dim={kb.image_dim}",
    ]
    if kb.text_embed_model or kb.image_embed_model:
        lines.append(
            f"  模型快照  {kb.text_embed_model or '-'} / {kb.image_embed_model or '-'}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    database = get_database()
    await database.init_schema()

    pipeline = IngestionPipeline(settings)
    try:
        kb = await pipeline.ensure_kb(args.kb, description=args.description or "")
        files, skipped = collect_files(args.path)

        if not files:
            print(
                _color("没有找到可导入的文件。", _RED),
                file=sys.stderr,
            )
            supported = ", ".join(sorted({p.suffix for p in skipped})) or "（无）"
            print(f"路径：{[str(p) for p in args.path]}", file=sys.stderr)
            print(f"被跳过的扩展名：{supported}", file=sys.stderr)
            return 2

        options = IngestOptions(
            embed=not args.no_embed,
            caption=not args.no_caption,
            reindex=args.reindex,
            dry_run=args.dry_run,
        )

        if not args.json:
            print(_render_kb(kb))
            mode = []
            if args.dry_run:
                mode.append("dry-run（不写库）")
            if args.no_embed:
                mode.append("no-embed（只写 BM25）")
            if args.no_caption:
                mode.append("no-caption")
            if args.reindex:
                mode.append("reindex")
            print(
                _color("模式  ", _DIM)
                + ("、".join(mode) if mode else "完整导入")
                + _color(f"  文件 {len(files)} 个", _DIM)
            )
            print()

        progress = None if args.json else _progress_printer()

        outcomes = await pipeline.ingest_paths(
            files, kb, options=options, progress=progress
        )
        if not args.json:
            # 进度行是用 \r 覆盖的，没有换行；收个尾再打表格
            print()

        failed = [item for item in outcomes if item.status is DocStatus.FAILED]
        # 任何一次导入都可能改写了磁盘上的索引，退出前落一次盘。
        # persist 只写脏索引，重复调用没有额外代价。
        await pipeline.bm25store.persist(kb.kb_id)

        if args.json:
            payload = {
                "kb": {
                    "kb_id": kb.kb_id,
                    "name": kb.name,
                    "text_collection": kb.text_collection,
                    "image_collection": kb.image_collection,
                    "text_dim": kb.text_dim,
                    "image_dim": kb.image_dim,
                },
                "options": {
                    "embed": options.embed,
                    "caption": options.caption,
                    "reindex": options.reindex,
                    "dry_run": options.dry_run,
                },
                "results": [item.to_dict() for item in outcomes],
                "skipped": [str(path) for path in skipped],
                "summary": {
                    "total": len(outcomes),
                    "done": sum(
                        1 for item in outcomes if item.status is DocStatus.DONE
                    ),
                    "duplicate": sum(
                        1
                        for item in outcomes
                        if item.status is DocStatus.SKIPPED_DUPLICATE
                    ),
                    "failed": len(failed),
                    "chunks": sum(item.chunks for item in outcomes),
                    "images": sum(item.images for item in outcomes),
                    "tokens": sum(item.tokens for item in outcomes),
                },
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(_render_table(outcomes))
            print(_render_summary(outcomes, skipped))

        if failed:
            return 1
        if args.strict:
            noisy = [
                item
                for item in outcomes
                if item.warnings or item.degraded_reasons
            ]
            if noisy:
                return 2
        return 0
    finally:
        await pipeline.aclose()
        await database.dispose()


def _progress_printer() -> ProgressCallback:
    """单行进度。只用 ``\\r`` 覆盖，不刷屏。"""
    interactive = sys.stdout.isatty()

    def report(status: DocStatus, message: str) -> None:
        if not interactive:
            return
        sys.stdout.write("\r" + _color("··· ", _DIM) + _pad(message, 72))
        sys.stdout.flush()
        if status in (DocStatus.DONE, DocStatus.FAILED):
            sys.stdout.write("\n")

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest_cli.py",
        description="把文件或目录导入多模态知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "离线验证：\n"
            "  python scripts/ingest_cli.py --kb demo --path data/raw --no-embed\n\n"
            "只预览，不写库：\n"
            "  python scripts/ingest_cli.py --kb demo --path data/raw --dry-run"
        ),
    )
    parser.add_argument("--kb", required=True, help="知识库名称或 kb_ 开头的 ID")
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        type=Path,
        help="文件或目录；可重复传多次",
    )
    parser.add_argument("--description", default="", help="新建知识库时的描述")
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="跳过向量化，只写 SQLite 与 BM25（无需 API Key）",
    )
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="不调用视觉模型，图片使用上下文占位 caption",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="即使内容哈希已存在也重新导入",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析与切分，不写任何数据",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON（日志走 stderr）")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="存在警告或降级时以退出码 2 结束",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="打印 INFO 级别日志"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _force_utf8_stdio()

    # 日志永远走 stderr：--json 时 stdout 必须只有一份合法 JSON
    setup_logging(
        level="INFO" if args.verbose else "WARNING",
        stream=sys.stderr,
    )

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已中断。已经完成的文档会保持 DONE 状态。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
