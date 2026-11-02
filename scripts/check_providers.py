#!/usr/bin/env python
"""provider 连通性与向量维度自检。

**这个脚本会真实调用外部 API**（会消耗极少量额度），
它是「配置写对了没有」这个问题的唯一可靠答案：

* API Key 有没有开通对应模型；
* ``TEXT_EMBED_DIM`` / ``IMAGE_EMBED_DIM`` 与服务端实际返回的维度是否一致；
* 多模态端点能不能真的接受一张图片（base64 编码、尺寸、contents 结构）；
* ``reasoning_effort`` 这个扩展参数会不会被服务端拒绝。

维度配错是**最危险**的一类问题：不会报错，只会让整个向量库的相似度失效。
所以宁可在这里花几秒钟。

用法::

    python scripts/check_providers.py                # 全量自检
    python scripts/check_providers.py --skip-llm     # 跳过对话模型（省钱/省时）
    python scripts/check_providers.py --json         # 机器可读输出
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import (  # noqa: E402
    ConfigurationError,
    get_settings,
    reset_settings_cache,
)
from app.core.console import enable_ansi  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.providers import factory  # noqa: E402

_OK = "\033[32m✓\033[0m"
_FAIL = "\033[31m✗\033[0m"
_SKIP = "\033[90m－\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[90m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    return enable_ansi()


def _paint(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _supports_color() else text


def _render_matrix(matrix: dict[str, dict[str, object]]) -> str:
    titles = {
        "llm": ("对话模型", "LLM_PROVIDER / LLM_MODEL"),
        "text_embedding": ("文本向量", "TEXT_EMBED_* → {kb_id}__text"),
        "image_embedding": ("多模态向量", "IMAGE_EMBED_* → {kb_id}__image"),
        "rerank": ("重排", "RERANK_*"),
    }
    lines: list[str] = []
    for key, (title, env_hint) in titles.items():
        info = matrix[key]
        lines.append(_paint(f"[{title}]", _BOLD))
        lines.append(f"  provider      {info.get('provider')}")
        lines.append(f"  model         {info.get('model')}")
        if "dim" in info:
            suffix = info.get("collection_suffix", "")
            lines.append(f"  dim           {info['dim']}  {suffix}")
        if "enable_fusion" in info:
            lines.append(f"  enable_fusion {info['enable_fusion']}")
        if "base_url" in info:
            lines.append(f"  base_url      {info['base_url']}")
        configured = info.get("configured")
        mark = _paint("已配置", _OK) if configured else _paint("未配置", _FAIL)
        lines.append(f"  凭证          {mark}")
        lines.append(_paint(f"  env           {env_hint}", _DIM))
        lines.append("")
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    reset_settings_cache()
    try:
        settings = get_settings()
        bundle = factory.get_providers()
    except ConfigurationError as exc:
        print(_paint("配置未通过校验，无法进行 provider 自检：", _FAIL))
        print(exc)
        print()
        print(_paint("提示：先执行 python scripts/show_config.py 查看每一项的来源", _DIM))
        return 2

    matrix = factory.provider_matrix(settings)

    if args.json:
        report = await factory.verify_providers(bundle, include_llm=not args.skip_llm)
        payload = {
            "ok": report.ok,
            "providers": matrix,
            "checks": [
                {
                    "name": check.name,
                    "ok": check.ok,
                    "skipped": check.skipped,
                    "detail": check.detail,
                    "latency_ms": check.latency_ms,
                    "extra": check.extra,
                }
                for check in report.checks
            ],
            "degraded_reasons": [str(r) for r in bundle.degraded_reasons],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if report.ok else 1

    print(_paint("provider 配置", _BOLD))
    print("=" * 78)
    print(_render_matrix(matrix))

    print(_paint("连通性 / 维度自检", _BOLD))
    print("=" * 78)
    if args.skip_llm:
        print(_paint("  （已跳过对话模型）", _DIM))

    report = await factory.verify_providers(bundle, include_llm=not args.skip_llm)

    for check in report.checks:
        if check.skipped:
            mark, color = _SKIP, _DIM
        elif check.ok:
            mark, color = _OK, ""
        else:
            mark, color = _FAIL, "\033[31m"
        timing = f"  {check.latency_ms} ms" if check.latency_ms and not check.skipped else ""
        detail = f"  {check.detail}" if check.detail else ""
        print(f"  {mark} {check.name}{_paint(timing, _DIM)}{_paint(detail, color)}")

    passed = sum(1 for c in report.checks if c.ok and not c.skipped)
    skipped = sum(1 for c in report.checks if c.skipped)
    failed = len(report.failures)
    print()
    print(f"  合计：{passed} 通过 / {skipped} 跳过 / {failed} 失败")

    reasons = bundle.degraded_reasons
    if reasons:
        print()
        print(_paint("当前降级状态（不是错误，是可用的功能变少）：", _BOLD))
        for reason in reasons:
            print(f"  - {reason}")
        print(_paint("  补齐对应的 API Key 或把 *_ENABLED 置为 true 即可恢复", _DIM))

    if not report.ok:
        for check in report.failures:
            upstream = (check.extra or {}).get("detail")
            if upstream:
                print()
                print(_paint(f"  {check.name} 上游详情：", _BOLD))
                print(f"  {json.dumps(upstream, ensure_ascii=False)}")
        print()
        print(_paint("自检未通过。常见原因：", _BOLD))
        print("  - API Key 被限制了模型访问范围（报错含 AccessDenied / API-Key restrictions）")
        print("  - API Key 未开通该模型（去控制台确认模型已开通，不只是 Key 有效）")
        print("  - 维度配置与服务端不一致（按报错里的 actual 改 .env 的 *_DIM）")
        print("  - 网络不通 / 需要代理（curl 一下 base_url 看能不能通）")

    return 0 if report.ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="provider 连通性与向量维度自检")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（供 CI / 脚本消费）")
    parser.add_argument("--skip-llm", action="store_true", help="跳过对话模型自检")
    parser.add_argument("--debug", action="store_true", help="打印 DEBUG 级日志")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.debug else "WARNING", json_output=False)
    enable_ansi()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
