#!/usr/bin/env python
"""打印当前生效的配置（密钥脱敏）与配置校验结果。

用法::

    python scripts/show_config.py                 # 人类可读
    python scripts/show_config.py --json          # 机器可读
    python scripts/show_config.py --init-dirs     # 顺带创建所有数据目录
    python scripts/show_config.py --strict        # 有致命配置问题时退出码 1

Batch 1 的验收命令就是 ``python scripts/show_config.py``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import (  # noqa: E402
    PROJECT_ROOT,
    Settings,
    collect_config_problems,
    get_settings,
)
from app.core.console import enable_ansi, paint  # noqa: E402

# --------------------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------------------

_GROUPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "应用",
        "app",
        [
            ("运行环境", "env"),
            ("日志级别", "log_level"),
            ("项目根目录", "project_root"),
        ],
    ),
    (
        "生成模型 LLM",
        "llm",
        [
            ("provider", "provider"),
            ("base_url", "base_url"),
            ("model", "model"),
            ("reasoning_effort", "reasoning_effort"),
            ("思考模式开启", "thinking_enabled"),
            ("temperature", "temperature"),
            ("max_tokens", "max_tokens"),
            ("timeout_s", "timeout_s"),
            ("max_retries", "max_retries"),
            ("max_concurrency", "max_concurrency"),
            ("API Key 已配置", "api_key_set"),
        ],
    ),
    (
        "视觉模型 Vision（caption 生成）",
        "vision",
        [
            ("启用", "enabled"),
            ("base_url", "base_url"),
            ("model", "model"),
            ("reasoning_effort", "reasoning_effort"),
            ("max_concurrency", "max_concurrency"),
            ("API Key 已配置", "api_key_set"),
            ("继承 LLM 的 Key", "inherits_llm_key"),
        ],
    ),
    (
        "文本向量 Text Embedding",
        "text_embed",
        [
            ("启用", "enabled"),
            ("provider", "provider"),
            ("base_url", "base_url"),
            ("model", "model"),
            ("dim", "dim"),
            ("batch_size", "batch_size"),
            ("normalize", "normalize"),
            ("API Key 已配置", "api_key_set"),
        ],
    ),
    (
        "多模态向量 Image Embedding（原生 API）",
        "image_embed",
        [
            ("启用", "enabled"),
            ("provider", "provider"),
            ("endpoint", "endpoint"),
            ("model", "model"),
            ("dim", "dim"),
            ("enable_fusion", "enable_fusion"),
            ("batch_size", "batch_size"),
            ("normalize", "normalize"),
            ("API Key 已配置", "api_key_set"),
        ],
    ),
    (
        "重排 Rerank",
        "rerank",
        [
            ("启用", "enabled"),
            ("provider", "provider"),
            ("endpoint", "endpoint"),
            ("model", "model"),
            ("max_docs_per_request", "max_docs_per_request"),
            ("图片候选模板", "image_doc_template"),
            ("API Key 已配置", "api_key_set"),
        ],
    ),
    (
        "向量库",
        "vectorstore",
        [
            ("backend", "backend"),
            ("chroma_dir", "chroma_dir"),
            # 显示 effective_metric 而不是 chroma_space：后者只是旧名字的兼容入口，
            # 直接打印它会在「只设了 VECTORSTORE_METRIC」时显示空值，误导排查
            ("距离度量", "metric"),
            ("PG DSN 已配置", "pg_dsn_set"),
        ],
    ),
    ("元数据存储", "docstore", [("sqlite_path", "sqlite_path")]),
    (
        "图片处理",
        "blob",
        [
            ("images_dir", "images_dir"),
            ("最小尺寸", "min_size"),
            ("phash 边长", "phash_size"),
            ("phash 汉明阈值", "phash_hamming_threshold"),
            ("纯色 std 阈值", "blank_std_threshold"),
            ("去重范围", "dedupe_scope"),
        ],
    ),
    (
        "切分",
        "chunking",
        [
            ("min_tokens", "min_tokens"),
            ("max_tokens", "max_tokens"),
            ("overlap_tokens", "overlap_tokens"),
            ("token 编码", "encoding"),
        ],
    ),
    (
        "检索与融合",
        "retrieval",
        [
            ("BM25 路", "enable_bm25"),
            ("文本向量路", "enable_dense_text"),
            ("多模态向量路", "enable_dense_image"),
            ("重排", "enable_rerank"),
            ("每路召回数", "top_k_per_route"),
            ("RRF k", "rrf_k"),
            ("三路权重", "route_weights"),
            ("重排输入数", "rerank_input_k"),
            ("最终 top-k", "final_top_k"),
        ],
    ),
    (
        "BM25",
        "bm25",
        [
            ("分词器", "tokenizer"),
            ("k1", "k1"),
            ("b", "b"),
            ("index_dir", "index_dir"),
            ("索引落盘", "persist"),
        ],
    ),
    (
        "Agent",
        "agent",
        [
            ("最大检索轮数", "max_rounds"),
            ("相关性阈值", "relevance_threshold"),
            ("相关性打分", "judge_enabled"),
            ("多查询改写", "query_rewrite_enabled"),
            ("改写条数", "query_rewrite_n"),
        ],
    ),
    (
        "生成",
        "generation",
        [
            ("temperature", "temperature"),
            ("强制引用", "require_citations"),
            ("无依据必须拒答", "refuse_when_empty"),
            ("允许标注后的模型知识", "allow_model_knowledge"),
            ("最大引用数", "max_citations"),
        ],
    ),
]

_YES = "是"
_NO = "否"


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return _YES if value else _NO
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]"
    if value is None:
        return "-"
    return str(value)


def _mark(value: object) -> str:
    """给布尔值加颜色提示：false 的健康检查项标黄。"""
    if value is True:
        return paint(_YES, "green")
    if value is False:
        return paint(_NO, "yellow")
    return _fmt(value)


def render(snapshot: dict, errors: list[str], warnings: list[str]) -> None:
    width = 74
    print()
    print("=" * width)
    print("  mmkb 配置快照（密钥已脱敏）")
    print("=" * width)

    for title, key, fields in _GROUPS:
        group = snapshot.get(key)
        if not group:
            continue
        print("\n" + paint(f"[{title}]", "bold"))
        for label, field in fields:
            if field not in group:
                continue
            raw = group[field]
            shown = _mark(raw) if isinstance(raw, bool) else _fmt(raw)
            print(f"  {label:<26} {shown}")

    paths = snapshot.get("paths", {})
    if paths:
        print("\n" + paint("[数据目录]", "bold"))
        for name, value in sorted(paths.items()):
            exists = "存在" if Path(value).exists() else paint("待创建", "yellow")
            print(f"  {name:<26} {value}  ({exists})")

    print("\n" + "-" * width)
    if errors:
        print(paint(f"✗ 配置校验未通过：{len(errors)} 项致命问题", "red"))
        for item in errors:
            print(f"    - {item}")
    else:
        print(paint("✓ 配置校验通过，无致命问题", "green"))

    if warnings:
        print("\n" + paint(f"! 提醒：{len(warnings)} 项", "yellow"))
        for item in warnings:
            print(f"    - {item}")

    print("-" * width + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="打印 mmkb 生效配置与校验结果")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--init-dirs", action="store_true", help="创建所有数据目录")
    parser.add_argument(
        "--strict", action="store_true", help="存在致命配置问题时以退出码 1 结束"
    )
    args = parser.parse_args(argv)

    enable_ansi()
    settings: Settings = get_settings()

    if args.init_dirs:
        created = settings.paths.ensure()
        for d in created:
            print(f"[init] 已创建 {d}")

    errors, warnings = collect_config_problems(settings)

    if args.json:
        print(
            json.dumps(
                {
                    "project_root": str(PROJECT_ROOT),
                    "config": settings.redacted(),
                    "errors": errors,
                    "warnings": warnings,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    else:
        render(settings.redacted(), errors, warnings)

    return 1 if (args.strict and errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
