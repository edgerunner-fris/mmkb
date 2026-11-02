"""配置层回归测试。

这里锁死的两类 bug 都在 Batch 1 真实验证时发生过：

1. ``.env`` 行尾注释与值之间漏了空格 —— ``python-dotenv`` 只在 ``#`` 前有空白时
   才把它当注释，于是 ``RETRIEVAL_ENABLE_DENSE_IMAGE=true# 【有默认】...``
   整个被当成布尔值去解析。
2. 空值 ``KEY=`` 被 pydantic 从 ``""`` 强转成 ``Path('.')`` —— 它不是 ``None``，
   于是所有「留空即自动派生」的路径一起塌缩到项目根目录。
"""

from __future__ import annotations

import re

import pytest
from dotenv import dotenv_values

from app import config as cfg
from app.config import (
    PROJECT_ROOT,
    PathSettings,
    Settings,
    _blank_to_none,
    collect_config_problems,
)

ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

#: 所有会被 .env 填充的配置模型（ResolvedEndpoint 是纯数据载体，不在此列）
_SETTINGS_MODELS = [
    cfg.AppSettings,
    cfg.LLMSettings,
    cfg.VisionSettings,
    cfg.TextEmbeddingSettings,
    cfg.ImageEmbeddingSettings,
    cfg.RerankSettings,
    cfg.VectorStoreSettings,
    cfg.DocStoreSettings,
    cfg.BlobSettings,
    cfg.ChunkingSettings,
    cfg.RetrievalSettings,
    cfg.BM25Settings,
    cfg.AgentSettings,
    cfg.GenerationSettings,
    cfg.IngestSettings,
    cfg.CacheSettings,
    cfg.EvalSettings,
    cfg.PathSettings,
]


def _documented_env_names() -> set[str]:
    names: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)\s*=", line)
        if m:
            names.add(m.group(1))
    return names


def _declared_env_names() -> set[str]:
    names: set[str] = set()
    for model in _SETTINGS_MODELS:
        prefix = model.model_config.get("env_prefix", "")
        for field in model.model_fields:
            names.add(f"{prefix}{field}".upper())
    return names


# --------------------------------------------------------------------------------------
# .env.example 的完整性
# --------------------------------------------------------------------------------------


def test_env_example_has_no_glued_inline_comments() -> None:
    """行尾注释必须与值之间有空白，否则会被解析进值里。"""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    offenders = [
        f"  L{i}: {line}"
        for i, line in enumerate(text.splitlines(), 1)
        if re.search(r"\S#", line)
    ]
    assert not offenders, (
        "以下行的 '#' 紧贴值，会被 python-dotenv 并进值里：\n" + "\n".join(offenders)
    )


def test_env_example_values_survive_parsing() -> None:
    values = dotenv_values(ENV_EXAMPLE)
    assert len(values) > 100, f"只解析出 {len(values)} 个变量，疑似文件被截断"
    polluted = {k: v for k, v in values.items() if v and "#" in v}
    assert not polluted, f"这些变量的值里混进了注释：{polluted}"


def test_env_example_and_config_agree() -> None:
    """.env.example 与 config.py 必须双向对齐，避免「写了但被静默忽略」。"""
    documented = _documented_env_names()
    declared = _declared_env_names()

    unknown = sorted(documented - declared)
    assert not unknown, f"这些变量在 config.py 里没有对应字段，会被静默忽略：{unknown}"

    missing = sorted(declared - documented)
    assert not missing, f"这些配置项没有写进 .env.example：{missing}"


# --------------------------------------------------------------------------------------
# 空值语义
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["", "   ", "\t", "\n", None])
def test_blank_to_none(raw: object) -> None:
    assert _blank_to_none(raw) is None


@pytest.mark.parametrize("raw", ["/tmp/x", "./rel", "C:/abs"])
def test_non_blank_passes_through(raw: str) -> None:
    assert _blank_to_none(raw) == raw


def test_blank_overrides_fall_back_to_data_dir(tmp_path) -> None:
    """这是 bug #2 的核心回归：空值必须回落，而不是变成 Path('.')。"""
    paths = PathSettings(
        data_dir=tmp_path,
        images_root="",
        chroma_root="",
        sqlite_file="   ",
        bm25_root=None,
    )
    assert paths.root == tmp_path
    assert paths.images_dir == tmp_path / "images"
    assert paths.chroma_dir == tmp_path / "chroma"
    assert paths.sqlite_path == tmp_path / "mmkb.db"
    assert paths.bm25_dir == tmp_path / "processed" / "bm25"


def test_derived_dirs_are_distinct_and_under_root(tmp_path) -> None:
    paths = PathSettings(data_dir=tmp_path)
    dirs = paths.all_dirs()
    assert len(set(dirs)) == len(dirs), f"派生目录发生塌缩：{dirs}"
    for d in dirs:
        assert d == tmp_path or tmp_path in d.parents, f"{d} 不在数据根目录之下"


# --------------------------------------------------------------------------------------
# 装配与校验
# --------------------------------------------------------------------------------------


def test_settings_loads_without_any_api_keys() -> None:
    """没有密钥也必须能构造出 Settings —— CLI、单测、CI 都依赖这一点。"""
    settings = Settings()

    assert settings.llm.model == "deepseek-flash"
    assert settings.llm.base_url == "https://api.deepseek.com"
    assert settings.llm.reasoning_effort == "high"
    assert settings.llm.thinking_enabled is True
    assert settings.llm.send_reasoning_effort is True
    assert settings.llm.effective_reasoning_effort == "high"

    assert settings.image_embed.model == "qwen3-vl-embedding"
    assert settings.image_embed.dim == 2560
    assert settings.image_embed.enable_fusion is True

    assert settings.text_embed.dim == 1024
    assert settings.text_embed.allowed_dim_set == {2048, 1536, 1024, 768, 512, 256}
    assert settings.image_embed.allowed_dim_set == {2560, 2048, 1536, 1024, 768, 512, 256}


def test_multimodal_endpoint_is_the_native_one() -> None:
    """多模态 embedding 必须走原生端点，不能用 compatible-mode。"""
    settings = Settings()
    endpoint = settings.image_embed.endpoint
    assert "compatible-mode" not in endpoint
    assert endpoint.startswith("https://dashscope.aliyuncs.com/api/v1/services/embeddings")
    assert endpoint.endswith("/multimodal-embedding/multimodal-embedding")


def test_vision_inherits_llm_when_blank(monkeypatch) -> None:
    monkeypatch.setenv("VISION_API_KEY", "")
    monkeypatch.setenv("VISION_BASE_URL", "")
    monkeypatch.setenv("VISION_MODEL", "")
    monkeypatch.setenv("LLM_API_KEY", "sk-llm-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")

    settings = Settings()
    resolved = settings.vision.resolve(settings.llm)
    assert resolved.api_key.get_secret_value() == "sk-llm-test"
    assert resolved.model == "deepseek-flash"
    assert resolved.base_url == "https://api.deepseek.com"


def test_missing_keys_are_reported_as_errors(monkeypatch) -> None:
    for var in (
        "LLM_API_KEY",
        "VISION_API_KEY",
        "TEXT_EMBED_API_KEY",
        "IMAGE_EMBED_API_KEY",
        "RERANK_API_KEY",
    ):
        monkeypatch.setenv(var, "")

    errors, _ = collect_config_problems(Settings())
    joined = " | ".join(errors)
    for expected in ("LLM_API_KEY", "TEXT_EMBED_API_KEY", "IMAGE_EMBED_API_KEY", "RERANK_API_KEY"):
        assert expected in joined, f"{expected} 的缺失没有被报告出来：{joined}"


def test_collect_config_problems_has_no_side_effects(tmp_path, monkeypatch) -> None:
    """校验函数只探测，不创建目录。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "not_yet"))
    settings = Settings()
    collect_config_problems(settings)
    assert not (tmp_path / "not_yet").exists()


def test_invalid_dim_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("TEXT_EMBED_DIM", "777")
    settings = Settings()
    errors, _ = collect_config_problems(settings)
    assert any("TEXT_EMBED_DIM=777" in e for e in errors)


def test_redacted_never_leaks_secrets(monkeypatch) -> None:
    secret = "sk-super-secret-value"
    monkeypatch.setenv("LLM_API_KEY", secret)
    monkeypatch.setenv("TEXT_EMBED_API_KEY", secret)
    monkeypatch.setenv("IMAGE_EMBED_API_KEY", secret)
    monkeypatch.setenv("RERANK_API_KEY", secret)

    snapshot = Settings().redacted()
    assert secret not in repr(snapshot)
    assert snapshot["llm"]["api_key_set"] is True
