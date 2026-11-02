"""Provider 装配。

**这是全项目唯一构造具体 provider 的地方。** 业务代码要么拿这里的
:class:`ProviderBundle`，要么只依赖 :mod:`app.providers.base` 的 ABC。

降级设计
--------
不是所有 provider 都是必需的：

============================  ==========  ==========================================
能力                           必需?      缺失时的行为
============================  ==========  ==========================================
LLM                           必需        启动即报错（没有它系统没有意义）
文本向量                       必需        启动即报错（没有它无法建索引）
多模态向量（图片）              可选        检索退化为两路，标记
                                          ``image_embed_unavailable``
重排                           可选        跳过重排直接用 RRF 分数，标记
                                          ``rerank_unavailable``
============================  ==========  ==========================================

「可选」不等于「静默失败」：缺失原因会一路带到 ``/api/health`` 与
``AnswerResult.degraded_reasons``，前端能明确看到「本次回答没有走重排」。
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.config import (
    ConfigurationError,
    Settings,
    get_settings,
)
from app.core.errors import MMKBError
from app.core.schemas import DegradedReason
from app.providers.base import (
    ChatMessage,
    EmbeddingProvider,
    LLMProvider,
    RerankProvider,
)
from app.providers.dashscope_embedding import (
    DashScopeImageEmbedding,
    DashScopeTextEmbedding,
)
from app.providers.deepseek_llm import DeepSeekLLM
from app.providers.rerank import DashScopeRerank, SiliconFlowRerank
from app.providers.siliconflow_embedding import SiliconFlowTextEmbedding

__all__ = [
    "CheckResult",
    "ProviderBundle",
    "VerifyReport",
    "build_image_embedding",
    "build_llm",
    "build_rerank",
    "build_text_embedding",
    "close_providers",
    "get_providers",
    "provider_matrix",
    "verify_providers",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# 单个 provider 的构造
# --------------------------------------------------------------------------------------


def build_llm(settings: Settings) -> LLMProvider:
    """构造对话模型。缺 Key 直接抛 :class:`ConfigurationError`。"""
    llm = settings.llm
    if not llm.has_api_key:
        raise ConfigurationError(["LLM_API_KEY 未配置，无法构造对话模型"])
    if llm.provider == "deepseek":
        return DeepSeekLLM(llm)
    raise ConfigurationError([f"不支持的 LLM_PROVIDER={llm.provider!r}（可选：deepseek）"])


def build_text_embedding(settings: Settings) -> EmbeddingProvider:
    """构造文本空间 embedding。"""
    cfg = settings.text_embed
    if not cfg.has_api_key:
        raise ConfigurationError(["TEXT_EMBED_API_KEY 未配置，无法构造文本向量模型"])
    if cfg.provider == "dashscope":
        return DashScopeTextEmbedding(cfg)
    if cfg.provider == "siliconflow":
        return SiliconFlowTextEmbedding(cfg)
    raise ConfigurationError(
        [f"不支持的 TEXT_EMBED_PROVIDER={cfg.provider!r}（可选：dashscope、siliconflow）"]
    )


def build_image_embedding(settings: Settings) -> EmbeddingProvider | None:
    """构造多模态（图像）空间 embedding。

    未启用或缺 Key 时返回 ``None`` —— 这是**允许的降级**，
    调用方必须据此摘掉以文搜图这一路，并用
    ``DegradedReason.IMAGE_EMBED_UNAVAILABLE`` 标记。
    """
    cfg = settings.image_embed
    if not cfg.enabled:
        logger.warning("多模态向量未启用，以文搜图/以图搜图将不可用")
        return None
    if not cfg.has_api_key:
        logger.warning("IMAGE_EMBED_API_KEY 未配置，以文搜图/以图搜图将不可用")
        return None
    if cfg.provider != "dashscope":
        raise ConfigurationError(
            [f"不支持的 IMAGE_EMBED_PROVIDER={cfg.provider!r}（目前仅 dashscope）"]
        )
    return DashScopeImageEmbedding(cfg)


def build_rerank(settings: Settings) -> RerankProvider | None:
    """构造重排 provider。未启用或缺 Key 时返回 ``None``（允许的降级）。"""
    cfg = settings.rerank
    if not cfg.enabled:
        logger.info("重排未启用，检索结果将直接使用 RRF 融合分数排序")
        return None
    if not cfg.has_api_key:
        logger.warning("RERANK_API_KEY 未配置，本次运行跳过重排")
        return None
    if cfg.provider == "dashscope":
        return DashScopeRerank(cfg)
    if cfg.provider == "siliconflow":
        return SiliconFlowRerank(cfg)
    raise ConfigurationError(
        [f"不支持的 RERANK_PROVIDER={cfg.provider!r}（可选：dashscope、siliconflow）"]
    )


# --------------------------------------------------------------------------------------
# 装配结果
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class ProviderBundle:
    """一次进程生命周期内共用的 provider 集合。

    所有 provider 内部都持有 ``httpx.AsyncClient`` 或 ``AsyncOpenAI``，
    连接池是复用的，**不要每次请求都重新构造**。
    """

    llm: LLMProvider
    text_embedding: EmbeddingProvider
    image_embedding: EmbeddingProvider | None = None
    rerank: RerankProvider | None = None

    @property
    def degraded_reasons(self) -> list[DegradedReason]:
        """必填 provider 齐备时，因可选能力缺失而产生的降级标记。"""
        reasons: list[DegradedReason] = []
        if self.image_embedding is None:
            reasons.append(DegradedReason.IMAGE_EMBED_UNAVAILABLE)
        if self.rerank is None:
            reasons.append(DegradedReason.RERANK_UNAVAILABLE)
        return reasons

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "llm": True,
            "dense_text": True,
            "dense_image": self.image_embedding is not None,
            "rerank": self.rerank is not None,
        }

    async def aclose(self) -> None:
        """关闭所有底层连接池。任一失败不影响其余。"""
        providers: list[Any] = [
            self.llm,
            self.text_embedding,
            self.image_embedding,
            self.rerank,
        ]
        for provider in providers:
            if provider is None:
                continue
            try:
                await provider.aclose()
            except Exception:  # noqa: BLE001 - 关闭失败不该阻止进程退出
                logger.debug("关闭 provider 失败", exc_info=True)


# --------------------------------------------------------------------------------------
# 缓存入口
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_providers() -> ProviderBundle:
    """构造并缓存 provider 集合。

    构造过程只创建客户端对象，不发网络请求，所以同步函数是安全的。
    真正的连通性校验在 :func:`verify_providers`，由 ``/api/health`` 与
    ``scripts/check_providers.py`` 显式触发。
    """
    settings = get_settings()

    # 四个都构造一遍再统一报错：缺 Key 时一次性列全，而不是让用户跑一次补一个。
    problems: list[str] = []
    built: dict[str, Any] = {}
    for key, builder in (
        ("llm", build_llm),
        ("text_embedding", build_text_embedding),
        ("image_embedding", build_image_embedding),
        ("rerank", build_rerank),
    ):
        try:
            built[key] = builder(settings)
        except ConfigurationError as exc:
            problems.extend(exc.problems)
            built[key] = None
    if problems:
        raise ConfigurationError(problems)

    bundle = ProviderBundle(
        llm=built["llm"],
        text_embedding=built["text_embedding"],
        image_embedding=built["image_embedding"],
        rerank=built["rerank"],
    )
    logger.info(
        "provider 装配完成",
        extra={
            "llm": f"{settings.llm.provider}/{settings.llm.model}",
            "text_embedding": f"{settings.text_embed.provider}/"
            f"{settings.text_embed.resolved_model}",
            "image_embedding": (
                f"{settings.image_embed.provider}/{settings.image_embed.model}"
                if bundle.image_embedding
                else "disabled"
            ),
            "rerank": (
                f"{settings.rerank.provider}/{settings.rerank.resolved_model}"
                if bundle.rerank
                else "disabled"
            ),
        },
    )
    return bundle


async def close_providers() -> None:
    """关闭并清空缓存。进程退出（lifespan shutdown）时调用。"""
    if get_providers.cache_info().currsize:
        await get_providers().aclose()
    get_providers.cache_clear()


def provider_matrix(settings: Settings | None = None) -> dict[str, dict[str, Any]]:
    """给 ``/api/health`` 用的 provider 一览（不含密钥）。"""
    settings = settings or get_settings()
    return {
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "base_url": settings.llm.base_url,
            "reasoning_effort": settings.llm.effective_reasoning_effort,
            "configured": settings.llm.has_api_key,
        },
        "text_embedding": {
            "provider": settings.text_embed.provider,
            "model": settings.text_embed.resolved_model,
            "dim": settings.text_embed.dim,
            "collection_suffix": "__text",
            "configured": settings.text_embed.has_api_key,
        },
        "image_embedding": {
            "provider": settings.image_embed.provider,
            "model": settings.image_embed.model,
            "dim": settings.image_embed.dim,
            "enable_fusion": settings.image_embed.enable_fusion,
            "collection_suffix": "__image",
            "enabled": settings.image_embed.enabled,
            "configured": settings.image_embed.has_api_key,
        },
        "rerank": {
            "provider": settings.rerank.provider,
            "model": settings.rerank.resolved_model,
            "enabled": settings.rerank.enabled,
            "configured": settings.rerank.has_api_key,
            "note": "重排器只接受纯文本，图片候选按其 caption + 章节路径参与重排",
        },
    }


# --------------------------------------------------------------------------------------
# 连通性 / 维度校验
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    latency_ms: int = 0
    skipped: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        if self.skipped:
            mark = "－"
        else:
            mark = "✓" if self.ok else "✗"
        tail = f"  {self.detail}" if self.detail else ""
        timing = f" ({self.latency_ms} ms)" if self.latency_ms and not self.skipped else ""
        return f"  {mark} {self.name}{timing}{tail}"


@dataclass(slots=True)
class VerifyReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """必需项全部通过即为通过；跳过（未启用）不算失败。"""
        return all(check.ok or check.skipped for check in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.ok and not check.skipped]

    def render(self) -> str:
        lines = [check.render() for check in self.checks]
        passed = sum(1 for c in self.checks if c.ok)
        skipped = sum(1 for c in self.checks if c.skipped)
        failed = len(self.failures)
        lines.append(f"  合计：{passed} 通过 / {skipped} 跳过 / {failed} 失败")
        return "\n".join(lines)


async def verify_providers(
    bundle: ProviderBundle | None = None,
    *,
    include_llm: bool = True,
) -> VerifyReport:
    """**真实调用**各 provider，验证连通性与向量维度。

    只做构造不校验是不够的：维度配错、模型名写错、Key 没开通，
    这三类问题在构造期都发现不了，却会让整个向量库默默失效。
    这里宁可多花几秒钟和几分钱，也要在启动时把它们逼出来。

    每一项都独立捕获异常 —— 一个 provider 挂了不应该让整份报告只剩一行错。
    """
    bundle = bundle or get_providers()
    report = VerifyReport()

    if include_llm:
        report.checks.append(await _check_llm(bundle.llm))
    report.checks.append(await _check_text_embedding(bundle.text_embedding))
    report.checks.append(await _check_image_embedding(bundle.image_embedding))
    report.checks.append(await _check_rerank(bundle.rerank))
    return report


async def _check_llm(provider: LLMProvider) -> CheckResult:
    started = time.perf_counter()
    try:
        result = await provider.chat(
            [ChatMessage.user("请只回复两个字：正常")], temperature=0.0, max_tokens=16
        )
    except MMKBError as exc:
        return _failed("LLM 对话", exc, started)
    except Exception as exc:  # noqa: BLE001
        return _failed("LLM 对话", exc, started)
    return CheckResult(
        name="LLM 对话",
        ok=True,
        detail=f"{provider.model} 返回“{_snippet(result.text)}”",
        latency_ms=_elapsed(started),
        extra={"degraded_reasons": result.degraded_reasons},
    )


async def _check_text_embedding(provider: EmbeddingProvider) -> CheckResult:
    started = time.perf_counter()
    try:
        result = await provider.embed_texts(["这是一条向量维度自检。"])
    except MMKBError as exc:
        return _failed("文本向量", exc, started)
    except Exception as exc:  # noqa: BLE001
        return _failed("文本向量", exc, started)

    actual = len(result.vectors[0]) if result.vectors else 0
    if actual != provider.dim:
        return CheckResult(
            name="文本向量",
            ok=False,
            detail=f"维度不符：配置 {provider.dim}，实际 {actual}",
            latency_ms=_elapsed(started),
        )
    return CheckResult(
        name="文本向量",
        ok=True,
        detail=f"{provider.model} dim={actual} → {{kb_id}}__text",
        latency_ms=_elapsed(started),
    )


async def _check_image_embedding(provider: EmbeddingProvider | None) -> CheckResult:
    if provider is None:
        return CheckResult(
            name="多模态向量",
            ok=True,
            skipped=True,
            detail="未启用，以文搜图将不可用（降级，非失败）",
        )

    started = time.perf_counter()
    fused_dim = 0
    try:
        # 1) 以文搜图路径：text-only 输入走融合空间
        text_result = await provider.embed_fused(["一张流程图的示意图"])
        fused_dim = len(text_result.vectors[0]) if text_result.vectors else 0

        # 2) 图片入库路径：现造一张小图，验证真实图片能被接受。
        #    只测文本路径是不够的 —— 图片的 base64 编码、尺寸限制、
        #    contents 结构都可能单独出问题。
        image_result = await provider.embed_images([_make_probe_image()])
        image_dim = len(image_result.vectors[0]) if image_result.vectors else 0
    except MMKBError as exc:
        return _failed("多模态向量", exc, started)
    except Exception as exc:  # noqa: BLE001
        return _failed("多模态向量", exc, started)

    if fused_dim != provider.dim or image_dim != provider.dim:
        return CheckResult(
            name="多模态向量",
            ok=False,
            detail=(
                f"维度不符：配置 {provider.dim}，文本查询 {fused_dim}，图片 {image_dim}"
            ),
            latency_ms=_elapsed(started),
        )
    return CheckResult(
        name="多模态向量",
        ok=True,
        detail=(
            f"{provider.model} dim={image_dim} enable_fusion="
            f"{getattr(provider, 'enable_fusion', '?')} → {{kb_id}}__image"
        ),
        latency_ms=_elapsed(started),
    )


async def _check_rerank(provider: RerankProvider | None) -> CheckResult:
    if provider is None:
        return CheckResult(
            name="重排",
            ok=True,
            skipped=True,
            detail="未启用，检索结果将直接使用 RRF 分数（降级，非失败）",
        )

    started = time.perf_counter()
    try:
        result = await provider.rerank(
            "系统支持哪些文件格式？",
            ["系统支持 PDF、Word、PPT、Markdown 等格式。", "今天天气不错。"],
            top_n=2,
        )
    except MMKBError as exc:
        return _failed("重排", exc, started)
    except Exception as exc:  # noqa: BLE001
        return _failed("重排", exc, started)

    if not result.hits:
        return CheckResult(
            name="重排",
            ok=False,
            detail="返回了空的 hits",
            latency_ms=_elapsed(started),
        )
    top = max(result.hits, key=lambda hit: hit.score)
    return CheckResult(
        name="重排",
        ok=True,
        detail=f"{provider.model} 命中 {len(result.hits)} 条，最高分 {top.score:.4f}",
        latency_ms=_elapsed(started),
    )


def _make_probe_image() -> bytes:
    """生成一张用于自检的小 JPEG。

    刻意画点东西而不是纯色块：纯色图在部分模型上会走特殊分支，
    测不出真实图片路径的问题。
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (160, 120), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([12, 12, 148, 108], outline=(60, 90, 160), width=3)
    draw.line([12, 60, 148, 60], fill=(60, 90, 160), width=2)
    draw.text((24, 30), "MMKB probe", fill=(30, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _snippet(text: str, limit: int = 40) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def _failed(name: str, exc: BaseException, started: float) -> CheckResult:
    extra: dict[str, Any] = {}
    if isinstance(exc, MMKBError):
        detail = f"[{exc.code}] {exc.message}"
        # 错误消息只放人话；结构化的 url / status / 上游原始 body
        # 留在 extra 里，供 --json 与自检脚本的「上游详情」打印。
        # 没有这一步，400 只剩一句「拒绝了请求（400）」，
        # 真正的原因（例如 Key 被限制模型）就丢了。
        if exc.detail:
            extra["detail"] = exc.detail
    else:
        detail = f"[{type(exc).__name__}] {exc}"
    return CheckResult(
        name=name,
        ok=False,
        detail=detail,
        latency_ms=_elapsed(started),
        extra=extra,
    )
