"""多查询改写（multi-query）。

用户的问题往往不适合直接拿去检索：口语化、带指代（「它」「这个方案」）、
术语与文档里的写法不一致。改写出几个**互补**的查询分别召回，再一起进 RRF，
通常能显著提升召回率 —— 代价是 N 倍的检索请求。

设计上的三个约束
----------------
1. **原问题永远排第一，权重最高。** 改写可能跑偏；把原问题当成「另一个候选」
   而不是「检索入口」，跑偏时的损失才可控。
2. **改写失败不是错误。** LLM 不可用、返回的不是 JSON、返回空列表 ——
   一律退回「只用原问题」，并记 ``LLM_UNAVAILABLE``。检索本身仍然完整。
3. **改写结果不直接给用户看**（除非 ``--debug``）。它只影响召回，不影响答案。

这个方法对应 A/B 消融里的 ``multi_query`` 组（见 Batch 8）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.config import Settings
from app.core.logging import get_logger
from app.core.schemas import DegradedReason
from app.providers.base import ChatMessage, LLMProvider

__all__ = ["QueryRewriteResult", "QueryRewriter"]

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "你是检索查询改写器。用户的问题会用来检索一个包含文档与图片的知识库。\n"
    "你的任务是把问题改写成若干条**互补**的检索查询，覆盖不同召回角度：\n"
    "1. 关键词式：去掉语气词与疑问结构，保留专有名词、型号、错误码；\n"
    "2. 同义改写：换一种常见说法，术语在正式表述与口语之间切换；\n"
    "3. 图表角度：如果问题可能在问截图、流程图、架构图或表格，给出一条\n"
    "   面向「图片内容/图注」的表达；问题与图片无关时不要硬凑。\n"
    "要求：\n"
    "- 不改变原意，不回答问题，不添加原问题里没有的事实；\n"
    "- 每条查询独立完整，不依赖上下文；\n"
    "- 不要编号，不要解释。\n"
    '只输出 JSON：{"queries": ["查询1", "查询2"]}'
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(slots=True)
class QueryRewriteResult:
    queries: list[str] = field(default_factory=list)
    degraded_reason: DegradedReason | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.degraded_reason is None


def _strip_fence(text: str) -> str:
    """去掉模型偶尔加的 ```json 围栏。

    结构化输出在多数供应商上已经可靠，但「偶尔多一层围栏」仍是常见故障 ——
    为这一个字符让整条改写链路降级不值得。
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_RE.sub("", cleaned).strip()
    return cleaned


def _extract_queries(raw: str) -> list[str]:
    """从模型输出里取出查询列表。兼容三种形状：
    ``{"queries": [...]}``、裸数组、以及前后带解释文字的 JSON。
    """
    text = _strip_fence(raw)
    if not text:
        return []

    data: object
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # 模型在 JSON 前后写了说明文字时，退一步只截取第一个 ``[]`` / ``{}``。
        # 这里不做更激进的猜测 —— 解析失败按降级处理，比猜出半句话继续检索安全。
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match is None:
            return []
        try:
            data = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            return []

    if isinstance(data, dict):
        data = data.get("queries") or data.get("query") or []
    if isinstance(data, str):
        data = [data]
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


class QueryRewriter:
    """LLM 驱动的多查询改写。"""

    def __init__(self, llm: LLMProvider, settings: Settings) -> None:
        self.llm = llm
        self._settings = settings

    async def rewrite(self, query: str, *, n: int | None = None) -> QueryRewriteResult:
        """返回 ``[原问题, 改写1, 改写2, ...]``，最多 ``n + 1`` 条。

        原问题永远是第一个元素，且调用方按「第一个权重 1.0、其余更低」
        处理 —— 这是本类唯一的硬契约。
        """
        query = query.strip()
        if not query:
            return QueryRewriteResult(queries=[])

        wanted = n if n is not None else self._settings.agent.query_rewrite_n
        wanted = max(1, min(6, wanted))

        try:
            result = await self.llm.chat(
                [
                    ChatMessage.system(_SYSTEM_PROMPT),
                    ChatMessage.user(f"用户问题：{query}"),
                ],
                temperature=0.3,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001 - 改写在任何意义上都是可选的
            logger.warning(
                "查询改写失败，只使用原问题", extra={"error": str(exc)[:200]}
            )
            return QueryRewriteResult(
                queries=[query],
                degraded_reason=DegradedReason.LLM_UNAVAILABLE,
                error=str(exc)[:500],
            )

        candidates = _extract_queries(result.text)
        variants: list[str] = []
        seen = {query}
        for candidate in candidates:
            # 过长的「改写」通常是模型开始回答问题了，直接丢掉
            if len(candidate) > 200 or candidate in seen:
                continue
            seen.add(candidate)
            variants.append(candidate)
            if len(variants) >= wanted:
                break

        if not variants:
            logger.info("查询改写没有产出可用变体，只使用原问题")
        return QueryRewriteResult(queries=[query, *variants])
