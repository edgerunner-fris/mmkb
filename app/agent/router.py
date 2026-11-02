"""意图路由与历史压缩。

**规则优先，模型兜底。** 三个意图各有非常稳定的字面特征（「这张图」、
```、寒暄词），规则能覆盖绝大多数请求且零延迟；只有在「看起来可能属于
另一类、但规则没命中」时才花一次 LLM 调用。把每一步都交给模型判断，
会让一次问答凭空多出 0.4–0.9 秒，而准确率未必更高。

路由结果的用途只有一个：决定**跑哪几路检索**、用哪套生成语气。
它不是权限边界，判错最多是召回范围不理想，不会导致错误或越权 ——
所以这里可以安心用宽松的启发式。

历史压缩解决的是另一类问题：「它为什么失败？」在没有上下文时无法检索。
把多轮对话压成一个独立问题后才送入检索与改写。压缩失败不阻断问答，
退回原问题并记 ``llm_unavailable``。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from app.agent.prompts import (
    HISTORY_COMPRESS_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    extract_json_object,
)
from app.config import Settings
from app.core.logging import get_logger
from app.core.schemas import DegradedReason, Route
from app.providers.base import ChatMessage, LLMProvider

__all__ = ["Intent", "Router", "RouterDecision", "StandaloneQuery"]

logger = get_logger(__name__)

History = Sequence[tuple[str, str]]

#: 寒暄必须**整句**匹配：用户说「你好，介绍一下检索架构」时，那是正经问题，
#: 不该被当成寒暄跳过检索。
_SMALLTALK_RE = re.compile(
    r"^\s*(?:"
    r"你好|您好|嗨|哈喽|哈啰|在吗|有人吗|"
    r"hi|hello|hey|"
    # 「谢谢你 / 多谢您」这类多了宾语的感谢同样要整句命中：
    # 漏掉它会让一句纯寒暄走完整条检索链路
    r"谢谢(?:你|您)?|谢了|多谢(?:你|您)?|感谢(?:你|您)?|thanks|thank\s+you|thx|"
    r"再见|拜拜|晚安|早上好|下午好|晚上好|"
    r"你是谁|你叫什么|你能做什么|你会什么|你是什么模型|怎么用你"
    r")\s*[!！。,.，?？~～]*\s*$",
    re.IGNORECASE,
)

#: 以图搜图 / 问图。刻意不收单独一个「图」字：「图片双向量化怎么实现」
#: 问的是实现，不是某张图。
_IMAGE_RE = re.compile(
    r"这张图|这幅图|那张图|那张图片|这张截图|这张照片|"
    r"图片里|图中|图里|截图里|图中文字|"
    r"以图搜图|类似(?:的)?图|"
    r"(?:找|搜|看|列出|显示|给我).{0,6}(?:图片|截图|照片|图表|架构图|流程图|示意图)"
)

_CODING_RE = re.compile(
    r"```|代码|函数|报错|异常|bug|调试|接口调用|"
    r"\b(?:python|javascript|typescript|java|golang|sql|shell|bash|"
    r"requests|fastapi|docker|pytest)\b",
    re.IGNORECASE,
)

#: 「可能属于非知识类，但规则没命中」的线索。命中这些词才值得问一次模型。
_UNCERTAIN_HINTS = ("图", "代码", "截图", "照片", "hello", "hi")


class Intent(StrEnum):
    SMALLTALK = "smalltalk"
    KNOWLEDGE = "knowledge"
    IMAGE_LOOKUP = "image_lookup"
    CODING = "coding"


@dataclass(slots=True)
class RouterDecision:
    intent: Intent
    source: Literal["rule", "llm", "default"] = "rule"
    confidence: float = 1.0
    reason: str = ""

    @property
    def routes(self) -> list[Route] | None:
        """该意图要跑的检索路；``None`` 表示三路全开。

        以图搜图仍要跑 BM25：图片的 caption 是文本空间里的一个分块，
        用户对图里文字的精确查询（型号、错误码）由 BM25 接住最稳。
        """
        if self.intent is Intent.IMAGE_LOOKUP:
            return [Route.DENSE_IMAGE, Route.BM25]
        return None


@dataclass(slots=True)
class StandaloneQuery:
    """补全指代后的独立问题。``rewritten`` 为 False 时就是原问题。"""

    text: str
    rewritten: bool = False
    degraded_reason: DegradedReason | None = None
    error: str = ""


class Router:
    def __init__(self, llm: LLMProvider | None, settings: Settings) -> None:
        self.llm = llm
        self._settings = settings

    # ------------------------------------------------------------------ 意图

    async def route(self, query: str) -> RouterDecision:
        text = query.strip()
        if not text:
            return RouterDecision(Intent.KNOWLEDGE, "default", 0.5, "空问题按知识问答处理")

        if _SMALLTALK_RE.match(text):
            return RouterDecision(Intent.SMALLTALK, "rule", 0.95, "整句是寒暄")
        if _IMAGE_RE.search(text):
            return RouterDecision(Intent.IMAGE_LOOKUP, "rule", 0.85, "命中看图/找图表达")
        if _CODING_RE.search(text):
            return RouterDecision(Intent.CODING, "rule", 0.8, "命中代码相关表达")

        cfg = self._settings.agent
        if (
            self.llm is not None
            and cfg.router_llm_enabled
            and self._looks_uncertain(text)
        ):
            decision = await self._route_with_llm(text)
            if decision is not None:
                return decision

        return RouterDecision(
            Intent.KNOWLEDGE, "default", 0.6, "未命中特定规则，按知识问答处理"
        )

    @staticmethod
    def _looks_uncertain(text: str) -> bool:
        """规则没命中、但字面上沾了图/代码的边 —— 值得问一次模型。"""
        lowered = text.lower()
        return len(text) <= 40 and any(hint in lowered for hint in _UNCERTAIN_HINTS)

    async def _route_with_llm(self, query: str) -> RouterDecision | None:
        llm = self.llm
        if llm is None:
            return None
        try:
            result = await llm.chat(
                [
                    ChatMessage.system(ROUTER_SYSTEM_PROMPT),
                    ChatMessage.user(query),
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001 - 路由失败退回规则结果
            logger.warning(
                "意图路由模型调用失败，按 knowledge 处理",
                extra={"error": str(exc)[:200]},
            )
            return None

        data = extract_json_object(result.text or "")
        if not data:
            return None
        raw_intent = str(data.get("intent", "")).strip().lower()
        try:
            intent = Intent(raw_intent)
        except ValueError:
            logger.warning("意图路由返回未知类别", extra={"intent": raw_intent})
            return None

        try:
            confidence = float(data.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        logger.debug(
            "意图路由由模型判定", extra={"intent": str(intent), "confidence": confidence}
        )
        return RouterDecision(
            intent,
            "llm",
            max(0.0, min(1.0, confidence)),
            str(data.get("reason", ""))[:200],
        )

    # ------------------------------------------------------------------ 历史

    async def standalone_query(
        self, question: str, history: History | None = None
    ) -> StandaloneQuery:
        """把带指代的问题补全成独立问题。

        只有**历史足够长**时才值得压缩：一两轮短对话里，指代通常也能被
        检索模型和生成模型自己处理，多一次调用只会增加延迟。
        """
        cfg = self._settings.agent
        text = question.strip()
        if not text or not history or not cfg.history_compress_enabled:
            return StandaloneQuery(text)
        if self.llm is None:
            return StandaloneQuery(text)

        recent = list(history)[-cfg.history_max_turns * 2 :] if cfg.history_max_turns else []
        total_chars = sum(len(content) for _, content in recent)
        if total_chars < cfg.history_compress_trigger_chars:
            return StandaloneQuery(text)

        transcript = "\n".join(
            f"{'用户' if role != 'assistant' else '助手'}：{content.strip()}"
            for role, content in recent
            if content.strip()
        )
        try:
            result = await self.llm.chat(
                [
                    ChatMessage.system(HISTORY_COMPRESS_SYSTEM_PROMPT),
                    ChatMessage.user(f"{transcript}\n\n当前问题：{text}"),
                ],
                temperature=0.0,
                max_tokens=300,
            )
        except Exception as exc:  # noqa: BLE001 - 压缩失败不阻断问答
            logger.warning("历史压缩失败，使用原问题", extra={"error": str(exc)[:200]})
            return StandaloneQuery(
                text,
                degraded_reason=DegradedReason.LLM_UNAVAILABLE,
                error=str(exc)[:500],
            )

        rewritten = " ".join((result.text or "").split()).strip("\"'“”")
        if not rewritten or len(rewritten) > 500:
            return StandaloneQuery(
                text,
                degraded_reason=DegradedReason.LLM_UNAVAILABLE,
                error="历史压缩返回空内容或过长",
            )
        return StandaloneQuery(rewritten, rewritten=True)
