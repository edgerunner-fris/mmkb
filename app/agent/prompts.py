"""Agent 阶段（意图路由 / 历史压缩 / 相关性判定）的提示词。

三处调用都要求模型输出**结构化 JSON**，但解析一律做成「宽容但失败即降级」：
多一层 ```json 围栏、前后多一句解释、字段名大小写不一致，都不该让整条
问答链路挂掉。解析器只接受它能确定的内容，其余交给调用方降级。
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "HISTORY_COMPRESS_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "ROUTER_SYSTEM_PROMPT",
    "extract_json_object",
]

ROUTER_SYSTEM_PROMPT = """你是知识库问答系统的意图分类器。把用户问题分到下列四类之一：

- smalltalk：寒暄、感谢、道别，或询问助手自身能力。不需要查知识库。
- image_lookup：用户想找图片、截图、照片、图表、流程图，或询问某张图里有什么。
- coding：用户在要代码，或在问报错、异常、某个 API/SDK/库怎么调用。
- knowledge：其余需要查知识库才能回答的问题。

拿不准时选 knowledge。只输出 JSON：
{"intent": "smalltalk|image_lookup|coding|knowledge", "confidence": 0.0, "reason": "一句话"}"""

JUDGE_SYSTEM_PROMPT = """你是检索结果的相关性评审。用户会给出一个问题和若干候选资料，
判断这些资料整体能否回答问题。评分标准：

- 1.0：至少一条资料直接回答了问题；
- 0.6–0.9：包含回答所需的关键信息，但需要综合或推理；
- 0.3–0.5：主题相关，但不足以回答问题；
- 0.0：与问题无关。

只输出 JSON：{"score": 0.0, "reason": "一句话说明依据"}"""

HISTORY_COMPRESS_SYSTEM_PROMPT = """把下面的多轮对话压缩成一个**独立、完整、可直接检索**的问题。

要求：
- 补全代词与省略（「它」「这个方案」「那第二步」要还原成具体对象）；
- 保留专有名词、型号、文件名、错误码；
- 不回答问题，只重写问题；
- 只输出改写后的问题本身，不要解释，不要加引号。"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """从模型输出里取一个 JSON 对象；失败返回 ``None``。

    与 :mod:`app.retrieval.query_rewrite` 的解析策略一致：先整体解析，
    失败再截取第一个 ``{...}``。不做更激进的猜测 —— 猜错一个意图、
    猜错一个分数，比直接降级更难排查。
    """
    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        match = _OBJECT_RE.search(text)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return data if isinstance(data, dict) else None
