"""Agent 层：意图路由、检索工具、相关性判定与问答循环。

分工：

* :mod:`app.agent.router` —— 规则优先的意图分类 + 多轮历史压缩；
* :mod:`app.agent.tools` —— 意图到检索策略的映射（``search_knowledge`` / ``search_images``）；
* :mod:`app.agent.graph` —— 最多两轮的检索-判定循环，产出 ``AnswerResult``；
* :mod:`app.agent.prompts` —— 三处结构化输出的提示词与 JSON 解析。

这一层不直接调用任何厂商 SDK，也不自己实现检索或生成 ——
它只负责「下一步做什么」。
"""

from app.agent.graph import AnswerAgent, Judge, JudgeOutcome
from app.agent.router import Intent, Router, RouterDecision, StandaloneQuery
from app.agent.tools import SearchTools, ToolCall, ToolName, tool_catalog

__all__ = [
    "AnswerAgent",
    "Intent",
    "Judge",
    "JudgeOutcome",
    "Router",
    "RouterDecision",
    "SearchTools",
    "StandaloneQuery",
    "ToolCall",
    "ToolName",
    "tool_catalog",
]
