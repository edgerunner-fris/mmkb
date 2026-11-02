"""答案生成层。

入口只有 :class:`~app.generation.answer.AnswerGenerator`：
输入「问题 + 排好序的检索结果」，输出答案、引用列表与置信度。
它不知道检索是怎么做的，也不知道答案最终从哪个接口返回 ——
这两件事分别属于 :mod:`app.retrieval` 与 :mod:`app.api`。

引用由答案正文里的 ``[n]`` 反解析，保证正文与引用列表永远一致；
流式与非流式共用同一套提示词与后处理，见 :mod:`app.generation.answer`。
"""

from app.generation.answer import (
    AnswerEvent,
    AnswerGenerator,
    GenerationOutcome,
    compute_confidence,
    parse_citations,
)

__all__ = [
    "AnswerEvent",
    "AnswerGenerator",
    "GenerationOutcome",
    "compute_confidence",
    "parse_citations",
]
