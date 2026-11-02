"""答案生成阶段的提示词与上下文拼装。

两条硬约束来自项目需求，任何改动都不能放松：

1. **只依据 context 回答。** 资料里没有的内容不许编造；
   确实回答不了就明确说「知识库中未找到相关内容」。
2. **关键结论必须带 ``[n]`` 引用。** 用户要能顺着编号点回原文 / 原图。
   使用模型自身知识时必须标注「以下不属于知识库内容」。

模型被要求输出**纯 Markdown**而不是 JSON。原因见
:mod:`app.generation.answer` 的模块文档：引用是从答案正文反解析的，
不是让模型自报的。
"""

from __future__ import annotations

from app.core.schemas import ItemType, RetrievedItem

__all__ = [
    "ANSWER_SYSTEM_PROMPT",
    "CONTEXT_EMPTY_TEXT",
    "REFUSAL_MARKER",
    "REFUSAL_TEXT",
    "SMALLTALK_FALLBACK_TEXT",
    "SMALLTALK_SYSTEM_PROMPT",
    "answer_user_prompt",
    "context_block",
    "format_context",
]

#: 拒答时的固定说法。评测里的拒答率直接按这个子串统计，
#: 所以它必须只出现在「确实没有依据」的场合。
REFUSAL_MARKER = "知识库中未找到相关内容"
REFUSAL_TEXT = f"{REFUSAL_MARKER}。"

CONTEXT_EMPTY_TEXT = "（本次没有检索到任何资料）"

SMALLTALK_FALLBACK_TEXT = (
    "你好，我是知识库问答助手。你可以直接问我知识库里的内容，"
    "包括文档正文、表格、代码以及文档中的图片。"
)

ANSWER_SYSTEM_PROMPT = """你是知识库问答助手。回答时必须遵守以下规则：

1. 只依据 <context> 中给出的资料回答。资料没有提到的内容不要编造，
   不要用常识补全具体数字、名称、版本号或结论。
2. 每个关键结论后面都要标注来源编号，格式为 [n]，可以连续标注多个，
   例如 [1][3]。编号只能取自 <context> 中已存在的编号。
3. 如果 <context> 无法回答问题，只输出这一句：知识库中未找到相关内容。
   不要猜测，也不要给出「可能」「大概」之类的替代答案。
4. 只有在确实使用了自身知识时才可以使用外部知识，并且必须在该段落前
   单独一行写明：以下不属于知识库内容。
5. <context> 中类型为「图片」的资料是文档里的原图及其图注。可以据此描述
   图片内容，并像引用文本一样引用它。
6. 用 Markdown 输出，直接给答案。不要复述这些规则，不要输出 JSON，
   不要虚构不存在的编号。"""

SMALLTALK_SYSTEM_PROMPT = """你是知识库问答助手，正在回应用户的寒暄或对你自身能力的询问。
要求：
- 友好、简洁，一到三句话；
- 不要编造知识库里的内容；
- 如果对方问的是知识库中的具体问题，提示他直接提问。"""

_ITEM_TYPE_LABEL: dict[ItemType, str] = {
    ItemType.TEXT: "正文",
    ItemType.CODE: "代码",
    ItemType.TABLE: "表格",
    ItemType.IMAGE: "图片",
}


def context_block(index: int, item: RetrievedItem, body: str | None = None) -> str:
    """一条检索结果在 prompt 里的样子。

    首行是元数据，正文另起一行。编号必须由调用方按最终入选顺序传入，
    保证与 ``[n]`` 引用、``Citation.index`` 三者一致。

    ``body`` 供预算截断时传入已裁剪的正文；默认用完整正文。
    """
    parts = [
        f"类型：{_ITEM_TYPE_LABEL.get(item.item_type, str(item.item_type))}",
        f"来源：{item.source_name or '未知文件'}",
    ]
    if item.heading_path:
        parts.append("章节：" + " > ".join(item.heading_path))
    if item.page is not None:
        parts.append(f"页码：{item.page}")
    header = f"[{index}] " + " | ".join(parts)
    return f"{header}\n{(item.text if body is None else body).strip()}"


def format_context(items: list[RetrievedItem]) -> str:
    """把已编号的结果拼成 ``<context>`` 内容。编号从 1 起。"""
    if not items:
        return CONTEXT_EMPTY_TEXT
    return "\n\n".join(
        context_block(index, item) for index, item in enumerate(items, start=1)
    )


def answer_user_prompt(question: str, context: str) -> str:
    return f"<context>\n{context}\n</context>\n\n问题：{question}"
