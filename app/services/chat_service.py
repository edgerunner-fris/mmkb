"""问答服务：会话、历史、Agent 调用与消息落库。

路由函数不应该知道「历史取最近几轮」「用户消息先写还是后写」
「流式结束后才落 assistant 消息」这些顺序问题。它们都集中在这里。

一个容易忽略的点：**当前问题不能进入历史**。历史必须在追加用户消息之前
读取，否则模型会在「历史」里看到用户刚刚问的同一句话，压缩器也可能把它
当成上一轮的问题再改写一次。顺序是：

1. 取历史 → 2. 写用户消息 → 3. 调 Agent → 4. 写 assistant 消息

流式模式下，assistant 消息只在**正常收到 done 之后**落库。客户端中途断开时
Python 会向生成器抛 ``GeneratorExit``，此时在 ``finally`` 里 await 数据库
写入既不稳定也没有意义；留下的用户消息足以说明「这次问答发生过但没完成」。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from app.core.errors import MMKBError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.core.schemas import AnswerResult, Citation, Message
from app.core.schemas import Session as ChatSession
from app.generation.answer import AnswerEvent

if TYPE_CHECKING:
    from app.services.container import AppContainer

__all__ = ["ChatService"]

logger = get_logger(__name__)


class ChatService:
    def __init__(self, container: AppContainer) -> None:
        self._c = container

    # ------------------------------------------------------------------ 预检

    async def preflight(self, kb_id: str):
        """在建立 SSE 连接**之前**确认知识库存在、LLM 可用。

        流式响应一旦发出 200 与 ``text/event-stream`` 头，就无法再改回
        404/503 的 JSON 错误。所有「可同步判定」的失败都必须在这里暴露。
        """
        kb = await self._c.docstore.require_kb(kb_id)
        self._c.require_agent()
        return kb

    # ------------------------------------------------------------------ 非流式

    async def ask(
        self,
        kb_id: str,
        question: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        rewrite: bool | None = None,
        debug: bool = False,
    ) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValidationError("问题不能为空")

        kb = await self._c.docstore.require_kb(kb_id)
        agent = self._c.require_agent()
        session = await self._c.docstore.get_or_create_session(kb.kb_id, session_id)
        history = await self._history(session.session_id)

        await self._c.docstore.add_message(session.session_id, "user", question)
        result = await agent.answer(
            kb,
            question,
            history=history,
            top_k=top_k,
            use_rerank=use_rerank,
            rewrite=rewrite,
            session_id=session.session_id,
        )
        await self._c.docstore.add_message(
            session.session_id,
            "assistant",
            result.answer_markdown,
            citations=result.citations,
        )
        if not debug:
            result = result.model_copy(update={"debug_trace": []})
        return result

    # ------------------------------------------------------------------ 流式

    async def stream(
        self,
        kb_id: str,
        question: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        rewrite: bool | None = None,
        debug: bool = False,
    ) -> AsyncIterator[AnswerEvent]:
        """产出 ``AnswerEvent``。落库在 ``done`` 之后完成。"""
        answer_parts: list[str] = []
        citations: list[Citation] = []
        try:
            question = question.strip()
            if not question:
                raise ValidationError("问题不能为空")

            kb = await self._c.docstore.require_kb(kb_id)
            agent = self._c.require_agent()
            session = await self._c.docstore.get_or_create_session(kb.kb_id, session_id)
            history = await self._history(session.session_id)
            await self._c.docstore.add_message(session.session_id, "user", question)

            async for event in agent.answer_stream(
                kb,
                question,
                history=history,
                top_k=top_k,
                use_rerank=use_rerank,
                rewrite=rewrite,
                session_id=session.session_id,
            ):
                if event.type == "token":
                    answer_parts.append(event.text)
                elif event.type == "citations":
                    citations = _parse_citations(event.data.get("citations"))
                if event.type == "done" and not debug:
                    event = AnswerEvent(
                        "done",
                        data={k: v for k, v in event.data.items() if k != "debug_trace"},
                    )
                yield event

            text = "".join(answer_parts).strip()
            if text:
                await self._c.docstore.add_message(
                    session.session_id, "assistant", text, citations=citations
                )
        except MMKBError as exc:
            logger.warning(
                "问答流失败", extra={"kb_id": kb_id, "code": exc.code, "error": str(exc)[:200]}
            )
            yield AnswerEvent("error", data=exc.to_dict()["error"])
        except Exception as exc:  # noqa: BLE001 - 流已开始，只能以事件告知
            logger.exception("问答流出现未预期异常", extra={"kb_id": kb_id})
            yield AnswerEvent(
                "error",
                data={
                    "code": "internal_error",
                    "message": "问答过程中发生未预期错误",
                    "detail": f"{type(exc).__name__}: {exc}"[:500],
                },
            )

    # ------------------------------------------------------------------ 会话

    async def messages(self, session_id: str, *, limit: int = 50) -> list[Message]:
        await self._c.docstore.require_session(session_id)
        return await self._c.docstore.list_messages(session_id, limit=limit)

    async def require_session(self, session_id: str) -> ChatSession:
        return await self._c.docstore.require_session(session_id)

    async def delete_session(self, session_id: str) -> None:
        removed = await self._c.docstore.delete_session(session_id)
        if not removed:
            raise NotFoundError(
                f"会话 {session_id!r} 不存在", detail={"session_id": session_id}
            )

    async def _history(self, session_id: str) -> list[tuple[str, str]]:
        turns = self._c.settings.agent.history_max_turns
        limit = max(2, turns * 2)
        messages = await self._c.docstore.list_messages(session_id, limit=limit)
        return [
            (message.role, message.content)
            for message in messages
            if message.content.strip()
        ]


def _parse_citations(raw: object) -> list[Citation]:
    if not isinstance(raw, list):
        return []
    out: list[Citation] = []
    for entry in raw:
        try:
            out.append(Citation.model_validate(entry))
        except Exception:  # noqa: BLE001 - 单条坏数据不该丢掉整份引用
            logger.warning("忽略无法解析的引用条目", extra={"entry": str(entry)[:200]})
    return out
