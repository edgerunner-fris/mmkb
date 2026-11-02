"""Server-Sent Events 序列化。

SSE 的格式本身很简单（``event:`` / ``data:`` / 空行），真正容易出错的是
**心跳**：检索 + 判定可能在第一个 token 之前花掉十几秒，中间的代理
（nginx、Cloudflare、公司网关）通常会在这个空档里把连接判死。心跳是一个
以 ``:`` 开头的注释行，浏览器会忽略它，但连接上的字节流动起来了。

心跳实现有个坑：不能用 ``asyncio.wait_for(anext(iterator), timeout)``。
超时时 ``wait_for`` 会 **取消** 那个 ``__anext__``，而取消会一路抛进
异步生成器内部，把整条流关掉 —— 表现为「每 15 秒断一次」。
正确做法是把 ``anext`` 放进一个独立 Task，超时只放弃等待、不取消它。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress

from app.generation.answer import AnswerEvent

__all__ = ["PING", "encode_event", "with_heartbeat"]

#: SSE 注释行。客户端不可见，只用于保活。
PING = ": ping\n\n"


def encode_event(event: AnswerEvent) -> str:
    """一个 ``AnswerEvent`` → 一个 SSE frame。

    JSON 序列化后是单行（``json.dumps`` 会把换行转义成 ``\\n``），
    所以不需要拆成多条 ``data:``。
    """
    payload = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
    return f"event: {event.type}\ndata: {payload}\n\n"


async def with_heartbeat(
    events: AsyncIterator[AnswerEvent],
    *,
    interval_s: float,
) -> AsyncIterator[str]:
    """把事件流转成 SSE 文本，空闲超过 ``interval_s`` 时发心跳。

    读到、读到 ``done`` / ``error`` 之后**继续读**，让上游生成器有机会
    执行 ``done`` 之后的收尾逻辑（例如把 assistant 消息落库）。上游自然
    结束后再关闭它。
    """
    interval = interval_s if interval_s and interval_s > 0 else 15.0
    iterator = events.__aiter__()
    pending: asyncio.Future[AnswerEvent] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield PING
                continue

            future, pending = pending, None
            try:
                event = future.result()
            except StopAsyncIteration:
                break
            yield encode_event(event)
    finally:
        # 客户端断开时 Starlette 会关闭这个生成器。此时必须收掉那个
        # 仍在等待上游的 Task，否则它会一直挂到上游自己结束。
        if pending is not None and not pending.done():
            pending.cancel()
            with suppress(BaseException):
                await pending
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                await aclose()
