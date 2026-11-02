"""Agent 可调用的检索工具。

工具层存在的意义不是「包一层」,而是把「意图 → 检索策略」这个映射集中到
一处。如果让 graph 直接调 :class:`HybridRetriever` 并自己拼 routes 列表,
那么新增一个意图时,召回策略会散落在分支里;而工具表让「这个意图查什么」
变成一行可读、可测试的声明。

当前只有两个工具:

* ``search_knowledge`` —— 三路全开,服务知识问答与代码问答;
* ``search_images`` —— 多模态向量 + BM25,服务「找图 / 看图」类问题。

**以图搜图仍然跑 BM25。** 图片的 caption 是文本空间里的分块,用户对
图内文字的精确查询(型号、错误码)由 BM25 接住比向量更稳;只跑多模态
一路会把这类召回丢掉。

工具不做降级:每一路失败已经在 :class:`HybridRetriever` 内被隔离并记入
``SearchOutcome.degraded_reasons``。工具再包一层 try/except 只会把
「某一路失败」升级成「整次检索失败」,方向反了。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.agent.router import Intent
from app.core.schemas import KnowledgeBase, Route
from app.retrieval.base import Stopwatch
from app.retrieval.hybrid import HybridRetriever, SearchOutcome

__all__ = ["SearchTools", "ToolCall", "ToolName", "tool_catalog"]


class ToolName(StrEnum):
    SEARCH_KNOWLEDGE = "search_knowledge"
    SEARCH_IMAGES = "search_images"


@dataclass(slots=True)
class ToolCall:
    """一次工具调用的审计记录，进日志与 debug_trace。"""

    name: ToolName
    query: str
    routes: list[str]
    latency_ms: int
    item_count: int
    rewrite: bool = False


#: 工具说明。给 ``/api/health`` 与前端调试面板用，不参与决策。
_TOOL_CATALOG: dict[ToolName, str] = {
    ToolName.SEARCH_KNOWLEDGE: "三路混合检索，覆盖文档正文、表格、代码与图片图注",
    ToolName.SEARCH_IMAGES: "多模态向量 + BM25 找图，返回文档中的原图",
}


def tool_catalog() -> list[dict[str, str]]:
    return [
        {"name": str(name), "description": description}
        for name, description in _TOOL_CATALOG.items()
    ]


class SearchTools:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    @staticmethod
    def tool_for(intent: Intent) -> tuple[ToolName, list[Route] | None]:
        """意图 → (工具名, 检索路)。``None`` 表示三路全开。"""
        if intent is Intent.IMAGE_LOOKUP:
            return ToolName.SEARCH_IMAGES, [Route.DENSE_IMAGE, Route.BM25]
        if intent in (Intent.KNOWLEDGE, Intent.CODING):
            return ToolName.SEARCH_KNOWLEDGE, None
        # 寒暄不该走到检索。真走到了，说明上游判断有 bug，按知识问答兜底，
        # 而不是抛异常把整个回答打断
        return ToolName.SEARCH_KNOWLEDGE, None

    async def run(
        self,
        intent: Intent,
        kb: KnowledgeBase,
        query: str,
        *,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        rewrite: bool = False,
    ) -> tuple[SearchOutcome, ToolCall]:
        name, routes = self.tool_for(intent)
        started = Stopwatch()
        outcome = await self.retriever.search(
            kb,
            query,
            top_k=top_k,
            routes=routes,
            use_rerank=use_rerank,
            rewrite=rewrite,
        )
        call = ToolCall(
            name=name,
            query=query,
            routes=[str(route) for route in routes] if routes else [],
            latency_ms=started.ms(),
            item_count=len(outcome.items),
            rewrite=bool(outcome.queries and len(outcome.queries) > 1),
        )
        return outcome, call
