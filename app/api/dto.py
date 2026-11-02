"""HTTP 请求模型。

它们与 :mod:`app.core.schemas` 里的领域模型刻意分开：

* 领域模型描述「系统里有什么」，字段随存储/检索演进而变；
* 请求模型描述「客户端能传什么」，多了长度上限与枚举约束，
  少了所有服务端派生字段（分数、时间、降级原因）。

混用会让「客户端能传 score」这种漏洞悄悄出现在 OpenAPI 文档里。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.schemas import RetrievalFilters

__all__ = [
    "ChatRequest",
    "CreateKBRequest",
    "RouteName",
    "SearchRequest",
]

#: 可作为检索路指定的路由。``hybrid`` 是融合结果标记，不是一路召回。
RouteName = Literal["bm25", "dense_text", "dense_image"]


class CreateKBRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=500)


class ChatRequest(BaseModel):
    kb_id: str = Field(min_length=1, description="知识库 ID")
    question: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(
        default=None,
        description="留空则新建会话；同一个 session_id 会带上之前的历史",
    )
    top_k: int | None = Field(default=None, ge=1, le=100)
    use_rerank: bool | None = Field(default=None, description="None 表示按配置")
    rewrite: bool | None = Field(default=None, description="None 表示按配置")
    debug: bool = Field(default=False, description="是否在响应里带每轮检索的调试轨迹")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    routes: list[RouteName] | None = Field(
        default=None, description="留空表示按配置跑全部启用的路"
    )
    top_k: int | None = Field(default=None, ge=1, le=100)
    use_rerank: bool | None = None
    rewrite: bool = False
    filters: RetrievalFilters | None = None
