"""业务服务层。

路由函数只做三件事：解析请求、调用服务、返回响应。所有跨模块的编排
（建库、导入、问答、检索、健康检查）都在这里，原因不是「分层好看」，而是：

1. 服务可以被 CLI 与测试直接调用，不需要起 HTTP 服务；
2. 一次问答要写会话、拼历史、跑 Agent、落消息，这些步骤必须在一个
   可重试、可测试的单元里，而不是散落在路由函数中；
3. 生命周期资源（provider / 后台任务 / 存储句柄）由
   :class:`~app.services.container.AppContainer` 统一持有，
   路由函数不接触构造过程。
"""

from __future__ import annotations

from app.services.container import AppContainer

__all__ = ["AppContainer"]
