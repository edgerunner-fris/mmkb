"""API 包：HTTP 边界。

这里只做协议转换（Pydantic 模型）与错误映射；业务编排在
:mod:`app.services`。这样同一套业务逻辑既能被 HTTP 调用，也能被
CLI / 评估脚本 / 测试直接调用。
"""

from __future__ import annotations

__all__: list[str] = []
