"""可量化的多模态检索评估。

这个包只做四件事，边界刻意划得很清楚：

1. **读题**（``dataset``）—— 只认人工标注过的用例，绝不替用户生成 ground truth；
2. **算分**（``metrics``）—— 检索指标是确定性纯函数，可用单测钉死；
   生成质量指标必须调用 LLM judge，拿不到分数就记为 ``None``，不猜；
3. **分组**（``ablation``）—— 五组消融只是「路由 + 重排」的不同组合；
4. **出报告**（``report``）—— Markdown + JSON + 柱状图，附运行参数与降级信息。

``run_eval`` 是 CLI 入口，不承担任何指标定义。
"""

from __future__ import annotations

__all__ = ["ablation", "dataset", "metrics", "report"]
