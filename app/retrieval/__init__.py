"""检索层：三路召回 → RRF 融合 → 重排。

对外只有两个入口：

* :class:`~app.retrieval.hybrid.HybridRetriever` —— 完整链路，业务代码用这个
* :class:`~app.retrieval.hybrid.SearchOutcome` —— 一次检索的完整快照
  （结果 + 每路状态 + 降级原因 + 分段耗时），
  既供 API 返回 ``debug_trace``，也是评估模块计算指标时的输入

三路各自的文件与它们遵守的铁律：

===========================  =====================================
文件                          方向
===========================  =====================================
``dense_text.py``            铁律 1：文本 query → ``{kb}__text``
``dense_image.py``           铁律 2：文本 query → 多模态融合空间 → ``{kb}__image``
``sparse.py``                BM25（jieba 分词），无向量空间概念
``fusion.py``                RRF，与后端无关
``rerank.py``                重排，只吃文本（已知取舍，见模块内说明）
===========================  =====================================

**铁律 3（禁止混用两个向量空间）在本层有三道防线**：
provider 的 ``space`` 属性、:func:`~app.retrieval.base.guard_kb_snapshot`
的快照校验、以及 ``VectorStore.query(space=...)`` 的运行时检查。
"""

from __future__ import annotations

from app.retrieval.hybrid import HybridRetriever, SearchOutcome

__all__ = ["HybridRetriever", "SearchOutcome"]
