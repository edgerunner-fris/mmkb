"""RRF（Reciprocal Rank Fusion）融合。

公式::

    score(d) = Σ_i  w_i · 1 / (k + rank_i(d))

``k`` 默认 60（原论文的经验值）。它压平了「第 1 名与第 2 名」的差距，
使融合结果对某一路的头部抖动不敏感；``w_i`` 是每路的权重，
来自 ``RETRIEVAL_ROUTE_WEIGHTS``。

**为什么用 RRF 而不是把三路分数加权求和**
三路的分数量纲完全不同：BM25 是无上界的正分，余弦相似度在 ``[-1, 1]``，
重排模型又是另一套 logits。直接相加等于让分数范围最大的那一路说了算，
而且换一个 embedding 模型之后权重含义又变了。RRF 只使用**名次**，
对分数尺度免疫 —— 这也是它在工业界混合检索里成为默认做法的原因。

去重键是 ``item_id``，而不是 ``chunk_id``：图片的 caption 分块与图像向量
在 :func:`~app.retrieval.base.canonical_identity` 里已经统一成同一个
``image_id``，所以同一张图被两路同时召回时，这里会**合并成一条**并累加分数，
而不是出现两次。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.schemas import RetrievedItem, Route

__all__ = ["RankedList", "reciprocal_rank_fusion"]

#: RRF 的默认平滑常数。改它会让所有历史评测结果失去可比性。
DEFAULT_RRF_K = 60


@dataclass(slots=True)
class RankedList:
    """一条已排好序的召回列表。

    ``label`` 是 ``route_scores`` / ``route_ranks`` 里的键。单 query 时就是
    ``route.value``；多 query 改写时是 ``"{route}#{n}"``。它必须**全局唯一**，
    否则后一条列表会覆盖前一条在同一条目上留下的分数。
    """

    label: str
    route: Route
    items: list[RetrievedItem] = field(default_factory=list)
    weight: float = 1.0


def _merge_fields(target: RetrievedItem, source: RetrievedItem) -> None:
    """把同一条目在另一路里的信息并进来。

    规则很简单：**谁有值用谁的，都有值时不覆盖先到的**。
    先到的是列表顺序靠前的那一路（由调用方决定顺序），
    它通常是作者最想展示的形态（例如文本分块比图像 metadata 带更多正文）。
    """
    if not target.doc_id and source.doc_id:
        target.doc_id = source.doc_id
    if target.chunk_id is None and source.chunk_id is not None:
        target.chunk_id = source.chunk_id
    if target.image_id is None and source.image_id is not None:
        target.image_id = source.image_id
    if not target.heading_path and source.heading_path:
        target.heading_path = list(source.heading_path)
    if target.page is None and source.page is not None:
        target.page = source.page
    if not target.image_url and source.image_url:
        target.image_url = source.image_url
    if not target.source_name and source.source_name:
        target.source_name = source.source_name
    # 正文取更长的：caption 分块与图像 metadata 里的 caption 通常一致，
    # 但旧数据可能只有一边完整，取长的不会丢信息
    if len(source.text) > len(target.text):
        target.text = source.text
    for key, value in source.metadata.items():
        target.metadata.setdefault(key, value)


def reciprocal_rank_fusion(
    ranked_lists: list[RankedList],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[RetrievedItem]:
    """融合多路召回，返回按 ``rrf_score`` 降序的条目。

    同分时按 ``item_id`` 排序，保证同一输入永远得到同一个顺序 ——
    评测集跑两次顺序不同会让人怀疑实现本身有问题。
    """
    safe_k = max(1, int(k))
    merged: dict[str, RetrievedItem] = {}
    scores: dict[str, float] = {}
    routes: dict[str, set[Route]] = {}

    for ranked in ranked_lists:
        if not ranked.items or ranked.weight <= 0:
            continue
        # 同一条列表里重复出现的条目只算一次：重复通常是上游分页/合并的
        # 缺陷，把它累加两次会让这条结果凭空获得双倍分数
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked.items, start=1):
            key = item.item_id
            if key in seen_in_list:
                continue
            seen_in_list.add(key)
            scores[key] = scores.get(key, 0.0) + ranked.weight / (safe_k + rank)
            routes.setdefault(key, set()).add(ranked.route)

            target = merged.get(key)
            if target is None:
                # 深拷贝：融合后的条目会被重排、回填、扩展邻居，
                # 不能改到调用方手里的原始结果
                target = item.model_copy(deep=True)
                target.route_scores = {}
                target.route_ranks = {}
                merged[key] = target
            else:
                _merge_fields(target, item)

            # 同一 label 在一路列表里只该出现一次；真有重复时保留更好的名次
            previous = target.route_ranks.get(ranked.label)
            if previous is None or rank < previous:
                target.route_ranks[ranked.label] = rank
                target.route_scores[ranked.label] = float(item.score)

    for key, item in merged.items():
        item.rrf_score = scores[key]
        item.score = item.rrf_score
        contributing = routes[key]
        # 只有一路参与时保留原名，让前端知道这条是从哪捡回来的；
        # 多路命中才是真正的 hybrid
        item.route = (
            Route.HYBRID if len(contributing) > 1 else next(iter(contributing))
        )

    fused = sorted(merged.values(), key=lambda entry: (-entry.rrf_score, entry.item_id))
    if top_k is not None and top_k > 0:
        return fused[:top_k]
    return fused
