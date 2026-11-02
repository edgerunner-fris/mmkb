"""检索层公共结构。

三路召回共享的东西只有四件：路由名、单路结果信封、命中 → ``RetrievedItem``
的转换、以及知识库快照校验。把它们集中在这里，是为了让三个 retriever
（``dense_text`` / ``dense_image`` / ``sparse``）只剩「怎么拿到候选」这一件事，
融合与重排则完全不需要知道后端是 Chroma 还是 pgvector。

**为什么 caption 分块要「改头换面」成图片**
-------------------------------------------
图片在库里有两个身份：

* 文本空间里的一条 ``image_caption`` 分块（chunk_id = ``ck_...``）；
* 图像空间里的一个图像向量（id = ``img_...``）。

如果不在检索层把两者统一成 ``item_id = image_id``，同一张图会以两个条目的身份
同时出现在结果里：RRF 无法给它叠加分数，去重也去不掉，最终挤掉真正不同的候选，
前端还会把同一张图展示两遍。所以 :func:`canonical_identity` 在这里做一次身份归一 ——
**这是「图文互搜」能干净工作的前提**。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.core.errors import RetrievalError
from app.core.logging import get_logger
from app.core.schemas import (
    BlockType,
    DegradedReason,
    ItemType,
    RetrievalFilters,
    RetrievedItem,
    Route,
)
from app.providers.base import EmbeddingProvider
from app.store.bm25store import BM25Hit
from app.store.docstore import DocStore
from app.store.vectorstore import VectorHit

__all__ = [
    "ROUTE_WEIGHT_ORDER",
    "RouteResult",
    "canonical_identity",
    "guard_kb_snapshot",
    "hydrate_items",
    "item_from_bm25_hit",
    "item_from_chunk_hit",
    "item_from_image_hit",
    "parse_json_list",
    "route_weight_map",
]

logger = get_logger(__name__)

#: ``RETRIEVAL_ROUTE_WEIGHTS`` 的固定顺序。改这个顺序等于悄悄改掉用户的权重含义。
ROUTE_WEIGHT_ORDER: tuple[Route, ...] = (
    Route.BM25,
    Route.DENSE_TEXT,
    Route.DENSE_IMAGE,
)

_ITEM_TYPE_BY_BLOCK: dict[str, ItemType] = {
    str(BlockType.TEXT): ItemType.TEXT,
    str(BlockType.CODE): ItemType.CODE,
    str(BlockType.TABLE): ItemType.TABLE,
    str(BlockType.IMAGE_CAPTION): ItemType.IMAGE,
}


# --------------------------------------------------------------------------------------
# 单路结果
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class RouteResult:
    """一路召回的结果信封。

    ``skipped`` / ``error`` / ``degraded_reason`` 三个字段刻意分开：

    * ``skipped`` —— 这一路本来就不该跑（配置关闭、库里没数据、权重为 0），
      **不是故障**；
    * ``error`` —— 这一路跑了但失败了。检索其余几路仍然有效，
      所以整次检索继续，只是响应要带 ``degraded_reasons``；
    * ``degraded_reason`` —— 失败或跳过的**机器可读原因**，来自
      :class:`~app.core.schemas.DegradedReason`，一路带到前端。
    """

    route: Route
    items: list[RetrievedItem] = field(default_factory=list)
    latency_ms: int = 0
    skipped: bool = False
    note: str = ""
    error: str = ""
    degraded_reason: DegradedReason | None = None
    #: 融合时的列表标签。单 query 是 ``route.value``；多 query 是 ``"{route}#{n}"``
    label: str = ""
    #: 该列表在 RRF 里的权重（已乘上改写折扣）
    weight: float = 1.0

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def count(self) -> int:
        return len(self.items)


# --------------------------------------------------------------------------------------
# metadata 解析
# --------------------------------------------------------------------------------------


def parse_json_list(value: Any) -> list[str]:
    """Chroma metadata 里的列表是 JSON 字符串（Chroma 只收标量）。

    解析失败返回 ``[]`` 而不是抛错：metadata 来自旧版本导入或第三方写入时，
    一条坏字段不该让整次检索挂掉。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # 不是合法的 JSON 就当作「没有这个字段」。**不要**退化成
        # ``[原始字符串]``：``image_ids`` 里一个坏值会变成一条不存在的
        # image_id，后续按它回填、按它去重都会指到空气上。
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def canonical_identity(
    metadata: dict[str, Any], fallback_id: str
) -> tuple[str, str | None]:
    """``(item_id, image_id)``。

    ``image_caption`` 分块在检索结果里代表的是**那张图**，不是一段文字，
    所以它的 ``item_id`` 用 ``image_id``。其余分块保持 ``chunk_id``。
    多条 ``image_ids`` 的情况只取第一张：caption 分块在切分阶段就是
    一图一块（见 ``chunker.chunk_document``），这里只是防御性写法。
    """
    block_type = str(metadata.get("block_type") or "")
    image_ids = parse_json_list(metadata.get("image_ids"))
    if block_type == str(BlockType.IMAGE_CAPTION) and image_ids:
        return image_ids[0], image_ids[0]
    return fallback_id, None


def _heading_path(metadata: dict[str, Any]) -> list[str]:
    return parse_json_list(metadata.get("heading_path"))


def _page(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("page_from", metadata.get("page"))
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _route_meta(route: Route, score: float, rank: int) -> tuple[dict[str, float], dict[str, int]]:
    label = route.value
    return {label: float(score)}, {label: int(rank)}


# --------------------------------------------------------------------------------------
# 命中 → RetrievedItem
# --------------------------------------------------------------------------------------


def item_from_chunk_hit(
    *,
    hit_id: str,
    text: str,
    score: float,
    metadata: dict[str, Any],
    kb_id: str,
    route: Route,
    rank: int,
) -> RetrievedItem:
    """文本空间命中（向量库或 BM25，两者的 metadata/payload 同构）。"""
    item_id, image_id = canonical_identity(metadata, hit_id)
    block_type = str(metadata.get("block_type") or str(BlockType.TEXT))
    route_scores, route_ranks = _route_meta(route, score, rank)

    return RetrievedItem(
        item_id=item_id,
        item_type=(
            ItemType.IMAGE if image_id else _ITEM_TYPE_BY_BLOCK.get(block_type, ItemType.TEXT)
        ),
        kb_id=str(metadata.get("kb_id") or kb_id),
        doc_id=str(metadata.get("doc_id") or ""),
        chunk_id=None if image_id else hit_id,
        image_id=image_id,
        text=text,
        score=float(score),
        route=route,
        route_scores=route_scores,
        route_ranks=route_ranks,
        heading_path=_heading_path(metadata),
        page=_page(metadata),
        image_url=str(metadata.get("url") or "") or None,
        metadata=dict(metadata),
    )


def item_from_vector_hit(
    hit: VectorHit, *, kb_id: str, route: Route, rank: int
) -> RetrievedItem:
    return item_from_chunk_hit(
        hit_id=hit.id,
        text=hit.document,
        score=hit.score,
        metadata=dict(hit.metadata),
        kb_id=kb_id,
        route=route,
        rank=rank,
    )


def item_from_bm25_hit(hit: BM25Hit, *, kb_id: str, route: Route, rank: int) -> RetrievedItem:
    return item_from_chunk_hit(
        hit_id=hit.id,
        text=hit.text,
        score=hit.score,
        metadata=dict(hit.payload),
        kb_id=kb_id,
        route=route,
        rank=rank,
    )


def item_from_image_hit(
    hit: VectorHit, *, kb_id: str, route: Route, rank: int
) -> RetrievedItem:
    """图像空间命中。``id`` 就是 ``image_id``，没有 chunk 的概念。"""
    metadata = dict(hit.metadata)
    image_id = str(metadata.get("image_id") or hit.id)
    # 图像 collection 的 document 字段存的是 caption 文本；metadata 里另有一份
    # 同名内容，优先用 metadata：它和入库时 ``build_caption_text`` 的结果逐字一致。
    text = str(metadata.get("caption") or hit.document or "")
    route_scores, route_ranks = _route_meta(route, hit.score, rank)

    return RetrievedItem(
        item_id=image_id,
        item_type=ItemType.IMAGE,
        kb_id=str(metadata.get("kb_id") or kb_id),
        doc_id=str(metadata.get("doc_id") or ""),
        chunk_id=None,
        image_id=image_id,
        text=text,
        score=float(hit.score),
        route=route,
        route_scores=route_scores,
        route_ranks=route_ranks,
        heading_path=_heading_path(metadata),
        page=_page(metadata),
        image_url=str(metadata.get("url") or "") or None,
        metadata=metadata,
    )


# --------------------------------------------------------------------------------------
# 知识库快照
# --------------------------------------------------------------------------------------


def guard_kb_snapshot(kb: Any, provider: EmbeddingProvider, *, space: str) -> None:
    """校验知识库快照与当前 provider 一致。**不一致直接抛错，不降级。**

    与 ``IngestionPipeline._guard_snapshot`` 是同一条规则的两端：写入时拦住，
    读取时再拦一次。只在写入时检查是不够的 —— 配置文件改了、进程重启了，
    旧库还在那里，检索会拿着新模型去查旧向量。两者的余弦相似度算得出来，
    但没有任何意义，而且不会报错。

    这里额外检查 ``provider.space``：文本 provider 去查图像库是铁律 3
    明令禁止的错误，而它的典型调用方式（误用 ``embed_texts`` 编码 query
    再查 ``__image``）在维度恰好相同时完全不会报错。
    """
    if provider.space != space:
        raise RetrievalError(
            f"provider {provider.name}（{provider.model}）属于 {provider.space!r} 空间，"
            f"不能用来检索 {space!r} 空间。文本向量与多模态向量禁止混用。",
            detail={"kb_id": getattr(kb, "kb_id", ""), "space": space},
        )

    if space == "text":
        expected_dim, expected_model, label = kb.text_dim, kb.text_embed_model, "文本向量"
    else:
        expected_dim, expected_model, label = kb.image_dim, kb.image_embed_model, "多模态向量"

    if expected_dim and provider.dim and expected_dim != provider.dim:
        raise RetrievalError(
            f"知识库 {kb.kb_id} 的{label}维度是 {expected_dim}，当前配置是 {provider.dim}。"
            "换维度等于换向量空间，旧向量无法与新 query 比较；"
            "请改回原配置，或新建知识库后重建索引。",
            detail={
                "kb_id": kb.kb_id,
                "space": space,
                "snapshot_dim": expected_dim,
                "configured_dim": provider.dim,
            },
        )
    if expected_model and provider.model and expected_model != provider.model:
        raise RetrievalError(
            f"知识库 {kb.kb_id} 的{label}模型是 {expected_model!r}，当前配置是 {provider.model!r}。"
            "不同模型产出的向量不在同一空间，不能互相检索。",
            detail={
                "kb_id": kb.kb_id,
                "space": space,
                "snapshot_model": expected_model,
                "configured_model": provider.model,
            },
        )


# --------------------------------------------------------------------------------------
# 权重
# --------------------------------------------------------------------------------------


def route_weight_map(settings: Settings) -> dict[Route, float]:
    """``RETRIEVAL_ROUTE_WEIGHTS`` → ``{Route: weight}``。

    顺序固定为 bm25 / dense_text / dense_image。数量不对或出现负数都直接报错：
    这两个错误都会让 RRF 的分数含义变得无法解释，而用户看到的只是「排序有点怪」。
    """
    values = settings.retrieval.route_weight_list
    if len(values) != len(ROUTE_WEIGHT_ORDER):
        raise RetrievalError(
            "RETRIEVAL_ROUTE_WEIGHTS 必须是 3 个数字（顺序：bm25, dense_text, dense_image），"
            f"当前解析出 {len(values)} 个：{values}",
        )
    weights = dict(zip(ROUTE_WEIGHT_ORDER, values, strict=True))
    negative = {route.value: weight for route, weight in weights.items() if weight < 0}
    if negative:
        raise RetrievalError(
            f"RETRIEVAL_ROUTE_WEIGHTS 不能为负数：{negative}。"
            "想关闭某一路请设成 0（或直接用 RETRIEVAL_ENABLE_* 开关）。"
        )
    return weights


# --------------------------------------------------------------------------------------
# 回填
# --------------------------------------------------------------------------------------


async def hydrate_items(items: list[RetrievedItem], docstore: DocStore) -> None:
    """把展示需要、但检索结果里没有的字段补齐。

    缺的两样东西都在 SQLite 里，而检索阶段只拿到了向量库/BM25 的 metadata：

    * ``source_name`` —— 引用里显示的文件名；
    * ``image_url`` —— 图片原图的访问地址（文本空间的 caption 命中没有它）。

    批量取：按 doc_id / image_id 去重后各发一次查询。**失败只记日志**：
    补不齐文件名不影响答案正确性，不该让整次检索失败。
    """
    if not items:
        return

    doc_ids = sorted({item.doc_id for item in items if item.doc_id})
    names: dict[str, str] = {}
    try:
        documents = await asyncio.gather(
            *(docstore.get_document(doc_id) for doc_id in doc_ids),
            return_exceptions=True,
        )
        for document in documents:
            if isinstance(document, BaseException) or document is None:
                continue
            names[document.doc_id] = document.source_name
    except Exception:  # noqa: BLE001 - 回填是装饰性的，失败不该影响检索
        logger.debug("回填文档名失败", exc_info=True)

    image_ids = sorted({item.image_id for item in items if item.image_id})
    urls: dict[str, str] = {}
    if image_ids:
        try:
            images = await docstore.get_images(image_ids)
            urls = {image.image_id: image.url for image in images if image.url}
        except Exception:  # noqa: BLE001
            logger.debug("回填图片 URL 失败", exc_info=True)

    for item in items:
        if not item.source_name and item.doc_id in names:
            item.source_name = names[item.doc_id]
        if item.image_id and not item.image_url:
            item.image_url = urls.get(item.image_id) or None


# --------------------------------------------------------------------------------------
# 计时
# --------------------------------------------------------------------------------------


class Stopwatch:
    """毫秒计时的小工具。检索链路里每一段都要单独报时，方便定位慢在哪一路。"""

    __slots__ = ("_started",)

    def __init__(self) -> None:
        self._started = time.perf_counter()

    def ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)


# 兼容旧调用：RetrievalFilters 在本模块被多处引用，集中再导出一次
__all__ += ["RetrievalFilters"]
