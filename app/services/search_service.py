"""检索服务：给前端调试面板与消融脚本用的原始检索入口。

它故意**不经过 Agent**：不路由、不判定、不生成。前端「检索调试」页签
要回答的是「三路各捞到了什么、RRF 怎么排的、重排改了哪些名次」，
套一层问答只会把中间结果藏起来。
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from app.core.errors import ProviderResponseError, ValidationError
from app.core.schemas import RetrievalFilters, Route
from app.providers.base import prepare_image_for_api
from app.retrieval.base import guard_kb_snapshot, hydrate_items, item_from_image_hit

if TYPE_CHECKING:
    from app.services.container import AppContainer

__all__ = ["SearchService"]

#: 与 :class:`app.core.schemas.Route` 的可用值保持一致（不含 ``hybrid``：
#: 它是融合结果的标记，指定它作为检索路没有意义）。
ALLOWED_ROUTES = ("bm25", "dense_text", "dense_image")


class SearchService:
    def __init__(self, container: AppContainer) -> None:
        self._c = container

    async def search(
        self,
        kb_id: str,
        query: str,
        *,
        routes: list[str] | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        rewrite: bool = False,
        filters: RetrievalFilters | None = None,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValidationError("检索词不能为空")
        if routes is not None:
            invalid = [route for route in routes if route not in ALLOWED_ROUTES]
            if invalid:
                raise ValidationError(
                    f"未知的检索路：{invalid}",
                    detail={"allowed": list(ALLOWED_ROUTES)},
                )

        kb = await self._c.docstore.require_kb(kb_id)
        outcome = await self._c.retriever.search(
            kb,
            query,
            top_k=top_k,
            filters=filters,
            routes=routes,
            use_rerank=use_rerank,
            rewrite=rewrite,
        )
        return outcome.to_dict()

    async def image_search(
        self,
        kb_id: str,
        data: bytes,
        *,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """以图搜图：图片 → 多模态向量 → ``{kb}__image``。

        这条路径**不经过文本检索器**，因为没有文本 query。它仍然必须：
        * 用 ``space="image"`` 的 provider（铁律 3）；
        * 先做知识库快照校验（模型/维度变化后旧向量不可比）；
        * 不做重排 —— reranker 只收文本，这里没有可拼的 query。
        """
        if not data:
            raise ValidationError("图片内容为空")
        provider = self._c.require_image_embedding()
        kb = await self._c.docstore.require_kb(kb_id)
        guard_kb_snapshot(kb, provider, space="image")

        if await self._c.vectorstore.count(kb.image_collection) <= 0:
            return {
                "query": "image",
                "queries": [],
                "count": 0,
                "items": [],
                "routes": [
                    {
                        "route": str(Route.DENSE_IMAGE),
                        "count": 0,
                        "latency_ms": 0,
                        "skipped": True,
                        "note": f"{kb.image_collection} 为空：知识库还没有图片向量",
                        "error": None,
                        "degraded_reason": None,
                    }
                ],
                "degraded_reasons": [],
                "timings": {"total": 0},
                "reranked": False,
                "fused_count": 0,
            }

        started = time.perf_counter()
        try:
            # Pillow 解码/缩放是 CPU 操作，不要占住事件循环
            prepared = await asyncio.to_thread(prepare_image_for_api, data)
        except ValueError as exc:
            raise ValidationError(f"无法解析图片：{exc}") from exc

        embedding = await provider.embed_images([prepared])
        if not embedding.vectors:
            raise ProviderResponseError(
                "多模态 embedding 返回空向量，无法执行以图搜图",
                detail={"provider": provider.name, "model": provider.model},
            )

        hits = await self._c.vectorstore.query(
            kb.image_collection,
            embedding.vectors[0],
            top_k=top_k,
            space="image",
        )
        items = [
            item_from_image_hit(
                hit, kb_id=kb.kb_id, route=Route.DENSE_IMAGE, rank=rank
            )
            for rank, hit in enumerate(hits, start=1)
        ]
        await hydrate_items(items, self._c.docstore)
        latency_ms = int((time.perf_counter() - started) * 1000)

        return {
            "query": "image",
            "queries": [],
            "count": len(items),
            "items": [item.model_dump(mode="json") for item in items],
            "routes": [
                {
                    "route": str(Route.DENSE_IMAGE),
                    "count": len(items),
                    "latency_ms": latency_ms,
                    "skipped": False,
                    "note": "以图搜图：没有文本 query，重排不参与",
                    "error": None,
                    "degraded_reason": None,
                }
            ],
            "degraded_reasons": [],
            "timings": {"total": latency_ms},
            "reranked": False,
            "fused_count": len(items),
        }
