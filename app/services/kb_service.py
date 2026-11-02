"""知识库生命周期服务。

删除是本模块里唯一复杂的东西。四类存储（SQLite / 向量库 / BM25 / 图片目录）
没有共同事务，所以采用**两阶段删除**：

1. ``status='deleting'`` 先落库 —— 这是「我打算删」的事实记录；
2. 逐个清派生数据（向量 → BM25 → 图片），最后删元数据行。

中途失败时行还在、状态是 ``deleting``，下一次启动
:meth:`KnowledgeBaseService.resume_deletions` 会接着删。
反过来（先删元数据再清向量）失败的话，向量库里会留下一批**没有任何
记录能指向**的数据，既查不到原因，也没有删除入口。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.errors import ConflictError, StorageError, ValidationError
from app.core.logging import get_logger
from app.core.schemas import KnowledgeBase

if TYPE_CHECKING:
    from app.services.container import AppContainer

__all__ = ["KnowledgeBaseService"]

logger = get_logger(__name__)


class KnowledgeBaseService:
    def __init__(self, container: AppContainer) -> None:
        self._c = container

    # ------------------------------------------------------------------ 读

    async def list_kbs(self, *, include_deleting: bool = False) -> list[KnowledgeBase]:
        return await self._c.docstore.list_kbs(include_deleting=include_deleting)

    async def get_kb(self, kb_id: str) -> KnowledgeBase:
        return await self._c.docstore.require_kb(kb_id)

    # ------------------------------------------------------------------ 写

    async def create_kb(self, name: str, description: str = "") -> KnowledgeBase:
        """按名称建库。同名视为冲突，而不是「返回已有的那个」。

        静默返回已有库会让调用方以为自己刚建了一个干净的空库，
        随后往里导入时才发现数据早就存在 —— 幂等应该由调用方显式选择
        （按 ID 取），而不是由服务端猜测。
        """
        name = name.strip()
        if not name:
            raise ValidationError("知识库名称不能为空")
        if len(name) > 64:
            raise ValidationError("知识库名称最长 64 个字符")

        existing = [
            kb
            for kb in await self._c.docstore.list_kbs(include_deleting=True)
            if kb.name == name
        ]
        if existing:
            raise ConflictError(
                f"已存在同名知识库：{name}",
                detail={"name": name, "kb_ids": [kb.kb_id for kb in existing]},
            )
        return await self._c.pipeline.create_kb(name, description=description)

    async def delete_kb(self, kb_id: str) -> dict[str, Any]:
        kb = await self._c.docstore.require_kb(kb_id)
        if kb.status == "deleting":
            # 上次没删完，继续删；不把它当成「正在删所以拒绝」
            logger.info("继续删除处于 deleting 状态的知识库", extra={"kb_id": kb_id})
        else:
            await self._c.docstore.set_kb_status(kb.kb_id, "deleting")

        purged = await self._purge_derived(kb)
        counts = await self._c.docstore.delete_kb(kb.kb_id)
        logger.info(
            "知识库已删除",
            extra={"kb_id": kb_id, "derived": purged, "rows": counts},
        )
        return {"kb_id": kb_id, "derived": purged, "rows": counts}

    async def resume_deletions(self) -> int:
        """启动时续做上次失败的删除。返回成功完成的数量。

        失败只记日志不抛出：一个删不干净的知识库不该阻止整个服务启动，
        它会在 ``/api/health`` 的 ``deleting_kbs`` 里持续可见。
        """
        completed = 0
        for kb in await self._c.docstore.list_kbs(include_deleting=True):
            if kb.status != "deleting":
                continue
            try:
                await self._purge_derived(kb)
                await self._c.docstore.delete_kb(kb.kb_id)
            except Exception:  # noqa: BLE001 - 启动流程不能被单个库拖垮
                logger.exception("续删知识库失败，保留 deleting 状态", extra={"kb_id": kb.kb_id})
                continue
            completed += 1
            logger.info("已续做完成知识库删除", extra={"kb_id": kb.kb_id})
        return completed

    async def _purge_derived(self, kb: KnowledgeBase) -> dict[str, int]:
        """清掉向量 / BM25 / 图片。失败抛 ``StorageError`` 并保留 deleting 标记。"""
        result: dict[str, int] = {}
        try:
            result["text_collection"] = (
                1 if await self._c.vectorstore.delete_collection(kb.text_collection) else 0
            )
            result["image_collection"] = (
                1 if await self._c.vectorstore.delete_collection(kb.image_collection) else 0
            )
            result["bm25"] = 1 if await self._c.bm25store.drop_kb(kb.kb_id) else 0
            result["image_files"] = self._c.blobstore.delete_kb(kb.kb_id)
        except Exception as exc:  # noqa: BLE001 - 统一转成可重试的存储错误
            logger.exception("清理知识库派生数据失败", extra={"kb_id": kb.kb_id})
            raise StorageError(
                "删除知识库失败，已保留 deleting 标记，下次启动会自动续删",
                detail={"kb_id": kb.kb_id, "error": f"{type(exc).__name__}: {exc}"},
            ) from exc
        return result
