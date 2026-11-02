"""健康检查与启动恢复。

``/api/health`` 的契约
----------------------
* **永远返回 200**：它回答的是「进程还活着吗」。即使所有 Key 都没配，
  能返回一份结构化的自检报告，本身就说明 Web 层是通的。
* 报告里的 ``status`` 回答「现在能不能干活」：
  ``ok`` / ``degraded``（可选能力缺失）/ ``not_ready``（必填能力缺失）。
* 需要给编排系统做就绪探针的，用 ``/api/health/ready`` —— 它在
  ``not_ready`` 时返回 503。

启动恢复
--------
进程可能在任意时刻被 kill -9。重启后有两类中间态需要处理：

1. ``tasks`` 里状态仍为 ``queued/running`` 的导入任务 —— 它们的执行者
   已经不存在了，继续显示「解析中」会永远转圈；标记为失败并说明原因。
2. ``knowledge_bases`` 里 ``status='deleting'`` 的库 —— 交给
   :meth:`KnowledgeBaseService.resume_deletions` 续删。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.config import collect_config_problems
from app.core.logging import get_logger
from app.core.schemas import TaskStatus

if TYPE_CHECKING:
    from app.services.container import AppContainer

__all__ = ["HealthService"]

logger = get_logger(__name__)

SERVICE_NAME = "mmkb"
SERVICE_VERSION = "0.1.0"


class HealthService:
    def __init__(self, container: AppContainer) -> None:
        self._c = container

    # ------------------------------------------------------------------ 恢复

    async def recover_interrupted_tasks(self) -> int:
        """把上次进程留下的 queued/running 任务标记为失败。返回处理条数。"""
        recovered = 0
        for kb in await self._c.docstore.list_kbs(include_deleting=True):
            for status in (TaskStatus.RUNNING, TaskStatus.QUEUED):
                tasks = await self._c.docstore.list_tasks(
                    kb.kb_id, limit=200, status=status
                )
                for task in tasks:
                    try:
                        await self._c.docstore.update_task(
                            task.task_id,
                            status=TaskStatus.FAILED,
                            stage="failed",
                            error="服务重启，任务在执行途中被中断（已保留进度，可重新导入）",
                            finish=True,
                        )
                    except Exception:  # noqa: BLE001 - 单条恢复失败不该阻止启动
                        logger.exception(
                            "恢复中断任务失败", extra={"task_id": task.task_id}
                        )
                        continue
                    recovered += 1
        if recovered:
            logger.warning("已将上次进程中断的任务标记为失败", extra={"count": recovered})
        return recovered

    # ------------------------------------------------------------------ 快照

    async def snapshot(self) -> dict[str, Any]:
        container = self._c
        settings = container.settings
        errors, warnings = collect_config_problems(settings)

        database = await self._database_snapshot()
        vectorstore = await self._vectorstore_snapshot()
        kbs = await self._kb_snapshot()
        capabilities = container.capabilities

        if not database.get("ok"):
            status = "not_ready"
        elif errors:
            status = "not_ready"
        elif warnings or not capabilities["dense_image"] or not capabilities["rerank"]:
            status = "degraded"
        else:
            status = "ok"

        started = container.started_at
        uptime = (
            int((datetime.now(UTC) - started).total_seconds()) if started else 0
        )

        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "env": settings.app.env,
            "status": status,
            "ready": status != "not_ready",
            "uptime_s": uptime,
            "started_at": started.isoformat() if started else None,
            "config": {
                "errors": errors,
                "warnings": warnings,
                "path": str(settings.sqlite_path),
            },
            "capabilities": capabilities,
            "providers": container.provider_matrix,
            "provider_errors": container.provider_errors,
            "stores": {
                "database": database,
                "vectorstore": vectorstore,
                "blobstore": container.blobstore.stats(),
                "bm25": container.bm25store.stats(),
            },
            "knowledge_bases": kbs,
            "tasks": {"running": container.ingest.active_task_count},
        }

    async def _database_snapshot(self) -> dict[str, Any]:
        try:
            info = await self._c.database.healthcheck()
            info["tables"] = await self._c.docstore.stats()
            return info
        except Exception as exc:  # noqa: BLE001 - 健康检查本身不能 500
            logger.warning("数据库健康检查失败", extra={"error": str(exc)[:200]})
            return {
                "ok": False,
                "url": self._c.database.safe_url,
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def _vectorstore_snapshot(self) -> dict[str, Any]:
        try:
            collections = await self._c.vectorstore.list_collections()
            return {"ok": True, "collections": collections, "count": len(collections)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("向量库健康检查失败", extra={"error": str(exc)[:200]})
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def _kb_snapshot(self) -> dict[str, Any]:
        try:
            kbs = await self._c.docstore.list_kbs(include_deleting=True)
        except Exception:  # noqa: BLE001
            return {"count": 0, "deleting": [], "error": "无法读取知识库列表"}
        active = [kb for kb in kbs if kb.status == "active"]
        return {
            "count": len(active),
            "deleting": [kb.kb_id for kb in kbs if kb.status == "deleting"],
            "items": [
                {
                    "kb_id": kb.kb_id,
                    "name": kb.name,
                    "doc_count": kb.doc_count,
                    "chunk_count": kb.chunk_count,
                    "image_count": kb.image_count,
                }
                for kb in active
            ],
        }
