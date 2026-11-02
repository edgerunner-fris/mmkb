"""导入服务：把 HTTP 上传变成一次可轮询的后台任务。

为什么必须是后台任务
--------------------
一份带图的 PDF 要走解析 → caption（多次 VLM 调用）→ 两路向量化 → 写索引，
几十秒到几分钟都正常。同步 HTTP 请求会在任何反向代理的默认超时处被掐断，
而客户端断开**不会**停止已经发出去的模型调用 —— 钱照花，结果还丢了。

任务状态全部落在 SQLite，前端轮询 ``GET /api/tasks/{id}``。这样即使前端
刷新页面、换一个标签页，进度仍然在。任务注册表（``_tasks``）只保存
``asyncio.Task`` 的强引用 —— 不保存它，事件循环之外没有别的引用，
任务可能在运行到一半时被 GC 回收。

上传文件落在 ``{upload_dir}/{kb_id}/{task_id}/`` 下，按序号前缀避免同名覆盖。
``INGEST_KEEP_UPLOADED_FILE=false`` 时任务结束后整目录删除。
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from app.core.errors import ConflictError, NotFoundError, UnsupportedFileTypeError, ValidationError
from app.core.logging import get_logger
from app.core.perf import get_metrics
from app.core.schemas import DocStatus, Document, KnowledgeBase, Task, TaskStatus
from app.ingestion.loaders.base import get_registry
from app.ingestion.pipeline import FileOutcome, IngestOptions

if TYPE_CHECKING:
    from app.services.container import AppContainer

__all__ = ["IncomingFile", "IngestService"]

logger = get_logger(__name__)

#: 文件名里允许保留的字符。其余（路径分隔符、引号、控制字符）统一替换成 ``_``。
#: 注意这是**上传文件名**，不参与 ``BlobStore`` 的 kb_id/image_id 路径契约，
#: 后者有更严格的白名单。
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+")
_MAX_NAME_CHARS = 120


class AsyncReadable(Protocol):
    """只依赖 ``read()``，避免服务层 import FastAPI 的 ``UploadFile``。"""

    async def read(self, size: int = -1) -> bytes: ...


@dataclass(slots=True)
class IncomingFile:
    """HTTP 层读完之后交给服务层的一份文件。"""

    filename: str
    data: bytes

    @property
    def safe_name(self) -> str:
        return safe_filename(self.filename)


def safe_filename(raw: str) -> str:
    """把客户端给的文件名压成一个安全的单段文件名。

    浏览器（或恶意客户端）可能送来 ``../../etc/passwd``、``C:\\x\\y.txt``；
    这里先按两种分隔符取最后一段，再做字符白名单替换。**不做「清洗后继续用
    原路径」**，只保留一个不带目录的名字。
    """
    name = re.split(r"[\\/]", raw or "")[-1].strip().strip(".")
    name = _UNSAFE_NAME_RE.sub("_", name)
    if not name:
        name = "upload"
    if len(name) > _MAX_NAME_CHARS:
        suffix = Path(name).suffix[:16]
        name = name[: _MAX_NAME_CHARS - len(suffix)] + suffix
    return name


class _TaskProgress:
    """把流水线的同步进度回调转成定时写库。

    回调必须同步（流水线在阶段切换时直接调用），而写 SQLite 是异步的。
    这里只改内存里的两个字段，再由一个 0.4 秒的心跳协程落库 ——
    既不会在回调里 ``create_task`` 出一堆乱序写，也不会让进度条卡在
    「解析中」直到整个文件结束。
    """

    def __init__(self, docstore: Any, task_id: str) -> None:
        self._docstore = docstore
        self._task_id = task_id
        self.stage = "queued"
        self.message = ""
        self.current = 0
        self._stop = asyncio.Event()
        self._worker = asyncio.create_task(self._loop())

    def callback(self, status: DocStatus, message: str) -> None:
        self.stage = str(status)
        self.message = message
        if status in (
            DocStatus.DONE,
            DocStatus.SKIPPED_DUPLICATE,
            DocStatus.FAILED,
        ):
            self.current += 1

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await self._flush()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=0.4)
            except TimeoutError:
                continue

    async def _flush(self) -> None:
        try:
            await self._docstore.update_task(
                self._task_id,
                stage=self.stage,
                progress_current=self.current,
            )
        except Exception:  # noqa: BLE001 - 进度写失败不该影响导入本身
            logger.debug("写任务进度失败", extra={"task_id": self._task_id}, exc_info=True)

    async def stop(self, *, current: int | None = None) -> None:
        self._stop.set()
        try:
            await self._worker
        except asyncio.CancelledError:  # pragma: no cover - 关停竞态
            pass
        if current is not None:
            self.current = current
        await self._flush()


class IngestService:
    def __init__(self, container: AppContainer) -> None:
        self._c = container
        self._tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------ 上传

    async def read_upload(self, upload: AsyncReadable) -> bytes:
        """分块读取，超过上限立即中止。

        ``len(await upload.read())`` 一次性读入会在一个 5 GB 的恶意请求上
        先把内存吃光，再来谈上限。这里每读 1 MB 检查一次。
        """
        limit = self._c.settings.ingest.max_file_bytes
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValidationError(
                    f"文件超过大小上限：{total / 1024 / 1024:.1f} MB > "
                    f"{limit / 1024 / 1024:.0f} MB",
                    detail={"max_bytes": limit},
                )
            chunks.append(chunk)
        if not chunks:
            raise ValidationError("上传的文件为空")
        return b"".join(chunks)

    def validate_name(self, filename: str) -> str:
        """只看文件名，先在不读字节的情况下拦住不支持的类型。

        上传 200 MB 的 ``.mp4`` 再告诉用户「不支持」是浪费带宽与内存；
        扩展名在读到第一个字节之前就能判断。
        """
        safe = safe_filename(filename)
        path = Path(safe)
        registry = get_registry()
        if not registry.supports(path):
            hint = registry.hint_for(path)
            message = f"不支持的文件类型：{path.suffix or '（无扩展名）'}（{path.name}）"
            if hint:
                message = f"{message}。{hint}"
            raise UnsupportedFileTypeError(
                message,
                detail={
                    "filename": path.name,
                    "supported": registry.supported_suffixes(),
                },
            )
        return safe

    def validate_files(self, files: list[IncomingFile]) -> None:
        """在创建任务之前拦住不支持的类型/空文件/超限文件。

        这类错误是**同步可判定的**，没有理由让用户先拿到一个任务 ID，
        再在轮询里看到失败。
        """
        if not files:
            raise ValidationError("没有收到任何文件")
        limit = self._c.settings.ingest.max_file_bytes
        for item in files:
            name = self.validate_name(item.filename)
            if not item.data:
                raise ValidationError(f"文件为空：{name}")
            if len(item.data) > limit:
                raise ValidationError(
                    f"文件 {name} 超过大小上限 {limit / 1024 / 1024:.0f} MB",
                    detail={"filename": name, "size": len(item.data)},
                )

    # ------------------------------------------------------------------ 任务

    async def submit(
        self,
        kb_id: str,
        files: list[IncomingFile],
        *,
        embed: bool = True,
        caption: bool = True,
        reindex: bool = False,
    ) -> Task:
        kb = await self._c.docstore.require_kb(kb_id)
        if kb.status != "active":
            raise ConflictError(f"知识库正在删除，无法导入：{kb_id}", detail={"kb_id": kb_id})
        self.validate_files(files)

        task = await self._c.docstore.create_task(
            kb.kb_id, kind="ingest", progress_total=len(files)
        )
        options = IngestOptions(embed=embed, caption=caption, reindex=reindex)
        background = asyncio.create_task(
            self._run(task.task_id, kb, files, options),
            name=f"ingest:{task.task_id}",
        )
        self._tasks[task.task_id] = background
        background.add_done_callback(
            lambda finished, task_id=task.task_id: self._forget(task_id, finished)
        )
        logger.info(
            "导入任务已创建",
            extra={
                "task_id": task.task_id,
                "kb_id": kb.kb_id,
                "files": len(files),
                "embed": embed,
                "caption": caption,
            },
        )
        return task

    def _forget(self, task_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(task_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:  # pragma: no cover - _run 内部已兜底
            logger.error(
                "导入任务异常退出",
                extra={"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"},
            )

    async def _run(
        self,
        task_id: str,
        kb: KnowledgeBase,
        files: list[IncomingFile],
        options: IngestOptions,
    ) -> None:
        settings = self._c.settings
        target_dir = settings.upload_dir / kb.kb_id / task_id
        progress: _TaskProgress | None = None
        task_started = time.perf_counter()
        try:
            await self._c.docstore.update_task(
                task_id, status=TaskStatus.RUNNING, stage="saving"
            )
            await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)
            saved = await self._save_files(target_dir, files)

            progress = _TaskProgress(self._c.docstore, task_id)
            outcomes = await self._c.pipeline.ingest_paths(
                saved,
                kb,
                options=options,
                progress=progress.callback,
            )
            summary = _summarize(outcomes)
            await progress.stop(current=len(outcomes))

            failed = int(summary["failed"])
            status = TaskStatus.FAILED if failed == len(outcomes) and outcomes else TaskStatus.DONE
            await self._c.docstore.update_task(
                task_id,
                status=status,
                stage="failed" if status is TaskStatus.FAILED else "done",
                progress_current=len(outcomes),
                result=summary,
                stage_timings=dict(summary["stage_timings"]),
                error=_first_error(outcomes) if status is TaskStatus.FAILED else None,
                finish=True,
            )
            get_metrics().observe(
                "ingest.task.total", (time.perf_counter() - task_started) * 1000
            )
        except asyncio.CancelledError:
            await self._safe_fail(task_id, "服务关闭，导入任务被中断", status=TaskStatus.CANCELLED)
            raise
        except Exception as exc:  # noqa: BLE001 - 后台任务必须自己收尾
            logger.exception("导入任务失败", extra={"task_id": task_id, "kb_id": kb.kb_id})
            await self._safe_fail(task_id, f"{type(exc).__name__}: {exc}")
        finally:
            if progress is not None:
                try:
                    await progress.stop()
                except Exception:  # noqa: BLE001
                    logger.debug("停止进度心跳失败", exc_info=True)
            if not settings.ingest.keep_uploaded_file:
                await asyncio.to_thread(shutil.rmtree, target_dir, True)

    async def _save_files(
        self, target_dir: Path, files: list[IncomingFile]
    ) -> list[Path]:
        saved: list[Path] = []
        for index, item in enumerate(files):
            path = target_dir / f"{index:03d}_{item.safe_name}"
            await asyncio.to_thread(path.write_bytes, item.data)
            saved.append(path)
        return saved

    async def _safe_fail(
        self,
        task_id: str,
        error: str,
        *,
        status: TaskStatus = TaskStatus.FAILED,
    ) -> None:
        try:
            await self._c.docstore.update_task(
                task_id,
                status=status,
                stage=str(status),
                error=error[:2000],
                finish=True,
            )
        except Exception:  # noqa: BLE001 - 数据库都写不了时只能记日志
            logger.error("写任务失败状态失败", extra={"task_id": task_id}, exc_info=True)

    async def shutdown(self) -> None:
        """取消所有在跑的导入任务，并等它们把 CANCELLED 状态写回数据库。"""
        running = [task for task in self._tasks.values() if not task.done()]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        self._tasks.clear()

    @property
    def active_task_count(self) -> int:
        return sum(1 for task in self._tasks.values() if not task.done())

    # ------------------------------------------------------------------ 查询

    async def get_task(self, task_id: str) -> Task:
        task = await self._c.docstore.get_task(task_id)
        if task is None:
            raise NotFoundError(f"任务 {task_id!r} 不存在", detail={"task_id": task_id})
        return task

    async def list_tasks(self, kb_id: str, *, limit: int = 50) -> list[Task]:
        await self._c.docstore.require_kb(kb_id)
        return await self._c.docstore.list_tasks(kb_id, limit=limit)

    async def list_documents(
        self,
        kb_id: str,
        *,
        status: DocStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        await self._c.docstore.require_kb(kb_id)
        return await self._c.docstore.list_documents(
            kb_id, status=status, limit=limit, offset=offset
        )

    async def get_document(self, doc_id: str) -> Document:
        return await self._c.docstore.require_document(doc_id)


# --------------------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------------------


def _summarize(outcomes: list[FileOutcome]) -> dict[str, Any]:
    done = sum(1 for out in outcomes if out.status is DocStatus.DONE)
    skipped = sum(1 for out in outcomes if out.status is DocStatus.SKIPPED_DUPLICATE)
    failed = sum(1 for out in outcomes if not out.ok)
    warnings = _dedupe(w for out in outcomes for w in out.warnings)
    degraded = _dedupe(r for out in outcomes for r in out.degraded_reasons)
    # 阶段耗时按文件求和。注意它可能超过任务总耗时：文件是并发导入的，
    # 求和得到的是「CPU / 外部请求的总工作量」，不是墙钟时间。
    # 报告里因此同时给出任务总耗时，两者的比值就是并发带来的收益。
    stage_timings: dict[str, int] = {}
    for out in outcomes:
        for key, value in out.stage_timings.items():
            stage_timings[key] = stage_timings.get(key, 0) + int(value)
    return {
        "files": len(outcomes),
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "chunks": sum(out.chunks for out in outcomes),
        "images": sum(out.images for out in outcomes),
        "text_vectors": sum(out.text_vectors for out in outcomes),
        "image_vectors": sum(out.image_vectors for out in outcomes),
        "elapsed_ms": sum(out.elapsed_ms for out in outcomes),
        "stage_timings": stage_timings,
        "warnings": warnings[:50],
        "degraded_reasons": degraded,
        "outcomes": [out.to_dict() for out in outcomes],
    }


def _first_error(outcomes: list[FileOutcome]) -> str | None:
    for out in outcomes:
        if out.error:
            return out.error[:2000]
    return None


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
