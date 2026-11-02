"""摄入流水线：文件 → 解析 → 图片处理 → 切分 → 向量与索引 → SQLite。

这一层只做「编排」，每一步的实现都在各自的模块里。它负责的是那些
**只有把它们串起来才会出现**的问题：

状态机
------
``QUEUED → PARSING → CAPTIONING → EMBEDDING → INDEXING → DONE``

每一步都落一次 SQLite。前端轮询文档状态时看到的是真实的阶段，
而不是一个从头到尾都是 ``parsing`` 的黑盒；失败时 ``error`` 字段里
是那一步的真实异常，而不是「导入失败」四个字。

幂等
----
去重按 ``(kb_id, content_sha256)``，不是文件名 —— 改名不改内容仍是同一份文档。
``DONE`` 直接跳过；``FAILED`` 或任何中途状态则**重试**，复用同一个
``doc_id``（``chunk_id`` 由它派生，重试因此不会产生孤儿向量）。
``--reindex`` 强制重跑 ``DONE`` 的文档。

提交顺序
--------
1. SQLite：分块与图片记录（``replace_document_chunks`` 先删后写）
2. 向量库：按 ``doc_id`` 删除旧向量，再 upsert 新向量
3. BM25：按 ``doc_id`` 删除旧文档，再加入新文档

SQLite 是事实源，所以先写它。向量与 BM25 都是**可以从 SQLite 重建**的派生索引，
顺序反过来会让「库里有分块、索引里没有」这种状态活得更久。

失败处理
--------
任何一步抛异常，文档标记 ``FAILED``，并尽力清掉本次写入的派生物
（向量、BM25）。分块与图片记录**保留**：它们是下一次重试的
``replace_document_chunks`` 要覆盖的旧数据，提前删掉只会让失败文档
在 UI 里变成「从未导入过」，丢失排障线索。

离线模式
--------
``IngestOptions.embed=False`` / ``caption=False`` 时完全不构造对应的 provider。
没有 API Key 的机器仍然可以完整跑通解析、过滤、切分、SQLite 与 BM25 ——
这也是本项目的验证命令能在离线环境执行的原因。缺 Key 属于**预期内降级**，
记进 ``degraded_reasons``；而「Key 配了、请求失败」属于事故，整份文档失败。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.config import ConfigurationError, Settings, get_settings
from app.core.errors import IngestError, MMKBError, UnsupportedFileTypeError
from app.core.ids import image_collection, is_valid_id, new_id, text_collection
from app.core.logging import get_logger
from app.core.perf import get_metrics
from app.core.schemas import (
    Chunk,
    DegradedReason,
    DocStatus,
    Document,
    ImageAsset,
    KnowledgeBase,
)
from app.ingestion.chunker import ChunkOutcome, chunk_document
from app.ingestion.image_processor import ImageProcessor, ImageProcessOutcome, ProcessedImage
from app.ingestion.loaders.base import (
    detect_mime,
    file_sha256,
    get_registry,
)
from app.ingestion.parser import ParsedDocument, parse_document
from app.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    PreparedImage,
    prepare_image_for_api,
)
from app.store.blobstore import BlobStore
from app.store.bm25store import BM25Doc, BM25Store
from app.store.docstore import DocStore
from app.store.vectorstore import VectorRecord, VectorStore

__all__ = [
    "FileOutcome",
    "IngestOptions",
    "IngestionPipeline",
    "ProviderSet",
    "collect_files",
]

logger = get_logger(__name__)

#: 进度回调：``(阶段, 给用户看的描述)``。回调里抛异常不会影响导入
#: （见 :meth:`IngestionPipeline._notify`）—— 一个画进度条的函数不该
#: 有能力让整批导入失败。
ProgressCallback = Callable[[DocStatus, str], None]


# --------------------------------------------------------------------------------------
# 选项与结果
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class IngestOptions:
    """一次导入的开关。

    ``embed=False`` 时连 provider 都不构造：没有 Key 的机器不该因为
    「想看看解析结果」而先被一句 ConfigurationError 拦住。
    """

    embed: bool = True
    caption: bool = True
    reindex: bool = False
    dry_run: bool = False


@dataclass(slots=True)
class FileOutcome:
    """单个文件的导入结果。CLI 的汇总表与 JSON 输出都直接用它。"""

    path: Path
    status: DocStatus = DocStatus.QUEUED
    doc_id: str = ""
    title: str = ""

    chunks: int = 0
    tokens: int = 0
    images: int = 0
    images_filtered: int = 0
    images_failed: int = 0
    text_vectors: int = 0
    image_vectors: int = 0
    page_count: int | None = None

    dry_run: bool = False
    elapsed_ms: int = 0
    #: 各阶段耗时（毫秒）。键：hash / parse / caption / chunk / embed / commit。
    #: 导入慢了到底是慢在解析 PDF、图片 caption、外部 embedding，还是写库，
    #: 这张表直接回答，而不是靠看日志猜。
    stage_timings: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    degraded_reasons: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        """能否算「这次导入成功了」。

        ``SKIPPED_DUPLICATE`` 算成功：重复导入是幂等语义，不是错误。
        """
        return self.status in (DocStatus.DONE, DocStatus.SKIPPED_DUPLICATE)

    @property
    def is_duplicate(self) -> bool:
        return self.status is DocStatus.SKIPPED_DUPLICATE

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "status": str(self.status),
            "doc_id": self.doc_id,
            "title": self.title,
            "chunks": self.chunks,
            "tokens": self.tokens,
            "images": self.images,
            "images_filtered": self.images_filtered,
            "images_failed": self.images_failed,
            "text_vectors": self.text_vectors,
            "image_vectors": self.image_vectors,
            "page_count": self.page_count,
            "dry_run": self.dry_run,
            "elapsed_ms": self.elapsed_ms,
            "stage_timings": dict(self.stage_timings),
            "warnings": list(self.warnings),
            "degraded_reasons": list(self.degraded_reasons),
            "error": self.error,
        }


# --------------------------------------------------------------------------------------
# Provider 组合
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class ProviderSet:
    """按需构造的三个 provider。

    刻意不用 :func:`~app.providers.factory.get_providers`：那个函数要求
    **四个 Key 全配齐**，而导入一份纯文本只需要文本 embedding。用全量 bundle
    会让「只想离线跑 BM25」的用户先被 LLM Key 拦住。
    """

    llm: LLMProvider | None = None
    text_embedding: EmbeddingProvider | None = None
    image_embedding: EmbeddingProvider | None = None
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# 路径收集
# --------------------------------------------------------------------------------------

#: 目录递归时直接忽略的文件。Office 的 ``~$`` 锁文件是二进制残片，
#: 交给 loader 只会得到一条毫无意义的解析错误。
_SKIP_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def collect_files(
    paths: Iterable[str | Path], *, recursive: bool = True
) -> tuple[list[Path], list[Path]]:
    """展开目录，返回 ``(可导入文件, 被跳过的文件)``。

    被跳过的文件不是错误 —— 一个资料目录里混着 .zip、.mp4、临时文件是常态。
    但调用方**应该**把数量报出来，否则用户会以为「我明明放进去了」。
    """
    registry = get_registry()
    accepted: list[Path] = []
    skipped: list[Path] = []
    seen: set[str] = set()

    for raw in paths:
        path = Path(raw).expanduser()
        candidates = _walk(path) if path.is_dir() and recursive else [path]
        if path.is_dir() and not recursive:
            candidates = sorted(child for child in path.iterdir() if child.is_file())
        for candidate in candidates:
            if not candidate.is_file():
                # 不存在的路径原样交给 ingest_file，由它给出带路径的明确错误
                if not path.is_dir():
                    accepted.append(candidate)
                continue
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            if candidate.name in _SKIP_NAMES or candidate.name.startswith("~$"):
                skipped.append(candidate)
            elif registry.supports(candidate):
                accepted.append(candidate)
            else:
                skipped.append(candidate)

    accepted.sort(key=lambda item: str(item).lower())
    return accepted, skipped


def _walk(directory: Path) -> list[Path]:
    out: list[Path] = []
    try:
        for child in sorted(directory.rglob("*")):
            # 隐藏目录（.git、.venv、__pycache__）里通常没有用户想导入的内容
            if any(part.startswith(".") for part in child.relative_to(directory).parts):
                continue
            if child.is_file():
                out.append(child)
    except OSError as exc:  # pragma: no cover - 权限问题属于部署环境
        raise IngestError(
            f"遍历目录失败：{directory}：{exc}", detail={"path": str(directory)}
        ) from exc
    return out


# --------------------------------------------------------------------------------------
# 流水线
# --------------------------------------------------------------------------------------


class IngestionPipeline:
    """把 loader / parser / chunker / image_processor 串成一次完整导入。

    依赖全部可注入：测试里传内存中的替代实现，生产里走各模块的工厂单例。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        docstore: DocStore | None = None,
        vectorstore: VectorStore | None = None,
        bm25store: BM25Store | None = None,
        blobstore: BlobStore | None = None,
        providers: ProviderSet | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.docstore = docstore or DocStore(self._settings)
        if vectorstore is None:
            from app.store.factory import get_vectorstore

            vectorstore = get_vectorstore()
        self.vectorstore = vectorstore
        if bm25store is None:
            from app.store.factory import get_bm25store

            bm25store = get_bm25store()
        self.bm25store = bm25store
        if blobstore is None:
            from app.store.factory import get_blobstore

            blobstore = get_blobstore()
        self.blobstore = blobstore

        self._providers = providers
        self._owns_providers = providers is None
        # 不同的 ``IngestOptions`` 需要不同的 provider 组合（--no-embed 不该
        # 顺手把 LLM 也构造出来）。按组合缓存，而不是缓存第一个碰到的组合。
        self._provider_cache: dict[tuple[bool, bool], ProviderSet] = {}

    # -- 生命周期 -------------------------------------------------------------------

    async def aclose(self) -> None:
        """只关自己构造的 provider。

        注入进来的对象归调用方管；把它一起关掉会让「同一个 provider 被
        两条流水线共用」的场景在第一条关闭后神秘失败。
        """
        if not self._owns_providers:
            return
        seen: set[int] = set()
        for providers in self._provider_cache.values():
            for provider in (
                providers.llm,
                providers.text_embedding,
                providers.image_embedding,
            ):
                if provider is None or id(provider) in seen:
                    continue
                seen.add(id(provider))
                try:
                    await provider.aclose()
                except Exception:  # noqa: BLE001 - 关闭失败不影响其余 provider
                    logger.warning("关闭 provider 失败", exc_info=True)

    # -- 知识库 ---------------------------------------------------------------------

    async def ensure_kb(self, name_or_id: str, *, description: str = "") -> KnowledgeBase:
        """按名称或 ID 取知识库，不存在就建。

        传入合法 ``kb_`` ID 时**必须**已存在：把 ID 当名称去新建一个同名库，
        会让「ID 打错一个字符」变成「悄悄多了一个空库」，比直接报错难查得多。
        """
        if is_valid_id(name_or_id, "kb"):
            kb = await self.docstore.get_kb(name_or_id)
            if kb is None:
                raise IngestError(
                    f"知识库 {name_or_id!r} 不存在",
                    detail={"kb_id": name_or_id},
                )
            await self._ensure_collections(kb)
            return kb

        for existing in await self.docstore.list_kbs():
            if existing.name == name_or_id:
                await self._ensure_collections(existing)
                return existing

        return await self.create_kb(name_or_id, description=description)

    async def create_kb(self, name: str, *, description: str = "") -> KnowledgeBase:
        """无条件新建一个知识库（不做同名检查）。

        单独抽出来给 API 的「新建知识库」用：它必须先拒绝同名请求，
        而不是像 :meth:`ensure_kb` 那样把同名的现有库直接返回。
        建库快照里的维度取**配置**而不是 provider 的实际返回值 ——
        快照必须能在离线环境复现，而 provider 在没有 Key 时根本不存在。
        """
        settings = self._settings
        kb_id = new_id("kb")
        kb = await self.docstore.create_kb(
            name,
            description=description,
            kb_id=kb_id,
            text_collection=text_collection(kb_id),
            image_collection=image_collection(kb_id),
            text_dim=settings.text_embed.dim,
            image_dim=settings.image_embed.dim,
            llm_model=settings.llm.model,
            text_embed_model=settings.text_embed.resolved_model,
            image_embed_model=settings.image_embed.model,
            rerank_model=settings.rerank.resolved_model,
            image_enable_fusion=settings.image_embed.enable_fusion,
        )
        logger.info("已创建知识库", extra={"kb_id": kb.kb_id, "name": kb.name})
        await self._ensure_collections(kb)
        return kb

    async def _ensure_collections(self, kb: KnowledgeBase) -> None:
        """幂等地建好两个 collection。

        为什么连已有的库也要调：Chroma 的数据目录可能被整个删掉，
        而 SQLite 里的 ``knowledge_bases`` 记录还在。此时不补建，
        第一次写入会得到一句来自 chromadb 内部的 ``Collection not found``。
        """
        await self.vectorstore.ensure_collection(
            kb.text_collection,
            dim=kb.text_dim,
            space="text",
            embed_model=kb.text_embed_model,
        )
        await self.vectorstore.ensure_collection(
            kb.image_collection,
            dim=kb.image_dim,
            space="image",
            embed_model=kb.image_embed_model,
        )

    # -- 批量入口 -------------------------------------------------------------------

    async def ingest_paths(
        self,
        paths: Sequence[str | Path],
        kb: KnowledgeBase | str,
        *,
        options: IngestOptions | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[FileOutcome]:
        """展开路径并逐个导入。目录会被递归扫描。"""
        files, skipped = collect_files(paths)
        for path in skipped:
            logger.info("跳过不支持的文件", extra={"path": str(path)})
        if not files:
            return []

        target = kb if isinstance(kb, KnowledgeBase) else await self.ensure_kb(kb)
        options = options or IngestOptions()
        limit = max(1, self._settings.ingest.max_concurrent_docs)
        semaphore = asyncio.Semaphore(limit)

        async def run(path: Path) -> FileOutcome:
            async with semaphore:
                return await self.ingest_file(
                    path,
                    target,
                    options=options,
                    progress=progress,
                    # 批量场景下每个文件都 refresh 一次是纯浪费：
                    # refresh_kb_stats 要跑三条 COUNT 并 UPDATE 知识库行，
                    # 一次导入 50 个文件就是 50 次。整批结束再算一次即可。
                    refresh_stats=False,
                )

        outcomes = list(await asyncio.gather(*(run(path) for path in files)))
        try:
            await self.docstore.refresh_kb_stats(target.kb_id)
        except Exception:  # noqa: BLE001 - 统计失败不该让导入结果变成失败
            logger.warning("批量导入后刷新知识库统计失败", exc_info=True)
        return outcomes

    # -- 单个文件 -------------------------------------------------------------------

    async def ingest_file(
        self,
        path: str | Path,
        kb: KnowledgeBase | str,
        *,
        options: IngestOptions | None = None,
        progress: ProgressCallback | None = None,
        refresh_stats: bool = True,
    ) -> FileOutcome:
        """导入一个文件。**不抛异常** —— 失败写进返回值的 ``error``。

        这样批量导入不会因为第三份 PDF 损坏而丢掉前两份的成功结果。
        调用方用 ``outcome.ok`` 判断，用 ``outcome.error`` 排障。

        ``refresh_stats=False`` 供批量入口使用：整批结束只刷新一次知识库统计。
        """
        started = time.perf_counter()
        # 只有路径里带 ``~`` 时才会读一次环境变量，开销远小于一次线程切换
        file_path = Path(path).expanduser()  # noqa: ASYNC240
        options = options or IngestOptions()
        outcome = FileOutcome(path=file_path, dry_run=options.dry_run)
        timings = outcome.stage_timings
        doc_id = ""
        collection_kb: KnowledgeBase | None = None

        def mark(stage: str, stage_started: float) -> None:
            """记录一个阶段耗时。同名阶段累加（重试 / 多次调用也只占一个键）。"""
            elapsed = (time.perf_counter() - stage_started) * 1000
            timings[stage] = timings.get(stage, 0) + int(elapsed)
            get_metrics().observe(f"ingest.stage.{stage}", elapsed)

        try:
            target = kb if isinstance(kb, KnowledgeBase) else await self.ensure_kb(kb)
            collection_kb = target
            self._check_file(file_path)

            hash_started = time.perf_counter()
            digest = await asyncio.to_thread(file_sha256, file_path)
            existing = await self.docstore.find_document_by_hash(target.kb_id, digest)
            mark("hash", hash_started)

            if (
                existing is not None
                and existing.status == DocStatus.DONE
                and not options.reindex
            ):
                outcome.status = DocStatus.SKIPPED_DUPLICATE
                outcome.doc_id = existing.doc_id
                outcome.title = existing.title or file_path.stem
                outcome.chunks = existing.chunk_count
                outcome.images = existing.image_count
                try:
                    outcome.tokens = int(existing.extra.get("tokens") or 0)
                except (TypeError, ValueError):
                    outcome.tokens = 0
                # 不写进 warnings：重复导入是**幂等语义**，不是异常。
                # 把它塞进 warnings 会让 --strict 在完全正常的重跑上失败。
                return self._finish(outcome, started)

            doc_id = existing.doc_id if existing is not None else new_id("doc")
            outcome.doc_id = doc_id

            if options.dry_run:
                parse_started = time.perf_counter()
                parsed = await self._parse(file_path, target, doc_id)
                mark("parse", parse_started)
                chunk_started = time.perf_counter()
                chunked = chunk_document(parsed, settings=self._settings.chunking)
                mark("chunk", chunk_started)
                outcome.title = parsed.title
                outcome.page_count = parsed.page_count
                outcome.chunks = len(chunked.chunks)
                outcome.tokens = sum(chunk.token_count for chunk in chunked.chunks)
                outcome.images = len(parsed.images)
                outcome.warnings.extend(parsed.warnings)
                outcome.warnings.extend(chunked.warnings)
                outcome.warnings.append("dry-run：未写入任何数据，图片未过滤")
                return self._finish(outcome, started)

            document = Document(
                doc_id=doc_id,
                kb_id=target.kb_id,
                source_path=str(file_path),
                source_name=file_path.name,
                mime_type=detect_mime(file_path),
                doc_hash=digest,
                size_bytes=file_path.stat().st_size,
                status=DocStatus.QUEUED,
            )
            saved, _created = await self.docstore.upsert_document(document)
            # upsert 命中既有记录时返回的是旧对象，doc_id 以它为准
            doc_id = saved.doc_id
            outcome.doc_id = doc_id

            providers = self._build_providers(options)
            await self._set_status(doc_id, DocStatus.PARSING, progress, file_path)

            parse_started = time.perf_counter()
            parsed = await self._parse(file_path, target, doc_id)
            mark("parse", parse_started)
            outcome.title = parsed.title
            outcome.page_count = parsed.page_count
            outcome.warnings.extend(parsed.warnings)

            caption_enabled = (
                options.caption
                and self._settings.vision.enabled
                and providers.llm is not None
            )
            if options.caption and providers.llm is None:
                outcome.degraded_reasons.append(DegradedReason.VISION_UNAVAILABLE)
                outcome.warnings.append(
                    "对话模型不可用，图片使用上下文占位 caption（图片仍可被检索）"
                )

            image_outcome = ImageProcessOutcome()
            if parsed.images:
                await self._set_status(
                    doc_id, DocStatus.CAPTIONING, progress, file_path
                )
                caption_started = time.perf_counter()
                image_outcome = await self._process_images(
                    parsed,
                    target,
                    providers,
                    caption_enabled=caption_enabled,
                )
                mark("caption", caption_started)
                outcome.warnings.extend(image_outcome.warnings)

            outcome.images = len(image_outcome.kept)
            outcome.images_filtered = len(image_outcome.filtered)
            outcome.images_failed = len(image_outcome.failed)

            chunk_started = time.perf_counter()
            chunked = chunk_document(
                parsed,
                settings=self._settings.chunking,
                keep_image_ids={item.asset.image_id for item in image_outcome.kept},
                caption_images=[item.asset for item in image_outcome.kept],
            )
            mark("chunk", chunk_started)
            outcome.chunks = len(chunked.chunks)
            outcome.tokens = sum(chunk.token_count for chunk in chunked.chunks)
            outcome.warnings.extend(chunked.warnings)
            _bind_images_to_chunks(image_outcome.kept, chunked)

            text_records: list[VectorRecord] = []
            image_records: list[VectorRecord] = []
            text_active = False
            image_active = False
            if options.embed:
                await self._set_status(
                    doc_id, DocStatus.EMBEDDING, progress, file_path
                )
                embed_started = time.perf_counter()
                (
                    text_records,
                    image_records,
                    text_active,
                    image_active,
                ) = await self._embed(
                    target,
                    providers,
                    chunked.chunks,
                    image_outcome.kept,
                    outcome,
                )
                mark("embed", embed_started)
            else:
                outcome.warnings.append(
                    "--no-embed：只写 SQLite 与 BM25，向量检索对本文档不可用"
                )

            await self._set_status(doc_id, DocStatus.INDEXING, progress, file_path)
            commit_started = time.perf_counter()
            await self._commit(
                target,
                doc_id,
                chunked.chunks,
                image_outcome,
                text_records,
                image_records,
                text_active=text_active,
                image_active=image_active,
            )
            mark("commit", commit_started)

            await self.docstore.update_document_status(
                doc_id,
                DocStatus.DONE,
                error=None,
                chunk_count=len(chunked.chunks),
                image_count=outcome.images,
                skipped_image_count=outcome.images_filtered + outcome.images_failed,
                title=parsed.title,
                page_count=parsed.page_count,
                extra={
                    "warnings": outcome.warnings[:20],
                    "degraded_reasons": _dedupe(outcome.degraded_reasons),
                    "text_vectors": len(text_records),
                    "image_vectors": len(image_records),
                    "tokens": outcome.tokens,
                    "page_count": parsed.page_count,
                    "source_name": file_path.name,
                },
            )
            outcome.status = DocStatus.DONE
            outcome.text_vectors = len(text_records)
            outcome.image_vectors = len(image_records)
            if refresh_stats:
                stats_started = time.perf_counter()
                await self.docstore.refresh_kb_stats(target.kb_id)
                mark("stats", stats_started)
            logger.info(
                "文档导入完成",
                extra={
                    "doc_id": doc_id,
                    "chunks": len(chunked.chunks),
                    "images": outcome.images,
                    "text_vectors": len(text_records),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            return self._finish(outcome, started)

        except MMKBError as exc:
            outcome.status = DocStatus.FAILED
            outcome.error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "文档导入失败",
                extra={"path": str(file_path), "doc_id": doc_id, "error": outcome.error},
            )
        except Exception as exc:  # noqa: BLE001 - 批量导入不能因单文件而中断
            outcome.status = DocStatus.FAILED
            outcome.error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "文档导入出现未预期异常",
                extra={"path": str(file_path), "doc_id": doc_id},
            )

        if collection_kb is not None and doc_id:
            await self._mark_failed(collection_kb, doc_id, outcome.error or "未知错误")
        return self._finish(outcome, started)

    # -- 各阶段 ---------------------------------------------------------------------

    def _check_file(self, path: Path) -> None:
        if not path.exists():
            raise IngestError(f"文件不存在：{path}", detail={"path": str(path)})
        if not path.is_file():
            raise IngestError(f"不是普通文件：{path}", detail={"path": str(path)})

        size = path.stat().st_size
        limit = self._settings.ingest.max_file_bytes
        if limit and size > limit:
            raise IngestError(
                f"文件超过大小上限：{size / 1024 / 1024:.1f} MB > "
                f"{limit / 1024 / 1024:.0f} MB",
                detail={"path": str(path), "size": size, "max": limit},
            )

        registry = get_registry()
        if registry.supports(path):
            return
        hint = registry.hint_for(path)
        message = f"不支持的文件类型：{path.suffix or '（无扩展名）'}（{path.name}）"
        if hint:
            message = f"{message}。{hint}"
        raise UnsupportedFileTypeError(
            message,
            detail={"path": str(path), "supported": registry.supported_suffixes()},
        )

    async def _parse(
        self, path: Path, kb: KnowledgeBase, doc_id: str
    ) -> ParsedDocument:
        """解析在**线程**里做：PyMuPDF / openpyxl 都是同步且 CPU 密集的。"""
        loader = get_registry().loader_for(path)
        result = await asyncio.to_thread(loader.load, path)
        return parse_document(
            result,
            doc_id=doc_id,
            kb_id=kb.kb_id,
            fallback_title=path.stem,
        )

    async def _process_images(
        self,
        parsed: ParsedDocument,
        kb: KnowledgeBase,
        providers: ProviderSet,
        *,
        caption_enabled: bool,
    ) -> ImageProcessOutcome:
        existing_phashes: dict[str, str] | None = None
        if self._settings.blob.dedupe_scope == "kb":
            mapping = await self.docstore.find_images_by_phash(
                kb.kb_id, exclude_doc_id=parsed.doc_id
            )
            existing_phashes = {
                phash: image_ids[0] for phash, image_ids in mapping.items() if image_ids
            }

        processor = ImageProcessor(
            self._settings,
            blobstore=self.blobstore,
            llm=providers.llm,
        )
        return await processor.process(
            parsed.images,
            kb_id=kb.kb_id,
            doc_id=parsed.doc_id,
            doc_title=parsed.title,
            existing_phashes=existing_phashes,
            caption_enabled=caption_enabled,
        )

    async def _embed(
        self,
        kb: KnowledgeBase,
        providers: ProviderSet,
        chunks: Sequence[Chunk],
        images: Sequence[ProcessedImage],
        outcome: FileOutcome,
    ) -> tuple[list[VectorRecord], list[VectorRecord], bool, bool]:
        """两路 embedding，返回 ``(文本记录, 图像记录, 文本空间可写, 图像空间可写)``。

        「可写」与「有记录」是两件事：文件改版后可能一张图都不剩，
        此时仍然需要删掉上一次导入留下的图像向量。所以提交阶段用
        ``*_active`` 决定是否清理旧向量，而不是用 ``records`` 是否为空。

        文本向量失败必须让整份文档失败：只有一半分块有向量时，检索会
        静默地漏掉另一端，而用户会把「答不出来」归咎于模型。
        图像向量失败则是**允许的降级** —— 图片仍然能通过 caption 走文本检索。

        **两路并发**：文本向量与图像向量请求的是两个不同的端点、
        用的是两份额度，彼此没有依赖。串行时一份「正文很长 + 图片很多」的
        文档要等两段网络时间相加；并发后总耗时约等于较慢的那一路。
        """
        # return_exceptions 而不是让第一路异常取消第二路：
        # 两路都已经发出去的请求，取消其中一路既浪费配额又拿不到它的降级信息。
        text_result, image_result = await asyncio.gather(
            self._embed_text(kb, providers, chunks, outcome),
            self._embed_images(kb, providers, images, outcome),
            return_exceptions=True,
        )
        # 文本错误优先抛出：它才是「文档不可检索」的原因，
        # 图像那一路即便也失败了，信息量也远小于它。
        for result in (text_result, image_result):
            if isinstance(result, BaseException):
                raise result

        text_records, text_active = text_result  # type: ignore[misc]
        image_records, image_active = image_result  # type: ignore[misc]
        return text_records, image_records, text_active, image_active

    async def _embed_text(
        self,
        kb: KnowledgeBase,
        providers: ProviderSet,
        chunks: Sequence[Chunk],
        outcome: FileOutcome,
    ) -> tuple[list[VectorRecord], bool]:
        text_provider = providers.text_embedding
        if text_provider is None:
            outcome.degraded_reasons.append(DegradedReason.TEXT_EMBED_UNAVAILABLE)
            outcome.warnings.append(
                "文本向量模型不可用，本文档只能通过 BM25 检索"
            )
            return [], False

        self._guard_snapshot(kb, text_provider, space="text")
        await self.vectorstore.ensure_collection(
            kb.text_collection,
            dim=text_provider.dim,
            space="text",
            embed_model=text_provider.model,
        )
        started = time.perf_counter()
        result = await text_provider.embed_texts([chunk.text for chunk in chunks])
        get_metrics().observe(
            "ingest.embed.text", (time.perf_counter() - started) * 1000
        )
        if len(result.vectors) != len(chunks):
            raise IngestError(
                "文本向量条数与分块数不一致："
                f"{len(result.vectors)} != {len(chunks)}",
                detail={"doc_id": chunks[0].doc_id if chunks else ""},
            )
        records = [
            VectorRecord(
                id=chunk.chunk_id,
                vector=vector,
                document=chunk.text,
                metadata=chunk.to_metadata(),
            )
            for chunk, vector in zip(chunks, result.vectors, strict=True)
        ]
        return records, True

    async def _embed_images(
        self,
        kb: KnowledgeBase,
        providers: ProviderSet,
        images: Sequence[ProcessedImage],
        outcome: FileOutcome,
    ) -> tuple[list[VectorRecord], bool]:
        image_provider = providers.image_embedding
        if image_provider is None:
            if images:
                outcome.degraded_reasons.append(
                    DegradedReason.IMAGE_EMBED_UNAVAILABLE
                )
                outcome.warnings.append(
                    "多模态向量模型不可用，图片无法参与以文搜图/以图搜图"
                )
            return [], False

        # 快照校验与 collection 建立**不吞异常**：维度 / 模型不一致是知识库
        # 完整性问题，不是「这一次请求运气不好」，必须让整份文档失败。
        self._guard_snapshot(kb, image_provider, space="image")
        await self.vectorstore.ensure_collection(
            kb.image_collection,
            dim=image_provider.dim,
            space="image",
            embed_model=image_provider.model,
        )

        # 到这里说明多模态空间本次可写：即使一张图都没剩，
        # 也要允许提交阶段清掉该文档的旧图像向量。
        active = True
        usable = [item for item in images if item.asset.image_id]
        if not usable:
            return [], active

        payloads, ready = await self._image_payloads(usable, outcome)
        if not ready:
            return [], active

        started = time.perf_counter()
        try:
            result = await image_provider.embed_images(payloads)
            if len(result.vectors) != len(ready):
                raise IngestError(
                    "图像向量条数与图片数不一致："
                    f"{len(result.vectors)} != {len(ready)}"
                )
        except Exception as exc:  # noqa: BLE001 - 见 docstring：图像降级是允许的
            get_metrics().observe(
                "ingest.embed.image",
                (time.perf_counter() - started) * 1000,
                error=True,
            )
            get_metrics().incr("ingest.embed.image_failures")
            outcome.degraded_reasons.append(DegradedReason.IMAGE_EMBED_UNAVAILABLE)
            outcome.warnings.append(
                f"图像向量生成失败，图片仍可经 caption 参与文本检索：{exc}"
            )
            return [], active
        get_metrics().observe(
            "ingest.embed.image", (time.perf_counter() - started) * 1000
        )

        records = [
            VectorRecord(
                id=item.asset.image_id,
                vector=vector,
                document=item.asset.caption_text,
                metadata=item.asset.to_metadata(),
            )
            for item, vector in zip(ready, result.vectors, strict=True)
        ]
        return records, active

    async def _image_payloads(
        self, items: Sequence[ProcessedImage], outcome: FileOutcome
    ) -> tuple[list[PreparedImage], list[ProcessedImage]]:
        """准备送进多模态端点的图片。

        caption 阶段已经预处理过的图直接复用；``--no-caption`` 时
        才在这里读文件并预处理。单张图失败只跳过它 ——
        一张读不出来的图不该让整份文档的文本也进不了库。
        """
        payloads: list[PreparedImage] = []
        ready: list[ProcessedImage] = []
        blob = self._settings.blob
        for item in items:
            try:
                if item.prepared is None:
                    raw = await asyncio.to_thread(
                        Path(item.asset.file_path).read_bytes
                    )
                    item.prepared = await asyncio.to_thread(
                        prepare_image_for_api,
                        raw,
                        max_side=self._settings.vision.max_image_side,
                        jpeg_quality=blob.jpeg_quality,
                    )
            except Exception as exc:  # noqa: BLE001 - 单张图降级
                outcome.warnings.append(
                    f"图片 {item.asset.image_id} 预处理失败，跳过图像向量：{exc}"
                )
                continue
            payloads.append(item.prepared)
            ready.append(item)
        return payloads, ready

    async def _commit(
        self,
        kb: KnowledgeBase,
        doc_id: str,
        chunks: Sequence[Chunk],
        image_outcome: ImageProcessOutcome,
        text_records: Sequence[VectorRecord],
        image_records: Sequence[VectorRecord],
        *,
        text_active: bool,
        image_active: bool,
    ) -> None:
        """写库。顺序见模块 docstring，不要随手调换。"""
        # 1. SQLite：事实源
        await self.docstore.replace_document_chunks(doc_id, chunks)
        assets = [item.asset for item in image_outcome.items]
        if assets:
            await self.docstore.upsert_images(assets)
        await self._delete_stale_images(kb.kb_id, doc_id, assets)

        # 2. 向量库：按空间分别判断。
        #    --no-embed 重导时删掉旧向量 = 把上一次导入的成果也一起毁了。
        if text_active:
            await self.vectorstore.delete(kb.text_collection, where={"doc_id": doc_id})
            if text_records:
                await self.vectorstore.upsert(
                    kb.text_collection, list(text_records), space="text"
                )
        if image_active:
            await self.vectorstore.delete(
                kb.image_collection, where={"doc_id": doc_id}
            )
            if image_records:
                await self.vectorstore.upsert(
                    kb.image_collection, list(image_records), space="image"
                )

        # 3. BM25：先删后加。文件改短时旧的高序号分块不会被覆盖。
        await self.bm25store.remove_documents_by_doc(kb.kb_id, doc_id)
        if chunks:
            await self.bm25store.add_documents(
                kb.kb_id,
                [
                    BM25Doc(id=chunk.chunk_id, text=chunk.text, payload=chunk.to_metadata())
                    for chunk in chunks
                ],
            )

    async def _delete_stale_images(
        self, kb_id: str, doc_id: str, current: Sequence[ImageAsset]
    ) -> None:
        """删掉重导之后不再存在的图片记录（文件改版时会出现）。"""
        keep = {asset.image_id for asset in current}
        existing = await self.docstore.list_images(kb_id, doc_id=doc_id, limit=10_000)
        stale = [asset.image_id for asset in existing if asset.image_id not in keep]
        if stale:
            await self.docstore.delete_images(stale)

    async def _mark_failed(
        self, kb: KnowledgeBase, doc_id: str, error: str
    ) -> None:
        """尽力清理派生物并标记失败。

        清理失败只记日志，不覆盖原始错误 —— 用户需要看到的是
        「为什么导入失败」，而不是「清理时又失败了一次」。
        """
        try:
            await self.vectorstore.delete(
                kb.text_collection, where={"doc_id": doc_id}
            )
        except Exception:  # noqa: BLE001 - 集合可能尚未创建
            logger.debug("回滚文本向量失败", exc_info=True)
        try:
            await self.vectorstore.delete(
                kb.image_collection, where={"doc_id": doc_id}
            )
        except Exception:  # noqa: BLE001
            logger.debug("回滚图像向量失败", exc_info=True)
        try:
            await self.bm25store.remove_documents_by_doc(kb.kb_id, doc_id)
        except Exception:  # noqa: BLE001
            logger.debug("回滚 BM25 失败", exc_info=True)

        try:
            await self.docstore.update_document_status(
                doc_id,
                DocStatus.FAILED,
                error=error[:1000],
            )
        except Exception:  # noqa: BLE001 - 文档行可能都没写成
            logger.debug("写入失败状态失败", exc_info=True)

    # -- provider -------------------------------------------------------------------

    def _build_providers(self, options: IngestOptions) -> ProviderSet:
        if self._providers is not None:
            return self._providers

        key = (options.caption, options.embed)
        cached = self._provider_cache.get(key)
        if cached is not None:
            return cached

        from app.providers.factory import (
            build_image_embedding,
            build_llm,
            build_text_embedding,
        )

        providers = ProviderSet()
        settings = self._settings

        if options.caption and settings.vision.enabled:
            try:
                providers.llm = build_llm(settings)
            except ConfigurationError as exc:
                providers.errors.append(str(exc))
                logger.info("对话模型不可用，图片 caption 降级为上下文占位")

        if options.embed and settings.text_embed.enabled:
            try:
                providers.text_embedding = build_text_embedding(settings)
            except ConfigurationError as exc:
                providers.errors.append(str(exc))
                logger.info("文本向量模型不可用，本次导入只写 BM25")

        if options.embed and settings.image_embed.enabled:
            try:
                providers.image_embedding = build_image_embedding(settings)
            except ConfigurationError as exc:
                # 图像向量是可选能力，构造失败也只降级
                providers.errors.append(str(exc))
                logger.info("多模态向量模型不可用，图片不参与以文搜图")

        self._provider_cache[key] = providers
        return providers

    def _guard_snapshot(
        self, kb: KnowledgeBase, provider: EmbeddingProvider, *, space: str
    ) -> None:
        """校验知识库快照与当前 provider 一致。

        这是「换了模型」事故的唯一拦路点：模型一变，向量空间就变了，
        新旧向量之间的余弦相似度没有意义。与其让检索悄悄返回垃圾，
        不如在写入前就拒绝。
        """
        if space == "text":
            expected_dim, expected_model, label = (
                kb.text_dim,
                kb.text_embed_model,
                "文本向量",
            )
        else:
            expected_dim, expected_model, label = (
                kb.image_dim,
                kb.image_embed_model,
                "多模态向量",
            )

        if expected_dim and provider.dim and expected_dim != provider.dim:
            raise IngestError(
                f"知识库 {kb.kb_id} 的{label}维度是 {expected_dim}，"
                f"当前配置是 {provider.dim}。换维度等于换向量空间，"
                "请新建知识库或清空后重建。",
                detail={"kb_id": kb.kb_id, "space": space},
            )
        if expected_model and provider.model and expected_model != provider.model:
            raise IngestError(
                f"知识库 {kb.kb_id} 建库时用的{label}模型是 {expected_model!r}，"
                f"当前配置是 {provider.model!r}。旧向量与新查询不再可比，"
                "请新建知识库或清空后重建。",
                detail={"kb_id": kb.kb_id, "space": space},
            )

    # -- 杂项 -----------------------------------------------------------------------

    async def _set_status(
        self,
        doc_id: str,
        status: DocStatus,
        progress: ProgressCallback | None,
        path: Path,
    ) -> None:
        await self.docstore.update_document_status(doc_id, status)
        self._notify(progress, status, f"{path.name}: {_STATUS_LABELS[status]}")

    def _notify(
        self,
        progress: ProgressCallback | None,
        status: DocStatus,
        message: str,
    ) -> None:
        if progress is None:
            return
        try:
            progress(status, message)
        except Exception:  # noqa: BLE001 - 回调不该影响导入
            logger.debug("进度回调抛异常，已忽略", exc_info=True)

    def _finish(self, outcome: FileOutcome, started: float) -> FileOutcome:
        outcome.elapsed_ms = int((time.perf_counter() - started) * 1000)
        outcome.warnings = _dedupe(outcome.warnings)
        outcome.degraded_reasons = _dedupe(outcome.degraded_reasons)
        metrics = get_metrics()
        metrics.observe(
            "ingest.file.total",
            outcome.elapsed_ms,
            error=outcome.status is DocStatus.FAILED,
        )
        metrics.incr("ingest.files")
        if outcome.status is DocStatus.FAILED:
            metrics.incr("ingest.files.failed")
        elif outcome.status is DocStatus.SKIPPED_DUPLICATE:
            metrics.incr("ingest.files.duplicate")
        elif outcome.status is DocStatus.DONE:
            metrics.incr("ingest.files.done")
            metrics.incr("ingest.chunks", outcome.chunks)
            metrics.incr("ingest.images", outcome.images)
            metrics.incr("ingest.text_vectors", outcome.text_vectors)
            metrics.incr("ingest.image_vectors", outcome.image_vectors)
        return outcome


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------

_STATUS_LABELS: dict[DocStatus, str] = {
    DocStatus.QUEUED: "排队中",
    DocStatus.PARSING: "解析中",
    DocStatus.CAPTIONING: "处理图片",
    DocStatus.EMBEDDING: "向量化",
    DocStatus.INDEXING: "写入索引",
    DocStatus.DONE: "完成",
    DocStatus.FAILED: "失败",
    DocStatus.SKIPPED_DUPLICATE: "跳过（重复）",
}


def _bind_images_to_chunks(
    images: Sequence[ProcessedImage], chunked: ChunkOutcome
) -> None:
    """把图片的 ``anchor_block_id`` 翻译成真实的 ``chunk_id``。

    ``block_to_chunk`` 里可能没有这个块（被 ``min_chunk_chars`` 丢掉了），
    此时留 ``None`` 而不是瞎猜相邻块 —— 一个错误的 ``text_chunk_id``
    会让前端「跳到原文」跳到完全无关的位置。
    """
    for item in images:
        block_id = item.anchor_block_id
        if not block_id or not block_id.startswith("b"):
            continue
        try:
            order = int(block_id[1:])
        except ValueError:
            continue
        chunk_id = chunked.block_to_chunk.get(order)
        if chunk_id:
            item.asset.anchor_chunk_id = chunk_id
            item.asset.text_chunk_id = chunk_id


def _dedupe(values: Iterable[str]) -> list[str]:
    """保序去重。警告列表里同一条重复十次只会挤掉真正有用的信息。"""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
