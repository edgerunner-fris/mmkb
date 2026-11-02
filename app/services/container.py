"""进程级依赖容器。

为什么需要它
------------
FastAPI 的 ``Depends`` 只解决「一个请求内怎么拿到依赖」，不解决
「一个进程内哪些东西只能有一份」。provider 内部的 ``httpx.AsyncClient``、
Chroma 的客户端、BM25 的内存索引都必须跨请求复用；而数据库 schema
初始化、上次崩溃留下的中间态清理又必须在**服务开始接请求之前**完成。

把这些放进 :class:`AppContainer`，``lifespan`` 只负责 ``startup()`` /
``shutdown()`` 两次调用，路由函数只通过 ``request.app.state.container``
取服务对象。

降级而不是崩溃
--------------
``build_llm`` / ``build_text_embedding`` 在缺 Key 时会抛
:class:`~app.config.ConfigurationError`。启动期把它们捕获下来记进
``provider_errors``：进程仍然能起，``/api/health`` 如实报告
``not_ready``，只有真正需要该能力的接口返回 503。
这让「没有 Key 也要能打开 /docs 看接口」成为默认行为，而不是要靠改代码。

``APP_ENV=prod`` 时相反：``lifespan`` 调 ``Settings.require_ready()``，
配置不全直接拒绝启动。开发期宽松、生产期严格，两种诉求本来就不一样。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from app.agent.graph import AnswerAgent
from app.config import (
    ConfigurationError,
    Settings,
    collect_config_problems,
    get_settings,
)
from app.core.logging import get_logger
from app.generation.answer import AnswerGenerator
from app.ingestion.pipeline import IngestionPipeline, ProviderSet
from app.providers.base import EmbeddingProvider, LLMProvider, RerankProvider
from app.providers.factory import (
    build_image_embedding,
    build_llm,
    build_rerank,
    build_text_embedding,
    provider_matrix,
)
from app.retrieval.hybrid import HybridRetriever
from app.services.chat_service import ChatService
from app.services.health_service import HealthService
from app.services.ingest_service import IngestService
from app.services.kb_service import KnowledgeBaseService
from app.services.perf_service import PerformanceService
from app.services.search_service import SearchService
from app.store.blobstore import BlobStore
from app.store.bm25store import BM25Store
from app.store.db import Database
from app.store.docstore import DocStore
from app.store.factory import build_blobstore, build_bm25store, build_vectorstore
from app.store.vectorstore import VectorStore

__all__ = ["AppContainer"]

logger = get_logger(__name__)

#: ``ConfigurationError.problems`` 里可能出现重复项（同一个 Key 被两个模块报），
#: 去重时保持顺序，让第一次出现的、通常也是最有上下文的那条留在前面。
def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


class AppContainer:
    """一次进程生命周期内的所有长生命周期对象。

    参数全部可注入，测试里可以给临时路径的 store；生产里全部由
    ``Settings`` 推导。**不读任何模块级单例**（``get_database`` 等），
    否则测试传进来的临时配置会被全局缓存覆盖。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        database: Database | None = None,
        vectorstore: VectorStore | None = None,
        bm25store: BM25Store | None = None,
        blobstore: BlobStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database = database or Database.from_settings(self.settings)
        self.docstore = DocStore(self.settings, db=self.database)
        self.vectorstore = vectorstore or build_vectorstore(self.settings)
        self.bm25store = bm25store or build_bm25store(self.settings)
        self.blobstore = blobstore or build_blobstore(self.settings)

        self._llm: LLMProvider | None = None
        self._text_embedding: EmbeddingProvider | None = None
        self._image_embedding: EmbeddingProvider | None = None
        self._rerank: RerankProvider | None = None
        self._provider_errors: list[str] = []
        self._providers_built = False

        self._retriever: HybridRetriever | None = None
        self._agent: AnswerAgent | None = None
        self._pipeline: IngestionPipeline | None = None

        self.started_at: datetime | None = None
        self._closed = False

        # 服务对象只持有容器的引用，本身无状态（IngestService 的后台任务
        # 注册表除外）。按实例缓存，避免每个请求都 new 一遍。
        self.kbs = KnowledgeBaseService(self)
        self.ingest = IngestService(self)
        self.chat = ChatService(self)
        self.search = SearchService(self)
        self.health = HealthService(self)
        self.perf = PerformanceService(self)

    # ==================================================================================
    # 生命周期
    # ==================================================================================

    async def startup(self, *, recover: bool = True) -> None:
        """建表、补目录、构造 provider、清理上次进程留下的中间态。

        顺序有讲究：schema 必须先于任何 ``docstore`` 调用；provider 的构造
        不依赖数据库，但恢复任务需要 ``TaskStatus``，所以放在建表之后。

        ``recover=False`` 供离线 CLI（基准测试、评测脚本）使用：它们可能与
        正在运行的 API 共用同一个数据库，此时把 ``running`` 任务标记为失败
        是在破坏另一个进程的状态，而不是「恢复」。
        """
        self.settings.paths.ensure()
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        await self.database.init_schema()
        self.blobstore.cleanup_tmp()
        self._build_providers()

        if recover:
            await self.health.recover_interrupted_tasks()
            await self.kbs.resume_deletions()

        self.started_at = datetime.now(UTC)
        errors, warnings = collect_config_problems(self.settings)
        logger.info(
            "服务启动完成",
            extra={
                "env": self.settings.app.env,
                "status": self.readiness(errors, warnings),
                "capabilities": self.capabilities,
                "config_errors": len(errors),
                "config_warnings": len(warnings),
            },
        )

    async def shutdown(self) -> None:
        """按「生产者 → 消费者 → 连接池」的顺序关停。

        先停后台导入任务：它们还会写 SQLite、BM25、向量库，先关存储会让
        任务在 ``CancelledError`` 之外再抛一堆「连接已关闭」，把真正的关闭
        原因淹没掉。BM25 在关之前会把内存索引落盘。
        """
        if self._closed:
            return
        self._closed = True

        try:
            await self.ingest.shutdown()
        except Exception:  # noqa: BLE001 - 关停阶段任何异常都不该阻止后续清理
            logger.warning("关闭导入服务失败", exc_info=True)

        if self._pipeline is not None:
            try:
                await self._pipeline.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("关闭摄入流水线失败", exc_info=True)

        for name, provider in (
            ("llm", self._llm),
            ("text_embedding", self._text_embedding),
            ("image_embedding", self._image_embedding),
            ("rerank", self._rerank),
        ):
            if provider is None:
                continue
            try:
                await provider.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("关闭 provider 失败", extra={"provider": name}, exc_info=True)

        try:
            await self.bm25store.aclose()
        except Exception:  # noqa: BLE001
            logger.warning("持久化 BM25 索引失败", exc_info=True)
        try:
            await self.vectorstore.aclose()
        except Exception:  # noqa: BLE001
            logger.warning("关闭向量库失败", exc_info=True)
        try:
            await self.database.dispose()
        except Exception:  # noqa: BLE001
            logger.warning("关闭数据库失败", exc_info=True)

        logger.info("服务已关闭")

    # ==================================================================================
    # Provider
    # ==================================================================================

    def _build_providers(self) -> None:
        """构造 provider。缺 Key 是**预期内的降级**，不是启动失败。

        四个 builder 都跑一遍再统一记录，缺多个 Key 时能一次列全 ——
        用户补一个、重启、再发现缺一个，体验很差。
        """
        if self._providers_built:
            return
        self._providers_built = True

        problems: list[str] = []
        builders = (
            ("llm", build_llm),
            ("text_embedding", build_text_embedding),
            ("image_embedding", build_image_embedding),
            ("rerank", build_rerank),
        )
        built: dict[str, Any] = {}
        for name, builder in builders:
            try:
                built[name] = builder(self.settings)
            except ConfigurationError as exc:
                built[name] = None
                problems.extend(exc.problems)
            except Exception as exc:  # noqa: BLE001 - 装配失败也不该让进程起不来
                built[name] = None
                problems.append(f"{name} 初始化失败：{exc}")
                logger.exception("provider 初始化出现未预期异常", extra={"provider": name})

        self._llm = built["llm"]
        self._text_embedding = built["text_embedding"]
        self._image_embedding = built["image_embedding"]
        self._rerank = built["rerank"]
        self._provider_errors = _dedupe(problems)

        logger.info(
            "provider 装配完成",
            extra={
                "llm": self._llm is not None,
                "text_embedding": self._text_embedding is not None,
                "image_embedding": self._image_embedding is not None,
                "rerank": self._rerank is not None,
                "errors": len(self._provider_errors),
            },
        )

    @property
    def llm(self) -> LLMProvider | None:
        return self._llm

    @property
    def text_embedding(self) -> EmbeddingProvider | None:
        return self._text_embedding

    @property
    def image_embedding(self) -> EmbeddingProvider | None:
        return self._image_embedding

    @property
    def rerank(self) -> RerankProvider | None:
        return self._rerank

    @property
    def provider_errors(self) -> list[str]:
        return list(self._provider_errors)

    @property
    def capabilities(self) -> dict[str, bool]:
        """当前进程**实际**具备的能力（不是配置里写了什么）。"""
        return {
            "llm": self._llm is not None,
            "dense_text": self._text_embedding is not None,
            "dense_image": self._image_embedding is not None,
            "rerank": self._rerank is not None,
            "vision": self._llm is not None and self.settings.vision.enabled,
            "bm25": True,
        }

    @property
    def provider_matrix(self) -> dict[str, dict[str, Any]]:
        return provider_matrix(self.settings)

    # ==================================================================================
    # 依赖对象
    # ==================================================================================

    @property
    def retriever(self) -> HybridRetriever:
        """三路检索器。没有 LLM 也能构造 —— 只影响查询改写与判定。"""
        if self._retriever is None:
            self._retriever = HybridRetriever(
                self.settings,
                docstore=self.docstore,
                vectorstore=self.vectorstore,
                bm25store=self.bm25store,
                text_embedding=self._text_embedding,
                image_embedding=self._image_embedding,
                rerank=self._rerank,
                llm=self._llm,
            )
        return self._retriever

    @property
    def pipeline(self) -> IngestionPipeline:
        """摄入流水线。provider 由容器持有，流水线只借用不关闭。"""
        if self._pipeline is None:
            self._pipeline = IngestionPipeline(
                self.settings,
                docstore=self.docstore,
                vectorstore=self.vectorstore,
                bm25store=self.bm25store,
                blobstore=self.blobstore,
                providers=ProviderSet(
                    llm=self._llm,
                    text_embedding=self._text_embedding,
                    image_embedding=self._image_embedding,
                ),
            )
        return self._pipeline

    def require_llm(self) -> LLMProvider:
        if self._llm is None:
            # 只挑与对话模型相关的问题：把「文本向量缺 Key」当成问答接口的
            # 错误原因返回，会让用户去修一个与本次请求无关的配置。
            relevant = [error for error in self._provider_errors if "LLM" in error]
            raise ConfigurationError(
                relevant
                or ["LLM_API_KEY 未配置：问答与相关性判定都依赖对话模型"]
            )
        return self._llm

    def require_image_embedding(self) -> EmbeddingProvider:
        """以图搜图直接依赖多模态向量 provider，缺失时明确 503。"""
        if self._image_embedding is None:
            relevant = [
                error for error in self._provider_errors if "IMAGE_EMBED" in error
            ]
            raise ConfigurationError(
                relevant
                or [
                    "IMAGE_EMBED_API_KEY 未配置：以图搜图与图像入库都依赖多模态向量模型。"
                    "若不需要图像检索，可设 IMAGE_EMBED_ENABLED=false"
                ]
            )
        return self._image_embedding

    def require_agent(self) -> AnswerAgent:
        """问答编排器。缺 LLM 时抛配置错误，由 API 层转成 503。"""
        if self._agent is not None:
            return self._agent
        llm = self.require_llm()
        self._agent = AnswerAgent(
            self.settings,
            retriever=self.retriever,
            generator=AnswerGenerator(llm, self.settings),
            llm=llm,
        )
        return self._agent

    @property
    def agent_available(self) -> bool:
        return self._agent is not None or self._llm is not None

    # ==================================================================================
    # 状态
    # ==================================================================================

    @staticmethod
    def readiness(errors: list[str], warnings: list[str]) -> str:
        if errors:
            return "not_ready"
        if warnings:
            return "degraded"
        return "ok"

    async def stats(self) -> dict[str, Any]:
        """轻量状态摘要，``/api/health`` 与启动日志共用。"""
        errors, warnings = collect_config_problems(self.settings)
        return {
            "status": self.readiness(errors, warnings),
            "capabilities": self.capabilities,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }
