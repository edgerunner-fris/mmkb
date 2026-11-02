"""ChromaDB PersistentClient 后端（默认）。

三个必须说清楚的实现决定
------------------------

**1. 绝不用 Chroma 的默认 embedding function。**
   ``chromadb`` 在不显式传 ``embedding_function`` 时会挂上
   ``DefaultEmbeddingFunction``（一个 ONNX MiniLM 模型），第一次调用就去下载
   几十 MB 的模型文件。本项目的铁律是「镜像内不下载任何本地模型」，
   所有向量都由云端 API 产出。所以这里显式传 ``embedding_function=None``，
   并且在自己的代码里先做 ``_require_vectors`` 检查 ——
   就算哪天有人忘了传向量，报错的也是我们自己的、看得懂的那一句，
   而不是一个「正在下载 onnx 模型」的进度条。

**2. 同步客户端丢进线程池，写操作用锁串行化。**
   ``PersistentClient`` 是同步 API，在 async 里必须 ``asyncio.to_thread``。
   并发问题在于：Chroma 本地模式内部是一个 SQLite 文件 + 段目录，
   多个线程同时写并不被官方保证安全。所以**写操作共用一把锁**，
   读操作不加锁。
   这个取舍的代价很小：写入的总耗时由上游的 embedding 网络请求主导，
   真正的写库时间只占几个百分点；而收益是「不会出现诡异的段损坏」。

**3. distance → score 的换算依赖 space，必须显式处理。**
   Chroma 返回的是**距离**，不是相似度，而且不同 space 的换算公式不同：

   ==========  ==========================  ====================
   space       含义                        换算
   ==========  ==========================  ====================
   cosine      ``1 - cos_sim``             ``score = 1 - d``
   l2          欧氏距离（未开方）             ``score = 1 / (1 + d)``
   ip          内积取负                     ``score = -d``
   ==========  ==========================  ====================

   把距离当相似度用，会让「最不相关」排到最前面，而且不会有任何报错。
   本模块统一把三者都换算成**越大越相关**的 score。
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from app.config import Settings, get_settings
from app.core.cache import TTLCache
from app.core.errors import VectorDimensionMismatchError
from app.core.logging import get_logger
from app.core.perf import get_metrics
from app.store.vectorstore import (
    CollectionInfo,
    VectorHit,
    VectorRecord,
    VectorSpace,
    VectorSpaceMismatchError,
    VectorStore,
    distance_to_score,
    is_mmkb_collection,
)

__all__ = ["ChromaVectorStore"]

logger = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    """Chroma 持久化后端。

    **不要在每次请求里构造它** —— 它持有客户端与一把写锁，
    应当是进程级单例（见 :func:`app.store.factory.get_vectorstore`）。
    """

    backend = "chroma"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._dir = self._settings.chroma_dir
        # 用 effective_metric 而不是 chroma_space：后者只是老名字的兼容入口，
        # 直接读它会在「只设了 VECTORSTORE_METRIC」时拿到默认值 cosine
        self._space = self._settings.vectorstore.effective_metric

        if client is not None:
            self._client = client
        else:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._dir),
                settings=ChromaSettings(
                    # 关掉匿名遥测：容器里没有外网出口，开着只会在日志里刷连接错误
                    anonymized_telemetry=False,
                    allow_reset=True,  # 测试要用；生产不会调用 reset
                ),
            )

        #: 写操作的串行化锁。见模块 docstring 第 2 点。
        self._write_lock = asyncio.Lock()
        self._batch_limit = self._resolve_batch_limit()

        #: collection 元信息（含条数）的短 TTL 缓存。
        #: 一次 hybrid 检索原本要把同一个 collection 描述 3~4 遍
        #: （route.probe → query → _query_sync），每遍都含一次
        #: ``get_collection`` + ``count``。元信息只在写入时变化，
        #: 所以本地每次 upsert / delete 后主动失效，TTL 只作为跨进程兜底。
        cache_cfg = self._settings.cache
        self._meta_cache: TTLCache[CollectionInfo] = TTLCache(
            "chroma.collection_meta",
            ttl_s=cache_cfg.collection_meta_ttl_s,
            max_entries=256,
            enabled=cache_cfg.enabled,
        )
        #: 已解析的 collection 句柄。Chroma 的句柄本身是轻量对象，
        #: 但每次 ``get_collection`` 都要走一次目录查找 + 反序列化 metadata。
        self._handles: dict[str, Any] = {}
        self._handle_lock = threading.Lock()

    def _resolve_batch_limit(self) -> int:
        """单次 upsert 的条数上限。

        Chroma 自己会报 ``get_max_batch_size()``（当前版本是 5461）；
        这里再取 ``min`` 一次是为了留出余量 —— 上限是**条数**，
        而每条记录还带着自己的 metadata 与 document，实际请求体会更大。
        """
        try:
            return max(1, min(int(self._client.get_max_batch_size()), 2048))
        except Exception:  # noqa: BLE001 - 拿不到就用保守值
            return 512

    @property
    def directory(self) -> Any:
        return self._dir

    # ------------------------------------------------------------------ 生命周期

    async def ensure_collection(
        self,
        name: str,
        *,
        dim: int,
        space: VectorSpace,
        embed_model: str = "",
        recreate: bool = False,
    ) -> CollectionInfo:
        async with self._write_lock:
            return await asyncio.to_thread(
                self._ensure_sync, name, dim, space, embed_model, recreate
            )

    def _ensure_sync(
        self, name: str, dim: int, space: VectorSpace, embed_model: str, recreate: bool
    ) -> CollectionInfo:
        # 这里刻意绕过缓存：ensure 是写入路径，必须看到磁盘上的真实状态，
        # 否则「刚刚在另一个进程里改了维度」会被缓存掩盖，直到 TTL 过期才报错。
        existing = self._describe_sync(name, use_cache=False)
        if existing is not None and not recreate:
            self._assert_compatible(existing, dim, space, name)
            return existing

        if existing is not None:
            logger.warning(
                "重建 collection（原有向量将全部丢弃）",
                extra={"collection": name, "old_count": existing.count},
            )
            self._client.delete_collection(name)
            self._forget_collection(name)

        metadata: dict[str, Any] = {
            "dim": int(dim),
            "space": space,
            "embed_model": embed_model,
            "backend": "chroma",
        }
        # 注意：hnsw 参数走 configuration，不要混进 metadata。
        # 混进去的话 Chroma 1.x 会把 "hnsw:space" 从 metadata 里剥掉，
        # 而我们自己写的 dim/space 会留下来 —— 看起来能用，
        # 但 collection 的实际距离度量可能与我们以为的不一致。
        collection = self._client.get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": self._space}},
            metadata=metadata,
            embedding_function=None,  # ⚠️ 见模块 docstring 第 1 点，绝不能省
        )
        logger.info(
            "已创建 collection",
            extra={"collection": name, "dim": dim, "space": space, "metric": self._space},
        )
        with self._handle_lock:
            self._handles[name] = collection
        self._meta_cache.invalidate(name)
        return CollectionInfo(
            name=name,
            count=collection.count(),
            dim=int(dim),
            space=space,  # type: ignore[arg-type]
            embed_model=embed_model,
            metadata=dict(collection.metadata or {}),
        )

    def _assert_compatible(
        self, info: CollectionInfo, dim: int, space: VectorSpace, name: str
    ) -> None:
        """已存在的 collection 必须与本次要写入的向量空间完全一致。

        这里**抛错而不是自动重建**。自动重建看起来「更智能」，
        实际上等于把整个知识库的向量悄悄作废 —— 用户下次检索发现结果全变了，
        却没有任何线索指向「早上启动时维度变了」。

        允许的一种情况：旧 collection 是我们早期的版本、metadata 里没写 dim。
        这时补写一次 metadata 而不是报错。
        """
        if info.dim is None or info.space is None:
            logger.warning(
                "collection 缺少 dim/space 元数据，按本次参数补写",
                extra={"collection": name},
            )
            self._patch_metadata_sync(name, dim, space)
            return
        if info.dim != dim:
            raise VectorDimensionMismatchError(
                collection=name,
                expected=info.dim,
                actual=dim,
                hint="该 collection 已存在且维度不同。换了 embedding 模型或改了 *_DIM？"
                "确认后请重建知识库（会自动重建 collection）。",
            )
        if info.space != space:
            raise VectorSpaceMismatchError(
                expected=str(info.space), actual=space, collection=name
            )

    def _patch_metadata_sync(self, name: str, dim: int, space: VectorSpace) -> None:
        collection = self._collection_sync(name)
        merged = dict(collection.metadata or {})
        merged.update({"dim": int(dim), "space": space})
        collection.modify(metadata=merged)
        self._meta_cache.invalidate(name)

    async def get_collection_info(self, name: str) -> CollectionInfo | None:
        return await asyncio.to_thread(self._describe_sync, name)

    def _collection_sync(self, name: str) -> Any:
        """取 collection 句柄，带进程内句柄缓存。

        句柄缓存只在**本进程写操作**之后失效（见 ``_forget_collection``）。
        Chroma 的句柄指向的是同一个底层 segment；外部进程删除 collection 后
        旧句柄会在使用时抛出异常，调用方按「collection 不存在」处理。
        """
        with self._handle_lock:
            handle = self._handles.get(name)
        if handle is not None:
            return handle
        handle = self._client.get_collection(name=name, embedding_function=None)
        with self._handle_lock:
            self._handles.setdefault(name, handle)
            return self._handles[name]

    def _forget_collection(self, name: str) -> None:
        """写入 / 删除后让元信息与句柄一起失效。"""
        self._meta_cache.invalidate(name)
        with self._handle_lock:
            self._handles.pop(name, None)

    def _describe_sync(self, name: str, *, use_cache: bool = True) -> CollectionInfo | None:
        if use_cache:
            cached = self._meta_cache.get(name)
            if cached is not None:
                return _copy_info(cached)
        try:
            collection = self._collection_sync(name)
            count = collection.count()
        except Exception:  # noqa: BLE001 - chromadb 用多种异常表示「不存在」
            self._forget_collection(name)
            return None
        meta = dict(collection.metadata or {})
        raw_dim = meta.get("dim")
        raw_space = meta.get("space")
        info = CollectionInfo(
            name=name,
            count=count,
            dim=int(raw_dim) if isinstance(raw_dim, (int, float, str)) else None,
            space=str(raw_space) if raw_space else None,  # type: ignore[arg-type]
            embed_model=str(meta.get("embed_model") or ""),
            metadata=meta,
        )
        self._meta_cache.set(name, info)
        return _copy_info(info)

    async def delete_collection(self, name: str) -> bool:
        async with self._write_lock:
            return await asyncio.to_thread(self._delete_collection_sync, name)

    def _delete_collection_sync(self, name: str) -> bool:
        try:
            self._client.delete_collection(name)
        except Exception:  # noqa: BLE001 - 不存在时删除是幂等的
            self._forget_collection(name)
            return False
        self._forget_collection(name)
        logger.info("已删除 collection", extra={"collection": name})
        return True

    async def list_collections(self, *, only_mmkb: bool = True) -> list[str]:
        return await asyncio.to_thread(self._list_sync, only_mmkb)

    def _list_sync(self, only_mmkb: bool) -> list[str]:
        names = [c.name for c in self._client.list_collections()]
        if only_mmkb:
            names = [n for n in names if is_mmkb_collection(n)]
        return sorted(names)

    # ------------------------------------------------------------------ 写

    async def upsert(
        self,
        name: str,
        records: list[VectorRecord],
        *,
        space: VectorSpace | None = None,
    ) -> int:
        if not records:
            return 0

        info = await self.get_collection_info(name)
        if info is None:
            raise LookupError(f"collection {name!r} 不存在，请先 ensure_collection")
        self.guard_space(space, info.space, name)
        self.guard_dim([r.vector for r in records], info.dim, collection=name)

        # 按 id 去重：同一批里出现重复 id 时，Chroma 的行为是「后者覆盖前者」
        # 但顺序不做保证。显式去重让结果可预期。
        deduped: dict[str, VectorRecord] = {}
        for record in records:
            deduped[record.id] = record
        items = list(deduped.values())
        if len(items) != len(records):
            logger.debug(
                "同一批 upsert 中出现重复 id，已保留最后一条",
                extra={"collection": name, "dropped": len(records) - len(items)},
            )

        written = 0
        started = time.perf_counter()
        async with self._write_lock:
            for start in range(0, len(items), self._batch_limit):
                shard = items[start : start + self._batch_limit]
                await asyncio.to_thread(self._upsert_sync, name, shard)
                written += len(shard)
            # 条数变了。主动失效，让紧接着的检索 / probe 看到真实 count，
            # 而不是等 TTL 到期。
            self._meta_cache.invalidate(name)
        get_metrics().observe("vectorstore.upsert", (time.perf_counter() - started) * 1000)
        get_metrics().incr("vectorstore.upsert_records", written)
        return written

    def _upsert_sync(self, name: str, shard: list[VectorRecord]) -> None:
        collection = self._collection_sync(name)

        # Chroma 1.5 起，metadata 传**空 dict** 会被直接拒绝：
        #   ValueError: Expected metadata to be a non-empty dict,
        #               got 0 metadata attributes in upsert.
        # 而 VectorRecord.metadata 的默认值就是 {}，所以「不传 metadata」
        # 这条看起来最无辜的路径反而会炸。空 dict 一律降级成 None，
        # 整批都没有 metadata 时连 kwarg 都不传。
        metadatas: list[dict[str, Any] | None] = [dict(r.metadata) or None for r in shard]
        payload: dict[str, Any] = {
            "ids": [r.id for r in shard],
            "embeddings": [r.vector for r in shard],
            "documents": [r.document for r in shard],
        }
        if any(meta is not None for meta in metadatas):
            payload["metadatas"] = metadatas
        collection.upsert(**payload)

    async def delete(
        self,
        name: str,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> int:
        if not ids and not where:
            raise ValueError("delete 至少要给 ids 或 where 之一，否则会清空整个 collection")
        async with self._write_lock:
            deleted = await asyncio.to_thread(self._delete_sync, name, ids, where)
            self._meta_cache.invalidate(name)
            return deleted

    def _delete_sync(
        self, name: str, ids: list[str] | None, where: dict[str, Any] | None
    ) -> int:
        info = self._describe_sync(name, use_cache=False)
        if info is None:
            return 0
        collection = self._collection_sync(name)
        before = collection.count()
        if ids:
            for start in range(0, len(ids), self._batch_limit):
                collection.delete(ids=ids[start : start + self._batch_limit])
        elif where is not None:
            # Chroma 的 where 删除没有 limit 参数可以放心用，
            # 但一次删太多会长时间持锁，分批按 count 变化判断是否删干净
            collection.delete(where=where)
        return max(0, before - collection.count())

    # ------------------------------------------------------------------ 读

    async def query(
        self,
        name: str,
        vector: list[float],
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        space: VectorSpace | None = None,
    ) -> list[VectorHit]:
        if not vector:
            raise ValueError("query 的向量为空。空向量查不出任何有意义的结果，请在上游拦住")

        info = await self.get_collection_info(name)
        if info is None:
            logger.debug("查询了不存在的 collection", extra={"collection": name})
            return []
        self.guard_space(space, info.space, name)
        self.guard_dim([vector], info.dim, collection=name)

        # 读操作不加写锁：Chroma 的读走的是自己的连接，与写并发是安全的，
        # 而加锁会让「用户提问」被「后台导入」阻塞 —— 体感上完全不可接受。
        started = time.perf_counter()
        try:
            return await asyncio.to_thread(
                self._query_sync, name, vector, top_k, where, info.count
            )
        finally:
            get_metrics().observe(
                "vectorstore.query", (time.perf_counter() - started) * 1000
            )

    def _query_sync(
        self,
        name: str,
        vector: list[float],
        top_k: int,
        where: dict[str, Any] | None,
        known_count: int | None = None,
    ) -> list[VectorHit]:
        if known_count is not None and known_count <= 0:
            # 元信息缓存刚失效过、或 collection 确实为空：不必再问一次 Chroma。
            return []
        collection = self._collection_sync(name)
        if collection.count() == 0:
            # n_results=0 之外的另一个边界：空 collection 查询在部分版本会抛错
            return []
        result = collection.query(
            query_embeddings=[vector],
            n_results=max(1, top_k),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        return _hits_from_query(result, self._space)

    async def get(self, name: str, ids: list[str]) -> list[VectorRecord]:
        if not ids:
            return []
        return await asyncio.to_thread(self._get_sync, name, ids)

    def _get_sync(self, name: str, ids: list[str]) -> list[VectorRecord]:
        info = self._describe_sync(name)
        if info is None:
            return []
        collection = self._collection_sync(name)
        result = collection.get(
            ids=list(ids), include=["documents", "metadatas", "embeddings"]
        )
        found: dict[str, VectorRecord] = {}
        raw_ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")
        for index, record_id in enumerate(raw_ids):
            vector = None if embeddings is None else _nth(embeddings, index, default=None)
            found[str(record_id)] = VectorRecord(
                id=str(record_id),
                vector=[float(x) for x in vector] if vector is not None else [],
                document=str(_nth(documents, index, default="") or ""),
                metadata=dict(_nth(metadatas, index, default={}) or {}),
            )
        # 按调用方给的顺序返回，缺失的跳过 —— 调用方往往要用它对齐另一份列表
        return [found[i] for i in ids if i in found]

    async def count(self, name: str) -> int:
        info = await self.get_collection_info(name)
        return info.count if info else 0

    def cache_stats(self) -> dict[str, Any]:
        """命中率归性能报告用。句柄缓存不计入命中率，只报当前数量。"""
        return {
            "collection_meta": self._meta_cache.stats().to_dict(),
            "collection_handles": len(self._handles),
        }

    def clear_cache(self) -> None:
        """清空元信息与句柄缓存。

        句柄也一起丢：清缓存的目的通常是「怀疑进程内缓存与磁盘不一致」，
        只清元信息而继续复用旧句柄，等于只解决了一半问题。
        """
        self._meta_cache.clear()
        self._meta_cache.reset_stats()
        with self._handle_lock:
            self._handles.clear()

    # ------------------------------------------------------------------ 运维

    async def aclose(self) -> None:
        """Chroma 的 PersistentClient 没有显式 close。

        这里什么都不做，让 GC 去关底层 SQLite 连接。
        写锁不影响 —— 它不持有任何系统资源。
        """

    async def stats(self) -> dict[str, Any]:
        """给 ``/api/health`` 用：每个 collection 的条数与维度。"""
        names = await self.list_collections()
        out: dict[str, Any] = {"backend": self.backend, "dir": str(self._dir)}
        for name in names:
            info = await self.get_collection_info(name)
            if info is not None:
                out[name] = {
                    "count": info.count,
                    "dim": info.dim,
                    "space": info.space,
                    "embed_model": info.embed_model,
                }
        return out


# --------------------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------------------


def _copy_info(info: CollectionInfo) -> CollectionInfo:
    """缓存返回一律给副本：调用方修改 metadata 不应污染缓存里的那一份。"""
    return CollectionInfo(
        name=info.name,
        count=info.count,
        dim=info.dim,
        space=info.space,
        embed_model=info.embed_model,
        metadata=dict(info.metadata),
    )


def _hits_from_query(result: Any, metric: str) -> list[VectorHit]:
    """把 Chroma 的 ``{ids, documents, metadatas, distances}`` 转成 VectorHit 列表。

    ``metric`` 是**距离度量**（cosine/l2/ip），不是向量空间（text/image）——
    两者恰好都来自配置里的一个叫 ``space`` 的字段，看调用点时容易混。

    Chroma 的结果是「每个 query 一个子列表」，我们只发一个 query，所以取 ``[0][i]``。
    某个字段没被 include 时 Chroma 返回 ``None`` 而不是空列表，
    所以四条都走 :func:`_nth` 而不是直接下标。
    """
    ids = _nth(result.get("ids"), 0, default=[]) or []
    documents = _nth(result.get("documents"), 0, default=[]) or []
    metadatas = _nth(result.get("metadatas"), 0, default=[]) or []
    distances = _nth(result.get("distances"), 0, default=[]) or []

    hits: list[VectorHit] = []
    for index, record_id in enumerate(ids):
        distance = float(_nth(distances, index, default=0.0))
        hits.append(
            VectorHit(
                id=str(record_id),
                distance=distance,
                score=distance_to_score(distance, metric),
                document=str(_nth(documents, index, default="") or ""),
                metadata=dict(_nth(metadatas, index, default={}) or {}),
            )
        )
    hits.sort(key=lambda hit: (-hit.score, hit.id))
    return hits


def _nth(container: Any, index: int, *, default: Any) -> Any:
    """安全取 ``container[index]``；越界 / 类型不符 / ``None`` 一律返回 ``default``。"""
    if container is None:
        return default
    try:
        return container[index]
    except (IndexError, KeyError, TypeError):
        return default
