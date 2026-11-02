"""存储层装配。

和 :mod:`app.providers.factory` 同一个思路：**这里是唯一构造具体后端的地方**。
业务代码只认 :class:`~app.store.vectorstore.VectorStore` 这层 ABC。

三个单例各自有理由必须是单例，而不是每个请求 new 一个：

* ``get_vectorstore()`` —— 持有 Chroma 客户端（内部是一个 SQLite 连接 + 一把写锁）
  或 pgvector 连接池。每个请求新建一个等于每次请求开一次库。
* ``get_blobstore()`` —— 无状态，但它做路径解析，缓存住省掉重复的 ``resolve()``。
* ``get_bm25store()`` —— 持有**内存里的 BM25 索引**。这个最要紧：
  每次请求重建索引意味着每次请求把整个知识库重新分词一遍。
* ``get_database()`` —— 在 :mod:`app.store.db` 里，它持有连接池。

测试用 ``reset_*`` 清缓存。**注意它们必须清掉，而且顺序有讲究**：
先 ``await vectorstore.aclose()`` 再 ``reset_vectorstore_cache()``，
否则连接池会漏在那儿（单例被丢掉后没人再持有它的引用，
但 asyncio 的池子不会因为 GC 就自己关掉）。

装配拆成两半：``build_*`` 是纯函数（给什么配置造什么对象，可测试），
``get_*`` 是零参数的缓存单例。**``get_*`` 刻意不接收 settings 参数** ——
``lru_cache`` 要求参数可哈希，而 pydantic 的 ``BaseSettings`` 默认不可哈希，
写成 ``get_vectorstore(settings)`` 会在**第一次调用**就抛
``TypeError: unhashable type: 'Settings'``。要换配置请用
``build_*``，或者改环境变量后 ``reset_*_cache()``。
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import ConfigurationError, Settings, get_settings
from app.core.errors import StorageError
from app.core.logging import get_logger
from app.store.vectorstore import VectorStore

if TYPE_CHECKING:
    # 只为类型标注。运行时靠函数体内的延迟导入 —— 见 get_vectorstore 的说明。
    from app.store.blobstore import BlobStore
    from app.store.bm25store import BM25Store

__all__ = [
    "build_bm25store",
    "build_blobstore",
    "build_vectorstore",
    "get_blobstore",
    "get_bm25store",
    "get_vectorstore",
    "reset_blobstore_cache",
    "reset_bm25store_cache",
    "reset_store_caches",
    "reset_vectorstore_cache",
]

logger = get_logger(__name__)


def build_vectorstore(settings: Settings) -> VectorStore:
    """按 ``VECTORSTORE_BACKEND`` 构造向量库后端。**不做缓存，不读全局配置。**

    ``vectorstore_pg`` 是**延迟导入**的：它在模块顶层 ``import psycopg`` /
    ``psycopg_pool``，而这两个包在 ``[pg]`` extra 里。默认安装（只装基础依赖）
    的用户不该因为「装了个没用的后端」而 import 失败 ——
    import 这个 factory 模块本身必须是安全的。
    """
    backend = settings.vectorstore.backend

    if backend == "pg":
        try:
            from app.store.vectorstore_pg import PgVectorStore
        except ImportError as exc:
            # ``ConfigurationError`` 收的是一个「问题列表」，与
            # ``collect_config_problems`` 的输出格式一致 —— 这样终端里看到的
            # 报错长得和启动自检完全一样，不必学第二套读法
            raise ConfigurationError(
                [
                    "VECTORSTORE_BACKEND=pg 但缺少 PostgreSQL 驱动或连接池："
                    f"{exc}",
                    '请执行：pip install -e ".[pg]"',
                ]
            ) from exc
        store = PgVectorStore(settings)
    elif backend == "chroma":
        from app.store.vectorstore_chroma import ChromaVectorStore

        store = ChromaVectorStore(settings)
    else:  # pragma: no cover - Literal 已在配置层拦住了
        raise ConfigurationError(
            [f"未知的 VECTORSTORE_BACKEND={backend!r}，可选：chroma / pg"]
        )

    logger.info(
        "向量库后端已装配",
        extra={"backend": store.backend, "metric": settings.vectorstore.effective_metric},
    )
    return store


@lru_cache(maxsize=1)
def get_vectorstore() -> VectorStore:
    """进程级单例。要换配置请用 :func:`build_vectorstore`。"""
    return build_vectorstore(get_settings())


def reset_vectorstore_cache() -> None:
    """清掉单例缓存。**调用方负责先 ``await`` 它的 ``aclose()``。**"""
    get_vectorstore.cache_clear()


def build_blobstore(settings: Settings) -> BlobStore:
    from app.store.blobstore import BlobStore

    return BlobStore(settings)


@lru_cache(maxsize=1)
def get_blobstore() -> BlobStore:
    return build_blobstore(get_settings())


def reset_blobstore_cache() -> None:
    get_blobstore.cache_clear()


def build_bm25store(settings: Settings) -> BM25Store:
    from app.store.bm25store import BM25Store

    return BM25Store(settings)


@lru_cache(maxsize=1)
def get_bm25store() -> BM25Store:
    return build_bm25store(get_settings())


def reset_bm25store_cache() -> None:
    get_bm25store.cache_clear()


async def reset_store_caches() -> None:
    """测试与优雅关停用：关掉持有外部资源的单例，再清缓存。"""
    try:
        store = get_vectorstore()
    except (StorageError, ConfigurationError, ImportError):
        # 构造都失败了，说明压根没建起来，没什么可关的
        pass
    else:
        await store.aclose()
    reset_vectorstore_cache()
    reset_blobstore_cache()
    reset_bm25store_cache()
