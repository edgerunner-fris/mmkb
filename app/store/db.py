"""SQLAlchemy 异步引擎与会话工厂。

为什么单独一个文件
------------------
引擎是**进程级单例**（连接池必须复用），session 是**请求级**的。
把两者混在 ``docstore.py`` 里，很容易写出「在模块顶部建一个 session 全局共用」
这种在并发下必然出错的结构。这里把生命周期边界画清楚：

``Database``（进程级，一个）  →  ``session()``（每次操作一个）  →  ``DocStore``（无状态，随便建）

SQLite 的三个必设 PRAGMA
------------------------
1. ``journal_mode=WAL`` —— 否则读会阻塞写。导入文档时前端还在轮询任务进度，
   没有 WAL 就会出现「查询任务状态把导入卡住」。
2. ``foreign_keys=ON`` —— **SQLite 默认关闭外键约束**。
   不打开的话 ``ON DELETE CASCADE`` 完全不生效，删知识库会留下一堆孤儿行，
   而且不报任何错。
3. ``busy_timeout`` —— 并发写入时不要立刻抛 ``database is locked``，
   而是等一会儿。写操作已经由导入链路的信号量串行化了，这里只是兜底。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings
from app.core.logging import get_logger

__all__ = ["Database", "get_database", "reset_database_cache"]

logger = get_logger(__name__)

#: SQLite 忙等超时（毫秒）。比默认的 0 宽容得多，又不会让请求永久挂住。
_BUSY_TIMEOUT_MS = 30_000


def _sqlite_url(path: Path) -> str:
    """把本地路径转成 aiosqlite 的 URL。

    Windows 的 ``C:\\Users\\...`` 直接拼进 URL 会被当成主机名的一部分，
    必须用正斜杠形式 ``C:/Users/...``（三个斜杠之后是绝对路径）。
    """
    return f"sqlite+aiosqlite:///{path.as_posix()}"


class Database:
    """进程级数据库句柄。

    刻意**不在构造时连接**：``create_async_engine`` 是惰性的，
    真正的连接在第一次使用时建立。这让「配置对不对」和「库通不通」
    成为两个独立的、可以分别报错的问题。
    """

    def __init__(
        self,
        url: str,
        *,
        echo: bool = False,
        is_sqlite: bool = True,
        connect_args: dict[str, Any] | None = None,
    ) -> None:
        self.url = url
        self.is_sqlite = is_sqlite
        self._engine: AsyncEngine = create_async_engine(
            url,
            echo=echo,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args or {},
        )
        if is_sqlite:
            self._install_sqlite_pragmas(self._engine)
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,  # commit 后还要读对象属性，否则会触发额外查询
            autoflush=False,
        )

    @staticmethod
    def _install_sqlite_pragmas(engine: AsyncEngine) -> None:
        """在每个新连接上设置 PRAGMA。见模块 docstring 的三条理由。"""

        @event.listens_for(engine.sync_engine, "connect")
        def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
                cursor.execute("PRAGMA synchronous=NORMAL")
            finally:
                cursor.close()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Database:
        settings = settings or get_settings()
        path = settings.sqlite_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(_sqlite_url(path), echo=settings.docstore.echo_sql, is_sqlite=True)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """一个工作单元。

        出错回滚、正常提交。**不做嵌套事务**：SQLite 的 SAVEPOINT 语义
        与「导入失败要回滚整个文档」的诉求不完全对得上，
        所以回滚粒度就定在「一次 ``async with db.session()``」。
        """
        session = self._sessionmaker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def init_schema(self) -> list[str]:
        """建表（幂等）。返回本次实际创建的表名。

        用 ``create_all`` 而不是迁移工具：这是单机应用，
        表结构在第一个正式版本之后才会变化，到那时再引入 Alembic 也不迟。
        """
        from app.store.models import Base

        async with self._engine.begin() as conn:
            before = await conn.run_sync(
                lambda sync_conn: set(_inspector_tables(sync_conn))
            )
            await conn.run_sync(Base.metadata.create_all)
            after = await conn.run_sync(
                lambda sync_conn: set(_inspector_tables(sync_conn))
            )
        created = sorted(after - before)
        if created:
            logger.info("已创建数据表", extra={"tables": created})
        return created

    async def drop_all(self) -> None:
        """删掉所有表。**只给测试用**，生产代码不该调用。"""
        from app.store.models import Base

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def healthcheck(self) -> dict[str, Any]:
        """轻量探活：能执行一条语句就算通。"""
        async with self._engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.scalar_one()
        return {"ok": True, "url": self.safe_url, "dialect": self._engine.dialect.name}

    @property
    def safe_url(self) -> str:
        """去掉密码的连接串，可以安全地写进日志与 ``/api/health``。"""
        return self._engine.url.render_as_string(hide_password=True)

    async def dispose(self) -> None:
        await self._engine.dispose()
        logger.debug("数据库连接池已关闭")


def _inspector_tables(sync_conn: Any) -> list[str]:
    from sqlalchemy import inspect

    return inspect(sync_conn).get_table_names()


@lru_cache(maxsize=1)
def get_database() -> Database:
    """进程级单例。测试里换数据库请用 :func:`reset_database_cache`。"""
    return Database.from_settings()


async def reset_database_cache() -> None:
    """清空单例并关闭既有连接池（仅测试与热重载使用）。"""
    if get_database.cache_info().currsize:
        await get_database().dispose()
    get_database.cache_clear()
