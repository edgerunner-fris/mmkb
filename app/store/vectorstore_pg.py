"""PostgreSQL + pgvector 后端（可选）。

默认走 Chroma；只有当 ``VECTORSTORE_BACKEND=pg`` 且装了 ``pip install -e ".[pg]"``
时才会用到这里。存在的意义是证明 :class:`~app.store.vectorstore.VectorStore`
这层抽象是真的能被第二个实现替换掉的，而不是一个「为抽象而抽象」的空壳。

本模块**在顶层 import psycopg**，所以没装 ``[pg]`` 时它自身导入会失败 ——
这是刻意的，由 :func:`app.store.factory.get_vectorstore` 在
``VECTORSTORE_BACKEND=pg`` 时才延迟导入，并捕获 ``ImportError``
给出「请执行 pip install -e ".[pg]"」这句人话。

与 Chroma 实现的三个实质差异
----------------------------

**1. 表名是 SQL 标识符，必须走 ``sql.Identifier``。**
   ``collection`` 派生自用户可见的 ``kb_id``。虽然
   :func:`~app.store.vectorstore.text_collection` 已经把字符集收窄到
   ``[A-Za-z0-9._-]``，但**拼接 SQL 字符串这个动作本身就是错的习惯** ——
   今天安全不代表以后加字段的人不会把它放宽。这里一律让驱动去转义。

**2. 元数据过滤要翻译成 jsonb 表达式。**
   Chroma 的 ``where`` 是 ``{"doc_id": {"$in": [...]}}`` 这种 Mongo 风味，
   本项目的上游（``RetrievalFilters.to_chroma_where``）也只产出这一种形状。
   两个后端必须接受同一份 ``where``，否则「换个后端就要改检索层」——
   那这层抽象就白做了。翻译逻辑见 :func:`_where_to_sql`。

**3. 距离算子与 opclass 必须成对出现。**
   ``<=>`` 配 ``vector_cosine_ops``，``<->`` 配 ``vector_l2_ops``，
   ``<#>`` 配 ``vector_ip_ops``。**配错不会报错**，
   只会让索引失效退化成全表扫描 —— 数据量小的时候完全看不出来。
   见 :data:`_METRIC_SPECS`。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import Settings, get_settings
from app.core.errors import StorageError, VectorDimensionMismatchError
from app.core.logging import get_logger
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

__all__ = ["PgVectorStore"]

logger = get_logger(__name__)

#: 度量 → (距离算子, 索引 opclass)
_METRIC_SPECS: dict[str, tuple[str, str]] = {
    "cosine": ("<=>", "vector_cosine_ops"),
    "l2": ("<->", "vector_l2_ops"),
    "ip": ("<#>", "vector_ip_ops"),
}

#: PostgreSQL 标识符长度上限。超了就截断 + 哈希后缀。
_MAX_IDENT = 63

#: ``where`` 支持的比较算子。
_COMPARISON_OPS: dict[str, str] = {
    "$eq": "=",
    "$ne": "<>",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


class PgVectorStore(VectorStore):
    """pgvector 后端。**进程级单例**，内部维护连接池。"""

    backend = "pg"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        pool: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        cfg = self._settings.vectorstore
        if not cfg.pg_dsn.strip():
            raise StorageError(
                "VECTORSTORE_BACKEND=pg 但 VECTORSTORE_PG_DSN 为空",
                detail={"hint": "填上 postgresql://user:pass@host:5432/dbname"},
            )
        self._dsn = cfg.pg_dsn
        self._prefix = cfg.pg_table_prefix
        self._metric = cfg.effective_metric
        self._owns_pool = pool is None
        self._pool = pool or AsyncConnectionPool(
            conninfo=self._dsn,
            min_size=1,
            max_size=cfg.pg_pool_size,
            # 显式 open：构造函数里连数据库会让「导入这个模块」变成一次网络操作
            open=False,
            kwargs={"row_factory": dict_row, "autocommit": True},
        )
        self._opened = not self._owns_pool
        self._lock = asyncio.Lock()  # DDL 串行化：建表 / 删表不该并发

    # ------------------------------------------------------------------ 连接

    async def _ensure_pool(self) -> None:
        if self._opened:
            return
        async with self._lock:
            if self._opened:
                return
            await self._pool.open(wait=True, timeout=10.0)
            self._opened = True
            async with self._pool.connection() as conn:
                # 扩展只需建一次。放在这里而不是要求运维手工执行：
                # 「跑起来才发现忘了 CREATE EXTENSION」是最常见的部署失败原因之一。
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("pgvector 连接池已建立", extra={"metric": self._metric})

    @property
    def table_prefix(self) -> str:
        return self._prefix

    @property
    def metric(self) -> str:
        return self._metric

    def _table(self, name: str) -> str:
        """collection 名 → 表名。超长时截断并加哈希后缀以保证唯一。"""
        raw = f"{self._prefix}_{name}"
        if len(raw) <= _MAX_IDENT:
            return raw
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        return f"{raw[: _MAX_IDENT - len(digest) - 1]}_{digest}"

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
        await self._ensure_pool()
        table = self._table(name)

        existing = await self.get_collection_info(name)
        if existing is not None and not recreate:
            if existing.dim is not None and existing.dim != dim:
                raise VectorDimensionMismatchError(
                    collection=name,
                    expected=existing.dim,
                    actual=dim,
                    hint="该表已存在且维度不同，换了 embedding 模型？确认后请重建知识库。",
                )
            if existing.space is not None and existing.space != space:
                raise VectorSpaceMismatchError(
                    expected=str(existing.space), actual=space, collection=name
                )
            return existing

        _, opclass = _METRIC_SPECS.get(self._metric, _METRIC_SPECS["cosine"])
        async with self._lock:
            async with self._pool.connection() as conn:
                if recreate:
                    await conn.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table))
                    )
                # 维度用 Literal 拼进类型里，不能用占位符 —— DDL 的类型参数不支持参数化
                await conn.execute(
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {table} ("
                        "  id        TEXT PRIMARY KEY,"
                        "  embedding vector({dim}) NOT NULL,"
                        "  document  TEXT NOT NULL DEFAULT '',"
                        "  metadata  JSONB NOT NULL DEFAULT '{{}}'::jsonb"
                        ")"
                    ).format(
                        table=sql.Identifier(table),
                        dim=sql.Literal(int(dim)),
                    )
                )
                await conn.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {index} ON {table} "
                        "USING hnsw (embedding {opclass})"
                    ).format(
                        index=sql.Identifier(f"{table}_embedding_idx"),
                        table=sql.Identifier(table),
                        opclass=sql.SQL(opclass),
                    )
                )
                # 元数据过滤走 GIN，否则 where 一多就退化成全表扫描
                await conn.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {index} ON {table} USING gin (metadata)"
                    ).format(
                        index=sql.Identifier(f"{table}_metadata_idx"),
                        table=sql.Identifier(table),
                    )
                )
                # dim / space / model 存在表注释里，而不是另建一张元数据表：
                # 注释随表一起 DROP，不会留下孤儿行；psql \d+ 就能看到，排查少一步 JOIN
                await conn.execute(
                    sql.SQL("COMMENT ON TABLE {table} IS {comment}").format(
                        table=sql.Identifier(table),
                        comment=sql.Literal(
                            f"collection={name} space={space} dim={dim} "
                            f"model={embed_model} metric={self._metric}"
                        ),
                    )
                )
        logger.info(
            "已创建向量表",
            extra={"collection": name, "table": table, "dim": dim, "space": space},
        )
        return CollectionInfo(
            name=name,
            count=await self.count(name),
            dim=dim,
            space=space,
            embed_model=embed_model,
            metadata={"table": table, "metric": self._metric, "opclass": opclass},
        )

    async def get_collection_info(self, name: str) -> CollectionInfo | None:
        await self._ensure_pool()
        table = self._table(name)
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT obj_description(c.oid, 'pg_class') AS comment "
                    "FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE c.relname = %s AND c.relkind = 'r' "
                    "AND n.nspname = current_schema()",
                    (table,),
                )
            ).fetchone()
            if row is None:
                return None
            count_row = await (
                await conn.execute(
                    sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(table))
                )
            ).fetchone()

        meta = _parse_comment(str(row.get("comment") or ""))
        return CollectionInfo(
            name=name,
            count=int((count_row or {}).get("n") or 0),
            dim=int(meta["dim"]) if meta.get("dim") else None,
            space=meta.get("space") or None,  # type: ignore[arg-type]
            embed_model=meta.get("model", ""),
            metadata={"table": table, "metric": meta.get("metric", self._metric)},
        )

    async def delete_collection(self, name: str) -> bool:
        await self._ensure_pool()
        table = self._table(name)
        existed = await self.get_collection_info(name) is not None
        async with self._lock:
            async with self._pool.connection() as conn:
                await conn.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table))
                )
        if existed:
            logger.info("已删除向量表", extra={"collection": name, "table": table})
        return existed

    async def list_collections(self, *, only_mmkb: bool = True) -> list[str]:
        await self._ensure_pool()
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT c.relname AS name, obj_description(c.oid, 'pg_class') AS comment "
                    "FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE c.relkind = 'r' AND n.nspname = current_schema() "
                    "AND c.relname LIKE %s "
                    "ORDER BY c.relname",
                    (f"{self._prefix}_%",),
                )
            ).fetchall()

        names: list[str] = []
        for row in rows:
            meta = _parse_comment(str(row.get("comment") or ""))
            # 没有我们写的注释 → 不是 mmkb 建的表（或很老的版本），从表名反推
            collection = meta.get("collection") or str(row["name"])[len(self._prefix) + 1 :]
            if only_mmkb and not is_mmkb_collection(collection):
                continue
            names.append(collection)
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
        await self._ensure_pool()

        info = await self.get_collection_info(name)
        if info is None:
            raise LookupError(f"collection {name!r} 不存在，请先 ensure_collection")
        self.guard_space(space, info.space, name)
        self.guard_dim([r.vector for r in records], info.dim, collection=name)

        # 同一批里重复 id 时保留最后一条，与 Chroma 后端的行为对齐
        items = list({r.id: r for r in records}.values())
        table = self._table(name)

        # ON CONFLICT DO UPDATE 让「导入失败重跑」在两种后端下都是幂等的
        statement = sql.SQL(
            "INSERT INTO {table} (id, embedding, document, metadata) "
            "VALUES (%s, %s::vector, %s, %s::jsonb) "
            "ON CONFLICT (id) DO UPDATE SET "
            "  embedding = EXCLUDED.embedding, "
            "  document  = EXCLUDED.document, "
            "  metadata  = EXCLUDED.metadata"
        ).format(table=sql.Identifier(table))

        rows = [
            (
                record.id,
                _vector_literal(record.vector),
                record.document,
                json.dumps(record.metadata, ensure_ascii=False),
            )
            for record in items
        ]
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(statement, rows)
        return len(items)

    async def delete(
        self,
        name: str,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> int:
        if not ids and not where:
            raise ValueError("delete 至少要给 ids 或 where 之一，否则会清空整个 collection")
        await self._ensure_pool()

        clauses: list[sql.Composable] = []
        params: list[Any] = []
        if ids:
            clauses.append(sql.SQL("id = ANY(%s)"))
            params.append(list(ids))
        if where:
            clause, where_params = _where_to_sql(where)
            clauses.append(clause)
            params.extend(where_params)

        statement = sql.SQL("DELETE FROM {table} WHERE {cond}").format(
            table=sql.Identifier(self._table(name)),
            cond=sql.SQL(" AND ").join(clauses),
        )
        async with self._pool.connection() as conn:
            result = await conn.execute(statement, params)
        return max(0, int(result.rowcount or 0))

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
        await self._ensure_pool()

        info = await self.get_collection_info(name)
        if info is None:
            logger.debug("查询了不存在的 collection", extra={"collection": name})
            return []
        self.guard_space(space, info.space, name)
        self.guard_dim([vector], info.dim, collection=name)

        operator, _ = _METRIC_SPECS.get(self._metric, _METRIC_SPECS["cosine"])
        literal = _vector_literal(vector)

        # 参数顺序必须与 SQL 里占位符出现的顺序完全一致：
        #   1) SELECT 的距离表达式
        #   2) where 里的条件（若有）
        #   3) ORDER BY 的算子
        #   4) LIMIT
        params: list[Any] = [literal]
        where_sql = sql.SQL("")
        if where:
            clause, where_params = _where_to_sql(where)
            where_sql = sql.SQL("WHERE {}").format(clause)
            params.extend(where_params)
        params.extend([literal, int(max(1, top_k))])

        # 距离在 SQL 里算（排序必须由数据库做），换算成相似度放在 Python ——
        # 与 Chroma 后端共用 distance_to_score，避免两边分数含义悄悄漂移
        statement = sql.SQL(
            "SELECT id, document, metadata, "
            "       (embedding {op} %s::vector) AS distance "
            "FROM {table} "
            "{where} "
            "ORDER BY embedding {op} %s::vector "
            "LIMIT %s"
        ).format(
            table=sql.Identifier(self._table(name)),
            where=where_sql,
            op=sql.SQL(operator),
        )

        async with self._pool.connection() as conn:
            rows = await (await conn.execute(statement, params)).fetchall()

        hits = [
            VectorHit(
                id=str(row["id"]),
                distance=float(row["distance"]),
                score=distance_to_score(float(row["distance"]), self._metric),
                document=str(row["document"] or ""),
                metadata=dict(row["metadata"] or {}),
            )
            for row in rows
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.id))
        return hits

    async def get(self, name: str, ids: list[str]) -> list[VectorRecord]:
        if not ids:
            return []
        await self._ensure_pool()
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute(
                    sql.SQL(
                        "SELECT id, embedding::text AS embedding, document, metadata "
                        "FROM {table} WHERE id = ANY(%s)"
                    ).format(table=sql.Identifier(self._table(name))),
                    (list(ids),),
                )
            ).fetchall()

        found: dict[str, VectorRecord] = {
            str(row["id"]): VectorRecord(
                id=str(row["id"]),
                vector=_parse_vector_literal(str(row["embedding"] or "")),
                document=str(row["document"] or ""),
                metadata=dict(row["metadata"] or {}),
            )
            for row in rows
        }
        # 按调用方给的顺序返回，缺失的跳过
        return [found[i] for i in ids if i in found]

    async def count(self, name: str) -> int:
        await self._ensure_pool()
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(
                    sql.SQL("SELECT count(*) AS n FROM {}").format(
                        sql.Identifier(self._table(name))
                    )
                )
            ).fetchone()
        return int((row or {}).get("n") or 0)

    # ------------------------------------------------------------------ 运维

    async def aclose(self) -> None:
        """只关自己建的池。外部传进来的池由外部负责关闭。"""
        if self._opened and self._owns_pool:
            await self._pool.close()
            self._opened = False


# --------------------------------------------------------------------------------------
# where 翻译
# --------------------------------------------------------------------------------------


def _where_to_sql(where: dict[str, Any]) -> tuple[sql.Composable, list[Any]]:
    """把 Chroma 风格的 ``where`` 翻译成 jsonb 条件。

    支持 ``$and`` / ``$or`` / ``$in`` / ``$nin`` / ``$eq`` / ``$ne`` /
    ``$gt`` / ``$gte`` / ``$lt`` / ``$lte``，也就是
    ``RetrievalFilters.to_chroma_where()`` 会产出的全部形状。

    **类型是这里唯一有点绕的地方**：值都存在 jsonb 里，``metadata->>'k'``
    取出来是 *text*。字符串比较与数字比较的语义完全不同 ——
    ``'10' < '9'`` 为真，而 ``10 < 9`` 为假。所以按 Python 值的类型
    决定要不要套 ``::numeric``。不处理这个，「按时间过滤」这类条件会静默地错。

    返回的 ``params`` 顺序与生成的 SQL 中占位符出现的顺序严格一致。
    """
    params: list[Any] = []

    def build(node: dict[str, Any]) -> sql.Composable:
        parts: list[sql.Composable] = []
        for key, condition in node.items():
            if key in ("$and", "$or"):
                clauses = [build(item) for item in condition]
                if not clauses:
                    continue
                joiner = sql.SQL(" AND ") if key == "$and" else sql.SQL(" OR ")
                parts.append(sql.SQL("(") + joiner.join(clauses) + sql.SQL(")"))
                continue
            parts.append(_field_condition(key, condition, params))
        if not parts:
            return sql.SQL("TRUE")
        return sql.SQL(" AND ").join(parts)

    return build(where), params


def _field_condition(key: str, condition: Any, params: list[Any]) -> sql.Composable:
    # -> 取 jsonb，->> 取 text。这里统一用 ->>，因为所有比较都在标量层面进行
    field = sql.SQL("metadata->>{}").format(sql.Literal(key))

    if not isinstance(condition, dict):
        condition = {"$eq": condition}

    clauses: list[sql.Composable] = []
    for op, value in condition.items():
        if op in ("$in", "$nin"):
            values = list(value)
            if not values:
                # 空集合：IN () 在 SQL 里是语法错误；语义上 IN 空集恒假、NOT IN 空集恒真
                clauses.append(
                    sql.SQL("FALSE") if op == "$in" else sql.SQL("TRUE")
                )
                continue
            params.extend(values)
            placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in values)
            expr = _casted(field, values)
            operator = sql.SQL("IN") if op == "$in" else sql.SQL("NOT IN")
            clauses.append(
                sql.SQL("{} {} ({})").format(expr, operator, placeholders)
            )
            continue

        operator_sql = _COMPARISON_OPS.get(op)
        if operator_sql is None:
            raise StorageError(
                f"pgvector 后端不支持的操作符 {op!r}",
                detail={
                    "supported": [
                        "$and", "$or", "$in", "$nin",
                        "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",
                    ],
                    "got": op,
                },
            )
        params.append(value)
        clauses.append(
            sql.SQL("{} {} {}").format(
                _casted(field, [value]), sql.SQL(operator_sql), sql.Placeholder()
            )
        )

    if not clauses:
        return sql.SQL("TRUE")
    return sql.SQL("(") + sql.SQL(" AND ").join(clauses) + sql.SQL(")")


def _casted(field: sql.Composable, values: list[Any]) -> sql.Composable:
    """按值的类型决定要不要把 jsonb 文本转成数字/布尔。"""
    concrete = [v for v in values if v is not None]
    if not concrete:
        return field
    if all(isinstance(v, bool) for v in concrete):
        # jsonb 的 true/false 与字符串 'true' 是两回事，
        # jsonb 类型本身就能与 boolean 比较，不需要转换
        return field
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in concrete):
        return sql.SQL("({})::numeric").format(field)
    return field


# --------------------------------------------------------------------------------------
# vector 字面量
# --------------------------------------------------------------------------------------


def _vector_literal(vector: list[float]) -> str:
    """pgvector 接受 ``'[1,2,3]'`` 这种文本字面量。

    不直接把 Python list 传给 ``::vector``：psycopg 会把 list 当 ARRAY 适配，
    到了服务端就是数组而不是 vector，报的错还很难懂。
    """
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _parse_vector_literal(text: str) -> list[float]:
    body = text.strip().lstrip("[").rstrip("]")
    if not body:
        return []
    return [float(part) for part in body.split(",") if part.strip()]


def _parse_comment(comment: str) -> dict[str, str]:
    """解析 ``COMMENT ON TABLE`` 里写的 ``k=v k=v``。"""
    out: dict[str, str] = {}
    for token in comment.split():
        if "=" in token:
            key, _, value = token.partition("=")
            out[key.strip()] = value.strip()
    return out
