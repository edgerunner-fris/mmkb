"""带 TTL 的进程内 LRU 缓存。

检索链路上有三类「同一输入必然得到同一输出」的昂贵操作：

* 把 query 文本编码成向量（一次外部 HTTP，100~400ms）；
* 对候选集做重排（一次外部 HTTP，200~800ms）；
* 读取 Chroma collection 的元信息与条数（一次磁盘/元数据查询）。

它们都不随时间变化，只随写入变化。因此用 **短 TTL + 写时失效** 两个手段
同时控制正确性与收益：

* TTL 兜底：即使某次写入没有走到失效逻辑（例如另一个进程写了库），
  陈旧数据最多存活 ``ttl_s`` 秒；
* 写时失效：本进程内 upsert / delete 之后主动清掉相关键，让下一次读立刻正确。

为什么是手写而不是 ``cachetools``
--------------------------------
项目依赖清单里没有 ``cachetools``，为 60 行逻辑新增一个依赖不划算；
而且这里需要暴露命中率给性能报告，自己实现反而更直接。
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

__all__ = ["CacheStats", "TTLCache", "cache_key"]


def cache_key(*parts: Any) -> str:
    """把任意片段拼成定长键。

    每个片段先编码再带长度前缀，避免 ``("ab", "c")`` 与 ``("a", "bc")``
    拼出同一个字符串——用分隔符拼接的缓存键出过太多这种事故。
    """
    digest = hashlib.sha256()
    for part in parts:
        chunk = str(part).encode("utf-8", errors="replace")
        digest.update(len(chunk).to_bytes(4, "big"))
        digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class CacheStats:
    name: str
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_entries: int = 0
    ttl_s: float = 0.0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "lookups": self.lookups,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "expirations": self.expirations,
            "size": self.size,
            "max_entries": self.max_entries,
            "ttl_s": round(self.ttl_s, 3),
        }


@dataclass(slots=True)
class _Entry[T]:
    value: T
    expires_at: float


class TTLCache[T]:
    """线程安全的小型 LRU + TTL 缓存。

    ``enabled=False`` 时所有读都落空、所有写都丢弃，调用方不需要到处写
    ``if cache_enabled`` 分支。这一点很重要：缓存是优化，关掉它绝不能
    改变业务语义。
    """

    def __init__(
        self,
        name: str,
        *,
        ttl_s: float = 60.0,
        max_entries: int = 256,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.ttl_s = max(0.0, float(ttl_s))
        self.max_entries = max(1, int(max_entries))
        self.enabled = bool(enabled) and self.ttl_s > 0
        self._data: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    # ------------------------------------------------------------------ 读写

    def get(self, key: str) -> T | None:
        """返回缓存值；未命中或已过期返回 ``None``。

        用 ``None`` 表示未命中意味着**不能缓存 None**。本项目的缓存值
        （向量、重排分数、collection 元信息）都不是 None，所以这个简化是安全的。
        """
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expires_at <= now:
                del self._data[key]
                self._expirations += 1
                self._misses += 1
                return None
            # 命中后移到队尾：队首即最久未使用。
            self._data.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            self._data[key] = _Entry(value=value, expires_at=now + self.ttl_s)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self._evictions += 1

    def get_or_set(self, key: str, factory: Any) -> T:
        """同步工厂版本。用于 collection 元信息这类即时计算。"""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value)
        return value

    # ------------------------------------------------------------------ 失效

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> int:
        """失效所有以 ``prefix`` 开头的键，返回删除条数。

        本项目的键都是 ``cache_key(...)`` 产生的十六进制串，没有可读前缀。
        真正的「按 collection 失效」靠调用方把 collection 名放进键里并用
        固定的第一个片段做命名空间（见 ``ChromaVectorStore``）。
        """
        with self._lock:
            doomed = [key for key in self._data if key.startswith(prefix)]
            for key in doomed:
                del self._data[key]
            return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    # ------------------------------------------------------------------ 统计

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                name=self.name,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                expirations=self._expirations,
                size=len(self._data),
                max_entries=self.max_entries,
                ttl_s=self.ttl_s,
            )

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0
