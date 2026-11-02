"""进程内性能指标登记表。

为什么不用 Prometheus
---------------------
这个项目的性能报告有两个使用场景：

1. **演示时打开页面就能看到** —— 一次问答、一次导入各花了多少毫秒，
   慢在检索、重排还是生成；
2. **压测脚本跑完立刻出 Markdown** —— 不依赖任何额外的 exporter / 存储。

Prometheus 适合长期留存与跨进程聚合，但它需要额外的 scrape 配置，
而本项目的指标全部属于「当前进程从启动到现在」这一个时间窗。
一个没有外部依赖、线程安全的环形缓冲 + 计数器就足够，也更容易解释清楚。

指标命名
--------
``observe(name, ms)`` 记的是**耗时样本**，``incr(name)`` 记的是**次数**。
两者分开：一次检索可能耗时 800ms（样本），但只算一次请求（计数）。
``snapshot()`` 输出每个指标的 count / error / p50 / p90 / p95 / p99 / max。

样本上限
--------
每个指标只保留最近 ``DEFAULT_MAX_SAMPLES`` 个样本（环形缓冲）。
这不是为了省内存，而是为了让报告回答「**现在**有多快」：
进程启动时的冷查询与一小时后的热查询混在一起，p95 会失真。
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "MetricSummary",
    "MetricsRegistry",
    "get_metrics",
    "reset_metrics",
]

#: 每个指标保留的最近样本数。2048 个 float 约 16KB，几十个指标也只有几 MB。
DEFAULT_MAX_SAMPLES = 2048


@dataclass(slots=True)
class MetricSummary:
    """一个指标的聚合结果。单位统一是毫秒（counter 除外）。"""

    name: str
    count: int = 0
    errors: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "errors": self.errors,
            "error_rate": round(self.error_rate, 4),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }


def _percentile(ordered: list[float], percent: float) -> float:
    """最近秩百分位（nearest-rank）。

    样本很少时（例如只跑了一次），p95 就等于最大值。这比线性插值更诚实：
    它不会用一个不存在的「中间值」让报告看起来更平滑。
    """
    if not ordered:
        return 0.0
    if percent <= 0:
        return ordered[0]
    if percent >= 100:
        return ordered[-1]
    rank = math.ceil(percent / 100.0 * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


@dataclass(slots=True)
class _Series:
    samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_MAX_SAMPLES)
    )
    count: int = 0
    errors: int = 0


class MetricsRegistry:
    """线程安全的耗时样本与计数器登记表。

    加锁的原因不是「异步代码会多线程并发」，而是图片检查、哈希、BM25 打分
    都跑在 ``asyncio.to_thread`` 里；只要有一个调用点在线程里记指标，
    就必须保证环形缓冲不会被两个线程同时改坏。
    """

    def __init__(self, *, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        self._max_samples = max(1, int(max_samples))
        self._series: dict[str, _Series] = {}
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()
        self._started_monotonic = time.monotonic()
        self._started_at = datetime.now(UTC)

    # ------------------------------------------------------------------ 写入

    def observe(self, name: str, ms: float, *, error: bool = False) -> None:
        """记录一个耗时样本。负值按 0 处理（计时器精度问题不应制造负延迟）。"""
        value = max(0.0, float(ms))
        with self._lock:
            series = self._series.get(name)
            if series is None:
                series = _Series(
                    samples=deque(maxlen=self._max_samples),
                )
                self._series[name] = series
            series.samples.append(value)
            series.count += 1
            if error:
                series.errors += 1

    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + int(value)

    def reset(self) -> None:
        with self._lock:
            self._series.clear()
            self._counters.clear()
            self._started_monotonic = time.monotonic()
            self._started_at = datetime.now(UTC)

    # ------------------------------------------------------------------ 读取

    def summary(self, name: str) -> MetricSummary | None:
        with self._lock:
            series = self._series.get(name)
            if series is None or not series.samples:
                return None
            values = sorted(series.samples)
            count, errors = series.count, series.errors
        return MetricSummary(
            name=name,
            count=count,
            errors=errors,
            min_ms=values[0],
            max_ms=values[-1],
            mean_ms=sum(values) / len(values),
            p50_ms=_percentile(values, 50),
            p90_ms=_percentile(values, 90),
            p95_ms=_percentile(values, 95),
            p99_ms=_percentile(values, 99),
        )

    def snapshot(self) -> dict[str, Any]:
        """完整快照。指标按名字排序，保证两次调用之间输出稳定。"""
        with self._lock:
            names = sorted(self._series)
            raw = {
                name: (
                    sorted(series.samples),
                    series.count,
                    series.errors,
                )
                for name, series in self._series.items()
                if series.samples
            }
            counters = dict(sorted(self._counters.items()))
            started_at = self._started_at
            started = self._started_monotonic

        metrics: dict[str, Any] = {}
        for name in names:
            if name not in raw:
                continue
            values, count, errors = raw[name]
            metrics[name] = MetricSummary(
                name=name,
                count=count,
                errors=errors,
                min_ms=values[0],
                max_ms=values[-1],
                mean_ms=sum(values) / len(values),
                p50_ms=_percentile(values, 50),
                p90_ms=_percentile(values, 90),
                p95_ms=_percentile(values, 95),
                p99_ms=_percentile(values, 99),
            ).to_dict()

        return {
            "started_at": started_at.isoformat(timespec="seconds"),
            "uptime_s": int(time.monotonic() - started),
            "window_samples": self._max_samples,
            "metrics": metrics,
            "counters": counters,
        }


_METRICS = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    """进程级单例。

    指标不是业务状态，不需要（也不应该）由依赖注入容器持有：
    provider、store、pipeline 都可能需要记一笔，而它们并不都拿得到容器。
    """
    return _METRICS


def reset_metrics() -> None:
    """测试与演示用：清空所有样本和计数器。"""
    _METRICS.reset()
