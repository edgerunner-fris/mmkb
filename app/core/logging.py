"""结构化日志与耗时打点。

两个目标：

1. **结构化**：容器里输出单行 JSON，本地开发输出人类可读的单行文本。同一个
   ``logging`` 调用两种格式，靠 ``APP_LOG_JSON`` 切换。
2. **可还原的耗时**：一次导入或一次问答会穿过十几个模块，出问题时必须能回答
   「慢在哪一步」。:class:`Timer` 负责收集每步毫秒数，最终原样进入
   ``debug_trace`` / ``Task.stage_timings``。

请求级上下文（``request_id`` / ``kb_id`` / ``session_id``）用 ``contextvars`` 传递，
不用一层层往函数签名里塞参数。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, TextIO

__all__ = [
    "JsonFormatter",
    "TextFormatter",
    "Timer",
    "bind_context",
    "clear_context",
    "current_context",
    "get_logger",
    "new_request_id",
    "setup_logging",
    "timed",
]

# --------------------------------------------------------------------------------------
# 上下文
# --------------------------------------------------------------------------------------

#: 默认值用不可变映射：上下文从来是整体替换（``_context.set(merged)``）而不是就地修改，
#: 一个可被误改的共享空 dict 只会带来「不同请求的日志串味」这种极难排查的问题。
_EMPTY_CONTEXT: Mapping[str, Any] = MappingProxyType({})

_context: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "mmkb_log_context", default=_EMPTY_CONTEXT
)

#: 会被自动附加到每条日志上的上下文字段
_CONTEXT_FIELDS = ("request_id", "kb_id", "doc_id", "task_id", "session_id", "stage")

_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATE_FORMAT = "%H:%M:%S"

#: LogRecord 的内建属性，用于把用户传入的 extra 挑出来
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def current_context() -> dict[str, Any]:
    return dict(_context.get())


def bind_context(**fields: Any) -> None:
    """把字段并入当前上下文。值是 ``None`` 的字段会被忽略。"""
    merged = dict(_context.get())
    merged.update({k: v for k, v in fields.items() if v is not None})
    _context.set(merged)


def clear_context() -> None:
    """退出一次请求/任务时调用，避免上下文在长驻进程里串味。"""
    _context.set({})


@contextmanager
def _scoped_context(**fields: Any) -> Iterator[None]:
    token = _context.set({**_context.get(), **{k: v for k, v in fields.items() if v}})
    try:
        yield
    finally:
        _context.reset(token)


# --------------------------------------------------------------------------------------
# Formatter
# --------------------------------------------------------------------------------------


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


def _payload(record: logging.LogRecord) -> dict[str, Any]:
    data: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
        + f".{int(record.msecs):03d}",
        "level": record.levelname,
        "logger": record.name,
        "msg": record.getMessage(),
    }
    data.update(_context.get())
    data.update(_extras(record))
    if record.exc_info:
        data["exc"] = logging.Formatter().formatException(record.exc_info)
    return data


class JsonFormatter(logging.Formatter):
    """单行 JSON。容器日志采集用这个。"""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(_payload(record), ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """人类可读单行：``12:00:01 INFO    app.ingestion.pipeline  解析完成 step=parse ms=812``"""

    def __init__(self) -> None:
        super().__init__(fmt=_TEXT_FORMAT, datefmt=_DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {**_context.get(), **_extras(record)}
        if not extras:
            return base
        tail = " ".join(f"{k}={_short(v)}" for k, v in extras.items())
        return f"{base}  {tail}"


def _short(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    stream: TextIO | None = None,
) -> None:
    """配置 root logger。重复调用会先清掉旧 handler，避免日志翻倍。"""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())

    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
        old.close()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # 这几个库在 INFO 级别噪音很大，压到 WARNING
    # jieba 在首次分词时打 DEBUG 级别的词典加载日志，且自带一个 handler，
    # 会绕开本项目的格式器直接往 stderr 打一行原文。压到 WARNING 最省事。
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "chromadb",
        "openai",
        "matplotlib",
        "jieba",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------------------------------------------
# 耗时打点
# --------------------------------------------------------------------------------------


class Timer:
    """收集一次任务里每一步的毫秒耗时。

    ::

        timer = Timer()
        with timer.step("parse"):
            blocks = parse(...)
        with timer.step("embed", count=len(chunks)):
            ...
        timer.timings      # {"parse": 812, "embed": 2130}
        timer.total_ms     # 2942
    """

    __slots__ = ("_timings", "_started")

    def __init__(self) -> None:
        self._timings: dict[str, int] = {}
        self._started = time.perf_counter()

    @contextmanager
    def step(self, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """计时一个阶段。阶段内可以往返回的 dict 里塞数据，退出时会一并打日志。"""
        box: dict[str, Any] = {}
        started = time.perf_counter()
        logger = get_logger("mmkb.timing")
        with _scoped_context(stage=name):
            try:
                yield box
            finally:
                elapsed = int((time.perf_counter() - started) * 1000)
                # 同名阶段累加（例如逐文档循环里的 parse）
                self._timings[name] = self._timings.get(name, 0) + elapsed
                logger.info(
                    "step finished",
                    extra={"step": name, "ms": elapsed, **fields, **box},
                )

    @property
    def timings(self) -> dict[str, int]:
        return dict(self._timings)

    @property
    def total_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {"stage_timings": self.timings, "total_ms": self.total_ms}


@contextmanager
def timed(logger: logging.Logger, label: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """轻量计时：只打一条日志，不收集结果。用于零散的外部调用。"""
    box: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        yield box
    finally:
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info(label, extra={"ms": elapsed, **fields, **box})
