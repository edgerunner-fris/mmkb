"""终端着色。

**为什么需要这个模块**：Windows 控制台默认不开 VT 处理，
``print("\\033[31mxxx\\033[0m")`` 打出来的是字面量 ``[31mxxx[0m`` —— 比不着色更难看。
所以在脚本入口调一次 :func:`enable_ansi`，它尝试为 stdout 打开
``ENABLE_VIRTUAL_TERMINAL_PROCESSING``；打不开就老实地返回 ``False``，
由 :func:`paint` 原样透传文本。

这样同一份脚本在 Windows Terminal / PowerShell / Linux / 重定向到文件
四种情况下都能正确输出：前两者着色，后两者是干净纯文本。
"""

from __future__ import annotations

import os
import sys
from typing import Any, TextIO

__all__ = ["paint", "enable_ansi", "supports_color"]

_CODES: dict[str, str] = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[90m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
}

#: 终态缓存：探测过一次就不再重复走 ctypes（每个 print 都探一次会明显拖慢输出）。
_probed: bool = False
_enabled: bool = False

_STD_OUTPUT_HANDLE = -11
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


def _enable_windows_vt() -> bool:
    """尝试为当前控制台打开 VT 处理。失败返回 False（不是异常）。"""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:  # 非 Windows 或精简运行时
        return False

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
        # INVALID_HANDLE_VALUE 是 -1，0 表示没有控制台
        if handle in (0, -1):
            return False
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & _ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return True
        return bool(
            kernel32.SetConsoleMode(
                handle, mode.value | _ENABLE_VIRTUAL_TERMINAL_PROCESSING
            )
        )
    except OSError:
        return False


def enable_ansi(stream: TextIO | None = None, *, force: bool = False) -> bool:
    """探测并启用 ANSI 着色。

    返回最终是否着色。**幂等**：首次调用后结果被缓存，后续调用直接返回缓存值。

    :param stream: 目标流，默认 ``sys.stdout``
    :param force: 跳过探测直接开启（给已知支持 ANSI 的环境用，例如 CI 的彩色日志）
    """
    global _probed, _enabled
    if _probed and not force:
        return _enabled

    _probed = True
    target = stream if stream is not None else sys.stdout

    if os.environ.get("NO_COLOR"):  # https://no-color.org/
        _enabled = False
        return _enabled
    if os.environ.get("FORCE_COLOR"):
        _enabled = True
        return _enabled
    if not (hasattr(target, "isatty") and target.isatty()):
        # 重定向到文件 / 被管道接走：输出纯文本，别往日志里塞转义码
        _enabled = False
        return _enabled

    _enabled = _enable_windows_vt() if sys.platform == "win32" else True
    return _enabled


def supports_color() -> bool:
    """当前是否着色。未调用过 :func:`enable_ansi` 时返回 False。"""
    return _enabled


def paint(text: Any, *styles: str) -> str:
    """套上 ANSI 样式；未启用着色时原样返回 ``str(text)``。

    ``styles`` 接受样式名（``"red"`` / ``"bold"``）或裸转义码，未知样式名原样拼上。
    """
    rendered = text if isinstance(text, str) else str(text)
    if not _enabled or not styles:
        return rendered
    prefix = "".join(_CODES.get(style, style) for style in styles)
    return f"{prefix}{rendered}{_CODES['reset']}"
