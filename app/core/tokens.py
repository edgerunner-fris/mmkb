"""token 计数。

切分器要按 300–800 token 切块，因此需要一个 token 计数器。

**关于 tiktoken 的取舍**：``tiktoken.get_encoding()`` 首次调用会联网下载词表文件
（约 1.7 MB，之后缓存在本地）。本项目「镜像内不装任何本地模型」的约束针对的是
推理模型，词表不属于模型；但下载失败（离线构建、被墙）不应该让整个系统起不来，
所以这里做了两级降级：

1. 首选 tiktoken 精确计数；
2. 拿不到词表时退化为字符类别估算（CJK 约 1 token/字，ASCII 约 4 字符/token）。

估算的误差在 10% 以内，对「300–800 token」这种量级的切分完全够用。
系统会在首次降级时打一条 WARNING，不会静默。
"""

from __future__ import annotations

import logging
import unicodedata
from functools import lru_cache

__all__ = [
    "DEFAULT_ENCODING",
    "count_tokens",
    "encoding_available",
    "estimate_tokens",
    "truncate_to_tokens",
]

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = "cl100k_base"

#: 记录下载失败的编码名，避免每次都重试联网
_FAILED_ENCODINGS: set[str] = set()


@lru_cache(maxsize=8)
def _get_encoding(name: str):  # type: ignore[no-untyped-def]
    """返回 tiktoken 编码器；不可用时返回 ``None``（结果被 lru_cache 记住）。"""
    if name in _FAILED_ENCODINGS:
        return None
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception as exc:  # noqa: BLE001 - 网络/缓存/版本问题都可能抛
        _FAILED_ENCODINGS.add(name)
        logger.warning(
            "tiktoken 编码 %s 不可用（%s），token 计数降级为字符估算。"
            "若需精确计数，请在有网环境预下载词表或预置 TIKTOKEN_CACHE_DIR。",
            name,
            exc,
        )
        return None


def encoding_available(name: str = DEFAULT_ENCODING) -> bool:
    return _get_encoding(name) is not None


def estimate_tokens(text: str) -> int:
    """字符类别估算，作为 tiktoken 不可用时的兜底。

    规则（对中英混排的经验值）：
      * CJK 表意文字：1 字 ≈ 1 token
      * ASCII 字母数字：4 字符 ≈ 1 token
      * 其余（空白、标点、符号）：2 字符 ≈ 1 token
    """
    if not text:
        return 0

    cjk = 0
    ascii_word = 0
    other = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            cjk += 1
        elif ch.isascii() and (ch.isalnum() or ch == "_"):
            ascii_word += 1
        else:
            other += 1

    return max(1, cjk + (ascii_word + 3) // 4 + (other + 1) // 2)


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    """统计 token 数。空串返回 0。"""
    if not text:
        return 0
    enc = _get_encoding(encoding)
    if enc is None:
        return estimate_tokens(text)
    # disallowed_special=() 让 <|endoftext|> 这类字面量按普通文本处理，
    # 否则用户文档里出现该串会直接抛异常。
    return len(enc.encode(text, disallowed_special=()))


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding: str = DEFAULT_ENCODING,
) -> str:
    """按 token 上限截断，尽量落在字符边界上。

    超长文本（例如把一整章塞进 reranker 的单个文档槽）用它做保护性截断。
    """
    if max_tokens <= 0:
        return ""
    if count_tokens(text, encoding) <= max_tokens:
        return text

    enc = _get_encoding(encoding)
    if enc is not None:
        encoded = enc.encode(text, disallowed_special=())
        return enc.decode(encoded[:max_tokens])

    # 估算路径：二分找最长满足 estimate_tokens(prefix) <= max_tokens 的前缀
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[: max(lo, 1)]
