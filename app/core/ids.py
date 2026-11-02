"""ID 生成与校验，以及 Chroma collection 命名。

所有 ID 都是 ``{prefix}_{12位小写hex}``，前缀让人一眼看出类型，
在日志和 debug_trace 里不用回表就能读懂。
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Final, Literal

__all__ = [
    "COLLECTION_SEP",
    "ID_PREFIXES",
    "IdKind",
    "image_collection",
    "is_valid_id",
    "new_id",
    "parse_kind",
    "stable_id",
    "text_collection",
]

IdKind = Literal["kb", "doc", "chunk", "image", "task", "session", "message"]

#: ID 类型 → 前缀。``chunk`` 用 ``ck`` 是为了和 ``collection`` 的惯用缩写区分开。
ID_PREFIXES: Final[dict[str, str]] = {
    "kb": "kb",
    "doc": "doc",
    "chunk": "ck",
    "image": "img",
    "task": "task",
    "session": "sess",
    "message": "msg",
}

_ID_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<prefix>[a-z]{2,5})_(?P<body>[0-9a-f]{12})$")
_PREFIX_TO_KIND: Final[dict[str, str]] = {v: k for k, v in ID_PREFIXES.items()}

#: Chroma collection 名里文本空间与图像空间的分隔符。
#: 用双下划线是为了**可逆解析**：kb_id 本身不含下划线，``split("__")`` 一定恰好两段。
COLLECTION_SEP: Final[str] = "__"


def new_id(kind: IdKind) -> str:
    """生成 ``{prefix}_{12hex}``。12 位 hex 约 2^48 空间，单机规模下碰撞可忽略。"""
    try:
        prefix = ID_PREFIXES[kind]
    except KeyError as exc:  # pragma: no cover - 传错类型属于编程错误
        raise ValueError(
            f"未知的 ID 类型 {kind!r}，可用：{sorted(ID_PREFIXES)}"
        ) from exc
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def stable_id(kind: IdKind, *parts: str) -> str:
    """由输入内容派生**确定性** ID：同样的输入永远得到同样的 ID。

    与 :func:`new_id` 的分工：随机的用于「一次创建、以后只引用」的实体
    （知识库、会话）；确定性的用于「同一份内容重跑导入不该产生新记录」的实体
    （分块、图片）。

    分隔符用 ``:``，并依赖所有 part 都不含 ``:``（项目里的 part 只有
    kb_id / doc_id / sha256 / 序号，都不含）。否则 ``("a:b", "c")`` 与
    ``("a", "b:c")`` 会撞到同一个 ID。
    """
    digest = hashlib.sha1(":".join(parts).encode("utf-8")).hexdigest()
    return f"{ID_PREFIXES[kind]}_{digest[:12]}"


def is_valid_id(value: str, kind: IdKind | None = None) -> bool:
    """校验格式；给了 ``kind`` 就同时校验前缀类型。"""
    if not isinstance(value, str):
        return False
    m = _ID_RE.match(value)
    if m is None:
        return False
    if kind is None:
        return True
    return m.group("prefix") == ID_PREFIXES.get(kind)


def parse_kind(value: str) -> str | None:
    """从 ID 反推类型；格式不合法返回 ``None``。"""
    m = _ID_RE.match(value)
    if m is None:
        return None
    return _PREFIX_TO_KIND.get(m.group("prefix"))


# --------------------------------------------------------------------------------------
# 向量空间隔离
# --------------------------------------------------------------------------------------


def text_collection(kb_id: str) -> str:
    """文本空间 collection：正文块向量 + 图片 caption 向量。

    caption 也进文本空间，是为了让图片既能被跨模态检索命中，
    也能被普通的文本检索/BM25 命中 —— 这是双模态召回的基础。
    """
    _require_kb_id(kb_id)
    return f"{kb_id}{COLLECTION_SEP}text"


def image_collection(kb_id: str) -> str:
    """图像空间 collection：qwen3-vl-embedding 的融合空间图像向量。

    ⚠️ 这个空间的向量**只能**由多模态端点产生，且入库与查询的
    ``enable_fusion`` 必须一致。绝不可把 text-embedding-v4 的向量写进来。
    """
    _require_kb_id(kb_id)
    return f"{kb_id}{COLLECTION_SEP}image"


def _require_kb_id(kb_id: str) -> None:
    if not is_valid_id(kb_id, "kb"):
        raise ValueError(f"非法 kb_id：{kb_id!r}（期望 kb_ + 12 位 hex）")
