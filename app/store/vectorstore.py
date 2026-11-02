"""向量库抽象接口。

**向量空间的铁律在这里第二次落地。** 第一次在 ``app/providers/base.py``
（provider 只能产出自己那个空间的向量），这里是第二次（存储层按空间分 collection，
跨空间写入直接报错）。

铁律 1  文本查文本：query 用 text-embedding-v4 编码 → 检索 ``{kb_id}__text``
铁律 2  以文搜图：query 文本送进 qwen3-vl-embedding 原生端点（text-only 输入），
        ``enable_fusion`` 与图片入库时完全一致 → 检索 ``{kb_id}__image``
铁律 3  禁止混用：不得用 text-embedding-v4 的 query 向量查图像库；
        不得用 qwen3-vl-embedding 的向量查纯文本库。
        两者不是同一个向量空间，直接算余弦相似度是错的。

三维度防线
----------
1. **collection 命名**：``{kb_id}__text`` / ``{kb_id}__image`` 后缀由
   :func:`text_collection` / :func:`image_collection` 生成，业务代码不拼字符串。
2. **collection 元数据记 dim 与 space**：写入 / 查询前比对，
   不一致抛 :class:`DimensionMismatchError`，错误信息里带上「配置值 vs 实际值」。
3. **空间守卫**：:meth:`VectorStore.upsert` 接受 ``space=`` 参数，
   与 collection 建立时记录的空间不符则抛 :class:`VectorSpaceMismatchError`。

第 3 条看起来是多余的（collection 名字里已经带了后缀），
但它拦的是「把 1024 维文本向量写进图像 collection」这种事 ——
如果两个空间的维度恰好相同，光靠维度检查是发现不了的。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.errors import MMKBError, VectorDimensionMismatchError

__all__ = [
    "IMAGE_SUFFIX",
    "TEXT_SUFFIX",
    "CollectionInfo",
    "VectorDistanceMetric",
    "VectorHit",
    "VectorRecord",
    "VectorSpaceMismatchError",
    "VectorStore",
    "distance_to_score",
    "image_collection",
    "is_mmkb_collection",
    "kb_id_from_collection",
    "parse_collection",
    "space_of_collection",
    "text_collection",
]

VectorSpace = Literal["text", "image"]

#: 距离度量。Chroma 里叫 ``hnsw:space``，pgvector 里叫 opclass，
#: 但两者的三个取值和数学含义是同一套，所以共用一个类型。
VectorDistanceMetric = Literal["cosine", "l2", "ip"]

TEXT_SUFFIX = "__text"
IMAGE_SUFFIX = "__image"

#: collection 名的合法字符集。Chroma 对名字有长度与字符限制
#: （3–512 字符，只允许字母数字和 ``._-``），这里提前收敛到一个更窄的子集，
#: 免得用户取个中文知识库名就撞上 Chroma 的报错。
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}(__text|__image)$")


class VectorSpaceMismatchError(MMKBError):
    """往文本空间写图像向量，或反过来。

    这是铁律 3 在存储层的兜底。provider 层有一个语义相同的
    :class:`~app.core.errors.ProviderUnsupportedOperationError`
    （「这个 provider 产不出那个空间的向量」）；本异常管的是另一头 ——
    向量已经有了，但被送到了错误的 collection。

    两者刻意不共用继承关系：修复方式完全不同，
    一个要换 provider 调用，一个要改 collection 名。
    """

    code = "vector_space_mismatch"
    http_status = 500
    retryable = False

    def __init__(self, *, expected: str, actual: str, collection: str) -> None:
        super().__init__(
            f"向量空间不匹配：collection {collection!r} 是 {expected!r} 空间，"
            f"但收到 {actual!r} 空间的向量。"
            "文本向量与多模态向量不是同一个空间，混用会得到无意义的相似度。",
            detail={
                "collection": collection,
                "expected_space": expected,
                "actual_space": actual,
            },
        )
        self.expected = expected
        self.actual = actual
        self.collection = collection


# --------------------------------------------------------------------------------------
# collection 命名
# --------------------------------------------------------------------------------------


def text_collection(kb_id: str) -> str:
    """``{kb_id}__text`` —— 只放 text-embedding-v4 产出的向量。

    内容是：文档分块向量 + 图片 caption 向量。
    caption 之所以也在这里，是因为它本身就是一段文本，
    这样「图片」天然能被文本检索命中，不需要第三套索引。
    """
    return _checked(f"{kb_id}{TEXT_SUFFIX}")


def image_collection(kb_id: str) -> str:
    """``{kb_id}__image`` —— 只放 qwen3-vl-embedding + enable_fusion 产出的向量。

    以文搜图时，query 必须用**同一模型、同一 enable_fusion** 编码后查这里。
    """
    return _checked(f"{kb_id}{IMAGE_SUFFIX}")


def _checked(name: str) -> str:
    if not _COLLECTION_RE.match(name):
        raise ValueError(
            f"非法的 collection 名 {name!r}："
            "只允许字母数字与 . _ -，且必须以 __text 或 __image 结尾"
        )
    return name


def parse_collection(name: str) -> tuple[str, VectorSpace] | None:
    """``"kb_abc__image"`` → ``("kb_abc", "image")``；解析不了返回 ``None``。"""
    for suffix, space in ((IMAGE_SUFFIX, "image"), (TEXT_SUFFIX, "text")):
        if name.endswith(suffix):
            kb_id = name[: -len(suffix)]
            if kb_id:
                return kb_id, space  # type: ignore[return-value]
    return None


def kb_id_from_collection(name: str) -> str | None:
    parsed = parse_collection(name)
    return parsed[0] if parsed else None


def space_of_collection(name: str) -> VectorSpace | None:
    parsed = parse_collection(name)
    return parsed[1] if parsed else None


def is_mmkb_collection(name: str) -> bool:
    """别的程序也用同一个 Chroma 目录时，靠这个把我们的 collection 挑出来。"""
    return parse_collection(name) is not None


# --------------------------------------------------------------------------------------
# 距离 → 分数
# --------------------------------------------------------------------------------------


def distance_to_score(distance: float, metric: str) -> float:
    """把「距离」换算成「越大越相关」的分数。

    **这一步不能省，也不能想当然。** 向量库返回的是距离，不是相似度，
    而且不同度量的换算公式不同：

    ==========  ==========================  ====================
    metric      含义                        换算
    ==========  ==========================  ====================
    cosine      ``1 - cos_sim``             ``score = 1 - d``
    l2          欧氏距离（未开方）             ``score = 1 / (1 + d)``
    ip          内积取负                     ``score = -d``
    ==========  ==========================  ====================

    把距离当相似度用，会让「最不相关」排到最前面，而且不会有任何报错 ——
    这类错误只能靠肉眼看检索结果才能发现。两个后端共用这一个实现，
    避免出现「Chroma 上对、PG 上错」这种只在切换后端时才暴露的问题。

    l2 刻意用 ``1/(1+d)`` 而不是取负：取负会得到负数，
    而 ``RETRIEVAL_MIN_SCORE_AFTER_RERANK`` 这类分数阈值是按「分数在 0 附近」
    来设的，混入负数会让阈值的含义变得难以解释。
    """
    if metric == "cosine":
        # cosine distance ∈ [0, 2] → 相似度 ∈ [-1, 1]
        return 1.0 - distance
    if metric == "l2":
        return 1.0 / (1.0 + max(0.0, distance))
    if metric == "ip":
        # pgvector 的 <#> 与 Chroma 的 ip 距离都是「取负的内积」
        return -distance
    return 1.0 - distance


# --------------------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class VectorRecord:
    """要写进向量库的一条记录。

    ``document`` 是 Chroma 的 document 字段：对 ``__text`` 是分块原文，
    对 ``__image`` 是 caption 文本（图谱回显与重排都要用）。
    **它不是 metadata** —— metadata 只放可过滤的标量，放长文本会让过滤变慢。
    """

    id: str
    vector: list[float]
    document: str = ""
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("VectorRecord.id 不能为空")
        if not self.vector:
            raise ValueError(f"VectorRecord({self.id}) 的 vector 为空")
        # Chroma 的 metadata 只接受标量，这里早一点拦住，
        # 免得错误在 to_thread 里抛出、堆栈指向 chromadb 内部
        for key, value in self.metadata.items():
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise TypeError(
                    f"VectorRecord({self.id}).metadata[{key!r}] 是 "
                    f"{type(value).__name__}，Chroma 只接受 str/int/float/bool；"
                    "列表与字典请先 json.dumps"
                )


@dataclass(slots=True)
class VectorHit:
    """一次向量查询的命中。``score`` 越大越相关。"""

    id: str
    score: float
    document: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0

    @property
    def chunk_id(self) -> str | None:
        value = self.metadata.get("chunk_id")
        return str(value) if value is not None else None

    @property
    def image_id(self) -> str | None:
        value = self.metadata.get("image_id")
        return str(value) if value is not None else None


@dataclass(slots=True)
class CollectionInfo:
    """collection 的现状。``dim`` / ``space`` 来自建立时写入的元数据。"""

    name: str
    count: int = 0
    dim: int | None = None
    space: VectorSpace | None = None
    embed_model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.dim is not None


# --------------------------------------------------------------------------------------
# 抽象接口
# --------------------------------------------------------------------------------------


class VectorStore(ABC):
    """向量库后端。

    **所有方法都是 async**，即使 Chroma 的客户端是同步的 ——
    实现方负责把它丢进线程池（``asyncio.to_thread``）。
    接口上不同步、实现里再看情况，是为了让业务代码在换后端时不用改调用方式。
    """

    #: 后端标识，用于日志与 ``/api/health``
    backend: str = "abstract"

    # ------------------------------------------------------------------ 生命周期

    @abstractmethod
    async def ensure_collection(
        self,
        name: str,
        *,
        dim: int,
        space: VectorSpace,
        embed_model: str = "",
        recreate: bool = False,
    ) -> CollectionInfo:
        """建 collection（幂等）。

        ``recreate=True`` 会先删再建 —— 只在「重建索引」这类显式操作里用。
        已存在且 ``dim`` / ``space`` 与参数不符时**抛错而不是覆盖**：
        静默覆盖等于把整个库的向量悄悄作废，这是最难发现的一类事故。
        """

    @abstractmethod
    async def get_collection_info(self, name: str) -> CollectionInfo | None:
        """返回 ``None`` 表示 collection 不存在。"""

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """删除整个 collection。返回是否真的删掉了。"""

    @abstractmethod
    async def list_collections(self, *, only_mmkb: bool = True) -> list[str]:
        """列出 collection。``only_mmkb=True`` 时只返回 ``*__text`` / ``*__image``。"""

    # ------------------------------------------------------------------ 写

    @abstractmethod
    async def upsert(
        self,
        name: str,
        records: list[VectorRecord],
        *,
        space: VectorSpace | None = None,
    ) -> int:
        """按 id upsert（已存在则覆盖）。返回写入条数。

        upsert 而非 add：导入失败重跑时，同一批 id 会再写一次，
        ``add`` 会因主键冲突报错，而 ``upsert`` 让整个导入链路天然幂等。
        """

    @abstractmethod
    async def delete(
        self,
        name: str,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> int:
        """按 id 或元数据条件删除。返回删除条数。``ids`` 与 ``where`` 至少给一个。"""

    # ------------------------------------------------------------------ 读

    @abstractmethod
    async def query(
        self,
        name: str,
        vector: list[float],
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        space: VectorSpace | None = None,
    ) -> list[VectorHit]:
        """向量检索，返回按 ``score`` 降序的结果。

        ``space`` 传了就必须与 collection 的空间一致 —— 这是铁律 3 的运行时兜底。
        """

    @abstractmethod
    async def get(self, name: str, ids: list[str]) -> list[VectorRecord]:
        """按 id 取回向量与元数据。**按传入 ids 的顺序**返回，缺失的跳过。"""

    @abstractmethod
    async def count(self, name: str) -> int:
        """collection 里的记录数。不存在返回 0。"""

    # ------------------------------------------------------------------ 关闭

    async def aclose(self) -> None:  # noqa: B027 - 有意的空默认实现
        """释放连接。默认什么都不做（Chroma 是本地文件，没有连接池）。"""

    # ------------------------------------------------------------------ 共用工具

    @staticmethod
    def guard_dim(vectors: list[list[float]], expected: int | None, *, collection: str) -> None:
        """校验一批向量的维度。``expected`` 为 ``None`` 时跳过。

        空向量单独报错：它不会撞上维度检查（``len([]) == 0`` 只有在 expected=0
        时才相等），但写进去之后所有相似度都会变成 0 或 NaN，
        而且不会报任何错 —— 属于最难查的一类问题。
        """
        if expected is None:
            return
        for index, vector in enumerate(vectors):
            if not vector:
                raise VectorDimensionMismatchError(
                    collection=collection,
                    expected=expected,
                    actual=0,
                    hint="空向量通常意味着上游把「编码失败」当成了「结果为空」返回。",
                )
            if len(vector) != expected:
                raise VectorDimensionMismatchError(
                    collection=collection,
                    expected=expected,
                    actual=len(vector),
                    hint=f"第 {index} 条记录。",
                )

    @staticmethod
    def guard_space(
        declared: VectorSpace | None, actual: VectorSpace | None, collection: str
    ) -> None:
        """铁律 3 的运行时检查。两边都有值且不同才报错。"""
        if declared is None or actual is None:
            return
        if declared != actual:
            raise VectorSpaceMismatchError(
                expected=actual, actual=declared, collection=collection
            )
