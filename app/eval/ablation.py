"""五组 A/B 消融定义。

消融**不改全局配置**，只改每次检索的 ``routes`` 与 ``use_rerank``。
理由：``Settings`` 是进程级单例，临时改开关再改回来，一旦中途异常就会污染
后续所有请求；而 ``HybridRetriever.search`` 本来就接受这两个参数，
按请求覆盖既安全又能保证所有组跑在完全相同的知识库快照上。

五组对应 README 里的对比表：

====================  ====================================  ==========
组名                   召回路                                 重排
====================  ====================================  ==========
``bm25_only``          BM25                                  否
``dense_text_only``   文本向量                               否
``dense_image_only``  多模态向量                             否
``hybrid_rrf``        三路 + RRF                             否
``hybrid_rrf_rerank`` 三路 + RRF + 重排（线上默认配置）        是
====================  ====================================  ==========
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.schemas import Route

__all__ = [
    "PRESETS",
    "AblationError",
    "AblationPreset",
    "resolve_presets",
]


class AblationError(ValueError):
    """未知的消融组名。"""


@dataclass(frozen=True, slots=True)
class AblationPreset:
    name: str
    routes: tuple[Route, ...] | None
    use_rerank: bool
    description: str
    #: 以图搜图只依赖图像空间；不含 dense_image 的组不应参与该任务
    supports_image_query: bool

    @property
    def route_labels(self) -> list[str]:
        if self.routes is None:
            return [str(route) for route in (Route.BM25, Route.DENSE_TEXT, Route.DENSE_IMAGE)]
        return [str(route) for route in self.routes]

    @property
    def route_text(self) -> str:
        return " + ".join(self.route_labels)


PRESETS: dict[str, AblationPreset] = {
    "bm25_only": AblationPreset(
        name="bm25_only",
        routes=(Route.BM25,),
        use_rerank=False,
        description="纯稀疏检索，衡量关键词基线",
        supports_image_query=False,
    ),
    "dense_text_only": AblationPreset(
        name="dense_text_only",
        routes=(Route.DENSE_TEXT,),
        use_rerank=False,
        description="纯文本向量，衡量语义召回基线",
        supports_image_query=False,
    ),
    "dense_image_only": AblationPreset(
        name="dense_image_only",
        routes=(Route.DENSE_IMAGE,),
        use_rerank=False,
        description="纯多模态向量，衡量跨模态召回能力",
        supports_image_query=True,
    ),
    "hybrid_rrf": AblationPreset(
        name="hybrid_rrf",
        routes=None,
        use_rerank=False,
        description="三路召回 + RRF，衡量融合收益",
        supports_image_query=True,
    ),
    "hybrid_rrf_rerank": AblationPreset(
        name="hybrid_rrf_rerank",
        routes=None,
        use_rerank=True,
        description="三路 + RRF + 重排，线上默认配置",
        supports_image_query=True,
    ),
}

#: 报告与 README 都按这个顺序展示，避免 dict 顺序随改动漂移
PRESET_ORDER: tuple[str, ...] = (
    "bm25_only",
    "dense_text_only",
    "dense_image_only",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
)


def resolve_presets(raw: str | None) -> list[AblationPreset]:
    """把 ``all`` 或逗号分隔的组名解析成预设列表，保持调用方顺序。

    未知组名直接报错并列出合法值 —— 静默忽略会让报告少一行而没人发现。
    """
    if raw is None or not raw.strip() or raw.strip().lower() == "all":
        return [PRESETS[name] for name in PRESET_ORDER]

    names: list[str] = []
    for part in raw.split(","):
        name = part.strip()
        if name and name not in names:
            names.append(name)

    unknown = [name for name in names if name not in PRESETS]
    if unknown:
        raise AblationError(
            f"未知的消融组：{', '.join(unknown)}；可选值：{', '.join(PRESET_ORDER)}"
        )
    return [PRESETS[name] for name in names]
