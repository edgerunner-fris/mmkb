"""全局配置中心。

设计要点
--------
1. 每个子系统一个 ``BaseSettings`` 子类，各自带独立的 ``env_prefix``，
   于是 ``TEXT_EMBED_MODEL`` / ``IMAGE_EMBED_MODEL`` 这类同名字段不会互相污染。
2. 顶层 :class:`Settings` 只做组合，不持有任何自己的环境变量
   （它的前缀被设成 ``MMKB_ROOT_``，确保不可能误匹配到真实变量）。
3. 厂商默认值（base_url / model / endpoint path）以「可被环境变量覆盖的默认值」形式
   集中在模块常量里，业务代码与 provider 实现一律读 ``resolved_*`` 属性，
   因此 **换模型、换厂商只改 .env，不改代码**。
4. 配置问题不抛在 import 期。:func:`collect_config_problems` 返回 (errors, warnings)，
   由 ``GET /api/health`` 汇报、由 ``lifespan`` 决定是否拒绝启动。
   这样 ``python -c "from app.config import get_settings"`` 在没有密钥时也能正常工作。

注意：本模块 **不得** import ``app`` 包内任何其他模块，保持零依赖、可单独导入。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------------------
# 路径与 .env 定位
# --------------------------------------------------------------------------------------

#: 项目根目录（app/config.py → app/ → 项目根）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

_ENV_FILES: tuple[str, ...] = (
    str(PROJECT_ROOT / ".env"),
    str(PROJECT_ROOT / ".env.local"),  # 后者优先级更高，用于本地覆盖
)


class ConfigurationError(RuntimeError):
    """配置缺失或自相矛盾。由 lifespan / /api/health 抛出并汇报。"""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        detail = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"配置校验未通过（{len(self.problems)} 项）：\n{detail}")


def _cfg(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        env_prefix=prefix,
        case_sensitive=False,
        extra="ignore",
    )


def _has_secret(value: SecretStr) -> bool:
    return bool(value.get_secret_value().strip())


def _parse_int_set(raw: str) -> set[int]:
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def _parse_float_list(raw: str) -> list[float]:
    return [float(p.strip()) for p in raw.split(",") if p.strip()]


def _blank_to_none(value: Any) -> Any:
    """把 ``.env`` 里的空值（``KEY=``）还原成 ``None``。

    这是本项目踩过的一个坑：对 ``Path | None`` 字段，pydantic 会把空字符串
    强转成 ``Path('.')``。它不是 ``None``，于是 ``override or default`` 这类
    回落逻辑全部失效，所有「留空即自动派生」的路径会一起塌缩到当前工作目录。
    这里在解析前拦一道，把空白字符串统一变回 ``None``。
    """
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


#: 「留空即自动派生」的可选路径字段。所有 *Root / *Dir / *Path 配置项都用它，
#: 保证 ``ENV_VAR=``（空值）等价于「未配置」。
OptPath = Annotated[Path | None, BeforeValidator(_blank_to_none)]


# --------------------------------------------------------------------------------------
# 厂商默认值（可被 .env 覆盖；不含任何密钥）
# --------------------------------------------------------------------------------------

#: 文本向量 provider → (base_url, model)
TEXT_EMBED_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "text-embedding-v4"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "BAAI/bge-m3"),
}

#: 重排 provider → (base_url, path, model)
RERANK_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "dashscope": (
        "https://dashscope.aliyuncs.com",
        "/api/v1/services/rerank/text-rerank/text-rerank",
        "gte-rerank-v2",
    ),
    "siliconflow": ("https://api.siliconflow.cn/v1", "/rerank", "BAAI/bge-reranker-v2-m3"),
}

#: 多模态 embedding 必须走 DashScope 原生端点，OpenAI 兼容模式不支持
IMAGE_EMBED_NATIVE_PATH = (
    "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
)


class _PathAware(BaseSettings):
    """把 .env 里的相对路径统一解析到项目根目录下，避免受 CWD 影响。"""

    @staticmethod
    def _resolve(p: Path | None) -> Path | None:
        if p is None:
            return None
        p = Path(p).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# --------------------------------------------------------------------------------------
# 1. 应用进程
# --------------------------------------------------------------------------------------


class AppSettings(BaseSettings):
    model_config = _cfg("APP_")

    env: Literal["dev", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = "*"
    static_url_prefix: str = "/static"
    api_prefix: str = "/api"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


# --------------------------------------------------------------------------------------
# 2. 生成模型
# --------------------------------------------------------------------------------------

ReasoningEffort = Literal["low", "high", "max", "none"]


class LLMSettings(BaseSettings):
    """DeepSeek（OpenAI 兼容 SDK）。``deepseek-flash`` 原生支持图像输入、思考模式、1M 上下文。

    ⚠️ 不支持视频输入，本项目不实现视频链路。
    ⚠️ DeepSeek 官方 API 没有 embedding 端点，向量一律走第 4/5 节的独立 provider。
    """

    model_config = _cfg("LLM_")

    provider: Literal["deepseek"] = "deepseek"
    api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"

    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)
    stream_timeout_s: float = Field(default=600.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_concurrency: int = Field(default=4, ge=1)
    rpm_limit: int = Field(default=0, ge=0)  # 0 = 不限流

    # 思考模式：low / high / max 为开启；none 为关闭（Anthropic 格式下用 none 关闭）
    reasoning_effort: ReasoningEffort = "high"
    # 是否把 reasoning_effort 发出去。取值 none 时本项自动失效（不发该参数即为关闭）。
    # 若某个 OpenAI 兼容网关不认这个参数，把它设 false 即可整体关闭，无需改代码。
    send_reasoning_effort: bool = True

    @property
    def effective_reasoning_effort(self) -> str | None:
        """实际要发出去的 reasoning_effort；None 表示不发送该参数。"""
        if not self.send_reasoning_effort or self.reasoning_effort == "none":
            return None
        return self.reasoning_effort

    @property
    def thinking_enabled(self) -> bool:
        return self.reasoning_effort != "none"

    @property
    def has_api_key(self) -> bool:
        return _has_secret(self.api_key)

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/")


class VisionSettings(BaseSettings):
    """图像理解（caption 生成）。默认整体继承 ``LLM_*``，需要独立模型时才覆盖。

    任意一项留空即回落到对应的 ``LLM_*``；这意味着可以只把 ``VISION_MODEL``
    指向另一个 OpenAI 兼容的视觉端点，而不用动生成模型。
    """

    model_config = _cfg("VISION_")

    enabled: bool = True
    provider: str = "deepseek"
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    model: str = ""

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1500, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_concurrency: int = Field(default=4, ge=1)

    max_image_side: int = Field(default=1600, ge=64)
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    reasoning_effort: ReasoningEffort = "low"

    @property
    def has_api_key(self) -> bool:
        return _has_secret(self.api_key)

    def resolve(self, llm: LLMSettings) -> ResolvedEndpoint:
        """把留空项回落到 LLM 配置，返回真正生效的三元组。"""
        return ResolvedEndpoint(
            provider=self.provider,
            api_key=self.api_key if self.has_api_key else llm.api_key,
            base_url=(self.base_url or llm.base_url).rstrip("/"),
            model=self.model or llm.model,
        )


class ResolvedEndpoint(BaseModel):
    """provider 解析后的实际生效配置（不含未解析的留空项）。

    刻意继承 ``BaseModel`` 而非 ``BaseSettings``：它是纯数据载体，
    所有字段都由 ``resolve()`` 显式填入，不应再去读环境变量。
    """

    provider: str = ""
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    path: str = ""
    model: str = ""

    @property
    def has_api_key(self) -> bool:
        return _has_secret(self.api_key)

    @property
    def endpoint(self) -> str:
        if not self.path:
            return self.base_url.rstrip("/")
        return self.base_url.rstrip("/") + "/" + self.path.lstrip("/")


# --------------------------------------------------------------------------------------
# 3. 文本向量
# --------------------------------------------------------------------------------------


class TextEmbeddingSettings(BaseSettings):
    """文本向量。DashScope compatible-mode 或 SiliconFlow，两者都是 OpenAI 兼容。"""

    model_config = _cfg("TEXT_EMBED_")

    enabled: bool = True
    provider: Literal["dashscope", "siliconflow"] = "dashscope"
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    model: str = ""

    dim: int = Field(default=1024, ge=1)
    allowed_dims: str = "2048,1536,1024,768,512,256"

    batch_size: int = Field(default=10, ge=1, le=256)
    max_concurrency: int = Field(default=3, ge=1)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    rpm_limit: int = Field(default=0, ge=0)
    normalize: bool = True
    truncate_chars: int = Field(default=6000, ge=1)

    @property
    def has_api_key(self) -> bool:
        return _has_secret(self.api_key)

    @property
    def allowed_dim_set(self) -> set[int]:
        return _parse_int_set(self.allowed_dims)

    @property
    def resolved(self) -> ResolvedEndpoint:
        default_url, default_model = TEXT_EMBED_PROVIDER_DEFAULTS[self.provider]
        return ResolvedEndpoint(
            provider=self.provider,
            api_key=self.api_key,
            base_url=(self.base_url or default_url).rstrip("/"),
            model=self.model or default_model,
        )

    @property
    def resolved_base_url(self) -> str:
        return self.resolved.base_url

    @property
    def resolved_model(self) -> str:
        return self.resolved.model


# --------------------------------------------------------------------------------------
# 4. 多模态向量（图像）
# --------------------------------------------------------------------------------------


class ImageEmbeddingSettings(BaseSettings):
    """图像向量。

    ⚠️ DashScope 的 OpenAI 兼容模式 **只支持文本 embedding**，多模态 embedding
       必须调用原生端点（见 ``IMAGE_EMBED_NATIVE_PATH``），不能用 OpenAI SDK 调。

    ⚠️ ``enable_fusion`` 在入库与查询时必须完全一致：图片入库用
       ``qwen3-vl-embedding + enable_fusion=true`` 生成向量存入 ``{kb_id}__image``，
       以文搜图时 query 文本也必须用 **同一个模型、同一个 enable_fusion**
       （text-only 输入）编码，才能落在同一融合空间。
    """

    model_config = _cfg("IMAGE_EMBED_")

    enabled: bool = True
    provider: Literal["dashscope"] = "dashscope"
    api_key: SecretStr = SecretStr("")
    base_url: str = "https://dashscope.aliyuncs.com"
    path: str = IMAGE_EMBED_NATIVE_PATH
    model: str = "qwen3-vl-embedding"

    dim: int = Field(default=2560, ge=1)
    allowed_dims: str = "2560,2048,1536,1024,768,512,256"
    enable_fusion: bool = True

    # ⚠️ enable_fusion=true 时端点会把整批 contents 融合成 1 个向量，
    # 只能逐条请求；只有 enable_fusion=false 才允许 >1。
    # 即使配成 >1，provider 也会强制按 1 发送并打警告。
    batch_size: int = Field(default=1, ge=1, le=64)
    max_concurrency: int = Field(default=2, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    rpm_limit: int = Field(default=0, ge=0)
    max_image_side: int = Field(default=1568, ge=64)
    normalize: bool = True

    @property
    def has_api_key(self) -> bool:
        return _has_secret(self.api_key)

    @property
    def allowed_dim_set(self) -> set[int]:
        return _parse_int_set(self.allowed_dims)

    @property
    def resolved(self) -> ResolvedEndpoint:
        return ResolvedEndpoint(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url.rstrip("/"),
            path=self.path,
            model=self.model,
        )

    @property
    def endpoint(self) -> str:
        return self.resolved.endpoint


# --------------------------------------------------------------------------------------
# 5. 重排
# --------------------------------------------------------------------------------------


class RerankSettings(BaseSettings):
    """重排。

    ⚠️ 已知限制：gte-rerank / bge-reranker 这类模型 **只接受纯文本输入，无法理解图像**。
       因此图片类候选必须用其 caption 文本 + 章节路径参与重排，
       模板见 ``image_doc_template``。这是本项目明确记录的取舍，
       它带来的精度损失由 Batch 8 的 A/B 消融（hybrid_rrf vs hybrid_rrf_rerank）量化。
    """

    model_config = _cfg("RERANK_")

    enabled: bool = True
    provider: Literal["dashscope", "siliconflow"] = "dashscope"
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    path: str = ""
    model: str = ""

    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_docs_per_request: int = Field(default=50, ge=1, le=500)
    max_chars_per_doc: int = Field(default=4000, ge=64)
    rpm_limit: int = Field(default=0, ge=0)

    image_doc_template: str = "[图片] {heading} | {caption}"

    @property
    def has_api_key(self) -> bool:
        return _has_secret(self.api_key)

    @property
    def resolved(self) -> ResolvedEndpoint:
        default_url, default_path, default_model = RERANK_PROVIDER_DEFAULTS[self.provider]
        return ResolvedEndpoint(
            provider=self.provider,
            api_key=self.api_key,
            base_url=(self.base_url or default_url).rstrip("/"),
            path=self.path or default_path,
            model=self.model or default_model,
        )

    @property
    def endpoint(self) -> str:
        return self.resolved.endpoint

    @property
    def resolved_model(self) -> str:
        return self.resolved.model


# --------------------------------------------------------------------------------------
# 6. 存储
# --------------------------------------------------------------------------------------


class VectorStoreSettings(BaseSettings):
    model_config = _cfg("VECTORSTORE_")

    backend: Literal["chroma", "pg"] = "chroma"
    chroma_dir: OptPath = None

    #: 距离度量。**两个后端共用** —— 「余弦相似度」在 Chroma 和 pgvector 里
    #: 是同一个数学对象，只是前者叫 hnsw:space、后者叫 opclass。
    #: 拆成两个配置项只会制造「换了后端分数含义就变了」这种事故。
    metric: Literal["cosine", "l2", "ip"] = "cosine"
    #: 兼容最初的名字。留空则用 ``metric``；填了就以它为准（老 .env 不必改）。
    chroma_space: Literal["cosine", "l2", "ip"] | None = None

    pg_dsn: str = ""
    pg_table_prefix: str = "mmkb"
    pg_pool_size: int = Field(default=5, ge=1)

    @property
    def effective_metric(self) -> Literal["cosine", "l2", "ip"]:
        """实际生效的距离度量。取值的唯一入口 —— 两个后端都必须用它。"""
        return self.chroma_space or self.metric


class DocStoreSettings(_PathAware):
    model_config = _cfg("DOCSTORE_")

    sqlite_path: OptPath = None
    echo_sql: bool = False

    @property
    def resolved_path(self) -> Path | None:
        return self._resolve(self.sqlite_path)


class BlobSettings(_PathAware):
    """图片过滤与去重阈值。phash 用 numpy 手写 DCT 实现，不引入 imagehash/scipy。"""

    model_config = _cfg("BLOB_")

    images_dir: OptPath = None
    min_width: int = Field(default=100, ge=1)
    min_height: int = Field(default=100, ge=1)
    phash_size: int = Field(default=8, ge=4, le=16)
    phash_hamming_threshold: int = Field(default=5, ge=0, le=64)
    blank_std_threshold: float = Field(default=6.0, ge=0.0)
    min_unique_color_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    dedupe_scope: Literal["doc", "kb"] = "kb"
    max_images_per_doc: int = Field(default=0, ge=0)  # 0 = 不限
    convert_to_png: bool = False
    jpeg_quality: int = Field(default=90, ge=1, le=100)

    @property
    def resolved_dir(self) -> Path | None:
        return self._resolve(self.images_dir)


# --------------------------------------------------------------------------------------
# 7. 切分 / 检索 / 生成
# --------------------------------------------------------------------------------------


class ChunkingSettings(BaseSettings):
    model_config = _cfg("CHUNK_")

    min_tokens: int = Field(default=300, ge=1)
    max_tokens: int = Field(default=800, ge=1)
    overlap_tokens: int = Field(default=100, ge=0)
    encoding: str = "cl100k_base"
    table_max_tokens: int = Field(default=1500, ge=1)
    table_repeat_header: bool = True
    keep_code_atomic: bool = True
    keep_table_atomic: bool = True
    merge_small_siblings: bool = True
    min_chunk_chars: int = Field(default=20, ge=1)


class RetrievalSettings(BaseSettings):
    """三路召回 + RRF 融合。

    前四个开关同时就是 A/B 消融的配置面：
    纯 BM25 / 纯文本向量 / 纯多模态向量 / 三路 RRF / 三路 RRF+重排。
    """

    model_config = _cfg("RETRIEVAL_")

    enable_bm25: bool = True
    enable_dense_text: bool = True
    enable_dense_image: bool = True
    enable_rerank: bool = True

    top_k_per_route: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    route_weights: str = "1.0,1.0,1.0"  # 顺序固定：bm25, dense_text, dense_image

    rerank_input_k: int = Field(default=50, ge=1, le=500)
    final_top_k: int = Field(default=8, ge=1, le=100)
    min_score_after_rerank: float = 0.0
    neighbor_expand: int = Field(default=0, ge=0, le=3)
    context_max_chars: int = Field(default=12000, ge=500)
    image_query_top_k: int = Field(default=10, ge=1)

    @property
    def route_weight_list(self) -> list[float]:
        return _parse_float_list(self.route_weights)


class BM25Settings(_PathAware):
    """稀疏检索。中文必须走 jieba，否则整句会退化成一个 token，BM25 路直接失效。"""

    model_config = _cfg("BM25_")

    tokenizer: Literal["auto", "jieba", "regex"] = "auto"
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    index_dir: OptPath = None
    persist: bool = True
    stopwords: str = ""
    min_token_len: int = Field(default=1, ge=1)

    @property
    def resolved_dir(self) -> Path | None:
        return self._resolve(self.index_dir)

    @property
    def stopword_list(self) -> list[str]:
        return [w.strip() for w in self.stopwords.split(",") if w.strip()]


class AgentSettings(BaseSettings):
    model_config = _cfg("AGENT_")

    max_rounds: int = Field(default=2, ge=1, le=5)
    relevance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    judge_enabled: bool = True
    judge_max_context_items: int = Field(default=8, ge=1)
    query_rewrite_enabled: bool = True
    query_rewrite_n: int = Field(default=3, ge=1, le=6)
    router_llm_enabled: bool = True
    smalltalk_skip_retrieval: bool = True
    history_max_turns: int = Field(default=6, ge=0)
    history_compress_enabled: bool = True
    history_compress_trigger_chars: int = Field(default=2000, ge=100)


class GenerationSettings(BaseSettings):
    model_config = _cfg("GEN_")

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    require_citations: bool = True
    refuse_when_empty: bool = True
    allow_model_knowledge: bool = True
    max_citations: int = Field(default=8, ge=1)
    snippet_chars: int = Field(default=300, ge=20)
    stream_heartbeat_s: float = Field(default=15.0, gt=0)


class IngestSettings(_PathAware):
    model_config = _cfg("INGEST_")

    max_file_mb: int = Field(default=200, ge=1)
    max_concurrent_docs: int = Field(default=1, ge=1, le=8)
    #: 单篇文档内图片检查（解码 + phash + 模糊判定）的并发度。
    #: 这部分是纯 CPU/IO、不耗外部配额，可以比 caption 并发更高。
    image_inspect_concurrency: int = Field(default=4, ge=1, le=32)
    keep_uploaded_file: bool = True
    upload_dir: OptPath = None

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def resolved_dir(self) -> Path | None:
        return self._resolve(self.upload_dir)


class CacheSettings(BaseSettings):
    """检索链路的进程内缓存。

    默认开启，但 TTL 都很短（分钟级）。缓存只服务于「同一 query 反复出现」
    与「collection 元信息被每次查询重复读取」这两类浪费；它不是正确性的一部分，
    把 ``CACHE_ENABLED=false`` 之后所有路径必须与未加缓存时完全一致。
    """

    model_config = _cfg("CACHE_")

    enabled: bool = True
    query_embedding_ttl_s: float = Field(default=600.0, ge=0.0)
    query_embedding_max_entries: int = Field(default=512, ge=1)
    rerank_ttl_s: float = Field(default=300.0, ge=0.0)
    rerank_max_entries: int = Field(default=256, ge=1)
    collection_meta_ttl_s: float = Field(default=30.0, ge=0.0)


class EvalSettings(_PathAware):
    model_config = _cfg("EVAL_")

    dataset_path: OptPath = None
    output_dir: OptPath = None
    recall_ks: str = "5,10"
    mrr_k: int = Field(default=10, ge=1)
    ndcg_k: int = Field(default=10, ge=1)
    use_ragas: bool = False
    ragas_metrics: str = "faithfulness,answer_relevancy,context_precision"
    max_questions: int = Field(default=0, ge=0)
    judge_model: str = ""
    ablation_presets: str = (
        "bm25_only,dense_text_only,dense_image_only,hybrid_rrf,hybrid_rrf_rerank"
    )

    @property
    def recall_k_list(self) -> list[int]:
        return sorted(_parse_int_set(self.recall_ks))

    @property
    def ragas_metric_list(self) -> list[str]:
        return [m.strip() for m in self.ragas_metrics.split(",") if m.strip()]

    @property
    def ablation_preset_list(self) -> list[str]:
        return [p.strip() for p in self.ablation_presets.split(",") if p.strip()]

    @property
    def resolved_dataset_path(self) -> Path | None:
        return self._resolve(self.dataset_path)

    @property
    def resolved_output_dir(self) -> Path | None:
        return self._resolve(self.output_dir)


class PathSettings(_PathAware):
    """所有落盘位置的唯一真相源。除 DATA_DIR 外全部可留空自动派生。"""

    model_config = _cfg("")

    data_dir: Path = PROJECT_ROOT / "data"
    raw_root: OptPath = None
    processed_root: OptPath = None
    images_root: OptPath = None
    chroma_root: OptPath = None
    eval_root: OptPath = None
    bm25_root: OptPath = None
    logs_root: OptPath = None
    sqlite_file: OptPath = None

    @property
    def root(self) -> Path:
        return self._resolve(self.data_dir) or (PROJECT_ROOT / "data")

    @property
    def raw_dir(self) -> Path:
        return self._resolve(self.raw_root) or (self.root / "raw")

    @property
    def processed_dir(self) -> Path:
        return self._resolve(self.processed_root) or (self.root / "processed")

    @property
    def images_dir(self) -> Path:
        return self._resolve(self.images_root) or (self.root / "images")

    @property
    def chroma_dir(self) -> Path:
        return self._resolve(self.chroma_root) or (self.root / "chroma")

    @property
    def eval_dir(self) -> Path:
        return self._resolve(self.eval_root) or (self.root / "eval")

    @property
    def bm25_dir(self) -> Path:
        return self._resolve(self.bm25_root) or (self.processed_dir / "bm25")

    @property
    def logs_dir(self) -> Path:
        return self._resolve(self.logs_root) or (self.root / "logs")

    @property
    def sqlite_path(self) -> Path:
        return self._resolve(self.sqlite_file) or (self.root / "mmkb.db")

    def all_dirs(self) -> list[Path]:
        return [
            self.root,
            self.raw_dir,
            self.processed_dir,
            self.images_dir,
            self.chroma_dir,
            self.eval_dir,
            self.bm25_dir,
            self.logs_dir,
        ]

    def ensure(self) -> list[Path]:
        """创建所有数据目录，返回实际创建的（不含已存在的）。"""
        created: list[Path] = []
        for d in self.all_dirs():
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                created.append(d)
        return created


# --------------------------------------------------------------------------------------
# 顶层组合
# --------------------------------------------------------------------------------------


class Settings(BaseSettings):
    """组合根。前缀设成 MMKB_ROOT_ 以确保不会误匹配到任何真实环境变量。"""

    model_config = _cfg("MMKB_ROOT_")

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    text_embed: TextEmbeddingSettings = Field(default_factory=TextEmbeddingSettings)
    image_embed: ImageEmbeddingSettings = Field(default_factory=ImageEmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    docstore: DocStoreSettings = Field(default_factory=DocStoreSettings)
    blob: BlobSettings = Field(default_factory=BlobSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    bm25: BM25Settings = Field(default_factory=BM25Settings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    paths: PathSettings = Field(default_factory=PathSettings)

    # ---- 派生的现实落点（把分散在子系统里的路径统一收敛到 paths）----

    @property
    def vision_endpoint(self) -> ResolvedEndpoint:
        return self.vision.resolve(self.llm)

    @property
    def chroma_dir(self) -> Path:
        return self.vectorstore.chroma_dir or self.paths.chroma_dir

    @property
    def images_dir(self) -> Path:
        return self.blob.resolved_dir or self.paths.images_dir

    @property
    def sqlite_path(self) -> Path:
        return self.docstore.resolved_path or self.paths.sqlite_path

    @property
    def bm25_dir(self) -> Path:
        return self.bm25.resolved_dir or self.paths.bm25_dir

    @property
    def upload_dir(self) -> Path:
        return self.ingest.resolved_dir or self.paths.raw_dir

    @property
    def eval_dataset_path(self) -> Path:
        return self.eval.resolved_dataset_path or (self.paths.eval_dir / "eval_set.jsonl")

    @property
    def eval_output_dir(self) -> Path:
        return self.eval.resolved_output_dir or self.paths.eval_dir

    def require_ready(self) -> None:
        """启动期调用。有致命配置问题时抛 :class:`ConfigurationError`。"""
        errors, _ = collect_config_problems(self)
        if errors:
            raise ConfigurationError(errors)

    def redacted(self) -> dict[str, Any]:
        """给 ``/api/health`` 与日志用的、不含任何密钥的配置快照。"""
        vision = self.vision.resolve(self.llm)
        return {
            "app": {
                "env": self.app.env,
                "log_level": self.app.log_level,
                "static_url_prefix": self.app.static_url_prefix,
                "api_prefix": self.app.api_prefix,
                "project_root": str(PROJECT_ROOT),
            },
            "llm": {
                "provider": self.llm.provider,
                "base_url": self.llm.endpoint,
                "model": self.llm.model,
                "reasoning_effort": self.llm.reasoning_effort,
                "thinking_enabled": self.llm.thinking_enabled,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "timeout_s": self.llm.timeout_s,
                "max_retries": self.llm.max_retries,
                "max_concurrency": self.llm.max_concurrency,
                "api_key_set": self.llm.has_api_key,
            },
            "vision": {
                "enabled": self.vision.enabled,
                "base_url": vision.base_url,
                "model": vision.model,
                "max_concurrency": self.vision.max_concurrency,
                "reasoning_effort": self.vision.reasoning_effort,
                "api_key_set": vision.has_api_key,
                "inherits_llm_key": not self.vision.has_api_key,
            },
            "text_embed": {
                "enabled": self.text_embed.enabled,
                "provider": self.text_embed.provider,
                "base_url": self.text_embed.resolved_base_url,
                "model": self.text_embed.resolved_model,
                "dim": self.text_embed.dim,
                "batch_size": self.text_embed.batch_size,
                "normalize": self.text_embed.normalize,
                "api_key_set": self.text_embed.has_api_key,
            },
            "image_embed": {
                "enabled": self.image_embed.enabled,
                "provider": self.image_embed.provider,
                "endpoint": self.image_embed.endpoint,
                "model": self.image_embed.model,
                "dim": self.image_embed.dim,
                "enable_fusion": self.image_embed.enable_fusion,
                "batch_size": self.image_embed.batch_size,
                "normalize": self.image_embed.normalize,
                "api_key_set": self.image_embed.has_api_key,
            },
            "rerank": {
                "enabled": self.rerank.enabled,
                "provider": self.rerank.provider,
                "endpoint": self.rerank.endpoint,
                "model": self.rerank.resolved_model,
                "max_docs_per_request": self.rerank.max_docs_per_request,
                "image_doc_template": self.rerank.image_doc_template,
                "api_key_set": self.rerank.has_api_key,
            },
            "vectorstore": {
                "backend": self.vectorstore.backend,
                "chroma_dir": str(self.chroma_dir),
                "metric": self.vectorstore.effective_metric,
                "pg_table_prefix": self.vectorstore.pg_table_prefix,
                "pg_dsn_set": bool(self.vectorstore.pg_dsn.strip()),
            },
            "docstore": {"sqlite_path": str(self.sqlite_path)},
            "blob": {
                "images_dir": str(self.images_dir),
                "min_size": [self.blob.min_width, self.blob.min_height],
                "phash_size": self.blob.phash_size,
                "phash_hamming_threshold": self.blob.phash_hamming_threshold,
                "blank_std_threshold": self.blob.blank_std_threshold,
                "dedupe_scope": self.blob.dedupe_scope,
            },
            "chunking": {
                "min_tokens": self.chunking.min_tokens,
                "max_tokens": self.chunking.max_tokens,
                "overlap_tokens": self.chunking.overlap_tokens,
                "encoding": self.chunking.encoding,
            },
            "retrieval": {
                "enable_bm25": self.retrieval.enable_bm25,
                "enable_dense_text": self.retrieval.enable_dense_text,
                "enable_dense_image": self.retrieval.enable_dense_image,
                "enable_rerank": self.retrieval.enable_rerank,
                "top_k_per_route": self.retrieval.top_k_per_route,
                "rrf_k": self.retrieval.rrf_k,
                "route_weights": self.retrieval.route_weights,
                "rerank_input_k": self.retrieval.rerank_input_k,
                "final_top_k": self.retrieval.final_top_k,
            },
            "bm25": {
                "tokenizer": self.bm25.tokenizer,
                "k1": self.bm25.k1,
                "b": self.bm25.b,
                "index_dir": str(self.bm25_dir),
                "persist": self.bm25.persist,
            },
            "agent": {
                "max_rounds": self.agent.max_rounds,
                "relevance_threshold": self.agent.relevance_threshold,
                "judge_enabled": self.agent.judge_enabled,
                "query_rewrite_enabled": self.agent.query_rewrite_enabled,
                "query_rewrite_n": self.agent.query_rewrite_n,
            },
            "generation": {
                "temperature": self.generation.temperature,
                "require_citations": self.generation.require_citations,
                "refuse_when_empty": self.generation.refuse_when_empty,
                "allow_model_knowledge": self.generation.allow_model_knowledge,
                "max_citations": self.generation.max_citations,
            },
            "ingest": {
                "max_file_mb": self.ingest.max_file_mb,
                "max_concurrent_docs": self.ingest.max_concurrent_docs,
                "image_inspect_concurrency": self.ingest.image_inspect_concurrency,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "query_embedding_ttl_s": self.cache.query_embedding_ttl_s,
                "query_embedding_max_entries": self.cache.query_embedding_max_entries,
                "rerank_ttl_s": self.cache.rerank_ttl_s,
                "rerank_max_entries": self.cache.rerank_max_entries,
                "collection_meta_ttl_s": self.cache.collection_meta_ttl_s,
            },
            # 显式列出，不用 d.name 推导：派生目录可能与根目录同名而被字典键吃掉
            "paths": {
                "data_dir": str(self.paths.root),
                "raw_dir": str(self.paths.raw_dir),
                "processed_dir": str(self.paths.processed_dir),
                "images_dir": str(self.paths.images_dir),
                "chroma_dir": str(self.paths.chroma_dir),
                "eval_dir": str(self.paths.eval_dir),
                "bm25_dir": str(self.paths.bm25_dir),
                "logs_dir": str(self.paths.logs_dir),
                "sqlite_path": str(self.paths.sqlite_path),
            },
        }


# --------------------------------------------------------------------------------------
# 配置校验
# --------------------------------------------------------------------------------------


def collect_config_problems(settings: Settings) -> tuple[list[str], list[str]]:
    """返回 ``(errors, warnings)``。

    errors  —— 致命，会导致 ``lifespan`` 拒绝启动、``/api/health`` 返回 not_ready；
    warnings —— 不致命，但可能意味着某条链路会降级，需要让使用者看见。
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ---------- 必填密钥 ----------
    if not settings.llm.has_api_key:
        errors.append("LLM_API_KEY 未配置：生成模型必填（https://platform.deepseek.com）")

    if settings.text_embed.enabled and not settings.text_embed.has_api_key:
        errors.append(
            "TEXT_EMBED_API_KEY 未配置：文本向量必填。"
            "若暂时不需要文本向量检索，可设 TEXT_EMBED_ENABLED=false"
        )
    if settings.image_embed.enabled and not settings.image_embed.has_api_key:
        errors.append(
            "IMAGE_EMBED_API_KEY 未配置：多模态向量必填。"
            "若暂时不需要图片检索，可设 IMAGE_EMBED_ENABLED=false"
        )
    if settings.rerank.enabled and not settings.rerank.has_api_key:
        errors.append("RERANK_API_KEY 未配置：可设 RERANK_ENABLED=false 关闭重排")
    if settings.vision.enabled and not (
        settings.vision.has_api_key or settings.llm.has_api_key
    ):
        errors.append(
            "VISION_API_KEY 与 LLM_API_KEY 均未配置：caption 生成需要其中之一，"
            "或设 VISION_ENABLED=false 降级为「前后文合成描述」"
        )

    if settings.vectorstore.backend == "pg" and not settings.vectorstore.pg_dsn.strip():
        errors.append("VECTORSTORE_BACKEND=pg 时必须配置 VECTORSTORE_PG_DSN")
    if settings.vectorstore.backend == "pg":
        try:
            import psycopg  # noqa: F401
        except ImportError:
            errors.append(
                "VECTORSTORE_BACKEND=pg 但未安装驱动，请执行：pip install -e \".[pg]\""
            )

    # ---------- 向量维度档位 ----------
    allowed_text = settings.text_embed.allowed_dim_set
    if settings.text_embed.dim not in allowed_text:
        errors.append(
            f"TEXT_EMBED_DIM={settings.text_embed.dim} 不在允许档位 {sorted(allowed_text)} 内"
        )
    allowed_image = settings.image_embed.allowed_dim_set
    if settings.image_embed.dim not in allowed_image:
        errors.append(
            f"IMAGE_EMBED_DIM={settings.image_embed.dim} "
            f"不在允许档位 {sorted(allowed_image)} 内"
        )
    if settings.text_embed.dim == settings.image_embed.dim:
        warnings.append(
            "文本向量与多模态向量维度恰好相同，但它们是不同模型的不同空间、"
            "分别落在 __text / __image，禁止跨空间比对余弦相似度"
        )

    # ---------- 切分 ----------
    if settings.chunking.min_tokens >= settings.chunking.max_tokens:
        errors.append(
            f"CHUNK_MIN_TOKENS({settings.chunking.min_tokens}) "
            f"必须小于 CHUNK_MAX_TOKENS({settings.chunking.max_tokens})"
        )
    if settings.chunking.overlap_tokens >= settings.chunking.max_tokens:
        errors.append(
            f"CHUNK_OVERLAP_TOKENS({settings.chunking.overlap_tokens}) "
            f"必须小于 CHUNK_MAX_TOKENS({settings.chunking.max_tokens})"
        )
    if settings.chunking.table_max_tokens < settings.chunking.max_tokens:
        warnings.append(
            f"CHUNK_TABLE_MAX_TOKENS({settings.chunking.table_max_tokens}) 小于 "
            f"CHUNK_MAX_TOKENS({settings.chunking.max_tokens})，表格块会比正文块更早被切开"
        )

    # ---------- 检索 ----------
    r = settings.retrieval
    if not (r.enable_bm25 or r.enable_dense_text or r.enable_dense_image):
        errors.append(
            "RETRIEVAL_ENABLE_BM25 / _DENSE_TEXT / _DENSE_IMAGE 三路全部关闭，检索链路无数据来源"
        )
    try:
        weights = r.route_weight_list
    except ValueError:
        weights = []
        errors.append(f"RETRIEVAL_ROUTE_WEIGHTS=\"{r.route_weights}\" 不是合法的逗号分隔数字")
    if weights and len(weights) != 3:
        errors.append(
            f"RETRIEVAL_ROUTE_WEIGHTS 必须是 3 个数字（bm25, dense_text, dense_image），"
            f"当前 {len(weights)} 个"
        )
    if r.final_top_k > r.rerank_input_k:
        warnings.append(
            f"RETRIEVAL_FINAL_TOP_K({r.final_top_k}) 大于 "
            f"RETRIEVAL_RERANK_INPUT_K({r.rerank_input_k})，重排后无法凑满 top-k"
        )
    if r.rerank_input_k < r.top_k_per_route:
        warnings.append(
            f"RETRIEVAL_RERANK_INPUT_K({r.rerank_input_k}) 小于 "
            f"RETRIEVAL_TOP_K_PER_ROUTE({r.top_k_per_route})，单路召回结果会被重排阶段截断"
        )

    if r.enable_rerank and not settings.rerank.enabled:
        warnings.append(
            "RETRIEVAL_ENABLE_RERANK=true 但 RERANK_ENABLED=false："
            "将跳过重排并直接用 RRF 分数，响应会标记 degraded"
        )
    if r.enable_dense_text and not settings.text_embed.enabled:
        warnings.append(
            "RETRIEVAL_ENABLE_DENSE_TEXT=true 但 TEXT_EMBED_ENABLED=false：文本向量路会被跳过"
        )
    if r.enable_dense_image and not settings.image_embed.enabled:
        warnings.append(
            "RETRIEVAL_ENABLE_DENSE_IMAGE=true 但 IMAGE_EMBED_ENABLED=false："
            "以文搜图/以图搜图会被跳过"
        )
    if r.enable_dense_image and not settings.image_embed.enable_fusion:
        warnings.append(
            "IMAGE_EMBED_ENABLE_FUSION=false：以文搜图依赖图文融合空间，"
            "关闭后跨模态召回质量会显著下降"
        )

    # ---------- Agent ----------
    if settings.agent.judge_enabled and settings.agent.max_rounds < 2:
        warnings.append(
            "AGENT_MAX_ROUNDS=1 时相关性打分不会触发第二轮检索，judge 只影响 confidence"
        )
    if settings.agent.query_rewrite_enabled and settings.agent.query_rewrite_n < 1:
        warnings.append("AGENT_QUERY_REWRITE_N<1，多查询改写实际不会产生任何新 query")

    # ---------- 存储路径可写性（只探测，不创建目录，保持本函数无副作用）----------
    root = settings.paths.root
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.is_dir():
        errors.append(f"数据根目录的最近存在祖先不是目录：{probe}")
    elif not os.access(probe, os.W_OK):
        errors.append(f"数据根目录不可写：{root}（最近存在的祖先 {probe} 无写权限）")

    return errors, warnings


# --------------------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试里改完环境变量请调用 :func:`reset_settings_cache`。"""
    return Settings()


def reset_settings_cache() -> None:
    """清空单例缓存（仅测试与热重载使用）。"""
    get_settings.cache_clear()


__all__ = [
    "PROJECT_ROOT",
    "AgentSettings",
    "AppSettings",
    "BM25Settings",
    "BlobSettings",
    "ChunkingSettings",
    "ConfigurationError",
    "DocStoreSettings",
    "EvalSettings",
    "GenerationSettings",
    "IMAGE_EMBED_NATIVE_PATH",
    "ImageEmbeddingSettings",
    "IngestSettings",
    "LLMSettings",
    "PathSettings",
    "RERANK_PROVIDER_DEFAULTS",
    "RerankSettings",
    "ResolvedEndpoint",
    "RetrievalSettings",
    "Settings",
    "TEXT_EMBED_PROVIDER_DEFAULTS",
    "TextEmbeddingSettings",
    "VectorStoreSettings",
    "VisionSettings",
    "collect_config_problems",
    "get_settings",
    "reset_settings_cache",
]
