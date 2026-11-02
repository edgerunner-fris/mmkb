"""Provider 抽象层。

**这是整个项目唯一允许出现厂商差异的地方。** 业务代码（摄入、检索、生成、评估）
只 import 本模块的 ABC，永远不 import ``openai`` / ``httpx`` 到具体端点。

三个抽象
--------
:class:`LLMProvider`
    对话与流式对话。视觉能力通过 ``vision_chat`` 暴露，默认实现是把图片拼成
    content parts 后走 ``chat``，所以任何支持图片输入的模型都自动具备视觉能力。

:class:`EmbeddingProvider`
    **一个实例只代表一个向量空间**。文本空间与多模态空间是两个不同的 provider，
    由 factory 分别构造：

    ====================  ================  ==========================
    方法                  文本空间 provider  多模态空间 provider
    ====================  ================  ==========================
    ``embed_texts``      ✅ 文档块 / query   ❌ 抛异常（会污染文本库）
    ``embed_images``     ❌ 抛异常           ✅ 图片入库
    ``embed_fused``      ❌ 抛异常           ✅ text-only query 检索图像库
    ====================  ================  ==========================

    不支持的操作会抛 :class:`ProviderUnsupportedOperationError`，而不是返回一个
    「维度看起来对、其实来自另一个空间」的向量。这是把「禁止混用向量空间」
    从口头约定变成运行期断言的手段。

:class:`RerankProvider`
    交叉编码器重排。**只吃文本**，无法理解图片 —— 所以图片候选必须靠
    caption + 章节路径拼出来的文本参与重排。这是本项目的已知取舍，
    在 retrieval 层和 README 中都会显式说明。

公共设施
--------
* :class:`RateLimiter` —— 并发上限 + 每分钟请求数限制
* :func:`retry_policy` —— 基于 :func:`app.core.errors.is_retryable` 的 tenacity 策略
* :func:`chunked` —— 批次分片
* :func:`prepare_image_for_api` —— Pillow 缩放 + 统一转 JPEG + base64 data URI
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from app.core.errors import (
    ProviderUnavailableError,
    ProviderUnsupportedOperationError,
    is_retryable,
)

__all__ = [
    "ChatDelta",
    "ChatMessage",
    "ChatResult",
    "ContentPart",
    "EmbeddingProvider",
    "EmbeddingResult",
    "ImagePart",
    "LLMProvider",
    "PreparedImage",
    "RateLimiter",
    "RerankHit",
    "RerankProvider",
    "RerankResult",
    "TextPart",
    "VectorSpace",
    "chunked",
    "prepare_image_for_api",
    "retry_policy",
    "run_with_retry",
    "to_data_uri",
]

logger = logging.getLogger(__name__)

#: 向量空间标识。文本空间与多模态空间**不可互换**。
VectorSpace = Literal["text", "image"]


# --------------------------------------------------------------------------------------
# 消息与内容
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class TextPart:
    text: str

    def to_openai(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass(slots=True)
class ImagePart:
    """一张已经准备好发出去的图（base64 data URI，不是本地路径）。"""

    data_uri: str
    detail: Literal["auto", "low", "high"] | None = None

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"url": self.data_uri}
        # detail 不是所有 OpenAI 兼容端点都认识，None 时干脆不发这个键。
        if self.detail is not None:
            payload["detail"] = self.detail
        return {"type": "image_url", "image_url": payload}


ContentPart = TextPart | ImagePart


@dataclass(slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]

    @classmethod
    def system(cls, text: str) -> ChatMessage:
        return cls("system", text)

    @classmethod
    def user(cls, text: str) -> ChatMessage:
        return cls("user", text)

    @classmethod
    def assistant(cls, text: str) -> ChatMessage:
        return cls("assistant", text)

    @classmethod
    def user_with_images(
        cls,
        text: str,
        images: Sequence[PreparedImage],
        detail: Literal["auto", "low", "high"] | None = None,
    ) -> ChatMessage:
        """图片在前、文字在后。

        顺序对多数 VLM 有实际影响：先看到图，再读到「请描述这张图」，
        比反过来更容易得到针对图片本身的回答，而不是泛泛而谈。
        """
        parts: list[ContentPart] = [ImagePart(img.data_uri, detail) for img in images]
        if text:
            parts.append(TextPart(text))
        return cls("user", parts)

    @property
    def text(self) -> str:
        """把内容里的纯文本部分取出来（用于日志与 token 估算）。"""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text for p in self.content if isinstance(p, TextPart))

    def to_openai(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": [p.to_openai() for p in self.content]}


# --------------------------------------------------------------------------------------
# 返回值
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class ChatResult:
    text: str = ""
    reasoning_text: str = ""
    model: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 1
    degraded: bool = False
    degraded_reasons: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatDelta:
    """流式增量。``reasoning_text`` 是思考过程（前端可选择折叠展示）。"""

    text: str = ""
    reasoning_text: str = ""
    done: bool = False
    finish_reason: str = ""


@dataclass(slots=True)
class EmbeddingResult:
    vectors: list[list[float]] = field(default_factory=list)
    model: str = ""
    dim: int = 0
    latency_ms: int = 0
    usage_tokens: int = 0
    attempts: int = 1

    def __len__(self) -> int:
        return len(self.vectors)


@dataclass(slots=True)
class RerankHit:
    index: int
    score: float


@dataclass(slots=True)
class RerankResult:
    """``hits`` 按 score 降序；``index`` 指向传入 ``documents`` 的下标。

    只返回命中的下标而不是「把文档也带回来」：文档本来就在调用方手里，
    来回搬运几十 KB 文本既浪费带宽也容易在前后端之间产生不一致。
    """

    hits: list[RerankHit] = field(default_factory=list)
    model: str = ""
    latency_ms: int = 0
    attempts: int = 1

    def scores_by_index(self) -> dict[int, float]:
        return {h.index: h.score for h in self.hits}


# --------------------------------------------------------------------------------------
# 图片准备
# --------------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class PreparedImage:
    data_uri: str
    mime_type: str
    width: int
    height: int
    byte_size: int
    source_bytes: int = 0

    @property
    def b64(self) -> str:
        """去掉 ``data:image/jpeg;base64,`` 前缀的裸 base64。

        DashScope 原生端点两种都收，但裸 base64 更通用。
        """
        return self.data_uri.split(",", 1)[-1]

    @property
    def compression_ratio(self) -> float:
        if self.source_bytes <= 0:
            return 1.0
        return self.byte_size / self.source_bytes


def to_data_uri(raw: bytes, mime_type: str = "image/jpeg") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"


def _flatten_to_rgb(image: Any) -> Any:
    """把任意模式压成 RGB。

    带 alpha 的图先合成到白底上：文档里的图片基本都是白底，直接丢弃 alpha 通道
    会让透明区域变黑，而合成到白底视觉效果等价且能存 JPEG（体积小得多）。
    """
    from PIL import Image

    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_alpha:
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.split()[-1])
    return background


def prepare_image_for_api(
    raw: bytes,
    *,
    max_side: int = 1568,
    jpeg_quality: int = 88,
) -> PreparedImage:
    """把任意图片字节转成可上行的 :class:`PreparedImage`。

    做三件事，每一件都有明确理由：

    1. **EXIF 方向矫正** —— 手机拍的图带着旋转标记，不矫正的话模型看到的是躺倒的。
    2. **限制长边** —— VLM 的视觉侧输入有像素预算，超过之后既更贵又不会更准。
       1568 是主流 VLM 的有效上限附近的一个稳妥值。
    3. **统一转 JPEG** —— 一个代码路径，体积可控，且原生端点与 OpenAI 兼容端点都收。

    失败时抛 ``ValueError``，由调用方决定是跳过这张图还是让整个文档失败。
    """
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - 依赖缺失属于部署问题
        raise RuntimeError("图片处理需要 Pillow，请确认已安装项目依赖") from exc

    if not raw:
        raise ValueError("图片字节为空")

    try:
        with Image.open(io.BytesIO(raw)) as opened:
            try:
                image = ImageOps.exif_transpose(opened) or opened
            except Exception:  # noqa: BLE001 - 损坏的 EXIF 不该让整张图失败
                image = opened
            image = _flatten_to_rgb(image)
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            width, height = image.size
    except OSError as exc:
        # Pillow 对非图片字节抛 UnidentifiedImageError，截断的图片抛 OSError；
        # 两者都归一到 ValueError，调用方只需处理一种异常。
        raise ValueError(str(exc)) from exc

    payload = buffer.getvalue()
    return PreparedImage(
        data_uri=to_data_uri(payload, "image/jpeg"),
        mime_type="image/jpeg",
        width=width,
        height=height,
        byte_size=len(payload),
        source_bytes=len(raw),
    )


# --------------------------------------------------------------------------------------
# 限流与分片
# --------------------------------------------------------------------------------------


class RateLimiter:
    """并发上限 + 每分钟请求数限制。

    并发用 :class:`asyncio.Semaphore`；RPM 用一个 60 秒滑动窗口。
    两者独立：并发管「同时在飞的请求数」，RPM 管「一分钟内总共发多少」。
    免费额度的 API 通常卡的是后者，而两者都超时会得到 429。

    ``max_concurrency <= 0`` 或 ``rpm <= 0`` 表示该项不限制。
    """

    __slots__ = ("_sem", "_rpm", "_stamps", "_lock", "_name")

    def __init__(self, max_concurrency: int = 0, rpm: int = 0, name: str = "") -> None:
        self._sem = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
        self._rpm = max(0, rpm)
        self._stamps: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._name = name or "provider"

    async def acquire(self) -> None:
        if self._sem is not None:
            await self._sem.acquire()
        try:
            await self._throttle()
        except BaseException:
            # 限流阶段失败也要把信号量还回去，否则一次异常会永久吃掉一个并发槽。
            if self._sem is not None:
                self._sem.release()
            raise

    def release(self) -> None:
        if self._sem is not None:
            self._sem.release()

    async def _throttle(self) -> None:
        if self._rpm <= 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] >= 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self._rpm:
                    self._stamps.append(now)
                    return
                wait = 60.0 - (now - self._stamps[0])
            logger.debug(
                "达到 RPM 上限，等待后再发请求", extra={"provider": self._name, "wait_s": round(wait, 2)}
            )
            await asyncio.sleep(max(wait, 0.05))

    async def __aenter__(self) -> RateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.release()


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """把序列切成固定大小的批次。``size <= 0`` 视为不分片。"""
    if size <= 0:
        yield items
        return
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --------------------------------------------------------------------------------------
# 重试
# --------------------------------------------------------------------------------------


def retry_policy(
    *,
    max_attempts: int = 3,
    min_wait: float = 0.8,
    max_wait: float = 20.0,
    label: str = "",
) -> AsyncRetrying:
    """构造一份 tenacity 异步重试策略。

    **每次调用都必须新建一个实例**：tenacity 的 ``AsyncRetrying`` 是有状态且不可
    重复迭代的，复用同一实例会在第二次调用时静默不重试。

    重试判据统一走 :func:`app.core.errors.is_retryable`：超时/连接失败/5xx/429 重试，
    400/401/维度不匹配这类「重试也没用」的错误立刻抛出。

    抖动（jitter）是必须的：多个并发请求同时被限流后，如果退避时间完全一致，
    它们会在同一毫秒再次一起打过去，形成周期性尖峰。
    """

    def _log_retry(state: Any) -> None:
        outcome = getattr(state, "outcome", None)
        exc = outcome.exception() if outcome is not None else None
        action = getattr(state, "next_action", None)
        logger.warning(
            "provider 调用失败，准备重试",
            extra={
                "provider": label,
                "attempt": state.attempt_number,
                "max_attempts": max_attempts,
                "error": type(exc).__name__ if exc else "unknown",
                "detail": str(exc)[:200] if exc else "",
                "sleep_s": round(getattr(action, "sleep", 0.0) or 0.0, 2),
            },
        )

    return AsyncRetrying(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(max(1, max_attempts)),
        wait=wait_exponential_jitter(multiplier=min_wait, max=max_wait),
        before_sleep=_log_retry,
        reraise=True,
    )


async def run_with_retry[T](
    op: str,
    call: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    limiter: RateLimiter | None = None,
    provider: str = "",
    model: str = "",
    min_wait: float = 0.8,
    max_wait: float = 20.0,
) -> tuple[T, int]:
    """限流 + 重试地执行 ``call``，返回 ``(结果, 实际尝试次数)``。

    所有 provider 共用这一个外壳，好处是：重试判据、限流位置、尝试次数统计
    只有一处实现。``call`` 内部必须把厂商异常翻译成本项目的异常层级 ——
    否则 :func:`app.core.errors.is_retryable` 认不出来，等于不重试。

    ``min_wait`` / ``max_wait`` 只在测试里需要调小；生产路径用默认值。
    """
    policy = retry_policy(
        max_attempts=max_attempts,
        min_wait=min_wait,
        max_wait=max_wait,
        label=f"{provider}:{op}" if provider else op,
    )
    attempts = 0
    async for attempt in policy:
        with attempt:
            attempts += 1
            if limiter is None:
                return await call(), attempts
            async with limiter:
                return await call(), attempts

    # tenacity 在 reraise=True 下会抛出最后一次异常，正常不会走到这里
    raise ProviderUnavailableError(
        f"{op} 重试 {attempts} 次后仍未成功", provider=provider, model=model
    )


# --------------------------------------------------------------------------------------
# 抽象基类
# --------------------------------------------------------------------------------------


class LLMProvider(ABC):
    """对话模型。"""

    #: 供日志与错误信息使用的 provider 名（如 ``deepseek``）
    name: str = "llm"
    #: 当前使用的模型名
    model: str = ""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        """一次性返回完整回复。"""

    @abstractmethod
    def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatDelta]:
        """流式返回。实现为 **async generator**（调用时不立即发请求）。"""

    async def vision_chat(
        self,
        prompt: str,
        images: Sequence[PreparedImage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """看图回答。默认实现把图片拼成 content parts 后复用 ``chat``。

        之所以给默认实现：对 OpenAI 兼容协议来说这就是唯一的做法，
        每个 provider 各写一遍只会重复。真要特殊处理（例如 DashScope 原生
        多模态端点）时再覆盖。
        """
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage.system(system))
        messages.append(ChatMessage.user_with_images(prompt, images))
        return await self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    @property
    def supports_vision(self) -> bool:
        return True

    # B027：这是**有意的空默认实现** —— 并非每个 provider 都持有需要关闭的资源，
    # 标记为 abstractmethod 会强迫所有子类写一遍无意义的空方法。
    async def aclose(self) -> None:  # noqa: B027
        """释放底层连接池。默认无操作，子类按需覆盖。"""


class EmbeddingProvider(ABC):
    """**单一向量空间**的 embedding。

    一个实例只服务一个空间。文本空间与多模态空间分别构造，
    见模块文档里的能力矩阵。
    """

    name: str = "embedding"
    model: str = ""
    #: 该 provider 服务的向量空间
    space: VectorSpace = "text"
    #: 期望维度（来自配置）
    dim: int = 0

    # -- 能力声明（用于 factory 自检与错误信息）-----------------------------------

    @property
    def supports_text(self) -> bool:
        return False

    @property
    def supports_image(self) -> bool:
        return False

    @property
    def supports_fused(self) -> bool:
        return False

    # -- 操作 --------------------------------------------------------------------

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        """编码纯文本。

        * 文本空间 provider：文档块 + 纯文本 query（铁律 1）
        * 多模态空间 provider：**不支持**，调用即抛异常（否则会污染 ``{kb}__text``）
        """

    @abstractmethod
    async def embed_images(
        self, images: Sequence[bytes | PreparedImage]
    ) -> EmbeddingResult:
        """编码图片。

        * 多模态空间 provider：图片入库（与 ``embed_fused`` 用同一套参数）
        * 文本空间 provider：**不支持**

        接受 ``bytes`` 或已处理好的 :class:`PreparedImage`：同一张图往往先送给
        VLM 生成 caption、再送去 embedding，允许传入已处理的实例可以省掉一次
        解码+缩放+JPEG 编码。
        """

    @abstractmethod
    async def embed_fused(self, texts: Sequence[str]) -> EmbeddingResult:
        """把**纯文本 query** 编码进多模态融合空间（铁律 2：以文搜图）。

        必须与图片入库时使用完全相同的模型与 ``enable_fusion``，
        否则 query 与图片不在同一空间，余弦相似度没有意义。

        * 多模态空间 provider：支持
        * 文本空间 provider：**不支持** —— 用 text-embedding-v4 的向量查图像库是本项目
          明令禁止的错误
        """

    # -- 内部工具 ----------------------------------------------------------------

    def _unsupported(self, op: str, hint: str) -> ProviderUnsupportedOperationError:
        return ProviderUnsupportedOperationError(
            f"{self.name}（{self.model}，{self.space} 空间）不支持 {op}。{hint}",
            provider=self.name,
            model=self.model,
            detail={"operation": op, "space": self.space},
        )

    async def aclose(self) -> None:  # noqa: B027 - 有意的空默认实现，理由同 LLMProvider
        """释放底层连接池。默认无操作，子类按需覆盖。"""


class RerankProvider(ABC):
    """交叉编码器重排。

    ⚠️ **只接受文本**。图片候选参与重排时，传进去的是 caption + 章节路径拼成的
    文本（见 :func:`app.core.schemas.build_caption_text`），不是图片本身。
    这意味着「视觉上相似但 caption 没写出来」的图片会被低估 —— 这是当前所有
    纯文本 reranker 的共性限制，README 中有专门说明。
    """

    name: str = "rerank"
    model: str = ""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> RerankResult:
        """对 ``documents`` 打分。``index`` 是传入序列的下标。"""

    async def aclose(self) -> None:  # noqa: B027 - 有意的空默认实现，理由同 LLMProvider
        """释放底层连接池。默认无操作，子类按需覆盖。"""
