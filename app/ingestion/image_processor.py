"""图片处理：感知哈希、质量过滤、去重、落盘、caption。

这一步的产出决定了三件事：
1. 哪些图值得花多模态 embedding 的钱（过滤）；
2. 每张图怎么被**文本**检索命中（caption_text）；
3. 前端拿到的 URL 指向哪个文件（blobstore）。

过滤规则为什么是这几条
----------------------
文档里的图片绝大多数是重复、装饰、或截图边角料。全量入库的代价是双重的：
钱（每张图一次 VLM + 一次多模态 embedding）与检索质量（几十张 logo 会把
真正的架构图挤出 top-k）。因此按「最便宜到最贵」的顺序过滤：

===========  ==============================================
规则          判定
===========  ==============================================
过小          任一边小于 ``BLOB_MIN_WIDTH/HEIGHT``
空白          灰度标准差 < ``BLOB_BLANK_STD_THRESHOLD``
装饰          去重后颜色占比 < ``BLOB_MIN_UNIQUE_COLOR_RATIO``
超配额        同一文档已收满 ``BLOB_MAX_IMAGES_PER_DOC`` 张
重复          phash 汉明距离 ≤ ``BLOB_PHASH_HAMMING_THRESHOLD``
===========  ==============================================

phash 用 numpy 手写 DCT-II，不引入 ``imagehash`` / ``scipy``：
需要的只是 32×32 DCT 的前 8×8 低频分量，一个 32×32 的余弦矩阵就够，
为这点计算装两个带 C 扩展的大包不划算（而且在 Windows 上装 scipy 常常失败）。

**重要限制**：phash 对「内容相同但尺寸/压缩不同」的图有效，对「同一页的
两种排版」无效。它解决的是「同一个 logo 出现在 200 个页脚」，不是
「两张从不同角度拍的同一块白板」。
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import Settings
from app.core.logging import get_logger
from app.core.perf import get_metrics
from app.core.schemas import (
    CaptionSource,
    FilteredReason,
    ImageAsset,
    ImageCaption,
    ImageStatus,
    ImageType,
    build_caption_text,
)
from app.ingestion.parser import ParsedImage
from app.providers.base import LLMProvider, PreparedImage, prepare_image_for_api
from app.store.blobstore import BlobStore

__all__ = [
    "ImageProcessOutcome",
    "ImageProcessor",
    "ProcessedImage",
    "dct2",
    "hamming_hex",
    "phash",
]

logger = get_logger(__name__)

#: DCT 的输入边长。32 是 phash 的经典取值：再大只是把「低频」的范围放大，
#: 对鲁棒性没有帮助，却让每张图多算 4 倍的余弦。
_DCT_INPUT_SIZE = 32

_CAPTION_SYSTEM_PROMPT = """你是文档图片理解助手。你的输出会被写进检索索引，
所以只写图中真实存在的信息，不推测、不补全、不评论。"""

_CAPTION_PROMPT_TEMPLATE = """请阅读这张图片，并输出**严格 JSON**（不要 Markdown 代码块，不要任何解释文字）。

文档标题：{doc_title}
所在章节：{heading_path}
图片上方的正文：{context}

输出格式：
{{
  "image_type": "chart | screenshot | diagram | photo | formula | table_image | logo | other",
  "summary": "用 1-3 句话说明这张图表达了什么。图表要写出主要趋势和关键数值；架构图要写出组件与连接关系。",
  "ocr_text": "图中所有可读文字，按阅读顺序拼接；没有文字则为空字符串。",
  "relation_to_section": "这张图与所在章节主题的关系，一句话。",
  "keywords": ["3-8 个便于检索的关键词，包含图中出现的专有名词"]
}}

要求：
1. summary 必须能脱离图片独立看懂 —— 读者看不到图，只能读你写的字。
2. 不要写「这是一张图片」这类无信息量的话。
3. 图中没有的信息一律不要编造；看不清就说看不清。
4. keywords 用图中真实出现的术语，不要用同义词凑数。"""


@dataclass(slots=True)
class ProcessedImage:
    """一张图处理完的结果。``prepared`` 可直接复用给多模态 embedding。"""

    asset: ImageAsset
    prepared: PreparedImage | None = None
    anchor_block_id: str | None = None

    @property
    def kept(self) -> bool:
        return self.asset.status is ImageStatus.DONE


@dataclass(slots=True)
class ImageProcessOutcome:
    items: list[ProcessedImage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def kept(self) -> list[ProcessedImage]:
        return [item for item in self.items if item.kept]

    @property
    def filtered(self) -> list[ProcessedImage]:
        return [
            item
            for item in self.items
            if item.asset.status is ImageStatus.FILTERED_OUT
        ]

    @property
    def failed(self) -> list[ProcessedImage]:
        return [item for item in self.items if item.asset.status is ImageStatus.FAILED]


# --------------------------------------------------------------------------------------
# 感知哈希
# --------------------------------------------------------------------------------------


def _dct_matrix(size: int) -> np.ndarray:
    """DCT-II 的正交基（行向量）。

    ``m[k, i] = c_k * cos(pi * (2i + 1) * k / (2N))``，``c_0 = 1/sqrt(2)``，
    其余 ``c_k = 1``，最后整体乘 ``sqrt(2/N)`` 归一化。
    """
    k = np.arange(size).reshape(-1, 1)
    i = np.arange(size).reshape(1, -1)
    matrix = np.cos(np.pi * (2 * i + 1) * k / (2 * size))
    matrix[0, :] *= 1.0 / math.sqrt(2.0)
    return matrix * math.sqrt(2.0 / size)


#: 32×32 的余弦矩阵在进程内只算一次。phash 会被每张图调用，
#: 每次都重算是纯粹的浪费。
_BASIS = _dct_matrix(_DCT_INPUT_SIZE)


def dct2(block: np.ndarray) -> np.ndarray:
    """二维 DCT-II：``C @ block @ C.T``。"""
    return _BASIS @ block @ _BASIS.T


def phash(image_bytes: bytes, *, size: int = 8) -> str:
    """感知哈希，返回十六进制字符串（默认 8×8 → 64 bit → 16 hex）。

    与 ``imagehash.phash`` 的算法一致：缩到 32×32 灰度 → DCT →
    取左上 ``size×size`` 低频 → 以中位数二值化。
    跳过直流分量（``flat[0]``）再取中位数：它代表整图平均亮度，
    参与中位数会让全图偏亮或偏暗的图出现大量相同的位。
    """
    from PIL import Image, ImageOps

    size = max(1, min(size, _DCT_INPUT_SIZE))
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = ImageOps.exif_transpose(opened) or opened
        gray = image.convert("L").resize(
            (_DCT_INPUT_SIZE, _DCT_INPUT_SIZE), Image.Resampling.LANCZOS
        )

    pixels = np.asarray(gray, dtype=np.float64)
    low = dct2(pixels)[:size, :size]
    flat = low.reshape(-1)
    if flat.size <= 1:
        return ""
    median = float(np.median(flat[1:]))
    return _bits_to_hex(flat > median)


def hamming_hex(a: str, b: str) -> int:
    """两个十六进制哈希的汉明距离。空串之间视为「不可比」，返回一个很大的值。"""
    if not a or not b:
        return 1 << 16
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _bits_to_hex(bits: np.ndarray) -> str:
    flat = bits.reshape(-1).astype(np.uint8)
    remainder = len(flat) % 4
    if remainder:
        flat = np.concatenate([flat, np.zeros(4 - remainder, dtype=np.uint8)])
    nibbles = flat.reshape(-1, 4) @ np.array([8, 4, 2, 1], dtype=np.uint8)
    return "".join(f"{int(value):x}" for value in nibbles)


# --------------------------------------------------------------------------------------
# 图片检查
# --------------------------------------------------------------------------------------

@dataclass(slots=True)
class _Inspection:
    width: int = 0
    height: int = 0
    phash: str = ""
    gray_std: float = 0.0
    unique_ratio: float = 1.0
    error: str = ""


def _inspect(data: bytes, *, hash_size: int) -> _Inspection:
    """一次解码，拿到过滤与去重需要的全部指标。

    刻意只解码一次：Pillow 的 ``open`` 是惰性的，``convert`` 才真正解压。
    每张图在这里解码一遍，之后都用结果。
    """
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened) or opened
            width, height = image.size
            rgb = image.convert("RGB")
            gray = rgb.convert("L")

            small_gray = np.asarray(
                gray.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float64
            )
            small_rgb = np.asarray(
                rgb.resize((64, 64), Image.Resampling.NEAREST), dtype=np.uint8
            )
            hashed = phash_from_image(gray, size=hash_size)
    except Exception as exc:  # noqa: BLE001 - 损坏图片不该打断整份文档
        return _Inspection(error=f"{type(exc).__name__}: {exc}")

    unique = np.unique(small_rgb.reshape(-1, 3), axis=0).shape[0]
    return _Inspection(
        width=int(width),
        height=int(height),
        phash=hashed,
        gray_std=float(small_gray.std()),
        unique_ratio=unique / float(small_rgb.shape[0] * small_rgb.shape[1]),
    )


def phash_from_image(gray_image: object, *, size: int = 8) -> str:
    """对已经转成灰度模式的 PIL 图片算哈希（避免重复解码）。

    ``gray_image`` 标注成 ``object`` 是为了不在模块顶层 import PIL ——
    Pillow 是必装依赖，但顶部 import 会让 CLI 的启动时间多出约 60ms。
    """
    from PIL import Image

    size = max(1, min(size, _DCT_INPUT_SIZE))
    resized = gray_image.resize(  # type: ignore[attr-defined]
        (_DCT_INPUT_SIZE, _DCT_INPUT_SIZE), Image.Resampling.LANCZOS
    )
    flat = dct2(np.asarray(resized, dtype=np.float64))[:size, :size].reshape(-1)
    if flat.size <= 1:
        return ""
    return _bits_to_hex(flat > float(np.median(flat[1:])))


# --------------------------------------------------------------------------------------
# 处理器
# --------------------------------------------------------------------------------------


class ImageProcessor:
    """把 :class:`~app.ingestion.parser.ParsedImage` 变成 :class:`ImageAsset`。

    ``llm`` 为 ``None`` 或 ``caption_enabled=False`` 时不做视觉描述，
    caption 退化为「章节路径 + 上方正文」的纯上下文形式（``caption_source=CONTEXT_ONLY``）。
    这个降级是有意义的：图片仍然能被「它所处的那段文字」检索到，
    只是丧失了「图中写了什么」的精确匹配能力。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        blobstore: BlobStore | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self._settings = settings
        self._blob = blobstore or BlobStore(settings)
        self._llm = llm

    # -- 主流程 ---------------------------------------------------------------------

    async def process(
        self,
        images: list[ParsedImage],
        *,
        kb_id: str,
        doc_id: str,
        doc_title: str = "",
        existing_phashes: dict[str, str] | None = None,
        caption_enabled: bool = True,
    ) -> ImageProcessOutcome:
        """处理一份文档里的全部图片。

        ``existing_phashes`` 是「已经在这个知识库里」的 ``{phash: image_id}``，
        由调用方查询后传入 —— 处理器本身不碰数据库，这样它可以被单独测试，
        也不会在异步调用链里偷偷发起查询。

        返回顺序与 ``images`` 一致；被过滤的图也在结果里（带着原因），
        这样上层能把「抽到 20 张图，留下 6 张」如实报告出来。
        """
        # 跨文档去重只在这两种情况下查：配置要求 kb 级，或调用方主动提供了索引
        known = dict(existing_phashes or {})
        doc_phashes: dict[str, str] = {}
        outcome = ImageProcessOutcome()
        kept_count = 0

        # 解码 + phash + 颜色统计是纯 CPU，且每张图互不依赖。先并发算完指标，
        # 再按原顺序做过滤决策 —— 顺序不能乱：去重与「每文档最多 N 张」
        # 都依赖「前面已经留下了哪些图」，并发决策会让结果不可复现。
        inspections = await self._inspect_all(images)

        for parsed, inspection in zip(images, inspections, strict=True):
            asset = ImageAsset(
                image_id=parsed.image_id,
                kb_id=kb_id,
                doc_id=doc_id,
                page=parsed.page,
                bbox=parsed.bbox,
                order_in_doc=parsed.order,
                mime_type=parsed.mime_type,
                size_bytes=len(parsed.data),
                sha256=parsed.sha256,
                phash=inspection.phash,
                context_text=parsed.context_text,
                heading_path=list(parsed.heading_path),
                image_vector_id=parsed.image_id,
            )
            item = ProcessedImage(asset=asset, anchor_block_id=parsed.anchor_block_id)

            if inspection.error:
                asset.status = ImageStatus.FAILED
                outcome.warnings.append(
                    f"图片 {parsed.image_id} 无法解码：{inspection.error[:160]}"
                )
                outcome.items.append(item)
                continue

            asset.width = inspection.width
            asset.height = inspection.height

            reason = self._filter_reason(
                inspection=inspection,
                sha256=parsed.sha256,
                kept_count=kept_count,
                known_phashes={**known, **doc_phashes},
            )
            if reason is not None:
                asset.status = ImageStatus.FILTERED_OUT
                asset.filtered_reason = reason
                outcome.items.append(item)
                continue

            try:
                path, url = await self._store(parsed, kb_id)
            except Exception as exc:  # noqa: BLE001 - 单张图落盘失败不废整份文档
                asset.status = ImageStatus.FAILED
                outcome.warnings.append(
                    f"图片 {parsed.image_id} 落盘失败：{type(exc).__name__}: {exc}"
                )
                outcome.items.append(item)
                continue

            asset.file_path = str(path)
            asset.url = url
            asset.size_bytes = path.stat().st_size if path.exists() else len(parsed.data)
            # **先标 DONE，再走 caption。** ``outcome.kept`` 是按
            # ``status is DONE`` 筛的，这里漏了这一步的话 kept 永远是空列表：
            # 图片既不进 caption，也不进向量库，却已经落盘成了孤儿文件。
            # 后面的 caption 步骤仍可把它改成 FAILED（描述为空时）。
            asset.status = ImageStatus.DONE
            kept_count += 1
            if inspection.phash:
                doc_phashes[inspection.phash] = parsed.image_id
            else:
                # phash 为空时用 sha256 占位：内容完全相同的图仍应去重
                doc_phashes[f"sha:{parsed.sha256}"] = parsed.image_id
            outcome.items.append(item)

        if caption_enabled and self._llm is not None:
            await self._caption_all(outcome.kept, doc_title=doc_title, outcome=outcome)
        else:
            for item in outcome.kept:
                self._apply_context_only_caption(item, doc_title=doc_title)

        return outcome

    # -- 过滤 -----------------------------------------------------------------------

    async def _inspect_all(self, images: list[ParsedImage]) -> list[_Inspection]:
        """并发算出每张图的过滤指标，结果顺序与输入一致。

        为什么用线程池而不是直接 ``await``：``_inspect`` 里的 Pillow 解码、
        缩放与 DCT 都是同步 CPU 操作，单张耗时随图片尺寸增长（几十到几百毫秒）。
        50 张图串行就是十几秒，而这段时间事件循环完全被阻塞。
        并发度由 ``INGEST_IMAGE_INSPECT_CONCURRENCY`` 控制：它不消耗任何外部配额，
        默认值（4）只受本机 CPU 核数影响。
        """
        if not images:
            return []
        limit = max(1, self._settings.ingest.image_inspect_concurrency)
        semaphore = asyncio.Semaphore(limit)
        hash_size = self._settings.blob.phash_size

        async def one(parsed: ParsedImage) -> _Inspection:
            async with semaphore:
                return await asyncio.to_thread(
                    _inspect, parsed.data, hash_size=hash_size
                )

        started = time.perf_counter()
        results = await asyncio.gather(*(one(parsed) for parsed in images))
        get_metrics().observe(
            "ingest.image.inspect", (time.perf_counter() - started) * 1000
        )
        return list(results)

    def _filter_reason(
        self,
        *,
        inspection: _Inspection,
        sha256: str,
        kept_count: int,
        known_phashes: dict[str, str],
    ) -> FilteredReason | None:
        blob = self._settings.blob

        if inspection.width < blob.min_width or inspection.height < blob.min_height:
            return FilteredReason.SMALL
        if inspection.gray_std < blob.blank_std_threshold:
            return FilteredReason.BLANK
        if inspection.unique_ratio < blob.min_unique_color_ratio:
            return FilteredReason.DECORATIVE
        if blob.max_images_per_doc and kept_count >= blob.max_images_per_doc:
            return FilteredReason.OVER_QUOTA

        # 字节完全相同的图先命中 sha256。它比 phash 更严格也更便宜 ——
        # 同一张图被 Word 重复嵌入两次时，两者的哈希前缀都会一致。
        if f"sha:{sha256}" in known_phashes:
            return FilteredReason.DUPLICATE

        for known_hash in known_phashes:
            if known_hash.startswith("sha:"):
                continue
            if hamming_hex(inspection.phash, known_hash) <= blob.phash_hamming_threshold:
                return FilteredReason.DUPLICATE
        return None

    # -- 落盘 -----------------------------------------------------------------------

    async def _store(self, parsed: ParsedImage, kb_id: str) -> tuple[Path, str]:
        payload, ext = parsed.data, parsed.ext
        if self._settings.blob.convert_to_png:
            payload, ext = await asyncio.to_thread(
                _to_png, parsed.data, self._settings.blob.jpeg_quality
            )
        mime = "image/png" if ext == "png" else parsed.mime_type
        return await asyncio.to_thread(
            self._blob.put, kb_id, parsed.image_id, payload, mime_type=mime, ext=ext
        )

    # -- caption --------------------------------------------------------------------

    async def _caption_all(
        self,
        items: list[ProcessedImage],
        *,
        doc_title: str,
        outcome: ImageProcessOutcome,
    ) -> None:
        if not items:
            return

        # 视觉 provider 自己也有并发上限，这里的信号量是**第二道**闸门：
        # 一份 PDF 抽出 80 张图时，一次性 gather 80 个协程虽然不会同时发出去
        # （provider 会排队），却会同时持有 80 份 base64 图片在内存里。
        semaphore = asyncio.Semaphore(max(1, self._settings.vision.max_concurrency))

        async def run(item: ProcessedImage) -> None:
            async with semaphore:
                started = time.perf_counter()
                await self._caption_one(item, doc_title=doc_title, outcome=outcome)
                # 单张粒度而不是整批：性能报告要回答的是「一张图 caption 平均多久」，
                # 它直接决定批量导入的预估时间。
                get_metrics().observe(
                    "ingest.image.caption", (time.perf_counter() - started) * 1000
                )

        await asyncio.gather(*(run(item) for item in items), return_exceptions=False)

    async def _caption_one(
        self,
        item: ProcessedImage,
        *,
        doc_title: str,
        outcome: ImageProcessOutcome,
    ) -> None:
        asset = item.asset
        blob = self._settings.blob
        started = time.perf_counter()
        try:
            raw = await asyncio.to_thread(Path(asset.file_path).read_bytes)
            prepared = await asyncio.to_thread(
                prepare_image_for_api,
                raw,
                max_side=self._settings.vision.max_image_side,
                jpeg_quality=blob.jpeg_quality,
            )
        except Exception as exc:  # noqa: BLE001 - 预处理失败就退化为纯上下文
            outcome.warnings.append(
                f"图片 {asset.image_id} 预处理失败，使用上下文 caption：{exc}"
            )
            self._apply_context_only_caption(item, doc_title=doc_title)
            return

        item.prepared = prepared
        prompt = _CAPTION_PROMPT_TEMPLATE.format(
            doc_title=doc_title or "（未命名文档）",
            heading_path=" > ".join(asset.heading_path) or "（无章节信息）",
            context=_truncate(asset.context_text or "（图片上方没有正文）", 600),
        )

        llm = self._llm
        if llm is None:  # 调用方保证了非空，这里只是兜底，避免 assert 被 -O 剥掉
            self._apply_context_only_caption(item, doc_title=doc_title)
            return
        try:
            result = await llm.vision_chat(
                prompt,
                [prepared],
                system=_CAPTION_SYSTEM_PROMPT,
                temperature=self._settings.vision.temperature,
                max_tokens=self._settings.vision.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - 视觉失败不该让文档导入失败
            outcome.warnings.append(
                f"图片 {asset.image_id} 生成描述失败，使用上下文 caption："
                f"{type(exc).__name__}: {exc}"
            )
            self._apply_context_only_caption(item, doc_title=doc_title)
            return

        caption = _parse_caption(result.text)
        if caption is None:
            outcome.warnings.append(
                f"图片 {asset.image_id} 的描述不是合法 JSON，使用上下文 caption"
            )
            self._apply_context_only_caption(item, doc_title=doc_title)
            return

        caption.caption_model = result.model or llm.model
        caption.caption_latency_ms = int((time.perf_counter() - started) * 1000)
        asset.caption = caption
        asset.caption_source = CaptionSource.VLM
        asset.caption_text = build_caption_text(caption, asset.heading_path)
        asset.status = (
            ImageStatus.DONE if asset.caption_text.strip() else ImageStatus.FAILED
        )
        if asset.status is ImageStatus.FAILED:
            outcome.warnings.append(f"图片 {asset.image_id} 的 caption 为空，已跳过")

    def _apply_context_only_caption(self, item: ProcessedImage, *, doc_title: str) -> None:
        asset = item.asset
        context = _truncate(asset.context_text.strip(), 500)
        heading = " > ".join(asset.heading_path)
        summary = context or heading or doc_title
        caption = ImageCaption(
            image_type=ImageType.OTHER,
            summary=summary,
            relation_to_section=heading,
        )
        asset.caption = caption
        asset.caption_source = CaptionSource.CONTEXT_ONLY
        asset.caption_text = build_caption_text(caption, asset.heading_path)
        asset.status = (
            ImageStatus.DONE if asset.caption_text.strip() else ImageStatus.FAILED
        )
        if asset.status is ImageStatus.DONE:
            # 图片已经落盘，能展示；只是描述来自上下文而不是视觉模型
            logger.debug(
                "图片使用上下文 caption",
                extra={"image_id": asset.image_id, "doc_id": asset.doc_id},
            )


def _parse_caption(text: str) -> ImageCaption | None:
    """解析 VLM 的结构化输出。

    模型经常不听「不要 Markdown 代码块」这句，所以先剥围栏、再取第一个
    ``{`` 到最后一个 ``}``。两个兜底都不能省：有些模型会在 JSON 前面写
    「好的，以下是分析结果：」，直接 ``json.loads`` 会失败。
    """
    if not text:
        return None

    body = text.strip()
    if "```" in body:
        start = body.find("```")
        end = body.rfind("```")
        if end > start:
            body = body[start + 3 : end]
            if body.lstrip().lower().startswith("json"):
                body = body.lstrip()[4:]

    first = body.find("{")
    last = body.rfind("}")
    if first < 0 or last <= first:
        return None

    try:
        payload = json.loads(body[first : last + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    summary = _str_field(payload.get("summary"), 600)
    ocr_text = _str_field(payload.get("ocr_text"), 2000)
    if not summary and not ocr_text:
        return None

    return ImageCaption(
        image_type=_image_type(payload.get("image_type")),
        summary=summary,
        ocr_text=ocr_text,
        relation_to_section=_str_field(payload.get("relation_to_section"), 400),
        keywords=_keywords(payload.get("keywords")),
    )


def _str_field(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return _truncate(value.strip(), limit)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _image_type(value: object) -> ImageType:
    if not isinstance(value, str):
        return ImageType.OTHER
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return ImageType(normalized)
    except ValueError:
        return ImageType.OTHER


def _keywords(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        word = _truncate(item.strip(), 60)
        if not word or word in seen:
            continue
        seen.add(word)
        out.append(word)
        if len(out) >= 12:
            break
    return out


def _to_png(data: bytes, jpeg_quality: int) -> tuple[bytes, str]:
    """按配置统一转 PNG（``BLOB_CONVERT_TO_PNG=true`` 时）。"""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as opened:
        image = ImageOps.exif_transpose(opened) or opened
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), "png"
