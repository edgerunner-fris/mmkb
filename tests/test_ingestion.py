"""摄入层测试。

全部在临时目录里跑，**不碰 data/ 下的真实数据，也不需要任何 API Key**。
需要向量的用例注入一个确定性的假 provider，它遵守
:class:`~app.providers.base.EmbeddingProvider` 的契约、只在本进程里算数，
因此这里验证的是流水线本身（写入、幂等、快照校验），而不是任何厂商接口。

覆盖的四类问题，共同点是「错了不会立刻被发现」：

1. **顺序契约** —— 块与图片共用 ``order`` 计数器那条规则一旦被破坏，
   图片会锚到错误的小节，导入却仍然报告成功；
2. **图片生命周期** —— 过滤后仍在结果里（带原因）、保留的才是 DONE、
   ``kept`` 为空时 caption 与图像向量会一起消失；
3. **导入幂等** —— 同一文件重跑必须收敛到同一批 chunk_id，
   否则每跑一次就在向量库里多留一批再也删不掉的孤儿；
4. **失败不扩散** —— 一份损坏的 PDF 只能把自己标成 FAILED，
   不能中断批量导入，也不能把上一条成功记录改坏。
"""

from __future__ import annotations

import hashlib
import io
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.config import (
    BlobSettings,
    BM25Settings,
    ChunkingSettings,
    DocStoreSettings,
    ImageEmbeddingSettings,
    PathSettings,
    Settings,
    TextEmbeddingSettings,
    VectorStoreSettings,
)
from app.core.errors import UnsupportedFileTypeError
from app.core.schemas import (
    BlockType,
    CaptionSource,
    DocStatus,
    FilteredReason,
    ImageAsset,
    ImageStatus,
)
from app.core.tokens import count_tokens
from app.ingestion.chunker import chunk_document
from app.ingestion.image_processor import ImageProcessor, hamming_hex, phash
from app.ingestion.loaders.base import (
    LoadResult,
    RawBlock,
    RawImage,
    get_registry,
    read_image_source,
)
from app.ingestion.parser import ParsedImage, image_id_for, parse_document
from app.ingestion.pipeline import (
    IngestionPipeline,
    IngestOptions,
    ProviderSet,
    collect_files,
)
from app.ingestion.table_serializer import split_table_markdown, to_markdown
from app.providers.base import EmbeddingProvider, EmbeddingResult
from app.store.blobstore import BlobStore
from app.store.bm25store import BM25Store
from app.store.db import Database
from app.store.docstore import DocStore
from app.store.vectorstore import image_collection, text_collection
from app.store.vectorstore_chroma import ChromaVectorStore

# --------------------------------------------------------------------------------------
# 夹具与工具
# --------------------------------------------------------------------------------------

TEXT_MODEL = "fake-text"
IMAGE_MODEL = "fake-image"


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    """完全指向 tmp_path 的配置：不读 .env，不碰真实数据目录。"""
    values: dict[str, Any] = {
        "paths": PathSettings(data_dir=tmp_path),
        "docstore": DocStoreSettings(sqlite_path=tmp_path / "test.db"),
        "blob": BlobSettings(images_dir=tmp_path / "images"),
        "bm25": BM25Settings(index_dir=tmp_path / "bm25", persist=False),
        "vectorstore": VectorStoreSettings(chroma_dir=tmp_path / "chroma"),
        "text_embed": TextEmbeddingSettings(model=TEXT_MODEL, dim=8),
        "image_embed": ImageEmbeddingSettings(model=IMAGE_MODEL, dim=16),
        "chunking": ChunkingSettings(
            min_tokens=20,
            max_tokens=120,
            overlap_tokens=0,
            min_chunk_chars=20,
            table_max_tokens=120,
        ),
    }
    values.update(overrides)
    return Settings(**values)


def _png_bytes(size: tuple[int, int] = (200, 200), seed: int = 1) -> bytes:
    """随机噪点图：phash 稳定，且不会命中「空白 / 装饰」过滤。"""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _write_png(path: Path, *, size: tuple[int, int] = (200, 200), seed: int = 1) -> Path:
    path.write_bytes(_png_bytes(size, seed))
    return path


def _parsed_image(
    path: Path,
    *,
    kb_id: str = "kb_testingest",
    order: int = 1,
    heading_path: list[str] | None = None,
    context: str = "",
) -> ParsedImage:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return ParsedImage(
        image_id=image_id_for(kb_id, digest),
        order=order,
        data=data,
        ext="png",
        mime_type="image/png",
        sha256=digest,
        heading_path=list(heading_path or []),
        context_text=context,
    )


def _asset(
    image_id: str = "img_test0001",
    *,
    caption_text: str = "",
    heading_path: list[str] | None = None,
) -> ImageAsset:
    return ImageAsset(
        image_id=image_id,
        kb_id="kb_testingest",
        doc_id="doc_test0001",
        caption_text=caption_text,
        heading_path=list(heading_path or []),
        status=ImageStatus.DONE,
    )


def _vec(text: str, dim: int) -> list[float]:
    """确定性向量：同一段文字永远得到同一个向量。"""
    values = [0.0] * dim
    for index, char in enumerate(text):
        values[index % dim] += (ord(char) % 11) + 1.0
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


class _FakeEmbedding(EmbeddingProvider):
    """只服务一个空间的假 provider。能力边界与真实 provider 一致。"""

    name = "fake"

    def __init__(self, *, space: str, dim: int, model: str) -> None:
        self.space = space
        self.dim = dim
        self.model = model

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if self.space != "text":
            raise self._unsupported("embed_texts", "多模态 provider 不编码纯文本")
        return EmbeddingResult(
            vectors=[_vec(text, self.dim) for text in texts],
            model=self.model,
            dim=self.dim,
        )

    async def embed_images(self, images: list[Any]) -> EmbeddingResult:
        if self.space != "image":
            raise self._unsupported("embed_images", "文本 provider 不编码图片")
        vector = [1.0] + [0.0] * (self.dim - 1)
        return EmbeddingResult(
            vectors=[list(vector) for _ in images], model=self.model, dim=self.dim
        )

    async def embed_fused(self, texts: list[str]) -> EmbeddingResult:
        if self.space != "image":
            raise self._unsupported("embed_fused", "纯文本 provider 不进入融合空间")
        vector = [1.0] + [0.0] * (self.dim - 1)
        return EmbeddingResult(
            vectors=[list(vector) for _ in texts], model=self.model, dim=self.dim
        )


def _providers(*, text: bool = True, image: bool = True) -> ProviderSet:
    return ProviderSet(
        text_embedding=_FakeEmbedding(space="text", dim=8, model=TEXT_MODEL)
        if text
        else None,
        image_embedding=_FakeEmbedding(space="image", dim=16, model=IMAGE_MODEL)
        if image
        else None,
    )


@asynccontextmanager
async def _env(tmp_path: Path, *, providers: ProviderSet | None = None):
    """一套隔离的存储 + 流水线。所有对象都指向 tmp_path。"""
    settings = _settings(tmp_path)
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}", is_sqlite=True)
    await database.init_schema()

    docstore = DocStore(settings, db=database)
    vectorstore = ChromaVectorStore(settings)
    bm25store = BM25Store(settings)
    blobstore = BlobStore(settings)
    pipeline = IngestionPipeline(
        settings,
        docstore=docstore,
        vectorstore=vectorstore,
        bm25store=bm25store,
        blobstore=blobstore,
        providers=providers,
    )
    env = SimpleNamespace(
        settings=settings,
        database=database,
        docstore=docstore,
        vectorstore=vectorstore,
        bm25store=bm25store,
        blobstore=blobstore,
        pipeline=pipeline,
    )
    try:
        yield env
    finally:
        await pipeline.aclose()
        await vectorstore.aclose()
        await database.dispose()


def _markdown_doc(raw: Path) -> Path:
    """带标题、表格、代码与本地图片的 Markdown 文档。"""
    _write_png(raw / "figure.png", seed=42)
    path = raw / "guide.md"
    path.write_text(
        "\n".join(
            [
                "---",
                "title: 多模态检索指南",
                "---",
                "",
                "# 第一章 检索",
                "",
                "本系统使用三路召回：BM25、文本向量与多模态向量，",
                "最后用 RRF 融合并重排。",
                "",
                "## 1.1 召回路径",
                "",
                "| 路径 | 数据来源 |",
                "| --- | --- |",
                "| 文本 | text-embedding-v4 |",
                "| 图像 | qwen3-vl-embedding |",
                "",
                "```python",
                'print("hybrid retrieval")',
                "```",
                "",
                "![三路召回示意图](figure.png)",
                "",
                "以上是整体架构。",
            ]
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# 1. 表格序列化
# --------------------------------------------------------------------------------------


class TestTableSerializer:
    def test_normalizes_pads_and_escapes(self) -> None:
        rows = [["指标", "值"], ["准确率 | 召回率", "0.91\n0.88"], ["A"]]
        lines = to_markdown(rows).splitlines()
        assert lines[0] == "| 指标 | 值 |"
        assert lines[1] == "| --- | --- |"
        assert "准确率 \\| 召回率" in lines[2]
        assert "0.91<br>0.88" in lines[2]
        # 参差不齐的行补空串，否则列会错位
        assert lines[3] == "| A |  |"

    def test_empty_table(self) -> None:
        assert to_markdown([]) == ""
        assert to_markdown([["", ""], ["", ""]]) == ""

    def test_split_repeats_header(self) -> None:
        rows = [["编号", "内容"]] + [[str(i), "说明" * 30] for i in range(30)]
        pieces = split_table_markdown(rows, max_tokens=60)
        assert len(pieces) > 1
        for piece in pieces:
            assert piece.splitlines()[0] == "| 编号 | 内容 |"
            assert count_tokens(piece) <= 130

    def test_split_without_repeat_only_first_has_header(self) -> None:
        rows = [["编号", "内容"]] + [[str(i), "说明" * 30] for i in range(30)]
        pieces = split_table_markdown(rows, max_tokens=60, repeat_header=False)
        assert len(pieces) > 1
        assert pieces[0].startswith("| 编号 | 内容 |")
        assert "编号" not in pieces[1]

    def test_single_row_larger_than_budget_is_shrunk(self) -> None:
        rows = [["键", "值"], ["长", "x" * 5000]]
        pieces = split_table_markdown(rows, max_tokens=80)
        assert len(pieces) == 1
        assert count_tokens(pieces[0]) < 250


# --------------------------------------------------------------------------------------
# 2. 感知哈希
# --------------------------------------------------------------------------------------


class TestPhash:
    def test_identical_bytes(self) -> None:
        digest = phash(_png_bytes(seed=7))
        assert len(digest) == 16
        assert hamming_hex(digest, phash(_png_bytes(seed=7))) == 0

    def test_resize_and_recompress_are_close(self) -> None:
        original = Image.open(io.BytesIO(_png_bytes(size=(320, 320), seed=3)))
        resized = original.resize((180, 180))
        buffer = io.BytesIO()
        resized.convert("RGB").save(buffer, format="JPEG", quality=82)

        distance = hamming_hex(phash(_png_bytes(size=(320, 320), seed=3)), phash(buffer.getvalue()))
        assert distance <= 10

    def test_different_images_are_far(self) -> None:
        distance = hamming_hex(phash(_png_bytes(seed=1)), phash(_png_bytes(seed=2)))
        assert distance > 10

    def test_empty_hash_is_incomparable(self) -> None:
        assert hamming_hex("", "abcd") > 64


# --------------------------------------------------------------------------------------
# 3. 图片处理
# --------------------------------------------------------------------------------------


class TestImageProcessor:
    @pytest.mark.asyncio
    async def test_keeps_image_and_falls_back_to_context_caption(
        self, tmp_path: Path
    ) -> None:
        settings = _settings(tmp_path)
        path = _write_png(tmp_path / "ok.png", seed=11)
        image = _parsed_image(path, heading_path=["第一章", "1.1 架构"], context="架构如下")

        processor = ImageProcessor(settings, blobstore=BlobStore(settings))
        outcome = await processor.process(
            [image],
            kb_id="kb_testingest",
            doc_id="doc_test0001",
            doc_title="测试文档",
            caption_enabled=False,
        )

        assert len(outcome.kept) == 1
        asset = outcome.kept[0].asset
        assert asset.status is ImageStatus.DONE
        assert asset.caption_source is CaptionSource.CONTEXT_ONLY
        assert "1.1 架构" in asset.caption_text
        # 测试里同步查一次落盘结果即可，不需要再包一层线程
        assert Path(asset.file_path).is_file()  # noqa: ASYNC240
        assert asset.url.startswith(settings.app.static_url_prefix)

    @pytest.mark.asyncio
    async def test_small_image_filtered(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        path = _write_png(tmp_path / "small.png", size=(80, 80))
        outcome = await ImageProcessor(settings).process(
            [_parsed_image(path)],
            kb_id="kb_testingest",
            doc_id="doc_test0001",
            caption_enabled=False,
        )
        assert outcome.kept == []
        assert outcome.filtered[0].asset.filtered_reason is FilteredReason.SMALL

    @pytest.mark.asyncio
    async def test_blank_image_filtered(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.png"
        Image.new("RGB", (200, 200), (255, 255, 255)).save(path)
        settings = _settings(tmp_path)
        outcome = await ImageProcessor(settings).process(
            [_parsed_image(path)],
            kb_id="kb_testingest",
            doc_id="doc_test0001",
            caption_enabled=False,
        )
        assert outcome.filtered[0].asset.filtered_reason is FilteredReason.BLANK

    @pytest.mark.asyncio
    async def test_decorative_image_filtered(self, tmp_path: Path) -> None:
        # 黑白棋盘：灰度标准差很高（不是空白），但颜色种类只有 2 种。
        image = Image.new("RGB", (240, 240), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        for y in range(0, 240, 24):
            for x in range(0, 240, 24):
                if (x // 24 + y // 24) % 2 == 0:
                    draw.rectangle([x, y, x + 23, y + 23], fill=(255, 255, 255))
        path = tmp_path / "checker.png"
        image.save(path)

        settings = _settings(tmp_path)
        outcome = await ImageProcessor(settings).process(
            [_parsed_image(path)],
            kb_id="kb_testingest",
            doc_id="doc_test0001",
            caption_enabled=False,
        )
        assert outcome.filtered[0].asset.filtered_reason is FilteredReason.DECORATIVE

    @pytest.mark.asyncio
    async def test_duplicate_inside_document(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        path = _write_png(tmp_path / "dup.png", seed=5)
        # 上下文 caption 非空，图片才会被标成 DONE —— 没有任何可检索文本的
        # 图片在真实的流水线里本来就不该算「保留」
        first = _parsed_image(path, order=1, heading_path=["第一章"], context="上方正文")
        second = _parsed_image(path, order=2, heading_path=["第一章"], context="上方正文")
        outcome = await ImageProcessor(settings).process(
            [first, second],
            kb_id="kb_testingest",
            doc_id="doc_test0001",
            caption_enabled=False,
        )
        assert len(outcome.kept) == 1
        assert outcome.filtered[0].asset.filtered_reason is FilteredReason.DUPLICATE

    @pytest.mark.asyncio
    async def test_duplicate_against_existing_kb_image(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        path = _write_png(tmp_path / "known.png", seed=6)
        known = phash(path.read_bytes())
        outcome = await ImageProcessor(settings).process(
            [_parsed_image(path)],
            kb_id="kb_testingest",
            doc_id="doc_test0001",
            existing_phashes={known: "img_other"},
            caption_enabled=False,
        )
        assert outcome.kept == []
        assert outcome.filtered[0].asset.filtered_reason is FilteredReason.DUPLICATE

    @pytest.mark.asyncio
    async def test_over_quota(self, tmp_path: Path) -> None:
        settings = _settings(
            tmp_path, blob=BlobSettings(images_dir=tmp_path / "images", max_images_per_doc=1)
        )
        first = _write_png(tmp_path / "a.png", seed=21)
        second = _write_png(tmp_path / "b.png", seed=22)
        outcome = await ImageProcessor(settings).process(
            [
                _parsed_image(first, order=1, heading_path=["第一章"]),
                _parsed_image(second, order=2, heading_path=["第一章"]),
            ],
            kb_id="kb_testingest",
            doc_id="doc_test0001",
            caption_enabled=False,
        )
        assert len(outcome.kept) == 1
        assert outcome.filtered[0].asset.filtered_reason is FilteredReason.OVER_QUOTA

    @pytest.mark.asyncio
    async def test_broken_image_is_failed_not_fatal(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        broken = ParsedImage(
            image_id="img_broken0001",
            order=1,
            data=b"this is not an image",
            ext="png",
            mime_type="image/png",
            sha256=hashlib.sha256(b"this is not an image").hexdigest(),
        )
        outcome = await ImageProcessor(settings).process(
            [broken],
            kb_id="kb_testingest",
            doc_id="doc_test0001",
            caption_enabled=False,
        )
        assert outcome.failed[0].asset.status is ImageStatus.FAILED
        assert outcome.warnings


# --------------------------------------------------------------------------------------
# 4. 解析归一
# --------------------------------------------------------------------------------------


class TestParser:
    def test_heading_stack_and_table(self) -> None:
        result = LoadResult(
            title="示例",
            blocks=[
                RawBlock(text="第一章", heading_level=1, order=0),
                RawBlock(text="正文 A", order=1),
                RawBlock(text="1.1 小节", heading_level=2, order=2),
                RawBlock(
                    text="",
                    table=[["列", "值"], ["a", "1"]],
                    block_type=BlockType.TABLE,
                    order=3,
                ),
                RawBlock(text="正文 B", order=4),
            ],
        )
        parsed = parse_document(result, doc_id="doc_test0001", kb_id="kb_testingest")

        assert parsed.title == "示例"
        assert [block.heading_path for block in parsed.blocks] == [
            ["第一章"],
            ["第一章", "1.1 小节"],
            ["第一章", "1.1 小节"],
        ]
        assert parsed.blocks[1].block_type is BlockType.TABLE
        assert "| 列 | 值 |" in parsed.blocks[1].text

    def test_image_is_anchored_to_preceding_block(self) -> None:
        image = RawImage(data=_png_bytes(seed=9), order=2)
        result = LoadResult(
            blocks=[
                RawBlock(text="第一章", heading_level=1, order=0),
                RawBlock(text="架构图如下，展示三路召回。", order=1),
            ],
            images=[image],
        )
        parsed = parse_document(result, doc_id="doc_test0001", kb_id="kb_testingest")

        assert len(parsed.images) == 1
        anchored = parsed.images[0]
        assert anchored.heading_path == ["第一章"]
        assert anchored.anchor_block_id == "b1"
        assert anchored.context_text == "架构图如下，展示三路召回。"
        assert parsed.blocks[0].image_ids == [anchored.image_id]
        assert anchored.image_id == image_id_for(
            "kb_testingest", hashlib.sha256(image.data).hexdigest()
        )

    def test_heading_clears_anchor(self) -> None:
        result = LoadResult(
            blocks=[
                RawBlock(text="第一章", heading_level=1, order=0),
                RawBlock(text="上一节正文", order=1),
                RawBlock(text="第二章", heading_level=1, order=2),
            ],
            images=[RawImage(data=_png_bytes(seed=10), order=3)],
        )
        parsed = parse_document(result, doc_id="doc_test0001", kb_id="kb_testingest")
        assert parsed.images[0].anchor_block_id is None
        assert parsed.images[0].context_text == ""

    def test_duplicate_images_are_deduplicated(self) -> None:
        payload = _png_bytes(seed=12)
        result = LoadResult(
            blocks=[RawBlock(text="正文", order=0)],
            images=[
                RawImage(data=payload, order=1),
                RawImage(data=payload, order=2),
            ],
        )
        parsed = parse_document(result, doc_id="doc_test0001", kb_id="kb_testingest")
        assert len(parsed.images) == 1
        assert any("重复图片" in warning for warning in parsed.warnings)

    def test_empty_document_warns(self) -> None:
        parsed = parse_document(LoadResult(), doc_id="doc_test0001", kb_id="kb_testingest")
        assert parsed.has_content is False
        assert any("没有任何可入库" in warning for warning in parsed.warnings)


# --------------------------------------------------------------------------------------
# 5. 切分
# --------------------------------------------------------------------------------------


def _chunk_settings(**overrides: Any) -> ChunkingSettings:
    values: dict[str, Any] = {
        "min_tokens": 20,
        "max_tokens": 120,
        "overlap_tokens": 0,
        "min_chunk_chars": 20,
        "table_max_tokens": 120,
    }
    values.update(overrides)
    return ChunkingSettings(**values)


def _doc(blocks: list[RawBlock], images: list[RawImage] | None = None):
    return parse_document(
        LoadResult(blocks=blocks, images=images or []),
        doc_id="doc_test0001",
        kb_id="kb_testingest",
    )


class TestChunker:
    def test_heading_path_prefix_and_boundary(self) -> None:
        doc = _doc(
            [
                RawBlock(text="第一章", heading_level=1, order=0),
                RawBlock(text="属于第一章的正文，长度足够形成块。" * 2, order=1),
                RawBlock(text="第二章", heading_level=1, order=2),
                RawBlock(text="属于第二章的正文，内容与上一节无关。" * 2, order=3),
            ]
        )
        outcome = chunk_document(doc, settings=_chunk_settings())
        assert len(outcome.chunks) == 2
        assert outcome.chunks[0].text.startswith("第一章\n\n")
        assert "第一章" in outcome.chunks[0].heading_path
        assert outcome.chunks[1].text.startswith("第二章\n\n")

    def test_chunk_ids_are_deterministic(self) -> None:
        doc = _doc(
            [
                RawBlock(text="第一章", heading_level=1, order=0),
                RawBlock(text="一段用于测试幂等的正文内容。" * 3, order=1),
            ]
        )
        first = chunk_document(doc, settings=_chunk_settings())
        second = chunk_document(doc, settings=_chunk_settings())
        assert [chunk.chunk_id for chunk in first.chunks] == [
            chunk.chunk_id for chunk in second.chunks
        ]

    def test_image_ids_are_limited_to_kept(self) -> None:
        image = RawImage(data=_png_bytes(seed=13), order=1)
        doc = _doc(
            [RawBlock(text="图片上方有一段正文，长度足够。" * 2, order=0)],
            images=[image],
        )
        kept = doc.images[0].image_id
        outcome = chunk_document(
            doc, settings=_chunk_settings(), keep_image_ids={kept}
        )
        assert outcome.chunks[0].image_ids == [kept]

        dropped = chunk_document(doc, settings=_chunk_settings(), keep_image_ids=set())
        assert dropped.chunks[0].image_ids == []

    def test_caption_chunk_and_neighbors(self) -> None:
        doc = _doc([RawBlock(text="正文段落，长度足够形成独立的分块。" * 2, order=0)])
        asset = _asset(caption_text="架构图：查询经三路召回后由 RRF 融合。")
        outcome = chunk_document(
            doc, settings=_chunk_settings(), caption_images=[asset]
        )
        caption = outcome.chunks[-1]
        assert caption.block_type is BlockType.IMAGE_CAPTION
        assert caption.image_ids == [asset.image_id]
        assert caption.source_ref["synthetic"] is True
        # caption 是合成块，不参与前后链表
        assert outcome.chunks[0].next_chunk_id is None

    def test_short_fragment_is_dropped(self) -> None:
        doc = _doc([RawBlock(text="短", order=0)])
        outcome = chunk_document(doc, settings=_chunk_settings())
        assert outcome.chunks == []
        assert any("过短" in warning for warning in outcome.warnings)

    def test_large_table_splits_and_repeats_header(self) -> None:
        rows = [["编号", "说明"]] + [[str(i), "内容" * 25] for i in range(40)]
        doc = _doc(
            [
                RawBlock(text="表格", heading_level=1, order=0),
                RawBlock(text="", table=rows, block_type=BlockType.TABLE, order=1),
            ]
        )
        outcome = chunk_document(doc, settings=_chunk_settings(table_max_tokens=80))
        tables = [chunk for chunk in outcome.chunks if chunk.block_type is BlockType.TABLE]
        assert len(tables) >= 2
        for chunk in tables:
            assert "| 编号 | 说明 |" in chunk.text

    def test_small_table_stays_atomic(self) -> None:
        rows = [["列", "值"], ["a", "1"], ["b", "2"]]
        doc = _doc([RawBlock(text="", table=rows, block_type=BlockType.TABLE, order=0)])
        outcome = chunk_document(doc, settings=_chunk_settings())
        assert len(outcome.chunks) == 1
        assert outcome.chunks[0].block_type is BlockType.TABLE
        assert "| a | 1 |" in outcome.chunks[0].text

    def test_code_fences_close_on_every_piece(self) -> None:
        code = "\n".join(f"value_{i} = compute({i})" for i in range(40))
        doc = _doc(
            [RawBlock(text=f"```python\n{code}\n```", block_type=BlockType.CODE, order=0)]
        )
        outcome = chunk_document(doc, settings=_chunk_settings(max_tokens=60))
        assert len(outcome.chunks) >= 2
        for chunk in outcome.chunks:
            text = chunk.text
            assert text.startswith("```")
            assert text.endswith("```")
            assert chunk.block_type is BlockType.CODE


# --------------------------------------------------------------------------------------
# 6. Loader（七种格式 + 注册表）
# --------------------------------------------------------------------------------------


class TestLoaders:
    def test_markdown_loader(self, tmp_path: Path) -> None:
        path = _markdown_doc(tmp_path)
        result = get_registry().loader_for(path).load(path)

        assert result.title == "多模态检索指南"
        assert result.images
        assert any(block.heading_level == 1 for block in result.blocks)
        assert any(block.heading_level == 2 for block in result.blocks)
        assert any(block.table for block in result.blocks)
        assert any(block.block_type is BlockType.CODE for block in result.blocks)

    def test_text_loader_detects_chinese_headings(self, tmp_path: Path) -> None:
        path = tmp_path / "faq.txt"
        path.write_text(
            "第一章 常见问题\n\n"
            "1.1 支持哪些格式\n\n"
            "支持 PDF、DOCX 与 Markdown。\n\n"
            "第二章 数据\n\n"
            "数据都保存在本机。\n",
            encoding="utf-8",
        )
        result = get_registry().loader_for(path).load(path)
        headings = [block.text for block in result.blocks if block.heading_level]
        assert "第一章 常见问题" in headings
        assert "1.1 支持哪些格式" in headings
        assert result.title == "第一章 常见问题"

    def test_csv_loader_becomes_table(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.csv"
        path.write_text("指标,数值\n召回率,0.91\n准确率,0.88\n", encoding="utf-8")
        result = get_registry().loader_for(path).load(path)
        assert len(result.blocks) == 1
        assert result.blocks[0].table == [
            ["指标", "数值"],
            ["召回率", "0.91"],
            ["准确率", "0.88"],
        ]

    def test_html_loader(self, tmp_path: Path) -> None:
        _write_png(tmp_path / "fig.png", seed=31)
        path = tmp_path / "page.html"
        path.write_text(
            "<html><head><title>网页示例</title></head><body>"
            "<h1>第一章</h1><p>正文段落。</p>"
            "<h2>1.1 表格</h2>"
            "<table><tr><th>列</th><th>值</th></tr><tr><td>a</td><td>1</td></tr></table>"
            '<p><img src="fig.png" alt="示意图"></p>'
            "<nav>导航不应进入正文</nav>"
            "</body></html>",
            encoding="utf-8",
        )
        result = get_registry().loader_for(path).load(path)
        assert result.title == "网页示例"
        assert any(block.heading_level == 1 for block in result.blocks)
        assert any(block.table for block in result.blocks)
        assert result.images
        assert all("导航不应进入正文" not in block.text for block in result.blocks)

    def test_docx_loader(self, tmp_path: Path) -> None:
        from docx import Document as open_docx

        _write_png(tmp_path / "pic.png", seed=41)
        document = open_docx()
        document.core_properties.title = "Word 示例"
        document.add_heading("第一章 架构", level=1)
        document.add_paragraph("这是一段正文，介绍了系统的组成部分。")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "模块"
        table.cell(0, 1).text = "职责"
        table.cell(1, 0).text = "检索"
        table.cell(1, 1).text = "三路召回"
        document.add_picture(str(tmp_path / "pic.png"))
        path = tmp_path / "demo.docx"
        document.save(str(path))

        result = get_registry().loader_for(path).load(path)
        assert result.title == "Word 示例"
        assert any(block.heading_level == 1 for block in result.blocks)
        assert any(block.table for block in result.blocks)
        assert result.images

    def test_pptx_loader(self, tmp_path: Path) -> None:
        from pptx import Presentation
        from pptx.util import Inches

        presentation = Presentation()
        presentation.core_properties.title = "PPT 示例"
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "检索模块"
        slide.placeholders[1].text = "三路召回与 RRF 融合"
        slide.shapes.add_picture(
            str(_write_png(tmp_path / "slide.png", seed=51)),
            Inches(5),
            Inches(3),
        )
        path = tmp_path / "deck.pptx"
        presentation.save(str(path))

        result = get_registry().loader_for(path).load(path)
        assert result.page_count == 1
        assert result.title == "PPT 示例"
        assert any(block.text == "检索模块" for block in result.blocks)
        assert result.images

    def test_xlsx_loader(self, tmp_path: Path) -> None:
        from openpyxl import Workbook

        workbook = Workbook()
        workbook.properties.title = "Excel 示例"
        sheet = workbook.active
        sheet.title = "指标"
        sheet.append(["指标", "数值"])
        sheet.append(["召回率", 0.91])
        sheet.append(["准确率", 0.88])
        path = tmp_path / "metrics.xlsx"
        workbook.save(str(path))

        result = get_registry().loader_for(path).load(path)
        assert result.page_count == 1
        assert any(block.table for block in result.blocks)
        table = next(block.table for block in result.blocks if block.table)
        assert table[0] == ["指标", "数值"]
        assert table[1][0] == "召回率"

    def test_pdf_loader(self, tmp_path: Path) -> None:
        import fitz

        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Multimodal retrieval evaluation", fontsize=14)
        page.insert_text((72, 100), "Three-way recall with RRF fusion.", fontsize=11)
        path = tmp_path / "demo.pdf"
        document.save(str(path))
        document.close()

        result = get_registry().loader_for(path).load(path)
        assert result.page_count == 1
        joined = "\n".join(block.text for block in result.blocks)
        assert "Multimodal retrieval" in joined

    def test_image_loader(self, tmp_path: Path) -> None:
        path = _write_png(tmp_path / "standalone.png", size=(320, 200), seed=61)
        result = get_registry().loader_for(path).load(path)
        assert len(result.images) == 1
        assert result.images[0].ext == "png"
        assert result.metadata["width"] == 320
        assert result.title == "standalone"

    def test_registry_rejects_with_actionable_hint(self, tmp_path: Path) -> None:
        legacy = tmp_path / "old.doc"
        legacy.write_bytes(b"\xd0\xcf\x11\xe0")
        with pytest.raises(UnsupportedFileTypeError) as excinfo:
            get_registry().loader_for(legacy)
        assert "另存为 .docx" in str(excinfo.value)

    def test_video_is_explicitly_unsupported(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        with pytest.raises(UnsupportedFileTypeError) as excinfo:
            get_registry().loader_for(video)
        assert "不支持视频输入" in str(excinfo.value)

    def test_remote_image_is_skipped_with_warning(self, tmp_path: Path) -> None:
        warnings: list[str] = []
        found = read_image_source(
            "https://example.com/a.png", tmp_path, warnings=warnings, label="测试"
        )
        assert found is None
        assert warnings and "远程图片" in warnings[0]


# --------------------------------------------------------------------------------------
# 7. 目录展开
# --------------------------------------------------------------------------------------


class TestCollectFiles:
    def test_expands_directories_and_skips_noise(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        nested = raw / "nested"
        hidden = raw / ".git"
        nested.mkdir(parents=True)
        hidden.mkdir()

        (raw / "a.md").write_text("# A", encoding="utf-8")
        (nested / "b.txt").write_text("正文", encoding="utf-8")
        (hidden / "secret.md").write_text("# 不应导入", encoding="utf-8")
        (raw / ".DS_Store").write_bytes(b"\x00")
        (raw / "~$lock.docx").write_bytes(b"\x00")
        (raw / "archive.zip").write_bytes(b"PK")

        accepted, skipped = collect_files([raw])
        names = [path.name for path in accepted]
        assert names == ["a.md", "b.txt"]
        # .DS_Store 在「隐藏文件」这一步就被排除了，不会进入 skipped 列表 ——
        # 它和 .git/ 里的内容一样，属于「根本没被看见」
        assert {path.name for path in skipped} == {"~$lock.docx", "archive.zip"}
        assert ".DS_Store" not in names

    def test_missing_path_is_kept_for_explicit_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.md"
        accepted, skipped = collect_files([missing])
        assert accepted == [missing]
        assert skipped == []


# --------------------------------------------------------------------------------------
# 8. 流水线端到端（离线）
# --------------------------------------------------------------------------------------


class TestPipelineOffline:
    @pytest.mark.asyncio
    async def test_ingest_without_embedding(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        path = _markdown_doc(raw)

        async with _env(tmp_path, providers=ProviderSet()) as env:
            kb = await env.pipeline.ensure_kb("离线知识库")
            outcome = await env.pipeline.ingest_file(
                path, kb, options=IngestOptions(embed=False, caption=False)
            )

            assert outcome.status is DocStatus.DONE
            assert outcome.chunks > 0
            assert outcome.images == 1
            assert outcome.text_vectors == 0
            assert outcome.image_vectors == 0

            document = await env.docstore.require_document(outcome.doc_id)
            assert document.status is DocStatus.DONE
            assert document.chunk_count == outcome.chunks
            assert document.image_count == 1

            chunks = await env.docstore.get_document_chunks(outcome.doc_id)
            assert any(
                chunk.block_type is BlockType.IMAGE_CAPTION for chunk in chunks
            )
            caption = next(
                chunk for chunk in chunks if chunk.block_type is BlockType.IMAGE_CAPTION
            )
            assert caption.image_ids

            hits = await env.bm25store.search(kb.kb_id, "三路召回", top_k=5)
            assert hits
            assert all(hit.score > 0 for hit in hits)

            # 没有 embedding 时两个向量空间都不该被写入
            assert await env.vectorstore.count(kb.text_collection) == 0
            assert await env.vectorstore.count(kb.image_collection) == 0

    @pytest.mark.asyncio
    async def test_second_run_is_skipped_as_duplicate(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        path = _markdown_doc(raw)

        async with _env(tmp_path, providers=ProviderSet()) as env:
            kb = await env.pipeline.ensure_kb("幂等知识库")
            options = IngestOptions(embed=False, caption=False)
            first = await env.pipeline.ingest_file(path, kb, options=options)
            second = await env.pipeline.ingest_file(path, kb, options=options)

            assert first.status is DocStatus.DONE
            assert second.status is DocStatus.SKIPPED_DUPLICATE
            assert second.is_duplicate and second.ok
            assert second.doc_id == first.doc_id
            assert second.chunks == first.chunks

    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        path = _markdown_doc(raw)

        async with _env(tmp_path, providers=ProviderSet()) as env:
            kb = await env.pipeline.ensure_kb("预览知识库")
            outcome = await env.pipeline.ingest_file(
                path,
                kb,
                options=IngestOptions(embed=False, caption=False, dry_run=True),
            )

            assert outcome.dry_run is True
            assert outcome.chunks > 0
            assert any("dry-run" in warning for warning in outcome.warnings)
            documents = await env.docstore.list_documents(kb.kb_id)
            assert documents == []
            assert await env.vectorstore.count(kb.text_collection) == 0
            assert await env.vectorstore.count(kb.image_collection) == 0

    @pytest.mark.asyncio
    async def test_corrupt_pdf_fails_alone(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"not a real pdf")

        async with _env(tmp_path, providers=ProviderSet()) as env:
            kb = await env.pipeline.ensure_kb("容错知识库")
            outcome = await env.pipeline.ingest_file(
                broken, kb, options=IngestOptions(embed=False, caption=False)
            )

            assert outcome.status is DocStatus.FAILED
            assert outcome.error
            document = await env.docstore.require_document(outcome.doc_id)
            assert document.status is DocStatus.FAILED
            assert document.error

    @pytest.mark.asyncio
    async def test_unsupported_file_reports_hint(self, tmp_path: Path) -> None:
        legacy = tmp_path / "old.doc"
        legacy.write_bytes(b"\xd0\xcf\x11\xe0")

        async with _env(tmp_path, providers=ProviderSet()) as env:
            kb = await env.pipeline.ensure_kb("提示知识库")
            outcome = await env.pipeline.ingest_file(
                legacy, kb, options=IngestOptions(embed=False, caption=False)
            )
            assert outcome.status is DocStatus.FAILED
            assert "另存为 .docx" in (outcome.error or "")


class TestPipelineWithVectors:
    @pytest.mark.asyncio
    async def test_embeds_both_spaces(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        path = _markdown_doc(raw)

        async with _env(tmp_path, providers=_providers()) as env:
            kb = await env.pipeline.ensure_kb("向量知识库")
            outcome = await env.pipeline.ingest_file(
                path, kb, options=IngestOptions(embed=True, caption=False)
            )

            assert outcome.status is DocStatus.DONE
            assert outcome.text_vectors == outcome.chunks
            assert outcome.image_vectors == 1
            assert await env.vectorstore.count(kb.text_collection) == outcome.chunks
            assert await env.vectorstore.count(kb.image_collection) == 1

            # 图像空间里存的是多模态向量，查询必须用同一空间（铁律 3）
            hits = await env.vectorstore.query(
                kb.image_collection,
                [1.0] + [0.0] * 15,
                top_k=3,
                space="image",
            )
            assert len(hits) == 1
            assert hits[0].metadata["doc_id"] == outcome.doc_id

    @pytest.mark.asyncio
    async def test_reindex_is_idempotent(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        path = _markdown_doc(raw)

        async with _env(tmp_path, providers=_providers()) as env:
            kb = await env.pipeline.ensure_kb("重导知识库")
            first = await env.pipeline.ingest_file(
                path, kb, options=IngestOptions(embed=True, caption=False)
            )
            first_ids = {
                chunk.chunk_id
                for chunk in await env.docstore.get_document_chunks(first.doc_id)
            }

            second = await env.pipeline.ingest_file(
                path,
                kb,
                options=IngestOptions(embed=True, caption=False, reindex=True),
            )
            second_ids = {
                chunk.chunk_id
                for chunk in await env.docstore.get_document_chunks(second.doc_id)
            }

            assert second.status is DocStatus.DONE
            assert second_ids == first_ids
            assert await env.vectorstore.count(kb.text_collection) == len(first_ids)
            assert await env.vectorstore.count(kb.image_collection) == 1

    @pytest.mark.asyncio
    async def test_snapshot_mismatch_is_refused(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        path = _markdown_doc(raw)

        async with _env(tmp_path, providers=_providers()) as env:
            # 建一个「用另一个维度建好的」知识库：用 4 维文本空间，
            # 而当前 provider 是 8 维。写入前必须被拦住。
            kb = await env.docstore.create_kb(
                "旧维度知识库",
                kb_id="kb_oldsnapshot01",
                text_collection=text_collection("kb_oldsnapshot01"),
                image_collection=image_collection("kb_oldsnapshot01"),
                text_dim=4,
                image_dim=16,
                llm_model="deepseek-flash",
                text_embed_model=TEXT_MODEL,
                image_embed_model=IMAGE_MODEL,
            )
            outcome = await env.pipeline.ingest_file(
                path, kb, options=IngestOptions(embed=True, caption=False)
            )

            assert outcome.status is DocStatus.FAILED
            assert "维度" in (outcome.error or "")
            # 半成品向量不能留在库里
            assert await env.vectorstore.count(kb.text_collection) == 0

    @pytest.mark.asyncio
    async def test_missing_image_provider_degrades(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        path = _markdown_doc(raw)

        async with _env(tmp_path, providers=_providers(text=True, image=False)) as env:
            kb = await env.pipeline.ensure_kb("降级知识库")
            outcome = await env.pipeline.ingest_file(
                path, kb, options=IngestOptions(embed=True, caption=False)
            )

            assert outcome.status is DocStatus.DONE
            assert outcome.image_vectors == 0
            assert "image_embed_unavailable" in outcome.degraded_reasons
            # 图片仍以 caption 形式进入文本空间
            assert outcome.text_vectors == outcome.chunks
            assert await env.vectorstore.count(kb.image_collection) == 0
            caption = next(
                chunk
                for chunk in await env.docstore.get_document_chunks(outcome.doc_id)
                if chunk.block_type is BlockType.IMAGE_CAPTION
            )
            assert caption.image_ids

