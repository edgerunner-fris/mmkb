"""Loader 抽象、通用数据结构与按扩展名分发。

为什么结构这么「薄」
------------------
Loader 只做一件事：把一种格式摊平成 **顺序化的块与图片**。标题层级、图片锚定、
表格序列化、切分全部留给后面的模块 —— 那些步骤对所有格式是同一套逻辑，
塞进每个 loader 里只会被复制七遍，然后以七种略有差异的方式各错一次。

顺序契约（非常重要）
------------------
``RawBlock.order`` 与 ``RawImage.order`` 取自**同一个递增计数器**，
所以「第 3 个块之后出现了一张图」这件事只靠 order 就能表达。
parser 依赖这一点把图片锚定到它上方最近的文本块；如果两者各用一个计数器，
锚定结果会随机漂移到别的小节去，而且不会报错。

同一规则也用于 PDF 里「表格区域内的文字」与「独立图片」的定序。
"""

from __future__ import annotations

import hashlib
import mimetypes
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from app.core.errors import IngestError, UnsupportedFileTypeError
from app.core.schemas import BlockType

__all__ = [
    "BaseLoader",
    "LoadResult",
    "LoaderRegistry",
    "OrderCounter",
    "RawBlock",
    "RawImage",
    "clean_block_text",
    "clean_inline_text",
    "detect_mime",
    "file_sha256",
    "get_registry",
    "headings_to_markdown",
    "read_image_source",
    "read_text_file",
    "suffix_of",
]

# --------------------------------------------------------------------------------------
# 文本清洗
# --------------------------------------------------------------------------------------

#: 控制字符（保留 \t \n \r）。PDF 解析出的文本里偶尔夹着 \x00 与 \x0c（分页符），
#: 它们会让后续的 JSON 序列化与 token 计数一起变得诡异。
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
#: 行内空白。  是不换行空格（网页复制来的文本里很常见），
#: 　 是全角空格 —— 两者都该按普通空格处理，否则分词器会把它们当成词的一部分。
_INLINE_SPACE_RE = re.compile(r"[ \t 　]+")


def clean_inline_text(text: str) -> str:
    """清洗单行文本：去控制字符、折叠空白、去首尾空白。"""
    if not text:
        return ""
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _INLINE_SPACE_RE.sub(" ", text).strip()


def clean_block_text(text: str) -> str:
    """清洗多行文本：保留段落内的换行，但去掉空行与行尾空白。

    保留换行是有意义的：Markdown/HTML 里的列表、代码、换行敏感的结构
    都要靠它还原；而空行对检索没有价值，只会虚增 token。
    """
    if not text:
        return ""
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_INLINE_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def headings_to_markdown(level: int, text: str) -> str:
    """把标题还原成 Markdown 形式。仅用于源引用展示，不参与切分。"""
    return f"{'#' * max(1, min(6, level))} {text}"


# --------------------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class RawBlock:
    """Loader 输出的一块内容，尚未决定归属哪个标题。

    ``heading_level > 0`` 表示这是一个标题块，它的 ``text`` 会进标题栈而不是正文。
    ``table`` 非空表示这是结构化表格，优先用它而不是 ``text``。
    """

    text: str
    block_type: BlockType = BlockType.TEXT
    page: int | None = None
    order: int = 0
    heading_level: int = 0
    table: list[list[str]] | None = None
    source_ref: dict[str, Any] = field(default_factory=dict)

    @property
    def is_heading(self) -> bool:
        return self.heading_level > 0

    @property
    def is_table(self) -> bool:
        return self.table is not None


@dataclass(slots=True)
class RawImage:
    """从文档里抽出的一张图片（原始字节，未做任何清洗）。"""

    data: bytes
    order: int = 0
    page: int | None = None
    bbox: list[float] | None = None
    ext: str = "png"
    mime_type: str = "image/png"
    source_ref: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ext = self.ext.lower().lstrip(".") or "png"
        if self.mime_type == "image/png" and self.ext != "png":
            guessed = mimetypes.types_map.get(f".{self.ext}")
            if guessed:
                self.mime_type = guessed


@dataclass(slots=True)
class LoadResult:
    """一个文件的摊平结果。"""

    blocks: list[RawBlock] = field(default_factory=list)
    images: list[RawImage] = field(default_factory=list)
    page_count: int | None = None
    title: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.blocks and not self.images


class OrderCounter:
    """块与图片共享的递增序号。

    单独封装而不是用 ``len(blocks) + len(images)``：后者在「先收块再收图」
    的实现里永远得到单调但错位的结果，而错位的锚定不会报错。
    """

    __slots__ = ("_value",)

    def __init__(self, start: int = 0) -> None:
        self._value = start

    def next(self) -> int:
        value = self._value
        self._value += 1
        return value

    @property
    def value(self) -> int:
        return self._value


# --------------------------------------------------------------------------------------
# 文件工具
# --------------------------------------------------------------------------------------


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """流式计算文件哈希。大文件不整读进内存。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise IngestError(
            f"读取文件失败：{exc}", detail={"path": str(path)}
        ) from exc
    return digest.hexdigest()


def suffix_of(path: Path) -> str:
    return path.suffix.lower()


def detect_mime(path: Path) -> str:
    """按扩展名猜 MIME。猜不出时退化为 octet-stream。

    刻意不用 ``python-magic``：它依赖系统 libmagic，在 Windows 与 slim 镜像上
    装起来都不稳定，而本项目对 MIME 的用途只有「告诉前端怎么预览」。
    """
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


#: 按顺序尝试的编码。中文技术文档的三大来源：
#: UTF-8（现代）、GB18030（Windows 中文环境）、UTF-16（记事本另存为）。
#: 不做 chardet 之类的统计猜测：猜错时的后果是**静默乱码**，
#: 而按固定顺序试解码，全都失败才报错，行为可预期。
_TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "big5", "utf-16")


def read_text_file(path: Path, *, max_bytes: int = 0) -> tuple[str, str]:
    """读取文本文件，返回 ``(文本, 实际使用的编码)``。

    ``max_bytes > 0`` 时只读前 ``max_bytes`` 字节 —— 用于给超大日志文件兜底。
    按字节截断可能切在多字节字符中间，因此解码时用 ``errors="replace"``。
    """
    try:
        raw = path.read_bytes() if max_bytes <= 0 else _read_head(path, max_bytes)
    except OSError as exc:
        raise IngestError(f"读取文件失败：{exc}", detail={"path": str(path)}) from exc

    if not raw:
        return "", "utf-8"

    for encoding in _TEXT_ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError):
            continue
    # 全试一遍都失败：用 latin-1 兜底（它对任意字节都不会抛），并标记出来。
    return raw.decode("latin-1", errors="replace"), "latin-1"


def _read_head(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(max_bytes)


#: 位图扩展名。SVG 刻意不在其中：它不是位图，Pillow 解不了，
#: 送进多模态 embedding 只会得到一张空白图。
_RASTER_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
)

_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+)?(?:;[^,]*)?;base64,(?P<payload>.*)$",
    re.DOTALL,
)


def read_image_source(
    src: str,
    base_dir: Path,
    *,
    warnings: list[str] | None = None,
    label: str = "",
) -> tuple[bytes, str] | None:
    """把 ``<img src>`` / ``![](...)`` 解析成 ``(字节, 扩展名)``。

    支持三种来源：``data:`` URI、本地相对/绝对路径、以及**跳过**远程 URL。
    远程 URL 不做下载 —— 一次导入几十个外链图片意味着几十次不可控的网络请求，
    在离线构建、内网部署、以及「链接已失效」三种情况下都会把导入卡住。
    跳过时会写一条 warning，让人知道少了几张图，而不是以为解析漏了。
    """
    text = (src or "").strip().strip("'\"")
    if not text:
        return None

    if text.startswith(("http://", "https://", "//")):
        if warnings is not None:
            warnings.append(f"{label}引用了远程图片，未下载：{text[:120]}")
        return None

    match = _DATA_URI_RE.match(text)
    if match:
        import base64

        try:
            payload = base64.b64decode(match.group("payload"), validate=False)
        except Exception:  # noqa: BLE001
            if warnings is not None:
                warnings.append(f"{label}内嵌图片的 base64 无法解码")
            return None
        mime = (match.group("mime") or "image/png").lower()
        ext = "jpg" if mime.endswith("jpeg") else mime.rsplit("/", 1)[-1]
        return payload, ext

    # 去掉 query / fragment：`pic.png?raw=true` 与 `pic.png#frag` 都指向同一个文件
    cleaned = text.split("#", 1)[0].split("?", 1)[0]
    from urllib.parse import unquote

    cleaned = unquote(cleaned)
    candidate = Path(cleaned)
    path = candidate if candidate.is_absolute() else (base_dir / candidate)
    try:
        path = path.resolve()
    except OSError:
        return None

    if not path.is_file():
        if warnings is not None:
            warnings.append(f"{label}引用的图片不存在：{text[:120]}")
        return None

    suffix = path.suffix.lower()
    if suffix not in _RASTER_SUFFIXES:
        if warnings is not None:
            warnings.append(
                f"{label}引用的图片格式 {suffix or '（未知）'} 不是位图，已跳过：{text[:120]}"
            )
        return None
    try:
        return path.read_bytes(), suffix.lstrip(".")
    except OSError:
        if warnings is not None:
            warnings.append(f"{label}读取图片失败：{text[:120]}")
        return None


# --------------------------------------------------------------------------------------
# 抽象
# --------------------------------------------------------------------------------------


class BaseLoader(ABC):
    """一种文件格式的 loader。

    ``load`` 是**同步**的：解析是 CPU 密集的，调用方负责 ``asyncio.to_thread``。
    这样 loader 内部不必关心事件循环，也方便单独测试。
    """

    #: 该 loader 负责的扩展名（含点，全小写）
    suffixes: ClassVar[frozenset[str]] = frozenset()
    #: 该 loader 的展示名，用于日志与错误信息
    kind: ClassVar[str] = "base"
    #: 该 loader 的主 MIME
    mime_type: ClassVar[str] = "application/octet-stream"

    @abstractmethod
    def load(self, path: Path) -> LoadResult:
        """解析文件。失败时抛 :class:`~app.core.errors.IngestError`。"""

    # -- 子类常用的小工具 ------------------------------------------------------------

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.exists():
            raise IngestError(f"文件不存在：{path}", detail={"path": str(path)})
        if not path.is_file():
            raise IngestError(f"不是普通文件：{path}", detail={"path": str(path)})

    @staticmethod
    def _check_size(path: Path, max_bytes: int) -> None:
        if max_bytes <= 0:
            return
        size = path.stat().st_size
        if size > max_bytes:
            raise IngestError(
                f"文件超过大小上限：{size / 1024 / 1024:.1f} MB > "
                f"{max_bytes / 1024 / 1024:.0f} MB",
                detail={"path": str(path), "size": size, "max": max_bytes},
            )


# --------------------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------------------

#: 扩展名 → (模块名, 类名)。**惰性导入**：装一个只用 TXT 的部署不该被迫
#: 在 import 期加载 fitz / python-docx / openpyxl。
_LOADER_SPECS: dict[str, tuple[str, str]] = {
    # PDF
    ".pdf": ("app.ingestion.loaders.pdf_loader", "PdfLoader"),
    # Word
    ".docx": ("app.ingestion.loaders.office_loader", "DocxLoader"),
    # PowerPoint
    ".pptx": ("app.ingestion.loaders.office_loader", "PptxLoader"),
    # Excel
    ".xlsx": ("app.ingestion.loaders.office_loader", "XlsxLoader"),
    ".xlsm": ("app.ingestion.loaders.office_loader", "XlsxLoader"),
    # Markdown
    ".md": ("app.ingestion.loaders.markdown_loader", "MarkdownLoader"),
    ".markdown": ("app.ingestion.loaders.markdown_loader", "MarkdownLoader"),
    # HTML
    ".html": ("app.ingestion.loaders.html_loader", "HtmlLoader"),
    ".htm": ("app.ingestion.loaders.html_loader", "HtmlLoader"),
    # 纯文本与源码
    ".txt": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".text": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".log": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".rst": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".csv": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".tsv": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".json": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".yaml": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".yml": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".toml": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".ini": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".xml": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".py": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".js": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".ts": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".java": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".go": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".rs": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".c": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".cpp": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".h": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".sql": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".sh": ("app.ingestion.loaders.text_loader", "TextLoader"),
    ".ps1": ("app.ingestion.loaders.text_loader", "TextLoader"),
    # 图片
    ".png": ("app.ingestion.loaders.image_loader", "ImageLoader"),
    ".jpg": ("app.ingestion.loaders.image_loader", "ImageLoader"),
    ".jpeg": ("app.ingestion.loaders.image_loader", "ImageLoader"),
    ".gif": ("app.ingestion.loaders.image_loader", "ImageLoader"),
    ".bmp": ("app.ingestion.loaders.image_loader", "ImageLoader"),
    ".webp": ("app.ingestion.loaders.image_loader", "ImageLoader"),
    ".tif": ("app.ingestion.loaders.image_loader", "ImageLoader"),
    ".tiff": ("app.ingestion.loaders.image_loader", "ImageLoader"),
}

#: 明确不支持、但要给出**可操作**提示的格式。
#: 只说「不支持」等于让使用者自己去猜下一步，这里直接告诉他怎么转。
_UNSUPPORTED_HINTS: dict[str, str] = {
    ".doc": "旧版 Word 二进制格式。请另存为 .docx 后再导入（Word/WPS 均可）。",
    ".ppt": "旧版 PowerPoint 二进制格式。请另存为 .pptx 后再导入。",
    ".xls": "旧版 Excel 二进制格式。请另存为 .xlsx 后再导入。",
    ".rtf": "RTF 格式未实现解析。请另存为 .docx 或 .txt。",
    ".epub": "EPUB 未实现解析。可先用 calibre 转成 PDF 或 TXT。",
    ".wps": "WPS 私有格式未实现解析。请另存为 .docx。",
    ".mp4": "不支持视频输入 —— 生成模型不接受视频，本项目也没有视频链路。",
    ".mp3": "不支持音频输入。",
    ".zip": "请先解压，再逐个导入其中的文件。",
}


class LoaderRegistry:
    """扩展名 → loader 实例。实例无状态，可复用。"""

    def __init__(self) -> None:
        self._specs: dict[str, tuple[str, str]] = dict(_LOADER_SPECS)
        self._instances: dict[str, BaseLoader] = {}

    def register(
        self, suffixes: frozenset[str] | set[str], module: str, class_name: str
    ) -> None:
        for suffix in suffixes:
            self._specs[suffix.lower()] = (module, class_name)

    def supports(self, path: Path) -> bool:
        return suffix_of(path) in self._specs

    def supported_suffixes(self) -> list[str]:
        return sorted(self._specs)

    def hint_for(self, path: Path) -> str | None:
        return _UNSUPPORTED_HINTS.get(suffix_of(path))

    def loader_for(self, path: Path) -> BaseLoader:
        suffix = suffix_of(path)
        loader = self._instances.get(suffix)
        if loader is not None:
            return loader

        spec = self._specs.get(suffix)
        if spec is None:
            hint = _UNSUPPORTED_HINTS.get(suffix)
            message = f"不支持的文件类型 {suffix or '（无扩展名）'}：{path.name}"
            if hint:
                message = f"{message}。{hint}"
            raise UnsupportedFileTypeError(
                message,
                detail={
                    "path": str(path),
                    "suffix": suffix,
                    "supported": self.supported_suffixes(),
                },
            )

        module_name, class_name = spec
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except ImportError as exc:  # pragma: no cover - 依赖缺失属于部署问题
            raise IngestError(
                f"解析 {suffix} 需要额外的依赖，但导入失败：{exc}。"
                "请确认已按 pyproject.toml 安装完整依赖。",
                detail={"suffix": suffix, "module": module_name},
            ) from exc

        instance = cls()
        self._instances[suffix] = instance
        return instance


_DEFAULT_REGISTRY: LoaderRegistry | None = None


def get_registry() -> LoaderRegistry:
    """进程级默认注册表。"""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = LoaderRegistry()
    return _DEFAULT_REGISTRY
