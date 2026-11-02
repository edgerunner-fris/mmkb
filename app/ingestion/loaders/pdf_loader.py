"""PDF loader（PyMuPDF / fitz）。

产出的三样东西
--------------
1. **文本块** —— 按 ``page.get_text("dict")`` 的行几何还原段落；
2. **表格块** —— ``page.find_tables()``，结构化行列表交给
   :mod:`app.ingestion.table_serializer`；
3. **内嵌图片** —— ``page.get_images(full=True)`` 拿 xref →
   ``doc.extract_image`` 拿字节 → ``page.get_image_rects`` 拿 bbox。

三个容易踩的坑
--------------

**坑 1：页眉页脚。** 每页顶部的书名、底部的页码会被解析成正文，于是每一页都
多出一段与内容无关的文字，还会被 BM25 当成高频词。这里用「同一行在 60% 以上
的页面出现」来识别并丢弃，识别前把数字统一成 ``#``（否则页码各不相同，
永远不会被认成重复）。

**坑 2：表格里的文字被解析两遍。** ``get_text("dict")`` 会把表格单元格的文字
当普通文本块返回，和 ``find_tables`` 的结果重复。这里按 bbox 相交面积跳过
表格区域内的文本行 —— 不跳的话，表格内容在检索里会出现两次，引用也会重复。

**坑 3：同一个 xref 在多页重复出现。** 水印、页眉 logo 的 xref 是全文档共享的。
用 ``seen_xrefs`` 去重，只保留第一次出现的页与 bbox。

**不支持的东西**（明确写在这里，避免以为漏了）：矢量图（``page.get_drawings()``）
不是位图，无法送进多模态 embedding；扫描版 PDF 若无文本层，只会产出整页图片，
由 image_processor 走 caption 路径补救 —— 这是有意的降级，不是遗漏。
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.errors import IngestError
from app.core.schemas import BlockType
from app.ingestion.loaders.base import (
    BaseLoader,
    LoadResult,
    OrderCounter,
    RawBlock,
    RawImage,
    clean_block_text,
    clean_inline_text,
    file_sha256,
)

__all__ = ["PdfLoader"]

#: PyMuPDF 的 span.flags 第 4 位（值 16）表示粗体。
_FLAG_BOLD = 16

#: 编号型标题。PDF 里最常见的标题形式就是「有编号 + 字号正常」，
#: 光看字号会漏掉一大批。
_NUMBERED_HEADING_PATTERNS: tuple[str, ...] = (
    r"^第\s*[一二三四五六七八九十百千零〇0-9]+\s*[章篇部节讲]",
    r"^[0-9]+(\.[0-9]+){1,3}[\s、.．]",
    r"^[一二三四五六七八九十]+[、.．]",
    r"^[（(][一二三四五六七八九十0-9]+[）)]",
)

#: 列表项标记：遇到它们时保留换行，而不是按「PDF 换行是排版产物」合并。
_BULLET_PREFIXES: tuple[str, ...] = ("•", "·", "●", "○", "▪", "▫", "①", "②", "③", "-", "*")

#: 句子结束符。合并行时用来决定是否保留换行。
_SENTENCE_ENDS = "。！？.!?；;：:"


@dataclass(slots=True)
class _Line:
    """一行文本及其几何信息。"""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    page: int

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)


@dataclass(slots=True)
class _PageData:
    number: int
    height: float
    lines: list[_Line] = field(default_factory=list)
    tables: list[tuple[tuple[float, float, float, float], list[list[str]]]] = field(
        default_factory=list
    )
    images: list[RawImage] = field(default_factory=list)
    has_text: bool = False


class PdfLoader(BaseLoader):
    """单文件 PDF。"""

    suffixes = frozenset({".pdf"})
    kind = "pdf"
    mime_type = "application/pdf"

    #: 页面文本块的最小字符数。低于它的页视为「没有文本层」。
    _MIN_PAGE_CHARS = 20

    def load(self, path: Path) -> LoadResult:
        self._require_file(path)

        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - 依赖缺失属于部署问题
            raise IngestError("解析 PDF 需要 PyMuPDF，请执行 pip install PyMuPDF") from exc

        try:
            document = fitz.open(path)
        except Exception as exc:  # noqa: BLE001 - fitz 的异常类型随版本变化
            raise IngestError(
                f"无法打开 PDF：{exc}", detail={"path": str(path)}
            ) from exc

        try:
            if document.needs_pass:
                # 空密码能解开的「仅限制编辑」PDF 很常见，先试一次再放弃。
                if not document.authenticate(""):
                    raise IngestError(
                        "PDF 已加密，需要密码才能读取",
                        detail={"path": str(path)},
                    )

            pages = self._read_pages(document)
            body_size = self._body_font_size(pages)
            repeated = self._repeated_margin_lines(pages)

            counter = OrderCounter()
            result = LoadResult(page_count=document.page_count)
            heading_levels: list[int] = []

            for page in pages:
                if page.has_text and self._text_length(page) < self._MIN_PAGE_CHARS:
                    result.warnings.append(
                        f"第 {page.number} 页几乎提取不到文本，可能是扫描页或整页图片"
                    )
                self._emit_page(page, body_size, repeated, counter, result, heading_levels)

            title = self._document_title(document, result)
            if title:
                result.title = title

            result.metadata = {
                "loader": self.kind,
                "sha256": file_sha256(path),
                "page_count": document.page_count,
                "font_body_size": body_size,
            }
            if result.is_empty:
                result.warnings.append("PDF 中没有提取到任何文本或图片")
            return result
        finally:
            document.close()

    # ------------------------------------------------------------------ 读取

    def _read_pages(self, document: Any) -> list[_PageData]:
        pages: list[_PageData] = []
        for index in range(document.page_count):
            page = document.load_page(index)
            data = _PageData(number=index + 1, height=float(page.rect.height))
            data.tables = self._extract_tables(page)

            raw_text = page.get_text("dict")
            for block in raw_text.get("blocks", []):
                if block.get("type") != 0:
                    continue  # type=1 是位图块，图片走 get_images 统一处理
                for line in block.get("lines", []):
                    text = ""
                    sizes: list[float] = []
                    bold = False
                    for span in line.get("spans", []):
                        piece = span.get("text", "")
                        if not piece:
                            continue
                        text += piece
                        sizes.append(float(span.get("size", 0.0)))
                        if int(span.get("flags", 0)) & _FLAG_BOLD:
                            bold = True
                    text = clean_inline_text(text)
                    if not text:
                        continue
                    bbox = line.get("bbox") or (0.0, 0.0, 0.0, 0.0)
                    data.lines.append(
                        _Line(
                            text=text,
                            x0=float(bbox[0]),
                            y0=float(bbox[1]),
                            x1=float(bbox[2]),
                            y1=float(bbox[3]),
                            size=round(statistics.mean(sizes), 1) if sizes else 0.0,
                            bold=bold,
                            page=index + 1,
                        )
                    )
                    data.has_text = True

            data.images = self._extract_images(document, page, index + 1)
            pages.append(data)
        return pages

    @staticmethod
    def _text_length(page: _PageData) -> int:
        return sum(len(line.text) for line in page.lines)

    def _extract_tables(
        self, page: Any
    ) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
        """返回 ``[(bbox, rows), ...]``。找不到表格时返回空列表。

        ``find_tables`` 在 1.23 才稳定下来；更老的版本没有这个属性，
        此时静默退化为「没有表格」，而不是让整份 PDF 解析失败。
        """
        finder_factory = getattr(page, "find_tables", None)
        if finder_factory is None:
            return []
        try:
            finder = finder_factory()
            tables = list(finder.tables)
        except Exception:  # noqa: BLE001 - 表格识别失败不该毁掉整页文本
            return []

        out: list[tuple[tuple[float, float, float, float], list[list[str]]]] = []
        for table in tables:
            try:
                raw_rows = table.extract()
            except Exception:  # noqa: BLE001
                continue
            rows = [
                [clean_inline_text(cell or "") for cell in row]
                for row in raw_rows
                if row is not None
            ]
            rows = [row for row in rows if any(cell for cell in row)]
            # 一行或一列的表基本是排版用的分隔线，不是真表格
            if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
                continue
            bbox = tuple(float(v) for v in table.bbox)
            out.append((bbox, rows))
        return out

    def _extract_images(
        self, document: Any, page: Any, page_number: int
    ) -> list[RawImage]:
        images: list[RawImage] = []
        try:
            infos = page.get_images(full=True)
        except Exception:  # noqa: BLE001
            return images

        for info in infos:
            xref = int(info[0])
            try:
                base = document.extract_image(xref)
            except Exception:  # noqa: BLE001 - 单个损坏的图不该中断整页
                continue
            payload = base.get("image") or b""
            if not payload:
                continue
            ext = str(base.get("ext") or "png").lower()
            rects = []
            try:
                rects = page.get_image_rects(xref)
            except Exception:  # noqa: BLE001
                rects = []
            bbox = None
            if rects:
                rect = rects[0]
                bbox = [
                    float(rect.x0),
                    float(rect.y0),
                    float(rect.x1),
                    float(rect.y1),
                ]
            images.append(
                RawImage(
                    data=payload,
                    page=page_number,
                    bbox=bbox,
                    ext=ext,
                    source_ref={"xref": xref},
                )
            )
        return images

    # ------------------------------------------------------------------ 统计

    @staticmethod
    def _body_font_size(pages: list[_PageData]) -> float:
        """按**字符数**加权选出正文字号。

        不用平均字号：标题少、正文多，平均会被标题拉高；按字符数加权才等价于
        「这份文档大部分字是多大的」。
        """
        weight: dict[float, int] = {}
        for page in pages:
            for line in page.lines:
                if line.size <= 0:
                    continue
                weight[line.size] = weight.get(line.size, 0) + len(line.text)
        if not weight:
            return 10.0
        return max(weight.items(), key=lambda item: item[1])[0]

    def _repeated_margin_lines(self, pages: list[_PageData]) -> set[str]:
        """识别页眉页脚：出现在 60% 以上页面的短行。"""
        if len(pages) < 3:
            return set()

        counts: dict[str, int] = {}
        for page in pages:
            seen: set[str] = set()
            for line in page.lines:
                if len(line.text) > 120:
                    continue
                in_margin = line.y1 < page.height * 0.10 or line.y0 > page.height * 0.90
                if not in_margin:
                    continue
                key = _normalize_for_repeat(line.text)
                if key:
                    seen.add(key)
            for key in seen:
                counts[key] = counts.get(key, 0) + 1

        threshold = max(2, int(len(pages) * 0.6))
        return {key for key, count in counts.items() if count >= threshold}

    @staticmethod
    def _document_title(document: Any, result: LoadResult) -> str:
        meta = getattr(document, "metadata", None) or {}
        title = clean_inline_text(str(meta.get("title") or ""))
        # 不少 PDF 把标题写成文件名（含扩展名）或者干脆是 "untitled"
        if title and not title.lower().endswith(".pdf") and title.lower() != "untitled":
            return title
        for block in result.blocks:
            if block.is_heading:
                return block.text
        return ""

    # ------------------------------------------------------------------ 输出

    def _emit_page(
        self,
        page: _PageData,
        body_size: float,
        repeated: set[str],
        counter: OrderCounter,
        result: LoadResult,
        heading_levels: list[int],
    ) -> None:
        """把一页的行/表/图按纵坐标合并成有序事件流。"""
        events: list[tuple[float, int, str, Any]] = []
        for line in page.lines:
            if _normalize_for_repeat(line.text) in repeated:
                continue
            if self._inside_table(line, page.tables):
                continue
            events.append((line.y0, 1, "line", line))
        for bbox, rows in page.tables:
            events.append((bbox[1], 0, "table", (bbox, rows)))
        for image in page.images:
            y = image.bbox[1] if image.bbox else float("inf")
            events.append((y, 2, "image", image))

        # 同一纵坐标时先表格、再文本、后图片：表格标题常与表头同高。
        events.sort(key=lambda item: (item[0], item[1]))

        paragraph: list[_Line] = []

        def flush() -> None:
            if not paragraph:
                return
            text = _join_lines([line.text for line in paragraph])
            if text:
                result.blocks.append(
                    RawBlock(
                        text=text,
                        block_type=BlockType.TEXT,
                        page=paragraph[0].page,
                        order=counter.next(),
                        source_ref={"loader": self.kind, "page": paragraph[0].page},
                    )
                )
            paragraph.clear()

        for _, _, kind, payload in events:
            if kind == "line":
                line: _Line = payload
                level = self._heading_level(line, body_size)
                if level:
                    flush()
                    result.blocks.append(
                        RawBlock(
                            text=line.text,
                            block_type=BlockType.TEXT,
                            page=line.page,
                            order=counter.next(),
                            heading_level=level,
                            source_ref={"loader": self.kind, "page": line.page},
                        )
                    )
                    heading_levels.append(level)
                    continue
                if paragraph and _starts_new_paragraph(paragraph[-1], line):
                    flush()
                paragraph.append(line)
            elif kind == "table":
                flush()
                bbox, rows = payload
                result.blocks.append(
                    RawBlock(
                        text="",
                        block_type=BlockType.TABLE,
                        page=page.number,
                        order=counter.next(),
                        table=rows,
                        source_ref={
                            "loader": self.kind,
                            "page": page.number,
                            "bbox": list(bbox),
                        },
                    )
                )
            else:
                flush()
                image: RawImage = payload
                image.order = counter.next()
                result.images.append(image)
        flush()

    @staticmethod
    def _inside_table(
        line: _Line,
        tables: list[tuple[tuple[float, float, float, float], list[list[str]]]],
    ) -> bool:
        for (x0, y0, x1, y1), _rows in tables:
            overlap = _intersection_ratio(line.x0, line.y0, line.x1, line.y1, x0, y0, x1, y1)
            if overlap > 0.5:
                return True
        return False

    def _heading_level(self, line: _Line, body_size: float) -> int:
        """判断一行是不是标题，是则返回层级（1 最高），否则 0。"""
        text = line.text
        if len(text) > 120:
            return 0
        if text.endswith(("，", ",", "、", "。", "．", ".")) and len(text) > 30:
            # 以逗号结尾的长行几乎一定是正文
            return 0

        ratio = line.size / body_size if body_size > 0 else 1.0
        if ratio >= 1.8:
            return 1
        if ratio >= 1.45:
            return 2
        if ratio >= 1.18:
            return 3

        # 字号不突出时，靠「粗体」或「编号」补判。
        is_numbered = _looks_numbered(text)
        if ratio >= 1.02 and line.bold and len(text) <= 60:
            return 4
        if is_numbered and len(text) <= 60 and ratio >= 0.98:
            return 4
        return 0


def _looks_numbered(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in _NUMBERED_HEADING_PATTERNS)


def _normalize_for_repeat(text: str) -> str:
    """把数字统一成 ``#``，让「第 3 页 / 第 4 页」这类页脚能对上。"""
    if not text:
        return ""
    out = []
    for ch in text:
        out.append("#" if ch.isdigit() else ch)
    return "".join(out).strip()


def _intersection_ratio(
    ax0: float, ay0: float, ax1: float, ay1: float,
    bx0: float, by0: float, bx1: float, by1: float,
) -> float:
    """A 被 B 覆盖的面积占 A 总面积的比例。"""
    width = max(0.0, ax1 - ax0)
    height = max(0.0, ay1 - ay0)
    area = width * height
    if area <= 0:
        return 0.0
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    return (inter_w * inter_h) / area


def _starts_new_paragraph(previous: _Line, current: _Line) -> bool:
    """靠几何间距判断是否换段。

    PDF 里没有「段落」这个对象，只有行。段间距明显大于行内间距是最可靠的信号；
    比它更复杂的方法（缩进、字体变化）在双栏排版里会大面积误判。
    """
    gap = current.y0 - previous.y1
    if gap <= 0:
        return False
    return gap > max(previous.height, current.height) * 0.75


def _join_lines(lines: list[str]) -> str:
    """把同一段的多行合并成一段。

    规则有实际来源：
      * 英文行尾连字符 ``exam-`` + ``ple`` 要拼成 ``example``，否则会多出一个错词；
      * 中文换行不加空格 —— 加了空格会让「检 索」这种东西进 BM25 语料；
      * 以列表符号开头的行保留换行，否则列表会被压成一句话。
    """
    if not lines:
        return ""
    out = lines[0]
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped.startswith(_BULLET_PREFIXES):
            out += "\n" + stripped
            continue
        if out.endswith("-") and stripped[:1].islower():
            out = out[:-1] + stripped
            continue
        if _is_cjk(out[-1:]) or _is_cjk(stripped[:1]):
            out += stripped
            continue
        if out.endswith(_SENTENCE_ENDS):
            out += "\n" + stripped
            continue
        out += " " + stripped
    return clean_block_text(out)


def _is_cjk(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK 统一表意文字
        or 0x3400 <= code <= 0x4DBF  # 扩展 A
        or 0x3000 <= code <= 0x303F  # CJK 标点
        or 0xFF00 <= code <= 0xFFEF  # 全角字符
    )
