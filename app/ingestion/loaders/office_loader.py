"""Office 三件套：docx / pptx / xlsx（含 xlsm）。

统一约定：**一份文件产出一份 LoadResult**，标题层级用 ``heading_level`` 表达，
表格用 ``table`` 字段表达，图片用 ``RawImage`` 表达。三种格式的差异只体现在
「怎么遍历」这一件事上。
"""

from __future__ import annotations

import io
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
)

__all__ = ["DocxLoader", "PptxLoader", "XlsxLoader"]

#: 常见的等宽字体。用来识别代码段 —— 比「看样式名」可靠，
#: 因为很多文档直接把代码贴进正文，样式仍是 Normal。
_MONO_FONTS = frozenset(
    {
        "consolas",
        "courier new",
        "courier",
        "monaco",
        "menlo",
        "source code pro",
        "fira code",
        "jetbrains mono",
        "dejavu sans mono",
        "lucida console",
        "cascadia code",
        "cascadia mono",
        "sf mono",
        "ubuntu mono",
        "roboto mono",
        "simsun",  # 宋体在中文代码块里常被当等宽用，宁可多判一点
    }
)


def _heading_level_from_style(style_name: str) -> int:
    """``Heading 1`` / ``标题 1`` / ``Heading1`` → 1。认不出返回 0。"""
    if not style_name:
        return 0
    lowered = style_name.strip().lower().replace(" ", "").replace("-", "")
    for prefix in ("heading", "标题", "h"):
        if lowered.startswith(prefix.lower()):
            tail = lowered[len(prefix) :]
            if tail.isdigit() and 1 <= int(tail) <= 6:
                return int(tail)
    if lowered in ("title", "标题"):
        return 1
    return 0


def _is_code_style(style_name: str) -> bool:
    lowered = (style_name or "").lower()
    return any(token in lowered for token in ("code", "代码", "listing", "source"))


def _guess_image_ext(content_type: str, fallback: str = "png") -> str:
    content_type = (content_type or "").lower()
    for candidate in ("png", "jpeg", "jpg", "gif", "bmp", "webp", "tiff"):
        if candidate in content_type:
            return "jpg" if candidate == "jpeg" else candidate
    return fallback


# --------------------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------------------


class DocxLoader(BaseLoader):
    """Word 文档。

    按 ``document.element.body`` 的**子元素顺序**遍历，而不是先取
    ``document.paragraphs`` 再取 ``document.tables`` —— 后者会把所有表格
    挪到所有段落之后，图片与标题的相对位置就全乱了。
    """

    suffixes = frozenset({".docx"})
    kind = "docx"
    mime_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    def load(self, path: Path) -> LoadResult:
        self._require_file(path)
        try:
            from docx import Document as open_docx
            from docx.oxml.ns import qn
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:  # pragma: no cover
            raise IngestError("解析 DOCX 需要 python-docx，请安装项目依赖") from exc

        try:
            document = open_docx(str(path))
        except Exception as exc:  # noqa: BLE001
            raise IngestError(
                f"无法打开 DOCX：{exc}", detail={"path": str(path)}
            ) from exc

        result = LoadResult()
        counter = OrderCounter()
        image_index = 0

        for child in document.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]

            if tag == "p":
                paragraph = Paragraph(child, document)
                text = clean_block_text(paragraph.text)
                # 图片要先收，再判断段落是否为空：Word 里插图最常见的形式
                # 就是「图片独占一个段落」，``paragraph.text`` 此时为空字符串。
                # 先 continue 的话，整份文档的图片会一张都抽不到，而且不会报错。
                image_index = self._collect_paragraph_images(
                    child, document, qn, result, counter, image_index
                )
                if not text:
                    continue
                style_name = paragraph.style.name if paragraph.style is not None else ""
                level = _heading_level_from_style(style_name)
                block_type = (
                    BlockType.CODE
                    if not level and self._paragraph_is_code(paragraph)
                    else BlockType.TEXT
                )
                if not level and _is_code_style(style_name):
                    block_type = BlockType.CODE
                result.blocks.append(
                    RawBlock(
                        text=text,
                        block_type=block_type,
                        order=counter.next(),
                        heading_level=level,
                        source_ref={"loader": self.kind},
                    )
                )

            elif tag == "tbl":
                table = Table(child, document)
                rows = self._table_rows(table)
                if rows:
                    result.blocks.append(
                        RawBlock(
                            text="",
                            block_type=BlockType.TABLE,
                            order=counter.next(),
                            table=rows,
                            source_ref={"loader": self.kind},
                        )
                    )

        title = clean_inline_text(document.core_properties.title or "")
        result.title = title or _first_heading(result)
        result.metadata = {"loader": self.kind}
        if result.is_empty:
            result.warnings.append("DOCX 中没有提取到文本、表格或图片")
        return result

    @staticmethod
    def _paragraph_is_code(paragraph: Any) -> bool:
        """所有非空 run 都用等宽字体 → 当作代码。

        要求「全部」而不是「任一」：正文里偶尔会内联一段等宽文本（变量名、路径），
        那不是代码块。
        """
        names: list[str] = []
        for run in paragraph.runs:
            if not (run.text or "").strip():
                continue
            font = run.font.name
            if not font:
                return False
            names.append(font.lower())
        return bool(names) and all(name in _MONO_FONTS for name in names)

    @staticmethod
    def _table_rows(table: Any) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in table.rows:
            cells = [clean_inline_text(cell.text) for cell in row.cells]
            if any(cells):
                rows.append(cells)
        return rows

    @staticmethod
    def _collect_paragraph_images(
        element: Any,
        document: Any,
        qn: Any,
        result: LoadResult,
        counter: OrderCounter,
        image_index: int,
    ) -> int:
        for blip in element.findall(".//" + qn("a:blip")):
            rid = blip.get(qn("r:embed"))
            if not rid:
                continue
            part = document.part.related_parts.get(rid)
            payload = getattr(part, "blob", b"") if part is not None else b""
            if not payload:
                continue
            ext = _guess_image_ext(getattr(part, "content_type", ""))
            result.images.append(
                RawImage(
                    data=payload,
                    order=counter.next(),
                    ext=ext,
                    source_ref={"loader": "docx", "index": image_index},
                )
            )
            image_index += 1
        return image_index


# --------------------------------------------------------------------------------------
# PPTX
# --------------------------------------------------------------------------------------


class PptxLoader(BaseLoader):
    """PowerPoint。

    每一页幻灯片是一个 ``page``；幻灯片标题作为二级标题（演示文稿标题是一级）。
    这样「第三章 架构 > 3.2 检索模块」这种跨页主题在 ``heading_path`` 里仍然连续。
    """

    suffixes = frozenset({".pptx"})
    kind = "pptx"
    mime_type = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )

    def load(self, path: Path) -> LoadResult:
        self._require_file(path)
        try:
            from pptx import Presentation
            from pptx.enum.shapes import MSO_SHAPE_TYPE
        except ImportError as exc:  # pragma: no cover
            raise IngestError("解析 PPTX 需要 python-pptx，请安装项目依赖") from exc

        try:
            presentation = Presentation(str(path))
        except Exception as exc:  # noqa: BLE001
            raise IngestError(
                f"无法打开 PPTX：{exc}", detail={"path": str(path)}
            ) from exc

        result = LoadResult(page_count=len(presentation.slides))
        counter = OrderCounter()
        image_index = 0

        deck_title = clean_inline_text(presentation.core_properties.title or "")
        if deck_title:
            result.blocks.append(
                RawBlock(
                    text=deck_title,
                    order=counter.next(),
                    heading_level=1,
                    source_ref={"loader": self.kind},
                )
            )

        for number, slide in enumerate(presentation.slides, start=1):
            title_shape = slide.shapes.title
            title = clean_inline_text(
                title_shape.text_frame.text if title_shape is not None else ""
            )
            if title:
                result.blocks.append(
                    RawBlock(
                        text=title,
                        order=counter.next(),
                        heading_level=2,
                        page=number,
                        source_ref={"loader": self.kind, "slide": number},
                    )
                )

            for shape in slide.shapes:
                image_index = self._emit_shape(
                    shape, result, counter, image_index, number, MSO_SHAPE_TYPE
                )

        result.title = deck_title or _first_heading(result)
        result.metadata = {"loader": self.kind, "slides": len(presentation.slides)}
        if result.is_empty:
            result.warnings.append("PPTX 中没有提取到文本、表格或图片")
        return result

    def _emit_shape(
        self,
        shape: Any,
        result: LoadResult,
        counter: OrderCounter,
        image_index: int,
        page: int,
        mso: Any,
    ) -> int:
        # 组合形状要递归 —— 否则组合里的图和表会被整体漏掉
        if shape.shape_type == mso.GROUP:
            for child in shape.shapes:
                image_index = self._emit_shape(
                    child, result, counter, image_index, page, mso
                )
            return image_index

        if getattr(shape, "has_table", False):
            rows = [
                [clean_inline_text(cell.text) for cell in row.cells]
                for row in shape.table.rows
            ]
            rows = [row for row in rows if any(row)]
            if rows:
                result.blocks.append(
                    RawBlock(
                        text="",
                        block_type=BlockType.TABLE,
                        order=counter.next(),
                        page=page,
                        table=rows,
                        source_ref={"loader": self.kind, "slide": page},
                    )
                )
            return image_index

        if getattr(shape, "has_text_frame", False):
            text = clean_block_text(shape.text_frame.text)
            if text:
                result.blocks.append(
                    RawBlock(
                        text=text,
                        order=counter.next(),
                        page=page,
                        source_ref={"loader": self.kind, "slide": page},
                    )
                )

        if shape.shape_type == mso.PICTURE:
            try:
                payload = shape.image.blob
                ext = shape.image.ext
            except Exception:  # noqa: BLE001 - 损坏的图片对象不该中断整页
                return image_index
            if payload:
                result.images.append(
                    RawImage(
                        data=payload,
                        order=counter.next(),
                        page=page,
                        ext=ext,
                        source_ref={"loader": self.kind, "slide": page},
                    )
                )
                image_index += 1
        return image_index


# --------------------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------------------


class XlsxLoader(BaseLoader):
    """Excel 工作簿。

    每个工作表是一张「表」块，表名作为二级标题。**不做逐单元格切分** ——
    电子表格的行是天然的结构化记录，用 Markdown 表表达后既能被检索命中，
    也能让模型看到列名。

    行数上限默认 5000 行：再多的表基本是数据导出，不是知识库内容，
    全量塞进去会让一次导入的花费失控，也会把 BM25 的语料分布带偏。
    超限时截断并在 ``warnings`` 里留痕。
    """

    suffixes = frozenset({".xlsx", ".xlsm"})
    kind = "xlsx"
    mime_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    _MAX_ROWS = 5000
    _MAX_COLS = 80

    def load(self, path: Path) -> LoadResult:
        self._require_file(path)
        try:
            from openpyxl import load_workbook
        except ImportError as exc:  # pragma: no cover
            raise IngestError("解析 XLSX 需要 openpyxl，请安装项目依赖") from exc

        try:
            # read_only=False 才能拿到 ws._images；代价是整本载入内存，
            # 所以这里配合 _MAX_ROWS 做限制。
            workbook = load_workbook(str(path), data_only=True, read_only=False)
        except Exception as exc:  # noqa: BLE001
            raise IngestError(
                f"无法打开 XLSX：{exc}", detail={"path": str(path)}
            ) from exc

        result = LoadResult(page_count=len(workbook.worksheets))
        counter = OrderCounter()
        image_index = 0

        for sheet in workbook.worksheets:
            rows, truncated = self._sheet_rows(sheet)
            if truncated:
                result.warnings.append(
                    f"工作表 {sheet.title!r} 超过 {self._MAX_ROWS} 行，已截断"
                )
            if not rows:
                continue
            result.blocks.append(
                RawBlock(
                    text=clean_inline_text(sheet.title) or "Sheet",
                    order=counter.next(),
                    heading_level=2,
                    source_ref={"loader": self.kind, "sheet": sheet.title},
                )
            )
            result.blocks.append(
                RawBlock(
                    text="",
                    block_type=BlockType.TABLE,
                    order=counter.next(),
                    table=rows,
                    source_ref={"loader": self.kind, "sheet": sheet.title},
                )
            )
            image_index = self._collect_sheet_images(
                sheet, result, counter, image_index
            )

        result.title = clean_inline_text(workbook.properties.title or "")
        result.metadata = {"loader": self.kind, "sheets": len(workbook.worksheets)}
        try:
            workbook.close()
        except Exception:  # noqa: BLE001
            pass
        if result.is_empty:
            result.warnings.append("XLSX 中没有提取到任何非空单元格或图片")
        return result

    def _sheet_rows(self, sheet: Any) -> tuple[list[list[str]], bool]:
        rows: list[list[str]] = []
        truncated = False
        for index, row in enumerate(sheet.iter_rows(values_only=True)):
            if index >= self._MAX_ROWS:
                truncated = True
                break
            cells = [_cell_text(value) for value in row[: self._MAX_COLS]]
            while cells and not cells[-1]:
                cells.pop()
            if any(cells):
                rows.append(cells)
        if not rows:
            return [], truncated
        width = max(len(row) for row in rows)
        return [row + [""] * (width - len(row)) for row in rows], truncated

    @staticmethod
    def _collect_sheet_images(
        sheet: Any, result: LoadResult, counter: OrderCounter, image_index: int
    ) -> int:
        """取出嵌入式图片。

        ``ws._images`` 在 openpyxl 里是**私有属性**，但公开 API 没有等价物。
        这里把每条路径都用 try 包住：拿不到就退化为「这张图没被提取」，
        而不是让整份工作簿导入失败。取字节的调用顺序覆盖了 openpyxl
        3.1 前后的两种实现（``_data()`` 与 ``ref`` 流）。
        """
        images = getattr(sheet, "_images", None) or []
        for image in images:
            payload = _openpyxl_image_bytes(image)
            if not payload:
                continue
            result.images.append(
                RawImage(
                    data=payload,
                    order=counter.next(),
                    ext="png",
                    source_ref={"loader": "xlsx", "sheet": sheet.title},
                )
            )
            image_index += 1
        return image_index


def _openpyxl_image_bytes(image: Any) -> bytes:
    data_method = getattr(image, "_data", None)
    if callable(data_method):
        try:
            payload = data_method()
            if isinstance(payload, (bytes, bytearray)):
                return bytes(payload)
        except Exception:  # noqa: BLE001
            pass

    ref = getattr(image, "ref", None)
    if ref is not None:
        try:
            if isinstance(ref, (bytes, bytearray)):
                return bytes(ref)
            position = None
            if hasattr(ref, "tell") and hasattr(ref, "seek"):
                position = ref.tell()
                ref.seek(0)
            payload = ref.read()
            if position is not None:
                ref.seek(position)
            if payload:
                return bytes(payload)
        except Exception:  # noqa: BLE001
            pass

    pil_image = getattr(image, "_image", None)
    if pil_image is not None:
        try:
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception:  # noqa: BLE001
            pass
    return b""


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        # Excel 的浮点几乎都是 12.0 这种；去掉无意义的 ".0"
        return str(int(value)) if value.is_integer() else f"{value:g}"
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001
            return str(value)
    return clean_inline_text(str(value))


def _first_heading(result: LoadResult) -> str:
    for block in result.blocks:
        if block.is_heading:
            return block.text
    return ""
