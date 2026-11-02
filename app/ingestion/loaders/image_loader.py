"""独立图片文件（png / jpg / gif / bmp / webp / tif）。

这类文件没有正文，产出就是一个 :class:`RawImage`。后面的流程会给它配一段
由视觉模型生成的描述（caption），并把描述放进文本向量库 —— 图片本身则进
多模态向量库。两个空间各存一份，正是为了 ``以文搜图`` 与 ``以图搜图``
都能成立。

尺寸信息尽量用 Pillow 读（失败不影响导入）：它决定了图片会不会在
:mod:`app.ingestion.image_processor` 里被「过小」规则过滤掉 ——
如果这里读不到尺寸，那张图就少了这道判断，所以我们把宽高写进
``metadata`` 而不是让下游再解一次。
"""

from __future__ import annotations

import io
from pathlib import Path

from app.core.errors import IngestError
from app.ingestion.loaders.base import BaseLoader, LoadResult, RawImage

__all__ = ["ImageLoader"]

#: 单张图片的大小上限。超过的图不像是知识库素材，更像是扫描件或设计稿，
#: 送进多模态 embedding 会被压缩到失去细节，而且按 token 计费并不便宜。
_MAX_IMAGE_BYTES = 20 << 20


class ImageLoader(BaseLoader):
    """一张独立的位图。"""

    suffixes = frozenset(
        {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
    )
    kind = "image"
    mime_type = "image/png"

    def load(self, path: Path) -> LoadResult:
        self._require_file(path)
        self._check_size(path, _MAX_IMAGE_BYTES)

        suffix = path.suffix.lower().lstrip(".")
        if suffix == "jpeg":
            suffix = "jpg"

        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise IngestError(
                f"读取图片失败：{exc}", detail={"path": str(path)}
            ) from exc

        result = LoadResult(metadata={"loader": self.kind})
        width, height, animated = _probe_image(payload)

        result.images.append(
            RawImage(
                data=payload,
                order=0,
                ext=suffix,
                source_ref={"loader": self.kind, "file": path.name},
            )
        )
        result.title = path.stem
        result.metadata.update(
            {
                "width": width,
                "height": height,
                "bytes": len(payload),
                "animated": animated,
            }
        )
        if animated:
            # GIF/APNG 只取首帧。多帧动画的每一帧都投进向量库既贵又没意义：
            # 同一个对象的十几种姿态会把检索结果挤满。
            result.warnings.append("动图只按首帧处理")
        if not width or not height:
            result.warnings.append("无法读取图片尺寸，跳过尺寸过滤")
        return result


def _probe_image(payload: bytes) -> tuple[int, int, bool]:
    """返回 ``(宽, 高, 是否动图)``。读不到时宽高为 0。"""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow 是必装依赖
        return 0, 0, False

    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            animated = bool(getattr(image, "is_animated", False))
            return int(width), int(height), animated
    except Exception:  # noqa: BLE001 - 损坏的图不该让整个导入失败
        return 0, 0, False
