"""图片二进制落盘。

原图**不存数据库**。三个理由：

1. SQLite 里塞 BLOB 会让每次备份、每次 ``VACUUM`` 都拖着几百 MB 图片；
2. 前端要的是 ``<img src="...">``，得有一个能被静态服务直接吐出去的路径；
3. 图片是不可变的（``image_id`` 由内容哈希派生），没有「就地更新」的需求，
   文件系统恰好擅长这个。

路径契约
--------
::

    {images_dir}/{kb_id}/{image_id}.{ext}
    ↕
    {static_url_prefix}/images/{kb_id}/{image_id}.{ext}

左边是磁盘、右边是 URL，两者由 :meth:`BlobStore.url_for` 与
:meth:`BlobStore.resolve` 互相推导。**这是整个项目里唯一一处拼接图片路径的地方**，
别处（API、前端、导出）一律走这两个方法 —— 否则改一次目录结构就要全局搜字符串。

安全
----
``kb_id`` 来自用户请求，会被直接用来拼路径。**``../../`` 这类输入必须在这里被拦住**，
不能指望上游没传恶意值：这是那一类「只要有一个调用点忘了校验就全线失守」的漏洞。
本模块用两道防线：

1. 白名单正则 —— ``kb_id`` / ``image_id`` 只允许 ``[A-Za-z0-9._-]``，
   且不能以 ``.`` 开头。``..`` 直接过不了字符检查；
2. ``resolve()`` 之后再断言 ``is_relative_to(root)`` ——
   万一将来有人放宽了正则（比如为了支持中文知识库名），
   这一层仍然会把逃逸出去的路径挡回来。

写盘用 ``.tmp`` + ``os.replace``：``os.replace`` 在同一文件系统内是**原子**的，
所以读者永远看不到「写了一半的图片」。断电时最多留下一个 ``.tmp`` 垃圾文件，
不会留下一张损坏的、看起来正常但打不开的图。
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.core.errors import StorageError
from app.core.logging import get_logger

__all__ = ["BlobStore"]

logger = get_logger(__name__)

#: 路径段白名单。刻意不允许以 ``.`` 开头：``.`` / ``..`` / ``.hidden`` 一并拒掉。
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")

#: 允许落盘的图片扩展名。不在表里的一律按 ``bin`` 存 —— 但我们不会写 ``bin``，
#: 因为上游的 mime 一定在这张表里，不在就说明抽出来的是我们没打算处理的东西。
_EXT_BY_MIME: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/x-emf": "emf",
    "image/emf": "emf",
    "image/wmf": "wmf",
    "image/x-wmf": "wmf",
}

#: 反查用：按扩展名给 Content-Type。
_MIME_BY_EXT: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "emf": "image/emf",
    "wmf": "image/wmf",
}

_TMP_SUFFIX = ".tmp"


def normalize_segment(value: str, *, kind: str) -> str:
    """校验一个路径段。不合法直接抛，**不做「清洗」**。

    不做清洗是有意的：把 ``a/../b`` 清洗成 ``b`` 会让调用方以为自己的输入被接受了，
    而它实际想要的可能完全是另一回事。宁可响亮地失败。
    """
    if not value or not _SEGMENT_RE.match(value):
        raise StorageError(
            f"非法的 {kind}：{value!r}",
            detail={
                "kind": kind,
                "allowed": "字母数字与 . _ -，必须以字母数字或下划线开头，长度 1–128",
                "hint": "这个值会参与文件路径拼接，含 / \\ .. 等一律拒绝",
            },
        )
    return value


def extension_for_mime(mime_type: str) -> str:
    """mime → 扩展名。不认识的一律用 ``png``。

    为什么是 png 而不是 ``bin``：下游的 OCR / VLM 与浏览器都靠扩展名或
    Content-Type 判断怎么解码，一个 ``.bin`` 会让整条链路都得多写一层特判。
    png 至少是最可能正确的猜测，而且我们上游的抽取器基本只吐 png/jpeg。
    """
    normalized = (mime_type or "").split(";")[0].strip().lower()
    return _EXT_BY_MIME.get(normalized, "png")


def mime_for_extension(ext: str) -> str:
    return _MIME_BY_EXT.get(ext.lstrip(".").lower(), "application/octet-stream")


class BlobStore:
    """图片仓库。无状态（除了缓存下来的根目录），可以放心当单例。"""

    def __init__(self, settings: Settings | None = None, *, root: Path | None = None) -> None:
        self._settings = settings or get_settings()
        self._root = Path(root) if root is not None else self._settings.images_dir
        self._prefix = self._settings.app.static_url_prefix.rstrip("/")

    @property
    def root(self) -> Path:
        return self._root

    def ensure_root(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    # ------------------------------------------------------------------ 路径

    def dir_for(self, kb_id: str) -> Path:
        """``{root}/{kb_id}``。存在性不保证 —— 空知识库没有目录。"""
        kb_id = normalize_segment(kb_id, kind="kb_id")
        path = (self._root / kb_id).resolve()
        self._assert_inside(path)
        return path

    def path_for(self, kb_id: str, image_id: str, ext: str = "png") -> Path:
        """``{root}/{kb_id}/{image_id}.{ext}``。"""
        kb_id = normalize_segment(kb_id, kind="kb_id")
        # image_id 可能已经带了扩展名（``abc123.png``），剥掉再拼，
        # 免得出现 ``abc123.png.png`` —— 那种文件能写、能读，只是在目录里极难看
        stem = image_id.rsplit(".", 1)[0] if "." in image_id else image_id
        image_id = normalize_segment(stem, kind="image_id")
        ext = normalize_segment(ext.lstrip("."), kind="扩展名")
        path = (self._root / kb_id / f"{image_id}.{ext}").resolve()
        self._assert_inside(path)
        return path

    def _assert_inside(self, path: Path) -> None:
        """第二道防线。见模块 docstring 的「安全」。"""
        root = self._root.resolve()
        if path != root and not path.is_relative_to(root):
            raise StorageError(
                "路径逃逸：拼接出的图片路径落在 images_dir 之外",
                detail={"path": str(path), "root": str(root)},
            )

    def url_for(self, kb_id: str, image_id: str, ext: str = "png") -> str:
        """磁盘路径 → 前端可访问的 URL。与 :meth:`path_for` 严格同构。"""
        kb_id = normalize_segment(kb_id, kind="kb_id")
        stem = image_id.rsplit(".", 1)[0] if "." in image_id else image_id
        image_id = normalize_segment(stem, kind="image_id")
        ext = normalize_segment(ext.lstrip("."), kind="扩展名")
        return f"{self._prefix}/images/{kb_id}/{image_id}.{ext}"

    def resolve(self, kb_id: str, image_id: str, ext: str = "png") -> Path | None:
        """URL ↔ 路径的另一种用法：已知 kb/image id，找出实际落盘的哪个文件。

        扩展名可能猜错（上游说的 mime 与实际存的不一致，或历史数据用了别的扩展名），
        所以这里会把候选扩展名都试一遍。返回 ``None`` 表示确实不存在。
        """
        candidates = [ext.lstrip("."), *_MIME_BY_EXT.keys()]
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                path = self.path_for(kb_id, image_id, candidate)
            except StorageError:
                continue
            if path.is_file():
                return path
        return None

    # ------------------------------------------------------------------ 写

    def put(
        self,
        kb_id: str,
        image_id: str,
        data: bytes,
        *,
        mime_type: str = "image/png",
        ext: str | None = None,
    ) -> tuple[Path, str]:
        """写入一张图片，返回 ``(实际路径, URL)``。

        扩展名优先取显式传入的 ``ext``，否则从 ``mime_type`` 推。
        同一 ``image_id`` 重复写入是安全的（``image_id`` 由内容哈希派生，
        内容相同则写入的字节也相同）。
        """
        if not data:
            raise StorageError(
                "图片内容为空，拒绝写入",
                detail={"kb_id": kb_id, "image_id": image_id},
            )
        suffix = ext.lstrip(".") if ext else extension_for_mime(mime_type)
        path = self.path_for(kb_id, image_id, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 同目录下的临时文件 —— os.replace 的原子性只在同一文件系统内成立，
        # 用 tempfile 默认的 /tmp 会跨设备，退化成「复制 + 删除」，就不原子了
        tmp = path.with_name(path.name + _TMP_SUFFIX)
        try:
            with tmp.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise StorageError(
                f"写图片失败：{exc}",
                detail={"path": str(path), "bytes": len(data)},
            ) from exc

        return path, self.url_for(kb_id, image_id, suffix)

    # ------------------------------------------------------------------ 读

    def read(self, kb_id: str, image_id: str, ext: str = "png") -> bytes:
        path = self.resolve(kb_id, image_id, ext)
        if path is None:
            raise StorageError(
                "图片不存在",
                detail={"kb_id": kb_id, "image_id": image_id},
            )
        try:
            return path.read_bytes()
        except OSError as exc:
            raise StorageError(
                f"读图片失败：{exc}", detail={"path": str(path)}
            ) from exc

    def exists(self, kb_id: str, image_id: str, ext: str = "png") -> bool:
        return self.resolve(kb_id, image_id, ext) is not None

    def mime_of(self, path: Path) -> str:
        return mime_for_extension(path.suffix)

    def size_of(self, path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    # ------------------------------------------------------------------ 删

    def delete(self, kb_id: str, image_id: str, ext: str = "png") -> bool:
        path = self.resolve(kb_id, image_id, ext)
        if path is None:
            return False
        path.unlink(missing_ok=True)
        return True

    def delete_kb(self, kb_id: str) -> int:
        """删掉整个知识库的图片目录。返回删掉的文件数。

        **删知识库必须调它** —— 只删数据库行会留下一堆永远没人引用的图片，
        在「重建索引」反复进行之后能积累到几个 GB。
        """
        directory = self.dir_for(kb_id)
        if not directory.is_dir():
            return 0
        count = sum(1 for p in directory.rglob("*") if p.is_file())
        shutil.rmtree(directory, ignore_errors=True)
        logger.info("已删除知识库图片目录", extra={"kb_id": kb_id, "files": count})
        return count

    def cleanup_tmp(self) -> int:
        """清掉上次崩溃留下的 ``.tmp``。启动时调一次，成本很低。"""
        root = self._root
        if not root.is_dir():
            return 0
        removed = 0
        for path in root.rglob(f"*{_TMP_SUFFIX}"):
            try:
                path.unlink()
                removed += 1
            except OSError:  # pragma: no cover - 并发删除时才会走到
                continue
        if removed:
            logger.info("清理了残留的临时图片文件", extra={"count": removed})
        return removed

    # ------------------------------------------------------------------ 运维

    def stats(self) -> dict[str, Any]:
        """给 ``/api/health`` 用。大目录下 ``rglob`` 会慢，所以只数不 stat。"""
        root = self._root
        if not root.is_dir():
            return {"root": str(root), "exists": False, "kb_count": 0, "file_count": 0}
        kb_dirs = [p for p in root.iterdir() if p.is_dir()]
        return {
            "root": str(root),
            "exists": True,
            "kb_count": len(kb_dirs),
            "file_count": sum(1 for p in root.rglob("*") if p.is_file()),
        }
