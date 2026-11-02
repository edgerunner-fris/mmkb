"""前端 HTTP 封装的回归测试（不启动 Streamlit）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "mmkb_frontend", ROOT / "frontend" / "streamlit_app.py"
)
assert _SPEC is not None and _SPEC.loader is not None
frontend = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = frontend
_SPEC.loader.exec_module(frontend)


class _Response:
    status_code = 200
    content = b'{"ok": true}'

    def json(self) -> dict[str, Any]:
        return {"ok": True}


def test_request_accepts_a_per_call_timeout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """导入大文件时要能覆盖默认超时，不能再触发 timeout 重复传参的 TypeError。"""
    captured: dict[str, Any] = {}

    def fake_request(method: str, url: str, **kwargs: Any) -> _Response:
        captured.update(kwargs)
        captured["_method"] = method
        captured["_url"] = url
        return _Response()

    monkeypatch.setattr(frontend.httpx, "request", fake_request)
    custom = httpx.Timeout(connect=5.0, read=600.0, write=600.0, pool=5.0)

    result = frontend._request(
        "POST", "/api/kbs/kb_1/ingest", json={"a": 1}, timeout=custom
    )

    assert result == {"ok": True}
    assert captured["timeout"] is custom
    assert captured["json"] == {"a": 1}
    assert captured["_method"] == "POST"
    assert captured["_url"].endswith("/api/kbs/kb_1/ingest")


def test_request_falls_back_to_the_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_request(method: str, url: str, **kwargs: Any) -> _Response:
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr(frontend.httpx, "request", fake_request)

    frontend._request("GET", "/api/health")

    assert captured["timeout"] is frontend.REQUEST_TIMEOUT
