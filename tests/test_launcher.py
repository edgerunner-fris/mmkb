"""一键启动器的离线测试。

只覆盖纯函数与命令拼装，**不真的拉起 uvicorn / streamlit**：
真正的启动验证由 ``scripts/launch.py`` 的端到端冒烟完成。
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "mmkb_launch", ROOT / "scripts" / "launch.py"
)
assert _SPEC is not None and _SPEC.loader is not None
launcher = importlib.util.module_from_spec(_SPEC)
# dataclasses 会按 cls.__module__ 回查 sys.modules；不注册会直接报
# AttributeError: 'NoneType' object has no attribute '__dict__'
sys.modules[_SPEC.name] = launcher
_SPEC.loader.exec_module(launcher)


# --------------------------------------------------------------------------------------
# .env 引导
# --------------------------------------------------------------------------------------


def test_ensure_env_file_copies_example_once(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("A=1\n", encoding="utf-8")

    assert launcher.ensure_env_file(tmp_path) is True
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "A=1\n"
    assert launcher.ensure_env_file(tmp_path) is False


def test_ensure_env_file_never_overwrites_existing_keys(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=keep\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("SECRET=example\n", encoding="utf-8")

    assert launcher.ensure_env_file(tmp_path) is False
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "SECRET=keep\n"


# --------------------------------------------------------------------------------------
# 端口
# --------------------------------------------------------------------------------------


def _occupied_port() -> tuple[socket.socket, int]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    return sock, sock.getsockname()[1]


def test_pick_port_shifts_when_preferred_is_taken() -> None:
    sock, port = _occupied_port()
    try:
        picked = launcher.pick_port(port, "127.0.0.1", allow_shift=True)
        assert picked != port
        assert launcher.port_is_free(picked, "127.0.0.1")
    finally:
        sock.close()


def test_pick_port_refuses_to_shift_a_pinned_port() -> None:
    sock, port = _occupied_port()
    try:
        with pytest.raises(launcher.PortUnavailableError):
            launcher.pick_port(port, "127.0.0.1", allow_shift=False)
    finally:
        sock.close()


# --------------------------------------------------------------------------------------
# 子进程命令与环境
# --------------------------------------------------------------------------------------


def test_api_command_targets_uvicorn_app() -> None:
    command = launcher.build_api_command(host="127.0.0.1", port=9123, python="python")

    assert command[:4] == ["python", "-m", "uvicorn", "app.main:app"]
    assert command[command.index("--port") + 1] == "9123"
    assert "--reload" not in command


def test_api_command_can_enable_reload() -> None:
    command = launcher.build_api_command(
        host="127.0.0.1", port=9123, python="python", reload=True
    )
    assert "--reload" in command


def test_frontend_command_is_headless_and_uses_the_given_port() -> None:
    command = launcher.build_frontend_command(port=8601, python="python")

    assert command[:3] == ["python", "-m", "streamlit"]
    assert command[command.index("--server.port") + 1] == "8601"
    assert command[command.index("--server.headless") + 1] == "true"
    assert command[command.index("--server.address") + 1] == "127.0.0.1"
    assert str(ROOT / "frontend" / "streamlit_app.py") in command


def test_frontend_env_points_both_urls_at_the_api() -> None:
    env = launcher.build_frontend_env("http://127.0.0.1:9123")

    assert env["MMKB_API_BASE"] == "http://127.0.0.1:9123"
    assert env["MMKB_PUBLIC_BASE"] == "http://127.0.0.1:9123"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_port_argument_validation_rejects_out_of_range() -> None:
    assert launcher._validate_port_arg(None, "--api-port") is True
    assert launcher._validate_port_arg(8000, "--api-port") is True
    assert launcher._validate_port_arg(0, "--api-port") is False
    assert launcher._validate_port_arg(65536, "--api-port") is False
