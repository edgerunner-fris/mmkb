#!/usr/bin/env bash
# mmkb 一键启动（macOS / Linux）
#
#   ./start.sh
#   ./start.sh --no-browser
#   ./start.sh --check
#
# 首次运行会自动创建 .venv 并安装依赖；之后直接启动。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

PY=".venv/bin/python"
# Git Bash / MSYS 里目录也是 .venv，但 Windows 版虚拟环境的解释器在 Scripts/ 下。
# 这种情况必须交给 start.bat，否则会往同一个 .venv 里重复创建一套 POSIX 布局。
if [ ! -x "$PY" ] && [ -f ".venv/Scripts/python.exe" ]; then
    echo "[mmkb] 检测到 Windows 版 .venv，请改用 start.bat 启动。" >&2
    exit 1
fi
if [ ! -x "$PY" ]; then
    echo "[mmkb] 未找到 .venv，正在创建 Python 3.12 虚拟环境..."
    if command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv .venv
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m venv .venv
    else
        echo "[mmkb] 未找到 Python。请安装 Python 3.12。" >&2
        exit 1
    fi
fi

if ! "$PY" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)"; then
    echo "[mmkb] .venv 里的 Python 不是 3.12，请删除 .venv 后重试。" >&2
    exit 1
fi

if ! "$PY" -c 'import fastapi, uvicorn, streamlit, pydantic_settings, httpx' >/dev/null 2>&1; then
    echo "[mmkb] 首次启动：正在安装依赖（可能需要几分钟）..."
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -e ".[dev]"
fi

exec "$PY" scripts/launch.py "$@"
