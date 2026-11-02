@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title mmkb
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ============================================================================
REM  mmkb 一键启动（Windows）
REM
REM  双击本文件，或在终端执行：
REM      .\start.bat
REM      .\start.bat --no-browser
REM      .\start.bat --check
REM
REM  首次运行会自动创建 .venv 并安装依赖；之后直接启动。
REM ============================================================================

set "PY=.venv\Scripts\python.exe"
if exist "%PY%" goto :have_venv

echo [mmkb] 未找到 .venv，正在创建 Python 3.12 虚拟环境...
py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 goto :try_python
py -3.12 -m venv .venv
goto :check_venv

:try_python
python -c "import sys" >nul 2>nul
if errorlevel 1 goto :no_python
python -m venv .venv
goto :check_venv

:check_venv
if not exist "%PY%" goto :venv_failed

:have_venv
"%PY%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if errorlevel 1 goto :wrong_python

"%PY%" -c "import fastapi, uvicorn, streamlit, pydantic_settings, httpx" >nul 2>nul
if not errorlevel 1 goto :run

echo [mmkb] 首次启动：正在安装依赖（可能需要几分钟）...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :install_failed
"%PY%" -m pip install -e ".[dev]"
if errorlevel 1 goto :install_failed

:run
"%PY%" scripts\launch.py %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [mmkb] 启动失败，退出码 %EXIT_CODE%。请查看上方日志。
    pause
)
exit /b %EXIT_CODE%

:no_python
echo [mmkb] 未找到 Python。请安装 Python 3.12，并勾选 "Add python.exe to PATH"。
echo        下载地址：https://www.python.org/downloads/
pause
exit /b 1

:wrong_python
echo [mmkb] .venv 里的 Python 不是 3.12：
"%PY%" --version
echo        请删除 .venv 后重新运行本脚本。
pause
exit /b 1

:venv_failed
echo [mmkb] 创建 .venv 失败。请确认 Python 3.12 可用，并查看上方错误。
pause
exit /b 1

:install_failed
echo [mmkb] 依赖安装失败。请检查网络或代理后重试。
pause
exit /b 1
