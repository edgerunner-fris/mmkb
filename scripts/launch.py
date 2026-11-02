#!/usr/bin/env python
"""一条命令启动 API 与 Streamlit 前端。

这是「实际使用」的入口，不是又一个调试脚本：

* 同时拉起 ``uvicorn`` 与 ``streamlit``，日志合并成 ``[api]`` / ``[web]`` 两路；
* 端口被占用时自动顺延，不会因为上一次没退干净就启动失败；
* 等 API 的 ``/api/health`` 与前端端口真正就绪后再打开浏览器；
* ``Ctrl+C`` 一次停掉两个子进程（Windows 连子进程树一起收）。

日常用法::

    python scripts/launch.py

Windows 直接双击根目录的 ``start.bat``；macOS / Linux 用 ``./start.sh``。
两者都会先确保 ``.venv`` 与依赖存在，再调用本文件。

常用参数::

    --api-only             只启动 API（配合 /docs 调试接口）
    --frontend-only        只启动前端（连接已在运行的 API）
    --check                只做配置 + provider 连通性自检，不启动服务
    --no-browser           就绪后不自动打开浏览器
    --api-port 9000        固定 API 端口（被占用则报错，不自动顺延）
    --frontend-port 8601   固定前端端口
    --host 0.0.0.0         允许局域网访问（默认只监听 127.0.0.1）
    --reload               uvicorn 热重载（开发时用）
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 启动前必须能 import 的包。缺任何一个都说明依赖没装好，
#: 此时继续只会得到一个看不懂的 traceback，不如直接给出安装命令。
_REQUIRED_MODULES = ("fastapi", "uvicorn", "streamlit", "pydantic_settings", "httpx")

_REEXEC_FLAG = "MMKB_LAUNCH_REEXEC"
_DEFAULT_FRONTEND_PORT = 8501
_PORT_SEARCH_SPAN = 20


# --------------------------------------------------------------------------------------
# 环境准备
# --------------------------------------------------------------------------------------


def venv_python(root: Path = ROOT) -> Path | None:
    """返回项目虚拟环境里的解释器；不存在则返回 ``None``。"""
    candidate = (
        root / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else root / ".venv" / "bin" / "python"
    )
    return candidate if candidate.exists() else None


def missing_modules(modules: tuple[str, ...] = _REQUIRED_MODULES) -> list[str]:
    """检查依赖是否可导入，**不实际 import**（避免副作用与慢启动）。"""
    missing: list[str] = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def bootstrap_venv(argv: list[str]) -> None:
    """如果当前解释器缺依赖、而项目里有 ``.venv``，切换到 ``.venv`` 重跑。

    这样用户在任意终端里执行 ``python scripts/launch.py`` 都能工作，
    不必先手动激活虚拟环境。已经处于 ``.venv`` 时不做任何事。
    """
    if os.environ.get(_REEXEC_FLAG) == "1":
        return
    target = venv_python()
    if target is None:
        return
    try:
        if Path(sys.executable).resolve() == target.resolve():
            return
    except OSError:
        return
    if not missing_modules():
        return

    env = dict(os.environ, **{_REEXEC_FLAG: "1"})
    completed = subprocess.run(
        [str(target), str(Path(__file__).resolve()), *argv],
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    raise SystemExit(completed.returncode)


def ensure_env_file(root: Path = ROOT) -> bool:
    """确保 ``.env`` 存在；返回是否刚刚从 ``.env.example`` 复制了一份。"""
    env_path = root / ".env"
    if env_path.exists():
        return False
    example = root / ".env.example"
    if not example.exists():
        return False
    shutil.copyfile(example, env_path)
    return True


def python_version_ok() -> bool:
    return sys.version_info[:2] == (3, 12)


# --------------------------------------------------------------------------------------
# 端口选择
# --------------------------------------------------------------------------------------


class PortUnavailableError(RuntimeError):
    """固定端口被占用，且调用方不允许顺延。"""


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """探测 ``host:port`` 能否绑定。

    刻意**不设** ``SO_REUSEADDR``：Windows 上它允许两个 socket 绑同一个端口，
    会让「端口被占用」永远探测不出来。
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
    except OSError:
        return False
    return True


def pick_port(
    preferred: int,
    host: str = "127.0.0.1",
    *,
    allow_shift: bool = True,
    span: int = _PORT_SEARCH_SPAN,
) -> int:
    """返回可用的端口。

    ``allow_shift=False`` 用于用户显式指定的端口：占用了就报错，
    而不是悄悄换一个让用户找不到。默认端口则允许顺延 ——
    「上次的服务还没退干净」不该阻止今天启动。
    """
    candidates = (
        range(preferred, min(65535, preferred + span) + 1)
        if allow_shift
        else [preferred]
    )
    for port in candidates:
        if port_is_free(port, host):
            return port
    raise PortUnavailableError(
        f"端口 {preferred} 已被占用"
        + (f"，往后 {span} 个端口也不可用" if allow_shift else "")
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[mmkb] 忽略无效的 {name}={raw!r}，改用 {default}")
        return default
    return value if 1 <= value <= 65535 else default


# --------------------------------------------------------------------------------------
# 子进程命令
# --------------------------------------------------------------------------------------


def build_api_command(
    *, host: str, port: int, reload: bool = False, python: str | None = None
) -> list[str]:
    command = [
        python or sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        command.append("--reload")
    return command


def build_frontend_command(
    *, port: int, address: str = "127.0.0.1", python: str | None = None
) -> list[str]:
    return [
        python or sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "frontend" / "streamlit_app.py"),
        "--server.address",
        address,
        "--server.port",
        str(port),
        # headless 不仅是不弹浏览器，还会跳过首次运行的邮箱订阅提示
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def build_frontend_env(api_base: str) -> dict[str, str]:
    """前端子进程的环境。

    浏览器加载引用图片走 ``MMKB_PUBLIC_BASE``，所以本地启动时两者相同；
    Docker 里 ``MMKB_API_BASE`` 是容器名、``MMKB_PUBLIC_BASE`` 是宿主机地址。
    """
    env = os.environ.copy()
    env["MMKB_API_BASE"] = api_base
    env["MMKB_PUBLIC_BASE"] = api_base
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def build_process_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# --------------------------------------------------------------------------------------
# 进程管理
# --------------------------------------------------------------------------------------


@dataclass
class Child:
    name: str
    prefix: str
    proc: subprocess.Popen[str]


def _pump(stream: Any, prefix: str) -> None:
    """把子进程输出逐行转发到当前终端；线程结束时静默退出。"""
    try:
        for line in iter(stream.readline, ""):
            text = line.rstrip()
            if text:
                print(f"{prefix} {text}", flush=True)
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def spawn(name: str, command: list[str], env: dict[str, str], prefix: str) -> Child:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        # 关键：不要和启动器共享 Ctrl+C 处理，否则两个子进程会各自退出，
        # 启动器来不及按顺序收尾。清理统一由 _stop() 负责。
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **kwargs,
    )
    assert proc.stdout is not None
    threading.Thread(
        target=_pump,
        args=(proc.stdout, prefix),
        daemon=True,
        name=f"mmkb-log-{name}",
    ).start()
    return Child(name=name, prefix=prefix, proc=proc)


def _stop(child: Child, *, grace_s: float = 8.0) -> None:
    """先礼后兵地停掉一个子进程（Windows 下连子进程树一起收）。"""
    proc = child.proc
    if proc.poll() is not None:
        return

    print(f"[mmkb] 正在停止 {child.name} ...", flush=True)
    if os.name == "nt":
        proc.terminate()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()

    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def _wait_http(url: str, child: Child, timeout_s: float) -> bool:
    """轮询一个 HTTP 地址，任何响应（包括 4xx/5xx）都算「服务已起来」。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if child.proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                response.read(1024)
            return True
        except urllib.error.HTTPError:
            return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.3)
    return False


def _wait_tcp(host: str, port: int, child: Child, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if child.proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _api_status(api_base: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"{api_base}/api/health", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------------------


def _print_config_problems(settings: Any) -> None:
    from app.config import collect_config_problems

    errors, warnings = collect_config_problems(settings)
    if errors:
        print("[mmkb] 配置不完整，将以 dev 模式降级启动：")
        for problem in errors:
            print(f"         - {problem}")
        print("         前端侧栏与 GET /api/health 会列出具体缺项；补齐后重启即可。")
    elif warnings:
        print("[mmkb] 配置告警（不影响启动）：")
        for warning in warnings:
            print(f"         - {warning}")
    if settings.app.is_prod and errors:
        print("[mmkb] 注意：APP_ENV=prod，API 启动时会因配置不全而拒绝服务。")


def _print_api_state(api_base: str) -> None:
    snapshot = _api_status(api_base)
    if snapshot is None:
        print("[mmkb] API 已启动，但健康检查暂时不可读；请查看上方 [api] 日志。")
        return
    status = snapshot.get("status", "unknown")
    if snapshot.get("ready"):
        print(f"[mmkb] API 就绪：{status}")
    else:
        print(f"[mmkb] API 已启动但未就绪：{status}")
        print("         缺 Key / 数据库不可用时会这样；前端仍可打开并显示原因。")


def _print_banner(
    *,
    api_base: str | None,
    frontend_url: str | None,
    api_only: bool,
    no_browser: bool,
) -> None:
    line = "=" * 68
    print()
    print(line)
    print("  mmkb 已启动")
    if api_base:
        print(f"    API    {api_base}")
        print(f"    文档   {api_base}/docs")
    if frontend_url:
        print(f"    前端   {frontend_url}")
    if api_only:
        print("    提示   --api-only：只启动了 API，按 Ctrl+C 停止")
    elif no_browser:
        print("    提示   --no-browser：请手动打开上面的前端地址，按 Ctrl+C 停止")
    else:
        print("    提示   浏览器已打开；按 Ctrl+C 一次性停止所有服务")
    print(line)
    print(flush=True)


# --------------------------------------------------------------------------------------
# 自检模式
# --------------------------------------------------------------------------------------


def run_provider_check(settings: Any) -> int:
    """真实调用各 provider 后退出；返回进程退出码。"""
    from app.config import collect_config_problems
    from app.providers import factory

    errors, _ = collect_config_problems(settings)
    if errors:
        print("[mmkb] 配置不完整，无法进行 provider 自检：")
        for problem in errors:
            print(f"  - {problem}")
        return 2

    try:
        bundle = factory.get_providers()
    except Exception as exc:  # noqa: BLE001 - CLI 需要把原因完整打给用户
        print(f"[mmkb] provider 装配失败：{type(exc).__name__}: {exc}")
        return 2

    async def _verify() -> int:
        try:
            report = await factory.verify_providers(bundle)
            print(report.render())
            if not report.ok:
                for check in report.failures:
                    upstream = (check.extra or {}).get("detail")
                    if upstream:
                        print(
                            f"\n  {check.name} 上游详情：\n"
                            f"  {json.dumps(upstream, ensure_ascii=False)}"
                        )
            return 0 if report.ok else 1
        finally:
            await bundle.aclose()

    return asyncio.run(_verify())


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一键启动 mmkb（FastAPI + Streamlit）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-only", action="store_true", help="只启动 API")
    parser.add_argument("--frontend-only", action="store_true", help="只启动前端")
    parser.add_argument("--check", action="store_true", help="只做配置与 provider 自检")
    parser.add_argument("--no-browser", action="store_true", help="就绪后不打开浏览器")
    parser.add_argument("--reload", action="store_true", help="uvicorn 热重载（开发用）")
    parser.add_argument("--api-port", type=int, default=None, help="固定 API 端口")
    parser.add_argument("--frontend-port", type=int, default=None, help="固定前端端口")
    parser.add_argument(
        "--host",
        default=None,
        help="API 监听地址；默认 127.0.0.1（局域网访问用 0.0.0.0）",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=90.0,
        help="等待服务就绪的秒数",
    )
    return parser


def _raise_keyboard_interrupt(signum: int, frame: Any) -> None:
    """把 SIGTERM 归一到 Ctrl+C 的处理路径，保证 finally 里的清理会执行。"""
    raise KeyboardInterrupt


def _validate_port_arg(value: int | None, name: str) -> bool:
    if value is None:
        return True
    if 1 <= value <= 65535:
        return True
    print(f"[mmkb] {name} 必须是 1–65535，收到 {value}")
    return False


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap_venv(argv)
    args = _parser().parse_args(argv)

    if args.api_only and args.frontend_only:
        print("[mmkb] --api-only 与 --frontend-only 不能同时使用")
        return 2
    if not _validate_port_arg(args.api_port, "--api-port"):
        return 2
    if not _validate_port_arg(args.frontend_port, "--frontend-port"):
        return 2

    # Ctrl+C 在 Windows 上走 SIGINT，Python 默认就转成 KeyboardInterrupt；
    # 但 Ctrl+Break 走 SIGBREAK，默认动作是直接终止进程，会让 finally 里的
    # 子进程清理被跳过。这里把 SIGTERM / SIGBREAK 都归一到同一条清理路径。
    for sig_name in ("SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _raise_keyboard_interrupt)
        except (ValueError, OSError):
            pass  # 非主线程运行时无法安装信号处理器，不影响正常使用

    if not python_version_ok():
        print(
            f"[mmkb] 需要 Python 3.12，当前是 "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        print("       请用 Python 3.12 创建 .venv 后重试（Windows 双击 start.bat 即可）。")
        return 3

    absent = missing_modules()
    if absent:
        print(f"[mmkb] 缺少依赖：{', '.join(absent)}")
        print("       请先运行 start.bat（Windows）或 ./start.sh（macOS/Linux）完成安装，")
        print("       或手动执行：python -m pip install -e \".[dev]\"")
        return 3

    from app.config import ConfigurationError, get_settings

    if ensure_env_file():
        print("[mmkb] 未找到 .env，已从 .env.example 复制一份。")
        print("       请填入 LLM / TEXT_EMBED / IMAGE_EMBED / RERANK 的 API Key；")
        print("       本次仍会以 dev 模式启动，前端会显示 not_ready。")

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print("[mmkb] 配置校验失败：")
        for problem in exc.problems:
            print(f"  - {problem}")
        print("       可运行 python scripts/show_config.py 查看每一项的来源。")
        return 2

    if args.check:
        return run_provider_check(settings)

    _print_config_problems(settings)

    from app.core.console import enable_ansi, paint

    enable_ansi()
    api_prefix = f"[{paint('api', 'cyan')}]"
    web_prefix = f"[{paint('web', 'blue')}]"

    # ---- 端口：显式指定则严格，否则占用后顺延 --------------------------------------
    api_host = args.host or "127.0.0.1"
    display_host = "127.0.0.1" if api_host in ("0.0.0.0", "::", "") else api_host

    api_port = 0
    api_base = ""
    if not args.frontend_only:
        preferred_api = args.api_port or settings.app.port
        try:
            api_port = pick_port(
                preferred_api,
                api_host,
                allow_shift=args.api_port is None,
            )
        except PortUnavailableError as exc:
            print(f"[mmkb] {exc}。请先停止占用端口的进程，或用 --api-port 指定别的端口。")
            return 4
        if api_port != preferred_api:
            print(f"[mmkb] 端口 {preferred_api} 已被占用，API 改用 {api_port}")
        api_base = f"http://{display_host}:{api_port}"
    else:
        api_base = os.getenv(
            "MMKB_API_BASE", f"http://127.0.0.1:{settings.app.port}"
        ).rstrip("/")

    frontend_port = 0
    frontend_url = ""
    if not args.api_only:
        preferred_ui = args.frontend_port or _env_int(
            "MMKB_FRONTEND_PORT", _DEFAULT_FRONTEND_PORT
        )
        try:
            frontend_port = pick_port(
                preferred_ui,
                "127.0.0.1",
                allow_shift=args.frontend_port is None,
            )
        except PortUnavailableError as exc:
            print(f"[mmkb] {exc}。请先停止占用端口的进程，或用 --frontend-port 指定别的端口。")
            return 4
        if frontend_port != preferred_ui:
            print(f"[mmkb] 端口 {preferred_ui} 已被占用，前端改用 {frontend_port}")
        frontend_url = f"http://127.0.0.1:{frontend_port}"

    # ---- 启动 ----------------------------------------------------------------------
    children: list[Child] = []
    api_child: Child | None = None
    web_child: Child | None = None

    try:
        if not args.frontend_only:
            print(f"[mmkb] 启动 API：{api_host}:{api_port}")
            api_child = spawn(
                "API",
                build_api_command(
                    host=api_host, port=api_port, reload=args.reload
                ),
                build_process_env(),
                api_prefix,
            )
            children.append(api_child)

        if not args.api_only:
            print(f"[mmkb] 启动前端：{frontend_url}（后端 {api_base}）")
            web_child = spawn(
                "前端",
                build_frontend_command(port=frontend_port),
                build_frontend_env(api_base),
                web_prefix,
            )
            children.append(web_child)

        if api_child is not None:
            if not _wait_http(
                f"{api_base}/api/health", api_child, args.startup_timeout
            ):
                print(
                    "[mmkb] API 在 "
                    f"{args.startup_timeout:.0f}s 内没有就绪"
                    + (
                        f"（进程已退出，code={api_child.proc.returncode}）"
                        if api_child.proc.poll() is not None
                        else "（仍在运行，请查看上方日志）"
                    )
                )
                return 1
            _print_api_state(api_base)

        if web_child is not None:
            if not _wait_tcp(
                "127.0.0.1", frontend_port, web_child, args.startup_timeout
            ):
                print(
                    "[mmkb] 前端在 "
                    f"{args.startup_timeout:.0f}s 内没有开始监听"
                    + (
                        f"（进程已退出，code={web_child.proc.returncode}）"
                        if web_child.proc.poll() is not None
                        else "（仍在运行，请查看上方日志）"
                    )
                )
                return 1

        _print_banner(
            api_base=api_base if api_child is not None else None,
            frontend_url=frontend_url if web_child is not None else None,
            api_only=args.api_only,
            no_browser=args.no_browser,
        )

        if web_child is not None and not args.no_browser:
            try:
                webbrowser.open(frontend_url)
            except Exception as exc:  # noqa: BLE001 - 打不开浏览器不是致命问题
                print(f"[mmkb] 自动打开浏览器失败（{exc}），请手动访问 {frontend_url}")

        # ---- 值守：任一进程退出就收掉另一个 -------------------------------------
        while True:
            for child in children:
                code = child.proc.poll()
                if code is not None:
                    print(
                        f"[mmkb] {child.name} 进程已退出（code={code}），"
                        "正在停止其余服务 ..."
                    )
                    return 0 if code == 0 else 1
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[mmkb] 收到 Ctrl+C，正在停止 ...")
        return 0
    finally:
        for child in reversed(children):
            _stop(child)


if __name__ == "__main__":
    raise SystemExit(main())
