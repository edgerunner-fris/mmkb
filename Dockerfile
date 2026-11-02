# mmkb 运行镜像：API 与 Streamlit 前端共用同一个镜像，只是 command 不同。
#
# 设计取舍：
# * 不把 .env 打进镜像（.dockerignore 已排除），密钥只在运行时注入；
# * 所有推理都走云端 API，镜像里没有任何本地模型，因此不需要 GPU 基础镜像；
# * 非 root 运行。Linux 宿主机上如果 ./data 由 root 创建，需要
#   `sudo chown -R 1000:1000 data`（UID 与镜像内 appuser 一致）。

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# curl 只用于 compose 的 healthcheck；装完就删 apt 列表，避免镜像里留垃圾
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 先抄依赖声明，尽量让依赖层在源码变更时命中缓存。
# pip install . 需要 app 源码，所以源码必须一起进来；这里不做
# `--no-deps` 拆分，因为本项目依赖不多，可读性比几秒构建时间重要。
COPY pyproject.toml README.md ./
COPY app ./app
COPY frontend ./frontend
COPY eval ./eval

RUN pip install .

# 数据目录在 compose 里由 ./data 挂载覆盖；镜像内自建一份，
# 这样 `docker run` 不带 volume 也能直接启动。
RUN mkdir -p /app/data \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

# 默认起 API；前端服务在 compose 里覆盖 command。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
