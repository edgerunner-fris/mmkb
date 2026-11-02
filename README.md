# mmkb —— 通用多模态知识库问答系统

文档 + 图片双模态摄入 · 文本向量 / 多模态向量 / BM25 三路混合检索 + RRF 融合 + 重排 ·
带溯源引用的答案生成 · 可量化的多模态检索 A/B 消融评估。

> **状态：Batch 1–8 / 8 已交付。**
> 一键启动 · 配置层 · provider 抽象层 · 存储层 · 摄入层 · 检索层 · 问答层 ·
> HTTP 服务层 · Streamlit 前端 · Docker Compose · 五组消融评测。
> 离线测试 **372 passed, 5 skipped**（不需要任何 API Key）。架构设计见 [DESIGN.md](DESIGN.md)。

---

## 快速开始

### 方式一：一键启动（推荐）

Windows 直接**双击 `start.bat`**，或在终端执行：

```powershell
.\start.bat
```

macOS / Linux：

```bash
./start.sh
```

首次运行会自动完成这些事：创建 `.venv` → 安装 `.[dev]` 依赖 → 从 `.env.example`
复制出 `.env`（**已存在则绝不覆盖**）。填入 Key 后再次启动即可。脚本会同时拉起
API 与前端、合并带前缀的日志、自动打开浏览器，按 **Ctrl+C 一次停掉两个服务**。

```
[mmkb] 端口 8000 已被占用，API 改用 8001
[mmkb] 启动 API：127.0.0.1:8001
[mmkb] 启动前端：http://127.0.0.1:8501（后端 http://127.0.0.1:8001）
...
====================================================================
  mmkb 已启动
    API    http://127.0.0.1:8001
    文档   http://127.0.0.1:8001/docs
    前端   http://127.0.0.1:8501
====================================================================
```

| 参数 | 作用 |
|---|---|
| 无参数 | 启动 API + 前端，并自动打开浏览器 |
| `--no-browser` | 不自动打开浏览器（远程桌面 / 服务器） |
| `--api-only` | 只启动 API |
| `--frontend-only` | 只启动前端（假定 API 已在运行） |
| `--check` | 只做配置与 provider 自检，不启动服务 |
| `--api-port 8000` | 固定 API 端口；被占用直接报错，不自动换端口 |
| `--frontend-port 8501` | 固定前端端口；被占用直接报错 |
| `--host 0.0.0.0` | API 监听所有网卡（局域网访问；默认仅本机） |
| `--reload` | API 代码热重载（开发用） |
| `--startup-timeout 90` | 等待服务就绪的秒数 |

默认端口分别是 `APP_PORT`（8000）与 `MMKB_FRONTEND_PORT`（8501）；**只对默认端口**
自动顺延到下一个空闲端口，显式传入的端口不会偷偷改。前端拿到的 `MMKB_API_BASE`
和 `MMKB_PUBLIC_BASE` 会自动对齐实际 API 端口，不需要手动改环境变量。

> 没有 Key 也能启动：`/api/health` 返回 `not_ready` 并列出缺哪一项，前端侧栏同样显示。
> 此时只有依赖该 provider 的功能不可用，其余接口与页面照常打开。这是 dev 模式的预期行为。

### 方式二：本地手动（开发 / 调试）

```powershell
cd C:\Users\74021\Downloads\mmkb
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
Copy-Item .env.example .env      # 填入 LLM / TEXT_EMBED / IMAGE_EMBED / RERANK 的 Key

# 终端 1：HTTP 服务
uvicorn app.main:app --reload --port 8000

# 终端 2：前端
streamlit run frontend/streamlit_app.py
```

打开 http://127.0.0.1:8501 。前端会调用 `MMKB_API_BASE`（默认 `http://127.0.0.1:8000`）。

### 方式三：Docker Compose（一条命令）

```powershell
Copy-Item .env.example .env      # 只需在 .env 填 API Key
docker compose up --build
```

| 服务 | 地址 | 说明 |
|---|---|---|
| `api` | http://127.0.0.1:8000 | FastAPI；只有它读取 `.env` |
| `frontend` | http://127.0.0.1:8501 | Streamlit；通过 `MMKB_API_BASE=http://api:8000` 访问 API |

`frontend` 不接收任何模型密钥。浏览器加载引用图片走 `MMKB_PUBLIC_BASE`（默认
`http://localhost:8000`），因为容器名 `api` 在浏览器里不可解析。

Linux 上 `./data` 由容器内 UID 1000 写入；若宿主机目录属于其他用户，先
`sudo chown -R 1000:1000 ./data`，否则导入会因权限失败。

### 方式四：只验证，不启动

```powershell
python scripts/show_config.py          # 配置自检（密钥脱敏）
python scripts/check_providers.py      # 真实调用各 provider，校验返回维度
python -m pytest tests/                # 全量离线测试，不需要 Key
```

**自检报 `AccessDenied` / `Access denied by API-Key restrictions.` 时**：这不是请求参数或维度
配置的问题，而是**那把 Key 在百炼控制台被限制了可访问的模型**。处理方式二选一：

- 到百炼控制台编辑该 Key 的模型限制，放开对应模型（本文档里是 `text-embedding-v4`）；
- 换用一把已验证可用的同账号 Key（例如直接使用 `IMAGE_EMBED_API_KEY` 的值）。

自检现在会打印上游原始错误体，不会再只显示「拒绝了请求（400）」。

---

## 技术栈

| 层 | 选型 |
|---|---|
| 运行时 | Python 3.12 |
| Web | FastAPI + Uvicorn + Pydantic v2，接口全 async |
| 生成模型 | `deepseek-flash`（OpenAI 兼容 SDK，原生视觉 + 思考模式 + 1M 上下文 + 流式） |
| 文本向量 | DashScope `text-embedding-v4`（备选 SiliconFlow `BAAI/bge-m3`） |
| 多模态向量 | DashScope `qwen3-vl-embedding`（**原生 API**，`enable_fusion=true`） |
| 稀疏检索 | rank_bm25 + jieba |
| 重排 | DashScope `gte-rerank-v2`（备选 SiliconFlow `BAAI/bge-reranker-v2-m3`） |
| 向量库 | ChromaDB PersistentClient（另提供 PostgreSQL + pgvector 实现） |
| 元数据 | SQLite + SQLAlchemy 2.x |
| 前端 | Streamlit 单文件 |
| 部署 | Dockerfile + docker-compose |

**镜像内不装任何本地模型**：所有推理走云 API；Chroma 的默认 embedding function（会下载 ONNX 模型）
全程不使用，所有读写都显式传入向量。

---

## 三条向量空间铁律

```
铁律 1  文本查文本：query 用 text-embedding-v4 编码 → 检索 {kb_id}__text
铁律 2  以文搜图：query 文本送进 qwen3-vl-embedding 原生端点（text-only 输入），
        enable_fusion 与图片入库时完全一致 → 检索 {kb_id}__image
铁律 3  禁止混用：两者的向量空间不同，直接算余弦相似度是错的
```

## 已知限制

- **reranker 只接受纯文本，无法理解图像。** 图片类召回结果一律用 caption 文本 + 章节路径参与重排，
  该取舍带来的精度损失由 A/B 消融（`hybrid_rrf` vs `hybrid_rrf_rerank`）量化。
- DeepSeek 官方 API **没有 embedding 端点**（调用会 404），向量一律外挂独立 provider。
- 模型**不支持视频输入**，本项目不实现视频链路。
- `qwen3-vl-embedding` 在 `enable_fusion=true` 时会把**一次请求里的整批 contents 融合成 1 个向量**，
  所以图片与融合空间 query 必须逐条编码（`IMAGE_EMBED_BATCH_SIZE=1`）。配成大于 1 时
  provider 会强制按 1 发送并打警告，避免多张图共用一个向量写进库。

---

## 命令行调试入口（Batch 1–7）

```powershell
python scripts/show_config.py      # Batch 1：配置层自检
python scripts/check_providers.py  # Batch 2：真实调用各 provider 验证连通性 + 向量维度
python -m pytest tests/            # 存储 + 摄入 + 检索 + 问答 + HTTP + 评测单测（无需任何 API Key）

# Batch 7：启动 HTTP 服务。没有 Key 也能起，/api/health 会如实报告 not_ready
uvicorn app.main:app --reload --port 8000
# 浏览器打开 http://127.0.0.1:8000/docs 与 http://127.0.0.1:8000/api/health

# Batch 4：导入（--no-embed 完全离线，不需要任何 Key）
python scripts/ingest_cli.py --kb demo_batch4 --path data/raw --no-embed

# Batch 5：检索（同样可以完全离线：向量路会先探针，collection 为空自动跳过）
python scripts/search_cli.py --kb demo_batch4 --q "混合检索是如何融合三路结果的？" --debug

# Batch 6：问答（需要 LLM_API_KEY；没有 Key 时以退出码 2 明确告知，而不是抛栈）
python scripts/search_cli.py --kb demo_batch4 --q "混合检索是如何融合三路结果的？" --ask --debug
```

`search_cli.py` 是检索层最快的调试入口，不需要起服务、不需要前端：

| 参数 | 作用 |
|---|---|
| `--debug` | 每一路的状态/耗时/跳过原因，以及每条结果由哪几路以第几名捞上来 |
| `--routes bm25` | 只跑指定检索路（消融用，逗号分隔 `bm25,dense_text,dense_image`） |
| `--no-rerank` | 跳过重排，直接按 RRF 分数排序 |
| `--rewrite` | 用 LLM 做多查询改写（失败自动退回原问题，不中断检索） |
| `--ask` | 走完整问答链路（路由 → 检索 → 相关性判定 → 生成），打印答案、引用与轮次 |
| `--no-judge` | 问答模式下关闭相关性判定，只跑一轮检索 |
| `--json` | 机器可读输出（日志走 stderr，stdout 只有 JSON；`--ask` 时输出完整 `AnswerResult`） |

退出码：`0` 完成 · `1` 不可降级失败（知识库快照与配置不一致）· `2` 用法错误或配置缺失 · `130` 用户中断。

`show_config.py` 会打印全部生效配置（密钥脱敏）与配置校验结果。
在 `.env` 填好密钥之前，它会如实报告待填的致命问题 —— 这是预期行为，不是 bug。

`check_providers.py` 会**真的发请求**，逐项验证：

| 检查项 | 验证内容 | 缺 Key 时 |
|---|---|---|
| LLM 对话 | `deepseek-flash` 能返回文本 | 退出码 2（必需） |
| 文本向量 | 返回维度 == `TEXT_EMBED_DIM` | 退出码 2（必需） |
| 多模态向量 | `embed_fused` 与 `embed_images` **两条路径**都通、维度一致 | 跳过（降级） |
| 重排 | 能返回带分数的 hits | 跳过（降级） |

维度配错、模型名写错、Key 没开通这三类问题在构造期发现不了，却会让整个向量库
默默失效 —— 所以这里宁可多花几秒和几分钱，也要在启动时把它们逼出来。

### 存储层（Batch 3）

四类存储各有明确的所有权，删知识库时四者都要清：

| 模块 | 存什么 | 删除入口 |
|---|---|---|
| `docstore` | 知识库 / 文档 / 分块 / 图片记录 / 任务 / 会话 / 消息（SQLite） | `DocStore.delete_kb` |
| `vectorstore` | 文本向量 + 图像向量（Chroma，或 `[pg]` 的 pgvector） | `delete_collection` ×2 |
| `bm25store` | jieba 分词后的稀疏索引（内存 + JSON 落盘） | `drop_kb` |
| `blobstore` | 图片原图（`data/images/{kb}/{image_id}.{ext}`） | `delete_kb` |

**四者之间没有事务**，所以删除与重建都走「先标记 `deleting` → 逐个清 → 最后删记录」，
中途失败留下的是一个下次启动能识别并续做的中间态，而不是一个半死不活的知识库。

铁律 3 在这一层有第二道防线：两个向量空间落在 `{kb_id}__text` / `{kb_id}__image`
两个 collection 里，各自把 `dim` 与 `space` 写进 collection 元数据，
写入与查询前都比对。第 3 项检查（`space`）看着冗余 —— collection 名字里已经有后缀了 ——
但它拦的是「两个空间维度恰好相同」时把图像向量写进文本库这种事，光靠维度检查发现不了。

### 摄入层（Batch 4）

一条文件走完：**加载 → 解析 → 图片处理 → 分块 → 写存储**。每一步都可单独测试。

| 环节 | 模块 | 关键行为 |
|---|---|---|
| 加载 | `ingestion/loaders/` | PDF / DOCX / PPTX / XLSX / Markdown / HTML / TXT·CSV / 独立图片。`.doc`/`.ppt`/视频给出可执行的错误提示，而不是静默失败 |
| 解析 | `parser.py` | 归一化成 `Block` 流；维护标题栈生成 `heading_path`；图片锚定到前一个正文块，双向建立 `image_ids` |
| 表格 | `table_serializer.py` | 转 Markdown 表；超长表按行切分，每片重复表头，保证每片单独看都语义完整 |
| 图片 | `image_processor.py` | numpy 手写 DCT-II phash 去重（文档内 + 跨文档）→ 尺寸/纯色/装饰图过滤 → 每文档配额 → VLM caption → **双向量化**（caption 进文本空间，原图进图像空间） |
| 分块 | `chunker.py` | 按标题边界切；每个分块前缀标题路径；caption 生成独立 `synthetic` 分块并与图片块互链 |
| 编排 | `pipeline.py` | 幂等（同文件二次导入 `SKIPPED_DUPLICATE`）、dry-run、失败落 `FAILED` 文档行并带原因、降级原因写入 `degraded_reasons` |

图片的双向量化是「图文互搜」的数据基础：同一张图在文本空间里以 caption 形式存在，
在图像空间里以多模态向量形式存在 —— 两者维度不同，所以必须落在两个 collection（铁律 3）。

**离线可验证**：`--no-embed --no-caption` 时整条链路不碰任何外部 API，
只写 SQLite / 图片文件 / BM25 索引。测试套件（53 个用例）正是这样跑的。

### 检索层（Batch 5）

一次查询的完整链路：**查询规划 → 三路并行召回 → RRF 融合 → 重排 → 邻块扩展 → 元数据回填**。

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `dense_text.py` | 文本查文本 | 铁律 1：query 必须用文本向量模型编码，查 `{kb_id}__text` |
| `dense_image.py` | 以文搜图 | 铁律 2：query 送进多模态融合端点，`enable_fusion` 与入库时完全一致 |
| `sparse.py` | BM25 稀疏召回 | jieba 分词；过滤条件只下推 BM25 能表达的字段 |
| `fusion.py` | RRF | `Σ wᵢ / (k + rankᵢ)`，k=60；同分按 `item_id` 排序保证可复现 |
| `rerank.py` | 交叉编码器重排 | 图片用 caption + 章节路径拼文本参与重排（见「已知限制」） |
| `query_rewrite.py` | 多查询改写 | 原问题权重 1.0，改写查询 0.7；失败退回原问题 |
| `hybrid.py` | 编排 | 每路先探针再执行；单路失败降级，快照不一致直接报错 |

**探针（probe）先于检索。** 每一路在花掉 embedding 调用之前，先检查 `collection.count()`；
为空就跳过并说明原因。这让 `--no-embed` 导入的知识库也能完整跑通检索链路，
而不需要任何 API Key —— 也避免了为一个注定为空的结果集付钱。

**降级是分层的，不是二元的。** 单路网络失败 → 该路返回 `RouteResult.error`，
其余路照常融合，最终 `SearchOutcome.degraded_reasons` 列出 `text_embed_unavailable` /
`image_embed_unavailable` / `rerank_unavailable` / `bm25_unavailable` 等；
而知识库快照（维度、模型、空间）与当前配置不一致 → 抛 `RetrievalError`，因为那意味着
「继续跑只会返回一批错误的结果」，而不是「这一路暂时没用」。

**同一张图的身份归一。** 一张图会同时出现在文本空间（caption 分块）和图像空间（多模态向量），
可能被三路各捞到一次。`canonical_identity` 把 `image_caption` 分块的 `item_id` 归一为 `image_id`，
RRF 才能把它们当作同一候选合并，而不是让同一张图占三个名次。

检索层单测（32 个用例）覆盖 RRF 的数值、快照守卫、降级路径、重排回退、
查询改写与过滤条件下推。

### 问答层（Batch 6）

一次问答的主链路：**意图路由 → （历史压缩）→ 检索 → 相关性判定 →（最多两轮）→ 生成 → 引用回填**。

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `agent/router.py` | 四类意图：`smalltalk` / `knowledge` / `image_lookup` / `coding` | **规则优先**：整句寒暄、看图找图、代码相关各有稳定字面特征，命中即零延迟返回；只有「规则没命中但字面沾边」时才花一次 LLM 调用 |
| `agent/tools.py` | 意图 → 检索工具/路由的声明式映射 | `image_lookup` → 多模态向量 + BM25；其余 → 三路全开。工具不做降级，降级由检索层负责 |
| `agent/graph.py` | 编排循环 | 判定达标 / 判定不可用 / 无新改写 query / 达到 `AGENT_MAX_ROUNDS` 即停；**取历次判定分最高的一轮**，不是最后一轮 |
| `generation/answer.py` | 上下文预算、引用反解析、置信度、降级、流式 | 只用 context；关键结论必带 `[n]`；无依据必须回「知识库中未找到相关内容」 |
| `generation/prompts.py` | 提示词与 `<context>` 编号拼装 | `[n]` 的编号、`Citation.index`、正文引用三者由同一份入选列表产生 |

**引用不是模型自报的，而是从答案正文反解析的。** 模型只输出纯 Markdown，代码扫描
`[n]` 生成引用列表 —— 正文与引用因此不可能不一致；流式也能用同一套逻辑（JSON 生成到
一半无法解析，纯文本可以边收边发）。代码块与行内代码会先被剔除，`arr[1]` 不会被当成引用。

**置信度是算出来的，不是模型自评的**：`0.6 × 相关性分 + 0.3 × 引用覆盖率 + 0.1（重排生效）`。
模型对「自己编没编」的判断并不可靠；覆盖率按 `cited / min(available, 4)` 计算，
只引一条来源的长答案会被扣分。

**降级贯穿每一层**，且每一条都有机器可读的原因：

| 情况 | 行为 | `degraded_reasons` |
|---|---|---|
| 没有任何检索结果 | 不调生成模型，直接拒答 | 无（正常拒答） |
| 有资料但答案一个 `[n]` 都没标 | 如实返回答案 + 标记 | `citation_missing` |
| 生成模型失败 | 返回「生成模型暂时不可用」+ 资料原文 + 完整引用 | `llm_unavailable` |
| 判定模型失败 | 停止重检，用本轮结果 | `judge_unavailable` |
| 历史压缩失败 | 用原问题继续 | `llm_unavailable` |
| 单路检索失败 | 其余路照常融合 | `text_embed_unavailable` 等 |

流式事件（Batch 7 的 SSE 层只做序列化）：
`meta`（意图/轮次/工具调用）→ `token` / `reasoning`（可多次）→ `citations`（含置信度）→ `done`（汇总）。

问答层单测（54 个用例）覆盖上下文预算与编号连续性、引用反解析（含代码剔除、越界编号、
首次出现顺序）、置信度公式、生成失败降级、流式事件序列、路由规则与 LLM 兜底、
历史压缩、判定阈值与降级、两轮检索的查询选择与「保留最优轮」、寒暄跳过检索、
以图搜图只跑图像 + BM25。连同前五批、Batch 7 的 HTTP 用例、Batch 8 的评测与
以图搜图用例，当前离线测试套件为 **372 passed, 5 skipped**（无需任何 API Key）。

### 接口层（Batch 7）

分层原则：**路由只做参数校验，业务顺序全部在 `services/`**。
`AppContainer` 是进程级单例，由 lifespan 创建和关闭；路由通过 `Depends(get_container)` 拿到它，
测试则可以通过 `create_app(settings, container_factory=...)` 注入临时目录与假 provider。

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `services/container.py` | 配置、DB、四种存储、provider、retriever、agent 的唯一装配点 | provider 构造失败不阻止 dev 启动，错误进 `provider_errors`；shutdown 有明确顺序 |
| `services/kb_service.py` | 建库 / 删库 | 删除是两阶段：先标 `deleting`，再清向量、BM25、图片、元数据；进程被杀后下次启动续删 |
| `services/ingest_service.py` | 上传落盘、后台导入任务、进度心跳 | 上传立即返回 `202 + Task`；进度每 0.4 s 落一次 SQLite；服务关闭时取消运行中的任务 |
| `services/chat_service.py` | 会话、历史、Agent 调用、消息落库 | 历史在追加用户消息**之前**读取；流式只在正常 `done` 后落 assistant 消息 |
| `services/search_service.py` | 原始混合检索 + 以图搜图 | 直接暴露每路召回、RRF、重排的诊断信息；以图搜图独立走图像空间，先做快照校验 |
| `services/health_service.py` | 自检快照与启动恢复 | 重启时把遗留的 `queued/running` 任务标为失败，避免永久转圈 |

实际接口（完整定义以 `/docs` 为准）：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 存活 + 配置/存储/能力自检，**永远 200** |
| `GET` | `/api/health/ready` | 就绪探针，`not_ready` 时 503 |
| `GET` / `POST` | `/api/kbs` | 列表 / 创建（同名 409） |
| `GET` / `DELETE` | `/api/kbs/{kb_id}` | 详情 / 两阶段删除 |
| `POST` | `/api/kbs/{kb_id}/ingest` | multipart 上传，返回 `202 + Task` |
| `GET` | `/api/tasks/{task_id}` | 轮询导入进度 |
| `GET` | `/api/kbs/{kb_id}/tasks` | 任务列表 |
| `GET` | `/api/kbs/{kb_id}/documents` | 文档列表（可按状态过滤） |
| `GET` | `/api/documents/{doc_id}` | 文档详情 |
| `POST` | `/api/chat` | 非流式问答，返回 `AnswerResult` |
| `POST` | `/api/chat/stream` | SSE 流式问答 |
| `GET` | `/api/sessions/{id}/messages` | 会话历史 |
| `DELETE` | `/api/sessions/{id}` | 删除会话 |
| `POST` | `/api/kbs/{kb_id}/search` | 原始混合检索（含每路诊断） |
| `POST` | `/api/search/image` | multipart：以图搜图，直接查 `{kb}__image`，不重排 |
| `GET` | `/api/performance` | 进程内性能快照（指标 / 缓存 / 存储规模 / 最近任务） |
| `GET` | `/api/performance/report` | 同一份快照渲染出的 Markdown 报告（可直接贴进文档） |
| `POST` | `/api/performance/reset` | 重置指标与计数器（不停止服务） |
| `POST` | `/api/performance/cache/clear` | 清空 query 向量 / 重排 / collection 元信息缓存 |

**启动策略。** `APP_ENV=prod` 缺配置直接拒绝启动；`dev`（默认）降级启动 ——
没有 Key 也能打开 `/docs`，`/api/health` 返回 `status: not_ready` 并列出具体缺哪一项，
只有真正需要该能力的接口返回 503 `configuration_error`。
这不是放松要求，而是把「配置问题」和「代码问题」分开：前者该看健康检查，后者才是 500。

**SSE 帧。** 命名事件 `meta` / `token` / `reasoning` / `citations` / `done` / `error`，
统一编码为 `event: <type>\ndata: <json>\n\n`，空闲时发注释心跳 `: ping\n\n`。
心跳用「挂起 Task + `asyncio.wait(timeout)`」实现；**不能**用 `asyncio.wait_for(anext(...))` ——
超时会取消 `anext`，把 async generator 一并关闭，一次心跳就会截断整条回答。
流式接口在发出响应头**之前**先做预检（知识库存在、LLM 可用），否则一旦返回 200
就再也改不回 404/503 的 JSON。

**错误信封统一。** 普通 HTTP 与 SSE 的 `error` 事件共用
`{"error": {"code", "message", "detail"}}`；前端只写一次错误处理。

### 前端（Batch 8）

单文件 Streamlit，四个页签，全部走真实 HTTP 接口：

| 页签 | 能做什么 | 关键实现 |
|---|---|---|
| 问答 | 新建/切换知识库、流式问答、查看引用图片、置信度/轮次/耗时/降级原因 | 手写 SSE 客户端：`httpx.Client.stream` 逐行解析 `event:` / `data:`，忽略 `:` 心跳；`meta → token → citations → done` 顺序渲染 |
| 知识库 | 上传文件（可勾选 embed / caption / reindex）、查询导入进度、按状态过滤文档、两阶段删除 | 上传返回 `202 + Task` 后轮询 `/api/tasks/{id}`；删除需要二次确认 |
| 检索诊断 | 文本查询或**以图搜图**、选择检索路、`top_k`、是否重排/改写，查看每路状态与最终结果 | 直接展示 `routes[]` 的 `skipped / error / degraded_reason`；上榜来源标明 `bm25:1 dense_image:2`；以图搜图调用 `/api/search/image`，结果中的图片直接渲染原图 |
| 性能 | 端到端 / 检索 / 缓存命中率 / 导入任务的实时指标，检索与导入阶段耗时柱状图，最近任务，Markdown 报告下载 | 只读 `/api/performance`；重置指标与清缓存两个按钮分别打 `POST /reset`、`POST /cache/clear` |

侧栏显示 `/api/health` 的真实状态：`status`、各 provider 能力、缺失原因。
没有 Key 时页面仍能打开，只是相关操作会收到 503 并显示统一错误信息。

图片引用通过 `MMKB_PUBLIC_BASE` 拼绝对地址，因此 Docker 场景下浏览器也能加载。

### 评测（Batch 8）

评测是**纯 CLI**，不暴露 HTTP 接口 —— 它要跑几分钟到几十分钟、写报告文件、
读取用户指定路径的数据集，做成 Web 接口只会扩大攻击面。

```powershell
# 1. 复制并人工标注（5 条示例全部 labeled=false，不会参与评分）
Copy-Item eval/samples/eval_set.example.jsonl data/eval/eval_set.jsonl
notepad data/eval/eval_set.jsonl

# 2. 跑五组消融 + 指定组生成评测
python -m app.eval.run_eval --kb demo --ablation all --generation-group hybrid_rrf_rerank

# 常用变体
python -m app.eval.run_eval --kb demo --ablation hybrid_rrf,hybrid_rrf_rerank
python -m app.eval.run_eval --kb demo --ablation all --skip-generation   # 只算确定性指标
python -m app.eval.run_eval --kb demo --ablation all --use-ragas         # 有 RAGAS 时启用
python -m app.eval.run_eval --kb demo --max-questions 20 --no-chart
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `--kb` | 必填 | 知识库 ID，找不到时按名字查 |
| `--dataset` | `data/eval/eval_set.jsonl` | JSONL 数据集 |
| `--ablation` | `all` | 逗号分隔的五组之一/多组，或 `all` |
| `--top-k` | `10` | 会自动放大到至少 `max(Recall@k, MRR@k, nDCG@k)` |
| `--generation-group` | `hybrid_rrf_rerank` | 只对这一组生成 + LLM judge |
| `--skip-generation` | 关 | 只算检索指标 |
| `--use-ragas` | 关 | 依赖缺失时自动回退自实现指标并记录原因 |
| `--max-questions` | `0`（全部） | 只取前 N 条**已标注**用例 |
| `--output` | `data/eval` | 产物目录 |
| `--no-chart` | 关 | 不生成 matplotlib 图 |

产物：

| 文件 | 内容 |
|---|---|
| `report.md` | 六节：概览、五组检索指标、跨模态指标、分组对比、生成指标、复现说明 |
| `report.json` | 机器可读的完整结果（含每条用例的名次与降级原因） |
| `cases.jsonl` | 逐题结果，便于定位某一道题为什么没召回 |
| `ablation_text.png` | 文本任务上五组的 Recall@5 / MRR@10 对比 |
| `ablation_cross_modal.png` | 以文搜图 / 以图搜图的 Recall@5 对比 |

**未标注的题不会计 0 分，而是被跳过。** 指标在无相关样本时返回 `None`，报告显示 `—`。
`labeled=true` 但 `ground_truth` 为空的用例会在加载阶段直接报错，避免「以为标了其实没标」。

五组消融的配置与指标定义见 [DESIGN.md](DESIGN.md#五组-ab-消融batch-8)。

退出码：`0` 完成 · `1` 运行失败（含快照不一致）· `2` 数据集/用法/配置问题 · `130` 中断。

`RetrievalError`（维度、模型或空间与知识库快照不一致）会**直接中止整次评测**，
而不是给每道题记一条相同错误 —— 否则报告看起来像「效果差」，实际是配置错了。

### 性能：实时报告、缓存与基准

性能这件事被拆成三个互不混淆的口径：

| 口径 | 回答什么问题 | 入口 |
|---|---|---|
| 进程指标 | 这个服务**到现在**有多快、慢在哪一段、缓存有没有生效 | `GET /api/performance` · Streamlit「性能」页签 |
| 离线基准 | 固定 query 集、固定重复次数下的**可复现**延迟 | `python scripts/benchmark.py` |
| 检索质量 | 优化之后召回**是不是变差了** | `python -m app.eval.run_eval`（延迟也写进报告第 5 节） |

**进程指标**（`app/core/perf.py` + `app/services/perf_service.py`）：

- 进程内 `MetricsRegistry`，线程安全，每项指标保留最近 2048 个样本（`DEFAULT_MAX_SAMPLES`），
  用**最近秩**算 p50/p90/p95/p99；计数与延迟样本分开，错误数单独统计；
- 指标是**进程本地**的，重启即清零 —— 报告里如实标注，不假装是持久化监控；
- 一次 `snapshot()` 同时驱动 JSON 接口、Markdown 报告与前端页面，避免三个地方各算一遍导致口径漂移；
- 覆盖 `http.request`、`qa.total`、`retrieval.total` 及其六个阶段（`plan / retrieve / fuse / rerank / expand / hydrate`）、
  三路召回各自的延迟、`ingest.file.total` 与 `hash / parse / caption / chunk / embed / commit / stats`
  七个阶段，以及 BM25 / 向量库查询。

```powershell
# 实时快照 / Markdown 报告
curl.exe http://127.0.0.1:8000/api/performance
curl.exe http://127.0.0.1:8000/api/performance/report

# 改完代码重测前，先把指标与缓存归零
curl.exe -X POST http://127.0.0.1:8000/api/performance/reset
curl.exe -X POST http://127.0.0.1:8000/api/performance/cache/clear
```

报告中的两条读法必须记住：**三路召回并发执行，阶段耗时之和不等于墙钟时间**；
**均值近似可加，p95 不可加**。报告与代码注释里都写明了这一点。

**缓存。** 三处热点带 TTL + LRU 缓存，键里包含 provider / 模型 / 维度 / 向量空间 / fusion 开关 ——
铁律 3 因此不会因为缓存而被绕过（不同空间的向量永远不会落到同一个键上）。
正确性不靠 TTL：重排键包含全部候选文本，候选一变键就变；collection 元信息在导入/删除时
按 collection 名主动失效。TTL 只负责回收「不再有人问」的旧键。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `CACHE_ENABLED` | `true` | 总开关；关掉后所有缓存直接查底层 |
| `CACHE_QUERY_EMBEDDING_TTL_S` | `600` | query 向量缓存有效期（秒） |
| `CACHE_QUERY_EMBEDDING_MAX_ENTRIES` | `512` | query 向量缓存条目上限 |
| `CACHE_RERANK_TTL_S` | `300` | 重排结果缓存有效期 |
| `CACHE_RERANK_MAX_ENTRIES` | `256` | 重排缓存条目上限 |
| `CACHE_COLLECTION_META_TTL_S` | `30` | collection 计数/快照校验缓存有效期 |

**离线基准**（`scripts/benchmark.py`）与评测一样是纯 CLI，不暴露 HTTP：

```powershell
# 热缓存：先预热 1 次，再测 5 次
python scripts/benchmark.py --kb demo --queries "三路召回如何融合？" "架构图" --repeat 5

# 冷缓存：每次测量前清空 query 向量 / 重排 / collection 元信息缓存
python scripts/benchmark.py --kb demo --queries-file eval/queries.txt --repeat 3 --cold

# 顺带测一次导入；自动建临时知识库，跑完删除
python scripts/benchmark.py --kb demo --queries "RRF" --ingest data/raw/demo.pdf

# 消融对照：只跑 BM25、关闭重排
python scripts/benchmark.py --kb demo --queries "RRF" --routes bm25 --no-rerank
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `--kb` | 必填（`--ingest-only` 除外） | 知识库名称或 `kb_` ID |
| `--queries` / `--queries-file` | 至少一个 | 查询集；文件每行一个，支持 JSONL 的 `query` 字段，`#` 为注释 |
| `--repeat` / `--warmup` | `3` / `1` | 每个查询的测量次数 / 不计入统计的预热次数 |
| `--cold` | 关 | 每次测量前清缓存，测冷路径 |
| `--routes` / `--no-rerank` / `--top-k` | 全部路 / 开 / 配置默认 | 与 `search_cli.py` 同义 |
| `--ingest` / `--ingest-repeat` | 无 / `1` | 额外测导入；`reindex` 会自动打开以绕过去重 |
| `--no-embed` / `--no-caption` | 关 | 导入基准中跳过向量化 / 图片 caption |
| `--keep-kb` | 关 | 保留导入基准的临时知识库 |
| `--output` | `data/bench` | 产物目录 |

产物是 `benchmark.md` + `benchmark.json`：总延迟、阶段拆解、分路延迟、每次查询明细、缓存命中率；
跑了 `--ingest` 时再加一节导入耗时与阶段。退出码：`0` 完成 · `1` 有导入失败 · `2` 用法/配置问题 · `130` 中断。

基准脚本用 `container.startup(recover=False)` 启动：它可能和正在运行的 API 共用同一个数据库，
把对方的 `running` 任务标成失败不是「恢复」，而是破坏。

### Docker（Batch 8）

```powershell
docker compose config          # 先校验变量与挂载
docker compose up --build
```

- 镜像基于 `python:3.12-slim`，以非 root 用户 `appuser`（UID 1000）运行；
- 镜像内**没有任何本地模型**，也不包含 `.env`；
- `./data` 挂载到 `/app/data`，SQLite / Chroma / 图片 / BM25 索引 / 评测产物都在宿主机可见；
- `api` 有 `/api/health` 健康检查，`frontend` 等它健康后再启动。

### 测试

```powershell
python -m pytest tests/ -q
# 372 passed, 5 skipped
```

5 个 skip 全部来自未安装 `[pg]` extra 的 pgvector SQL 翻译测试（`pytest.importorskip`），
不是静默跳过；RAGAS 缺失则走 `try_ragas` 的显式回退路径，由普通用例覆盖。
测试全部离线：provider 用假实现，图片用 Pillow 现场生成，没有任何网络调用。

