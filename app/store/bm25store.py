"""BM25 稀疏检索索引。

**为什么中文必须走 jieba**
------------------------
BM25 的基础是词频。中文没有空格，`"检索模块的设计"` 如果不分词就是**一个** token ——
那么「检索」这个词永远匹配不上任何查询，整条 BM25 路会静默失效：
不报错、不抛异常，只是召回质量莫名其妙地差。这是最容易被忽略的一类 bug，
因为它在纯英文语料上完全正常。

**分词策略**（``BM25_TOKENIZER``）

==========  ==================================================================
取值         行为
==========  ==================================================================
``auto``    默认。**按脚本切分文本**：汉字段交给 jieba，其余段走正则小写。
            中英混排的文档因此能同时得到两种切分，而不是二选一。
``jieba``   整段交给 jieba（它自带英文处理），保留 jieba 的原始切分粒度。
``regex``   纯正则。汉字按**单字**切，英文按小写单词切。
            存在的意义是做 A/B —— 同一份评测集上跑 ``jieba`` vs ``regex``
            能直接量化「分词对中文检索的贡献」，这正是本项目要展示的东西之一。
==========  ==================================================================

**零分与负分必须过滤掉**
----------------------
``rank_bm25`` 对没有命中任何查询词的文档返回 ``0``。这些文档如果进入 RRF，
会以「第 300 名」的身份贡献一份融合分数，把真正相关的候选挤下去 ——
BM25 的「没命中」是**信息**，不是「排得比较靠后」，必须原样保留这个区别。

负分同样存在，而且更隐蔽：``rank_bm25`` 把 idf 为负的词（出现在超过一半文档里的词）
统一抬到一个下限 ``epsilon * average_idf``。但当语料很小、``average_idf`` 本身为负时，
这个「下限」是个负数 —— 只含这类词的文档就会得到负分。
所以过滤条件是 ``score > 0``，不是 ``score >= 0``。

**索引是增量的，但 BM25Okapi 不是**
----------------------------------
``BM25Okapi`` 在构造时一次性算好 idf 与平均文档长度，没有「加一篇文档」的接口。
所以本模块自己维护 ``{id: tokens}``，**标记脏位**，在下一次 ``search`` 时才重建 ——
连续导入 100 个文档只会触发 1 次重建，而不是 100 次。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.perf import get_metrics

__all__ = ["BM25Doc", "BM25Hit", "BM25Store", "tokenize"]

logger = get_logger(__name__)

#: 索引文件格式版本。改了结构就 +1，加载时版本不符会直接重建而不是读出乱码。
_INDEX_VERSION = 1

#: 汉字与常见扩展区。用它把文本按脚本切成若干段。
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿぀-ヿ]+")
#: 拉丁字母 / 数字词。保留词内的 . _ + # -，这样 ``gpt-4`` ``bge-m3`` ``v1.2``
#: ``c++`` 不会被切碎 —— 技术文档里这类词很常见，切碎了就再也匹配不上。
_WORD_RE = re.compile(r"[a-z0-9]+(?:[._+#-][a-z0-9]+)*")
#: 单个汉字。``regex`` 模式下用。
_SINGLE_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

TokenizerName = Literal["auto", "jieba", "regex"]

_jieba_lock = asyncio.Lock()
_jieba_ready = False


def _ensure_jieba() -> Any:
    """惰性 import + 首次调用时预热词典。

    jieba 第一次 ``cut`` 会加载词典并建前缀树，冷启动要几百毫秒到一秒。
    在请求线程里做这件事会让「第一次提问」明显变慢，
    所以在锁里预热一次，之后所有调用都是热的。
    """
    global _jieba_ready
    import jieba

    if not _jieba_ready:
        # jieba 在 **import 时**把自己的 logger 设成 DEBUG 并挂一个
        # StreamHandler。它比 setup_logging() 晚发生（这里是惰性导入），
        # 所以全局的日志级别压不住它 —— 每次冷启动都会往 stderr 打三行
        # 「Building prefix dict...」。在 import 之后显式调一次才有效。
        jieba.setLogLevel(logging.WARNING)
        jieba.initialize()
        _jieba_ready = True
    return jieba


def tokenize(
    text: str,
    *,
    tokenizer: TokenizerName = "auto",
    stopwords: frozenset[str] = frozenset(),
    min_len: int = 1,
) -> list[str]:
    """把一段文本切成 BM25 用的 token 列表。**纯函数**，方便单独测。

    三种模式的区别见模块 docstring。所有模式都会：转小写、去空白、
    丢掉长度小于 ``min_len`` 的 token、丢掉停用词。

    数字与纯标点不作为 token：它们在 BM25 里几乎不携带区分度，
    留在索引里只会让 ``avgdl`` 变大、把真正有信息量的词稀释掉。
    """
    if not text:
        return []

    raw: list[str] = []
    if tokenizer == "jieba":
        jieba = _ensure_jieba()
        raw = [piece.strip() for piece in jieba.lcut(text, cut_all=False)]
    elif tokenizer == "regex":
        # 汉字逐字切：至少在「检索」这种两字词上还能跟单字查询对上，
        # 比整段变一个 token 强得多。英文照常。
        raw = _SINGLE_CJK_RE.findall(text.lower())
        raw.extend(_WORD_RE.findall(text.lower()))
    else:  # auto
        # 先把文本按「汉字段 / 非汉字段」切开，两种段分别处理。
        # 这样中英混排的一句话里，中文得到词级切分、英文得到小写词切分。
        jieba = _ensure_jieba()
        cursor = 0
        for match in _CJK_RE.finditer(text):
            non_cjk = text[cursor : match.start()]
            if non_cjk:
                raw.extend(_WORD_RE.findall(non_cjk.lower()))
            raw.extend(piece.strip() for piece in jieba.lcut(match.group(0), cut_all=False))
            cursor = match.end()
        tail = text[cursor:]
        if tail:
            raw.extend(_WORD_RE.findall(tail.lower()))

    out: list[str] = []
    for token in raw:
        token = token.lower()
        if len(token) < min_len:
            continue
        if token in stopwords:
            continue
        if not any(ch.isalnum() for ch in token):
            # 纯标点。jieba 会把标点原样吐出来，Chroma 侧无所谓，BM25 侧必须滤掉
            continue
        out.append(token)
    return out


# --------------------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class BM25Doc:
    """要进索引的一条文档。``payload`` 与向量库 metadata 同规则：只放标量。

    带上 payload 而不是只存 id，是为了让 BM25 的命中**可以直接变成
    ``RetrievedItem``**，不必为每条命中再回查一次数据库 ——
    检索链路上每次回查都是一次 await，20 条命中就是 20 次往返。
    """

    id: str
    text: str
    payload: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("BM25Doc.id 不能为空")
        if not self.text.strip():
            # 空文档会让 avgdl 被拉低，还会在语料里留下一个永远命中不了的槽位
            raise ValueError(f"BM25Doc({self.id}) 的 text 为空")
        for key, value in self.payload.items():
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise TypeError(
                    f"BM25Doc({self.id}).payload[{key!r}] 是 {type(value).__name__}；"
                    "只接受 str/int/float/bool（要存列表请先 json.dumps）"
                )


@dataclass(slots=True)
class BM25Hit:
    """一次 BM25 命中。``rank`` 从 1 开始，与 RRF 的 ``1/(k + rank)`` 直接对接。"""

    id: str
    score: float
    rank: int
    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class _KBIndex:
    """一个知识库的内存索引。``bm25`` 为 ``None`` 表示脏，下次查询时重建。"""

    __slots__ = ("bm25", "docs", "dirty", "kb_id", "tokens")

    def __init__(self, kb_id: str) -> None:
        self.kb_id = kb_id
        #: id → BM25Doc。用 dict 而不是 list：删除是 O(1)，
        #: 且「重复导入同一文档」天然幂等（同 id 覆盖）。
        self.docs: dict[str, BM25Doc] = {}
        #: id → tokens。**与 docs 分开存**是为了持久化：
        #: 存 tokens 就不必在加载时重跑一遍 jieba。
        self.tokens: dict[str, list[str]] = {}
        self.bm25: Any | None = None
        self.dirty: bool = True

    @property
    def size(self) -> int:
        return len(self.docs)

    def mark_dirty(self) -> None:
        self.bm25 = None
        self.dirty = True


# --------------------------------------------------------------------------------------
# 索引
# --------------------------------------------------------------------------------------


class BM25Store:
    """BM25 索引仓库。进程级单例（见 :func:`app.store.factory.get_bm25store`）。

    内存里的索引是**权威副本**，磁盘上的 JSON 只是缓存 ——
    源数据始终在 SQLite 的 ``chunks`` 表里。所以索引文件丢了、格式变了、
    版本对不上，都只需要重建，不构成数据丢失。
    """

    def __init__(self, settings: Settings | None = None, *, index_dir: Path | None = None) -> None:
        self._settings = settings or get_settings()
        cfg = self._settings.bm25
        self._index_dir = Path(index_dir) if index_dir is not None else self._settings.bm25_dir
        self._tokenizer: TokenizerName = cfg.tokenizer
        self._k1 = cfg.k1
        self._b = cfg.b
        self._persist = cfg.persist
        self._stopwords = frozenset(w.lower() for w in cfg.stopword_list)
        self._min_len = cfg.min_token_len
        self._indexes: dict[str, _KBIndex] = {}
        self._lock = asyncio.Lock()
        self._warned_regex_on_cjk = False

    # ------------------------------------------------------------------ 属性

    @property
    def tokenizer(self) -> TokenizerName:
        return self._tokenizer

    @property
    def index_dir(self) -> Path:
        return self._index_dir

    def index_path(self, kb_id: str) -> Path:
        return self._index_dir / f"{_safe_name(kb_id)}.json"

    # ------------------------------------------------------------------ 分词

    def tokenize(self, text: str) -> list[str]:
        if self._tokenizer == "regex" and not self._warned_regex_on_cjk:
            if _CJK_RE.search(text):
                # 只提醒一次：这是配置选择，不是错误，但结果差异很大，值得知道
                logger.info(
                    "BM25_TOKENIZER=regex：汉字按单字切分，中文召回会明显弱于 jieba。"
                    "这是评测对比用的档位，生产建议用 auto。"
                )
            self._warned_regex_on_cjk = True
        return tokenize(
            text,
            tokenizer=self._tokenizer,
            stopwords=self._stopwords,
            min_len=self._min_len,
        )

    # ------------------------------------------------------------------ 索引生命周期

    async def ensure_kb(self, kb_id: str) -> bool:
        """确保内存里有这个知识库的索引。返回是否从磁盘**加载**成功。

        加载失败（文件不存在 / 版本不符 / JSON 损坏）不是错误 —— 返回 ``False``，
        调用方随后 :meth:`add_documents` 会把索引重新建起来。
        """
        async with self._lock:
            if kb_id in self._indexes:
                return True
            loaded = self._load_sync(kb_id)
            if not loaded:
                self._indexes[kb_id] = _KBIndex(kb_id)
            return loaded

    async def add_documents(self, kb_id: str, docs: list[BM25Doc]) -> int:
        """增量加入 / 覆盖文档。返回索引里当前的文档总数。

        **不在这里重建 BM25 索引** —— 只标脏位。导入 100 个文档时，
        每加一个都重建一次 idf 是 O(n²) 的无谓开销。
        """
        if not docs:
            return await self.count(kb_id)
        async with self._lock:
            # 新进程里第一次写入时必须先把磁盘上的索引捞回来。
            # 少了这一步，本次只导入一份新文件就会把持久化文件覆盖成
            # 「只含这一份文件」的版本 —— 之前导入的文档在 BM25 里全部消失，
            # 而向量库与 SQLite 里它们还在，表现为「检索偶尔漏掉整篇文档」。
            if kb_id not in self._indexes:
                self._load_sync(kb_id)
            index = self._indexes.setdefault(kb_id, _KBIndex(kb_id))
            for doc in docs:
                # 同一个 id 重复加入时覆盖，且必须让 tokens 跟着更新，
                # 否则 docs 与 tokens 会指向两个不同版本的文本
                index.tokens[doc.id] = self.tokenize(doc.text)
                index.docs[doc.id] = doc
            index.mark_dirty()
            return index.size

    async def remove_documents(self, kb_id: str, ids: list[str]) -> int:
        """按 id 删除。返回实际删掉的条数。"""
        if not ids:
            return 0
        async with self._lock:
            index = self._indexes.get(kb_id)
            if index is None:
                return 0
            removed = 0
            for doc_id in ids:
                if index.docs.pop(doc_id, None) is not None:
                    index.tokens.pop(doc_id, None)
                    removed += 1
            if removed:
                index.mark_dirty()
            return removed

    async def remove_documents_by_doc(self, kb_id: str, doc_id: str) -> int:
        """删掉某个文档带来的所有分块。

        重导同一份文件时必须先调它：``chunk_id`` 里带 ordinal，
        文件改短之后旧的高序号分块不会被新导入覆盖，会变成幽灵结果。
        """
        async with self._lock:
            index = self._indexes.get(kb_id)
            if index is None:
                # 同上：重导发生在另一个进程里时，不加载就等于没删干净
                self._load_sync(kb_id)
                index = self._indexes.get(kb_id)
            if index is None:
                return 0
            doomed = [
                key
                for key, doc in index.docs.items()
                if str(doc.payload.get("doc_id") or "") == doc_id
            ]
            for key in doomed:
                index.docs.pop(key, None)
                index.tokens.pop(key, None)
            if doomed:
                index.mark_dirty()
            return len(doomed)

    async def replace_kb(self, kb_id: str, docs: list[BM25Doc]) -> int:
        """全量替换。重建索引时用 —— 一次建好，而不是逐条 add。"""
        async with self._lock:
            index = _KBIndex(kb_id)
            for doc in docs:
                index.tokens[doc.id] = self.tokenize(doc.text)
                index.docs[doc.id] = doc
            index.mark_dirty()
            self._indexes[kb_id] = index
            return index.size

    async def drop_kb(self, kb_id: str) -> bool:
        """从内存与磁盘上一并删掉。删知识库时调用。"""
        async with self._lock:
            existed = self._indexes.pop(kb_id, None) is not None
            path = self.index_path(kb_id)
            if path.is_file():
                path.unlink(missing_ok=True)
                existed = True
            return existed

    async def count(self, kb_id: str) -> int:
        async with self._lock:
            index = self._indexes.get(kb_id)
            return index.size if index else 0

    async def list_kbs(self) -> list[str]:
        async with self._lock:
            return sorted(self._indexes)

    # ------------------------------------------------------------------ 检索

    async def search(
        self,
        kb_id: str,
        query: str,
        *,
        top_k: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[BM25Hit]:
        """BM25 检索。返回**按分数降序**、只含正分的结果。

        ``where`` 支持与向量库同一套形状的**等值**过滤
        （``{"doc_id": {"$in": [...]}}`` / ``{"block_type": "table"}``）。
        这里在打分**之后**过滤，而不是先缩小语料 ——
        后者会改变 idf（``corpus_size`` 变了），同一个查询在不同过滤条件下
        得到不可比的分数，RRF 融合时就成了噪声。
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        async with self._lock:
            index = self._indexes.get(kb_id)
            if index is None:
                # 检索进程可能是刚启动的：索引还在磁盘上，先加载再说。
                # 否则本进程的第一次查询会返回空列表，看起来像「知识库是空的」。
                self._load_sync(kb_id)
                index = self._indexes.get(kb_id)
            if index is None or not index.docs:
                return []
            self._build_sync(index)
            bm25 = index.bm25
            # 只复制键列表（O(n) 个字符串引用），不复制 payload / text。
            # 原实现每次查询都要构造两个与语料等大的字典，万级分块的库上
            # 每个查询白白多出几毫秒与一批 GC 压力。正文与 payload 在
            # 真正需要时按 id 取 —— 只有 top_k 条，通常不超过 20 条。
            docs = index.docs
            ids = list(docs)

        # 打分是纯 CPU 计算（O(命中词数 × 文档数)），丢线程池，
        # 免得一次大语料的检索把事件循环卡住几百毫秒
        started = time.perf_counter()
        scores = await asyncio.to_thread(bm25.get_scores, query_tokens)
        get_metrics().observe("bm25.search", (time.perf_counter() - started) * 1000)

        scored: list[tuple[str, float]] = []
        for position, raw in enumerate(scores):
            score = float(raw)
            # 见模块 docstring：「没命中」与「排得靠后」是两回事，不能混
            if not math.isfinite(score) or score <= 0.0:
                continue
            doc_id = ids[position]
            # 并发写入可能在这轮打分期间删掉了某条分块：跳过而不是 KeyError
            doc = docs.get(doc_id)
            if doc is None:
                continue
            if where is not None and not _match_payload(doc.payload, where):
                continue
            scored.append((doc_id, score))

        # 分数相同时按 id 排序：BM25 的浮点分数经常并列，
        # 不给定序的话同一查询两次运行可能给出不同顺序，评测结果就不可复现
        scored.sort(key=lambda pair: (-pair[1], pair[0]))

        hits: list[BM25Hit] = []
        for rank, (doc_id, score) in enumerate(scored[: max(1, top_k)], start=1):
            doc = docs.get(doc_id)
            if doc is None:
                continue
            hits.append(
                BM25Hit(
                    id=doc_id,
                    score=score,
                    rank=rank,
                    text=doc.text,
                    payload=dict(doc.payload),
                )
            )
        return hits

    def _build_sync(self, index: _KBIndex) -> None:
        """按需重建 BM25 索引。调用方必须已持有 ``self._lock``。"""
        if index.bm25 is not None and not index.dirty:
            return
        corpus = [index.tokens[doc_id] for doc_id in index.docs]
        if not corpus:
            # BM25Okapi([]) 会在 average_idf 的除法上 ZeroDivisionError
            index.bm25 = None
            index.dirty = False
            return
        index.bm25 = _PositiveIdfBM25(corpus, k1=self._k1, b=self._b)
        index.dirty = False
        logger.debug(
            "BM25 索引已重建",
            extra={
                "kb_id": index.kb_id,
                "docs": len(corpus),
                # avgdl 是排查「检索结果莫名其妙」时最有用的一个数：
                # 它异常小说明分词把整句切成了一个 token
                "avgdl": round(sum(len(t) for t in corpus) / len(corpus), 2),
            },
        )

    # ------------------------------------------------------------------ 持久化

    async def persist(self, kb_id: str) -> Path | None:
        """把索引写到磁盘。``BM25_PERSIST=false`` 时是空操作。

        写失败只 warning 不抛：索引是缓存，重建的代价是几秒 CPU，
        不值得让整个导入流程因为一个磁盘写失败而回滚。
        """
        if not self._persist:
            return None
        async with self._lock:
            index = self._indexes.get(kb_id)
            if index is None:
                return None
            payload = {
                "version": _INDEX_VERSION,
                "kb_id": kb_id,
                "tokenizer": self._tokenizer,
                "k1": self._k1,
                "b": self._b,
                "updated_at": datetime.now(UTC).isoformat(),
                "docs": [
                    {
                        "id": doc.id,
                        "text": doc.text,
                        "payload": doc.payload,
                        "tokens": index.tokens.get(doc.id, []),
                    }
                    for doc in index.docs.values()
                ],
            }
            path = self.index_path(kb_id)

        try:
            await asyncio.to_thread(_write_json_atomic, path, payload)
        except OSError as exc:
            logger.warning(
                "BM25 索引写盘失败，本次运行仍可用（索引在内存里）",
                extra={"kb_id": kb_id, "path": str(path), "error": str(exc)},
            )
            return None
        logger.info(
            "BM25 索引已持久化",
            extra={"kb_id": kb_id, "path": str(path), "docs": len(payload["docs"])},
        )
        return path

    def _load_sync(self, kb_id: str) -> bool:
        """从磁盘加载。调用方必须已持有 ``self._lock``。"""
        if not self._persist:
            return False
        path = self.index_path(kb_id)
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "BM25 索引文件读取失败，将重建",
                extra={"kb_id": kb_id, "path": str(path), "error": str(exc)},
            )
            return False

        if data.get("version") != _INDEX_VERSION:
            logger.info(
                "BM25 索引格式版本不符，将重建",
                extra={"kb_id": kb_id, "found": data.get("version")},
            )
            return False

        index = _KBIndex(kb_id)
        stored_tokenizer = data.get("tokenizer")
        # 分词器换了 → 存下来的 tokens 全部作废，但 text 还在，
        # 就地重新分词即可（这就是为什么索引文件里要存 text）
        retokenize = stored_tokenizer != self._tokenizer
        for entry in data.get("docs", []):
            try:
                doc = BM25Doc(
                    id=str(entry["id"]),
                    text=str(entry["text"]),
                    payload=dict(entry.get("payload") or {}),
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "BM25 索引里有损坏的记录，整份索引将重建",
                    extra={"kb_id": kb_id, "error": str(exc)},
                )
                return False
            index.docs[doc.id] = doc
            if retokenize:
                index.tokens[doc.id] = self.tokenize(doc.text)
            else:
                index.tokens[doc.id] = [str(t) for t in entry.get("tokens") or []]
        index.mark_dirty()

        if retokenize:
            logger.info(
                "BM25 索引的分词器与当前配置不同，已就地重新分词",
                extra={
                    "kb_id": kb_id,
                    "stored": stored_tokenizer,
                    "current": self._tokenizer,
                },
            )
        self._indexes[kb_id] = index
        logger.info(
            "BM25 索引已加载",
            extra={"kb_id": kb_id, "docs": index.size, "path": str(path)},
        )
        return True

    # ------------------------------------------------------------------ 运维

    def stats(self) -> dict[str, Any]:
        return {
            "tokenizer": self._tokenizer,
            "k1": self._k1,
            "b": self._b,
            "persist": self._persist,
            "index_dir": str(self._index_dir),
            "kbs": {
                kb_id: {"docs": index.size, "built": index.bm25 is not None}
                for kb_id, index in sorted(self._indexes.items())
            },
        }

    async def aclose(self) -> None:
        """把还没落盘的索引写出去。**优雅关停时必须调** ——
        否则下次启动要重跑一遍全部 jieba 分词。"""
        for kb_id in list(self._indexes):
            await self.persist(kb_id)


# --------------------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------------------


class _PositiveIdfBM25(BM25Okapi):
    """把 idf 换成**恒正**的写法。这是本项目对 ``rank_bm25`` 默认行为的一处刻意偏离。

``rank_bm25.BM25Okapi`` 用的是 Robertson idf::

    idf = log(N - df + 0.5) - log(df + 0.5)

它在 ``df`` 接近 ``N`` 时**会变成 0 甚至负数**。两个后果：

1. ``df = N - 1``（词只缺席一篇文档）时 ``idf`` 恰好落在 0 附近 ——
   只要 ``df = (N-1)/2``，正负就翻过去了。小语料上尤其致命：
   ``N = 2, df = 1`` 时 ``idf = log(1.5) - log(1.5) = 0``，
   **所有分数全部为 0**。而本模块会过滤掉零分（见模块 docstring），
   于是「刚建好、只导入了两三份文件的知识库」BM25 路完全静默失效 ——
   不报错，只是三路混合检索里少了一路，评测分数莫名其妙地低。
2. ``rank_bm25`` 对负 idf 的处理是「抬高到 ``epsilon * average_idf``」，
   但 ``average_idf`` 本身也可能是负的，于是这个「下限」是个负数，
   分数真的会为负。

这里改用 Lucene / Elasticsearch 实际使用的写法::

    idf = log(1 + (N - df + 0.5) / (df + 0.5))

它恒大于 0，且保持「词越稀有 idf 越大」的单调性，
同时不会因为语料大小而在「有效」和「全线归零」之间跳变。
这不是为了好看 —— 是为了让 BM25 这一路在**任何规模**的知识库上都有意义。

注：``_calc_idf`` 是 ``rank_bm25`` 的私有方法，覆盖它属于依赖内部实现。
选它是因为另一条路更差：自己从零实现 BM25 会引入一套需要单独验证的
分词统计与 k1/b 平滑逻辑，而这里要改的只有一行公式。
``test_store.py`` 里有一条测试专门守住「小语料下 BM25 仍然返回结果」，
所以上游一旦改了 ``_calc_idf`` 的签名，会立刻被测出来。
"""

    def _calc_idf(self, nd: Any) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
        self.average_idf = (
            sum(self.idf.values()) / len(self.idf) if self.idf else 0.0
        )


def _safe_name(kb_id: str) -> str:
    """``kb_id`` 会变成文件名。把不在白名单里的字符换掉。

    与 :mod:`app.store.blobstore` 不同，这里不做「非法就抛」——
    索引文件名是内部实现细节，用户看不到，没必要因为一个知识库名让导入失败。
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", kb_id).lstrip(".")
    return cleaned or "kb"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """``.tmp`` + ``os.replace``。理由同 blobstore：读者看不到写了一半的索引。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _match_payload(payload: dict[str, Any], where: dict[str, Any]) -> bool:
    """BM25 侧的 ``where`` 求值。

    只支持 ``$and``/``$or``/``$in``/``$nin``/``$eq``/``$ne`` ——
    范围比较（``$gt`` 等）在 BM25 上没有意义，因为 payload 里放的都是
    枚举型字段（doc_id / block_type / image_id）。
    """
    for key, condition in where.items():
        if key == "$and":
            if not all(_match_payload(payload, item) for item in condition):
                return False
            continue
        if key == "$or":
            if not any(_match_payload(payload, item) for item in condition):
                return False
            continue

        actual = payload.get(key)
        if not isinstance(condition, dict):
            if actual != condition:
                return False
            continue
        for op, expected in condition.items():
            if op == "$eq" and actual != expected:
                return False
            if op == "$ne" and actual == expected:
                return False
            if op == "$in" and actual not in list(expected):
                return False
            if op == "$nin" and actual in list(expected):
                return False
    return True
