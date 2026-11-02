"""评测模块的离线测试。

不联网、不建库、不调模型。覆盖三类最容易悄悄错掉的东西：

1. **指标定义** —— Recall/MRR/nDCG 的边界（去重、k 截断、空标注返回 None
   而不是 0）；
2. **数据校验** —— 坏行必须带行号报错，未标注必须跳过而不是计 0；
3. **报告渲染** —— 不可用指标显示 ``—``，绝不写成 0。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.config import (
    AppSettings,
    BlobSettings,
    BM25Settings,
    DocStoreSettings,
    EvalSettings,
    ImageEmbeddingSettings,
    LLMSettings,
    PathSettings,
    RerankSettings,
    Settings,
    TextEmbeddingSettings,
    VectorStoreSettings,
    VisionSettings,
)
from app.core.errors import RetrievalError
from app.core.schemas import ItemType, KnowledgeBase, RetrievedItem
from app.eval.ablation import PRESET_ORDER, AblationError, resolve_presets
from app.eval.dataset import (
    DatasetError,
    EvalCase,
    load_cases,
    load_dataset,
    resolve_query_image,
)
from app.eval.metrics import (
    JudgeScore,
    average,
    context_precision_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    retrieval_metrics,
)
from app.eval.report import (
    CaseResult,
    EvalReport,
    GroupResult,
    format_metric,
    render_markdown,
    write_outputs,
)
from app.eval.run_eval import EvalOptions, EvalRunner, _dedupe, _unit_key

SAMPLE = Path(__file__).resolve().parents[1] / "eval" / "samples" / "eval_set.example.jsonl"


# --------------------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------------------


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        app=AppSettings(env="dev", log_level="WARNING"),
        llm=LLMSettings(api_key=SecretStr("")),
        vision=VisionSettings(enabled=False),
        text_embed=TextEmbeddingSettings(api_key=SecretStr("")),
        image_embed=ImageEmbeddingSettings(api_key=SecretStr("")),
        rerank=RerankSettings(api_key=SecretStr("")),
        paths=PathSettings(data_dir=tmp_path),
        docstore=DocStoreSettings(sqlite_path=tmp_path / "test.db"),
        blob=BlobSettings(images_dir=tmp_path / "images"),
        bm25=BM25Settings(index_dir=tmp_path / "bm25", persist=False),
        vectorstore=VectorStoreSettings(chroma_dir=tmp_path / "chroma"),
        eval=EvalSettings(recall_ks="5,10", mrr_k=10, ndcg_k=10),
    )


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(payload, ensure_ascii=False) for payload in payloads),
        encoding="utf-8",
    )
    return path


def _case(case_id: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": case_id,
        "task": "text",
        "query": f"问题 {case_id}",
        "labeled": True,
        "ground_truth": {"item_ids": ["ck_1"], "doc_ids": [], "image_ids": []},
    }
    payload.update(overrides)
    return payload


def _kb() -> KnowledgeBase:
    return KnowledgeBase(
        kb_id="kb_test",
        name="测试库",
        text_collection="kb_test__text",
        image_collection="kb_test__image",
        text_dim=1024,
        image_dim=2560,
        llm_model="deepseek-flash",
        text_embed_model="text-embedding-v4",
        image_embed_model="qwen3-vl-embedding",
        rerank_model="gte-rerank-v2",
    )


def _item(**overrides: object) -> RetrievedItem:
    payload: dict[str, object] = {
        "item_id": "ck_1",
        "item_type": ItemType.TEXT,
        "kb_id": "kb_test",
        "doc_id": "doc_1",
        "text": "内容",
    }
    payload.update(overrides)
    return RetrievedItem.model_validate(payload)


class _Container:
    """EvalRunner 构造期只需要 settings 与 llm；这里不跑检索。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = None
        self.image_embedding = None


# --------------------------------------------------------------------------------------
# 检索指标
# --------------------------------------------------------------------------------------


def test_recall_perfect_and_partial() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k(["a", "x", "y"], {"a", "c"}, 3) == 0.5
    assert recall_at_k(["a", "x", "c"], {"a", "c"}, 1) == 0.5


def test_recall_dedupes_before_scoring() -> None:
    # 同一张图被三路各召回一次；不去重会把命中数算成 3。
    # 去重后 ["a", "b"] 全中；若按原始顺序截断前 2 条，会错算成 0.5。
    assert recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0
    assert recall_at_k(["a", "a", "a", "b"], {"a", "b"}, 10) == 1.0


def test_metrics_return_none_for_empty_relevance() -> None:
    # 必须返回 None 而不是 0：报告里显示「未评估」，不能伪造一个 0 分
    assert recall_at_k(["a"], set(), 5) is None
    assert mrr_at_k(["a"], set(), 5) is None
    assert ndcg_at_k(["a"], set(), 5) is None
    assert context_precision_at_k(["a"], set(), 5) is None


def test_metrics_reject_non_positive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, 0)
    with pytest.raises(ValueError):
        mrr_at_k(["a"], {"a"}, -1)


def test_mrr_uses_first_relevant_rank() -> None:
    assert mrr_at_k(["x", "a", "b"], {"a", "b"}, 10) == 0.5
    assert mrr_at_k(["x", "y"], {"a"}, 10) == 0.0
    assert mrr_at_k(["x", "a"], {"a"}, 1) == 0.0


def test_ndcg_perfect_and_ideal_capped_at_k() -> None:
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == 1.0
    # 标注 20 条、只看前 10：前 10 全中就应该得到 1.0，
    # 不能因为 IDCG 按 20 条算而永远达不到 1
    relevant = {f"d{i}" for i in range(20)}
    ranked = sorted(relevant)[:10]
    assert ndcg_at_k(ranked, relevant, 10) == pytest.approx(1.0)
    # 相关结果排第一比排第二得分高
    assert ndcg_at_k(["a", "x"], {"a"}, 10) > ndcg_at_k(["x", "a"], {"a"}, 10)


def test_context_precision_penalises_bad_order() -> None:
    # 相关项在第 1、3 位：precision@1=1, precision@3=2/3 → AP=(1+2/3)/2
    score = context_precision_at_k(["a", "x", "b"], {"a", "b"}, 10)
    assert score == pytest.approx((1 + 2 / 3) / 2)
    assert context_precision_at_k(["a", "b"], {"a", "b"}, 10) == 1.0
    assert context_precision_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_retrieval_metrics_keys_are_stable() -> None:
    metrics = retrieval_metrics(
        ["a"], {"a"}, recall_ks=[10, 5, 5], mrr_k=10, ndcg_k=10
    )
    assert list(metrics) == [
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "context_precision",
    ]


def test_average_ignores_none() -> None:
    assert average([1.0, None, 0.0]) == 0.5
    assert average([None, None]) is None


def test_judge_score_defaults_to_not_evaluated() -> None:
    score = JudgeScore(error="no key")
    assert score.score is None


# --------------------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------------------


def test_sample_dataset_has_no_fabricated_ground_truth() -> None:
    dataset = load_dataset(SAMPLE)
    assert len(dataset.cases) == 5
    assert dataset.labeled == []
    assert len(dataset.unlabeled) == 5
    assert {case.task for case in dataset.cases} == {
        "text",
        "text_to_image",
        "image_to_image",
    }
    for case in dataset.cases:
        assert case.labeled is False
        assert case.ground_truth.is_empty


def test_labeled_case_without_ids_is_rejected_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"q1","task":"text","query":"问题","labeled":true,'
        '"ground_truth":{"item_ids":[],"doc_ids":[],"image_ids":[]}}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError) as excinfo:
        load_cases(path)
    assert "bad.jsonl:1" in str(excinfo.value)
    assert "ground_truth" in str(excinfo.value)


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "dup.jsonl", [_case("q1"), _case("q1")])
    with pytest.raises(DatasetError) as excinfo:
        load_cases(path)
    assert "重复" in str(excinfo.value)
    assert ":2" in str(excinfo.value)


def test_invalid_json_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(_case("q1"), ensure_ascii=False) + "\n{not json}\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError) as excinfo:
        load_cases(path)
    assert ":2" in str(excinfo.value)
    assert "JSON" in str(excinfo.value)


def test_image_to_image_requires_query_image(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "img.jsonl",
        [_case("q1", task="image_to_image", query_image=None)],
    )
    with pytest.raises(DatasetError) as excinfo:
        load_cases(path)
    assert "query_image" in str(excinfo.value)


def test_text_task_requires_query(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "empty.jsonl", [_case("q1", query="  ")])
    with pytest.raises(DatasetError) as excinfo:
        load_cases(path)
    assert "query" in str(excinfo.value)


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "typo.jsonl", [_case("q1", lableled=True)])
    with pytest.raises(DatasetError):
        load_cases(path)


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "with_comments.jsonl"
    path.write_text(
        "# 说明\n\n" + json.dumps(_case("q1"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert [case.id for case in load_cases(path)] == ["q1"]


def test_missing_dataset_gives_copy_command(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as excinfo:
        load_dataset(tmp_path / "nope.jsonl")
    assert "cp eval/samples/eval_set.example.jsonl" in str(excinfo.value)


def test_limit_only_counts_labeled_cases(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "mixed.jsonl",
        [
            _case("q1"),
            _case("q2"),
            _case("q3", labeled=False, ground_truth={"item_ids": []}),
        ],
    )
    dataset = load_dataset(path)
    assert len(dataset.labeled) == 2
    limited = dataset.limit(1)
    assert [case.id for case in limited.cases] == ["q1"]
    assert [case.id for case in dataset.limit(0).labeled] == ["q1", "q2"]


def test_relevant_keys_depend_on_task_type() -> None:
    text_case = EvalCase.model_validate(
        {
            "id": "q1",
            "query": "问题",
            "labeled": True,
            "ground_truth": {
                "item_ids": ["ck_1"],
                "doc_ids": ["doc_1"],
                "image_ids": ["img_1"],
            },
        }
    )
    assert text_case.relevant_keys == {"ck_1", "doc_1", "img_1"}

    image_case = EvalCase.model_validate(
        {
            "id": "q2",
            "task": "text_to_image",
            "query": "找图",
            "labeled": True,
            "ground_truth": {
                "item_ids": ["ck_1"],
                "doc_ids": ["doc_1"],
                "image_ids": ["img_1"],
            },
        }
    )
    # 跨模态只认图片 ID：命中 caption 分块不等于找到图
    assert image_case.relevant_keys == {"img_1"}


def test_resolve_query_image_tries_dataset_dir_then_project_root(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "eval"
    dataset_dir.mkdir()
    image = dataset_dir / "query.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    case = EvalCase.model_validate(
        {
            "id": "q1",
            "task": "image_to_image",
            "query_image": "query.png",
            "labeled": False,
        }
    )
    assert resolve_query_image(case, dataset_dir / "set.jsonl") == image

    missing = case.model_copy(update={"query_image": "definitely-missing.png"})
    assert resolve_query_image(missing, dataset_dir / "set.jsonl") is None


# --------------------------------------------------------------------------------------
# 消融组
# --------------------------------------------------------------------------------------


def test_resolve_presets_all_in_fixed_order() -> None:
    expected = list(PRESET_ORDER)
    assert [preset.name for preset in resolve_presets(None)] == expected
    assert [preset.name for preset in resolve_presets("all")] == expected
    assert [preset.name for preset in resolve_presets("  ")] == expected


def test_resolve_presets_keeps_caller_order_and_dedupes() -> None:
    presets = resolve_presets("hybrid_rrf,bm25_only,hybrid_rrf")
    assert [preset.name for preset in presets] == ["hybrid_rrf", "bm25_only"]


def test_resolve_presets_rejects_unknown_name() -> None:
    with pytest.raises(AblationError) as excinfo:
        resolve_presets("bm25_only,no_such_group")
    message = str(excinfo.value)
    assert "no_such_group" in message
    assert "hybrid_rrf_rerank" in message


def test_image_query_support_matches_routes() -> None:
    presets = {preset.name: preset for preset in resolve_presets("all")}
    assert presets["bm25_only"].supports_image_query is False
    assert presets["dense_text_only"].supports_image_query is False
    assert presets["dense_image_only"].supports_image_query is True
    assert presets["hybrid_rrf_rerank"].supports_image_query is True


# --------------------------------------------------------------------------------------
# Runner 纯逻辑
# --------------------------------------------------------------------------------------


def test_dedupe_preserves_order() -> None:
    assert _dedupe(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_unit_key_folds_document_for_text_task() -> None:
    case = EvalCase.model_validate(
        {
            "id": "q1",
            "query": "问题",
            "labeled": True,
            "ground_truth": {"item_ids": ["ck_1"], "doc_ids": ["doc_1"]},
        }
    )
    # 标注了文档级相关：同一文档的两个分块折叠成一个单元
    assert _unit_key(_item(item_id="ck_1", doc_id="doc_1"), case) == "doc_1"
    assert _unit_key(_item(item_id="ck_2", doc_id="doc_1"), case) == "doc_1"
    # 别的文档仍按分块计
    assert _unit_key(_item(item_id="ck_9", doc_id="doc_9"), case) == "ck_9"


def test_unit_key_for_image_task_uses_image_id() -> None:
    case = EvalCase.model_validate(
        {
            "id": "q1",
            "task": "text_to_image",
            "query": "找图",
            "labeled": True,
            "ground_truth": {"image_ids": ["img_1"]},
        }
    )
    item = _item(item_id="img_1", image_id="img_1", item_type=ItemType.IMAGE)
    assert _unit_key(item, case) == "img_1"
    # 即使文本块 id 恰好长得像图片 id，也必须走 image_id
    assert _unit_key(_item(item_id="img_2", image_id=None), case) == "img_2"


def test_runner_widens_top_k_to_metric_depth(tmp_path: Path) -> None:
    dataset = load_dataset(_write_jsonl(tmp_path / "set.jsonl", [_case("q1")]))
    runner = EvalRunner(
        _Container(_settings(tmp_path)),
        _kb(),
        dataset,
        _settings(tmp_path),
        options=EvalOptions(top_k=3),
    )
    # Recall@10 需要 10 条候选，不能被 --top-k 3 截断
    assert runner.top_k == 10


class _FatalRetriever:
    """模拟快照不一致：每道题都会以同样的错误失败。"""

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        raise RetrievalError("知识库维度与当前配置不一致，不能继续检索")


async def test_runner_aborts_on_snapshot_error_instead_of_repeating(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    container = _Container(settings)
    retriever = _FatalRetriever()
    container.retriever = retriever  # type: ignore[attr-defined]
    dataset = load_dataset(
        _write_jsonl(tmp_path / "set.jsonl", [_case("q1"), _case("q2")])
    )
    runner = EvalRunner(
        container,  # type: ignore[arg-type]
        _kb(),
        dataset,
        settings,
        options=EvalOptions(skip_generation=True),
    )

    with pytest.raises(RetrievalError):
        await runner.run(resolve_presets("bm25_only"))
    # 配置级错误不该按题重试：第一题就中止
    assert retriever.calls == 1


# --------------------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------------------


def _report(
    *,
    generation_skipped: str = "显式跳过（--skip-generation）",
    ragas_error: str = "",
) -> EvalReport:
    preset = resolve_presets("hybrid_rrf_rerank")[0]
    group = GroupResult(
        preset=preset,
        cases=[
            CaseResult(
                case_id="q1",
                task="text",
                query="问题",
                ranked=["ck_1"],
                relevant=["ck_1"],
                metrics={
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "mrr@10": 1.0,
                    "ndcg@10": 1.0,
                    "context_precision": 1.0,
                },
                generation={
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": 1.0,
                },
                item_count=1,
                latency_ms=120,
            ),
            CaseResult(
                case_id="q2",
                task="text_to_image",
                query="找图",
                ranked=["img_9"],
                relevant=["img_1"],
                metrics={
                    "recall@5": 0.0,
                    "recall@10": 0.0,
                    "mrr@10": 0.0,
                    "ndcg@10": 0.0,
                    "context_precision": 0.0,
                },
                latency_ms=340,
            ),
        ],
    )
    return EvalReport(
        dataset_path="data/eval/eval_set.jsonl",
        kb_id="kb_test",
        kb_name="测试库",
        top_k=10,
        recall_ks=[5, 10],
        mrr_k=10,
        ndcg_k=10,
        total_cases=3,
        labeled_cases=2,
        unlabeled_cases=1,
        groups=[group],
        generation_group="hybrid_rrf_rerank",
        generation_skipped_reason=generation_skipped,
        ragas_error=ragas_error,
        warnings=["1 条用例未标注"],
        duration_ms=1234,
    )


def test_format_metric_uses_dash_for_none() -> None:
    assert format_metric(None) == "—"
    assert format_metric(0.5) == "0.5000"


def test_markdown_contains_tables_and_skips_unavailable() -> None:
    text = render_markdown(_report())
    assert "# mmkb 检索与生成评估报告" in text
    assert "## 2. 文本检索" in text
    assert "## 3. 跨模态检索" in text
    assert "hybrid_rrf_rerank" in text
    assert "1.0000" in text
    assert "## 5. 检索延迟" in text
    assert "| 文本 p50 | 文本 p95 |" in text
    assert "跳过原因：显式跳过" in text
    assert "reranker 只接受纯文本" in text
    assert "2 / 3 条计入" in text


def test_markdown_shows_dash_not_zero_for_unavailable_generation() -> None:
    # 生成了、但 judge 不可用时：指标必须显示 —，不能伪造成 0.0000
    text = render_markdown(_report(generation_skipped=""))
    assert "| Faithfulness | — |" in text
    assert "| Answer Relevancy | — |" in text
    assert "未评估不会被写成 0" in text


def test_report_to_dict_is_json_serialisable() -> None:
    payload = json.dumps(_report().to_dict(), ensure_ascii=False)
    decoded = json.loads(payload)
    assert decoded["groups"][0]["name"] == "hybrid_rrf_rerank"
    assert decoded["groups"][0]["metrics"]["text"]["recall@5"] == 1.0
    assert decoded["groups"][0]["metrics"]["text_to_image"]["recall@5"] == 0.0
    assert decoded["groups"][0]["latency"]["text"]["p50_ms"] == 120.0
    assert decoded["groups"][0]["latency"]["all"]["count"] == 2
    assert decoded["counts"] == {"total": 3, "labeled": 2, "unlabeled": 1}
    # ragas_samples 是运行期数据，不应进入报告
    assert "ragas_samples" not in decoded["groups"][0]


def test_write_outputs_creates_three_artifacts(tmp_path: Path) -> None:
    written = write_outputs(_report(), tmp_path, make_chart=False)
    names = {path.name for path in written}
    assert names == {"report.md", "report.json", "cases.jsonl"}
    lines = (tmp_path / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["group"] == "hybrid_rrf_rerank"
    assert first["case_id"] == "q1"
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))[
        "kb_id"
    ] == "kb_test"
