"""评测集读取与校验。

**不生成任何 ground truth。** 本模块能做的只有：按约定格式解析、检查字段、
把「未标注」的条目单独计数并跳过。空标注绝不被当成「没有相关结果」，
否则 Recall 会被一堆 0 分静默拉低，看起来像模型很差，实际是数据没标。

文件格式：JSONL，一行一个对象，空行与 ``#`` 开头的行会被忽略::

    {
      "id": "q001",
      "task": "text",                       // text | text_to_image | image_to_image
      "query": "系统如何融合三路召回？",
      "query_image": null,                  // task=image_to_image 时必填
      "kb_id": null,                        // 可选，填了就必须与 --kb 一致
      "labeled": false,                     // 人工标注完成后改成 true
      "ground_truth": {
        "item_ids": [],                     // 精确到分块（ck_...）或图片（img_...）
        "doc_ids": [],                      // 文档级标注：该文档的任一分块命中都算相关
        "image_ids": [],                    // 图片级标注，跨模态指标只认这个
        "answer": ""                        // 参考答案，可留空（生成指标会跳过）
      },
      "notes": "为什么这条题值得评测"
    }

判定规则：

* ``labeled=false`` → 永远跳过，并在报告里显示数量；
* ``labeled=true`` 但四类标注全空 → 解析期直接报错，避免「标了但什么都没标」；
* ``text`` / ``text_to_image`` 必须有 ``query``；
* ``image_to_image`` 必须有 ``query_image``，相对路径以数据集文件所在目录为基准，
  找不到时再尝试项目根目录。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config import PROJECT_ROOT

__all__ = [
    "DatasetError",
    "EvalCase",
    "EvalDataset",
    "GroundTruth",
    "load_dataset",
    "load_cases",
    "resolve_query_image",
]

TaskType = Literal["text", "text_to_image", "image_to_image"]


class DatasetError(ValueError):
    """评测集格式错误。消息里一定包含行号，便于人工修。"""


class GroundTruth(BaseModel):
    """人工标注。多个字段可以同时使用，按任务类型取不同的集合。"""

    model_config = ConfigDict(extra="forbid")

    item_ids: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)
    answer: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.item_ids or self.doc_ids or self.image_ids)


class EvalCase(BaseModel):
    # 拼错字段名（lableled / ground_truths）必须报错。Pydantic 默认忽略多余字段，
    # 那会让一道标注过的题静默变成「未标注」，评估结果少一条却看不出原因。
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    task: TaskType = "text"
    query: str = ""
    query_image: str | None = None
    kb_id: str | None = None
    labeled: bool = False
    ground_truth: GroundTruth = Field(default_factory=GroundTruth)
    notes: str = ""

    @model_validator(mode="after")
    def _validate_shape(self) -> EvalCase:
        if self.task == "image_to_image":
            if not (self.query_image or "").strip():
                raise ValueError("task=image_to_image 时必须提供 query_image")
        elif not self.query.strip():
            raise ValueError(f"task={self.task} 时 query 不能为空")

        if self.labeled and self.ground_truth.is_empty:
            raise ValueError(
                "labeled=true 但 ground_truth 中 item_ids / doc_ids / image_ids 全为空；"
                "请完成人工标注，或把 labeled 改为 false"
            )
        return self

    @property
    def has_labels(self) -> bool:
        """真正参与计分的条件：显式标注 + 至少一个相关 ID。"""
        return self.labeled and not self.ground_truth.is_empty

    @property
    def relevant_keys(self) -> set[str]:
        """检索结果与标注比对时使用的 key 集合。

        跨模态任务只认 ``image_ids``：文本分块即使命中了图片 caption，
        也不是「找到了那张图」。
        """
        gt = self.ground_truth
        if self.task in {"text_to_image", "image_to_image"}:
            return set(gt.image_ids)
        return set(gt.item_ids) | set(gt.doc_ids) | set(gt.image_ids)


@dataclass(slots=True)
class EvalDataset:
    path: Path
    cases: list[EvalCase]
    warnings: list[str] = field(default_factory=list)

    @property
    def labeled(self) -> list[EvalCase]:
        return [case for case in self.cases if case.has_labels]

    @property
    def unlabeled(self) -> list[EvalCase]:
        return [case for case in self.cases if not case.has_labels]

    def limit(self, max_questions: int) -> EvalDataset:
        """只取前 N 条**已标注**用例。``0`` 表示不限制。"""
        if max_questions <= 0:
            return self
        kept = self.labeled[:max_questions]
        return EvalDataset(path=self.path, cases=kept, warnings=list(self.warnings))


def load_cases(path: Path) -> list[EvalCase]:
    """解析 JSONL，保留文件顺序。任何一条坏数据都报行号，不静默跳过。"""
    cases: list[EvalCase] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_no} JSON 解析失败：{exc.msg}") from exc
            if not isinstance(payload, dict):
                raise DatasetError(f"{path}:{line_no} 每行必须是 JSON 对象")
            try:
                case = EvalCase.model_validate(payload)
            except ValidationError as exc:
                detail = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                    for err in exc.errors()
                )
                raise DatasetError(f"{path}:{line_no} 用例校验失败：{detail}") from exc
            if case.id in seen:
                raise DatasetError(f"{path}:{line_no} 用例 id 重复：{case.id!r}")
            seen.add(case.id)
            cases.append(case)

    if not cases:
        raise DatasetError(f"{path} 里没有任何用例")
    return cases


def load_dataset(path: Path) -> EvalDataset:
    """读取并统计。路径不存在时给出可直接照抄的下一步命令。"""
    if not path.is_file():
        raise DatasetError(
            f"评测集不存在：{path}\n"
            "先复制示例并人工标注（不要指望代码替你编 ground truth）：\n"
            "  mkdir -p data/eval\n"
            "  cp eval/samples/eval_set.example.jsonl data/eval/eval_set.jsonl\n"
            "然后把 labeled 改为 true 并填写 item_ids / doc_ids / image_ids。"
        )

    cases = load_cases(path)
    warnings: list[str] = []
    unlabeled = [case for case in cases if not case.has_labels]
    if unlabeled:
        warnings.append(
            f"{len(unlabeled)} 条用例未标注（labeled=false 或标注为空），已跳过计分："
            + ", ".join(case.id for case in unlabeled[:10])
            + (" ..." if len(unlabeled) > 10 else "")
        )

    for case in cases:
        if case.task in {"text_to_image", "image_to_image"} and not case.ground_truth.image_ids:
            warnings.append(
                f"{case.id}: task={case.task} 但没有 image_ids，跨模态 Recall 无法计算"
            )
        if case.task == "text" and not (
            case.ground_truth.item_ids or case.ground_truth.doc_ids
        ):
            warnings.append(
                f"{case.id}: task=text 但没有 item_ids/doc_ids，文本检索指标无法计算"
            )

    return EvalDataset(path=path, cases=cases, warnings=warnings)


def resolve_query_image(case: EvalCase, dataset_path: Path) -> Path | None:
    """把 ``query_image`` 解析成实际路径。

    先相对数据集目录找，再相对项目根目录找。两个都找不到返回 ``None``，
    由调用方记成「样本缺失」而不是抛异常 —— 一条坏数据不该中断整轮评测。
    """
    raw = (case.query_image or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for base in (dataset_path.parent, PROJECT_ROOT):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return None


def iter_task(cases: Iterable[EvalCase], task: TaskType) -> list[EvalCase]:
    return [case for case in cases if case.task == task]


def as_jsonl(cases: Iterable[EvalCase]) -> str:
    """把用例序列化回 JSONL，供「标注模板生成」使用。"""
    return "\n".join(
        json.dumps(case.model_dump(mode="json"), ensure_ascii=False) for case in cases
    )


def dump_raw(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
