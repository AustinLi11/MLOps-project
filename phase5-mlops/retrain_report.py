"""新旧模型对比报告：只提供决策依据，**不做任何决策**。

安全边界（本阶段的核心设计）
---------------------------
本模块是**只读**的：它从 MLflow 读取两个模型、在同一个测试集上评估、输出一份
带建议性结论的 markdown 报告。它**不会**：

* 调用任何模型注册表的写接口（alias / stage / tag 一律不动）；
* 自动替换线上模型或触发部署。

「把哪个模型标记为 Production」是人工动作：报告只给出结论、核对清单，以及
「去 README 哪一节拿命令」的指引，命令本身由人复制、确认后手动执行。
本文件里连注册表写接口的名字都不出现，可用 README「验收 3」里的 grep 命令自证。

用法::

    # 与上一次 run 自动对比
    python retrain_report.py --candidate-run-id <NEW_RUN_ID> --output report.md

    # 与指定基线（run 或注册表别名）对比
    python retrain_report.py --candidate-run-id <NEW> --baseline <OLD_RUN_ID>
    python retrain_report.py --candidate-run-id <NEW> --baseline "models:/iris-classifier@champion"

    # 用独立留出集评估（最严谨）
    python retrain_report.py --candidate-run-id <NEW> --test-csv holdout.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PHASE5_ROOT: Path = Path(__file__).resolve().parent
PHASE1_ROOT: Path = PHASE5_ROOT.parent / "phase1-mlops"
if str(PHASE1_ROOT) not in sys.path:
    # 复用第一阶段的数据切分，保证「同一测试集」的定义只有一份实现
    sys.path.insert(0, str(PHASE1_ROOT))

from mlops_shared.features import FEATURE_COLUMNS, TARGET_NAMES, build_feature_frame  # noqa: E402
from mlops_shared.tracking import MODEL_ARTIFACT_NAME, normalize_model_uri  # noqa: E402

#: 报告里默认使用的评估划分（与第一阶段默认值一致）
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42

#: f1_macro 提升达到该值才建议考虑替换，低于其绝对值视为噪音
DEFAULT_MIN_IMPROVEMENT: float = 0.01

METRIC_LABELS: dict[str, str] = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
}


class ReportError(RuntimeError):
    """报告生成过程中的可预期失败。"""


@dataclass(frozen=True)
class ModelUnderTest:
    """参与对比的一个模型及其可追溯信息。"""

    role: str  # "candidate" / "baseline"
    reference: str  # 用户给的原始引用（run_id 或 model URI）
    model_uri: str
    run_id: Optional[str]
    params: Mapping[str, str] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    logged_metrics: Mapping[str, float] = field(default_factory=dict)
    model: Any = None

    @property
    def git_commit(self) -> str:
        return self.tags.get("git_commit", "unknown")

    def param(self, key: str) -> str:
        return self.params.get(key, "?")


@dataclass(frozen=True)
class Comparison:
    """两个模型在同一测试集上的评估结果。"""

    candidate: ModelUnderTest
    baseline: Optional[ModelUnderTest]
    candidate_metrics: Mapping[str, float]
    baseline_metrics: Mapping[str, float]
    candidate_per_class_f1: Mapping[str, float]
    baseline_per_class_f1: Mapping[str, float]
    test_set_description: str
    test_set_size: int
    min_improvement: float
    warnings: Sequence[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def delta(self, metric: str) -> Optional[float]:
        if not self.baseline_metrics:
            return None
        return self.candidate_metrics[metric] - self.baseline_metrics[metric]

    @property
    def verdict(self) -> tuple[str, str]:
        """(结论标签, 建议文本)。只是建议，执行与否由人决定。"""
        if self.baseline is None:
            return (
                "无基线可比",
                "仓库里没有可用于对比的旧模型（这看起来是第一个模型）。"
                "建议人工检查指标绝对值是否达到上线标准后，再手动标记。",
            )

        f1_delta = self.delta("f1_macro") or 0.0
        acc_delta = self.delta("accuracy") or 0.0

        if f1_delta >= self.min_improvement and acc_delta >= -self.min_improvement:
            return (
                "建议人工审核后替换",
                f"新模型 f1_macro 提升 {f1_delta:+.2%}、accuracy {acc_delta:+.2%}，"
                f"超过约定的最小提升阈值 {self.min_improvement:.2%}。"
                "**建议人工审核后手动**把候选模型标记为 Production（命令见下节）。"
                "审核要点：确认本次重训练所用的数据不是异常数据（漂移告警可能来自上游故障）。",
            )
        if f1_delta <= -self.min_improvement:
            return (
                "建议不要上线",
                f"新模型 f1_macro 下降 {f1_delta:+.2%}，明显劣于现网模型。"
                "**建议不要上线**，优先排查训练数据是否因上游埋点/采样异常而失真。",
            )
        return (
            "建议暂不替换",
            f"新旧模型差异在噪音范围内（f1_macro {f1_delta:+.2%}，阈值 "
            f"±{self.min_improvement:.2%}）。**建议保持现网模型不变**，"
            "把本次 run 留作记录；若确有业务理由替换，需人工说明依据。",
        )


# --------------------------------------------------------------------------- #
# MLflow 读取（只读）
# --------------------------------------------------------------------------- #
def _client(tracking_uri: Optional[str]):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def load_model_under_test(
    reference: str, role: str, tracking_uri: Optional[str], artifact_name: str
) -> ModelUnderTest:
    """加载一个待比较的模型，并带出它的 params / tags / 已记录指标。

    Raises:
        ReportError: 引用无法解析或模型加载失败。
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.exceptions import MlflowException
    from mlops_shared.errors import ModelLoadError
    from mlops_shared.tracking import run_id_from_uri

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        model_uri = normalize_model_uri(reference, artifact_name=artifact_name)
    except ModelLoadError as exc:
        raise ReportError(f"{role} 引用无法解析：{exc}") from exc

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:  # noqa: BLE001 - mlflow 抛的类型不稳定
        raise ReportError(
            f"{role} 模型加载失败（{model_uri}）：{type(exc).__name__}: {exc}。"
            "请确认 run_id / 别名以及 MLFLOW_TRACKING_URI 是否正确。"
        ) from exc

    run_id = run_id_from_uri(model_uri)
    params: dict[str, str] = {}
    tags: dict[str, str] = {}
    metrics: dict[str, float] = {}
    if run_id:
        try:
            run = _client(tracking_uri).get_run(run_id)
            params = dict(run.data.params)
            tags = dict(run.data.tags)
            metrics = dict(run.data.metrics)
        except MlflowException as exc:
            # 元数据只是加分项，缺了不影响对比
            print(f"[warn] 无法读取 run {run_id} 的元数据：{exc}", file=sys.stderr)

    return ModelUnderTest(
        role=role,
        reference=reference,
        model_uri=model_uri,
        run_id=run_id,
        params=params,
        tags=tags,
        logged_metrics=metrics,
        model=model,
    )


def find_previous_run(
    candidate_run_id: str, tracking_uri: Optional[str], experiment_name: Optional[str]
) -> Optional[str]:
    """在同一 experiment 里找候选之前最近一次已完成的 run，作为默认基线。"""
    from mlflow.exceptions import MlflowException

    client = _client(tracking_uri)
    try:
        candidate = client.get_run(candidate_run_id)
        experiment_id = candidate.info.experiment_id
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=25,
        )
    except MlflowException as exc:
        print(f"[warn] 自动寻找基线失败：{exc}", file=sys.stderr)
        return None

    for run in runs:
        if run.info.run_id != candidate_run_id:
            return run.info.run_id
    return None


# --------------------------------------------------------------------------- #
# 测试集与评估
# --------------------------------------------------------------------------- #
def build_test_set(
    test_csv: Optional[Path], test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.Series, str]:
    """构造评估用测试集。

    优先使用 ``--test-csv``（独立留出集，最严谨）；否则复用第一阶段
    ``train.load_dataset`` 的确定性分层切分，保证两个模型评估在完全相同的数据上。

    Raises:
        ReportError: CSV 缺少标签列，或数据集无法构造。
    """
    if test_csv is not None:
        try:
            frame = pd.read_csv(test_csv)
        except FileNotFoundError as exc:
            raise ReportError(f"找不到测试集 {test_csv}。") from exc
        except Exception as exc:  # noqa: BLE001
            raise ReportError(f"读取测试集 {test_csv} 失败：{type(exc).__name__}: {exc}") from exc

        label_column = next((c for c in ("target", "label", "y") if c in frame.columns), None)
        if label_column is None:
            raise ReportError(
                f"测试集 {test_csv} 缺少标签列（需要 target / label / y 之一），无法评估。"
            )
        try:
            features = build_feature_frame(frame.drop(columns=[label_column]))
        except Exception as exc:  # noqa: BLE001
            raise ReportError(f"测试集 {test_csv} 不满足共享特征契约：{exc}") from exc
        return features, frame[label_column].astype(int), f"独立留出集 {test_csv}（{len(frame)} 行）"

    try:
        import train  # 第一阶段训练脚本
    except ImportError as exc:  # pragma: no cover
        raise ReportError(
            f"无法导入第一阶段 train.py（期望位于 {PHASE1_ROOT}）：{exc}"
        ) from exc

    try:
        dataset = train.load_dataset(test_size=test_size, random_state=random_state)
    except train.DataLoadError as exc:
        raise ReportError(f"构造测试集失败：{exc}") from exc

    description = (
        f"iris 分层切分（test_size={test_size}, random_state={random_state}，"
        f"{len(dataset.X_test)} 行），与第一阶段 train.py 使用同一份切分实现"
    )
    return dataset.X_test, dataset.y_test, description


def evaluate(model: Any, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """在给定测试集上评估，返回四个宏观指标。"""
    predictions = model.predict(build_feature_frame(features))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
    }


def per_class_f1(model: Any, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """逐类 f1，用于发现「整体持平但某一类明显退化」的情况。"""
    predictions = model.predict(build_feature_frame(features))
    scores = f1_score(labels, predictions, average=None, labels=list(range(len(TARGET_NAMES))))
    return {name: float(score) for name, score in zip(TARGET_NAMES, scores)}


def collect_warnings(
    candidate: ModelUnderTest,
    baseline: Optional[ModelUnderTest],
    test_csv: Optional[Path],
    random_state: int,
) -> list[str]:
    """提示可能影响结论可信度的因素，避免报告被误读。"""
    warnings: list[str] = []
    if test_csv is None:
        seeds = {
            model.param("random_state")
            for model in (candidate, baseline)
            if model is not None and model.param("random_state") != "?"
        }
        if seeds - {str(random_state)}:
            warnings.append(
                f"评估划分用的 random_state={random_state}，但参与对比的模型训练时使用了 "
                f"{sorted(seeds)}。训练种子不同意味着某个模型可能在训练时见过这里的部分样本，"
                "指标会偏乐观。要得到严格结论，请用 `--test-csv` 提供独立留出集。"
            )
    if baseline is not None and candidate.git_commit != baseline.git_commit:
        warnings.append(
            f"两个模型来自不同代码版本（候选 {candidate.git_commit[:8]}，"
            f"基线 {baseline.git_commit[:8]}），指标差异可能来自代码变更而非数据。"
        )
    if baseline is None:
        warnings.append("未找到基线模型，本报告只能给出候选模型的绝对指标。")
    return warnings


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
def _fmt(value: Optional[float], *, pct: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value:.2%}" if pct else f"{value:.6f}"


def render_markdown(comparison: Comparison) -> str:
    """渲染对比报告（markdown）。报告只陈述事实与建议，不含任何执行动作。"""
    candidate, baseline = comparison.candidate, comparison.baseline
    label, advice = comparison.verdict

    lines: list[str] = [
        "# 重训练对比报告",
        "",
        f"- 生成时间：{comparison.generated_at.astimezone(timezone.utc):%Y-%m-%d %H:%M:%S UTC}",
        f"- 测试集：{comparison.test_set_description}",
        f"- 最小提升阈值：f1_macro ±{comparison.min_improvement:.2%}",
        "",
        "> 本报告**只提供决策依据**。是否替换线上模型由人工判断并手动执行，",
        "> 报告与生成它的脚本都不会修改任何模型别名 / 阶段 / 标签。",
        "",
        "## 1. 参与对比的模型",
        "",
        "| 角色 | 引用 | run_id | git_commit | n_estimators | max_depth | random_state |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for model in (candidate, baseline):
        if model is None:
            lines.append("| baseline | （未找到可比基线） | — | — | — | — | — |")
            continue
        lines.append(
            f"| {model.role} | `{model.reference}` | `{model.run_id or '—'}` | "
            f"`{model.git_commit[:8]}` | {model.param('n_estimators')} | "
            f"{model.param('max_depth')} | {model.param('random_state')} |"
        )

    lines += [
        "",
        "## 2. 同一测试集上的指标对比",
        "",
        f"测试集样本数：**{comparison.test_set_size}**（两个模型评估的是完全相同的数据）",
        "",
        "| 指标 | 基线（现网） | 候选（新训练） | 变化 |",
        "| --- | --- | --- | --- |",
    ]
    for metric in METRIC_LABELS:
        delta = comparison.delta(metric)
        arrow = "" if delta is None else ("🔺" if delta > 0 else ("🔻" if delta < 0 else "➖"))
        lines.append(
            f"| {METRIC_LABELS[metric]} | {_fmt(comparison.baseline_metrics.get(metric))} | "
            f"{_fmt(comparison.candidate_metrics.get(metric))} | "
            f"{arrow} {_fmt(delta) if delta is not None else '—'} |"
        )

    lines += [
        "",
        "### 逐类 f1（防止「整体持平、某一类塌陷」被平均值掩盖）",
        "",
        "| 类别 | 基线 | 候选 | 变化 |",
        "| --- | --- | --- | --- |",
    ]
    for name in TARGET_NAMES:
        base = comparison.baseline_per_class_f1.get(name)
        cand = comparison.candidate_per_class_f1.get(name)
        delta = None if base is None or cand is None else cand - base
        lines.append(f"| {name} | {_fmt(base)} | {_fmt(cand)} | {_fmt(delta) if delta is not None else '—'} |")

    if candidate.logged_metrics or (baseline and baseline.logged_metrics):
        lines += [
            "",
            "### 训练时记录的指标（来自各自的 MLflow run，仅作参考）",
            "",
            "| 指标 | 基线 run | 候选 run |",
            "| --- | --- | --- |",
        ]
        keys = sorted(
            set(candidate.logged_metrics) | set(baseline.logged_metrics if baseline else {})
        )
        for key in keys:
            base_value = (baseline.logged_metrics.get(key) if baseline else None)
            lines.append(
                f"| {key} | {_fmt(base_value, pct=False)} | "
                f"{_fmt(candidate.logged_metrics.get(key), pct=False)} |"
            )

    if comparison.warnings:
        lines += ["", "## 3. 需要注意的前提", ""]
        lines += [f"- ⚠️ {item}" for item in comparison.warnings]

    lines += [
        "",
        f"## 4. 建议性结论：**{label}**",
        "",
        advice,
        "",
        "### 人工确认清单（逐项确认后再决定）",
        "",
        "- [ ] 本次重训练的数据窗口经过检查，漂移是真实分布变化，不是上游埋点/采样故障",
        "- [ ] 指标提升不是来自测试集泄漏（训练/评估划分见上文「需要注意的前提」）",
        "- [ ] 逐类指标没有出现某一类明显退化",
        "- [ ] 已知本次变更的代码版本（git_commit）与预期一致",
        "",
        "### 若决定替换（以下动作全部由人工执行，本脚本不会执行任何一步）",
        "",
        "1. 按 `phase5-mlops/README.md` →「人工把候选模型提升为 Production」一节，"
        "复制那段命令并把版本号替换为本次候选版本；",
        "2. 按 `phase2-mlops/README.md` →「人工确认部署」一节重启服务，把 `MODEL_URI` "
        "指向新模型：",
        "",
        "```text",
        f"MODEL_URI={candidate.model_uri}",
        "```",
        "",
        "回滚方式：把别名指回旧版本，或把服务的 `MODEL_URI` 改回基线模型后重启——",
        "镜像不需要重新构建，模型与镜像是解耦的。",
    ]
    return "\n".join(lines) + "\n"


def to_json(comparison: Comparison) -> dict[str, Any]:
    """机器可读版本，便于流水线把结论贴到 PR / summary。"""
    label, advice = comparison.verdict
    return {
        "generated_at": comparison.generated_at.astimezone(timezone.utc).isoformat(),
        "test_set": {
            "description": comparison.test_set_description,
            "size": comparison.test_set_size,
        },
        "candidate": {
            "reference": comparison.candidate.reference,
            "model_uri": comparison.candidate.model_uri,
            "run_id": comparison.candidate.run_id,
            "git_commit": comparison.candidate.git_commit,
            "metrics": dict(comparison.candidate_metrics),
            "per_class_f1": dict(comparison.candidate_per_class_f1),
        },
        "baseline": None
        if comparison.baseline is None
        else {
            "reference": comparison.baseline.reference,
            "model_uri": comparison.baseline.model_uri,
            "run_id": comparison.baseline.run_id,
            "git_commit": comparison.baseline.git_commit,
            "metrics": dict(comparison.baseline_metrics),
            "per_class_f1": dict(comparison.baseline_per_class_f1),
        },
        "deltas": {metric: comparison.delta(metric) for metric in METRIC_LABELS},
        "warnings": list(comparison.warnings),
        "recommendation": {"label": label, "advice": advice, "auto_applied": False},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrain_report.py",
        description="对比新旧模型在同一测试集上的表现，生成带建议的报告（只读，不改注册表）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-run-id", required=True, help="新训练出的 run_id（或模型 URI）")
    parser.add_argument(
        "--baseline",
        default=None,
        help="基线：run_id 或模型 URI（如 models:/iris-classifier@champion）。省略则自动取上一次 run",
    )
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI（默认取环境变量）")
    parser.add_argument(
        "--experiment-name", default=None, help="自动寻找基线时限定的 experiment 名"
    )
    parser.add_argument(
        "--artifact-name", default=MODEL_ARTIFACT_NAME, help="模型 artifact 名（默认 model）"
    )
    parser.add_argument("--test-csv", type=Path, default=None, help="独立留出集 CSV（含标签列）")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="无 CSV 时的切分比例")
    parser.add_argument(
        "--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="无 CSV 时的切分随机种子"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=DEFAULT_MIN_IMPROVEMENT,
        help="f1_macro 提升达到该值才建议考虑替换",
    )
    parser.add_argument("--output", type=Path, default=None, help="报告输出路径（markdown）")
    parser.add_argument("--json-output", type=Path, default=None, help="附带输出机器可读 JSON")
    return parser


def build_comparison(args: argparse.Namespace) -> Comparison:
    """执行完整对比流程（加载模型 → 构造测试集 → 评估 → 汇总）。"""
    candidate = load_model_under_test(
        args.candidate_run_id, "candidate", args.tracking_uri, args.artifact_name
    )

    baseline_reference = args.baseline
    if baseline_reference is None and candidate.run_id:
        found = find_previous_run(candidate.run_id, args.tracking_uri, args.experiment_name)
        if found:
            baseline_reference = found
            print(f"[info] 自动选定基线 run：{found}", file=sys.stderr)

    baseline = (
        load_model_under_test(baseline_reference, "baseline", args.tracking_uri, args.artifact_name)
        if baseline_reference
        else None
    )

    features, labels, description = build_test_set(args.test_csv, args.test_size, args.random_state)

    return Comparison(
        candidate=candidate,
        baseline=baseline,
        candidate_metrics=evaluate(candidate.model, features, labels),
        baseline_metrics=evaluate(baseline.model, features, labels) if baseline else {},
        candidate_per_class_f1=per_class_f1(candidate.model, features, labels),
        baseline_per_class_f1=per_class_f1(baseline.model, features, labels) if baseline else {},
        test_set_description=description,
        test_set_size=len(features),
        min_improvement=args.min_improvement,
        warnings=collect_warnings(candidate, baseline, args.test_csv, args.random_state),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。返回退出码，不抛裸栈；无论结论如何都不修改任何模型状态。"""
    args = build_parser().parse_args(argv)
    try:
        comparison = build_comparison(args)
        markdown = render_markdown(comparison)

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(markdown, encoding="utf-8")
            print(f"[info] 报告已写入 {args.output}", file=sys.stderr)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(to_json(comparison), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[info] JSON 已写入 {args.json_output}", file=sys.stderr)

        print(markdown)
    except ReportError as exc:
        print(f"retrain_report.py: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("retrain_report.py: 被用户中断", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
