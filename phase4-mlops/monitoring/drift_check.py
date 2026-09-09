"""数据/模型漂移检测（批处理）：参考数据 vs 当前线上数据。

这是**模型侧**监控，与 Prometheus/Grafana 的系统指标完全是两回事：

* 系统指标回答「服务是不是还活着、快不快、报不报错」——实时、秒级；
* 本脚本回答「喂进模型的数据还是不是模型见过的那种数据」——批处理、离线，
  跑一次看一批。

产出三类文件：

* ``drift_report.html``            Evidently 的完整可视化报告（分布对比图）
* ``drift.json``                   Evidently 原生 ``Report.run(...).dict()``，
                                   第五阶段的 ``alert_notifier.py`` 直接消费
* ``drift_summary.json``           精简结论：**哪个特征漂了、漂了多少**

判定与阈值（不使用「默认值但不解释」，全部显式固定并在下方说明）：

* **是否漂移**：数值特征用 **K-S 检验的 p 值**，``p < 0.05`` 判为漂移。
  Evidently 的默认行为会随样本量切换检验方法（≤1000 行用 K-S p 值，>1000 行
  改用 Wasserstein 距离），这里显式固定成 K-S，避免样本量一变结论口径就变。
* **漂移程度**：同时计算 **Wasserstein 距离**（按参考集标准差归一化），
  阈值 0.1 仅作「幅度大不大」的参考，不参与是否漂移的判定。
  p 值只说「差异是否显著」，样本量足够大时再小的差异也会显著，所以必须配着幅度看。
* **数据集整体漂移**：漂移特征占比 ≥ ``--drift-share``（默认 0.5，即 4 个特征里
  至少 2 个漂了）才判定「整体漂移」。单个特征漂移仍会在报告里逐条列出。
* **模型输出漂移**：``prediction`` 列（类别型）用 **Jensen-Shannon 距离**，
  阈值 0.1。距离型指标对「某个类别在一侧完全没出现」更稳健，也直接给出幅度。

用法::

    python monitoring/drift_check.py                                  # 用默认路径
    python monitoring/drift_check.py --current monitoring/current_drifted.csv
    python monitoring/drift_check.py --current ... --fail-on-drift    # 漂移时退出码 3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

PHASE4_ROOT = Path(__file__).resolve().parents[1]

from mlops_shared.errors import FeatureValidationError  # noqa: E402
from mlops_shared.features import (  # noqa: E402
    API_FIELD_TO_COLUMN,
    FEATURE_COLUMNS,
    build_feature_frame,
)

DEFAULT_REFERENCE = PHASE4_ROOT / "monitoring" / "reference_data.csv"
DEFAULT_CURRENT = PHASE4_ROOT / "monitoring" / "current_data.csv"
DEFAULT_REPORT_DIR = PHASE4_ROOT / "reports"

#: 判定是否漂移的检验（数值特征）与阈值：p 值低于它即判为漂移。
DEFAULT_NUM_METHOD = "ks"
DEFAULT_NUM_THRESHOLD = 0.05
#: 衡量漂移幅度的距离指标与「幅度较大」的参考线。
MAGNITUDE_METHOD = "wasserstein"
DEFAULT_MAGNITUDE_THRESHOLD = 0.1
#: 类别型（模型输出）漂移的检验与阈值。
DEFAULT_CAT_METHOD = "jensenshannon"
DEFAULT_CAT_THRESHOLD = 0.1
#: 数据集整体判定：漂移特征占比达到该比例才算「整体漂移」。
DEFAULT_DRIFT_SHARE = 0.5
#: 样本量低于此值时统计检验不可靠，只给提示不改变判定。
MIN_RELIABLE_ROWS = 30

PREDICTION_COLUMN = "prediction"


class DriftCheckError(RuntimeError):
    """漂移检测无法进行（文件缺失、列不匹配等）。"""


@dataclass
class FeatureFinding:
    """单个特征的漂移结论。字段名与第五阶段 alert_notifier 的解析格式保持一致。"""

    feature: str
    method: str
    score: float
    threshold: float
    drifted: bool
    magnitude: Optional[float] = None
    magnitude_method: Optional[str] = None
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    reference_std: Optional[float] = None
    current_std: Optional[float] = None
    mean_shift_pct: Optional[float] = None


@dataclass
class DriftSummary:
    """一次漂移检查的完整结论。"""

    generated_at: str
    reference_path: str
    current_path: str
    reference_rows: int
    current_rows: int
    thresholds: dict[str, Any]
    dataset_drift: dict[str, Any]
    drifted_features: list[dict[str, Any]] = field(default_factory=list)
    all_features: list[dict[str, Any]] = field(default_factory=list)
    prediction_drift: Optional[dict[str, Any]] = None
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 读取与校验
# --------------------------------------------------------------------------- #
def _read_csv(path: Path, role: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        hint = (
            "先运行 python monitoring/make_reference.py 生成参考数据。"
            if role == "参考"
            else "先运行 python monitoring/simulate_traffic.py 生成当前数据。"
        )
        raise DriftCheckError(f"{role}数据 '{path}' 不存在。{hint}") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DriftCheckError(f"{role}数据 '{path}' 不是合法 CSV：{exc}") from exc
    if frame.empty:
        raise DriftCheckError(f"{role}数据 '{path}' 没有数据行。")
    return frame


def extract_features(frame: pd.DataFrame, path: Path, role: str) -> pd.DataFrame:
    """按共享特征契约取出并规范化特征列（列名/列序/dtype 与训练时一致）。"""
    known = set(API_FIELD_TO_COLUMN) | set(FEATURE_COLUMNS)
    feature_part = frame[[column for column in frame.columns if column in known]]
    try:
        return build_feature_frame(feature_part)
    except FeatureValidationError as exc:
        raise DriftCheckError(
            f"{role}数据 '{path}' 不满足特征契约：{exc} "
            f"（需要这 {len(FEATURE_COLUMNS)} 列，API 名或 sklearn 名皆可：{list(API_FIELD_TO_COLUMN)}）"
        ) from exc


# --------------------------------------------------------------------------- #
# Evidently 报告
# --------------------------------------------------------------------------- #
def _build_reports(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    num_method: str,
    num_threshold: float,
    drift_share: float,
) -> Any:
    """特征漂移报告：K-S 判定 + Wasserstein 幅度，放在同一份报告里。"""
    from evidently import Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset

    metrics: list[Any] = [
        DataDriftPreset(
            num_method=num_method,
            num_threshold=num_threshold,
            drift_share=drift_share,
        )
    ]
    # 额外为每个特征算一次距离型指标，作为「漂移程度」
    metrics += [ValueDrift(column=column, method=MAGNITUDE_METHOD) for column in FEATURE_COLUMNS]
    return Report(metrics=metrics).run(current_data=current, reference_data=reference)


def _prediction_report(
    reference: pd.DataFrame, current: pd.DataFrame, cat_method: str, cat_threshold: float
) -> Any:
    """模型输出漂移报告：只看 prediction 一列（类别型）。"""
    from evidently import Report
    from evidently.metrics import ValueDrift

    return Report(
        metrics=[ValueDrift(column=PREDICTION_COLUMN, method=cat_method, threshold=cat_threshold)]
    ).run(current_data=current, reference_data=reference)


def _value_drift_entries(snapshot: Any) -> list[tuple[str, str, float, float]]:
    """从 snapshot 里抽出所有 ValueDrift：(column, method, threshold, value)。"""
    entries: list[tuple[str, str, float, float]] = []
    for metric in snapshot.dict().get("metrics", []):
        config = metric.get("config") or {}
        if not str(config.get("type", "")).endswith("ValueDrift"):
            continue
        value = metric.get("value")
        if config.get("column") is None or not isinstance(value, (int, float)):
            continue
        entries.append(
            (
                str(config["column"]),
                str(config.get("method", "unknown")),
                float(config.get("threshold", 0.0)),
                float(value),
            )
        )
    return entries


def analyse(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    snapshot: Any,
    num_threshold: float,
    magnitude_threshold: float,
) -> list[FeatureFinding]:
    """把 Evidently 的原始指标整理成「每个特征漂没漂、漂了多少」。"""
    decisions: dict[str, tuple[str, float, float]] = {}
    magnitudes: dict[str, float] = {}

    for column, method, threshold, value in _value_drift_entries(snapshot):
        if MAGNITUDE_METHOD in method.lower():
            magnitudes[column] = value
        else:
            decisions[column] = (method, threshold or num_threshold, value)

    findings: list[FeatureFinding] = []
    for column in FEATURE_COLUMNS:
        if column not in decisions:
            continue
        method, threshold, score = decisions[column]
        ref_mean = float(reference[column].mean())
        cur_mean = float(current[column].mean())
        findings.append(
            FeatureFinding(
                feature=column,
                method=method,
                score=score,
                threshold=threshold,
                # p 值型：低于阈值算漂移
                drifted=score < threshold,
                magnitude=magnitudes.get(column),
                magnitude_method=MAGNITUDE_METHOD if column in magnitudes else None,
                reference_mean=round(ref_mean, 4),
                current_mean=round(cur_mean, 4),
                reference_std=round(float(reference[column].std()), 4),
                current_std=round(float(current[column].std()), 4),
                mean_shift_pct=(
                    round((cur_mean - ref_mean) / ref_mean * 100, 2) if ref_mean else None
                ),
            )
        )
    findings.sort(key=lambda item: (not item.drifted, -(item.magnitude or 0.0)))
    return findings


def analyse_prediction(
    reference: pd.Series, current: pd.Series, snapshot: Any, threshold: float
) -> dict[str, Any]:
    """整理模型输出漂移，并附上两侧的类别占比对比。"""
    entries = _value_drift_entries(snapshot)
    method, thr, score = (entries[0][1], entries[0][2] or threshold, entries[0][3]) if entries else (
        "unknown",
        threshold,
        float("nan"),
    )
    ref_share = (reference.value_counts(normalize=True) * 100).round(2).to_dict()
    cur_share = (current.value_counts(normalize=True) * 100).round(2).to_dict()
    return {
        "column": PREDICTION_COLUMN,
        "method": method,
        "score": score,
        "threshold": thr,
        # 距离型：达到阈值算漂移
        "drifted": bool(score >= thr),
        "reference_share_pct": ref_share,
        "current_share_pct": cur_share,
    }


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def print_report(summary: DriftSummary) -> None:
    """把结论打成一张人能直接读的表。"""
    print("=" * 100)
    print("数据漂移报告（模型侧监控 · 批处理）")
    print("=" * 100)
    print(f"参考数据：{summary.reference_path}（{summary.reference_rows} 行）")
    print(f"当前数据：{summary.current_path}（{summary.current_rows} 行）")
    thresholds = summary.thresholds
    print(
        f"判定：{thresholds['decision_method']} p<{thresholds['decision_threshold']} "
        f"｜ 幅度：{thresholds['magnitude_method']}（参考线 {thresholds['magnitude_threshold']}）"
        f"｜ 数据集阈值：漂移特征占比 ≥ {thresholds['drift_share']:.0%}"
    )
    for warning in summary.warnings:
        print(f"⚠️  {warning}")

    print("\n【输入特征漂移】")
    header = (
        f"{'特征':<20}{'检验':<12}{'p 值':>10}{'阈值':>8}{'漂移':>6}"
        f"{'幅度(W)':>10}{'参考均值':>10}{'当前均值':>10}{'均值变化':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in summary.all_features:
        magnitude = "-" if item["magnitude"] is None else f"{item['magnitude']:.3f}"
        shift = "-" if item["mean_shift_pct"] is None else f"{item['mean_shift_pct']:+.1f}%"
        print(
            f"{item['feature']:<20}{item['method']:<12}{item['score']:>10.4g}"
            f"{item['threshold']:>8.3g}{'是' if item['drifted'] else '否':>6}"
            f"{magnitude:>10}{item['reference_mean']:>10.3f}{item['current_mean']:>10.3f}"
            f"{shift:>10}"
        )

    dataset = summary.dataset_drift
    verdict = "整体漂移" if dataset["drifted"] else "未达到整体漂移阈值"
    print(
        f"\n结论：{dataset['drifted_features']}/{dataset['total_features']} 个特征漂移"
        f"（占比 {dataset['drift_share']:.0%}）→ {verdict}"
    )
    if summary.drifted_features:
        worst = summary.drifted_features[0]
        print(
            f"最严重：{worst['feature']}（p={worst['score']:.3g}，"
            f"Wasserstein={worst['magnitude']:.3f}，"
            f"均值 {worst['reference_mean']} → {worst['current_mean']}）"
        )

    if summary.prediction_drift:
        pred = summary.prediction_drift
        print("\n【模型输出漂移】（预测类别分布，与输入特征漂移分开看）")
        print(
            f"  {pred['method']} 距离 = {pred['score']:.4f}（阈值 {pred['threshold']}）"
            f" → {'漂移' if pred['drifted'] else '未漂移'}"
        )
        print(f"  参考占比：{pred['reference_share_pct']}")
        print(f"  当前占比：{pred['current_share_pct']}")
    print("=" * 100)


def write_outputs(
    summary: DriftSummary,
    snapshot: Any,
    prediction_snapshot: Any,
    html_path: Path,
    json_path: Path,
    summary_path: Path,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    json_path.write_text(
        json.dumps(snapshot.dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nHTML 报告      : {html_path}")
    print(f"Evidently JSON : {json_path}  （第五阶段 alert_notifier.py 可直接消费）")
    print(f"结论摘要 JSON  : {summary_path}")
    if prediction_snapshot is not None:
        prediction_html = html_path.with_name(f"{html_path.stem}_prediction.html")
        prediction_snapshot.save_html(str(prediction_html))
        print(f"输出漂移报告   : {prediction_html}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    reference_raw = _read_csv(args.reference, "参考")
    current_raw = _read_csv(args.current, "当前")
    reference = extract_features(reference_raw, args.reference, "参考")
    current = extract_features(current_raw, args.current, "当前")

    warnings: list[str] = []
    if len(current) < MIN_RELIABLE_ROWS:
        warnings.append(
            f"当前数据只有 {len(current)} 行（<{MIN_RELIABLE_ROWS}），统计检验结论不稳定，"
            "建议积累更多线上样本再下判断。"
        )
    if len(current) > 5000:
        warnings.append(
            "样本量很大时 K-S 的 p 值几乎必然显著（微小差异也会被判为漂移），"
            "请以 Wasserstein 幅度为主要依据。"
        )

    snapshot = _build_reports(
        reference, current, args.num_method, args.num_threshold, args.drift_share
    )
    findings = analyse(reference, current, snapshot, args.num_threshold, args.magnitude_threshold)
    drifted = [item for item in findings if item.drifted]
    share = len(drifted) / len(findings) if findings else 0.0

    prediction_snapshot = None
    prediction_drift = None
    if (
        not args.skip_prediction_drift
        and PREDICTION_COLUMN in reference_raw.columns
        and PREDICTION_COLUMN in current_raw.columns
    ):
        prediction_snapshot = _prediction_report(
            reference_raw[[PREDICTION_COLUMN]],
            current_raw[[PREDICTION_COLUMN]],
            args.cat_method,
            args.cat_threshold,
        )
        prediction_drift = analyse_prediction(
            reference_raw[PREDICTION_COLUMN],
            current_raw[PREDICTION_COLUMN],
            prediction_snapshot,
            args.cat_threshold,
        )
    elif not args.skip_prediction_drift:
        warnings.append(
            f"两份数据里没有同时出现 '{PREDICTION_COLUMN}' 列，跳过模型输出漂移分析"
            "（参考数据可用 make_reference.py --model-uri 生成）。"
        )

    summary = DriftSummary(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        reference_path=str(args.reference),
        current_path=str(args.current),
        reference_rows=len(reference),
        current_rows=len(current),
        thresholds={
            "decision_method": args.num_method,
            "decision_threshold": args.num_threshold,
            "magnitude_method": MAGNITUDE_METHOD,
            "magnitude_threshold": args.magnitude_threshold,
            "categorical_method": args.cat_method,
            "categorical_threshold": args.cat_threshold,
            "drift_share": args.drift_share,
        },
        dataset_drift={
            "drifted": share >= args.drift_share,
            "drifted_features": len(drifted),
            "total_features": len(findings),
            "drift_share": round(share, 4),
            "threshold": args.drift_share,
        },
        drifted_features=[asdict(item) for item in drifted],
        all_features=[asdict(item) for item in findings],
        prediction_drift=prediction_drift,
        warnings=warnings,
    )

    print_report(summary)
    write_outputs(
        summary,
        snapshot,
        prediction_snapshot,
        args.output_html,
        args.output_json,
        args.summary_json,
    )

    if args.fail_on_drift and summary.dataset_drift["drifted"]:
        print("\n--fail-on-drift 生效：检测到数据集整体漂移，退出码 3。")
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drift_check.py",
        description="对比参考数据与当前数据，用 Evidently 生成特征级漂移报告。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="参考数据 CSV。")
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT, help="当前数据 CSV。")
    parser.add_argument(
        "--output-html", type=Path, default=DEFAULT_REPORT_DIR / "drift_report.html",
        help="Evidently HTML 报告输出路径。",
    )
    parser.add_argument(
        "--output-json", type=Path, default=DEFAULT_REPORT_DIR / "drift.json",
        help="Evidently 原生 JSON（供第五阶段告警脚本消费）。",
    )
    parser.add_argument(
        "--summary-json", type=Path, default=DEFAULT_REPORT_DIR / "drift_summary.json",
        help="精简结论 JSON。",
    )
    parser.add_argument(
        "--num-method", default=DEFAULT_NUM_METHOD,
        help="数值特征的漂移检验（ks/wasserstein/psi/kl_div/anderson 等）。",
    )
    parser.add_argument(
        "--num-threshold", type=float, default=DEFAULT_NUM_THRESHOLD,
        help="判定阈值：K-S 等 p 值型检验中，p 小于该值判为漂移。",
    )
    parser.add_argument(
        "--magnitude-threshold", type=float, default=DEFAULT_MAGNITUDE_THRESHOLD,
        help="Wasserstein 距离的「幅度较大」参考线（不参与是否漂移的判定）。",
    )
    parser.add_argument(
        "--cat-method", default=DEFAULT_CAT_METHOD, help="prediction 列（类别型）的漂移检验。"
    )
    parser.add_argument(
        "--cat-threshold", type=float, default=DEFAULT_CAT_THRESHOLD, help="类别型漂移阈值。"
    )
    parser.add_argument(
        "--drift-share", type=float, default=DEFAULT_DRIFT_SHARE,
        help="数据集整体判定：漂移特征占比达到该值即判为整体漂移。",
    )
    parser.add_argument(
        "--skip-prediction-drift", action="store_true", help="不分析 prediction 列。"
    )
    parser.add_argument(
        "--fail-on-drift", action="store_true",
        help="检测到数据集整体漂移时以退出码 3 结束（便于接入定时任务/CI）。",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except DriftCheckError as exc:
        print(f"drift_check.py: error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(
            f"drift_check.py: error: 缺少依赖（{exc}）。"
            "请执行 pip install -r requirements-monitoring.txt",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("drift_check.py: 已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
