"""给服务打流量，并记录这批「线上数据」用于漂移检测。

一个脚本同时服务两个验收目标：

* **系统指标**：持续发请求，Grafana 的 QPS / 延迟 / 错误率面板会动起来
  （``--error-rate`` 还能人为制造一批 422，用来演示「错误率涨了但没有漂移」）。
* **模型指标 / 漂移**：把发出去的特征行写成 CSV，交给 ``drift_check.py``。
  ``--mode drift`` 会人为给指定特征加一个偏移量，用来验证 Evidently 检得出来。

用法::

    # 正常流量（无漂移）：200 个请求，每个 5 行
    python monitoring/simulate_traffic.py --requests 200 --batch-size 5 \
        --out monitoring/current_data.csv

    # 人为制造特征偏移：花瓣长度 +2.0cm、花瓣宽度 +0.8cm
    python monitoring/simulate_traffic.py --mode drift --requests 200 \
        --out monitoring/current_drifted.csv

    # 只生成数据不发请求（服务没起来时）
    python monitoring/simulate_traffic.py --no-send --out /tmp/current.csv
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
from sklearn.datasets import load_iris

PHASE4_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE4_ROOT.parent

from mlops_shared.features import API_FIELD_TO_COLUMN, COLUMN_TO_API_FIELD  # noqa: E402

API_FIELDS: tuple[str, ...] = tuple(API_FIELD_TO_COLUMN)

#: --mode drift 时默认偏移的特征与幅度（单位 cm）。
#: 选花瓣特征是因为它们对鸢尾花分类最有区分度，偏移后预测分布也会跟着变，
#: 正好演示「特征漂移 -> 模型输出漂移」的传导。
DEFAULT_DRIFT: dict[str, float] = {"petal_length": 2.0, "petal_width": 0.8}

#: 正常模式下加的一点点观测噪声，避免线上数据和参考数据一模一样（不真实）。
NORMAL_NOISE_SCALE = 0.05


class TrafficError(RuntimeError):
    """流量模拟失败。"""


def sample_rows(n: int, seed: int) -> pd.DataFrame:
    """从 iris 里有放回地抽样，作为「线上请求」的原始特征。"""
    frame = load_iris(as_frame=True).data.rename(columns=COLUMN_TO_API_FIELD)
    return frame.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def apply_shift(
    frame: pd.DataFrame, shifts: dict[str, float], noise_scale: float, seed: int
) -> pd.DataFrame:
    """给指定特征加偏移量，并叠加一点观测噪声。"""
    rng = random.Random(seed)
    shifted = frame.copy()
    for column in shifted.columns:
        offset = shifts.get(column, 0.0)
        shifted[column] = [
            max(0.01, value + offset + rng.gauss(0.0, noise_scale)) for value in shifted[column]
        ]
    return shifted.round(4)


def _post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Optional[dict], float]:
    """发一次 POST，返回 (状态码, 响应体, 耗时秒)。网络异常不抛出。"""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body, time.perf_counter() - start
    except urllib.error.HTTPError as exc:  # 422 / 503 等业务错误
        return exc.code, None, time.perf_counter() - start
    except urllib.error.URLError as exc:
        raise TrafficError(
            f"连接 {url} 失败：{exc.reason}。先确认服务已启动（docker compose up -d）。"
        ) from exc


def invalid_payload() -> dict[str, Any]:
    """故意构造一条不合法请求（少一个特征），服务会返回 422。"""
    return {"instances": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}]}


def run(args: argparse.Namespace) -> int:
    shifts = DEFAULT_DRIFT if args.mode == "drift" else {}
    if args.drift_feature:
        shifts = {name: args.drift_shift for name in args.drift_feature}

    total_rows = args.requests * args.batch_size
    raw = sample_rows(total_rows, seed=args.seed)
    features = apply_shift(raw, shifts, args.noise, seed=args.seed)

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    status_counts: dict[int, int] = {}
    rng = random.Random(args.seed)
    predict_url = f"{args.api_url.rstrip('/')}/predict"

    for index in range(args.requests):
        batch = features.iloc[index * args.batch_size : (index + 1) * args.batch_size]
        records = batch.to_dict("records")

        if not args.no_send and args.error_rate > 0 and rng.random() < args.error_rate:
            # 人为制造一次「调用方参数错误」，只影响系统指标，不影响数据分布
            status, _, elapsed = _post(predict_url, invalid_payload(), args.timeout)
            status_counts[status] = status_counts.get(status, 0) + 1
            latencies.append(elapsed)
            continue

        predictions: list[Optional[str]] = [None] * len(records)
        confidences: list[Optional[float]] = [None] * len(records)

        if not args.no_send:
            status, body, elapsed = _post(predict_url, {"instances": records}, args.timeout)
            status_counts[status] = status_counts.get(status, 0) + 1
            latencies.append(elapsed)
            if body:
                predictions = [item["label_name"] for item in body["predictions"]]
                confidences = [item["confidence"] for item in body["predictions"]]

        for record, label, confidence in zip(records, predictions, confidences):
            row = dict(record)
            if label is not None:
                row["prediction"] = label
                row["confidence"] = confidence
            rows.append(row)

        if args.delay:
            time.sleep(args.delay)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"模式：{args.mode}" + (f"（偏移量 {shifts}）" if shifts else "（无偏移）"))
    print(f"当前数据已写入 {args.out}（{len(frame)} 行）")
    if not args.no_send:
        ordered = sorted(latencies)
        p50 = statistics.median(ordered) * 1000 if ordered else 0.0
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)] * 1000 if ordered else 0.0
        print(f"请求数：{len(latencies)}  状态码分布：{dict(sorted(status_counts.items()))}")
        print(f"客户端观测延迟：P50 {p50:.1f} ms / P95 {p95:.1f} ms")
    if "prediction" in frame:
        print("\n预测类别分布：")
        print(frame["prediction"].value_counts().to_string())
    print(f"\n下一步：python monitoring/drift_check.py --current {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulate_traffic.py",
        description="给推理服务打流量，同时产出漂移检测用的「当前数据」CSV。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-url", default="http://localhost:8000", help="服务地址。")
    parser.add_argument("--requests", type=int, default=200, help="发送的请求数。")
    parser.add_argument("--batch-size", type=int, default=5, help="每个请求携带的样本数。")
    parser.add_argument(
        "--mode",
        choices=("normal", "drift"),
        default="normal",
        help="normal=贴近训练分布；drift=人为给花瓣特征加偏移。",
    )
    parser.add_argument(
        "--drift-feature",
        action="append",
        choices=API_FIELDS,
        help=f"自定义要偏移的特征（可重复）。默认 {list(DEFAULT_DRIFT)}。",
    )
    parser.add_argument(
        "--drift-shift", type=float, default=2.0, help="配合 --drift-feature 使用的偏移量(cm)。"
    )
    parser.add_argument(
        "--noise", type=float, default=NORMAL_NOISE_SCALE, help="叠加的高斯噪声标准差(cm)。"
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.0,
        help="故意发送非法请求的比例（0~1），用于制造 422、演示「错误率上升但无漂移」。",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="请求之间的间隔秒数。")
    parser.add_argument("--timeout", type=float, default=10.0, help="单请求超时秒数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现。")
    parser.add_argument("--no-send", action="store_true", help="只生成 CSV，不发 HTTP 请求。")
    parser.add_argument(
        "--out",
        type=Path,
        default=PHASE4_ROOT / "monitoring" / "current_data.csv",
        help="当前数据 CSV 输出路径。",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.error_rate <= 1.0:
        print("simulate_traffic.py: error: --error-rate 必须在 0~1 之间", file=sys.stderr)
        return 2
    try:
        return run(args)
    except TrafficError as exc:
        print(f"simulate_traffic.py: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("simulate_traffic.py: 已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
