"""生成 `reference_data.csv`：训练数据分布的快照，作为漂移检测的基线。

参考数据取的是**第一阶段训练时真正用到的那一份训练集切分**（直接复用
`phase1-mlops/train.py` 的 `load_dataset()`，不重新实现切分逻辑），因此
"当前线上数据 vs 参考数据" 比较的是「线上输入」和「模型见过的输入」。

可选地再用一个模型给参考数据打分，把预测类别写进 `prediction` 列——它是
**模型输出漂移**（prediction drift）的基线：线上预测的类别结构如果明显偏离
这一列的分布，即使输入特征没漂，也说明模型行为变了。

用法::

    python monitoring/make_reference.py                      # 只有特征列
    python monitoring/make_reference.py --model-uri ../phase2-mlops/export/v1/model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

PHASE4_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE4_ROOT.parent
PHASE1_ROOT = REPO_ROOT / "phase1-mlops"
if str(PHASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE1_ROOT))

from mlops_shared.features import COLUMN_TO_API_FIELD, TARGET_NAMES  # noqa: E402

DEFAULT_OUTPUT = PHASE4_ROOT / "monitoring" / "reference_data.csv"


class ReferenceDataError(RuntimeError):
    """参考数据生成失败。"""


def build_reference_frame(test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    """取第一阶段的训练集切分作为参考分布。"""
    try:
        import train  # phase1-mlops/train.py
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise ReferenceDataError(
            f"无法导入第一阶段的 train.py（{exc}）。请确认仓库结构完整，"
            "并已安装 phase4-mlops/requirements-monitoring.txt 的依赖。"
        ) from exc

    dataset = train.load_dataset(test_size=test_size, random_state=random_state)
    # 列名换成 API 字段名（snake_case）：线上请求就是这么发的，两边保持一致
    features = dataset.X_train.rename(columns=COLUMN_TO_API_FIELD)
    return features.reset_index(drop=True), dataset.y_train.reset_index(drop=True)


def score_reference(features: pd.DataFrame, model_uri: str) -> list[str]:
    """用指定模型给参考数据打分，得到 prediction drift 的基线分布。"""
    sys.path.insert(0, str(REPO_ROOT / "phase2-mlops"))
    try:
        from app.model_loader import load_model_bundle, predict_batch
    except ImportError as exc:  # pragma: no cover
        raise ReferenceDataError(
            f"无法导入第二阶段的模型加载器（{exc}）。"
        ) from exc

    try:
        bundle = load_model_bundle(requested_uri=model_uri)
    except Exception as exc:  # noqa: BLE001 - 统一成可读错误
        raise ReferenceDataError(
            f"加载模型 '{model_uri}' 失败：{type(exc).__name__}: {exc}。"
            "可以先按 README 步骤 1 导出模型目录，或直接省略 --model-uri。"
        ) from exc

    results = predict_batch(bundle, features.to_dict("records"))
    return [result.label_name for result in results]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_reference.py",
        description="生成漂移检测用的参考数据（训练集分布快照）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 CSV 路径。")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="与训练时一致的测试集比例。"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="与训练时一致的随机种子。"
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default=None,
        help="可选：用该模型给参考数据打分，写入 prediction 列（模型输出漂移的基线）。",
    )
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="额外写入真实标签列 target（仅作参考，不参与漂移计算）。",
    )
    args = parser.parse_args(argv)

    try:
        features, target = build_reference_frame(args.test_size, args.random_state)
        frame = features.copy()
        if args.model_uri:
            frame["prediction"] = score_reference(features, args.model_uri)
        if args.include_target:
            frame["target"] = [TARGET_NAMES[int(label)] for label in target]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    except ReferenceDataError as exc:
        print(f"make_reference.py: error: {exc}", file=sys.stderr)
        return 1

    print(f"参考数据已写入 {args.output}（{len(frame)} 行，{len(frame.columns)} 列）")
    print(frame.describe().T.to_string())
    if "prediction" in frame:
        print("\nprediction 分布（模型输出漂移的基线）：")
        print(frame["prediction"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
