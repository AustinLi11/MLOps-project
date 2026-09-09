"""漂移告警通知：把 Evidently 的漂移检测结果发到可替换的通知渠道。

安全边界（本阶段的核心设计）
---------------------------
本模块**只发送通知**。它不会、也不应该触发重训练：线上数据异常（上游埋点故障、
采样偏差、节假日效应）同样会表现为「漂移」，自动重训练很可能产出更差的模型。
检测到漂移之后的下一步是**人工判断**——由人去 GitHub Actions 页面手动点击
``manual-retrain`` workflow。整条闭环里，本模块的位置是「自动检测 → 自动通知」，
到此为止。

通知渠道通过配置文件切换（``config/notifier_config.yaml``），不需要改代码：
console / file / webhook（通用 JSON）/ slack / wecom（企业微信机器人）。
Webhook URL 一律用 ``${ENV_VAR}`` 形式从环境变量读取，不落明文。

用法::

    # 1) 消费已有的 Evidently 报告（阶段 4 监控产出的 JSON）
    python alert_notifier.py --drift-report drift.json

    # 2) 现场对比两份数据并检测漂移
    python alert_notifier.py --reference ref.csv --current current.csv

    # 3) 演示/验收：用 iris 注入人为漂移，走完整通知链路
    python alert_notifier.py --demo

    # 只打印将要发送的内容，不真的发（排查配置用）
    python alert_notifier.py --demo --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "notifier_config.yaml"

#: 配置里 ${VAR} 形式的占位符，用于从环境变量注入 Webhook URL 等敏感值
_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Evidently 的 p 值型方法：分数**低于**阈值才算漂移；其余（距离型）是高于阈值算漂移
_P_VALUE_HINT = "p_value"

LOGGER = logging.getLogger("alert_notifier")


# --------------------------------------------------------------------------- #
# 错误类型
# --------------------------------------------------------------------------- #
class AlertError(RuntimeError):
    """本模块所有可预期失败的基类（调用方转成退出码，不抛裸栈）。"""


class ConfigError(AlertError):
    """配置文件缺失、格式错误或渠道未知。"""


class DriftReportError(AlertError):
    """漂移报告无法解析。"""


class NotificationError(AlertError):
    """通知发送失败。"""


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DriftFinding:
    """单个特征的漂移结论。"""

    feature: str
    method: str
    score: float
    threshold: float
    drifted: bool
    #: 距离型幅度（如 wasserstein 归一化距离），用于回答「漂移了多少」
    magnitude: Optional[float] = None
    magnitude_method: Optional[str] = None
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None

    @property
    def mean_shift_pct(self) -> Optional[float]:
        """当前均值相对参考均值的偏移百分比（参考均值为 0 时返回 None）。"""
        if self.reference_mean in (None, 0) or self.current_mean is None:
            return None
        return (self.current_mean - self.reference_mean) / abs(self.reference_mean) * 100.0

    def describe(self) -> str:
        """一行人类可读描述：特征名 + 幅度 + 判定依据。"""
        parts = [f"`{self.feature}`"]
        if self.magnitude is not None:
            parts.append(f"{self.magnitude_method}={self.magnitude:.4f}")
        parts.append(f"{self.method}={self.score:.3g} (阈值 {self.threshold:g})")
        shift = self.mean_shift_pct
        if shift is not None:
            parts.append(f"均值 {self.reference_mean:.3f} → {self.current_mean:.3f}（{shift:+.1f}%）")
        return " · ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "method": self.method,
            "score": self.score,
            "threshold": self.threshold,
            "drifted": self.drifted,
            "magnitude": self.magnitude,
            "magnitude_method": self.magnitude_method,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
            "mean_shift_pct": self.mean_shift_pct,
        }


@dataclass(frozen=True)
class DriftAlert:
    """一次漂移检测的完整结论，以及要发出去的通知内容。"""

    detected_at: datetime
    dataset: str
    findings: Sequence[DriftFinding]
    total_features: int
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    @property
    def drifted(self) -> list[DriftFinding]:
        return [f for f in self.findings if f.drifted]

    @property
    def drift_share(self) -> float:
        if not self.total_features:
            return 0.0
        return len(self.drifted) / self.total_features

    @property
    def detected_at_text(self) -> str:
        return self.detected_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def to_payload(self) -> dict[str, Any]:
        """通用 JSON webhook 的结构化载荷。"""
        return {
            "event": "data_drift_detected",
            "detected_at": self.detected_at.astimezone(timezone.utc).isoformat(),
            "dataset": self.dataset,
            "source": self.source,
            "summary": {
                "drifted_features": len(self.drifted),
                "total_features": self.total_features,
                "drift_share": round(self.drift_share, 4),
                "thresholds": dict(self.thresholds),
            },
            "drifted_features": [f.to_dict() for f in self.drifted],
            # 明确告诉接收方：这只是通知，后续动作需要人工决定
            "next_step": {
                "type": "manual_confirmation_required",
                "action": "人工确认后，在 GitHub Actions 手动触发 manual-retrain workflow",
                "auto_retrain": False,
                "auto_deploy": False,
            },
        }

    def to_markdown(self) -> str:
        """给 Slack / 企业微信 / 控制台看的 markdown 文本。"""
        lines = [
            f"**⚠️ 数据漂移告警** · {self.dataset}",
            "",
            f"- 检测时间：{self.detected_at_text}",
            f"- 漂移特征：{len(self.drifted)}/{self.total_features}"
            f"（占比 {self.drift_share:.0%}）",
        ]
        if self.source:
            lines.append(f"- 数据来源：{self.source}")
        if self.thresholds:
            thresholds = ", ".join(f"{k}={v}" for k, v in self.thresholds.items())
            lines.append(f"- 告警阈值：{thresholds}")
        lines.append("")
        lines.append("漂移明细（特征 · 幅度 · 判定依据）：")
        for finding in self.drifted:
            lines.append(f"- {finding.describe()}")
        lines += [
            "",
            "**下一步需要人工确认**：先排查是真实分布变化还是上游数据异常；",
            "确认需要重训练时，到 GitHub Actions 手动运行 `manual-retrain` workflow。",
            "本告警**不会**自动触发重训练，也不会自动上线任何模型。",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 通知渠道：新增渠道只需实现 send() 并注册到 CHANNELS
# --------------------------------------------------------------------------- #
class NotificationChannel(Protocol):
    """通知渠道接口。实现它即可接入新平台，调用方无需改动。"""

    name: str

    def send(self, alert: DriftAlert) -> None:
        """发送告警；失败时抛 :class:`NotificationError`。"""


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 10.0,
    channel: str = "webhook",
) -> None:
    """向 URL POST 一段 JSON（只用标准库，避免额外依赖）。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json; charset=utf-8")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            text = response.read(512).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read(512).decode("utf-8", errors="replace")
        raise NotificationError(
            f"渠道 {channel} 发送失败：HTTP {exc.code} {exc.reason}。响应：{detail[:200]}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NotificationError(
            f"渠道 {channel} 无法连接（{type(exc).__name__}: {exc}）。"
            "请检查 Webhook URL、网络与代理设置。"
        ) from exc
    if status >= 300:
        raise NotificationError(f"渠道 {channel} 返回异常状态 {status}：{text[:200]}")
    LOGGER.info("渠道 %s 发送成功（HTTP %s）", channel, status)


@dataclass
class ConsoleChannel:
    """打印到 stdout。默认渠道，本地调试与无网络环境用。"""

    name: str = "console"

    def send(self, alert: DriftAlert) -> None:
        print(alert.to_markdown())


@dataclass
class FileChannel:
    """追加写入 JSONL 文件，用于留档/审计，也方便自动化测试断言。"""

    path: Path
    name: str = "file"

    def send(self, alert: DriftAlert) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(alert.to_payload(), ensure_ascii=False) + "\n")
        except OSError as exc:
            raise NotificationError(f"写入告警文件 {self.path} 失败：{exc.strerror or exc}") from exc
        LOGGER.info("告警已写入 %s", self.path)


@dataclass
class WebhookChannel:
    """通用 JSON Webhook：POST :meth:`DriftAlert.to_payload` 的结构化内容。"""

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout: float = 10.0
    name: str = "webhook"

    def send(self, alert: DriftAlert) -> None:
        _post_json(
            self.url, alert.to_payload(), headers=self.headers, timeout=self.timeout, channel=self.name
        )


@dataclass
class SlackChannel:
    """Slack Incoming Webhook：``{"text": "<markdown>"}``。"""

    url: str
    timeout: float = 10.0
    name: str = "slack"

    def send(self, alert: DriftAlert) -> None:
        payload = {
            "text": f"⚠️ 数据漂移告警 · {alert.dataset}",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": alert.to_markdown()},
                }
            ],
        }
        _post_json(self.url, payload, timeout=self.timeout, channel=self.name)


@dataclass
class WeComChannel:
    """企业微信群机器人：``{"msgtype": "markdown", ...}``。"""

    url: str
    timeout: float = 10.0
    name: str = "wecom"

    def send(self, alert: DriftAlert) -> None:
        payload = {"msgtype": "markdown", "markdown": {"content": alert.to_markdown()}}
        _post_json(self.url, payload, timeout=self.timeout, channel=self.name)


def _build_console(_: Mapping[str, Any]) -> ConsoleChannel:
    return ConsoleChannel()


def _build_file(settings: Mapping[str, Any]) -> FileChannel:
    raw_path = settings.get("path")
    if not raw_path:
        raise ConfigError("渠道 file 需要配置 path")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return FileChannel(path=path)


def _require_url(settings: Mapping[str, Any], channel: str) -> str:
    url = str(settings.get("url", "")).strip()
    if not url:
        raise ConfigError(
            f"渠道 {channel} 需要配置 url。请在 notifier_config.yaml 里写成 "
            "${ENV_VAR} 形式，并把真实地址放到环境变量/GitHub Secrets 里。"
        )
    if url.startswith("${"):
        raise ConfigError(
            f"渠道 {channel} 的 url 占位符未被解析：{url}。"
            "对应的环境变量没有设置——先 export 该变量再运行。"
        )
    return url


def _build_webhook(settings: Mapping[str, Any]) -> WebhookChannel:
    return WebhookChannel(
        url=_require_url(settings, "webhook"),
        headers={str(k): str(v) for k, v in (settings.get("headers") or {}).items()},
        timeout=float(settings.get("timeout_seconds", 10)),
    )


def _build_slack(settings: Mapping[str, Any]) -> SlackChannel:
    return SlackChannel(
        url=_require_url(settings, "slack"), timeout=float(settings.get("timeout_seconds", 10))
    )


def _build_wecom(settings: Mapping[str, Any]) -> WeComChannel:
    return WeComChannel(
        url=_require_url(settings, "wecom"), timeout=float(settings.get("timeout_seconds", 10))
    )


#: 渠道注册表：配置文件里的名字 → 构造函数。接入新平台就在这里加一行。
CHANNELS: dict[str, Any] = {
    "console": _build_console,
    "file": _build_file,
    "webhook": _build_webhook,
    "slack": _build_slack,
    "wecom": _build_wecom,
}


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def _expand_env(value: Any) -> Any:
    """递归把 ``${VAR}`` 替换成环境变量值（未设置则保留原样，便于报错提示）。"""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(0))

        return _ENV_PLACEHOLDER.sub(replace, value)
    if isinstance(value, Mapping):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """读取 YAML 配置并展开环境变量占位符。

    Raises:
        ConfigError: 文件不存在、不可读或不是 YAML 映射。
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"找不到配置文件 {path}。可用 --config 指定路径。") from exc
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}：{exc.strerror or exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 {path} 不是合法 YAML：{exc}") from exc

    if not isinstance(data, Mapping):
        raise ConfigError(f"配置文件 {path} 的顶层必须是映射（key: value）。")
    return dict(_expand_env(dict(data)))


def build_channels(config: Mapping[str, Any]) -> list[NotificationChannel]:
    """按配置里的 ``active_channels`` 构造渠道实例（切换渠道不需要改代码）。

    Raises:
        ConfigError: 未配置渠道、渠道名未知或渠道参数缺失。
    """
    active = config.get("active_channels") or []
    if isinstance(active, str):
        active = [active]
    if not active:
        raise ConfigError(
            "配置里 active_channels 为空。至少填一个渠道，例如：active_channels: [console]"
        )

    settings_all = config.get("channels") or {}
    channels: list[NotificationChannel] = []
    for name in active:
        builder = CHANNELS.get(str(name))
        if builder is None:
            raise ConfigError(
                f"未知通知渠道 '{name}'。可选：{sorted(CHANNELS)}。"
                "接入新平台请实现 NotificationChannel 并注册到 CHANNELS。"
            )
        channels.append(builder(settings_all.get(str(name)) or {}))
    return channels


# --------------------------------------------------------------------------- #
# 解析 Evidently 报告
# --------------------------------------------------------------------------- #
def _is_drifted(method: str, score: float, threshold: float) -> bool:
    """判定方向：p 值型「低于阈值」算漂移，距离型「达到阈值」算漂移。

    Evidently 0.7 的 ``ValueDrift`` 只给分数不给判定标记，所以这里显式区分方向。
    """
    if _P_VALUE_HINT in method.lower():
        return score < threshold
    return score >= threshold


def parse_evidently_report(report: Mapping[str, Any]) -> list[DriftFinding]:
    """从 Evidently 0.7 的 ``Report.run(...).dict()`` 结构里抽取每个特征的漂移结论。

    Raises:
        DriftReportError: 结构不符合预期（既没有 metrics 也没有 drifted_features）。
    """
    metrics = report.get("metrics")
    if not isinstance(metrics, list):
        raise DriftReportError(
            "漂移报告里没有 metrics 列表。期望 Evidently 0.7 的 Report.run(...).dict() 结构。"
        )

    scores: dict[str, DriftFinding] = {}
    magnitudes: dict[str, tuple[str, float]] = {}
    for metric in metrics:
        if not isinstance(metric, Mapping):
            continue
        config = metric.get("config") or {}
        if not str(config.get("type", "")).endswith("ValueDrift"):
            continue
        column = config.get("column")
        value = metric.get("value")
        if column is None or not isinstance(value, (int, float)):
            continue
        method = str(config.get("method", "unknown"))
        threshold = float(config.get("threshold", 0.05))
        score = float(value)
        if _P_VALUE_HINT in method.lower():
            scores[str(column)] = DriftFinding(
                feature=str(column),
                method=method,
                score=score,
                threshold=threshold,
                drifted=_is_drifted(method, score, threshold),
            )
        else:
            # 距离型指标当作「幅度」附加到同一特征上
            magnitudes[str(column)] = (method, score)
            scores.setdefault(
                str(column),
                DriftFinding(
                    feature=str(column),
                    method=method,
                    score=score,
                    threshold=threshold,
                    drifted=_is_drifted(method, score, threshold),
                ),
            )

    if not scores:
        raise DriftReportError("漂移报告里没有任何 ValueDrift 指标，无法判断特征级漂移。")

    findings: list[DriftFinding] = []
    for column, finding in scores.items():
        method_magnitude = magnitudes.get(column)
        if method_magnitude:
            finding = DriftFinding(
                **{
                    **finding.to_dict(),
                    "magnitude_method": method_magnitude[0],
                    "magnitude": method_magnitude[1],
                }
            )
        findings.append(finding)
    # 漂移的排在前面，其中幅度大的优先
    findings.sort(key=lambda f: (not f.drifted, -(f.magnitude or 0.0)))
    return findings


def _finding_from_dict(data: Mapping[str, Any]) -> DriftFinding:
    return DriftFinding(
        feature=str(data["feature"]),
        method=str(data.get("method", "unknown")),
        score=float(data.get("score", 0.0)),
        threshold=float(data.get("threshold", 0.0)),
        drifted=bool(data.get("drifted", True)),
        magnitude=data.get("magnitude"),
        magnitude_method=data.get("magnitude_method"),
        reference_mean=data.get("reference_mean"),
        current_mean=data.get("current_mean"),
    )


def load_drift_report(path: Path, dataset: str, thresholds: Mapping[str, Any]) -> DriftAlert:
    """读取漂移报告 JSON。

    同时兼容两种输入：Evidently 原生 ``dict()`` 输出，以及本模块自己产出的
    ``to_payload()``（便于把上一次告警重放/补发）。
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DriftReportError(f"找不到漂移报告 {path}。") from exc
    except OSError as exc:
        raise DriftReportError(f"无法读取漂移报告 {path}：{exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise DriftReportError(f"漂移报告 {path} 不是合法 JSON：{exc}") from exc

    if not isinstance(report, Mapping):
        raise DriftReportError(f"漂移报告 {path} 的顶层必须是 JSON 对象。")

    if "drifted_features" in report and "metrics" not in report:
        findings = [_finding_from_dict(item) for item in report["drifted_features"]]
        summary = report.get("summary") or {}
        total = int(summary.get("total_features") or len(findings))
        detected_at = _parse_timestamp(report.get("detected_at"))
        return DriftAlert(
            detected_at=detected_at,
            dataset=str(report.get("dataset") or dataset),
            findings=findings,
            total_features=total,
            thresholds=thresholds,
            source=str(path),
        )

    findings = parse_evidently_report(report)
    return DriftAlert(
        detected_at=_parse_timestamp(report.get("timestamp")),
        dataset=dataset,
        findings=findings,
        total_features=len(findings),
        thresholds=thresholds,
        source=str(path),
    )


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            LOGGER.warning("无法解析报告里的时间戳 %r，改用当前时间", value)
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# 现场检测（可选路径：需要 evidently + pandas）
# --------------------------------------------------------------------------- #
def detect_drift(
    reference: "Any",
    current: "Any",
    dataset: str = "current-vs-reference",
    columns: Optional[Sequence[str]] = None,
    thresholds: Optional[Mapping[str, Any]] = None,
    save_report: Optional[Path] = None,
) -> DriftAlert:
    """用 Evidently 对比两份 DataFrame，返回归一化后的告警对象。

    除 K-S p 值外，额外对每个数值特征跑一次 wasserstein 距离作为「幅度」，
    并附上参考/当前均值，让通知里能回答「漂移了多少」。

    Raises:
        DriftReportError: Evidently 未安装或检测失败。
    """
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.metrics import ValueDrift
        from evidently.presets import DataDriftPreset
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise DriftReportError(
            "现场检测需要 evidently：pip install -r requirements.txt。"
            "若只想消费已有报告，请改用 --drift-report。"
        ) from exc

    feature_columns = list(columns or [c for c in reference.columns if c in current.columns])
    if not feature_columns:
        raise DriftReportError("参考数据与当前数据没有共同的特征列，无法比较。")

    definition = DataDefinition(numerical_columns=feature_columns)
    reference_ds = Dataset.from_pandas(reference[feature_columns], data_definition=definition)
    current_ds = Dataset.from_pandas(current[feature_columns], data_definition=definition)

    metrics: list[Any] = [DataDriftPreset()]
    metrics += [ValueDrift(column=column, method="wasserstein") for column in feature_columns]

    try:
        run = Report(metrics).run(current_ds, reference_ds)
        report_dict = run.dict()
    except Exception as exc:  # noqa: BLE001 - Evidently 内部异常类型不稳定
        raise DriftReportError(f"Evidently 漂移检测失败：{type(exc).__name__}: {exc}") from exc

    if save_report is not None:
        save_report.parent.mkdir(parents=True, exist_ok=True)
        save_report.write_text(
            json.dumps(report_dict, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        LOGGER.info("Evidently 原始报告已保存到 %s", save_report)

    findings = []
    for finding in parse_evidently_report(report_dict):
        findings.append(
            DriftFinding(
                **{
                    **finding.to_dict(),
                    "reference_mean": float(reference[finding.feature].mean()),
                    "current_mean": float(current[finding.feature].mean()),
                }
            )
        )

    return DriftAlert(
        detected_at=datetime.now(timezone.utc),
        dataset=dataset,
        findings=findings,
        total_features=len(feature_columns),
        thresholds=thresholds or {},
        source="evidently:DataDriftPreset+wasserstein",
    )


def _demo_frames() -> tuple["Any", "Any"]:
    """演示数据：iris 前 100 行作参考，后 100 行注入人为漂移作当前数据。"""
    try:
        from sklearn.datasets import load_iris
    except ImportError as exc:  # pragma: no cover
        raise DriftReportError("--demo 需要 scikit-learn：pip install -r requirements.txt") from exc

    frame = load_iris(as_frame=True).data
    reference = frame.iloc[:100].copy()
    current = frame.iloc[50:].copy()
    # 模拟「上游埋点把厘米写成了别的单位」这类真实故障
    current["petal length (cm)"] = current["petal length (cm)"] * 1.8 + 2.0
    current["sepal width (cm)"] = current["sepal width (cm)"] + 0.6
    return reference, current


# --------------------------------------------------------------------------- #
# 告警判定与发送
# --------------------------------------------------------------------------- #
def should_alert(alert: DriftAlert, config: Mapping[str, Any]) -> tuple[bool, str]:
    """按配置阈值判断是否需要告警，并给出原因（写进日志/summary）。"""
    detection = config.get("detection") or {}
    min_features = int(detection.get("min_drifted_features", 1))
    min_share = float(detection.get("drift_share_threshold", 0.0))

    drifted = len(alert.drifted)
    if drifted < min_features:
        return False, f"漂移特征 {drifted} 个 < 阈值 {min_features} 个，不告警"
    if alert.drift_share < min_share:
        return False, (
            f"漂移占比 {alert.drift_share:.0%} < 阈值 {min_share:.0%}，不告警"
        )
    return True, (
        f"漂移特征 {drifted}/{alert.total_features}（{alert.drift_share:.0%}）"
        f"达到阈值（≥{min_features} 个且 ≥{min_share:.0%}）"
    )


def send_alert(alert: DriftAlert, channels: Sequence[NotificationChannel]) -> None:
    """向所有渠道发送；任一渠道失败都抛错，但不影响已成功的渠道。

    Raises:
        NotificationError: 至少一个渠道发送失败。
    """
    failures: list[str] = []
    for channel in channels:
        try:
            channel.send(alert)
        except NotificationError as exc:
            failures.append(f"{getattr(channel, 'name', channel)}: {exc}")
            LOGGER.error("渠道发送失败：%s", exc)
    if failures:
        raise NotificationError("以下渠道发送失败 → " + " | ".join(failures))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alert_notifier.py",
        description="检测/读取数据漂移结果，并通过可配置的 Webhook 渠道发送告警（不触发重训练）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--drift-report", type=Path, help="已有的漂移报告 JSON（阶段 4 监控产出）")
    source.add_argument("--reference", type=Path, help="参考数据 CSV（与 --current 搭配）")
    source.add_argument("--demo", action="store_true", help="用 iris 注入人为漂移做端到端演示")

    parser.add_argument("--current", type=Path, help="当前数据 CSV（与 --reference 搭配）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="通知渠道配置")
    parser.add_argument(
        "--dataset-name", type=str, default="iris-inference@production", help="通知里显示的数据集名"
    )
    parser.add_argument("--save-report", type=Path, default=None, help="把 Evidently 原始报告存成 JSON")
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印将要发送的内容，不实际调用 Webhook"
    )
    parser.add_argument(
        "--force", action="store_true", help="忽略阈值判定，强制发送（用于验收/联调）"
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="发出告警时以退出码 2 结束（方便被监控任务感知）",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    return parser


def _load_csv(path: Path) -> "Any":
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise DriftReportError("读取 CSV 需要 pandas：pip install -r requirements.txt") from exc
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DriftReportError(f"找不到数据文件 {path}。") from exc
    except Exception as exc:  # noqa: BLE001
        raise DriftReportError(f"读取 {path} 失败：{type(exc).__name__}: {exc}") from exc


def resolve_alert(args: argparse.Namespace, config: Mapping[str, Any]) -> DriftAlert:
    """根据命令行参数得到一个归一化的 DriftAlert。"""
    thresholds = dict(config.get("detection") or {})
    if args.drift_report is not None:
        return load_drift_report(args.drift_report, args.dataset_name, thresholds)

    if args.demo:
        reference, current = _demo_frames()
        dataset = f"{args.dataset_name} (demo: 注入人为漂移)"
    else:
        if args.current is None:
            raise ConfigError("--reference 需要与 --current 搭配使用。")
        reference, current = _load_csv(args.reference), _load_csv(args.current)
        dataset = args.dataset_name

    return detect_drift(
        reference=reference,
        current=current,
        dataset=dataset,
        thresholds=thresholds,
        save_report=args.save_report,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。返回退出码，不向外抛裸栈。

    退出码：0 正常（含「未达阈值不告警」）；1 可预期错误；2 已告警且指定了 --fail-on-drift。
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    try:
        config = load_config(args.config)
        alert = resolve_alert(args, config)
        needed, reason = should_alert(alert, config)

        if not needed and not args.force:
            LOGGER.info("无需告警：%s", reason)
            print(f"未达告警阈值：{reason}")
            return 0
        if not needed and args.force:
            LOGGER.warning("--force 生效：尽管%s，仍然发送", reason)

        if args.dry_run:
            print("=== dry-run：以下内容不会真正发送 ===")
            print(f"渠道：{config.get('active_channels')}")
            print(alert.to_markdown())
            print("\n--- 通用 webhook 载荷 ---")
            print(json.dumps(alert.to_payload(), ensure_ascii=False, indent=2))
            return 0

        channels = build_channels(config)
        send_alert(alert, channels)
        print(
            f"已通过渠道 {[getattr(c, 'name', '?') for c in channels]} 发送告警："
            f"{len(alert.drifted)}/{alert.total_features} 个特征漂移（{reason}）"
        )
    except AlertError as exc:
        print(f"alert_notifier.py: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("alert_notifier.py: 被用户中断", file=sys.stderr)
        return 130

    return 2 if args.fail_on_drift else 0


if __name__ == "__main__":
    sys.exit(main())
