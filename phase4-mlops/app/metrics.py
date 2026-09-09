"""模型级（model-level）指标：回答「模型预测还可不可信」。

和系统级指标严格分开：

* **系统指标**（``http_*``）由 ``prometheus-fastapi-instrumentator`` 自动产出，
  回答「服务本身健不健康」——QPS、延迟、错误率。见 :mod:`app.asgi`。
* **模型指标**（本模块，全部以 ``model_`` 前缀）回答「模型在预测什么、有多确信」
  ——预测类别分布、置信度分布、纯推理耗时、当前加载的是哪个模型。

两类指标在 Grafana 里也分属两个 dashboard，解读方式不同（见 README）。

注意边界：这里的模型指标是**实时信号**，只能提示「输出分布看起来变了」；
「输入特征是否真的漂移、漂移多少」由 ``monitoring/drift_check.py`` 的
Evidently 批处理报告给出结论。两者互相印证，不互相替代。
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

from prometheus_client import Counter, Histogram
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector, REGISTRY

LOGGER = logging.getLogger(__name__)

#: 置信度分桶：低置信度区间划细一些，模型「拿不准」时最先在这里体现。
CONFIDENCE_BUCKETS: tuple[float, ...] = (
    0.34,  # 三分类的随机水平
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
    1.0,
)

#: 纯推理耗时分桶（秒）。不含 HTTP/序列化开销，与系统延迟对照可判断瓶颈在哪。
INFERENCE_BUCKETS: tuple[float, ...] = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
)

#: 单次请求的样本条数分桶。
BATCH_SIZE_BUCKETS: tuple[float, ...] = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)


PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "模型成功产出的预测条数（按预测类别拆分）。类别占比突变是模型侧的早期信号。",
    ["predicted_class"],
)

PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "每条预测的置信度（预测类别的概率，0~1）。低置信度占比上升往往先于漂移被发现。",
    buckets=CONFIDENCE_BUCKETS,
)

PREDICTION_BATCH_SIZE = Histogram(
    "model_prediction_batch_size",
    "单次 /predict 请求携带的样本条数。",
    buckets=BATCH_SIZE_BUCKETS,
)

INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds",
    "纯模型推理耗时（预处理+predict+predict_proba），不含 HTTP 开销。",
    buckets=INFERENCE_BUCKETS,
)


class ModelStateCollector(Collector):
    """把「当前加载了哪个模型」暴露成指标，抓取时才读取应用状态。

    产出两个指标：

    * ``model_loaded`` —— 1 表示模型已加载，0 表示服务处于降级状态；
    * ``model_info`` —— 恒为 1，信息全在标签里（run_id / model_uuid / git_commit
      / model_uri）。这样 Grafana 面板上能直接看出「这段曲线是哪个模型跑出来的」，
      出问题时可以一路追回训练 run 和代码 commit。
    """

    def __init__(self, fastapi_app: Any) -> None:
        self._app = fastapi_app

    def collect(self) -> Iterator[Metric]:  # noqa: D102 - Collector 接口
        bundle = getattr(self._app.state, "bundle", None)

        loaded = GaugeMetricFamily(
            "model_loaded",
            "1=模型已加载并可服务，0=降级（/health 返回 503）。",
        )
        loaded.add_metric([], 1.0 if bundle is not None else 0.0)
        yield loaded

        info = GaugeMetricFamily(
            "model_info",
            "当前服务的模型标识，值恒为 1，信息在标签里。",
            labels=["run_id", "model_uuid", "model_uri", "git_commit"],
        )
        if bundle is not None:
            info.add_metric(
                [
                    bundle.run_id or "none",
                    bundle.model_uuid or "none",
                    bundle.model_uri,
                    (bundle.run_tags or {}).get("git_commit", "none"),
                ],
                1.0,
            )
        yield info


def register_model_state_collector(fastapi_app: Any) -> None:
    """注册 :class:`ModelStateCollector`（重复调用安全）。"""
    for collector in list(getattr(REGISTRY, "_collector_to_names", {})):
        if isinstance(collector, ModelStateCollector):
            return
    REGISTRY.register(ModelStateCollector(fastapi_app))


def _observe(results: Iterable[Any], records: Sequence[Mapping[str, float]], elapsed: float) -> None:
    """记录一次成功推理的模型级指标。"""
    INFERENCE_DURATION.observe(elapsed)
    PREDICTION_BATCH_SIZE.observe(len(records))
    for result in results:
        PREDICTIONS_TOTAL.labels(predicted_class=result.label_name).inc()
        PREDICTION_CONFIDENCE.observe(result.confidence)


def instrument_predict_batch() -> None:
    """给第二阶段的 ``model_loader.predict_batch`` 套一层指标采集。

    在 *推理函数* 而不是 HTTP 路由上埋点的原因：
    ① 不用改第二阶段的业务代码，也不用解析响应体；
    ② 记录到的是纯推理耗时，和系统延迟形成对照。

    **调用时机**：必须在 ``import app.main`` 之前调用——``app/main.py`` 是用
    ``from app.model_loader import predict_batch`` 把函数取到自己命名空间的，
    晚于它包装就不生效了。:mod:`app.asgi` 保证了这个顺序。

    失败的请求不计入模型指标（它们会以 4xx/5xx 出现在系统指标里）。
    """
    from app import model_loader  # 延迟导入：确保包路径合并已生效

    original = model_loader.predict_batch
    if getattr(original, "_phase4_instrumented", False):
        return  # 幂等：重复导入不会套两层

    @functools.wraps(original)
    def instrumented(bundle: Any, records: Sequence[Mapping[str, float]]) -> Any:
        start = time.perf_counter()
        results = original(bundle, records)
        _observe(results, records, time.perf_counter() - start)
        return results

    instrumented._phase4_instrumented = True  # type: ignore[attr-defined]
    model_loader.predict_batch = instrumented  # type: ignore[assignment]
    LOGGER.info("模型级指标已挂载到 model_loader.predict_batch")


__all__ = [
    "INFERENCE_DURATION",
    "PREDICTIONS_TOTAL",
    "PREDICTION_BATCH_SIZE",
    "PREDICTION_CONFIDENCE",
    "ModelStateCollector",
    "instrument_predict_batch",
    "register_model_state_collector",
]
