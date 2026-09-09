"""ASGI 入口：第二阶段的 FastAPI 应用 + Prometheus 埋点。

    uvicorn app.asgi:app --host 0.0.0.0 --port 8000

这里只做三件事，业务逻辑一行都不重复：

1. 先给 ``model_loader.predict_batch`` 套上**模型级**指标（必须早于导入
   ``app.main``，原因见 :func:`app.metrics.instrument_predict_batch`）；
2. 导入第二阶段的 FastAPI 应用，挂上 ``prometheus-fastapi-instrumentator``
   采集**系统级**指标，并暴露 ``/metrics``；
3. 注册模型状态采集器，让 ``/metrics`` 里能看到当前加载的是哪个模型。

系统指标（``http_*``）和模型指标（``model_*``）共用一个 ``/metrics`` 端点
——Prometheus 的抓取模型如此——但**指标族、面板、解读方式完全分开**，
Grafana 里也是两个 dashboard。
"""

from __future__ import annotations

import logging

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator import metrics as instrumentator_metrics

from app import metrics as model_metrics

# ⚠ 顺序敏感：包装必须发生在 `app.main` 被导入之前。
model_metrics.instrument_predict_batch()

from app.main import app  # noqa: E402  (顺序见上方说明)

LOGGER = logging.getLogger("app.asgi")

#: HTTP 延迟分桶（秒）。覆盖 5ms~5s：本服务正常在 5~50ms，留出长尾观察空间。
#: Grafana 面板把结果 ×1000 显示成 ms。
LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.250,
    0.500,
    1.000,
    2.500,
    5.000,
)


def build_instrumentator() -> Instrumentator:
    """配置系统级指标采集器。

    显式声明要哪些指标（而不是用默认集合），这样 Grafana 里的 PromQL
    和这里的指标名一一对应，改动时不会对不上。
    """
    instrumentator = Instrumentator(
        should_group_status_codes=False,  # 保留具体状态码：422/500/503 要能分开看
        should_ignore_untemplated=True,  # 未匹配路由不建时间序列，避免高基数
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],  # 抓取自身不计入 QPS
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    )
    instrumentator.add(
        # 请求数：按 handler / method / 状态码分类 —— 错误率面板的数据来源
        instrumentator_metrics.requests(
            metric_name="http_requests_total",
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
        ),
        # 延迟直方图：QPS 与 P50/P95/P99 都从它推导
        instrumentator_metrics.latency(
            metric_name="http_request_duration_seconds",
            buckets=LATENCY_BUCKETS,
            should_include_handler=True,
            should_include_method=False,
            should_include_status=False,
        ),
    )
    return instrumentator


build_instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=True,
    tags=["monitoring"],
)

# 模型标识（run_id / model_uuid / git_commit）在抓取时从 app.state 读取
model_metrics.register_model_state_collector(app)

LOGGER.info("Prometheus 埋点已启用：/metrics（系统指标 http_*，模型指标 model_*）")

__all__ = ["app", "build_instrumentator", "LATENCY_BUCKETS"]
