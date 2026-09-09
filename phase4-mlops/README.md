# Phase 4 — 双重监控：系统指标 + 模型指标

给第二阶段的推理服务补上可观测性。本阶段最重要的一件事不是「把 Grafana 跑起来」，
而是**把两类完全不同的问题分开**：

| | 系统监控（system） | 模型监控（model / data） |
| --- | --- | --- |
| 回答的问题 | 服务**还活着吗**？快不快？报不报错？ | 模型的预测**还可信吗**？输入数据还是模型见过的那种吗？ |
| 数据来源 | FastAPI 埋点 `http_*`（prometheus-fastapi-instrumentator） | 推理埋点 `model_*` + Evidently 批处理报告 |
| 时效 | 实时，15 秒抓一次 | 实时信号（置信度/类别分布）+ 离线批处理（漂移报告） |
| 典型信号 | QPS、P95 延迟、5xx/422 错误率 | 预测类别分布、置信度下降、某个特征的分布偏移 |
| 出问题时先找谁 | 后端/SRE：进程、资源、依赖、上游调用方 | 算法/数据：上游数据源、特征口径、是否该重训 |
| 展示位置 | Grafana → **Service overview (系统指标)** | Grafana → **Model quality & drift signals (模型指标)** + Evidently HTML 报告 |
| 阈值语义 | 工程 SLO（如 P95 < 200ms、错误率 < 1%） | 统计检验（p 值、分布距离），见下文「阈值怎么定」 |

**两者互不替代**：服务 200 全绿、延迟很低，模型也可能在安静地预测错（漂移）；
反过来错误率飙升也可能跟数据分布毫无关系（见文末「错误率上升但没有漂移」）。
所以面板分成两块、报告分成两份，**不混在一起**。

> 本文档中的所有命令和数字都是在本机实测跑出来的（Docker 29.7.2 / linux-arm64，
> Prometheus v3.6.0，Grafana 12.2.0，Evidently 0.7.21）。

## 目录结构

```
phase4-mlops/
├── docker-compose.yml              # api + prometheus + grafana 三个容器
├── Dockerfile                      # 第二阶段服务 + 埋点（多阶段、非 root）
├── requirements.txt                # 服务运行时（= phase2 依赖 + instrumentator）
├── requirements-monitoring.txt     # 漂移分析用（Evidently），不进服务镜像
├── prometheus/
│   └── prometheus.yml              # scrape 配置（含 scrape_interval 选型说明）
├── grafana/
│   ├── provisioning/               # 数据源 + dashboard 自动加载，无需手动配置
│   └── dashboards/
│       ├── service-overview.json   # 系统指标：QPS / P95 / 错误率 …（11 个面板）
│       └── model-overview.json     # 模型指标：类别分布 / 置信度 / 推理耗时（10 个面板）
├── app/
│   ├── __init__.py                 # 把第二阶段的 app 包并入本包（复用，不复制）
│   ├── metrics.py                  # 模型级指标 + 对 predict_batch 的埋点
│   └── asgi.py                     # 入口：phase2 应用 + Instrumentator + /metrics
├── monitoring/
│   ├── make_reference.py           # 生成参考数据（训练集分布快照）
│   ├── simulate_traffic.py         # 打流量 / 人为制造特征偏移
│   ├── drift_check.py              # Evidently 漂移报告（HTML + JSON）
│   └── reference_data.csv          # 参考数据（120 行训练集 + 模型打分基线）
└── reports/                        # 运行产物（HTML/JSON，已 gitignore）
```

### 服务代码是怎么"复用"第二阶段的

`app/__init__.py` 把 `phase2-mlops/app` 追加进本包的 `__path__`，于是
`app.main` / `app.model_loader` / `app.schemas` **直接就是第二阶段的源码**，
本目录只新增 `metrics.py` 和 `asgi.py`。没有复制任何业务代码，第二阶段改了这里立刻同步。

埋点挂在 `model_loader.predict_batch` 上（而不是 HTTP 路由上）：
不用改第二阶段的代码，也不用解析响应体，而且量到的是**纯推理耗时**，
可以和 HTTP 延迟对照着看是慢在模型还是慢在 HTTP 层。

## 快速开始

### 步骤 1 · 准备一个模型（服务不含模型，运行时注入）

```bash
cd "/path/to/MLOps project"
# 若还没训练过：cd phase1-mlops && python train.py --n-estimators 100 --max-depth 3 --random-state 42
MLFLOW_TRACKING_URI="sqlite:///$PWD/phase1-mlops/mlflow.db" \
  mlflow artifacts download --artifact-uri "runs:/<RUN_ID>/model" \
  --dst-path phase2-mlops/export/v1
```

### 步骤 2 · 起监控栈

```bash
cd phase4-mlops
docker compose up -d --build
docker compose ps
```

| 服务 | 地址 | 说明 |
| --- | --- | --- |
| 推理服务 | <http://localhost:8000/docs> | `/metrics` 在 <http://localhost:8000/metrics> |
| Prometheus | <http://localhost:9090/targets> | 确认 `iris-inference` 是 **UP** |
| Grafana | <http://localhost:3000> | admin/admin，已开匿名只读；两份 dashboard 自动加载在 **MLOps** 文件夹 |

### 步骤 3 · 打流量看面板

```bash
pip install -r requirements-monitoring.txt          # 首次：装 Evidently 等
python monitoring/simulate_traffic.py --requests 400 --batch-size 4 --delay 0.22 --error-rate 0.08
```

### 步骤 4 · 跑一次漂移检查

```bash
python monitoring/drift_check.py --current monitoring/current_data.csv
open reports/drift_report.html
```

---

## 一、系统指标（服务健不健康）

### 暴露了哪些指标

| 指标 | 类型 | 标签 | 用途 |
| --- | --- | --- | --- |
| `http_requests_total` | Counter | `handler` / `method` / `status` | 请求数、QPS、**按状态码分类的错误计数** |
| `http_request_duration_seconds` | Histogram | `handler`, `le` | 延迟分布，推导 P50/P95/P99 |
| `http_requests_inprogress` | Gauge | `handler` / `method` | 并发处理中的请求数 |
| `up{job="iris-inference"}` | Gauge | — | Prometheus 抓取成功与否（服务挂了这里就是 0） |

状态码**不做分组**（`should_group_status_codes=False`），因为本服务的 422（入参不合法）、
503（模型未加载）、500（未预期异常）含义完全不同，合并成 `4xx/5xx` 会丢失排障信息。

延迟分桶固定为 `5ms ~ 5s` 十档（见 `app/asgi.py: LATENCY_BUCKETS`）：本服务正常在 5~25ms，
留出长尾观察空间。面板里把秒 ×1000 显示成 **ms**。

### `scrape_interval` 为什么是 15s

写在 `prometheus/prometheus.yml` 的注释里，摘要：

- **分辨率**：面板用 `rate(...[1m])`，窗口内至少要 4 个样本才稳定 → 15s 刚好 4 个点。
  改成 60s 就只剩 1 个点，`rate(1m)` 直接算不出来，P95 也会滞后一分钟以上。
- **开销**：本服务约 200 条时间序列，15s ≈ 115 万样本/天，单机每天几 MB，可忽略；
  降到 5s 则样本量 ×3，而这种服务的秒级抖动没有可解读的信息，纯浪费。
- **惯例**：15s 是 Prometheus 默认值，协作成本最低。
- 经验法则：`scrape_interval ≤ 告警评估窗口 / 4`，且 `≥ 单次抓取耗时 × 10`。

### 面板与解读（Service overview）

| 面板 | 单位 | 怎么读 |
| --- | --- | --- |
| 服务可用性 (up) | UP/DOWN | 0 = 抓不到 `/metrics`，服务或网络出问题 |
| 当前 QPS | req/s | 掉到 0 = 没流量或服务挂了；暴涨 = 上游异常重试 |
| P95 延迟 | ms | 上涨 = 变慢（资源不足 / 批量变大 / 依赖变慢） |
| 错误率 (5m) | % | 4xx+5xx 占比。**先看下面的状态码拆分再下结论** |
| QPS（按接口） | req/s | 定位是哪个端点在被打 |
| 请求延迟分位数 | ms | P99 与 P50 拉开 = 长尾问题，通常是 GC/排队/大批量 |
| 请求速率（按状态码） | req/s | 422=调用方参数错，503=模型没加载，5xx=服务端异常 |
| 错误率（时间序列） | % | 看趋势和突变点 |
| 并发处理中的请求数 | 个 | 持续 >0 说明在排队 |
| 按接口的 P95 延迟 | ms | 哪个端点慢 |

## 二、模型指标（预测还可不可信）

### 暴露了哪些指标

| 指标 | 类型 | 标签 | 用途 |
| --- | --- | --- | --- |
| `model_predictions_total` | Counter | `predicted_class` | 各类别预测数；**类别结构突变是最早的信号** |
| `model_prediction_confidence` | Histogram | `le` | 置信度分布；低置信度占比上升 = 模型开始拿不准 |
| `model_prediction_batch_size` | Histogram | `le` | 每次请求的样本数 |
| `model_inference_duration_seconds` | Histogram | `le` | 纯推理耗时（不含 HTTP） |
| `model_loaded` | Gauge | — | 1=已加载，0=降级 |
| `model_info` | Gauge | `run_id` / `model_uuid` / `git_commit` / `model_uri` | 当前跑的是哪个模型，可一路追回训练 run 与代码 commit |

### 面板与解读（Model quality & drift signals）

- **预测速率（按类别）/ 类别占比**：与训练集分布对照。某一类突然消失或翻倍，
  基本可以断定输入变了——实测中给花瓣特征加 +2cm 偏移后，`setosa` 的预测数在 3 分钟内**归零**。
- **置信度分位数 / 低置信度占比（<0.8）**：漂移的早期指标。正常流量下 P05 ≈ 0.98，
  偏移流量下 P05 掉到 0.83。
- **纯推理耗时 P95**：与系统 P95 对比。实测 23.4ms vs 24.1ms，说明耗时几乎全在模型上，
  HTTP 层不是瓶颈。
- **当前模型 / 训练 commit**：换了 `MODEL_URI` 重启后这里会变，用来解释「为什么曲线从某一刻起不一样了」。

> ⚠️ 这些实时信号只能**提示**「输出看起来变了」。**哪个特征漂了、漂了多少**，
> 以 `drift_check.py` 的 Evidently 报告为准——那是批处理，不是实时。

## 三、数据漂移检测（Evidently 批处理）

### 三个脚本的分工

| 脚本 | 作用 |
| --- | --- |
| `make_reference.py` | 生成参考数据：直接复用第一阶段 `train.py` 的训练集切分，可选地用模型打分产出 `prediction` 基线列 |
| `simulate_traffic.py` | 打流量（喂系统指标）+ 记录这批线上数据（喂漂移检测）；`--mode drift` 人为制造特征偏移 |
| `drift_check.py` | 对比参考 vs 当前，产出 HTML 报告 + 两份 JSON + 一张逐特征结论表 |

### 阈值怎么定（不使用"默认值但不解释"）

| 判定项 | 方法 | 阈值 | 含义与理由 |
| --- | --- | --- | --- |
| 单个数值特征**是否**漂移 | Kolmogorov–Smirnov 检验的 **p 值** | `p < 0.05` | 「两批样本来自同一分布」这个原假设被拒绝的显著性水平。Evidently 的默认行为会随样本量切换方法（≤1000 行用 K-S p 值，>1000 行改用 Wasserstein 距离），本脚本**显式固定为 K-S**，避免样本量一变结论口径就悄悄变了 |
| 漂移**程度** | **Wasserstein 距离**（按参考集标准差归一化） | 0.1 作参考线 | 回答「漂了多少」。p 值只说差异是否显著，**样本量足够大时再小的差异也会显著**，所以必须配着幅度一起看。此值不参与是否漂移的判定 |
| 数据集**整体**漂移 | 漂移特征占比 | `≥ 0.5` | 4 个特征里至少 2 个漂了才算「整体漂移」。单个特征漂移仍逐条列出，不会被吞掉 |
| **模型输出**漂移 | prediction 列的 **Jensen–Shannon 距离** | `≥ 0.1` | 距离型指标，对「某个类别在一侧完全没出现」更稳健，且直接给出幅度 |

全部可用命令行覆盖：`--num-method / --num-threshold / --magnitude-threshold /
--cat-method / --cat-threshold / --drift-share`。

两个统计陷阱脚本会主动提示：当前样本 <30 行时结论不稳定；>5000 行时 p 值几乎必然显著，
应以幅度为主要依据。

### 报告不是一个布尔值

`drift_check.py` 的 stdout 会打出逐特征结论（实测输出，见下一节），HTML 报告里有
每个特征的参考/当前分布对比图。同时产出两份 JSON：

| 文件 | 结构 | 用途 |
| --- | --- | --- |
| `reports/drift.json` | Evidently 原生 `Report.run(...).dict()` | 保留全部原始指标 |
| `reports/drift_summary.json` | 精简：`dataset_drift` + `drifted_features[]`（含 method/score/threshold/magnitude/均值变化） | 给告警/自动化消费，第五阶段 `alert_notifier.py` **已实测可直接读取** |

---

## 四、如何人为制造漂移并验证（验收 3 的完整操作）

### 步骤

```bash
cd phase4-mlops

# ① 基线：正常流量 → 应判定「未漂移」
python monitoring/simulate_traffic.py --requests 400 --batch-size 4 --delay 0.22 \
    --error-rate 0.08 --out monitoring/current_data.csv
python monitoring/drift_check.py --current monitoring/current_data.csv

# ② 人为制造特征偏移：花瓣长度 +2.0cm、花瓣宽度 +0.8cm
#    （选花瓣特征是因为它对鸢尾花分类最有区分度，偏移后预测分布也会跟着变）
python monitoring/simulate_traffic.py --mode drift --requests 250 --batch-size 4 \
    --delay 0.15 --out monitoring/current_drifted.csv

# ③ 重新检测 → 应精确指出是哪两个特征、偏移多大
python monitoring/drift_check.py --current monitoring/current_drifted.csv \
    --output-html reports/drift_report_drifted.html \
    --output-json reports/drift_drifted.json \
    --summary-json reports/drift_summary_drifted.json \
    --fail-on-drift

open reports/drift_report_drifted.html
```

自定义偏移：`--drift-feature sepal_width --drift-shift 1.5`（可重复指定多个特征）。

### 实测输出（②③ 步）

```
【输入特征漂移】
特征                  检验                 p 值      阈值    漂移     幅度(W)      参考均值      当前均值      均值变化
------------------------------------------------------------------------------------------------
petal length (cm)   ks            5.19e-28    0.05     是     1.147     3.770     5.790    +53.6%
petal width (cm)    ks           4.738e-18    0.05     是     1.062     1.205     2.011    +66.9%
sepal width (cm)    ks              0.3309    0.05     否     0.073     3.048     3.057     +0.3%
sepal length (cm)   ks              0.8843    0.05     否     0.058     5.842     5.853     +0.2%

结论：2/4 个特征漂移（占比 50%）→ 整体漂移
最严重：petal length (cm)（p=5.19e-28，Wasserstein=1.147，均值 3.77 → 5.7902）

【模型输出漂移】（预测类别分布，与输入特征漂移分开看）
  jensenshannon 距离 = 0.3765（阈值 0.1） → 漂移
  参考占比：{'virginica': 35.0, 'setosa': 33.33, 'versicolor': 31.67}
  当前占比：{'virginica': 67.5, 'versicolor': 32.5}
```

对照组（正常流量）实测为 `0/4 个特征漂移`，JS 距离 0.0099 → 未漂移。
**两个没被人为偏移的特征仍然判定为「否」**，说明报告没有草木皆兵。

`--fail-on-drift` 时退出码为 **3**（实测），可以直接挂到定时任务或 CI 里。

---

## 五、验收结果（全部实测）

### 验收 1 · Prometheus `/targets` 显示服务 UP ✅

```bash
docker compose up -d --build
curl -s 'http://localhost:9090/api/v1/targets?state=active' | python3 -m json.tool | grep -E '"health"|"job"'
# 或浏览器打开 http://localhost:9090/targets
```

实测：

```
job=iris-inference   instance=api:8000        health=UP   err=-
job=prometheus       instance=localhost:9090  health=UP   err=-
```

### 验收 2 · 压测后 QPS / 延迟面板出现变化 ✅

```bash
python monitoring/simulate_traffic.py --requests 400 --batch-size 4 --delay 0.22 --error-rate 0.08
# 打开 http://localhost:3000 → MLOps → Service overview（时间范围选 Last 15 minutes）
```

压测中实测（直接查 Prometheus，面板用的是同样的 PromQL）：

| 面板 | 表达式 | 压测前 | 压测中 |
| --- | --- | --- | --- |
| 当前 QPS | `sum(rate(http_requests_total[1m]))` | 0.022（只有健康检查） | **4.00 req/s** |
| QPS（按接口） | `sum by (handler) (rate(...))` | /predict = 0 | **/predict = 3.96 req/s** |
| P95 延迟 | `histogram_quantile(0.95, …) * 1000` | 4.75 ms | **24.06 ms** |
| 错误率 | `100 * 4xx5xx / all` | 0 % | **8.71 %**（与 `--error-rate 0.08` 一致） |
| 按状态码 | `sum by (status) (rate(...))` | 200=0.022 | **200=3.64 / 422=0.36** |

模型面板同时实测到：三类预测速率均 >2/s、置信度 P50=0.984、低置信度占比 7.8%、
纯推理 P95=23.4ms、`model_info` 显示 `model_uuid=m-25c18ca9…`。

### 验收 3 · 制造偏移后报告标出具体特征与幅度 ✅

见上一节实测输出：明确给出 `petal length (cm)` / `petal width (cm)` 两个特征名、
p 值、Wasserstein 幅度、均值从多少变到多少、变化百分比。

### 验收 4 · 「错误率上升但 Evidently 没检测到漂移」怎么解释

见下一章，这正是两类监控必须分开的原因。

---

## 六、错误率上升，但 Evidently 没检测到漂移 —— 可能是什么原因？

先记住一句话：**错误率是「请求有没有被正确处理」，漂移是「被成功处理的数据长什么样」。
一个请求如果失败了，它根本不会进入漂移分析的样本里。** 可能的原因按可能性排序：

### A. 根本不是模型的问题（最常见）

1. **调用方传参错了（4xx / 422）**。字段缺失、类型不对、批量超过 1000 条——
   请求在 Pydantic 校验阶段就被挡下，**没进模型**，Evidently 自然看不到。
   *怎么确认*：看「请求速率（按状态码）」面板，如果涨的是 422，去看请求样例。
2. **服务端故障（5xx）**。未预期异常、依赖超时、OOM、磁盘满。
   *怎么确认*：`model_loaded` 是否为 1，容器日志里的 traceback，`docker stats` 看资源。
3. **模型没加载（503）**。`MODEL_URI` 配错、artifact 卷没挂上、模型文件损坏。
   服务此时对**所有**请求返回 503，错误率 100%，而漂移分析连数据都收不到。
   *怎么确认*：`/health` 的 `detail` 字段直接会说原因。
4. **部署/配置变更**。刚换了镜像或模型版本、并发/资源被调小。
   *怎么确认*：Grafana 上「当前模型 (model_uuid)」和「训练 commit」面板在那个时间点变了没有。

### B. 是数据的问题，但漂移检测「看不见」（幸存者偏差）

5. **上游改了数据格式**：字段改名（`petal_len` vs `petal_length`）、单位从 cm 变 mm 但类型仍是数字、
   多送了一个字段。**改名/多字段会被 schema 拒成 422**，这些样本压根不在你的当前数据 CSV 里——
   错误率涨了，漂移报告却一片绿。这是最容易困住人的一种。
   *怎么确认*：抓一条失败请求的原始 body 看看。
6. **采样口径不一致**：当前数据只记录了成功请求（本项目的 `simulate_traffic.py` 就是这样），
   失败请求天然不在分布里。
   *怎么确认*：核对当前数据行数与 `http_requests_total{status="200"}` 的增量是否对得上。

### C. 时间窗口 / 阈值的问题

7. **窗口没对齐**：Grafana 看的是最近 5 分钟，漂移报告跑的可能是昨天的批次，还没覆盖到这段流量。
   *怎么确认*：看 `drift_summary.json` 里的 `generated_at` 和当前数据的时间范围。
8. **确实漂了，只是没过阈值**：单个特征漂移但占比没到 `drift_share=0.5`，
   或者 p 值刚好在 0.05 边上、幅度很小。
   *怎么确认*：看逐特征表格，别只看最后那句「整体漂移/未漂移」。
9. **样本量太小**：当前数据只有几十行，K-S 检验没有把握度，脚本会打 ⚠️ 提示。

### 反过来也要会看：Evidently 报漂移，但 Grafana 全绿

服务健康、延迟正常、零错误，但输入分布变了——这是**更危险**的情况：模型在安静地把
错误答案以 200 返回。系统指标永远发现不了它，只能靠漂移报告 + 模型指标
（置信度下降、类别分布突变）。本项目实测的偏移场景正是如此：偏移流量下
250 个请求**全部 200**、错误率 0，但 `setosa` 预测归零、两个花瓣特征显著漂移。

### 排障顺序（照着走）

```
错误率↑
 ├─ 看「请求速率（按状态码）」面板
 │   ├─ 422 为主 → 调用方/上游数据格式问题 → 抓失败请求 body（漂移报告帮不上忙）
 │   ├─ 503 为主 → 模型没加载 → 看 /health detail、model_loaded、卷挂载
 │   └─ 5xx 为主 → 服务端异常 → 看容器日志、资源、依赖
 ├─ 看「当前模型 / 训练 commit」面板 → 是不是刚换了模型版本
 └─ 都排除后，再看模型侧：置信度是否下降、类别分布是否突变 → 跑一次 drift_check.py
```

---

## 七、与第五阶段的衔接

`drift_check.py` 的 JSON 产物就是第五阶段告警脚本的输入：

```bash
# 已实测：精简摘要 JSON 可被 phase5 的告警脚本直接消费（dry-run 打印完整告警内容）
python ../phase5-mlops/alert_notifier.py \
  --drift-report reports/drift_summary_drifted.json --dry-run
```

> **已知问题（第五阶段代码，非本阶段）**：用 Evidently 原生的 `reports/drift.json`
> 走同一入口会抛
> `TypeError: DriftFinding.__init__() got an unexpected keyword argument 'mean_shift_pct'`。
> 原因是 `phase5-mlops/alert_notifier.py` 的 `DriftFinding.to_dict()`（第 125 行）
> 输出了派生属性 `mean_shift_pct`，而 `parse_evidently_report()`（第 520 行）又把
> `to_dict()` 的结果整个塞回构造函数。修法是构造前剔除该键，例如：
> `DriftFinding(**{k: v for k, v in {**finding.to_dict(), "magnitude": …}.items() if k != "mean_shift_pct"})`。
> 本阶段没有改动第五阶段的代码。

## 八、清理与运维

```bash
docker compose down            # 停止容器，保留 Prometheus/Grafana 数据卷
docker compose down -v         # 连同历史指标数据一起删除
docker compose logs -f api     # 看服务日志
curl -X POST http://localhost:9090/-/reload   # 改完 prometheus.yml 热加载，不用重启
```

生产化时要改的几件事（本阶段是本地演示栈，刻意从简）：

- Grafana 的 `admin/admin` 与匿名只读**必须**关掉，改用 SSO 或强密码 + Secret 管理；
- Prometheus 目前 `retention=7d`、无鉴权、无 Alertmanager——告警通道在第五阶段；
- 生产上 Prometheus/Grafana 通常是集群共享组件，服务侧只要暴露 `/metrics` 并被抓取即可，
  不需要每个服务自带一套；
- 漂移检查应该定时跑（cron / Airflow / GitHub Actions `schedule`），
  用 `--fail-on-drift` 的退出码 3 触发下游动作，而不是靠人记得跑。
