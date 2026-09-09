# Phase 2 — 容器化推理服务（FastAPI + Docker）

把第一阶段训练出的模型包装成 REST 服务。本阶段的两个重点：

1. **training / serving 预处理完全同源**——特征契约（列名、列序、dtype、标签解码）
   只有一份实现，放在 `shared/mlops_shared/features.py`，训练脚本和服务代码都是
   `import` 它，不存在第二份拷贝。
2. **构建过程可审计、可优化**——多阶段构建、非 root 运行、`.dockerignore` 精确裁剪，
   每一层的作用都写在 Dockerfile 注释里，体积优化有实测数据。

> 本文档中的所有命令与数字均为**实测**结果（Docker 29.7.2，linux/arm64，
> containerd 镜像存储；宿主机 macOS/arm64）。四条验收标准全部在容器中验证通过。

> **CI/CD**：本阶段的测试已接入 GitHub Actions，每次 push / PR 自动运行；流程说明见根目录 [README.md 的「CI/CD 流程说明」](../README.md#cicd-流程说明)。

## 目录结构

```
MLOps project/
├── shared/                       # 新增：训练与服务共用的代码（可安装包）
│   ├── pyproject.toml            #   name = mlops-shared，只依赖 pandas
│   └── mlops_shared/
│       ├── features.py           #   ★ 特征契约：唯一的预处理实现
│       ├── tracking.py           #   MLflow 路径/URI 解析（不 import mlflow）
│       └── errors.py             #   统一错误层次
├── phase1-mlops/                 # 第一阶段：训练（已改为从 shared 导入）
├── phase2-mlops/
│   ├── app/
│   │   ├── main.py               # FastAPI 应用入口（启动时一次性加载模型）
│   │   ├── schemas.py            # Pydantic 请求/响应 schema
│   │   └── model_loader.py       # 模型加载 + 预处理（调用 shared 的实现）
│   ├── Dockerfile                # 多阶段构建，逐层注释
│   ├── .dockerignore             # 构建上下文裁剪（根目录有指向它的符号链接）
│   ├── requirements.txt          # 运行时依赖
│   ├── requirements-dev.txt      # 仅测试用（镜像不安装）
│   ├── README.md
│   └── tests/test_api.py         # TestClient 测试 /health 与 /predict
└── .dockerignore -> phase2-mlops/.dockerignore
```

### 为什么根目录有个 `.dockerignore` 符号链接

镜像必须包含 `shared/`（服务要 `import mlops_shared`），因此**构建上下文是仓库根目录**。
而 Docker 只读取「上下文根目录」下的 `.dockerignore`。为了既满足本阶段的目录规范
（文件在 `phase2-mlops/.dockerignore`），又让 Docker 真的能读到它，根目录放了一个
指向它的符号链接——一份内容，两个位置可见。Windows 下若符号链接不可用，把
`phase2-mlops/.dockerignore` 复制到仓库根目录即可。

## 一、如何消除 training-serving skew

skew 的典型来源是「训练时用 DataFrame 的列名顺序 A，服务时手搓一个顺序 B」。
本项目的做法是把这件事变成不可能：

| 环节 | 代码位置 | 调用的函数 |
| --- | --- | --- |
| 训练前构造特征 | [phase1-mlops/train.py](../phase1-mlops/train.py) `load_dataset()` | `build_feature_frame()` |
| 离线预测 | [phase1-mlops/load_and_predict.py](../phase1-mlops/load_and_predict.py) `predict()` | `build_feature_frame()` |
| HTTP 预测 | [app/model_loader.py](app/model_loader.py) `prepare_features()` | `build_feature_frame()` |

`build_feature_frame()`（[shared/mlops_shared/features.py](../shared/mlops_shared/features.py)）负责：

- 把 API 字段名（`sepal_length`）翻译成模型列名（`sepal length (cm)`）——映射表
  `API_FIELD_TO_COLUMN` 同时被 Pydantic schema 使用，HTTP 契约与模型契约无法各说各话；
- 校验特征齐全、无多余列，缺列/多列直接报错；
- 统一 cast 成 `float64` 并**按训练时的列序重排**；
- 拒绝 NaN / inf。

另外两道保险：

- 训练时会核对 `sklearn` 的标签顺序是否仍与共享契约 `TARGET_NAMES` 一致，不一致就
  拒绝训练（否则服务会用错的类名解码预测结果）；
- 模型 artifact 里记录了 signature，可以直观看到服务发出的列名与训练完全一致：

```bash
grep -A3 '^signature' phase1-mlops/mlartifacts/models/*/artifacts/MLmodel | head -8
# inputs: '[{"type": "double", "name": "sepal length (cm)", ...
```

## 二、环境搭建（本地开发）

```bash
cd phase2-mlops
python3 -m venv .venv
source .venv/bin/activate                                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt     # 必须在 phase2-mlops/ 下执行
```

`requirements.txt` 的第一行是 `../shared`（相对路径按**当前工作目录**解析），所以
务必在 `phase2-mlops/` 目录下安装；Dockerfile 里也是先 `WORKDIR /build/phase2-mlops`
再 `pip install`，两边行为一致。

> 第一阶段的 `phase1-mlops/requirements.txt` 也新增了 `-e ../shared`。如果你的
> phase1 环境是本次改动之前建的，需要在 `phase1-mlops/` 下重新执行一次
> `pip install -r requirements.txt`，否则 `train.py` 会报 `No module named 'mlops_shared'`。

## 三、配置（全部通过环境变量，无硬编码 run_id）

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `MODEL_URI` | ✅ | 要服务的模型。支持：32 位 run_id、`runs:/<run_id>/model`、`models:/<name>/<version>`、本地模型目录。**未设置时服务不会崩，而是 `/health` 返回 503 并说明原因。** |
| `MLFLOW_TRACKING_URI` | 条件必填 | 仅当 `MODEL_URI` 是 `runs:/` / `models:/` 形式时需要，用于解析模型位置 |
| `MODEL_ARTIFACT_NAME` | ❌ | `MODEL_URI` 只给 run_id 时使用的 artifact 名，默认 `model` |
| `LOG_LEVEL` | ❌ | 默认 `INFO` |

模型只在**应用启动时加载一次**（FastAPI lifespan），请求路径上不再访问 tracking store。
`/health` 与 `/predict` 返回的 `loaded_at`、`model_uuid` 可用于验证这一点（见测试
`test_model_is_loaded_once_not_per_request`）。

## 四、本地运行（不用 Docker）

```bash
cd phase2-mlops
export MLFLOW_TRACKING_URI="sqlite:///$(cd ../phase1-mlops && pwd)/mlflow.db"
export MODEL_URI=<第一阶段的 run_id>          # 例如 111ad0c141ec462fb2cac8be10eb6ae9
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`GET /health`（本机实测输出）：

```json
{
  "status": "ok",
  "service": "iris-inference",
  "version": "0.2.0",
  "model": {
    "requested_uri": "111ad0c141ec462fb2cac8be10eb6ae9",
    "resolved_uri": "runs:/111ad0c141ec462fb2cac8be10eb6ae9/model",
    "run_id": "111ad0c141ec462fb2cac8be10eb6ae9",
    "model_uuid": "m-25c18ca9e1fd48a3a024eb16b304d526",
    "tracking_uri": "sqlite:////.../phase1-mlops/mlflow.db",
    "loaded_at": "2026-09-08T09:17:17.860894Z",
    "supports_confidence": true,
    "git_commit": "33a3c9f8f8999a98bd69e374393205e8d4cf00a9",
    "training_params": {"n_estimators": "100", "max_depth": "3", "random_state": "42", "test_size": "0.2"}
  },
  "detail": null
}
```

`git_commit` 与 `training_params` 直接来自第一阶段记录的 run，因此**一次线上预测可以
一路追到训练超参数和代码 commit**。

`POST /predict`：

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[
        {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2},
        {"sepal_length":6.7,"sepal_width":3.0,"petal_length":5.2,"petal_width":2.3}]}'
```

```json
{"model": {...}, "predictions": [
  {"label": 0, "label_name": "setosa",    "confidence": 1.0,
   "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}},
  {"label": 2, "label_name": "virginica", "confidence": 0.9877322260447259,
   "probabilities": {"setosa": 0.0, "versicolor": 0.012267773955273955, "virginica": 0.9877322260447259}}
]}
```

交互式文档：<http://127.0.0.1:8000/docs>。

### 状态码语义

| 场景 | 状态码 | 响应 |
| --- | --- | --- |
| 正常预测 / 模型已加载 | 200 | 业务响应 |
| 字段缺失、类型错误、多余字段、值越界、批量为空或超过 1000 条 | **422** | `{"error":"validation_error","detail":[{"location":[...],"message":...}]}` |
| 通过了 schema 但违反特征契约 | 422 | `{"error":"feature_contract_error", ...}` |
| 模型未加载（配置错误/artifact 丢失） | 503 | `{"error":"model_unavailable","detail":"<可读原因>"}` |
| 未预期异常 | 500 | `{"error":"internal_error","detail":"Internal server error. See service logs for details."}` |

原始异常和 traceback 只进服务日志，不会返回给客户端。实测：

```
missing field  -> 422    {"error":"validation_error","detail":[{"location":["body","instances",0,"petal_width"],"message":"Field required","type":"missing"}]}
extra field    -> 422
wrong type     -> 422
```

## 五、Docker 构建与运行

### 构建（必须在仓库根目录，`-f` 指定 Dockerfile）

```bash
cd "/path/to/MLOps project"
docker build -f phase2-mlops/Dockerfile -t iris-inference:phase2 .
docker images iris-inference:phase2          # 镜像大小
docker history iris-inference:phase2         # 逐层大小，核对下表
```

实测：冷构建 **37.8s**（依赖下载+安装 19.5s），改动 `app/` 代码后重建只需数秒
（依赖层命中缓存）。构建上下文实测只传 **84 KB / 10 个文件**（见第六节）。

### 运行方式 A（推荐）：挂载导出的模型目录，容器内不需要 tracking store

```bash
# 1) 从第一阶段导出模型（已实测）
cd "/path/to/MLOps project"
MLFLOW_TRACKING_URI="sqlite:///$PWD/phase1-mlops/mlflow.db" \
  mlflow artifacts download \
    --artifact-uri "runs:/<RUN_ID>/model" \
    --dst-path phase2-mlops/export/v1
# 产物位于 phase2-mlops/export/v1/model（约 1.7 MB）

# 2) 起容器：模型以只读卷挂进去，MODEL_URI 指向容器内路径
docker run --rm -p 8000:8000 \
  -e MODEL_URI=/models/v1 \
  -v "$PWD/phase2-mlops/export/v1/model:/models/v1:ro" \
  iris-inference:phase2

curl -s http://localhost:8000/health | python3 -m json.tool
```

这种方式下 `/health` 的 `run_id` 为 `null`（模型脱离了 run 上下文），模型标识由
`model_uuid`（读自 `MLmodel`）给出。容器内实测输出：

```json
{"status":"ok","service":"iris-inference","version":"0.2.0",
 "model":{"requested_uri":"/models/v1","resolved_uri":"/models/v1","run_id":null,
          "model_uuid":"m-25c18ca9e1fd48a3a024eb16b304d526","tracking_uri":null,
          "loaded_at":"2026-09-08T13:09:36.353827Z","supports_confidence":true,
          "git_commit":null,"training_params":{}},"detail":null}
```

### 运行方式 B（开发便利）：容器内直接用 run_id 访问本地 sqlite store

第一阶段的 artifact 位置在数据库里记录为**宿主机绝对路径**
（`file:///.../phase1-mlops/mlartifacts`），所以必须把它挂到容器内**同一绝对路径**，
否则能查到 run 却读不到 artifact：

```bash
docker run --rm -p 8001:8000 \
  -e MODEL_URI=<RUN_ID> \
  -e MLFLOW_TRACKING_URI="sqlite:///$PWD/phase1-mlops/mlflow.db" \
  -v "$PWD/phase1-mlops:$PWD/phase1-mlops" \
  iris-inference:phase2
```

（sqlite 后端初始化时可能写 journal，因此这里不加 `:ro`。）
**已实测可用**：容器日志
`Loading model from runs:/111ad0c1.../model (tracking store: sqlite:////.../phase1-mlops/mlflow.db)`
→ `Model ready: run_id=111ad0c1... classes=[0, 1, 2] confidence=True`，
`/health` 返回 200 且带上了 `git_commit: 33a3c9f8...` 与 `training_params`
（即容器内也能从一次预测追溯到训练超参数与代码 commit）。
生产环境不要这样做——第三阶段起应改为 MLflow tracking server + 对象存储 artifact
（`mlflow server --serve-artifacts`，artifact URI 变成 `mlflow-artifacts://`，与宿主机路径解耦）。

### 运行方式 C：模型注册表

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_URI="models:/iris-classifier/3" \
  -e MLFLOW_TRACKING_URI="http://mlflow.internal:5000" \
  iris-inference:phase2
```

## 六、镜像体积：优化手段与实测数据

采用的优化（对应 Dockerfile 里的 `[builder n/5]` / `[runtime n/8]` 注释）：

| # | 手段 | 效果 |
| --- | --- | --- |
| 1 | **多阶段构建**：builder 装依赖到 `/opt/venv`，final 只 `COPY --from=builder /opt/venv` | pip 缓存、构建目录、`shared/` 源码都不进最终镜像 |
| 2 | **`python:3.10-slim` 而非完整版** | slim ≈ 125 MB，完整版 `python:3.10` ≈ 1.0 GB（Docker Hub 公开数据，未在本机核实） |
| 3 | **`mlflow-skinny` 替代 `mlflow`** | site-packages 602 MB → 372 MB（**实测 −230 MB**）：去掉 pyarrow(125M)、matplotlib(26M)、fontTools(18M)、flask/gunicorn/docker 等推理不用的依赖 |
| 4 | **清理 `__pycache__` / `*.pyc`** | 372 MB → 264 MB（**实测 −108 MB**）；`PYTHONDONTWRITEBYTECODE=1` 防止运行时再生成 |
| 5 | **`.dockerignore` 裁剪上下文** | 仓库 618 MB → 进入上下文 **84 KB / 10 个文件**（实测）：排除 `.venv/`(608M)、`phase1-mlops/` 及其 `mlflow.db`/`mlartifacts`(9.5M)、`.git/`、`__pycache__`、tests、文档、导出的模型 |
| 6 | **测试依赖分离到 `requirements-dev.txt`** | 镜像不含 pytest/httpx |
| 7 | **依赖层与代码层分离**（先 COPY requirements，再 COPY app） | 改业务代码不触发重新安装依赖，构建缓存命中 |

### 构建前后对比（实测，linux/arm64）

「前」= 一个刻意不做优化的等价镜像：完整 `python:3.10` 基础镜像 + 完整 `mlflow` +
单阶段构建 + 保留 pip 缓存 + root 运行（`/tmp/naive/Dockerfile`，见下方复现命令）。
两个镜像功能相同，`/health` 都返回 200。

| 指标 | 未优化（`iris-inference:naive`） | 本项目（`iris-inference:phase2`） | 变化 |
| --- | --- | --- | --- |
| `docker images` DISK USAGE | **3.0 GB** | **721 MB** | −76% |
| CONTENT SIZE（推送/拉取的压缩体积） | 811 MB | **149 MB** | −82% |
| 各层未压缩之和（`docker history`） | 2192 MB | 572 MB | −74% |
| 镜像内 site-packages | 771 MB | 382 MB | −389 MB |
| 镜像内残留 pip 缓存 | 193 MB | **0** | 清零 |
| 镜像内残留 `__pycache__` 目录数 | 数千 | **0** | 清零 |
| 运行用户 | `uid=0(root)` | `uid=10001(appuser)` | 非特权 |
| 冷构建耗时 | 51.8s | 37.8s | — |

复现命令：

```bash
# 本项目镜像
docker build -f phase2-mlops/Dockerfile -t iris-inference:phase2 .

# 未优化对照镜像
mkdir -p /tmp/naive && cp -R shared /tmp/naive/shared && cp -R phase2-mlops/app /tmp/naive/app
cat > /tmp/naive/Dockerfile <<'EOF'
FROM python:3.10
WORKDIR /srv
COPY shared/ /build/shared/
COPY app/ ./app/
RUN pip install /build/shared mlflow scikit-learn pandas fastapi "uvicorn[standard]" skops
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
docker build -t iris-inference:naive /tmp/naive

docker images iris-inference                      # 对比体积
docker history iris-inference:phase2              # 逐层核对（venv 层 400 MB）
docker run --rm --entrypoint sh iris-inference:phase2 -c \
  "du -sh /opt/venv/lib/python3.10/site-packages; find /opt/venv -name __pycache__ | wc -l"
```

> **Docker 29 的体积口径**：本版本使用 containerd 镜像存储，`docker images` 输出
> `DISK USAGE`（含快照开销）与 `CONTENT SIZE`（压缩内容，约等于推送体积）两列，
> 和旧版单一 `SIZE` 不可直接比较；上表两种口径都给了。

依赖体积的宿主机侧实测（macOS/arm64 wheel，`du -sh site-packages`，用于说明第 3、4 项优化的来源）：

| 依赖组合 | 清理前 | 清理 `__pycache__` 后 |
| --- | --- | --- |
| 完整 `mlflow` + sklearn + pandas + fastapi + uvicorn | 602 MB | 474 MB |
| **本项目 `requirements.txt`**（mlflow-skinny + skops + sqlalchemy + …） | 372 MB | **264 MB** |

注：容器内（linux/arm64 wheel）同一份依赖是 382 MB，比宿主机 macOS/arm64 的 264 MB 更大，
主要来自 `scipy.libs`/`numpy.libs`（manylinux wheel 会把 BLAS 等共享库打进包内，
实测各 27 MB / 26 MB）。**镜像体积主要由 scipy+sklearn+pandas+numpy 这四个科学计算
包决定（约 240 MB），不是应用代码**；要再瘦身只能换更小的模型运行时（如导出 ONNX
后只装 onnxruntime），那属于后续阶段的取舍。

一个已验证的坑：**不要**在瘦身时删除 site-packages 里的 `tests/` 目录。
反序列化模型时 numpy 会 `import numpy._core.tests`，删掉后模型加载直接失败
（本机实测：`ModuleNotFoundError: No module named 'numpy._core.tests'`，`/health` 变 503）。
Dockerfile 注释里也标注了这一点。

## 七、验收验证

### 验收 1：构建成功且体积合理 ✅ 已实测

`docker build` 成功（37.8s），镜像 721 MB / 压缩 149 MB，相比未优化的等价镜像
（3.0 GB / 811 MB）减少 76%–82%，7 项优化手段与实测数据见上一节。

### 验收 2：容器起来后 `/health` 返回 200 且包含模型标识 ✅ 已实测

```bash
docker run -d --name iris -p 8000:8000 \
  -e MODEL_URI=/models/v1 \
  -v "$PWD/phase2-mlops/export/v1/model:/models/v1:ro" \
  iris-inference:phase2

curl -i http://localhost:8000/health          # 期望 HTTP/1.1 200 OK
curl -s http://localhost:8000/health | python3 -c "import json,sys; m=json.load(sys.stdin)['model']; print(m['resolved_uri'], m['model_uuid'], m['run_id'])"
docker inspect --format='{{.State.Health.Status}}' iris    # Dockerfile 内置 HEALTHCHECK
docker exec iris id                                        # 期望 uid=10001(appuser)，不是 root
```

实测结果：

```
GET /health -> 200        model_uuid=m-25c18ca9e1fd48a3a024eb16b304d526（方式 A）
docker exec iris id       uid=10001(appuser) gid=10001(appuser)      # 非 root
docker inspect ... Health  healthy                                    # 内置 HEALTHCHECK
```

用方式 B（`runs:/` + 挂载 store）启动时，`/health` 还会带上 `run_id`、
`git_commit=33a3c9f8...` 和 `training_params`，模型标识更完整。

**容器内的输入校验实测**：缺字段 / 类型错 / 多字段 / 空批量 → 全部 **422**；
未设置 `MODEL_URI` 时 `/health` 503 `degraded`、`/predict` 503 `model_unavailable`，
容器保持 `Up` 但被标记 `unhealthy`（readinessProbe 语义），不会 crash-loop。

### 验收 3：`/predict` 结果与直接用 Python 脚本加载同一模型完全一致 ✅ 已实测

```bash
# ① HTTP 预测
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"instances":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2},
                    {"sepal_length":6.7,"sepal_width":3.0,"petal_length":5.2,"petal_width":2.3}]}'

# ② 离线预测（第一阶段脚本，同一个 run_id）
printf 'sepal_length,sepal_width,petal_length,petal_width\n5.1,3.5,1.4,0.2\n6.7,3.0,5.2,2.3\n' > /tmp/sample.csv
cd phase1-mlops
python load_and_predict.py --run-id <RUN_ID> --input-csv /tmp/sample.csv
```

自动对比（labels + 置信度逐位比较）：

```bash
cd phase2-mlops
MLFLOW_TRACKING_URI="sqlite:///$(cd ../phase1-mlops && pwd)/mlflow.db" python - <<'PY'
import json, urllib.request, mlflow, mlflow.sklearn
from mlops_shared.features import build_feature_frame

rows = [{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2},
        {"sepal_length":6.7,"sepal_width":3.0,"petal_length":5.2,"petal_width":2.3}]
req = urllib.request.Request("http://127.0.0.1:8000/predict",
        data=json.dumps({"instances": rows}).encode(),
        headers={"Content-Type": "application/json"})
api = json.load(urllib.request.urlopen(req))
http_labels = [p["label"] for p in api["predictions"]]
http_conf   = [p["confidence"] for p in api["predictions"]]

model = mlflow.sklearn.load_model(f"runs:/{api['model']['run_id']}/model")
frame = build_feature_frame(rows)
off_labels = [int(v) for v in model.predict(frame)]
off_conf   = [float(max(r)) for r in model.predict_proba(frame)]
print("http    :", http_labels, http_conf)
print("offline :", off_labels, off_conf)
print("identical:", http_labels == off_labels and http_conf == off_conf)
PY
```

实测输出（左侧为**容器**里的服务，右侧为宿主机直接用 Python 加载同一模型）：

```
container /predict : [0, 2, 1] [1.0, 0.9877322260447259, 0.9680256896633047]
offline python     : [0, 2, 1] [1.0, 0.9877322260447259, 0.9680256896633047]
identical          : True
```

置信度浮点值逐位相同——预处理同源（`build_feature_frame`）+ 同一份 artifact，
跨「容器 linux/arm64」与「宿主机 macOS/arm64」都没有偏差。
同一断言也固化在测试里：`test_predict_matches_offline_prediction_for_the_same_model`。

### 验收 4：改 `MODEL_URI` 重启即可换模型版本（零代码改动）✅ 已实测

```bash
# 训练第二个版本，拿到新的 run_id
cd phase1-mlops && python train.py --n-estimators 300 --max-depth none --random-state 2024

# 导出并换容器（唯一变化是 -e MODEL_URI / 挂载路径）
MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db" mlflow artifacts download \
  --artifact-uri "runs:/<NEW_RUN_ID>/model" --dst-path ../phase2-mlops/export/v2
cd .. && docker rm -f iris && docker run -d --name iris -p 8000:8000 \
  -e MODEL_URI=/models/v2 \
  -v "$PWD/phase2-mlops/export/v2/model:/models/v2:ro" \
  iris-inference:phase2
curl -s http://localhost:8000/health | python3 -c "import json,sys; print(json.load(sys.stdin)['model'])"
```

容器实测（同一个镜像，只改 `-e MODEL_URI` 与挂载，未动任何代码）：

```
MODEL_URI=/models/v1  -> model_uuid m-25c18ca9e1fd48a3a024eb16b304d526, 预测 virginica 0.9877322260447259
MODEL_URI=/models/v2  -> model_uuid m-768992bee52f4664836de3006e09986d, 预测 virginica 0.98
```

方式 B（run_id 形式）同样实测通过，`/health` 会显示两个 run 各自的
`n_estimators=100 / max_depth=3` 与 `n_estimators=300 / max_depth=None`。

测试 `test_model_uri_switches_versions_without_code_changes` 固化了这一行为。

## 八、测试

```bash
cd phase2-mlops
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -q          # 实测：22 passed
```

覆盖内容：

- `/health` 200 + 模型标识（run_id / model_uuid / git_commit / 训练超参数）；
- 模型加载失败时 `/health` 503 `degraded`、`/predict` 503 `model_unavailable`，且不泄漏 traceback；
- `/predict` 标签 + 置信度（置信度等于预测类概率、概率和为 1）；
- **HTTP 与离线预测逐位一致**（验收 3）；
- 模型只加载一次（两次请求 `loaded_at` / `model_uuid` 相同）；
- **换 `MODEL_URI` 即换模型版本**（验收 4）；
- 本地模型目录形式的 `MODEL_URI`（容器推荐用法），无需 tracking store；
- 8 类非法输入全部 422（缺字段 / 类型错 / 多字段 / 值越界 / 空批量 / 超限批量 /
  顶层 key 错 / 容器类型错），且 422 不会打到模型上；
- `MODEL_URI` 未设置、无法解释时给出明确配置错误。

测试会 `import` 第一阶段的 `train.py` 来现场训练两个临时模型，因此顺带验证了
「phase1 训练 → phase2 加载」这条链路是通的。

## 九、生产注意事项

- **非 root**：镜像创建 `appuser`(uid/gid 10001) 并在 `USER appuser` 之后才启动服务；
  `docker exec <c> id` 可验证。
- **单 worker**：模型在进程内加载一次；要提升吞吐用多副本或 `--workers N`
  （每个 worker 各持一份模型，内存×N）。
- **就绪语义**：模型加载失败时进程不退出，而是 `/health` 返回 503。K8s 里用它做
  readinessProbe，容器不会进入 crash-loop，运维能直接从 `/health` 的 `detail` 看到原因。
- **HEALTHCHECK**：slim 镜像没有 curl/wget，探针用 `python -c urllib.request` 实现。
- **批量上限**：`instances` 最多 1000 条（`app/schemas.py: MAX_BATCH_SIZE`），防止单请求打满 worker。
- **基础镜像 CVE（实测）**：`docker scout quickview` 结果

  | 镜像 | 漏洞 |
  | --- | --- |
  | `python:3.10-slim`（基础镜像） | 0C **3H** 8M 25L |
  | `iris-inference:phase2`（本项目） | 0C **3H** 8M 25L |

  两者计数完全相同，说明**本项目新增的 60+ 个 Python 依赖没有引入任何新的
  high 级漏洞**，3 个 high 全部来自基础镜像自带的 `zlib` / `wheel 0.45.1` /
  `jaraco-context 5.3.0`。`python:3.10-slim` 已是该 tag 的最新版（`scout
  recommendations` 显示 "up to date"），试过在 runtime 层卸载基础镜像自带的
  pip/setuptools/wheel，扫描结果无变化，故未采纳。
  真正的修复路径是升基础镜像：scout 实测 `python:3.12-slim` 为 0C **1H** 6M 25L
  （少 2 个 high，且体积略小）。本阶段的约束是 `python:3.10-slim`，故保持不变；
  建议在第四阶段 CI 里把「基础镜像升级 + 镜像扫描门禁」做成流水线的一步。
  日常缓解：按 digest 固定（`python:3.10-slim@sha256:...`）以便可复现地升级，
  并定期重建镜像。
- 下一阶段可加：`--read-only` 根文件系统 + `tmpfs`、非特权端口、CI 中接入镜像扫描
  （trivy / docker scout）并构建推送镜像。
