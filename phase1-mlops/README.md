# Phase 1 — 可追踪、可复现的训练基线

MLOps 全链路学习项目的第一阶段：用 scikit-learn 在 iris 数据集上训练 RandomForest 基线，
所有超参数、指标、模型 artifact 与 **git commit** 都记录进本地 MLflow，并提供一个独立的
模型加载脚本作为后续阶段（Docker / FastAPI / CI-CD / 监控）的稳定接口。

## 目录结构

```
phase1-mlops/
├── train.py              # 训练入口，argparse 管理全部超参数
├── load_and_predict.py   # 通过 run_id 加载模型并预测（后续阶段的模型加载契约）
├── requirements.txt
├── README.md
├── .gitignore            # 忽略本地 tracking store 与 artifacts
└── tests/
    └── test_train.py

../shared/                # 与第二阶段共用的特征契约（第二阶段引入）
└── mlops_shared/
    ├── features.py       # 唯一的预处理实现：列名/列序/dtype/标签解码
    ├── tracking.py       # tracking URI 与 model URI 解析
    └── errors.py
```

> 第二阶段（推理服务）会 `import` 同一份 `mlops_shared.features`，所以特征处理逻辑
> 只有一处实现，训练与线上不会漂移。详见 [phase2-mlops/README.md](../phase2-mlops/README.md)。

> **CI/CD**：本阶段的测试已接入 GitHub Actions，每次 push / PR 自动运行；流程说明见根目录 [README.md 的「CI/CD 流程说明」](../README.md#cicd-流程说明)。

## 1. 环境搭建

要求 Python 3.10+。

```bash
cd phase1-mlops
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # 需在 phase1-mlops/ 下执行：含 `-e ../shared`
```

验证依赖就位：

```bash
python -c "import mlflow, sklearn, pandas, mlops_shared; print(mlflow.__version__, sklearn.__version__, mlops_shared.__version__)"
```

## 2. 快速开始

```bash
# 训练一次
python train.py --n-estimators 100 --max-depth 3 --random-state 42

# 用上一条命令输出的 run_id 加载模型并预测
python load_and_predict.py --run-id <run_id>
```

`train.py` 输出示例：

```
Run logged successfully.
  tracking_uri : sqlite:////abs/path/phase1-mlops/mlflow.db
  experiment   : iris-baseline
  run_id       : fd7b38aea86c4849b8bd8d1035c5f050
  accuracy     : 0.966667
  f1_macro     : 0.966583
  model_uri    : runs:/fd7b38aea86c4849b8bd8d1035c5f050/model
```

### train.py 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--n-estimators` | 正整数 | `100` | 森林中树的数量 |
| `--max-depth` | 正整数或 `none` | `none` | 最大树深，`none` 表示生长到叶子纯净 |
| `--random-state` | 整数 | `42` | 同时驱动数据切分与森林，决定可复现性 |
| `--test-size` | (0, 1) 浮点 | `0.2` | 测试集比例（分层切分） |
| `--experiment-name` | 字符串 | `iris-baseline` | MLflow experiment 名称 |
| `--run-name` | 字符串 | 无 | 可选的 run 名称 |
| `--tracking-uri` | 字符串 | 见下 | MLflow tracking URI |

### load_and_predict.py 参数

| 参数 | 说明 |
| --- | --- |
| `--run-id` | **必填**，要加载的 MLflow run id |
| `--tracking-uri` | 覆盖 tracking URI |
| `--artifact-name` | 模型 artifact 名，默认 `model` |
| `--input-csv` | 从 CSV 读取特征行；省略时用 iris 中的固定样本做演示 |
| `--n-samples` | 演示模式下预测的行数，默认 5 |

CSV 需要包含训练时的四个特征列（脚本会按模型 signature 校验列名并给出缺列提示）：

```bash
printf 'sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n5.1,3.5,1.4,0.2\n6.7,3.0,5.2,2.3\n' > sample.csv
python load_and_predict.py --run-id <run_id> --input-csv sample.csv
```

## 3. Tracking 后端（本地，无需 server）

优先级：`--tracking-uri` > 环境变量 `MLFLOW_TRACKING_URI` > 默认值。
默认值是 **脚本同目录下的 SQLite 文件**（由 `Path(__file__)` 推导，不是硬编码的绝对路径）：

```
sqlite:///<phase1-mlops>/mlflow.db      # 元数据
<phase1-mlops>/mlartifacts/             # 模型 artifact
```

覆盖示例：

```bash
export MLFLOW_TRACKING_URI="sqlite:///$HOME/mlops-demo/mlflow.db"
python train.py --n-estimators 50 --max-depth 2 --random-state 7
```

> **为什么默认是 SQLite 而不是经典的 `./mlruns`？**
> MLflow 3.16 起，文件存储（FileStore）进入维护模式，直接使用 `./mlruns` 会抛
> `MlflowException` 拒绝启动，且 FileStore 不支持后续阶段要用的 model registry。
> 两者都是纯本地文件、不需要跑远程 server。
> 如果你确实想用文件存储，指定 `file:` URI 即可，脚本会自动设置
> `MLFLOW_ALLOW_FILE_STORE=true` 优雅降级而不是崩掉：
>
> ```bash
> MLFLOW_TRACKING_URI="file://$(pwd)/mlruns" python train.py --n-estimators 20 --random-state 5
> ```

启动 UI（注意路径含空格时要加引号）：

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --port 5000
# 浏览器打开 http://127.0.0.1:5000
```

## 4. 每次 run 记录了什么

| 类别 | 内容 |
| --- | --- |
| params | `n_estimators`、`max_depth`、`random_state`、`test_size`、`model_type`、`dataset`、`n_train_samples`、`n_test_samples` |
| metrics | `accuracy`、`f1_macro` |
| tags | `git_commit`、`git_branch`、`git_dirty` |
| artifact | `model/`（带 signature 与 input_example，URI 为 `runs:/<run_id>/model`） |

`git_dirty=true` 表示当次训练的工作区有未提交改动，即该 commit 无法完整还原这次实验。
不在 git 仓库中运行时（或系统没有 git），三个 tag 降级为 `unknown`，**不会报错中断**。

## 5. 验收验证

以下命令均在 `phase1-mlops/` 目录、已激活虚拟环境的前提下执行。

### 验收 1：三次不同超参数 → 三条独立 run

```bash
python train.py --n-estimators 100 --max-depth 3 --random-state 42
python train.py --n-estimators 50  --max-depth 2 --random-state 7
python train.py --n-estimators 300 --max-depth none --random-state 2024

# UI 中查看（Experiments → iris-baseline）
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --port 5000

# 或者命令行核对：应输出 3 行、run_id 各不相同
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
df = mlflow.search_runs(experiment_names=['iris-baseline'])
print(df[['run_id','params.n_estimators','params.max_depth','params.random_state',
          'metrics.accuracy','metrics.f1_macro']].to_string(index=False))
"
```

> iris 只有 30 个测试样本且非常容易拟合，不同超参数下 `accuracy`/`f1_macro` 很可能完全相同
> （通常都是 0.966667）——这是数据集饱和，不是没记录成功；以 `run_id` 和 params 区分三条 run。

### 验收 2：同参数重跑，指标完全一致

```bash
python train.py --n-estimators 100 --max-depth 3 --random-state 42 | grep -E "accuracy|f1_macro"
python train.py --n-estimators 100 --max-depth 3 --random-state 42 | grep -E "accuracy|f1_macro"
# 两次输出必须逐位相同（误差为 0）
```

自动化断言版本（对比数据库里最新两条同参数 run）：

```bash
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
df = mlflow.search_runs(
    experiment_names=['iris-baseline'],
    filter_string=\"params.n_estimators='100' and params.max_depth='3' and params.random_state='42'\",
    order_by=['attributes.start_time DESC'], max_results=2)
a, b = df.iloc[0], df.iloc[1]
assert a['metrics.accuracy'] == b['metrics.accuracy']
assert a['metrics.f1_macro'] == b['metrics.f1_macro']
print('reproducible:', a['metrics.accuracy'], a['metrics.f1_macro'])
"
```

### 验收 3：`git_commit` tag 与 `git log` 对应

```bash
git log -1 --format=%H

python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
df = mlflow.search_runs(experiment_names=['iris-baseline'],
                        order_by=['attributes.start_time DESC'], max_results=1)
print(df[['run_id','tags.git_commit','tags.git_branch','tags.git_dirty']].to_string(index=False))
"
# 两条命令输出的 commit hash 应完全一致（UI 中在 run 详情页的 Tags 区域可见）
```

### 验收 4：通过 run_id 加载模型并预测

```bash
RUN_ID=$(python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
print(mlflow.search_runs(experiment_names=['iris-baseline'],
      order_by=['attributes.start_time DESC'], max_results=1)['run_id'][0])
")
python load_and_predict.py --run-id "$RUN_ID"
```

预期输出（每行给出预测类名与真实类名）：

```
Loaded model from run fd7b38aea86c4849b8bd8d1035c5f050; 5 prediction(s):

     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)   predicted      actual
7                  5.0               3.4                1.5               0.2      setosa      setosa
...
Matched 5/5 known labels.
```

### 运行测试

```bash
python -m pytest tests/ -q      # 16 passed
```

覆盖内容：指标处于合理区间、相同参数指标零误差、模型可经 `run_id` 重新加载并预测、
params/metrics/`git_commit` tag 确实落库、非 git 目录下的优雅降级、超参数全部来自命令行、
tracking 目录不可写与 run_id 不存在时的清晰报错。

## 6. 错误处理

所有预期失败都以单行提示 + 退出码 1 呈现，不抛裸 stack trace：

```bash
$ python load_and_predict.py --run-id deadbeef
load_and_predict.py: error: Could not load 'runs:/deadbeef/model' from tracking store
'sqlite:///.../mlflow.db': Run with id=deadbeef not found. Check the run id (`mlflow ui`)
and that the run finished successfully.

$ MLFLOW_TRACKING_URI="sqlite:///read-only-dir/mlflow.db" python train.py
train.py: error: The MLflow tracking directory '/read-only-dir' is not writable by the
current user. Fix its permissions or set --tracking-uri / MLFLOW_TRACKING_URI to a writable path.
```

## 7. 为后续阶段预留的接口

- **模型加载契约**：模型永远以 artifact 名 `model` 记录，可用 `runs:/<run_id>/model` 加载。
  下游代码只依赖 `load_and_predict.py` 的两个函数，不直接调用 MLflow：

  ```python
  from load_and_predict import load_model, predict

  model = load_model(run_id="...")            # 返回 mlflow.pyfunc 模型，与框架无关
  labels = predict(model, features_dataframe) # -> list[int]
  ```

  需要 `predict_proba` 等 sklearn 原生 API 时改用 `mlflow.sklearn.load_model(uri)`。
- **配置注入**：`train.py` 的 `TrainingConfig` 与 `resolve_tracking_uri()` 让 Docker/CI 只需
  传环境变量或命令行参数，无需改代码。
- **已在 Phase 2 完成的抽取**：`resolve_tracking_uri` / `prepare_tracking_backend` /
  `MODEL_ARTIFACT_NAME` 以及全部特征处理逻辑已移入 [../shared/mlops_shared/](../shared/mlops_shared/)，
  `train.py` 现在从那里导入（本文件里的同名函数只是提供本项目默认值的薄封装）。
  这样推理服务镜像只需安装 `mlops-shared`，不必依赖训练脚本，而两侧的预处理仍是同一份代码。
