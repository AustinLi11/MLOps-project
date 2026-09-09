# syntax=docker/dockerfile:1
#
# 第四阶段服务镜像 = 第二阶段的推理服务 + Prometheus 埋点。
#
# 构建上下文是 **仓库根目录**（镜像需要 shared/ 与 phase2-mlops/app，服务代码不复制）：
#   docker compose build                                   # compose 已配好 context
#   docker build -f phase4-mlops/Dockerfile -t iris-inference:phase4 .
#
# 与 phase2-mlops/Dockerfile 结构一致（多阶段 + 非 root + venv 单目录搬运），
# 差别只在于多装一个埋点库、多拷一个 app 目录。

# =============================================================================
# Stage 1 (builder)：把依赖装进独立 venv，构建产物只有这一个目录会进最终镜像。
# =============================================================================
FROM python:3.10-slim AS builder

# [builder 1/5] 构建期解释器设置：不写 .pyc、日志不缓冲、pip 不留缓存。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# [builder 2/5] 独立 venv：运行阶段整目录拷走，pip/构建工具留在本阶段。
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# [builder 3/5] 只拷依赖清单和共享包（缓存层）：改 app 代码不会触发重新装依赖。
# 注意三者的相对关系要和仓库一致，phase4 的清单里写的是 `-r ../phase2-mlops/requirements.txt`。
WORKDIR /build
COPY shared/ shared/
COPY phase2-mlops/requirements.txt phase2-mlops/requirements.txt
COPY phase4-mlops/requirements.txt phase4-mlops/requirements.txt

# [builder 4/5] 在 phase4-mlops/ 下安装：清单里的相对路径按当前工作目录解析
# （`../shared` -> /build/shared，与本地开发行为一致）。
# requirements-monitoring.txt（Evidently）不装：漂移分析是离线批处理，不进服务镜像。
WORKDIR /build/phase4-mlops
RUN pip install --no-cache-dir -r requirements.txt

# [builder 5/5] 清字节码缓存（phase2 实测省约 108 MB）。
# 不要删 site-packages 里的 tests/ 目录：numpy 反序列化模型时会 import numpy._core.tests。
RUN find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /opt/venv -name '*.pyc' -delete

# =============================================================================
# Stage 2 (runtime)：最终镜像，只有 venv + 两个 app 目录。
# =============================================================================
FROM python:3.10-slim AS runtime

# [runtime 1/8] 运行期设置。PYTHONPATH 指向 phase4-mlops，使 `app` 包可导入；
# app/__init__.py 会把 ../phase2-mlops/app 并进同一个包（详见该文件注释）。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/srv/phase4-mlops \
    MODEL_ARTIFACT_NAME=model
# MODEL_URI 仍然不给默认值：模型在 `docker run` / compose 时注入，镜像里不含任何模型引用。

# [runtime 2/8] 非特权用户（uid/gid 10001），服务不以 root 运行。
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

# [runtime 3/8] 从 builder 取走唯一产物：装好依赖的 venv。
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# [runtime 4/8] 业务代码：第二阶段的服务 + 第四阶段的埋点。
# 目录相对关系与仓库保持一致，app/__init__.py 才能找到 phase2 的代码。
WORKDIR /srv
COPY --chown=appuser:appuser phase2-mlops/app/ ./phase2-mlops/app/
COPY --chown=appuser:appuser phase4-mlops/app/ ./phase4-mlops/app/

# [runtime 5/8] 模型挂载点（MODEL_URI=/models/... 时用）。
RUN mkdir -p /models && chown appuser:appuser /models

# [runtime 6/8] 降权后再启动。
WORKDIR /srv/phase4-mlops
USER appuser

# [runtime 7/8] 端口与存活探针。slim 镜像没有 curl/wget，用解释器自带的 urllib。
# 模型未加载时 /health 返回 503 -> 容器被标记 unhealthy（readiness 语义）。
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# [runtime 8/8] 入口是带埋点的 ASGI 应用（app.asgi），不是第二阶段的 app.main。
CMD ["uvicorn", "app.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
