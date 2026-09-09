"""第四阶段的可观测性构建：在第二阶段服务之上叠加 Prometheus 埋点。

**这个包扩展（而不是复制）第二阶段的 `app` 包。** 做法是把 `phase2-mlops/app`
目录追加进本包的 ``__path__``，于是：

* ``app.main`` / ``app.model_loader`` / ``app.schemas`` —— 直接来自第二阶段的源码，
  一行都没有复制，第二阶段改了这里立刻同步；
* ``app.metrics`` / ``app.asgi`` —— 本目录新增的埋点模块。

两个目录里的模块名不重叠，唯一被遮蔽的是 ``__init__.py`` 本身，所以这里必须
补上第二阶段用到的 ``__version__``（``app/main.py`` 会 ``from app import __version__``）。

第二阶段代码的位置可以用环境变量 ``PHASE2_APP_DIR`` 覆盖；容器镜像里把两个阶段的
目录按仓库同样的相对布局摆放，所以本地开发和容器行为一致。
"""

from __future__ import annotations

import os
from pathlib import Path

#: 第二阶段 `app` 包所在目录（默认按仓库布局推导：<repo>/phase2-mlops/app）
PHASE2_APP_DIR: Path = Path(
    os.environ.get("PHASE2_APP_DIR")
    or (Path(__file__).resolve().parents[2] / "phase2-mlops" / "app")
)

if not (PHASE2_APP_DIR / "main.py").is_file():
    raise RuntimeError(
        f"找不到第二阶段的服务代码：'{PHASE2_APP_DIR / 'main.py'}' 不存在。"
        "第四阶段复用第二阶段的 FastAPI 应用，请在仓库内运行，"
        "或用环境变量 PHASE2_APP_DIR 指向 phase2-mlops/app 目录。"
    )

# 关键一行：让 `app.main` 等子模块解析到第二阶段的源码。
__path__.append(str(PHASE2_APP_DIR))

#: 本包遮蔽了第二阶段的 app/__init__.py，因此在这里提供同名常量。
#: /health 返回的 version 会是这个值，表示「跑的是带埋点的第四阶段构建」。
__version__ = "0.4.0"
