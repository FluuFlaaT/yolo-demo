# AGENTS.md

!!! 仅在 开发用工作机器 平台上运行，不要尝试在 RK3588 上尝试 调试 / 开发 / 适配 这个项目 !!!

## 基础命令

```bash
# 安装依赖
uv sync

# 安装开发依赖
uv sync --dev

# 运行 CLI（infer/train/export/webui/api）
uv run yolo-demo <command>

# 运行测试
uv run pytest

# 运行测试并查看覆盖率 HTML
uv run pytest --cov && open cov_html/index.html

# 代码检查
uv run ruff check src/
```

## 项目结构

- 包入口：`src/yolo_demo/`
- CLI 入口：`yolo_demo.main:cli`
- WebUI 入口：`yolo_demo.ui.webui:launch`
- API 入口：`yolo_demo.api.app:serve`
- 测试目录：`tests/`

## 关键约束

- Python 版本固定为 `3.9.*`（pyproject.toml 要求）
- `uv` 是推荐包管理器（不用 pip）
- 覆盖率要求：不低于 50%（`cov_fail_under = 50`）
- ruff 检查规则：E, F, I, N, W

## 推理后端选择逻辑

自动按优先级选择：CUDA > MPS > CPU。`create_engine()` 会自动检测可用后端。

## RKNN 导出注意

- 需要安装 `rknn-toolkit2>=2.0,<3.0`（可选依赖 `uv sync --extra rknn`）
- 仅支持 Linux x86_64 和 Python 3.8-3.10
- setuptools 版本需限制在 `<81`（rknn-toolkit2 依赖 pkg_resources）
