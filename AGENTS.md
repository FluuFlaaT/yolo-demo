# AGENTS.md — yolo-demo

!!! 仅在 开发用工作机器 平台上运行，不要尝试在 RK3588 上调试 / 开发 / 适配 这个项目 !!!

## 基础命令

```bash
# 安装依赖
uv sync

# 安装开发依赖
uv sync --extra dev

# 运行 CLI（infer/train/export/webui/api）
uv run yolo-demo <command>

# 运行测试（85 tests，60% 覆盖率）
uv run pytest

# 运行测试并查看覆盖率 HTML
uv run pytest && open cov_html/index.html

# 代码检查
uv run ruff check src/

# Docker 一键部署
docker compose up -d
```

## 项目结构

- 包入口：`src/yolo_demo/`
- CLI 入口：`yolo_demo.main:cli`
- WebUI 入口：`yolo_demo.ui.webui:launch`
- API 入口：`yolo_demo.api.app:serve`
- 测试目录：`tests/`（6 个文件，85 tests）
- Docker 部署：`Dockerfile` + `docker-compose.yml` + `.dockerignore`
- CI 工作流：`.github/workflows/ci.yml` + `docker-publish.yml`

## Gradio WebUI

4 个标签页：**Inference** / **Training** / **Export** / **Dataset Converter**。Video Stream 标签页已移除。

## 关键约束

- Python 版本：`>=3.9`（pyproject.toml 要求）
- `uv` 是推荐包管理器（不用 pip）
- 覆盖率要求：不低于 50%（`[tool.coverage.report] fail_under = 50`）
- ruff 检查规则：E, F, I, N, W

## 推理后端选择逻辑

自动按优先级选择：CUDA > MPS > CPU。`create_engine()` 会自动检测可用后端。

## 引擎缓存（API 服务）

- 最多缓存 5 个加载的模型
- 空闲 30 分钟自动释放 GPU 资源
- SIGTERM 时通过 FastAPI lifespan 自动清理

## CORS 配置

开发环境默认 `*`（credentials 禁用）。生产部署时通过 `CORS_ORIGINS` 环境变量设置白名单：
```bash
export CORS_ORIGINS="https://your-frontend.example.com"
```

## Docker 部署

```bash
# 启动 API + WebUI
docker compose up -d

# 仅 API
docker compose up -d api

# 注入模型文件
docker compose cp yolov8n.pt api:/models/

# 查看日志
docker compose logs -f
```

## CI/CD

- **ci.yml**: push/PR 时自动运行 ruff lint → pytest（ubuntu + macOS, py3.9）→ Docker build + smoke test
- **docker-publish.yml**: 推送 `v*` tag 时构建镜像并发布到 ghcr.io（SLSA L3 溯源）

## RKNN 导出注意

- 需要安装 `rknn-toolkit2>=2.0,<3.0`（可选依赖 `uv sync --extra rknn`）
- 仅支持 Linux x86_64 和 Python 3.8-3.10
- setuptools 版本需限制在 `<81`（rknn-toolkit2 依赖 pkg_resources）
