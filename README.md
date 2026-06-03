# YOLO Demo

轻量级实时目标检测系统，面向边缘计算设备优化。基于 Ultralytics YOLO 构建，支持 Mac (MPS)、NVIDIA (CUDA) 和 CPU 多种后端。

## 功能特性

- 跨平台推理：自动选择最优后端（CUDA > MPS > CPU）
- 增量训练：支持在自定义数据集上微调 YOLO 模型
- 模型导出：PT → ONNX → RKNN 全流程转换，支持 Rockchip NPU 多平台
- WebUI：基于 Gradio 的图形界面（推理、训练、导出、数据集转换）
- REST API：基于 FastAPI 的 RESTful 接口，引擎缓存 + 优雅关闭
- Docker 部署：多阶段构建镜像 + docker-compose 一键启动

## 安装

### 前置要求

- Python 3.9 或更高版本
- uv 包管理器（推荐）

### 安装步骤

```bash
git clone <repository-url>
cd yolo-demo
uv sync
```

### 开发环境

```bash
uv sync --extra dev
```

## 快速开始

### 命令行

```bash
# 推理
uv run yolo-demo infer image.jpg
uv run yolo-demo infer image.jpg --model yolov8s.pt --conf 0.5 -o output.jpg

# 训练
uv run yolo-demo train dataset.yaml --epochs 200 --batch 32

# 导出为 RKNN
uv run yolo-demo export model.pt --platform rk3588

# 启动 WebUI（端口 7860）
uv run yolo-demo webui

# 启动 API 服务（端口 8000）
uv run yolo-demo api
```

### Docker 一键部署

```bash
# 构建并启动 API + WebUI
docker compose up -d

# 仅启动 API
docker compose up -d api

# 仅启动 WebUI
docker compose up -d webui

# 将模型文件注入容器
docker compose cp yolov8n.pt api:/models/

# 查看日志
docker compose logs -f

# 停止并清理
docker compose down -v
```

服务启动后：
- **API 文档**：http://localhost:8000/docs
- **WebUI**：http://localhost:7860

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SERVICE_MODE` | `api` | 启动模式：`api` 或 `webui` |
| `CORS_ORIGINS` | `*` | 允许的跨域来源（逗号分隔）。生产环境应设置为前端域名 |
| `LOG_LEVEL` | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Python API

#### 推理

```python
from yolo_demo.inference import create_engine
import cv2

engine = create_engine("yolov8n.pt")
engine.load_model()

img = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = engine.predict(img_rgb)

print(f"检测到 {len(result.detections)} 个目标")
for det in result.detections:
    print(f"  {det.class_name}: {det.confidence:.2f}, 位置 {det.bbox}")
```

#### 训练

```python
from yolo_demo.training import TrainingConfig, Trainer

config = TrainingConfig(
    model="yolov8n.pt",
    data="dataset.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    lr0=0.01,
)

trainer = Trainer(config)
result = trainer.train()

if result.success:
    print(f"训练完成，模型保存至：{result.model_path}")
else:
    print(f"训练失败：{result.error}")
```

#### 导出模型（PT → RKNN）

```python
from yolo_demo.export import pt_to_rknn

# 直接导出为 RKNN 格式
rknn_path = pt_to_rknn("yolov8n.pt", target_platform="rk3588")
```

## WebUI 使用指南

启动 WebUI 后，在浏览器中访问 `http://localhost:7860`。共 4 个标签页：

### 推理（Inference）

1. 上传待检测图像，输入模型名称或上传自定义 .pt 文件
2. 调整置信度阈值
3. 点击 "Detect Objects" 查看检测结果和可视化输出

### 训练（Training）

1. 上传 YOLO 格式数据集配置文件（.yaml）
2. 选择基础模型（列表选择或上传 .pt 文件）
3. 展开 "Training Parameters" 配置训练超参数
4. 点击 "Start Training"，实时查看日志流输出
5. 训练完成后下载模型文件

### 导出（Export）

- **PT → RKNN 快速导出**：上传 .pt 文件，选择目标平台，一键导出
- **ONNX → RKNN 转换**：上传 .onnx 文件，可选 INT8 量化

### 数据集转换（Dataset Converter）

COCO / VOC 格式 → YOLO 格式，自动生成 dataset.yaml。

## 数据集格式

```
dataset/
├── dataset.yaml          # 数据集配置文件
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

```yaml
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 80
names:
  - person
  - bicycle
  - car
  ...
```

标注文件（.txt）每行格式：

```
<class_id> <x_center> <y_center> <width> <height>
```

坐标已归一化到 [0, 1]。

## REST API

### 启动

```bash
uv run yolo-demo api
# 或通过 Docker
docker compose up -d api
```

访问 `http://localhost:8000/docs` 查看交互式文档。

### 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | API 信息 |
| GET | `/health` | 健康检查（后端、CUDA/MPS 可用性） |
| GET | `/models` | 可用模型列表 |
| GET | `/api/v1/inference/backend` | 推理后端信息 |
| POST | `/api/v1/inference/image` | 图像推理（multipart） |
| POST | `/api/v1/inference/image/base64` | Base64 图像推理 |
| POST | `/api/v1/train` | 启动训练任务 |
| GET | `/api/v1/train/{job_id}/status` | 查询训练状态 |
| DELETE | `/api/v1/train/{job_id}` | 取消训练任务 |
| POST | `/api/v1/export/onnx` | 导出 ONNX |
| POST | `/api/v1/export/onnx/rk3588` | 导出 RK3588 优化模型 |

### 调用示例

```bash
# 图像推理
curl -X POST "http://localhost:8000/api/v1/inference/image" \
  -F "image=@image.jpg" \
  -F "model=yolov8n.pt" \
  -F "conf_threshold=0.25"

# 健康检查
curl http://localhost:8000/health

# 启动训练
curl -X POST "http://localhost:8000/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{"data_yaml": "dataset.yaml", "epochs": 100, "batch_size": 16}'

# 查询训练状态
curl "http://localhost:8000/api/v1/train/<job_id>/status"
```

### 安全配置

CORS 通过环境变量 `CORS_ORIGINS` 控制。开发环境下默认为 `*`（允许所有来源，credentials 禁用）。生产部署时设置：

```bash
export CORS_ORIGINS="https://your-frontend.example.com"
```

引擎缓存上限 5 个模型，空闲 30 分钟后自动释放 GPU 资源。服务收到 SIGTERM 时自动清理所有缓存引擎。

## 边缘设备部署

### RK3588 (Rockchip NPU)

```bash
# 1. PT → RKNN 一键导出
uv run yolo-demo export model.pt --platform rk3588

# 2. 传输到 RK3588
scp model.rknn user@rk3588:/path/to/

# 3. 在设备上使用 rknn-toolkit2-lite 进行推理
```

### NVIDIA Jetson (TensorRT)

```bash
uv run yolo-demo export model.pt --opset 13 --no-dynamic

trtexec --onnx=model.onnx \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:8x3x640x640 \
  --saveEngine=model.engine
```

## 配置

`configs/default.yaml`：

```yaml
model: yolov8n.pt
epochs: 100
imgsz: 640
batch: 16
lr0: 0.01

# 数据增强
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
fliplr: 0.5
```

## 测试

```bash
uv run pytest
uv run pytest --cov
uv run pytest tests/test_inference.py -v
```

## 项目结构

```
yolo-demo/
├── pyproject.toml              # 项目配置和依赖
├── Dockerfile                  # 多阶段构建镜像
├── docker-compose.yml          # 一键部署
├── .dockerignore               # Docker 构建排除
├── configs/
│   └── default.yaml            # 默认训练配置
├── scripts/
│   ├── coco2yolo.py            # COCO/VOC → YOLO 格式转换
│   ├── convert_to_rknn.py      # ONNX → RKNN 转换脚本
│   ├── export.py               # 模型导出脚本
│   └── train.py                # 训练脚本
├── src/yolo_demo/
│   ├── __init__.py
│   ├── main.py                 # CLI 入口（infer / train / export / webui / api）
│   ├── api/
│   │   ├── app.py              # FastAPI 应用（CORS、lifespan、健康检查）
│   │   ├── schemas.py          # Pydantic 数据模型
│   │   └── routes/
│   │       ├── inference.py    # 推理接口（引擎缓存 TTL）
│   │       ├── training.py     # 训练接口
│   │       └── export.py       # 导出接口
│   ├── inference/
│   │   ├── engine.py           # InferenceEngine 抽象基类
│   │   ├── cpu_backend.py      # CPU 后端
│   │   ├── cuda_backend.py     # CUDA 后端
│   │   └── mps_backend.py      # MPS 后端
│   ├── training/
│   │   └── trainer.py          # TrainingConfig + Trainer
│   ├── export/
│   │   └── rknn_exporter.py    # PT → RKNN 导出
│   ├── ui/
│   │   ├── webui.py            # Gradio WebUI（4 标签页）
│   │   └── dataset_converter.py # 数据集转换 UI
│   └── utils/
│       └── logging.py          # 日志配置
└── tests/
    ├── test_api.py
    ├── test_inference.py
    ├── test_export.py
    └── test_training.py
```

## 常见问题

### 内存不足

训练时 CUDA out of memory：
- 减小 batch size
- 减小 imgsz（如从 640 改为 416）
- 使用更小的模型（yolov8n 而非 yolov8x）

### 推理速度慢

- 确认使用了 GPU 后端（检查日志输出）
- 尝试减小输入图像尺寸
- 考虑导出为 TensorRT 或 ONNX 格式

### 数据集转换失败

- 检查 COCO JSON 或 VOC XML 格式是否正确
- 确认图像路径与标注中的引用一致

## 许可证

MIT License
