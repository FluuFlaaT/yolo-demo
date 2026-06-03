# YOLO Demo

轻量级实时目标检测系统，面向边缘计算设备优化。基于 Ultralytics YOLO 构建，支持 Mac (MPS)、NVIDIA (CUDA) 和 CPU 多种后端。

## 功能特性

- 跨平台推理：自动选择最优后端（CUDA 优先，其次 MPS，最后 CPU）
- 增量训练：支持在自定义数据集上微调 YOLO 模型
- ONNX 导出：导出模型用于边缘设备部署（RK3588、TensorRT、OpenVINO）
- RKNN 转换：支持将 ONNX 模型转换为 RKNN 格式（需安装 rknn-toolkit2）
- WebUI：基于 Gradio 的图形界面，支持推理、训练和导出
- REST API：基于 FastAPI 的 RESTful 接口，便于集成

## 安装

### 前置要求

- Python 3.9 或更高版本
- uv 包管理器（推荐）或 pip

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd yolo-demo

# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip 安装
pip install -e .
```

### 开发环境

```bash
# 安装开发依赖（包含测试工具）
uv sync --dev
```

## 快速开始

### 命令行使用

#### 目标检测推理

```bash
# 基本推理（使用默认 YOLOv8n 模型）
uv run yolo-demo infer image.jpg

# 指定模型
uv run yolo-demo infer image.jpg --model yolov8s.pt

# 保存结果
uv run yolo-demo infer image.jpg -o output.jpg

# 调整置信度阈值
uv run yolo-demo infer image.jpg --conf 0.5
```

#### 模型训练

```bash
# 基本训练
uv run yolo-demo train dataset.yaml

# 指定预训练模型和参数
uv run yolo-demo train dataset.yaml --model yolov8n.pt --epochs 200 --batch 32

# 指定输出目录
uv run yolo-demo train dataset.yaml -o runs/custom_training
```

#### 模型导出

```bash
# 导出为 ONNX 格式
uv run yolo-demo export model.pt

# 指定输出路径
uv run yolo-demo export model.pt -o ./models

# 指定 opset 版本
uv run yolo-demo export model.pt --opset 12

# 导出为 RK3588 优化格式
uv run yolo-demo export model.pt --rk3588
```

#### 启动服务

```bash
# 启动 WebUI（默认端口 7860）
uv run yolo-demo webui

# 指定端口
uv run yolo-demo webui --port 8080

# 启动 API 服务（默认端口 8000）
uv run yolo-demo api

# 启用自动重载（开发模式）
uv run yolo-demo api --reload
```

### Python API

#### 推理

```python
from yolo_demo.inference import create_engine
import cv2

# 创建推理引擎（自动选择最优后端）
engine = create_engine("yolov8n.pt")
engine.load_model()

# 读取图像
img = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 执行推理
result = engine.predict(img_rgb)

# 输出结果
print(f"检测到 {len(result.detections)} 个目标")
for det in result.detections:
    print(f"  {det.class_name}: 置信度 {det.confidence:.2f}, 位置 {det.bbox}")
```

#### 训练

```python
from yolo_demo.training import TrainingConfig, Trainer

# 配置训练参数
config = TrainingConfig(
    model="yolov8n.pt",      # 预训练模型
    data="dataset.yaml",     # 数据集配置文件
    epochs=100,              # 训练轮数
    batch=16,                # 批次大小
    imgsz=640,               # 输入图像尺寸
    lr0=0.01,                # 初始学习率
)

# 创建训练器并开始训练
trainer = Trainer(config)
result = trainer.train()

# 检查训练结果
if result.success:
    print(f"训练完成，模型保存至：{result.model_path}")
else:
    print(f"训练失败：{result.error}")
```

#### 导出模型

```python
from yolo_demo.export import ONNXExporter, prepare_for_rk3588

# 标准 ONNX 导出
exporter = ONNXExporter("yolov8n.pt")
onnx_path = exporter.export(opset=11, dynamic=True)

# RK3588 优化导出
onnx_path = prepare_for_rk3588("yolov8n.pt")
```

## WebUI 使用指南

启动 WebUI 后，在浏览器中访问 `http://localhost:7860`。

### 推理标签页

1. 上传待检测图像
2. 选择模型来源：
   - 预训练模型：从 YOLOv8n/s/m/l/x 中选择
   - 自定义模型：上传自己的 .pt 文件
3. 调整置信度阈值（可选）
4. 点击"Detect Objects"执行检测
5. 查看检测结果和可视化输出

### 训练标签页

1. 上传 YOLO 格式的数据集配置文件（.yaml）
2. 上传预训练模型（可选，默认使用 YOLOv8n.pt）
3. 展开"Training Parameters"设置训练参数：
   - Epochs：训练轮数
   - Batch Size：批次大小
   - Image Size：图像尺寸
   - Initial Learning Rate：初始学习率
   - Output Directory：输出目录（可选）
4. 点击"Start Training"开始训练
5. 训练完成后下载模型文件

### 导出标签页

#### 标准 ONNX 导出

1. 上传 .pt 模型文件
2. 选择 ONNX Opset 版本（推荐 11 或 12）
3. 勾选动态轴和简化选项
4. 指定输出文件名（可选）
5. 点击"Export to ONNX"

#### RK3588 快速导出

1. 上传 .pt 模型文件
2. 指定输出文件名（默认格式：`模型名-rk3588-export.onnx`）
3. 点击"Export for RK3588"

### 数据集转换标签页

用于将 COCO 或 VOC 格式数据集转换为 YOLO 格式。

1. 选择输入格式（COCO 或 VOC）
2. 上传对应的标注文件：
   - COCO：上传 annotations.json 文件
   - VOC：上传包含 VOCdevkit 的 zip 文件
3. 选择是否复制图像到输出目录
4. 指定输出数据集名称
5. 点击"Convert Dataset"执行转换
6. 下载生成的 dataset.yaml 文件

## 数据集格式

### YOLO 格式数据集结构

```
dataset/
├── dataset.yaml          # 数据集配置文件
├── images/
│   ├── train/           # 训练图像
│   └── val/             # 验证图像
└── labels/
    ├── train/           # 训练标注
    └── val/             # 验证标注
```

### dataset.yaml 配置文件

```yaml
path: /path/to/dataset    # 数据集根目录
train: images/train       # 训练集相对路径
val: images/val           # 验证集相对路径
test: images/test         # 测试集相对路径（可选）

nc: 80                    # 类别数量
names:                    # 类别名称列表
  - person
  - bicycle
  - car
  ...
```

### 标注文件格式

每个图像对应一个同名的.txt 标注文件，每行格式为：

```
<class_id> <x_center> <y_center> <width> <height>
```

坐标值已归一化到 [0, 1] 范围。

## REST API 接口

### 启动服务

```bash
uv run yolo-demo api
```

服务启动后访问 `http://localhost:8000/docs` 查看交互式 API 文档。

### 接口列表

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | API 信息 |
| GET | `/health` | 健康检查 |
| GET | `/models` | 可用模型列表 |
| GET | `/api/v1/inference/backend` | 后端信息 |
| POST | `/api/v1/inference/image` | 图像推理 |
| POST | `/api/v1/inference/image/base64` | Base64 图像推理 |
| POST | `/api/v1/train` | 启动训练任务 |
| GET | `/api/v1/train/{job_id}/status` | 查询训练状态 |
| DELETE | `/api/v1/train/{job_id}` | 取消训练任务 |
| POST | `/api/v1/export/onnx` | 导出 ONNX |
| POST | `/api/v1/export/onnx/rk3588` | 导出 RK3588 优化模型 |

### 接口调用示例

#### 图像推理

```bash
curl -X POST "http://localhost:8000/api/v1/inference/image" \
  -F "image=@image.jpg" \
  -F "model=yolov8n.pt" \
  -F "conf_threshold=0.25"
```

#### Base64 图像推理

```bash
# 将图像转换为 base64
base64 -i image.jpg

# 调用接口
curl -X POST "http://localhost:8000/api/v1/inference/image/base64" \
  -H "Content-Type: application/json" \
  -d '{"image_data": "<base64_string>", "model": "yolov8n.pt"}'
```

#### 启动训练

```bash
curl -X POST "http://localhost:8000/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data_yaml": "dataset.yaml",
    "epochs": 100,
    "batch_size": 16,
    "imgsz": 640,
    "lr0": 0.01
  }'
```

#### 查询训练状态

```bash
curl "http://localhost:8000/api/v1/train/<job_id>/status"
```

返回示例：

```json
{
  "job_id": "xxx-xxx-xxx",
  "status": "running",
  "progress": 0.45,
  "metrics": {"precision": 0.85, "recall": 0.82}
}
```

#### 导出 ONNX

```bash
curl -X POST "http://localhost:8000/api/v1/export/onnx" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "yolov8n.pt",
    "opset": 11,
    "dynamic": true,
    "simplify": true
  }'
```

## 边缘设备部署

### RK3588 (Rockchip NPU)

#### 快速转换流程

```bash
# 1. 导出 ONNX 模型（优化设置）
uv run yolo-demo export model.pt --rk3588

# 2. 转换为 RKNN 格式（需要安装 rknn-toolkit2）
# 方式一：使用 WebUI
uv run yolo-demo webui
# 然后在 Export 标签页中使用 "Convert to RKNN" 功能

# 方式二：使用 Python API
python3 -c "
from yolo_demo.export import RKNNExporter
exporter = RKNNExporter('model.onnx')
exporter.export('model.rknn')
"

# 3. 传输到 RK3588 设备
scp model.rknn user@rk3588:/path/to/

# 4. 在设备上使用 rknn-toolkit2-lite 运行推理
```

#### 安装 rknn-toolkit2

```bash
# 仅支持 Linux x86_64 和 Python 3.8-3.10
pip install 'yolo-demo[rknn]'
```

### NVIDIA Jetson (TensorRT)

1. 导出 ONNX 模型

```bash
uv run yolo-demo export model.pt --opset 13 --no-dynamic
```

2. 使用 trtexec 转换

```bash
trtexec --onnx=model.onnx \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:8x3x640x640 \
  --saveEngine=model.engine
```

## 配置选项

### 训练配置

可在 `configs/` 目录找到预设配置文件。

```yaml
# configs/default.yaml
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
# 运行所有测试
uv run pytest

# 运行测试并生成覆盖率报告
uv run pytest --cov

# 查看 HTML 覆盖率报告
open cov_html/index.html

# 运行特定测试文件
uv run pytest tests/test_inference.py -v
```

## 项目结构

```
yolo-demo/
├── pyproject.toml          # 项目配置和依赖
├── README.md               # 项目文档
├── configs/
│   └── default.yaml        # 默认训练配置
├── scripts/
│   ├── coco2yolo.py        # COCO/VOC 转 YOLO 格式工具
│   ├── export.py           # 导出脚本
│   └── train.py            # 训练脚本
├── src/yolo_demo/
│   ├── __init__.py
│   ├── main.py             # CLI 入口
│   ├── api/
│   │   ├── app.py          # FastAPI 应用
│   │   ├── schemas.py      # Pydantic 数据模型
│   │   └── routes/
│   │       ├── inference.py  # 推理接口
│   │       ├── training.py   # 训练接口
│   │       └── export.py     # 导出接口
│   ├── inference/
│   │   ├── engine.py       # 推理抽象基类
│   │   ├── cpu_backend.py  # CPU 后端
│   │   ├── cuda_backend.py # CUDA 后端
│   │   └── mps_backend.py  # MPS 后端
│   ├── training/
│   │   └── trainer.py      # 训练模块
│   ├── export/
│   │   ├── onnx_exporter.py # ONNX 导出模块
│   │   └── rknn_exporter.py # RKNN 转换模块
│   ├── ui/
│   │   ├── webui.py         # Gradio WebUI
│   │   └── dataset_converter.py  # 数据集转换 UI
│   └── utils/
│       └── logging.py       # 日志配置
└── tests/
    ├── test_api.py
    ├── test_inference.py
    └── test_export.py
```

## 日志配置

```python
from yolo_demo.utils import setup_logging, get_logger

# 配置日志
setup_logging(
    level=logging.INFO,
    log_file="app.log"
)

# 获取 logger
logger = get_logger(__name__)
logger.info("Application started")
```

## 常见问题

### 内存不足

训练时出现 CUDA out of memory：
- 减小 batch size
- 减小 imgsz（如从 640 改为 416）
- 使用更小的模型（如 yolov8n 而非 yolov8x）

### 推理速度慢

- 确认使用了 GPU 后端（检查日志输出）
- 尝试减小输入图像尺寸
- 考虑导出为 TensorRT 或 ONNX 格式

### 数据集转换失败

- 检查 COCO JSON 或 VOC XML 格式是否正确
- 确认图像路径与标注中的引用一致
- 检查类别名称是否匹配

## 许可证

MIT License
