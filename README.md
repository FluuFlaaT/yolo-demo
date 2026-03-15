# YOLO Demo

Lightweight real-time object detection system for edge computing. Built with Ultralytics YOLO, supporting Mac (MPS), NVIDIA (CUDA), and CPU backends.

## Features

- **Cross-platform inference**: Automatic backend selection (CUDA > MPS > CPU)
- **Incremental training**: Fine-tune YOLO models on custom datasets
- **ONNX export**: Export models for edge deployment (RK3588, TensorRT, OpenVINO)
- **WebUI**: Gradio-based interface for inference, training, and export
- **REST API**: FastAPI-based API for integration

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd yolo-demo

# Install dependencies with uv
uv sync

# Or install with pip
pip install -e .
```

### Development dependencies

```bash
uv sync --dev
```

## Quick Start

### Command Line

```bash
# Run inference on an image
uv run yolo-demo infer image.jpg --model yolov8n.pt

# Train a custom model
uv run yolo-demo train data.yaml --epochs 100 --batch 16

# Export to ONNX
uv run yolo-demo export yolov8n.pt --output ./models

# Export for RK3588
uv run yolo-demo export yolov8n.pt --rk3588

# Launch WebUI
uv run yolo-demo webui --port 7860

# Launch API server
uv run yolo-demo api --port 8000
```

### Python API

```python
from yolo_demo.inference import create_engine

# Create engine (auto-selects best backend)
engine = create_engine("yolov8n.pt")
engine.load_model()

# Run inference
import cv2
img = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = engine.predict(img_rgb)

print(f"Detected {len(result.detections)} objects")
for det in result.detections:
    print(f"  {det.class_name}: {det.confidence:.2f}")
```

### Training

```python
from yolo_demo.training import TrainingConfig, Trainer

config = TrainingConfig(
    model="yolov8n.pt",
    data="data.yaml",
    epochs=100,
    batch=16,
)

trainer = Trainer(config)
result = trainer.train()

if result.success:
    print(f"Model saved to: {result.model_path}")
```

### ONNX Export

```python
from yolo_demo.export import ONNXExporter, prepare_for_rk3588

# Standard export
exporter = ONNXExporter("yolov8n.pt")
onnx_path = exporter.export(opset=11, dynamic=True)

# RK3588-optimized export
onnx_path = prepare_for_rk3588("yolov8n.pt")
```

## REST API

Start the API server:

```bash
uv run yolo-demo api
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/api/v1/inference/image` | Image inference |
| POST | `/api/v1/inference/image/base64` | Base64 image inference |
| GET | `/api/v1/inference/backend` | Backend info |
| POST | `/api/v1/train` | Start training job |
| GET | `/api/v1/train/{job_id}/status` | Get training status |
| POST | `/api/v1/export/onnx` | Export to ONNX |
| POST | `/api/v1/export/onnx/rk3588` | Export for RK3588 |

### Example API Usage

```bash
# Inference
curl -X POST "http://localhost:8000/api/v1/inference/image" \
  -F "image=@image.jpg" \
  -F "model=yolov8n.pt"

# Start training
curl -X POST "http://localhost:8000/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{"data_yaml": "data.yaml", "epochs": 100}'

# Check training status
curl "http://localhost:8000/api/v1/train/{job_id}/status"
```

## Project Structure

```
yolo-demo/
├── src/yolo_demo/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── api/
│   │   ├── app.py           # FastAPI application
│   │   ├── schemas.py       # Pydantic models
│   │   └── routes/
│   │       ├── inference.py  # Inference endpoints
│   │       ├── training.py   # Training endpoints
│   │       └── export.py     # Export endpoints
│   ├── inference/
│   │   ├── engine.py        # Abstract base class
│   │   ├── cuda_backend.py  # NVIDIA CUDA backend
│   │   ├── mps_backend.py   # Apple MPS backend
│   │   └── cpu_backend.py   # CPU fallback
│   ├── training/
│   │   └── trainer.py       # Training module
│   ├── export/
│   │   └── onnx_exporter.py # ONNX export
│   └── ui/
│       └── webui.py         # Gradio WebUI
├── tests/
├── configs/
└── scripts/
```

## Configuration

### Default Config (`configs/default.yaml`)

```yaml
model: yolov8n.pt
epochs: 100
imgsz: 640
batch: 16
lr0: 0.01
```

### RK3588 Config (`configs/rk3588.yaml`)

Optimized settings for RK3588 edge deployment.

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=yolo_demo

# Run specific test file
uv run pytest tests/test_inference.py
```

## Edge Deployment

### RK3588 (Rockchip NPU)

1. Export model with RK3588 settings:
   ```bash
   uv run yolo-demo export model.pt --rk3588
   ```

2. Convert to RKNN on target device:
   ```python
   from rknn.api import RKNN

   rknn = RKNN()
   rknn.config(target_platform='rk3588')
   rknn.load_onnx(model='model.onnx')
   rknn.build(do_quantization=True, dataset='calib.txt')
   rknn.export_rknn('model.rknn')
   ```

### TensorRT (NVIDIA Jetson)

```bash
# Export with TensorRT settings
python -c "from yolo_demo.export import ONNXExporter; \
    e = ONNXExporter('yolov8n.pt'); \
    e.export(opset=13, dynamic=False)"
```

## License

MIT
