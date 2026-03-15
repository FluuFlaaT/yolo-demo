# RK3588 Deployment Guide

This guide covers the complete workflow for deploying YOLO models on RK3588 edge devices.

## Overview

The deployment process consists of three main steps:

1. **Export ONNX model** from PyTorch (.pt) format
2. **Convert to RKNN** format using rknn-toolkit2
3. **Run inference** on RK3588 device using the RKNN runtime

## Prerequisites

### For Conversion (Step 2)

RKNN conversion requires a Linux x86_64 environment with Python 3.6-3.8:

- **Option A**: Docker container (recommended for Mac/Windows users)
- **Option B**: Ubuntu 22.04 VM or physical machine
- **Option C**: Direct installation on RK3588 device

### For Inference (Step 3)

RK3588 device with:
- Rockchip RKNN runtime (pre-installed on most devices)
- Python 3.8+ (optional, for Python inference)

## Step 1: Export ONNX Model

### Using CLI

```bash
# Export with RK3588 optimized settings
uv run yolo-demo export yolov8n.pt --rk3588

# Or manually specify settings
uv run yolo-demo export yolov8n.pt -o yolov8n-rk3588.onnx --opset 11
```

### Using Python API

```python
from yolo_demo.export import prepare_for_rk3588

onnx_path = prepare_for_rk3588("yolov8n.pt")
print(f"Exported to: {onnx_path}")
```

### Export Settings for RK3588

| Setting | Value | Reason |
|---------|-------|--------|
| opset | 11 | Best RKNN compatibility |
| dynamic | True | Support variable batch sizes |
| simplify | True | Optimize graph structure |
| half | False | Use FP32 for better compatibility |

## Step 2: Convert ONNX to RKNN

### Option A: Using Docker (Recommended for Mac/Windows)

The Docker approach works on any platform that supports Docker Desktop or OrbStack.

```bash
# Make the script executable
chmod +x docker/build_and_convert.sh

# Basic conversion
./docker/build_and_convert.sh yolov8n-rk3588.onnx

# With custom output name
./docker/build_and_convert.sh yolov8n-rk3588.onnx -o yolov8n.rknn

# With INT8 quantization (requires calibration dataset)
./docker/build_and_convert.sh yolov8n-rk3588.onnx --quantize --dataset calib.txt
```

**Output**: RKNN model will be saved to `rknn_models/` directory.

**Logs**: Conversion logs are saved to `docker/logs/`.

### Option B: Manual Installation on Linux

```bash
# Install Python 3.8
sudo apt update
sudo apt install -y python3.8 python3-pip

# Create virtual environment
python3.8 -m venv rknn_env
source rknn_env/bin/activate

# Install rknn-toolkit2
pip install rknn-toolkit2==1.5.0

# Run conversion
python scripts/convert_to_rknn.py yolov8n-rk3588.onnx -o yolov8n.rknn
```

### Option C: On RK3588 Device

```bash
# Install rknn-toolkit2 (if not pre-installed)
sudo apt update
sudo apt install -y rknn-toolkit2

# Run conversion directly on device
python scripts/convert_to_rknn.py yolov8n-rk3588.onnx
```

### Conversion Script Options

```bash
python convert_to_rknn.py --help

# Arguments:
#   onnx_model           Path to ONNX model file
#   -o, --output         Output RKNN model path
#   --quantize           Enable INT8 quantization
#   --dataset            Calibration dataset file
#   --platform           Target platform (rk3588, rk3568, etc.)
#   --validate           Validate ONNX model before conversion
#   --log-file           Path to log file
#   --verbose, -v        Enable verbose output
```

### INT8 Quantization

INT8 quantization can reduce model size and improve inference speed, but requires a calibration dataset.

**Generate calibration dataset:**

```bash
# From a directory of images
python scripts/generate_calib_dataset.py /path/to/images -o calib.txt -n 500

# From COCO dataset
python scripts/generate_calib_dataset.py /path/to/coco --coco -o calib.txt

# With image copying
python scripts/generate_calib_dataset.py /path/to/images -o calib.txt --copy
```

**Convert with quantization:**

```bash
python convert_to_rknn.py model.onnx --quantize --dataset calib.txt
```

## Step 3: Run Inference on RK3588

### Transfer Model to Device

```bash
# Using scp
scp rknn_models/yolov8n.rknn user@rk3588:/path/to/models/

# Or using USB drive, network share, etc.
```

### Install Dependencies on Device

```bash
# Most RK3588 devices come with RKNN runtime pre-installed
# Verify installation:
python3 -c "from rknn.api import RKNN; print('RKNN installed')"

# If not installed, follow device manufacturer instructions
```

### Run Inference Script

```bash
# Copy the inference script to device
scp scripts/inference_rknn.py user@rk3588:/path/to/

# Single image inference
python3 inference_rknn.py image.jpg yolov8n.rknn

# Save output
python3 inference_rknn.py image.jpg yolov8n.rknn -o output.jpg

# Video inference
python3 inference_rknn.py video.mp4 yolov8n.rknn -o output.mp4

# Camera inference
python3 inference_rknn.py --camera 0 yolov8n.rknn

# Adjust thresholds
python3 inference_rknn.py image.jpg yolov8n.rknn --conf 0.5 --iou 0.5
```

### Inference Script Options

```bash
python3 inference_rknn.py --help

# Arguments:
#   image_or_video       Path to input image or video
#   model                Path to RKNN model file
#   -o, --output         Output path
#   --conf               Confidence threshold (default: 0.25)
#   --iou                IoU threshold for NMS (default: 0.45)
#   --classes            Path to class names file
#   --camera, -c         Camera device ID
#   --verbose, -v        Enable verbose output
```

### Using Python API

```python
from yolo_demo.inference import create_rknn_engine
import cv2

# Create engine
engine = create_rknn_engine(
    "yolov8n.rknn",
    conf_threshold=0.25,
    iou_threshold=0.45,
)

# Load model
engine.load_model()

# Run inference
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = engine.predict(image_rgb)

# Process results
print(f"Detected {len(result.detections)} objects")
for det in result.detections:
    print(f"  {det.class_name}: {det.confidence:.2f} at {det.bbox}")
print(f"Inference time: {result.inference_time_ms:.2f}ms")
```

## Performance Optimization

### Model Size vs. Speed

| Model | Size (MB) | mAP | RK3588 FPS |
|-------|-----------|-----|------------|
| YOLOv8n | ~6 | 37.3 | ~60 |
| YOLOv8s | ~22 | 44.9 | ~40 |
| YOLOv8m | ~50 | 50.2 | ~25 |
| YOLOv8l | ~88 | 52.9 | ~15 |
| YOLOv8x | ~136 | 53.9 | ~10 |

### INT8 vs. FP16

| Precision | Size | Speed | Accuracy |
|-----------|------|-------|----------|
| FP16 | 100% | 1.0x | 100% |
| INT8 | ~50% | 1.5-2.0x | 95-98% |

### Tips for Best Performance

1. **Use smaller models** (YOLOv8n or YOLOv8s) for real-time applications
2. **Enable INT8 quantization** if accuracy loss is acceptable
3. **Reduce input image size** (e.g., 416x416 instead of 640x640)
4. **Use batch inference** when processing multiple images
5. **Pre-allocate buffers** for camera/video streams

## Troubleshooting

### Conversion Fails

**Error**: "Failed to load ONNX model"

- Ensure ONNX model is valid: `python -c "import onnx; onnx.load('model.onnx')"`
- Try simplifying: `python -m onnxsim model.onnx model-simplified.onnx`
- Check opset version: RKNN works best with opset 11

**Error**: "Unsupported operation"

- Some ONNX operators are not supported by RKNN
- Try exporting with `simplify=True`
- Consider using a different model architecture

### Inference Fails

**Error**: "RKNN model not found"

- Verify model path is correct
- Ensure file permissions allow reading

**Error**: "Failed to load RKNN model"

- Model may be corrupted; re-run conversion
- Check RKNN runtime is installed correctly

### Slow Inference

- Verify NPU is being used (check `/sys/class/devcpu*/load`)
- Ensure model is converted with correct target_platform
- Try reducing input image size
- Check for thermal throttling

## File Structure

```
yolo-demo/
├── scripts/
│   ├── convert_to_rknn.py          # ONNX to RKNN conversion
│   ├── inference_rknn.py           # RKNN inference on device
│   └── generate_calib_dataset.py   # Calibration dataset generator
├── docker/
│   ├── Dockerfile.rknn-converter   # Docker image for conversion
│   └── build_and_convert.sh        # Build and run script
├── rknn_models/                    # Converted RKNN models
└── docker/logs/                    # Conversion logs
```

## Quick Start Summary

```bash
# 1. Export ONNX (on Mac/PC)
uv run yolo-demo export yolov8n.pt --rk3588

# 2. Convert to RKNN (using Docker)
./docker/build_and_convert.sh yolov8n-rk3588.onnx

# 3. Transfer to RK3588
scp rknn_models/yolov8n.rknn user@rk3588:/path/to/
scp scripts/inference_rknn.py user@rk3588:/path/to/

# 4. Run inference (on RK3588)
python3 inference_rknn.py image.jpg yolov8n.rknn -o result.jpg
```

## Additional Resources

- [RKNN Toolkit2 Documentation](https://github.com/airockchip/rknn-toolkit2)
- [Rockchip Developer Forum](https://forums.rock-chips.com/)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
