# RKNN 快速开始指南

## 一键转换流程

### 步骤 1: 导出 ONNX 模型

```bash
# 导出适用于 RK3588 的 ONNX 模型
uv run yolo-demo export yolov8n.pt --rk3588 -o yolov8n-rk3588.onnx
```

### 步骤 2: 使用 Docker 转换为 RKNN

```bash
# 构建 Docker 镜像并运行转换
./docker/build_and_convert.sh yolov8n-rk3588.onnx
```

转换完成后，RKNN 模型保存在 `rknn_models/` 目录下。

### 步骤 3: 传输到 RK3588 设备

```bash
# 使用 scp 传输
scp rknn_models/yolov8n.rknn user@rk3588:/home/user/models/
scp scripts/inference_rknn.py user@rk3588:/home/user/
```

### 步骤 4: 在 RK3588 上运行推理

```bash
# 在设备上运行
ssh user@rk3588
cd /home/user
python3 inference_rknn.py image.jpg yolov8n.rknn -o result.jpg
```

## 完整示例

### 图像检测

```bash
# 单张图像
python3 inference_rknn.py photo.jpg yolov8n.rknn

# 保存结果
python3 inference_rknn.py photo.jpg yolov8n.rknn -o detection_result.jpg

# 调整置信度阈值
python3 inference_rknn.py photo.jpg yolov8n.rknn --conf 0.5
```

### 视频检测

```bash
# 视频文件
python3 inference_rknn.py video.mp4 yolov8n.rknn -o output.mp4

# 显示实时 FPS
python3 inference_rknn.py video.mp4 yolov8n.rknn
```

### 摄像头检测

```bash
# 使用默认摄像头 (ID=0)
python3 inference_rknn.py --camera 0 yolov8n.rknn

# 使用外部摄像头 (ID=1)
python3 inference_rknn.py --camera 1 yolov8n.rknn
```

## INT8 量化（可选）

### 生成校准数据集

```bash
# 从图像目录生成 500 张校准图片
python3 scripts/generate_calib_dataset.py /path/to/images -o calib.txt -n 500

# 从 COCO 数据集生成
python3 scripts/generate_calib_dataset.py /path/to/coco --coco -o calib.txt
```

### 带量化转换

```bash
./docker/build_and_convert.sh yolov8n-rk3588.onnx --quantize --dataset calib.txt
```

## 故障排查

### Docker 构建失败

```bash
# 检查 Docker 是否运行
docker ps

# 手动构建镜像
docker build -f docker/Dockerfile.rknn-converter -t rknn-converter:latest .
```

### 转换失败

```bash
# 查看日志
ls -la docker/logs/
cat docker/logs/convert_*.log
```

### 在 RK3588 上运行失败

```bash
# 检查 RKNN 运行时
python3 -c "from rknn.api import RKNN; print('OK')"

# 检查模型文件
ls -la yolov8n.rknn
file yolov8n.rknn
```

## 性能参考

| 模型 | 输入尺寸 | FP16 FPS | INT8 FPS |
|------|----------|----------|----------|
| YOLOv8n | 640x640 | ~60 | ~90 |
| YOLOv8s | 640x640 | ~40 | ~60 |
| YOLOv8n | 416x416 | ~90 | ~120 |

## 常用命令速查

```bash
# 导出 ONNX
uv run yolo-demo export yolov8n.pt --rk3588

# Docker 转换
./docker/build_and_convert.sh yolov8n-rk3588.onnx

# 生成校准数据
python3 scripts/generate_calib_dataset.py ./images -o calib.txt -n 500

# 带量化转换
./docker/build_and_convert.sh yolov8n-rk3588.onnx --quantize --dataset calib.txt

# 图像推理
python3 inference_rknn.py image.jpg yolov8n.rknn -o result.jpg

# 视频推理
python3 inference_rknn.py video.mp4 yolov8n.rknn -o output.mp4

# 摄像头推理
python3 inference_rknn.py --camera 0 yolov8n.rknn
```
