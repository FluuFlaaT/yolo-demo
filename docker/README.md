# Docker RKNN Converter

本目录包含用于 RKNN 模型转换的 Docker 配置和脚本。

## 文件说明

| 文件 | 说明 |
|------|------|
| `Dockerfile.rknn-converter` | Docker 镜像定义，包含 rknn-toolkit2 环境 |
| `build_and_convert.sh` | 一键构建镜像并运行转换的脚本 |
| `logs/` | 转换日志输出目录 |

## 使用方法

### 方式一：使用一键脚本（推荐）

```bash
# 基本用法
./build_and_convert.sh model.onnx

# 指定输出文件名
./build_and_convert.sh model.onnx -o model.rknn

# 启用 INT8 量化
./build_and_convert.sh model.onnx --quantize --dataset calib.txt

# 指定目标平台
./build_and_convert.sh model.onnx --platform rk3568
```

### 方式二：手动运行 Docker

```bash
# 1. 构建镜像
docker build -f Dockerfile.rknn-converter -t rknn-converter:latest ..

# 2. 运行转换
docker run --rm \
  -v $(pwd)/../rknn_models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd):/workspace/input \
  rknn-converter:latest \
  /workspace/input/model.onnx \
  -o /workspace/models/model.rknn
```

## 输出文件

### RKNN 模型

转换完成后，RKNN 模型保存在项目根目录的 `rknn_models/` 文件夹中。

### 日志文件

转换日志保存在 `docker/logs/` 目录，文件名包含时间戳：

```
logs/
├── convert_20260315_143022.log
├── convert_20260315_145611.log
└── ...
```

## 支持的选项

### build_and_convert.sh 参数

```
Usage: ./build_and_convert.sh <onnx_model> [options]

Options:
  -o, --output <name>    输出 RKNN 模型文件名
  --quantize             启用 INT8 量化
  --dataset <file>       校准数据集文件（量化必需）
  --platform <name>      目标平台（默认：rk3588）
                         可选：rk3588, rk3568, rk3566
  -h, --help             显示帮助信息
```

### 支持的 RKNN 平台

- `rk3588` - Rockchip RK3588 (8TOPS NPU)
- `rk3568` - Rockchip RK3568 (1TOPS NPU)
- `rk3566` - Rockchip RK3566 (0.5TOPS NPU)
- `rv1126` - Rockchip RV1126 (0.5TOPS NPU)
- `rv1109` - Rockchip RV1109 (0.5TOPS NPU)

## INT8 量化

### 生成校准数据集

```bash
# 从图像目录生成
python3 ../scripts/generate_calib_dataset.py /path/to/images -o calib.txt -n 500

# 从 COCO 数据集生成
python3 ../scripts/generate_calib_dataset.py /path/to/coco --coco -o calib.txt
```

### 执行量化转换

```bash
./build_and_convert.sh model.onnx --quantize --dataset calib.txt
```

## 故障排查

### Docker 构建失败

```bash
# 检查 Docker 版本
docker --version

# 检查是否可以拉取基础镜像
docker pull ubuntu:22.04

# 清理构建缓存
docker builder prune -a
```

### 转换过程中报错

1. 查看日志文件：`cat logs/convert_*.log`

2. 常见错误：
   - **ONNX 模型加载失败**: 检查 ONNX 文件是否损坏
   - **不支持的操作符**: 尝试使用 `--opset 11` 重新导出
   - **内存不足**: 减小模型尺寸或使用更简单的模型

### rknn-toolkit2 安装失败

Docker 镜像中使用了官方 rknn-toolkit2 wheel。如果自动安装失败：

1. 从 Rockchip 官方下载 wheel 文件
2. 将 wheel 文件放在 `wheels/` 目录
3. 修改 Dockerfile 取消本地安装的注释

## 性能优化

### 构建优化

```bash
# 使用 BuildKit 加速构建
export DOCKER_BUILDKIT=1
docker build -f Dockerfile.rknn-converter -t rknn-converter:latest ..
```

### 转换优化

在 Dockerfile 中调整优化级别：

```dockerfile
rknn.config(
    target_platform='rk3588',
    optimization_level=3,  # 0-3, 3 为最高优化
)
```

## 相关文档

- [RK3588 部署指南](../docs/RK3588_DEPLOYMENT.md)
- [RKNN 快速开始](../docs/RKNN_QUICKSTART.md)
- [主项目 README](../README.md)
