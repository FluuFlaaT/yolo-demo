#!/bin/bash
# Build and run RKNN conversion Docker container
# Usage: ./build_and_convert.sh <onnx_model> [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
ONNX_MODEL=""
OUTPUT_NAME=""
QUANTIZE=""
DATASET=""
PLATFORM="rk3588"

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --quantize)
            QUANTIZE="--quantize"
            shift
            ;;
        --dataset)
            DATASET="--dataset $2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <onnx_model> [options]"
            echo ""
            echo "Options:"
            echo "  -o, --output <name>    Output RKNN model name"
            echo "  --quantize             Enable INT8 quantization"
            echo "  --dataset <file>       Calibration dataset file (required for --quantize)"
            echo "  --platform <name>      Target platform (default: rk3588)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            if [[ -z "$ONNX_MODEL" ]]; then
                ONNX_MODEL="$1"
            else
                log_error "Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [[ -z "$ONNX_MODEL" ]]; then
    log_error "Please provide an ONNX model file"
    echo "Usage: $0 <onnx_model> [options]"
    exit 1
fi

# Resolve absolute path
if [[ ! -f "$ONNX_MODEL" ]]; then
    log_error "ONNX model not found: $ONNX_MODEL"
    exit 1
fi
ONNX_MODEL_ABS="$(realpath "$ONNX_MODEL")"
ONNX_MODEL_NAME="$(basename "$ONNX_MODEL")"

# Determine output name
if [[ -z "$OUTPUT_NAME" ]]; then
    OUTPUT_NAME="${ONNX_MODEL_NAME%.onnx}.rknn"
fi

log_info "=== RKNN Conversion ==="
log_info "Input:  $ONNX_MODEL_ABS"
log_info "Output: $OUTPUT_NAME"
log_info "Platform: $PLATFORM"
if [[ -n "$QUANTIZE" ]]; then
    log_info "Quantization: ENABLED"
    log_info "Dataset: $DATASET"
fi

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/rknn_models"
mkdir -p "$OUTPUT_DIR"

# Create log directory
LOG_DIR="$PROJECT_ROOT/docker/logs"
mkdir -p "$LOG_DIR"

# Generate log filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/convert_${TIMESTAMP}.log"

log_info "Log file: $LOG_FILE"

# Build Docker image
log_info "Building Docker image..."
docker build \
    -f "$SCRIPT_DIR/Dockerfile.rknn-converter" \
    -t rknn-converter:latest \
    "$PROJECT_ROOT" 2>&1 | tee -a "$LOG_FILE"

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    log_error "Docker build failed"
    exit 1
fi

log_info "Docker image built successfully"

# Run conversion
log_info "Starting conversion..."

# Build docker run arguments
DOCKER_ARGS=(
    --rm
    -v "$OUTPUT_DIR:/workspace/models"
    -v "$LOG_DIR:/workspace/logs"
    -v "$(dirname "$ONNX_MODEL_ABS"):/workspace/input:ro"
)

# Mount dataset if provided
if [[ -n "$DATASET" ]]; then
    DATASET_PATH=$(echo "$DATASET" | awk '{print $2}')
    DATASET_ABS="$(realpath "$DATASET_PATH")"
    DOCKER_ARGS+=(-v "$(dirname "$DATASET_ABS"):/workspace/dataset:ro")
fi

# Run the container
docker run "${DOCKER_ARGS[@]}" \
    rknn-converter:latest \
    "/workspace/input/$ONNX_MODEL_NAME" \
    -o "/workspace/models/$OUTPUT_NAME" \
    --platform "$PLATFORM" \
    $QUANTIZE \
    $DATASET \
    --log-file "/workspace/logs/$(basename "$LOG_FILE")" \
    --verbose 2>&1 | tee -a "$LOG_FILE"

DOCKER_EXIT=${PIPESTATUS[0]}

if [[ $DOCKER_EXIT -ne 0 ]]; then
    log_error "Conversion failed! Check log file: $LOG_FILE"
    exit 1
fi

# Check if output file exists
RKNN_OUTPUT="$OUTPUT_DIR/$OUTPUT_NAME"
if [[ -f "$RKNN_OUTPUT" ]]; then
    RKNN_SIZE=$(du -h "$RKNN_OUTPUT" | cut -f1)
    log_info "=== Conversion Successful ==="
    log_info "Output: $RKNN_OUTPUT"
    log_info "Size: $RKNN_SIZE"
    log_info "Log: $LOG_FILE"
else
    log_error "Conversion completed but output file not found"
    exit 1
fi

echo ""
echo "To use this model on RK3588:"
echo "  1. Copy $RKNN_OUTPUT to your device"
echo "  2. Use the inference script: python inference_rknn.py $OUTPUT_NAME"
