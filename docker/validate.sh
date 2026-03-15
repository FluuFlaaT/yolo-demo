#!/bin/bash
# Run RKNN model validation inside the rknn-converter Docker container (CPU simulation).
#
# Usage:
#   ./validate.sh --rknn <path/to/model.rknn> --onnx <path/to/model.onnx> [--image <img>] [options]
#
# All paths are resolved on the host; the script handles Docker volume mounts.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- argument parsing --------------------------------------------------------
ONNX_PATH=""
IMAGE_PATH=""
CONF=0.25
IOU=0.45
INPUT_SIZE=640
CLASS_NAMES="pill"

while [[ $# -gt 0 ]]; do
    case $1 in
        --onnx)        ONNX_PATH="$2";    shift 2 ;;
        --image)       IMAGE_PATH="$2";   shift 2 ;;
        --conf)        CONF="$2";         shift 2 ;;
        --iou)         IOU="$2";          shift 2 ;;
        --input-size)  INPUT_SIZE="$2";   shift 2 ;;
        --class-names) CLASS_NAMES="$2";  shift 2 ;;
        -h|--help)
            echo "Usage: $0 --onnx <model.onnx> [--image <img>] [options]"
            echo ""
            echo "Required:"
            echo "  --onnx <path>         Path to ONNX model"
            echo ""
            echo "Optional:"
            echo "  --image <path>        Test image (generates synthetic image if omitted)"
            echo "  --conf <float>        Confidence threshold (default: 0.25)"
            echo "  --iou  <float>        IoU threshold for NMS (default: 0.45)"
            echo "  --input-size <int>    Model input size (default: 640)"
            echo "  --class-names <str>   Comma-separated class names (default: pill)"
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$ONNX_PATH" ]]; then
    log_error "--onnx is required"
    echo "Usage: $0 --onnx <model.onnx> [--image <img>]"
    exit 1
fi

# ---- resolve absolute paths --------------------------------------------------
ONNX_ABS="$(realpath "$ONNX_PATH")"
if [[ ! -f "$ONNX_ABS" ]]; then
    log_error "ONNX model not found: $ONNX_ABS"
    exit 1
fi

IMAGE_ABS=""
if [[ -n "$IMAGE_PATH" ]]; then
    IMAGE_ABS="$(realpath "$IMAGE_PATH")"
    if [[ ! -f "$IMAGE_ABS" ]]; then
        log_warn "Image not found ($IMAGE_ABS), will use synthetic test pattern"
        IMAGE_ABS=""
    fi
fi

# ---- print plan --------------------------------------------------------------
log_info "=== RKNN Validation ==="
log_info "  ONNX model : $ONNX_ABS"
[[ -n "$IMAGE_ABS" ]] && log_info "  Test image : $IMAGE_ABS" || log_info "  Test image : synthetic"
log_info "  conf/iou   : $CONF / $IOU"

# ---- ensure Docker image is built --------------------------------------------
log_info "Building Docker image (rknn-converter:latest)..."
docker build \
    --platform linux/amd64 \
    -f "$SCRIPT_DIR/Dockerfile.rknn-converter" \
    -t rknn-converter:latest \
    "$PROJECT_ROOT"

# ---- assemble docker run arguments ------------------------------------------
DOCKER_ARGS=(
    --rm
    --platform linux/amd64
    # Mount the validate script (read-only)
    -v "$PROJECT_ROOT/scripts/validate_rknn.py:/workspace/validate_rknn.py:ro"
    # Mount ONNX model directory
    -v "$(dirname "$ONNX_ABS"):/workspace/onnx_input:ro"
)

# Script arguments
SCRIPT_ARGS=(
    --onnx "/workspace/onnx_input/$(basename "$ONNX_ABS")"
    --conf "$CONF"
    --iou  "$IOU"
    --input-size "$INPUT_SIZE"
    --class-names "$CLASS_NAMES"
)

if [[ -n "$IMAGE_ABS" ]]; then
    DOCKER_ARGS+=(-v "$(dirname "$IMAGE_ABS"):/workspace/images:ro")
    SCRIPT_ARGS+=(--image "/workspace/images/$(basename "$IMAGE_ABS")")
fi

# ---- run validation ----------------------------------------------------------
log_info "Starting validation..."

docker run "${DOCKER_ARGS[@]}" \
    --entrypoint /opt/rknn_env/bin/python \
    rknn-converter:latest \
    /workspace/validate_rknn.py \
    "${SCRIPT_ARGS[@]}"

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    log_info "Validation PASSED"
else
    log_error "Validation FAILED (exit code $EXIT_CODE)"
fi

exit $EXIT_CODE
