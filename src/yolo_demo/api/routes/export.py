"""Export API routes."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from yolo_demo.export import ONNXExporter, prepare_for_rk3588
from yolo_demo.api.schemas import ExportRequest, ExportResponse

router = APIRouter(prefix="/export", tags=["export"])


@router.post("/onnx", response_model=ExportResponse)
async def export_onnx(request: ExportRequest):
    """
    Export a YOLO model to ONNX format.

    Args:
        request: ExportRequest with model path and options

    Returns:
        ExportResponse with path to exported ONNX file
    """
    try:
        model_path = request.model_path or "yolov8n.pt"

        exporter = ONNXExporter(model_path)
        onnx_path = exporter.export(
            opset=request.opset,
            dynamic=request.dynamic,
            simplify=request.simplify,
        )

        return ExportResponse(
            success=True,
            onnx_path=onnx_path,
        )

    except Exception as e:
        return ExportResponse(
            success=False,
            error=str(e),
        )


@router.post("/onnx/rk3588", response_model=ExportResponse)
async def export_onnx_rk3588(model_path: str = "yolov8n.pt"):
    """
    Export a YOLO model optimized for RK3588 deployment.

    Uses recommended settings for RKNN compatibility.

    Args:
        model_path: Path to YOLO model

    Returns:
        ExportResponse with path to exported ONNX file
    """
    try:
        onnx_path = prepare_for_rk3588(model_path)

        return ExportResponse(
            success=True,
            onnx_path=onnx_path,
        )

    except Exception as e:
        return ExportResponse(
            success=False,
            error=str(e),
        )
