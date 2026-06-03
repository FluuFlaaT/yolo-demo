"""Export API routes."""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from yolo_demo.export import RKNN_SUPPORTED_PLATFORMS

router = APIRouter(prefix="/export", tags=["export"])


class PTToRKNNRequest(BaseModel):
    """Request to export YOLO model (pt) to RKNN format."""

    model_path: str = Field(..., description="Path to YOLO model file (.pt)")
    target_platform: str = Field("rk3588", description="Target Rockchip platform")
    imgsz: int = Field(640, description="Input image size")


class ExportResponse(BaseModel):
    """Response from export endpoint."""

    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None


@router.post("/rknn", response_model=ExportResponse)
async def export_rknn(request: PTToRKNNRequest):
    """Export YOLO model (pt) to RKNN format using Ultralytics native export."""
    try:
        from yolo_demo.export import pt_to_rknn

        if request.target_platform not in RKNN_SUPPORTED_PLATFORMS:
            return ExportResponse(
                success=False,
                error=f"Unsupported platform: {request.target_platform}",
            )

        rknn_path = pt_to_rknn(
            request.model_path,
            target_platform=request.target_platform,
            imgsz=request.imgsz,
        )

        return ExportResponse(success=True, output_path=rknn_path)

    except Exception as e:
        return ExportResponse(success=False, error=str(e))
