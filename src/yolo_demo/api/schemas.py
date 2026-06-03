"""Pydantic schemas for API requests and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Single detection result."""

    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Confidence score")
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")


class InferenceResponse(BaseModel):
    """Response from inference endpoint."""

    success: bool
    detections: List[Detection] = Field(default_factory=list)
    inference_time_ms: float = 0.0
    device: str = ""
    image_width: int = 0
    image_height: int = 0
    error: Optional[str] = None


class TrainingRequest(BaseModel):
    """Request to start training."""

    data_yaml: Optional[str] = None
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    lr0: float = 0.01
    model: Optional[str] = None


class TrainingStatusResponse(BaseModel):
    """Response for training status check."""

    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    model_path: Optional[str] = None


class ExportRequest(BaseModel):
    """Request to export model."""

    model_path: Optional[str] = None
    opset: int = 11
    dynamic: bool = True
    simplify: bool = True


class ExportResponse(BaseModel):
    """Response from export endpoint."""

    success: bool
    onnx_path: Optional[str] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str
    path: str
    size_mb: float = 0.0
    input_shape: List[int] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    """Response listing available models."""

    models: List[ModelInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    backend: str
    cuda_available: bool = False
    mps_available: bool = False
