"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Any


class Detection(BaseModel):
    """Single detection result."""

    bbox: list[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Confidence score")
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")


class InferenceResponse(BaseModel):
    """Response from inference endpoint."""

    success: bool
    detections: list[Detection] = Field(default_factory=list)
    inference_time_ms: float = 0.0
    device: str = ""
    image_width: int = 0
    image_height: int = 0
    error: str | None = None


class TrainingRequest(BaseModel):
    """Request to start training."""

    data_yaml: str | None = None
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    lr0: float = 0.01
    model: str | None = None


class TrainingStatusResponse(BaseModel):
    """Response for training status check."""

    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    model_path: str | None = None


class ExportRequest(BaseModel):
    """Request to export model."""

    model_path: str | None = None
    opset: int = 11
    dynamic: bool = True
    simplify: bool = True


class ExportResponse(BaseModel):
    """Response from export endpoint."""

    success: bool
    onnx_path: str | None = None
    error: str | None = None


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str
    path: str
    size_mb: float = 0.0
    input_shape: list[int] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    """Response listing available models."""

    models: list[ModelInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    backend: str
    cuda_available: bool = False
    mps_available: bool = False
