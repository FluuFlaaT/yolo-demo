"""Inference API routes."""

import base64
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image

from yolo_demo.inference import create_engine, get_available_backend
from yolo_demo.api.schemas import Detection, InferenceResponse

router = APIRouter(prefix="/inference", tags=["inference"])

# Global engine cache
_engine_cache: dict[str, Any] = {}


def get_engine(model_path: str):
    """Get or create an inference engine."""
    if model_path not in _engine_cache:
        engine = create_engine(model_path)
        engine.load_model()
        _engine_cache[model_path] = engine
    return _engine_cache[model_path]


@router.post("/image", response_model=InferenceResponse)
async def infer_image(
    image: UploadFile,
    model: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
):
    """
    Perform object detection on an uploaded image.

    Args:
        image: Image file (JPEG, PNG, etc.)
        model: Model path or name (default: yolov8n.pt)
        conf_threshold: Confidence threshold for detections

    Returns:
        InferenceResponse with detections and metadata
    """
    try:
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)

        # Run inference
        engine = get_engine(model)
        start = time.perf_counter()
        result = engine.predict(img_np)
        elapsed = (time.perf_counter() - start) * 1000

        # Filter by confidence
        filtered_detections = [
            Detection(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
            )
            for det in result.detections
            if det.confidence >= conf_threshold
        ]

        return InferenceResponse(
            success=True,
            detections=filtered_detections,
            inference_time_ms=round(elapsed, 2),
            device=result.device,
            image_width=img_np.shape[1],
            image_height=img_np.shape[0],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image/base64", response_model=InferenceResponse)
async def infer_image_base64(
    image_data: str,
    model: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
):
    """
    Perform object detection on a base64-encoded image.

    Args:
        image_data: Base64-encoded image string
        model: Model path or name
        conf_threshold: Confidence threshold

    Returns:
        InferenceResponse with detections
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        # Run inference
        engine = get_engine(model)
        start = time.perf_counter()
        result = engine.predict(img_np)
        elapsed = (time.perf_counter() - start) * 1000

        filtered_detections = [
            Detection(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
            )
            for det in result.detections
            if det.confidence >= conf_threshold
        ]

        return InferenceResponse(
            success=True,
            detections=filtered_detections,
            inference_time_ms=round(elapsed, 2),
            device=result.device,
            image_width=img_np.shape[1],
            image_height=img_np.shape[0],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backend")
async def get_backend_info():
    """Get information about the current inference backend."""
    import torch

    return {
        "backend": get_available_backend(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }
