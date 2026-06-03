"""Inference API routes."""

import base64
import io
import logging
import time
from typing import Any, Dict, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image

from yolo_demo.api.schemas import Detection, InferenceResponse
from yolo_demo.inference import create_engine, get_available_backend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["inference"])

# -- Engine cache with TTL -----------------------------------------------------
# Cache entries: (engine, last_access_time, model_path)
# TTL: 30 minutes of inactivity before eviction.
# Max size: 5 engines (older idle entries evicted first).

_MAX_CACHE_SIZE = 5
_CACHE_TTL_SECONDS = 30 * 60  # 30 minutes

_engine_cache: Dict[str, Tuple[Any, float]] = {}  # model_path -> (engine, last_access)


def _evict_idle() -> None:
    """Evict cache entries that have been idle beyond TTL or when cache is full."""
    now = time.monotonic()

    # Evict expired
    expired = [
        k for k, (_, ts) in _engine_cache.items() if (now - ts) > _CACHE_TTL_SECONDS
    ]
    for key in expired:
        _close_engine(key)

    # Evict oldest if still over limit
    while len(_engine_cache) > _MAX_CACHE_SIZE:
        oldest = min(_engine_cache.items(), key=lambda x: x[1][1])
        _close_engine(oldest[0])


def _close_engine(model_path: str) -> None:
    """Gracefully close and remove a cached engine."""
    if model_path in _engine_cache:
        engine, _ = _engine_cache.pop(model_path)
        try:
            engine.__exit__(None, None, None)  # release GPU resources
            logger.info("Evicted cached engine: %s", model_path)
        except Exception:
            logger.warning("Error closing engine for %s", model_path, exc_info=True)


def get_engine(model_path: str):
    """Get or create an inference engine with cache eviction."""
    _evict_idle()

    if model_path not in _engine_cache:
        logger.info("Loading engine for model: %s", model_path)
        engine = create_engine(model_path)
        engine.load_model()
        _engine_cache[model_path] = (engine, time.monotonic())
        logger.info("Engine cached (total: %d)", len(_engine_cache))
    else:
        # Refresh access time
        engine, _ = _engine_cache[model_path]
        _engine_cache[model_path] = (engine, time.monotonic())

    return engine


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

    except Exception:
        logger.exception("Inference failed for model: %s", model)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during inference. Check server logs for details.",
        )


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

    except Exception:
        logger.exception("Base64 inference failed for model: %s", model)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during inference. Check server logs for details.",
        )


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
