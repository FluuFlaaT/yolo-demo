"""Inference service — decouples Gradio callbacks from inference engine calls."""

import logging
from typing import Any, Dict, Optional

import numpy as np

from ...inference import DetectionResult, create_engine

logger = logging.getLogger(__name__)


def resolve_model_path(model_name: str, custom_model: Optional[str]) -> str:
    """Resolve model path preferring custom file upload over text input.

    Args:
        model_name: Model name from text input (e.g., "yolov8n.pt").
        custom_model: Path to uploaded custom model file, or None.

    Returns:
        Resolved model path string.

    Raises:
        ValueError: If neither model_name nor custom_model is provided.
    """
    if custom_model:  # treat None and empty string as "not provided"
        return custom_model
    if model_name:
        return model_name
    raise ValueError("Please enter a model name or upload a custom model")


def format_detections(result: DetectionResult, conf_threshold: float) -> Dict[str, Any]:
    """Format detection results into a Gradio-compatible JSON dictionary.

    Args:
        result: DetectionResult from the inference engine.
        conf_threshold: Minimum confidence threshold for included detections.

    Returns:
        Dictionary with keys: count, inference_time_ms, device, detections.
    """
    return {
        "count": len(result.detections),
        "inference_time_ms": round(result.inference_time_ms, 2),
        "device": result.device,
        "detections": [
            {
                "class": det.class_name,
                "confidence": round(det.confidence, 3),
                "bbox": det.bbox,
            }
            for det in result.detections
            if det.confidence >= conf_threshold
        ],
    }


def run_inference(
    image: np.ndarray,
    model_name: str,
    custom_model: Optional[str],
    conf_threshold: float,
) -> Dict[str, Any]:
    """Run object detection inference and return formatted results.

    This is the main entry point for the inference tab callback.
    It handles the full inference pipeline: model resolution,
    engine creation, prediction, and result formatting.

    Args:
        image: Input image as numpy array (HWC, BGR or RGB).
        model_name: Model name from text input.
        custom_model: Path to uploaded custom model file, or None.
        conf_threshold: Minimum confidence for included detections.

    Returns:
        Dictionary with inference results or {"error": str} on failure.
    """
    try:
        model_path = resolve_model_path(model_name, custom_model)
    except ValueError as e:
        return {"error": str(e)}

    logger.info("Running inference with model: %s", model_path)

    try:
        engine = create_engine(model_path)
        engine.load_model()
        result: DetectionResult = engine.predict(image)
        return format_detections(result, conf_threshold)
    except FileNotFoundError as e:
        logger.error("Model not found: %s", e)
        return {"error": f"Model not found: {e}"}
    except Exception as e:
        logger.error("Inference failed: %s", e)
        return {"error": str(e)}
