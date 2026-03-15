"""Inference module with cross-platform support."""

from .cpu_backend import CPUBackend
from .cuda_backend import CUDABackend
from .engine import Detection, DetectionResult, InferenceEngine
from .mps_backend import MPSBackend
from .rknn_backend import RKNNBackend

import torch


def create_engine(model_path: str) -> InferenceEngine:
    """
    Create an inference engine based on available hardware.

    Priority: CUDA > MPS > CPU

    Args:
        model_path: Path to the YOLO model weights.

    Returns:
        An appropriate InferenceEngine instance.
    """
    # Check if it's an RKNN model (for RK3588 deployment)
    if model_path.endswith(".rknn"):
        return RKNNBackend(model_path)

    if torch.cuda.is_available():
        return CUDABackend(model_path)
    elif torch.backends.mps.is_available():
        return MPSBackend(model_path)
    else:
        return CPUBackend(model_path)


def create_rknn_engine(
    model_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
) -> RKNNBackend:
    """
    Create an RKNN inference engine for RK3588.

    Args:
        model_path: Path to RKNN model file
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        max_det: Maximum number of detections

    Returns:
        RKNNBackend instance
    """
    return RKNNBackend(
        model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
    )


def get_available_backend() -> str:
    """Get the name of the available backend."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


__all__ = [
    "InferenceEngine",
    "CUDABackend",
    "MPSBackend",
    "CPUBackend",
    "RKNNBackend",
    "Detection",
    "DetectionResult",
    "create_engine",
    "create_rknn_engine",
    "get_available_backend",
]
