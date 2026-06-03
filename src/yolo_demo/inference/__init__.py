"""Inference module with cross-platform support."""

import torch

from .cpu_backend import CPUBackend
from .cuda_backend import CUDABackend
from .engine import Detection, DetectionResult, InferenceEngine
from .mps_backend import MPSBackend


def create_engine(model_path: str) -> InferenceEngine:
    """
    Create an inference engine based on available hardware.

    Priority: CUDA > MPS > CPU

    Args:
        model_path: Path to the YOLO model weights.

    Returns:
        An appropriate InferenceEngine instance.
    """
    if torch.cuda.is_available():
        return CUDABackend(model_path)
    elif torch.backends.mps.is_available():
        return MPSBackend(model_path)
    else:
        return CPUBackend(model_path)


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
    "Detection",
    "DetectionResult",
    "create_engine",
    "get_available_backend",
]
