"""Inference engine with abstract base class for cross-platform support."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np


@dataclass
class Detection:
    """Single detection result."""

    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionResult:
    """Complete detection result for an image."""

    detections: list[Detection]
    image_shape: tuple[int, int]  # (height, width)
    inference_time_ms: float
    device: str


class InferenceEngine(ABC):
    """Abstract base class for YOLO inference engines."""

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.model = None
        self.device = "cpu"

    @abstractmethod
    def load_model(self) -> None:
        """Load the YOLO model."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> DetectionResult:
        """
        Run inference on an image.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format.

        Returns:
            DetectionResult containing detections and metadata.
        """
        pass

    @abstractmethod
    def get_device_info(self) -> dict[str, Any]:
        """Get information about the current device/backend."""
        pass

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model = None
