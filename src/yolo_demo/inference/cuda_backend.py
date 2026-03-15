"""NVIDIA CUDA backend for YOLO inference."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from .engine import Detection, DetectionResult, InferenceEngine


class CUDABackend(InferenceEngine):
    """YOLO inference backend using NVIDIA CUDA."""

    def __init__(self, model_path: str | Path):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")
        super().__init__(model_path)
        self.device = "cuda"

    def load_model(self) -> None:
        """Load the YOLO model onto CUDA device."""
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False)

    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run inference on CUDA device."""
        start = time.perf_counter()
        results = self.model.predict(
            image, device=self.device, verbose=False, conf=0.25
        )
        elapsed = (time.perf_counter() - start) * 1000

        result = results[0]
        detections = []
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                detections.append(
                    Detection(
                        bbox=box.xyxy[0].tolist(),
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                        class_name=self.model.names[int(box.cls[0])],
                    )
                )

        return DetectionResult(
            detections=detections,
            image_shape=(image.shape[0], image.shape[1]),
            inference_time_ms=elapsed,
            device=f"cuda:{torch.cuda.current_device()}",
        )

    def get_device_info(self) -> dict[str, Any]:
        """Get CUDA device information."""
        return {
            "backend": "cuda",
            "device": f"cuda:{torch.cuda.current_device()}",
            "available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "memory_allocated": torch.cuda.memory_allocated(0)
            if torch.cuda.is_available()
            else 0,
        }
