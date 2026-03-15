"""Mac MPS backend for YOLO inference."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from .engine import Detection, DetectionResult, InferenceEngine


class MPSBackend(InferenceEngine):
    """YOLO inference backend using Apple Metal Performance Shaders (MPS)."""

    def __init__(self, model_path: str | Path):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this system")
        super().__init__(model_path)
        self.device = "mps"

    def load_model(self) -> None:
        """Load the YOLO model onto MPS device."""
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False)

    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run inference on MPS device."""
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
            device="mps",
        )

    def get_device_info(self) -> dict[str, Any]:
        """Get MPS device information."""
        return {
            "backend": "mps",
            "device": "mps",
            "available": torch.backends.mps.is_available(),
            "memory_allocated": torch.mps.current_allocated_memory()
            if hasattr(torch.mps, "current_allocated_memory")
            else 0,
        }
