"""CPU backend for YOLO inference (fallback option)."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from .engine import Detection, DetectionResult, InferenceEngine


class CPUBackend(InferenceEngine):
    """YOLO inference backend using CPU (fallback option)."""

    def __init__(self, model_path: str | Path):
        super().__init__(model_path)
        self.device = "cpu"

    def load_model(self) -> None:
        """Load the YOLO model onto CPU."""
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device="cpu", verbose=False)

    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run inference on CPU."""
        start = time.perf_counter()
        results = self.model.predict(image, device="cpu", verbose=False, conf=0.25)
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
            device="cpu",
        )

    def get_device_info(self) -> dict[str, Any]:
        """Get CPU device information."""
        return {
            "backend": "cpu",
            "device": "cpu",
            "available": True,
        }
