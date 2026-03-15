"""RKNN backend for RK3588 NPU inference."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from yolo_demo.inference.engine import Detection, DetectionResult, InferenceEngine

logger = logging.getLogger(__name__)


class RKNNBackend(InferenceEngine):
    """
    RKNN inference engine for RK3588 NPU.

    This backend uses Rockchip's RKNN toolkit for hardware-accelerated
    inference on RK3588 and compatible SoCs.

    Requirements:
        - rknn-toolkit2 (runtime, included with RKNN SDK)
        - Rockchip NPU drivers (pre-installed on RK3588 devices)
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ):
        """
        Initialize RKNN backend.

        Args:
            model_path: Path to RKNN model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_det: Maximum number of detections
        """
        super().__init__(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.input_name = None
        self.input_shape = None
        self._rknn = None

    def load_model(self) -> None:
        """Load RKNN model."""
        try:
            from rknn.api import RKNN
        except ImportError as e:
            raise ImportError(
                "rknn-toolkit2 not installed. On RK3588, install via:\n"
                "  sudo apt-get install rknn-toolkit2\n"
                "or use the pre-built SDK from Rockchip."
            ) from e

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"RKNN model not found: {model_path}")

        logger.info(f"Loading RKNN model: {model_path}")

        self._rknn = RKNN()

        # Load model
        ret = self._rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model (error code: {ret})")

        # Get input info
        inputs = self._rknn.inputs
        if len(inputs) == 0:
            raise RuntimeError("No inputs found in RKNN model")

        self.input_name = inputs[0]["name"]
        self.input_shape = inputs[0]["shape"]  # [N, C, H, W]

        logger.info(f"Input name: {self.input_name}")
        logger.info(f"Input shape: {self.input_shape}")

        self.device = "rknn_npu"
        logger.info("RKNN model loaded successfully")

    def predict(self, image: np.ndarray) -> DetectionResult:
        """
        Run inference on an image.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format.

        Returns:
            DetectionResult containing detections and metadata.
        """
        if self._rknn is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        h, w = image.shape[:2]
        start_time = time.time()

        # Resize and normalize input
        input_tensor = self._preprocess(image)

        # Run inference
        outputs = self._rknn.inference(inputs=[input_tensor])

        # Post-process outputs
        detections = self._postprocess(outputs, h, w)

        inference_time_ms = (time.time() - start_time) * 1000

        return DetectionResult(
            detections=detections,
            image_shape=(h, w),
            inference_time_ms=inference_time_ms,
            device=self.device,
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for RKNN input.

        Args:
            image: RGB image as numpy array (H, W, C)

        Returns:
            Preprocessed tensor (1, C, H, W) normalized to [0, 1]
        """
        import cv2

        # Get target input size
        target_h, target_w = self.input_shape[2], self.input_shape[3]

        # Resize image
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Convert to CHW format and normalize
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension

        return tensor

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        img_h: int,
        img_w: int,
    ) -> list[Detection]:
        """
        Post-process RKNN outputs.

        Args:
            outputs: Raw outputs from RKNN model
            img_h: Original image height
            img_w: Original image width

        Returns:
            List of Detection objects
        """
        # YOLOv8 output format: [N, 84, 8400] or [N, 25200, 84]
        # Need to parse based on model output shape
        output = outputs[0]

        # Handle different output formats
        if output.ndim == 3:
            if output.shape[1] > output.shape[2]:
                # Format: [N, 84, 8400]
                output = np.transpose(output[0], (1, 0))  # [8400, 84]
            else:
                # Format: [N, 8400, 84]
                output = output[0]  # [8400, 84]
        else:
            output = output.reshape(-1, output.shape[-1])

        # Parse boxes and scores
        # YOLOv8: [x, y, w, h, scores...]
        boxes = output[:, :4]  # [N, 4]
        scores = output[:, 4:]  # [N, 80]

        # Get max scores and class ids
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # Filter by confidence threshold
        mask = max_scores > self.conf_threshold
        boxes = boxes[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert to [x1, y1, x2, y2]
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        box_w = boxes[:, 2]
        box_h = boxes[:, 3]

        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Scale boxes to original image size
        scale_factor = min(
            self.input_shape[2] / img_h,
            self.input_shape[3] / img_w,
        )
        boxes /= scale_factor

        # Clip to image bounds
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_h)

        # Apply NMS
        indices = self._nms(boxes, scores, self.iou_threshold)

        # Limit to max_det
        indices = indices[: self.max_det]

        # Build detection list
        detections = []
        for idx in indices:
            det = Detection(
                bbox=boxes[idx].tolist(),
                confidence=float(scores[idx]),
                class_id=int(class_ids[idx]),
                class_name=str(class_ids[idx]),  # Will be mapped by caller
            )
            detections.append(det)

        return detections

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """
        Apply Non-Maximum Suppression.

        Args:
            boxes: Boxes [N, 4]
            scores: Scores [N]
            iou_threshold: IoU threshold

        Returns:
            Indices of selected boxes
        """
        import cv2

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            iou_threshold,
        )

        if len(indices) > 0:
            return [int(i) for i in indices.flatten()]
        return []

    def get_device_info(self) -> dict[str, Any]:
        """Get information about the RKNN device."""
        return {
            "backend": "rknn_npu",
            "target_platform": "rk3588",
            "model_path": str(self.model_path),
            "input_shape": self.input_shape,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
        }

    def __del__(self):
        """Cleanup RKNN resources."""
        if hasattr(self, "_rknn") and self._rknn is not None:
            try:
                self._rknn.release()
            except Exception:
                pass
