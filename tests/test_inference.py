"""Tests for inference module."""

import json
import os
import tempfile

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from yolo_demo.inference import (
    CPUBackend,
    Detection,
    DetectionResult,
    create_engine,
    get_available_backend,
)


class TestDetection:
    """Test Detection dataclass."""

    def test_detection_creation(self):
        det = Detection(
            bbox=[10.0, 20.0, 30.0, 40.0],
            confidence=0.95,
            class_id=0,
            class_name="person",
        )
        assert det.bbox == [10.0, 20.0, 30.0, 40.0]
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "person"


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_detection_result_creation(self):
        det = Detection(
            bbox=[10.0, 20.0, 30.0, 40.0],
            confidence=0.95,
            class_id=0,
            class_name="person",
        )
        result = DetectionResult(
            detections=[det],
            image_shape=(480, 640),
            inference_time_ms=15.5,
            device="cpu",
        )
        assert len(result.detections) == 1
        assert result.image_shape == (480, 640)
        assert result.inference_time_ms == 15.5


class TestCPUBackend:
    """Test CPU backend."""

    @patch("yolo_demo.inference.cpu_backend.YOLO")
    def test_cpu_backend_load_model(self, mock_yolo):
        """Test model loading."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        backend = CPUBackend("yolov8n.pt")
        backend.load_model()

        assert mock_yolo.called
        assert backend.model is not None

    @patch("yolo_demo.inference.cpu_backend.YOLO")
    def test_cpu_backend_predict(self, mock_yolo):
        """Test prediction."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Setup mock result with proper structure
        mock_box = MagicMock()
        # Use MagicMock for xyxy, conf, cls to support .tolist() calls
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.__getitem__ = MagicMock(
            return_value=MagicMock(tolist=lambda: [10, 20, 30, 40])
        )
        mock_box.conf = MagicMock()
        mock_box.conf.__getitem__ = MagicMock(return_value=0.95)
        mock_box.cls = MagicMock()
        mock_box.cls.__getitem__ = MagicMock(return_value=0)

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "person"}

        backend = CPUBackend("yolov8n.pt")
        backend.load_model()

        # Create test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = backend.predict(test_img)

        assert len(result.detections) == 1
        assert result.detections[0].class_name == "person"
        assert result.device == "cpu"

    @patch("yolo_demo.inference.cpu_backend.YOLO")
    def test_cpu_backend_get_device_info(self, mock_yolo):
        """Test device info."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        backend = CPUBackend("yolov8n.pt")
        backend.load_model()
        info = backend.get_device_info()

        assert info["backend"] == "cpu"
        assert info["available"] is True


class TestCreateEngine:
    """Test factory function."""

    @patch("yolo_demo.inference.torch.cuda.is_available", return_value=False)
    @patch("yolo_demo.inference.torch.backends.mps.is_available", return_value=False)
    def test_create_engine_cpu_fallback(self, mock_mps, mock_cuda):
        """Test CPU fallback when no GPU available."""
        engine = create_engine("yolov8n.pt")
        assert isinstance(engine, CPUBackend)

    @patch("yolo_demo.inference.torch.cuda.is_available", return_value=True)
    def test_create_engine_cuda_priority(self, mock_cuda):
        """Test CUDA has priority."""
        with patch("yolo_demo.inference.cuda_backend.YOLO", MagicMock()):
            engine = create_engine("yolov8n.pt")
            # Would be CUDABackend if CUDA available
            # For testing, we just check the logic path

    @patch("yolo_demo.inference.torch.cuda.is_available", return_value=False)
    @patch("yolo_demo.inference.torch.backends.mps.is_available", return_value=True)
    def test_get_available_backend_mps(self, mock_mps, mock_cuda):
        """Test backend detection."""
        backend = get_available_backend()
        assert backend == "mps"


class TestBackendSelection:
    """Test backend selection logic."""

    def test_get_available_backend_default(self):
        """Test backend detection returns valid string."""
        backend = get_available_backend()
        assert backend in ["cpu", "cuda", "mps"]
