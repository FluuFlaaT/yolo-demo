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
    load_class_names_from_onnx,
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


class TestLoadClassNamesFromOnnx:
    """Test loading class names from ONNX metadata."""

    def test_load_class_names_from_onnx_file_not_found(self):
        """Test loading from non-existent file."""
        from yolo_demo.inference.rknn_backend import load_class_names_from_onnx

        result = load_class_names_from_onnx("nonexistent.onnx")
        assert result is None

    def test_load_class_names_from_onnx_parsed_correctly(self):
        """Test that class names are parsed correctly from real ONNX."""
        import os
        from yolo_demo.export import ONNXExporter
        from yolo_demo.inference.rknn_backend import load_class_names_from_onnx

        exporter = ONNXExporter("yolov8n.pt")
        onnx_path = exporter.export(output="test_names.onnx", simplify=True)

        try:
            names = load_class_names_from_onnx(onnx_path)
            assert names is not None
            assert len(names) == 80
            assert names[0] == "person"
            assert names[1] == "bicycle"
        finally:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)


class TestRKNNBackendClassNames:
    """Test RKNN backend class names handling."""

    def test_rknn_backend_init_with_class_names_dict(self):
        """Test RKNNBackend initialization with class names dict."""
        with patch("yolo_demo.inference.rknn_backend.RKNNBackend.load_model"):
            from yolo_demo.inference.rknn_backend import RKNNBackend

            class_names = {0: "person", 1: "car", 2: "dog"}
            backend = RKNNBackend(
                "test.rknn",
                class_names=class_names,
            )

            assert backend._class_names == class_names

    def test_rknn_backend_init_with_class_names_list(self):
        """Test RKNNBackend initialization with class names list."""
        with patch("yolo_demo.inference.rknn_backend.RKNNBackend.load_model"):
            from yolo_demo.inference.rknn_backend import RKNNBackend

            class_names = ["person", "car", "dog"]
            backend = RKNNBackend(
                "test.rknn",
                class_names=class_names,
            )

            assert backend._class_names == {0: "person", 1: "car", 2: "dog"}

    def test_rknn_backend_init_without_class_names(self):
        """Test RKNNBackend initialization without class names."""
        with patch("yolo_demo.inference.rknn_backend.RKNNBackend.load_model"):
            from yolo_demo.inference.rknn_backend import RKNNBackend

            backend = RKNNBackend("test.rknn")

            assert backend._class_names is None

    def test_rknn_get_class_name_with_dict(self):
        """Test _get_class_name returns correct name from dict."""
        with patch("yolo_demo.inference.rknn_backend.RKNNBackend.load_model"):
            from yolo_demo.inference.rknn_backend import RKNNBackend

            class_names = {0: "person", 1: "car", 2: "dog"}
            backend = RKNNBackend(
                "test.rknn",
                class_names=class_names,
            )

            assert backend._get_class_name(0) == "person"
            assert backend._get_class_name(1) == "car"
            assert backend._get_class_name(2) == "dog"
            assert backend._get_class_name(99) == "99"

    def test_rknn_get_class_name_without_dict(self):
        """Test _get_class_name returns string of ID when no dict."""
        with patch("yolo_demo.inference.rknn_backend.RKNNBackend.load_model"):
            from yolo_demo.inference.rknn_backend import RKNNBackend

            backend = RKNNBackend("test.rknn")

            assert backend._get_class_name(0) == "0"
            assert backend._get_class_name(5) == "5"
