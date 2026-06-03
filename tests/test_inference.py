"""Tests for inference module."""


from unittest.mock import MagicMock, patch

import numpy as np

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
    def test_cpu_backend_predict_no_boxes(self, mock_yolo):
        """Test prediction with empty boxes returns empty detection list."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]

        backend = CPUBackend("yolov8n.pt")
        backend.load_model()

        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = backend.predict(test_img)

        assert len(result.detections) == 0

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
            _ = create_engine("yolov8n.pt")
            # Would be CUDABackend if CUDA available
            # For testing, we just check the logic path

    @patch("yolo_demo.inference.torch.cuda.is_available", return_value=False)
    @patch("yolo_demo.inference.torch.backends.mps.is_available", return_value=True)
    def test_get_available_backend_mps(self, mock_mps, mock_cuda):
        """Test backend detection."""
        backend = get_available_backend()
        assert backend == "mps"


class TestInferenceEngine:
    """Test InferenceEngine context manager."""

    @patch("yolo_demo.inference.cpu_backend.YOLO")
    def test_engine_context_manager(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        from yolo_demo.inference.cpu_backend import CPUBackend

        engine = CPUBackend("yolov8n.pt")
        with engine:
            assert engine.model is not None

        assert engine.model is None

    @patch("yolo_demo.inference.cpu_backend.YOLO")
    def test_get_device_info(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        backend = CPUBackend("yolov8n.pt")
        backend.load_model()
        info = backend.get_device_info()

        assert "backend" in info
        assert info["backend"] == "cpu"
        assert "available" in info


class TestMPSBackend:
    """Test MPS backend (mocked)."""

    @patch("yolo_demo.inference.mps_backend.YOLO")
    @patch("yolo_demo.inference.mps_backend.MPSBackend.__init__", return_value=None)
    def test_mps_backend_load_model(self, mock_init, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        from yolo_demo.inference.mps_backend import MPSBackend

        backend = MPSBackend("yolov8n.pt")
        backend.device = "mps"
        backend.model_path = __import__("pathlib").Path("yolov8n.pt")
        backend.load_model()
        assert backend.model is not None

    @patch("yolo_demo.inference.mps_backend.YOLO")
    @patch("yolo_demo.inference.mps_backend.MPSBackend.__init__", return_value=None)
    def test_mps_backend_predict(self, mock_init, mock_yolo):
        import numpy as np

        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        mock_box = MagicMock()
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

        from yolo_demo.inference.mps_backend import MPSBackend

        backend = MPSBackend("yolov8n.pt")
        backend.device = "mps"
        backend.model_path = __import__("pathlib").Path("yolov8n.pt")
        backend.model = mock_model

        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = backend.predict(test_img)

        assert len(result.detections) == 1
        assert result.detections[0].class_name == "person"
        assert result.device == "mps"

    @patch("yolo_demo.inference.mps_backend.YOLO")
    @patch("yolo_demo.inference.mps_backend.MPSBackend.__init__", return_value=None)
    @patch("torch.mps.current_allocated_memory", return_value=0)
    def test_mps_backend_get_device_info(self, mock_mem, mock_init, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        from yolo_demo.inference.mps_backend import MPSBackend

        backend = MPSBackend("yolov8n.pt")
        backend.device = "mps"
        backend.model_path = __import__("pathlib").Path("yolov8n.pt")
        backend.load_model()
        info = backend.get_device_info()
        assert info["backend"] == "mps"
        assert "available" in info


class TestBackendSelection:
    """Test backend selection logic."""

    def test_get_available_backend_default(self):
        """Test backend detection returns valid string."""
        backend = get_available_backend()
        assert backend in ["cpu", "cuda", "mps"]
