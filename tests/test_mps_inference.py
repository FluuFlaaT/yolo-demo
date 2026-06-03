"""Real MPS (Apple Silicon GPU) inference tests.

These tests require a macOS machine with Apple Silicon and PyTorch built with MPS support.
On non-MPS systems, all tests are automatically skipped via ``pytest.skip``.

The model ``yolo11n.pt`` (~5.4 MB) is auto-downloaded by ultralytics if not already cached.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.mps

MODEL_NAME = "yolo11n.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_mps() -> None:
    """Skip the current test if MPS is unavailable."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this system")


def _ensure_model_downloaded() -> None:
    """Trigger ultralytics auto-download of the model if not already present."""
    from ultralytics import YOLO

    YOLO(MODEL_NAME)  # auto-downloads model weights if missing


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mps_engine():
    """Create and pre-warm an MPSBackend instance (shared across module tests)."""
    _require_mps()
    _ensure_model_downloaded()

    from yolo_demo.inference import MPSBackend

    engine = MPSBackend(MODEL_NAME)
    engine.load_model()  # includes warm-up inference
    yield engine
    # Cleanup – release model from GPU memory
    engine.model = None


@pytest.fixture(scope="module")
def blank_image() -> np.ndarray:
    """640×640 blank image (all zeros) for smoke‑level inference."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestMPSBackendInstantiation:
    """Test that MPSBackend can be created on Apple Silicon."""

    def test_instantiate_and_device(self) -> None:
        """MPSBackend sets device='mps' and raises on unsupported hardware."""
        _require_mps()
        _ensure_model_downloaded()

        from yolo_demo.inference import MPSBackend

        backend = MPSBackend(MODEL_NAME)
        assert backend.device == "mps"
        assert str(backend.model_path).endswith(MODEL_NAME)

    def test_raises_without_mps(self, monkeypatch) -> None:
        """Verify RuntimeError when MPS is falsely reported as unavailable."""
        monkeypatch.setattr(
            "yolo_demo.inference.mps_backend.torch.backends.mps.is_available",
            lambda: False,
        )
        from yolo_demo.inference import MPSBackend

        with pytest.raises(RuntimeError, match="MPS is not available"):
            MPSBackend("does_not_exist.pt")


class TestMPSModelLoading:
    """Test model load and warm-up on MPS."""

    def test_load_model(self, mps_engine) -> None:
        """Model attribute is set after load_model()."""
        assert mps_engine.model is not None
        assert mps_engine.device == "mps"


class TestMPSInference:
    """Test real MPS inference pipeline."""

    def test_predict_returns_result(self, mps_engine, blank_image) -> None:
        """predict() returns a DetectionResult with device='mps'."""
        result = mps_engine.predict(blank_image)
        assert result.device == "mps"
        assert result.image_shape == (640, 640)
        assert isinstance(result.inference_time_ms, float)
        assert result.inference_time_ms > 0
        assert isinstance(result.detections, list)

    def test_predict_with_visible_object(self, mps_engine) -> None:
        """Inference on a synthetic image containing a salient shape."""
        # Create a 640×640 image with a prominent white rectangle
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[200:440, 200:440, :] = 255  # large white square

        result = mps_engine.predict(img)
        assert result.device == "mps"
        assert result.image_shape == (640, 640)
        # Some detections *may* appear; at minimum the call must not crash
        assert isinstance(result.detections, list)

    def test_multiple_inferences(self, mps_engine, blank_image) -> None:
        """Multiple consecutive inferences are stable."""
        results = []
        for _ in range(5):
            results.append(mps_engine.predict(blank_image))

        times = [r.inference_time_ms for r in results]
        # After warm-up all calls should complete in under 500 ms each
        for t in times:
            assert 0 < t < 500, f"Inference took {t:.1f} ms, expected < 500 ms"
        assert all(r.device == "mps" for r in results)


class TestMPSDeviceInfo:
    """Test get_device_info() for MPS backend."""

    def test_get_device_info(self, mps_engine) -> None:
        """Returns dict with backend='mps' and availability flag."""
        info = mps_engine.get_device_info()
        assert info["backend"] == "mps"
        assert info["available"] is True
        assert "memory_allocated" in info
        assert isinstance(info["memory_allocated"], int)

    def test_get_device_info_before_load(self) -> None:
        """get_device_info() works even before load_model() (availability only)."""
        _require_mps()
        from yolo_demo.inference import MPSBackend

        backend = MPSBackend(MODEL_NAME)
        info = backend.get_device_info()
        assert info["backend"] == "mps"
        assert info["available"] is True


class TestMPSEngineSelection:
    """Test that create_engine() auto‑selects MPS on Apple Silicon."""

    def test_create_engine_selects_mps(self) -> None:
        """create_engine() returns MPSBackend when MPS is available."""
        _require_mps()
        _ensure_model_downloaded()

        from yolo_demo.inference import MPSBackend, create_engine

        engine = create_engine(MODEL_NAME)
        assert isinstance(engine, MPSBackend)

    def test_get_available_backend_returns_mps(self) -> None:
        """get_available_backend() returns 'mps' on Apple Silicon."""
        _require_mps()
        from yolo_demo.inference import get_available_backend

        assert get_available_backend() == "mps"
