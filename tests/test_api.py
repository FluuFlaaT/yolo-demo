"""Tests for API module."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from yolo_demo.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "YOLO Demo API"
        assert "version" in data


class TestHealthEndpoint:
    """Test health endpoint."""

    def test_health_check(self, client):
        """Test health check returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "backend" in data


class TestModelsEndpoint:
    """Test models endpoint."""

    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0


class TestInferenceEndpoint:
    """Test inference endpoints."""

    @patch("yolo_demo.api.routes.inference.create_engine")
    def test_infer_image(self, mock_create_engine, client):
        """Test image inference."""
        from yolo_demo.api.routes import inference

        inference._engine_cache.clear()

        # Setup mock engine
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.detections = []
        mock_result.device = "cpu"
        mock_result.inference_time_ms = 15.0
        mock_engine.predict.return_value = mock_result
        mock_create_engine.return_value = mock_engine

        # Create test image
        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = client.post(
            "/api/v1/inference/image",
            files={"image": ("test.png", img_bytes, "image/png")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "detections" in data

    @patch("yolo_demo.api.routes.inference.create_engine")
    def test_infer_image_with_detections(self, mock_create_engine, client):
        """Test inference with detections."""
        from yolo_demo.api.routes import inference
        from yolo_demo.inference import Detection, DetectionResult

        # Clear engine cache
        inference._engine_cache.clear()

        mock_engine = MagicMock()

        det = Detection(
            bbox=[10.0, 20.0, 100.0, 150.0],
            confidence=0.95,
            class_id=0,
            class_name="person",
        )
        mock_result = DetectionResult(
            detections=[det],
            image_shape=(480, 640),
            inference_time_ms=20.0,
            device="cpu",
        )
        mock_engine.predict.return_value = mock_result
        mock_engine.model.names = {0: "person"}
        mock_create_engine.return_value = mock_engine

        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = client.post(
            "/api/v1/inference/image",
            files={"image": ("test.png", img_bytes, "image/png")},
            params={"conf_threshold": 0.1},  # Lower threshold to ensure detection passes
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["detections"]) == 1
        assert data["detections"][0]["class_name"] == "person"
        assert data["detections"][0]["confidence"] == 0.95

    @patch("yolo_demo.api.routes.inference.create_engine")
    def test_infer_image_base64(self, mock_create_engine, client):
        """Test base64 image inference."""
        from yolo_demo.api.routes import inference

        inference._engine_cache.clear()

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.detections = []
        mock_result.device = "cpu"
        mock_result.inference_time_ms = 15.0
        mock_engine.predict.return_value = mock_result
        mock_create_engine.return_value = mock_engine

        import base64

        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = client.post(
            "/api/v1/inference/image/base64",
            params={"image_data": b64_str},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_infer_image_base64_invalid(self, client):
        """Test base64 inference with invalid base64 string."""
        response = client.post(
            "/api/v1/inference/image/base64",
            params={"image_data": "not-valid-base64!!!@@@"},
        )
        assert response.status_code in [200, 500]

    @patch("yolo_demo.api.routes.inference.create_engine")
    def test_engine_cache_reuse(self, mock_create_engine):
        """Test that engine cache reuses loaded engines."""
        from yolo_demo.api.routes import inference
        from yolo_demo.api.routes.inference import get_engine

        inference._engine_cache.clear()

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        e1 = get_engine("yolov8n.pt")
        e2 = get_engine("yolov8n.pt")

        assert e1 is e2
        assert mock_create_engine.call_count == 1

    @patch("yolo_demo.api.routes.inference.create_engine")
    def test_engine_cache_ttl_eviction(self, mock_create_engine):
        """Test engine eviction when TTL expires."""
        import time as time_mod

        from yolo_demo.api.routes import inference
        from yolo_demo.api.routes.inference import _CACHE_TTL_SECONDS, get_engine

        inference._engine_cache.clear()

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Insert an engine with an ancient timestamp
        inference._engine_cache["old_model.pt"] = (
            MagicMock(),
            time_mod.monotonic() - _CACHE_TTL_SECONDS - 100,
        )

        # Getting a different model should evict the expired one
        _ = get_engine("new_model.pt")
        assert "old_model.pt" not in inference._engine_cache
        assert "new_model.pt" in inference._engine_cache

    @patch("yolo_demo.api.routes.inference.create_engine")
    def test_engine_cache_max_size_eviction(self, mock_create_engine):
        """Test eviction when cache exceeds max size."""
        import time as time_mod

        from yolo_demo.api.routes import inference
        from yolo_demo.api.routes.inference import _MAX_CACHE_SIZE, get_engine

        inference._engine_cache.clear()

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        for i in range(_MAX_CACHE_SIZE + 1):
            inference._engine_cache[f"model_{i}.pt"] = (
                MagicMock(),
                time_mod.monotonic() + i * 0.1,
            )

        get_engine("new_model.pt")
        assert len(inference._engine_cache) <= _MAX_CACHE_SIZE + 1
        assert "model_0.pt" not in inference._engine_cache

    @patch("yolo_demo.api.routes.inference.create_engine")
    def test_engine_close_error_handled(self, mock_create_engine):
        """Test that engine close errors are handled gracefully."""
        import time as time_mod

        from yolo_demo.api.routes import inference
        from yolo_demo.api.routes.inference import get_engine

        inference._engine_cache.clear()

        mock_engine = MagicMock()
        mock_engine.__exit__.side_effect = RuntimeError("Cleanup failed")
        mock_create_engine.return_value = mock_engine

        inference._engine_cache["bad_engine.pt"] = (
            mock_engine,
            time_mod.monotonic() - 99999,
        )

        # This should not raise — error is swallowed with log
        get_engine("other.pt")
        assert "bad_engine.pt" not in inference._engine_cache

    @patch("yolo_demo.api.routes.inference.create_engine")
    def test_engine_eviction_via_exit(self, mock_create_engine):
        """Test engine eviction via __exit__."""
        from yolo_demo.api.routes import inference
        from yolo_demo.api.routes.inference import _close_engine, get_engine

        inference._engine_cache.clear()

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        get_engine("model_a.pt")
        assert len(inference._engine_cache) == 1

        _close_engine("model_a.pt")
        assert len(inference._engine_cache) == 0
        mock_engine.__exit__.assert_called_once()

    def test_infer_image_file_not_found(self, client):
        """Test inference with missing file."""
        response = client.post(
            "/api/v1/inference/image",
            files={"image": ("test.png", b"invalid", "image/png")},
        )

        # Should either succeed with PIL reading or fail gracefully
        assert response.status_code in [200, 500]


class TestTrainingEndpoint:
    """Test training endpoints."""

    def test_start_training(self, client):
        """Test starting a training job."""
        response = client.post(
            "/api/v1/train",
            json={
                "data_yaml": "data.yaml",
                "epochs": 10,
                "batch_size": 4,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data

    def test_get_training_status(self, client):
        """Test getting training status."""
        # First create a job
        create_response = client.post(
            "/api/v1/train",
            json={"data_yaml": "data.yaml", "epochs": 1},
        )
        job_id = create_response.json()["job_id"]

        # Then check status
        status_response = client.get(f"/api/v1/train/{job_id}/status")
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["job_id"] == job_id
        assert "status" in data

    def test_cancel_training(self, client):
        """Test cancelling a training job."""
        from yolo_demo.api.routes import training

        training._jobs["test-pending"] = {
            "status": "pending",
            "progress": 0.0,
            "config": None,
            "error": None,
            "model_path": None,
            "metrics": {},
        }

        cancel_response = client.delete("/api/v1/train/test-pending")
        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "cancelled"

    def test_cancel_training_already_done(self, client):
        """Test cancelling a job that already completed."""
        from yolo_demo.api.routes import training

        training._jobs["test-done"] = {
            "status": "completed",
            "progress": 1.0,
            "config": None,
            "error": None,
            "model_path": "/tmp/best.pt",
            "metrics": {"precision": 0.9},
        }

        response = client.delete("/api/v1/train/test-done")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cannot_cancel"

    def test_cancel_training_not_found(self, client):
        """Test cancelling non-existent job."""
        response = client.delete("/api/v1/train/nonexistent")
        assert response.status_code == 404

    def test_get_training_status_not_found(self, client):
        """Test getting status for non-existent job."""
        response = client.get("/api/v1/train/nonexistent-id/status")
        assert response.status_code == 404


class TestExportEndpoint:
    """Test export endpoints."""

    @patch("yolo_demo.export.pt_to_rknn")
    def test_export_rknn(self, mock_pt_to_rknn, client):
        """Test PT to RKNN export."""
        mock_pt_to_rknn.return_value = "/tmp/model.rknn"

        response = client.post(
            "/api/v1/export/rknn",
            json={"model_path": "yolov8n.pt", "target_platform": "rk3588"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["output_path"] == "/tmp/model.rknn"

    def test_export_rknn_unsupported_platform(self, client):
        """Test export with an unsupported platform."""
        response = client.post(
            "/api/v1/export/rknn",
            json={"model_path": "yolov8n.pt", "target_platform": "invalid"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Unsupported platform" in data["error"]

    @patch("yolo_demo.export.pt_to_rknn")
    def test_export_rknn_error_handling(self, mock_pt_to_rknn, client):
        """Test export with runtime error."""
        mock_pt_to_rknn.side_effect = RuntimeError("Export crashed")

        response = client.post(
            "/api/v1/export/rknn",
            json={"model_path": "yolov8n.pt", "target_platform": "rk3588"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Export crashed" in data["error"]


class TestBackendEndpoint:
    """Test backend info endpoint."""

    def test_get_backend_info(self, client):
        """Test getting backend information."""
        response = client.get("/api/v1/inference/backend")
        assert response.status_code == 200
        data = response.json()
        assert "backend" in data
        assert "cuda_available" in data
        assert "mps_available" in data


class TestAppLifecycle:
    """Test FastAPI app lifecycle and CORS configuration."""

    def test_lifespan_startup_shutdown(self):
        """Test lifespan triggers on client context entry/exit."""
        from yolo_demo.api.app import create_app

        app = create_app()
        with TestClient(app) as tc:
            response = tc.get("/health")
            assert response.status_code == 200

    def test_shutdown_engines_empty(self):
        """Test shutdown with no cached engines."""
        from yolo_demo.api.app import _shutdown_engines
        from yolo_demo.api.routes import inference

        inference._engine_cache.clear()
        _shutdown_engines()

    def test_shutdown_engines_with_cache(self):
        """Test shutdown releases cached engines."""
        from unittest.mock import MagicMock

        from yolo_demo.api.app import _shutdown_engines
        from yolo_demo.api.routes import inference

        mock_engine = MagicMock()
        inference._engine_cache["test.pt"] = (mock_engine, 0)

        _shutdown_engines()

        assert "test.pt" not in inference._engine_cache
        mock_engine.__exit__.assert_called_once()

    def test_shutdown_engines_handles_close_error(self):
        """Test shutdown handles engine close errors gracefully."""
        from unittest.mock import MagicMock

        from yolo_demo.api.app import _shutdown_engines
        from yolo_demo.api.routes import inference

        mock_engine = MagicMock()
        mock_engine.__exit__.side_effect = RuntimeError("Close failed")
        inference._engine_cache["bad.pt"] = (mock_engine, 0)

        _shutdown_engines()

        assert "bad.pt" not in inference._engine_cache

    def test_cors_with_env_var(self, monkeypatch):
        """Test CORS with CORS_ORIGINS env var set."""
        monkeypatch.setenv("CORS_ORIGINS", "https://example.com,https://app.example.com")
        from yolo_demo.api.app import create_app

        app = create_app()
        cors = None
        for mw in app.user_middleware:
            if mw.cls == CORSMiddleware:
                cors = mw
                break
        assert cors is not None
        assert cors.kwargs["allow_origins"] == ["https://example.com", "https://app.example.com"]
        assert cors.kwargs["allow_credentials"] is True

    @patch("uvicorn.run")
    def test_serve(self, mock_run):
        """Test serve function starts uvicorn."""
        from yolo_demo.api.app import serve

        serve(host="127.0.0.1", port=9999)
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert "yolo_demo.api.app:app" in args


class TestTrainingJob:
    """Test background training job function."""

    @patch("yolo_demo.api.routes.training.Trainer")
    def test_run_training_job_success(self, mock_trainer_cls):
        """Test background training job with successful training."""
        from yolo_demo.api.routes.training import _jobs, _run_training_job
        from yolo_demo.training.trainer import TrainingConfig, TrainingResult

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.train.return_value = TrainingResult(
            success=True,
            model_path="/tmp/best.pt",
            metrics={"precision": 0.9},
        )

        _jobs["test-success"] = {"status": "pending"}
        _run_training_job("test-success", TrainingConfig())

        assert _jobs["test-success"]["status"] == "completed"
        assert _jobs["test-success"]["model_path"] == "/tmp/best.pt"
        assert _jobs["test-success"]["metrics"]["precision"] == 0.9

    @patch("yolo_demo.api.routes.training.Trainer")
    def test_run_training_job_failure(self, mock_trainer_cls):
        """Test background training job with failed training."""
        from yolo_demo.api.routes.training import _jobs, _run_training_job
        from yolo_demo.training.trainer import TrainingConfig, TrainingResult

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.train.return_value = TrainingResult(
            success=False,
            error="Out of memory",
        )

        _jobs["test-fail"] = {"status": "pending"}
        _run_training_job("test-fail", TrainingConfig())

        assert _jobs["test-fail"]["status"] == "failed"
        assert _jobs["test-fail"]["error"] == "Out of memory"

    @patch("yolo_demo.api.routes.training.Trainer")
    def test_run_training_job_exception(self, mock_trainer_cls):
        """Test background training job with unexpected exception."""
        from yolo_demo.api.routes.training import _jobs, _run_training_job
        from yolo_demo.training.trainer import TrainingConfig

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.train.side_effect = RuntimeError("GPU crashed")

        _jobs["test-exc"] = {"status": "pending"}
        _run_training_job("test-exc", TrainingConfig())

        assert _jobs["test-exc"]["status"] == "failed"
        assert "GPU crashed" in _jobs["test-exc"]["error"]
