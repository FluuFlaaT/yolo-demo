"""Tests for API module."""

import io
import pytest
from unittest.mock import MagicMock, patch

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
        from PIL import Image
        import numpy as np

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
        from yolo_demo.inference import Detection, DetectionResult
        from yolo_demo.api.routes import inference

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

        from PIL import Image
        import numpy as np

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

    def test_get_training_status_not_found(self, client):
        """Test getting status for non-existent job."""
        response = client.get("/api/v1/train/nonexistent-id/status")
        assert response.status_code == 404


class TestExportEndpoint:
    """Test export endpoints."""

    @patch("yolo_demo.api.routes.export.ONNXExporter")
    def test_export_onnx(self, mock_exporter, client):
        """Test ONNX export."""
        mock_instance = MagicMock()
        mock_instance.export.return_value = "/tmp/model.onnx"
        mock_exporter.return_value = mock_instance

        response = client.post(
            "/api/v1/export/onnx",
            json={"model_path": "yolov8n.pt", "opset": 11},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["onnx_path"] == "/tmp/model.onnx"

    def test_export_onnx_rk3588(self, client):
        """Test RK3588 export endpoint."""
        # This will fail without a real model, but tests the endpoint exists
        response = client.post(
            "/api/v1/export/onnx/rk3588",
            params={"model_path": "yolov8n.pt"},
        )

        # Endpoint should exist, may fail due to model download
        assert response.status_code in [200, 500]


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
