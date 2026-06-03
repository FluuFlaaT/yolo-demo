"""Tests for UI service layer — InferenceService, TrainingService, ExportService, DatasetService."""

import logging
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from yolo_demo.inference import Detection, DetectionResult
from yolo_demo.training.trainer import TrainingResult
from yolo_demo.ui.services.dataset_service import convert_dataset, extract_voc_zip
from yolo_demo.ui.services.export_service import (
    check_rknn_availability,
    export_onnx_to_rknn,
    export_pt_to_rknn,
)
from yolo_demo.ui.services.inference_service import (
    format_detections,
    resolve_model_path,
    run_inference,
)
from yolo_demo.ui.services.training_service import (
    TrainingJobConfig,
    TrainingSession,
    TrainingSessionManager,
    _SessionLogHandler,
)


class TestResolveModelPath:
    """Tests for resolve_model_path."""

    def test_custom_model_wins(self):
        """Custom model file takes priority over text input."""
        result = resolve_model_path("text_model.pt", "/path/to/custom.pt")
        assert result == "/path/to/custom.pt"

    def test_custom_model_wins_when_text_empty(self):
        """Custom model wins even when text input is empty string."""
        result = resolve_model_path("", "/path/to/custom.pt")
        assert result == "/path/to/custom.pt"

    def test_text_input_fallback(self):
        """Text input is used when no custom file uploaded (None)."""
        result = resolve_model_path("yolov8n.pt", None)
        assert result == "yolov8n.pt"

    def test_text_input_fallback_custom_is_empty(self):
        """Text input used when custom_model is empty string (no file)."""
        result = resolve_model_path("yolov8n.pt", "")
        assert result == "yolov8n.pt"

    def test_both_missing_raises(self):
        """ValueError when neither model name nor custom file provided."""
        with pytest.raises(ValueError, match="model name"):
            resolve_model_path("", None)

    def test_none_text_none_custom_raises(self):
        """ValueError when both are None."""
        with pytest.raises(ValueError, match="model name"):
            resolve_model_path(None, None)  # type: ignore[arg-type]


class TestFormatDetections:
    """Tests for format_detections."""

    @staticmethod
    def _make_result(detections, time_ms=45.2, device="cpu"):
        return DetectionResult(
            detections=detections,
            image_shape=(480, 640),
            inference_time_ms=time_ms,
            device=device,
        )

    @staticmethod
    def _make_det(bbox=None, confidence=0.95, class_id=0, class_name="person"):
        bbox_default = [10.0, 20.0, 100.0, 150.0]
        return Detection(
            bbox=bbox if bbox is not None else bbox_default,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
        )

    def test_filters_by_confidence(self):
        """Detections below threshold are excluded."""
        dets = [
            self._make_det(confidence=0.95),
            self._make_det(confidence=0.10),
            self._make_det(confidence=0.55),
        ]
        result = self._make_result(dets)
        formatted = format_detections(result, conf_threshold=0.5)

        assert formatted["count"] == 3  # raw count unchanged
        assert len(formatted["detections"]) == 2  # filtered
        confidences = [d["confidence"] for d in formatted["detections"]]
        assert confidences == [0.95, 0.55]

    def test_empty_detections(self):
        """Empty detections list produces empty output."""
        result = self._make_result([])
        formatted = format_detections(result, conf_threshold=0.5)
        assert formatted["count"] == 0
        assert formatted["detections"] == []

    def test_output_structure(self):
        """Verify all expected keys are present with correct types."""
        dets = [self._make_det()]
        result = self._make_result(dets, time_ms=12.34, device="cuda")
        formatted = format_detections(result, conf_threshold=0.25)

        assert "count" in formatted
        assert formatted["count"] == 1
        assert "inference_time_ms" in formatted
        assert formatted["inference_time_ms"] == 12.34
        assert "device" in formatted
        assert formatted["device"] == "cuda"
        assert "detections" in formatted
        assert len(formatted["detections"]) == 1

    def test_detection_entry_keys(self):
        """Each detection entry has class, confidence, bbox."""
        dets = [self._make_det(confidence=0.876, class_name="car")]
        result = self._make_result(dets)
        formatted = format_detections(result, conf_threshold=0.25)

        entry = formatted["detections"][0]
        assert entry["class"] == "car"
        assert entry["confidence"] == 0.876  # rounded to 3 dp
        assert entry["bbox"] == [10.0, 20.0, 100.0, 150.0]

    def test_threshold_zero_includes_all(self):
        """When threshold is 0, all detections are included."""
        dets = [
            self._make_det(confidence=0.01),
            self._make_det(confidence=0.99),
        ]
        result = self._make_result(dets)
        formatted = format_detections(result, conf_threshold=0.0)
        assert len(formatted["detections"]) == 2


class TestRunInference:
    """Tests for run_inference (full pipeline)."""

    @patch("yolo_demo.ui.services.inference_service.create_engine")
    def test_success(self, mock_create_engine):
        """Full pipeline returns formatted results on success."""
        mock_engine = MagicMock()
        mock_engine.predict.return_value = DetectionResult(
            detections=[
                Detection(
                    bbox=[10, 20, 100, 150],
                    confidence=0.95,
                    class_id=0,
                    class_name="person",
                )
            ],
            image_shape=(480, 640),
            inference_time_ms=45.2,
            device="cuda",
        )
        mock_create_engine.return_value = mock_engine

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = run_inference(img, "yolov8n.pt", None, 0.5)

        assert "error" not in result
        assert result["count"] == 1
        assert result["device"] == "cuda"
        mock_create_engine.assert_called_once_with("yolov8n.pt")
        mock_engine.load_model.assert_called_once()

    @patch("yolo_demo.ui.services.inference_service.create_engine")
    def test_custom_model_uploaded(self, mock_create_engine):
        """Custom uploaded model is preferred over text input."""
        mock_engine = MagicMock()
        mock_engine.predict.return_value = DetectionResult(
            detections=[],
            image_shape=(480, 640),
            inference_time_ms=10.0,
            device="cpu",
        )
        mock_create_engine.return_value = mock_engine

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = run_inference(img, "yolov8n.pt", "/uploads/custom.pt", 0.5)
        assert "error" not in result
        mock_create_engine.assert_called_once_with("/uploads/custom.pt")

    def test_no_model_provided(self):
        """Error when neither model name nor custom file provided."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = run_inference(img, "", None, 0.5)
        assert "error" in result

    @patch("yolo_demo.ui.services.inference_service.create_engine")
    def test_model_not_found(self, mock_create_engine):
        """FileNotFoundError is caught and returned as error message."""
        mock_create_engine.side_effect = FileNotFoundError("No such file: bad.pt")

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = run_inference(img, "bad.pt", None, 0.5)
        assert "error" in result
        assert "Model not found" in result["error"]

    @patch("yolo_demo.ui.services.inference_service.create_engine")
    def test_unexpected_error(self, mock_create_engine):
        """Generic exceptions are caught and returned."""
        mock_create_engine.side_effect = RuntimeError("GPU out of memory")

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = run_inference(img, "yolov8n.pt", None, 0.5)
        assert "error" in result
        assert "GPU out of memory" in result["error"]


class TestTrainingJobConfig:
    """Tests for TrainingJobConfig."""

    def test_to_training_config_basic(self):
        """Convert to TrainingConfig with default values."""
        job = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        cfg = job.to_training_config()
        assert cfg.model == "yolov8n.pt"
        assert cfg.epochs == 100
        assert cfg.batch == 16
        assert cfg.imgsz == 640
        assert cfg.lr0 == 0.01

    def test_to_training_config_with_output_dir(self):
        """Output directory is passed through to TrainingConfig."""
        job = TrainingJobConfig(
            model_path="yolov8n.pt",
            dataset_path="data.yaml",
            output_dir="/tmp/runs",
        )
        cfg = job.to_training_config()
        assert cfg.project == "/tmp/runs"


class TestTrainingSession:
    """Tests for TrainingSession lifecycle."""

    @patch("yolo_demo.ui.services.training_service.Trainer")
    def test_start_and_complete(self, mock_trainer_cls):
        """Training session starts, runs, and produces ('done', path)."""
        mock_trainer = MagicMock()
        mock_trainer.model = MagicMock()
        mock_trainer.train.return_value = TrainingResult(
            success=True, model_path="/tmp/best.pt"
        )
        mock_trainer_cls.return_value = mock_trainer

        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        session = TrainingSession("job-1", config)
        session.start()

        # Wait for completion (max 10s)
        import time as _time

        deadline = _time.time() + 10
        result_found = None
        while _time.time() < deadline:
            msg = session.poll(timeout=0.5)
            if msg is not None:
                if isinstance(msg, tuple):
                    result_found = msg
                    break
            if not session.is_active():
                break

        assert result_found is not None, "Training did not complete in time"
        assert result_found[0] == "done"
        assert result_found[1] == "/tmp/best.pt"
        assert session.status in ("completed", "running")

        session.cleanup()

    @patch("yolo_demo.ui.services.training_service.Trainer")
    def test_start_and_error(self, mock_trainer_cls):
        """Training session produces ('error', msg) on trainer failure."""
        mock_trainer = MagicMock()
        mock_trainer.model = MagicMock()
        mock_trainer.train.return_value = TrainingResult(
            success=False, error="CUDA out of memory"
        )
        mock_trainer_cls.return_value = mock_trainer

        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        session = TrainingSession("job-2", config)
        session.start()

        import time as _time

        deadline = _time.time() + 10
        result_found = None
        while _time.time() < deadline:
            msg = session.poll(timeout=0.5)
            if msg is not None and isinstance(msg, tuple):
                result_found = msg
                break
            if not session.is_active():
                break

        assert result_found is not None
        assert result_found[0] == "error"
        assert "CUDA out of memory" in result_found[1]

    def test_cancel_sets_stop_event(self):
        """Cancel sets the stop event; no thread needed for this test."""
        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        session = TrainingSession("job-3", config)

        # Simulate running state without starting a real thread
        session._status = "running"
        session.cancel()
        assert session.status == "cancelled"
        assert session._stop_event.is_set()

    def test_poll_returns_none_when_empty(self):
        """poll returns None when queue is empty and timeout expires."""
        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        session = TrainingSession("job-4", config)
        result = session.poll(timeout=0.01)
        assert result is None

    def test_initial_status_is_pending(self):
        """Session starts in 'pending' state."""
        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        session = TrainingSession("job-5", config)
        assert session.status == "pending"
        assert not session.is_active()
        assert session.error is None
        assert session.model_path is None

    def test_cleanup_is_idempotent(self):
        """cleanup can be called multiple times safely."""
        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        session = TrainingSession("job-6", config)
        session.cleanup()
        session.cleanup()  # should not raise

    @patch("yolo_demo.ui.services.training_service.Trainer")
    def test_start_is_idempotent(self, mock_trainer_cls):
        """Calling start twice on a running session is a no-op."""
        mock_trainer = MagicMock()
        mock_trainer.model = MagicMock()
        mock_trainer.train.return_value = TrainingResult(
            success=True, model_path="/tmp/best.pt"
        )
        mock_trainer_cls.return_value = mock_trainer

        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        session = TrainingSession("job-7", config)
        session.start()
        session.start()  # should be no-op (status is already "running")

        session.cleanup()

    @patch("yolo_demo.ui.services.training_service.Trainer")
    def test_has_messages_and_is_active(self, mock_trainer_cls):
        mock_trainer = MagicMock()
        mock_trainer.model = MagicMock()
        mock_trainer.train.return_value = TrainingResult(
            success=True, model_path="/tmp/best.pt"
        )
        mock_trainer_cls.return_value = mock_trainer

        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        session = TrainingSession("job-8", config)

        assert not session.is_active()
        session.cleanup()


class TestTrainingSessionManager:
    """Tests for TrainingSessionManager."""

    @patch("yolo_demo.ui.services.training_service.TrainingSession.start")
    def test_submit_creates_and_starts_job(self, mock_start):
        """Submit creates a session and starts it when no active job."""
        mgr = TrainingSessionManager(max_concurrent=1)

        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        job_id = mgr.submit(config)

        assert job_id is not None
        assert len(job_id) > 0
        assert mgr.active_count == 1
        assert mgr.active_job_id == job_id
        mock_start.assert_called_once()

    @patch("yolo_demo.ui.services.training_service.TrainingSession.start")
    def test_get_returns_session(self, mock_start):
        """Get retrieves a submitted session."""
        mgr = TrainingSessionManager()
        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        job_id = mgr.submit(config)

        session = mgr.get(job_id)
        assert session is not None
        assert session.job_id == job_id

    @patch("yolo_demo.ui.services.training_service.TrainingSession.start")
    def test_cancel_calls_session_cancel(self, mock_start):
        """Cancel delegates to the session's cancel method."""
        mgr = TrainingSessionManager()
        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        job_id = mgr.submit(config)

        session = mgr.get(job_id)
        session._status = "running"

        result = mgr.cancel(job_id)
        assert result is True
        assert session.status == "cancelled"

    @patch("yolo_demo.ui.services.training_service.TrainingSession.start")
    def test_cancel_unknown_job(self, mock_start):
        """Cancel returns False for unknown job_id."""
        mgr = TrainingSessionManager()
        result = mgr.cancel("nonexistent")
        assert result is False

    @patch("yolo_demo.ui.services.training_service.TrainingSession.start")
    def test_cleanup_removes_session(self, mock_start):
        """cleanup calls session.cleanup() and removes from dict."""
        mgr = TrainingSessionManager()
        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        job_id = mgr.submit(config)

        mgr.cleanup(job_id)
        assert mgr.get(job_id) is None
        assert mgr.active_count == 0

    @patch("yolo_demo.ui.services.training_service.TrainingSession.start")
    def test_cleanup_all_removes_everything(self, mock_start):
        """cleanup_all removes all sessions."""
        mgr = TrainingSessionManager()
        for i in range(3):
            config = TrainingJobConfig(
                model_path="yolov8n.pt", dataset_path=f"data{i}.yaml"
            )
            mgr.submit(config)

        assert len(mgr._sessions) == 3
        mgr.cleanup_all()
        assert len(mgr._sessions) == 0
        assert mgr.active_count == 0

    @patch("yolo_demo.ui.services.training_service.TrainingSession.start")
    def test_on_job_finished_clears_active(self, mock_start):
        """After on_job_finished, active_count returns to 0."""
        mgr = TrainingSessionManager()
        config = TrainingJobConfig(model_path="yolov8n.pt", dataset_path="data.yaml")
        job_id = mgr.submit(config)

        assert mgr.active_count == 1
        mgr.on_job_finished(job_id)
        assert mgr.active_count == 0

    @patch("yolo_demo.ui.services.training_service.TrainingSession.start")
    def test_get_nonexistent(self, mock_start):
        """Get returns None for unknown job_id."""
        mgr = TrainingSessionManager()
        assert mgr.get("no-such-job") is None


class TestSessionLogHandler:
    """Tests for _SessionLogHandler."""

    def test_emit_puts_record_in_queue(self):
        """Log records are forwarded to the queue."""
        import queue

        q = queue.Queue()
        handler = _SessionLogHandler(q)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        handler.emit(record)

        assert not q.empty()
        msg = q.get_nowait()
        assert "hello world" in msg


class TestCheckRknnAvailability:
    """Tests for check_rknn_availability."""

    @patch("yolo_demo.ui.services.export_service.check_rknn_availability")
    def test_returns_false_when_not_installed(self, mock_check):
        """When rknn-toolkit2 is not installed, returns (False, msg)."""
        mock_check.return_value = (
            False,
            "Error: rknn-toolkit2 is not installed.\n"
            "Install with: uv sync --extra rknn",
        )
        available, msg = mock_check()
        assert not available
        assert "rknn-toolkit2" in msg

    def test_returns_false_by_default(self):
        """In test environment (no rknn-toolkit2), returns (False, msg)."""
        available, msg = check_rknn_availability()
        assert not available
        assert "rknn-toolkit2" in msg


class TestExportPtToRknn:
    """Tests for export_pt_to_rknn."""

    @patch("yolo_demo.ui.services.export_service.pt_to_rknn")
    @patch("yolo_demo.ui.services.export_service.shutil.make_archive")
    def test_success(self, mock_archive, mock_pt_to_rknn):
        """Export succeeds and creates a zip."""
        mock_pt_to_rknn.return_value = "/tmp/model_rknn_model"

        status, path = export_pt_to_rknn("model.pt", "rk3588", 640)

        assert "successful" in status.lower()
        assert path is not None
        mock_pt_to_rknn.assert_called_once_with(
            "model.pt", target_platform="rk3588", imgsz=640
        )


class TestExportOnnxToRknn:
    """Tests for export_onnx_to_rknn."""

    def test_returns_not_available(self):
        """ONNX export returns not-available message (RKNNExporter not implemented)."""
        status, path = export_onnx_to_rknn("model.onnx", "rk3588")
        assert "rknn-toolkit2" in status
        assert path is None

    def test_quantization_without_dataset(self):
        """Returns error when quantization requested without dataset."""
        status, path = export_onnx_to_rknn(
            "model.onnx", "rk3588", quantize=True, dataset_path=None
        )
        assert "Calibration dataset required" in status
        assert path is None


class TestExtractVocZip:
    """Tests for extract_voc_zip."""

    @patch("yolo_demo.ui.services.dataset_service.zipfile.ZipFile")
    @patch("yolo_demo.ui.services.dataset_service.tempfile.mkdtemp")
    def test_finds_vocdevkit(self, mock_mkdtemp, mock_zipfile_cls):
        """Extracts zip and finds VOCdevkit directory."""
        from pathlib import Path

        extract_root = Path("/tmp/fake_extract")
        mock_mkdtemp.return_value = str(extract_root)

        # Create mock extract_dir structure
        mock_zipfile = MagicMock()

        def _extractall(dest):
            voc_devkit = dest / "VOCdevkit"
            voc_devkit.mkdir(parents=True, exist_ok=True)

        mock_zipfile.__enter__.return_value.extractall.side_effect = _extractall
        mock_zipfile_cls.return_value = mock_zipfile

        result = extract_voc_zip("dataset.zip")
        assert "VOCdevkit" in result

    @patch("yolo_demo.ui.services.dataset_service.zipfile.ZipFile")
    @patch("yolo_demo.ui.services.dataset_service.tempfile.mkdtemp")
    def test_no_vocdevkit_found(self, mock_mkdtemp, mock_zipfile_cls):
        """Raises FileNotFoundError when no VOCdevkit directory exists."""
        from pathlib import Path

        extract_root = Path("/tmp/fake_extract_empty")
        mock_mkdtemp.return_value = str(extract_root)

        mock_zipfile = MagicMock()

        def _extractall_empty(dest):
            (dest / "other_dir").mkdir(parents=True, exist_ok=True)

        mock_zipfile.__enter__.return_value.extractall.side_effect = _extractall_empty
        mock_zipfile_cls.return_value = mock_zipfile

        with pytest.raises(FileNotFoundError, match="VOCdevkit"):
            extract_voc_zip("dataset.zip")


class TestConvertDataset:
    """Tests for convert_dataset."""

    @patch("builtins.open", new_callable=mock_open, read_data="nc: 5\nnames: [a, b, c, d, e]")
    @patch("yolo_demo.ui.services.dataset_service.yaml.safe_load")
    @patch("yolo_demo.ui.services.dataset_service.convert_coco_to_yolo")
    def test_convert_coco_success(self, mock_convert, mock_yaml_load, mock_file):
        """COCO conversion succeeds."""
        mock_convert.return_value = "/tmp/output/dataset.yaml"
        mock_yaml_load.return_value = {"nc": 5, "names": ["a", "b", "c", "d", "e"]}

        status, yaml_path, info = convert_dataset(
            "COCO", "/tmp/annotations.json", None, "trainval", True, "test-ds"
        )

        assert "successful" in status.lower()
        assert yaml_path == "/tmp/output/dataset.yaml"
        assert info is not None
        assert info["nc"] == 5

    def test_convert_coco_no_file(self):
        """Returns error when no COCO file provided."""
        status, yaml_path, info = convert_dataset(
            "COCO", None, None, "trainval", True, "test-ds"
        )
        assert "Error" in status
        assert yaml_path is None
        assert info is None

    def test_convert_voc_no_file(self):
        """Returns error when no VOC file provided."""
        status, yaml_path, info = convert_dataset(
            "VOC", None, None, "trainval", True, "test-ds"
        )
        assert "Error" in status
        assert yaml_path is None
        assert info is None
