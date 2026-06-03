"""Tests for training module."""

from unittest.mock import MagicMock, patch

import pytest

from yolo_demo.training.trainer import Trainer, TrainingConfig, TrainingResult


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.batch == 16
        assert config.imgsz == 640
        assert config.lr0 == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            model="yolov8s.pt",
            epochs=50,
            batch=32,
            imgsz=320,
        )
        assert config.model == "yolov8s.pt"
        assert config.epochs == 50
        assert config.batch == 32
        assert config.imgsz == 320

    def test_from_yaml(self, tmp_path):
        """Test loading config from YAML."""
        yaml_content = """
model: yolov8s.pt
epochs: 50
batch: 32
imgsz: 320
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = TrainingConfig.from_yaml(yaml_file)
        assert config.model == "yolov8s.pt"
        assert config.epochs == 50
        assert config.batch == 32

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TrainingConfig(epochs=50, batch=32)
        d = config.to_dict()
        assert d["epochs"] == 50
        assert d["batch"] == 32
        assert d["imgsz"] == 640

    def test_to_dict_with_extra_kwargs(self):
        """Test to_dict includes extra_kwargs."""
        config = TrainingConfig(extra_kwargs={"custom_param": 42, "freeze": 10})
        d = config.to_dict()
        assert d["custom_param"] == 42
        assert d["freeze"] == 10

    def test_to_dict_excludes_none(self):
        """Test to_dict excludes optional fields when None."""
        config = TrainingConfig(project=None, name=None, device=None)
        d = config.to_dict()
        assert "project" not in d
        assert "name" not in d
        assert "device" not in d

    def test_to_dict_includes_optional_when_set(self):
        """Test to_dict includes optional fields when set."""
        config = TrainingConfig(project="/tmp/runs", name="exp1", device="cpu")
        d = config.to_dict()
        assert d["project"] == "/tmp/runs"
        assert d["name"] == "exp1"
        assert d["device"] == "cpu"


class TestTrainer:
    """Test Trainer class."""

    @patch("yolo_demo.training.trainer.YOLO")
    def test_trainer_load_pretrained(self, mock_yolo):
        """Test loading pretrained model."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        trainer = Trainer()
        trainer.load_pretrained("yolov8n.pt")

        assert mock_yolo.called
        assert trainer.model is not None

    @patch("yolo_demo.training.trainer.YOLO")
    def test_trainer_train(self, mock_yolo):
        """Test training."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.model_save_dir = "/tmp/runs/detect/train"

        # Setup mock train result
        mock_result = MagicMock()
        mock_result.result_dict = {"metrics/precision": 0.9, "metrics/recall": 0.85}
        mock_model.train.return_value = mock_result

        trainer = Trainer(TrainingConfig(epochs=10, batch=8))
        trainer.load_pretrained("yolov8n.pt")

        result = trainer.train(data_yaml="data.yaml")

        assert result.success is True
        assert trainer.model is not None

    @patch("yolo_demo.training.trainer.YOLO")
    def test_trainer_train_failure(self, mock_yolo):
        """Test training failure handling."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.train.side_effect = Exception("Training failed")

        trainer = Trainer()
        trainer.load_pretrained("yolov8n.pt")

        result = trainer.train(data_yaml="data.yaml")

        assert result.success is False
        assert result.error is not None

    @patch("yolo_demo.training.trainer.YOLO")
    def test_trainer_export_onnx(self, mock_yolo):
        """Test ONNX export."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.export.return_value = "/tmp/model.onnx"

        trainer = Trainer()
        trainer.load_pretrained("yolov8n.pt")
        onnx_path = trainer.export_onnx(output="/tmp/model.onnx")

        assert mock_model.export.called
        assert onnx_path == "/tmp/model.onnx"

    @patch("yolo_demo.training.trainer.YOLO")
    def test_trainer_export_without_model(self, mock_yolo):
        """Test export without loaded model."""
        trainer = Trainer()

        with pytest.raises(RuntimeError, match="No model loaded"):
            trainer.export_onnx()

    @patch("yolo_demo.training.trainer.YOLO")
    def test_trainer_train_auto_loads_model(self, mock_yolo):
        """Test that train() auto-loads model when none is preloaded."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.model_save_dir = "/tmp/runs/detect/train"
        mock_result = MagicMock()
        mock_result.result_dict = {}
        mock_model.train.return_value = mock_result

        trainer = Trainer()
        result = trainer.train(data_yaml="data.yaml")

        assert result.success is True
        assert mock_yolo.call_count == 1  # auto-load via train

    @patch("yolo_demo.training.trainer.YOLO")
    def test_trainer_export_on_none_path(self, mock_yolo):
        """Test export without explicit output path."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.export.return_value = "/tmp/model.onnx"

        trainer = Trainer()
        trainer.load_pretrained("yolov8n.pt")
        onnx_path = trainer.export_onnx()

        assert mock_model.export.called
        assert onnx_path == "/tmp/model.onnx"


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_success_result(self):
        """Test successful training result."""
        result = TrainingResult(
            success=True,
            model_path="/tmp/best.pt",
            metrics={"precision": 0.9},
        )
        assert result.success is True
        assert result.model_path == "/tmp/best.pt"
        assert result.metrics["precision"] == 0.9

    def test_failure_result(self):
        """Test failed training result."""
        result = TrainingResult(
            success=False,
            error="Out of memory",
        )
        assert result.success is False
        assert result.error == "Out of memory"
        assert result.model_path is None
