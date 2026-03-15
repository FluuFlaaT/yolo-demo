"""Tests for export module."""

import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch


class TestONNXExporter:
    """Test ONNX exporter."""

    @patch("yolo_demo.export.onnx_exporter.YOLO")
    def test_onnx_exporter_init(self, mock_yolo):
        """Test ONNXExporter initialization."""
        from yolo_demo.export import ONNXExporter

        with patch("yolo_demo.export.onnx_exporter._ensure_onnx_metadata"):
            exporter = ONNXExporter("yolov8n.pt")
            assert exporter.model_path is not None

    @patch("yolo_demo.export.onnx_exporter.YOLO")
    def test_onnx_exporter_load_model(self, mock_yolo):
        """Test loading model."""
        from yolo_demo.export import ONNXExporter

        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        exporter = ONNXExporter("yolov8n.pt")
        exporter.load_model()

        assert exporter.model is not None

    @patch("yolo_demo.export.onnx_exporter.YOLO")
    @patch("yolo_demo.export.onnx_exporter._ensure_onnx_metadata")
    @patch("yolo_demo.export.onnx_exporter.onnx")
    def test_onnx_export_with_metadata(self, mock_onnx, mock_ensure_metadata, mock_yolo):
        """Test ONNX export includes metadata handling."""
        from yolo_demo.export import ONNXExporter

        mock_model = MagicMock()
        mock_model.export.return_value = "model.onnx"
        mock_yolo.return_value = mock_model

        exporter = ONNXExporter("yolov8n.pt")
        exporter.load_model()
        result = exporter.export()

        assert result == "model.onnx"
        mock_ensure_metadata.assert_called_once()

    @patch("yolo_demo.export.onnx_exporter.YOLO")
    def test_onnx_exporter_from_file(self, mock_yolo):
        """Test static method export_from_file."""
        from yolo_demo.export import ONNXExporter

        mock_model = MagicMock()
        mock_model.export.return_value = "model.onnx"
        mock_yolo.return_value = mock_model

        with patch("yolo_demo.export.onnx_exporter._ensure_onnx_metadata"):
            result = ONNXExporter.export_from_file("yolov8n.pt")

        assert result == "model.onnx"


class TestEnsureOnnxMetadata:
    """Test _ensure_onnx_metadata function."""

    @patch("yolo_demo.export.onnx_exporter.onnx")
    def test_ensure_metadata_adds_names(self, mock_onnx):
        """Test adding names to ONNX metadata when missing."""
        from yolo_demo.export.onnx_exporter import _ensure_onnx_metadata
        from pathlib import Path

        mock_model = MagicMock()
        mock_model.metadata_props = []
        mock_onnx.load.return_value = mock_model

        mock_yolo_model = MagicMock()
        mock_yolo_model.names = {0: "person", 1: "car"}

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = Path(f.name)

        try:
            _ensure_onnx_metadata(path, mock_yolo_model)
            mock_onnx.save.assert_called_once()
        finally:
            path.unlink(missing_ok=True)

    @patch("yolo_demo.export.onnx_exporter.onnx")
    def test_ensure_metadata_skips_when_names_exist(self, mock_onnx):
        """Test skipping when names already exist."""
        from yolo_demo.export.onnx_exporter import _ensure_onnx_metadata
        from pathlib import Path

        mock_prop = MagicMock()
        mock_prop.key = "names"
        mock_prop.value = "{0: 'person'}"

        mock_model = MagicMock()
        mock_model.metadata_props = [mock_prop]
        mock_onnx.load.return_value = mock_model

        mock_yolo_model = MagicMock()
        mock_yolo_model.names = {0: "person", 1: "car"}

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = Path(f.name)

        try:
            _ensure_onnx_metadata(path, mock_yolo_model)
            mock_onnx.save.assert_not_called()
        finally:
            path.unlink(missing_ok=True)


class TestPrepareForRK3588:
    """Test prepare_for_rk3588 function."""

    @patch("yolo_demo.export.onnx_exporter.ONNXExporter.export_from_file")
    def test_prepare_for_rk3588_default_params(self, mock_export):
        """Test prepare_for_rk3588 with default parameters."""
        from yolo_demo.export import prepare_for_rk3588

        mock_export.return_value = "model.onnx"

        result = prepare_for_rk3588("yolov8n.pt")

        assert result == "model.onnx"
        mock_export.assert_called_once()
        call_args = mock_export.call_args
        assert call_args[0][0] == "yolov8n.pt"
        assert call_args[1]["opset"] == 11
        assert call_args[1]["dynamic"] is True
        assert call_args[1]["simplify"] is True

    @patch("yolo_demo.export.onnx_exporter.ONNXExporter.export_from_file")
    def test_prepare_for_rk3588_custom_output(self, mock_export):
        """Test prepare_for_rk3588 with custom output path."""
        from yolo_demo.export import prepare_for_rk3588

        mock_export.return_value = "/path/to/model.onnx"

        result = prepare_for_rk3588("yolov8n.pt", output="/custom/path")

        assert result == "/path/to/model.onnx"
        mock_export.assert_called_once()
        call_args = mock_export.call_args
        assert call_args[0][0] == "yolov8n.pt"
        assert call_args[1]["output"] == "/custom/path"
