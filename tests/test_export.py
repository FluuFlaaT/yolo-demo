"""Tests for export module."""

import pytest
from unittest.mock import MagicMock, patch

from yolo_demo.export.rknn_exporter import RKNN_SUPPORTED_PLATFORMS, pt_to_rknn


class TestPtToRKNN:
    """Test pt_to_rknn function."""

    def test_pt_to_rknn_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            pt_to_rknn("/nonexistent/model.pt")

    def test_pt_to_rknn_unsupported_platform(self):
        with pytest.raises(ValueError, match="Unsupported platform"):
            pt_to_rknn("/nonexistent/model.pt", target_platform="invalid")

    @patch("ultralytics.YOLO")
    @patch("pathlib.Path.exists", return_value=True)
    def test_pt_to_rknn_success(self, mock_exists, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        result = pt_to_rknn("/fake/model.pt", target_platform="rk3588")

        mock_model.export.assert_called_once()
        assert isinstance(result, str)
        assert "model_rknn_model" in result

    @patch("ultralytics.YOLO")
    @patch("pathlib.Path.exists", return_value=True)
    def test_pt_to_rknn_with_batch(self, mock_exists, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        pt_to_rknn("/fake/model.pt", target_platform="rk3568", batch=4)

        call_kwargs = mock_model.export.call_args.kwargs
        assert call_kwargs["batch"] == 4
        assert call_kwargs["name"] == "rk3568"

    def test_all_platforms_supported(self):
        for p in RKNN_SUPPORTED_PLATFORMS:
            assert isinstance(p, str)
            assert len(p) >= 5


class TestRKNNExporterImport:
    """Test that the export module is importable."""

    def test_export_init_exports(self):
        from yolo_demo.export import pt_to_rknn as exported_fn
        from yolo_demo.export.rknn_exporter import pt_to_rknn as source_fn

        assert exported_fn is source_fn
