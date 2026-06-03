"""Tests for export module."""

import pytest


class TestPtToRKNN:
    """Test pt_to_rknn function."""

    def test_pt_to_rknn_file_not_found(self):
        """Test pt_to_rknn raises error when file not found."""
        from yolo_demo.export.rknn_exporter import pt_to_rknn

        with pytest.raises(FileNotFoundError):
            pt_to_rknn("/nonexistent/model.pt")

    def test_pt_to_rknn_unsupported_platform(self):
        """Test pt_to_rknn raises error for unsupported platform."""
        from yolo_demo.export.rknn_exporter import pt_to_rknn

        with pytest.raises(ValueError, match="Unsupported platform"):
            pt_to_rknn("/nonexistent/model.pt", target_platform="invalid")
