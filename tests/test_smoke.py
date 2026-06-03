"""Smoke tests for CLI entry point and UI modules."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestMainModule:
    """Test CLI entry point module-level coverage."""

    @patch("sys.exit")
    def test_main_version(self, mock_exit):
        """Test --version flag works."""
        with patch.object(sys, "argv", ["yolo-demo", "--version"]):
            from yolo_demo.main import main

            main()
            assert mock_exit.called

    @patch("sys.exit")
    def test_main_no_command(self, mock_exit):
        """Test no command prints help and exits."""
        with patch.object(sys, "argv", ["yolo-demo"]):
            from yolo_demo.main import main

            main()
            mock_exit.assert_called_once_with(0)

    @patch("yolo_demo.main.run_training")
    def test_main_train_command(self, mock_train):
        """Test train subcommand parsing."""
        with patch.object(sys, "argv", ["yolo-demo", "train", "data.yaml"]):
            from yolo_demo.main import main

            main()
            mock_train.assert_called_once()

    @patch("yolo_demo.main.run_export")
    def test_main_export_command(self, mock_export):
        """Test export subcommand parsing."""
        with patch.object(sys, "argv", ["yolo-demo", "export", "model.pt"]):
            from yolo_demo.main import main

            main()
            mock_export.assert_called_once()

    @patch("yolo_demo.main.run_webui")
    def test_main_webui_command(self, mock_webui):
        """Test webui subcommand parsing."""
        with patch.object(sys, "argv", ["yolo-demo", "webui"]):
            from yolo_demo.main import main

            main()
            mock_webui.assert_called_once()

    @patch("yolo_demo.main.run_api")
    def test_main_api_command(self, mock_api):
        """Test api subcommand parsing."""
        with patch.object(sys, "argv", ["yolo-demo", "api"]):
            from yolo_demo.main import main

            main()
            mock_api.assert_called_once()

    def test_cli_function(self):
        """Test cli() entry point exists."""
        from yolo_demo.main import cli

        assert callable(cli)


class TestWebUIModule:
    """Smoke tests for WebUI module imports."""

    @patch("gradio.Blocks")
    @patch("gradio.Markdown")
    @patch("gradio.Tabs")
    @patch("yolo_demo.ui.webui.create_inference_tab")
    @patch("yolo_demo.ui.webui.create_training_tab")
    @patch("yolo_demo.ui.webui.create_export_tab")
    @patch("yolo_demo.ui.webui.create_dataset_converter_tab")
    def test_create_webui(self, *mocks):
        """Test create_webui assembles tabs."""
        from yolo_demo.ui.webui import create_webui

        app = create_webui()
        assert app is not None

    def test_draw_detections(self):
        """Test draw_detections overlays boxes on image."""
        from yolo_demo.ui.webui import draw_detections
        from yolo_demo.inference import Detection

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [
            Detection(bbox=[10, 20, 100, 150], confidence=0.95, class_id=0, class_name="person")
        ]
        result = draw_detections(img, dets)
        assert result is not None


class TestDatasetConverterModule:
    """Smoke tests for dataset converter module."""

    def test_import_dataset_converter(self):
        """Test dataset converter module is importable."""
        from yolo_demo.ui import dataset_converter

        assert hasattr(dataset_converter, "create_dataset_converter_tab")
