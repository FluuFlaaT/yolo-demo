"""ONNX export module for YOLO models."""

import logging
import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export YOLO models to ONNX format."""

    def __init__(self, model_path: str | Path | None = None):
        """
        Initialize the exporter.

        Args:
            model_path: Optional path to a YOLO model. Can also be provided in export().
        """
        self.model_path = Path(model_path) if model_path else None
        self.model: YOLO | None = None

    def load_model(self, model_path: str | Path | None = None) -> None:
        """
        Load a YOLO model for export.

        Args:
            model_path: Path to YOLO model weights (.pt file).
        """
        path = model_path or self.model_path
        if path is None:
            raise ValueError("model_path must be provided")
        self.model = YOLO(str(path))

    def export(
        self,
        output: str | Path | None = None,
        opset: int = 11,
        dynamic: bool = True,
        simplify: bool = True,
        include: list[str] | None = None,
        output_filename: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output: Output directory or file path. If None, exports to runs/export.
            opset: ONNX opset version (11 or 12 recommended for RK3588).
            dynamic: Enable dynamic axes for batch and image size.
            simplify: Simplify the exported ONNX model.
            include: Additional formats to include (e.g., ['onnx', 'openvino']).
            output_filename: Custom output filename (e.g., 'model-rk3588-export.onnx').
            **kwargs: Additional arguments passed to ultralytics export.

        Returns:
            Path to the exported ONNX file.

        Example:
            ```python
            exporter = ONNXExporter('yolov8n.pt')
            onnx_path = exporter.export(opset=11, dynamic=True)
            ```
        """
        if self.model is None:
            self.load_model()

        export_kwargs = {
            "format": "onnx",
            "opset": opset,
            "dynamic": dynamic,
            "simplify": simplify,
        }

        if output:
            output_path = Path(output)
            if output_path.suffix == ".onnx":
                export_kwargs["save_dir"] = str(output_path.parent)
            else:
                export_kwargs["save_dir"] = str(output)

        if include:
            export_kwargs["include"] = include

        export_kwargs.update(kwargs)

        onnx_path = self.model.export(**export_kwargs)
        onnx_path = Path(onnx_path)

        # Rename if custom filename is provided
        if output_filename:
            new_path = onnx_path.parent / output_filename
            if new_path != onnx_path:
                shutil.move(str(onnx_path), str(new_path))
                logger.info(f"Renamed exported model to: {new_path}")
                return str(new_path)

        return str(onnx_path)

    @staticmethod
    def export_from_file(
        model_path: str | Path,
        output: str | Path | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Static method to export a model in one call.

        Args:
            model_path: Path to YOLO model weights.
            output: Output path.
            **kwargs: Additional export arguments.

        Returns:
            Path to the exported ONNX file.
        """
        exporter = ONNXExporter(model_path)
        return exporter.export(output=output, **kwargs)


def prepare_for_rk3588(
    model_path: str | Path,
    output: str | Path | None = None,
) -> str:
    """
    Export a YOLO model optimized for RK3588 (RKNN) deployment.

    Uses ONNX opset 11 which has good RKNN compatibility.

    Args:
        model_path: Path to YOLO model weights.
        output: Output path for ONNX file.

    Returns:
        Path to the exported ONNX file ready for RKNN conversion.
    """
    return ONNXExporter.export_from_file(
        model_path,
        output=output,
        opset=11,
        dynamic=True,
        simplify=True,
        half=False,  # RKNN typically uses FP16 but keep FP32 for compatibility
    )
