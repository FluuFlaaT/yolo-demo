"""Export service — decouples Gradio export callbacks from export logic."""

import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

from ...export import pt_to_rknn

logger = logging.getLogger(__name__)


def check_rknn_availability() -> Tuple[bool, str]:
    """Pre-flight check: is rknn-toolkit2 importable?

    Returns:
        (available, message): (True, "") if installed,
                              (False, install_instructions) if not.
    """
    try:
        from ...export.rknn_exporter import RKNNExporter  # noqa: F401

        return True, ""
    except ImportError:
        return (
            False,
            "Error: rknn-toolkit2 is not installed.\n"
            "Install with: uv sync --extra rknn\n"
            "Note: Only supports Linux x86_64 and Python 3.8-3.10",
        )


def export_pt_to_rknn(
    model_path: str,
    platform: str,
    imgsz: int,
    output_dir: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Export a .pt model to RKNN format.

    Args:
        model_path: Path to the .pt model file.
        platform: Target Rockchip platform (e.g., "rk3588").
        imgsz: Input image size.
        output_dir: Optional output directory (uses default if None).

    Returns:
        (status_message, file_path): Status text and path to the exported
        file (or zip if multiple files), or None on failure.
    """
    rknn_path = pt_to_rknn(
        model_path,
        target_platform=platform,
        imgsz=imgsz,
    )

    rknn_dir = Path(rknn_path)
    zip_path = str(rknn_dir.with_suffix(".zip"))
    shutil.make_archive(
        str(rknn_dir.with_suffix("")),
        "zip",
        rknn_dir,
    )

    logger.info("RKNN model exported to: %s", rknn_path)
    status = (
        f"RKNN export successful!\n\n"
        f"Platform: {platform}\n"
        f"Output: {zip_path}"
    )
    return status, zip_path


def _resolve_output_filename(
    onnx_path: str, output_filename: Optional[str]
) -> str:
    """Resolve output .rknn filename from input or provided name."""
    if output_filename:
        if not output_filename.endswith(".rknn"):
            output_filename += ".rknn"
        return output_filename
    return Path(onnx_path).stem + ".rknn"


def export_onnx_to_rknn(
    onnx_path: str,
    platform: str,
    quantize: bool = False,
    dataset_path: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Export an .onnx model to RKNN format.

    NOTE: The RKNNExporter class is not yet implemented. This function
    always returns an availability error until the class is created.

    Args:
        onnx_path: Path to the .onnx file.
        platform: Target Rockchip platform.
        quantize: Enable INT8 quantization (requires calibration dataset).
        dataset_path: Path to calibration dataset .txt file (for quantization).
        output_filename: Optional output .rknn filename.

    Returns:
        (status_message, rknn_path): Status text and path to .rknn file,
        or (error_message, None) on failure.
    """
    if quantize and not dataset_path:
        return "Error: Calibration dataset required for INT8 quantization", None

    _, msg = check_rknn_availability()
    return msg, None
