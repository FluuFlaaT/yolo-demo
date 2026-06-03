"""RKNN export module using Ultralytics native export."""

import logging
import time
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

RKNN_SUPPORTED_PLATFORMS = [
    "rk3588",
    "rk3576",
    "rk3566",
    "rk3568",
    "rk3562",
    "rv1103",
    "rv1106",
    "rv1103b",
    "rv1106b",
    "rk2118",
    "rv1126b",
]


def pt_to_rknn(
    pt_path: Union[str, Path],
    target_platform: str = "rk3588",
    imgsz: int = 640,
    batch: int = 1,
) -> str:
    """
    Export YOLO model (.pt) to RKNN format using Ultralytics native export.

    Args:
        pt_path: Path to YOLO model file (.pt)
        target_platform: Target Rockchip platform
        imgsz: Input image size
        batch: Batch size

    Returns:
        Path to exported RKNN model directory

    Example:
        rknn_dir = pt_to_rknn("yolov8n.pt", target_platform="rk3588")
    """
    from ultralytics import YOLO

    if target_platform not in RKNN_SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Unsupported platform: {target_platform}. "
            f"Supported: {', '.join(RKNN_SUPPORTED_PLATFORMS)}"
        )

    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Model not found: {pt_path}")

    logger.info(f"Exporting {pt_path} to RKNN for {target_platform}...")

    model = YOLO(str(pt_path))
    export_dir = pt_path.parent / f"{pt_path.stem}_rknn_model"

    start = time.time()
    model.export(format="rknn", name=target_platform, imgsz=imgsz, batch=batch, exist_ok=True)
    elapsed = time.time() - start

    logger.info(f"Export completed in {elapsed:.1f}s -> {export_dir}")
    return str(export_dir)
