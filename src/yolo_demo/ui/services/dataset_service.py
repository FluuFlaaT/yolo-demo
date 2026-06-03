"""Dataset service — decouples Gradio dataset converter callback from conversion logic."""

import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from ..coco2yolo import convert_coco_to_yolo, convert_voc_to_yolo

logger = logging.getLogger(__name__)


def extract_voc_zip(zip_path: str) -> str:
    """Extract a VOC dataset zip and find the VOCdevkit directory.

    Args:
        zip_path: Path to the .zip file.

    Returns:
        Path to the VOCdevkit directory as string.

    Raises:
        FileNotFoundError: If no VOCdevkit directory is found after extraction.
    """
    extract_dir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    # Look for VOCdevkit directory
    voc_dir = extract_dir / "VOCdevkit"
    if voc_dir.exists():
        return str(voc_dir)

    # Fallback: find any directory containing "VOC" in its name
    for d in extract_dir.iterdir():
        if d.is_dir() and "VOC" in d.name:
            return str(d)

    raise FileNotFoundError(
        f"No VOCdevkit directory found in {zip_path}"
    )


def convert_dataset(
    format_type: str,
    coco_file_path: Optional[str],
    voc_file_path: Optional[str],
    voc_split: str,
    copy_images: bool,
    output_name: str,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Convert COCO or VOC format dataset to YOLO format.

    Args:
        format_type: "COCO" or "VOC".
        coco_file_path: Path to COCO annotations JSON file (or None).
        voc_file_path: Path to VOC zip file or directory (or None).
        voc_split: Dataset split for VOC ("trainval", "train", "val", "test").
        copy_images: Whether to copy images to the output directory.
        output_name: Name for the output dataset directory.

    Returns:
        (status_message, yaml_path, dataset_info): Status text, path to the
        generated dataset.yaml, and a dict with dataset metadata (classes,
        image count, label count).
    """
    output_dir = Path(tempfile.mkdtemp()) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if format_type == "COCO":
        if not coco_file_path:
            return "Error: Please upload COCO annotations JSON", None, None

        yaml_path = convert_coco_to_yolo(
            coco_json_path=coco_file_path,
            output_dir=str(output_dir),
            copy_images=copy_images,
        )
    else:  # VOC
        if not voc_file_path:
            return "Error: Please upload VOCdevkit directory", None, None

        # Handle zip extraction
        if voc_file_path.endswith(".zip"):
            voc_dir = extract_voc_zip(voc_file_path)
        else:
            voc_dir = voc_file_path

        yaml_path = convert_voc_to_yolo(
            voc_devkit_dir=voc_dir,
            output_dir=str(output_dir),
            copy_images=copy_images,
            split=voc_split,
        )

    # Load and preview dataset YAML
    with open(yaml_path) as f:
        dataset_info: Dict[str, Any] = yaml.safe_load(f)

    # Count images and labels
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    num_images = (
        len(list(images_dir.glob("**/*.*"))) if images_dir.exists() else 0
    )
    num_labels = (
        len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
    )

    dataset_info["_stats"] = {
        "num_images": num_images,
        "num_labels": num_labels,
        "num_classes": dataset_info.get("nc", 0),
    }

    status = "Conversion successful!\n\n"
    status += f"Classes: {dataset_info.get('nc', 0)}\n"
    status += f"Images: {num_images}\n"
    status += f"Labels: {num_labels}\n"
    status += f"\nDataset YAML:\n{yaml_path}"

    logger.info("Dataset converted: %s", yaml_path)
    return status, yaml_path, dataset_info
