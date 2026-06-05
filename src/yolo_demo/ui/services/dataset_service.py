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
    """Extract a VOC dataset zip and return the path containing Annotations + JPEGImages.

    Handles three common layouts:
      - zip/VOCdevkit/VOC2007/Annotations/
      - zip/VOC2007/Annotations/
      - zip/Annotations/ + JPEGImages/ at root

    Returns:
        Path to the directory that should be passed to convert_voc_to_yolo
        (i.e., the parent of Annotations/ if flat, or VOCdevkit/ if present).

    Raises:
        FileNotFoundError: If neither Annotations/ nor VOCdevkit is found.
    """
    extract_dir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    known = {p.name for p in extract_dir.iterdir()}

    # Case A: flat layout — Annotations/ + JPEGImages/ at root
    if "Annotations" in known:
        return str(extract_dir)

    # Case B: VOCdevkit/ wrapper present
    vocdevkit = extract_dir / "VOCdevkit"
    if vocdevkit.is_dir():
        return str(vocdevkit)

    # Case C: year subdirectory (e.g. VOC2007/ directly)
    for year in ("VOC2007", "VOC2012", "VOC2010"):
        year_dir = extract_dir / year
        if year_dir.is_dir():
            return str(extract_dir)

    raise FileNotFoundError(
        "No VOC dataset found in zip. Expected Annotations/ and JPEGImages/ "
        "at the root, or inside VOCdevkit/, VOC2007/, VOC2012/, or VOC2010/."
    )


def _extract_coco_zip(zip_path: str) -> str:
    """Extract a COCO zip and return the path to the annotations JSON file."""
    extract_dir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    for f in extract_dir.rglob("*.json"):
        return str(f)

    top_level = sorted(p.name for p in extract_dir.iterdir())
    raise FileNotFoundError(
        "No .json file found in COCO zip archive.\n"
        f"Top-level contents: {top_level}"
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
        coco_file_path: Path to COCO zip or JSON file (or None).
        voc_file_path: Path to VOC zip file or directory (or None).
        voc_split: Dataset split for VOC ("trainval", "train", "val", "test").
        copy_images: Whether to copy images to the output directory.
        output_name: Name for the output dataset directory.

    Returns:
        (status_message, yaml_path, dataset_info): Status text, path to the
        generated dataset.yaml, and a dict with dataset metadata.
    """
    output_dir = Path(tempfile.mkdtemp()) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if format_type == "COCO":
        if not coco_file_path:
            return "Error: Please upload a COCO zip or JSON file", None, None

        if coco_file_path.lower().endswith(".zip"):
            json_path = _extract_coco_zip(coco_file_path)
        else:
            json_path = coco_file_path

        yaml_path = convert_coco_to_yolo(
            coco_json_path=json_path,
            output_dir=str(output_dir),
            copy_images=copy_images,
        )
    else:
        if not voc_file_path:
            return "Error: Please upload a VOC zip or directory", None, None

        src = Path(voc_file_path)
        if src.suffix.lower() == ".zip":
            voc_root = extract_voc_zip(voc_file_path)
        elif src.is_dir():
            voc_root = voc_file_path
        else:
            return f"Error: Path not found: {voc_file_path}", None, None

        yaml_path = convert_voc_to_yolo(
            voc_devkit_dir=voc_root,
            output_dir=str(output_dir),
            copy_images=copy_images,
            split=voc_split,
        )

    with open(yaml_path) as f:
        dataset_info: Dict[str, Any] = yaml.safe_load(f)

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
