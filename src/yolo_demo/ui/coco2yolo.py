#!/usr/bin/env python3
"""Convert COCO or VOC format datasets to YOLO format."""

import argparse
import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_coco_annotations(coco_json_path: str) -> Tuple[List[dict], List[str]]:
    """Parse COCO JSON annotations.

    Args:
        coco_json_path: Path to COCO format JSON file.

    Returns:
        Tuple of (annotations list, class names list).
    """
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    categories = sorted(coco_data["categories"], key=lambda x: x["id"])
    class_names = [cat["name"] for cat in categories]

    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}

    images = {img["id"]: img for img in coco_data["images"]}
    annotations = []

    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in images:
            continue

        img_info = images[img_id]
        class_idx = cat_id_to_idx[ann["category_id"]]

        bbox = ann["bbox"]
        img_width = img_info["width"]
        img_height = img_info["height"]

        x_center = (bbox[0] + bbox[2] / 2) / img_width
        y_center = (bbox[1] + bbox[3] / 2) / img_height
        width = bbox[2] / img_width
        height = bbox[3] / img_height

        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        annotations.append(
            {
                "image_id": img_id,
                "filename": img_info["file_name"],
                "width": img_width,
                "height": img_height,
                "class_idx": class_idx,
                "bbox": [x_center, y_center, width, height],
            }
        )

    return annotations, class_names


def parse_voc_annotations(voc_xml_path: str) -> Tuple[List[dict], Set[str]]:
    """Parse VOC XML annotations.

    Args:
        voc_xml_path: Path to VOC format XML file.

    Returns:
        Tuple of (annotations list, set of class names).
    """
    tree = ET.parse(voc_xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)
    filename = root.find("filename").text

    annotations = []
    class_names: Set[str] = set()

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_names.add(class_name)

        bbox_elem = obj.find("bndbox")
        xmin = float(bbox_elem.find("xmin").text)
        ymin = float(bbox_elem.find("ymin").text)
        xmax = float(bbox_elem.find("xmax").text)
        ymax = float(bbox_elem.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        annotations.append(
            {
                "filename": filename,
                "width": img_width,
                "height": img_height,
                "class_name": class_name,
                "bbox": [x_center, y_center, width, height],
            }
        )

    return annotations, class_names


def convert_coco_to_yolo(
    coco_json_path: str,
    output_dir: str,
    copy_images: bool = False,
    image_dir: Optional[str] = None,
) -> str:
    """Convert COCO format dataset to YOLO format.

    Args:
        coco_json_path: Path to COCO annotations JSON file.
        output_dir: Output directory for YOLO format dataset.
        copy_images: Whether to copy images to output directory.
        image_dir: Source image directory (if different from COCO paths).

    Returns:
        Path to created dataset YAML file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Parsing COCO annotations from {coco_json_path}")
    annotations, class_names = parse_coco_annotations(coco_json_path)

    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    annotations_by_img: Dict[str, List[dict]] = {}
    for ann in annotations:
        filename = ann["filename"]
        if filename not in annotations_by_img:
            annotations_by_img[filename] = []
        annotations_by_img[filename].append(ann)

    logger.info(f"Converting {len(annotations_by_img)} images")
    for filename, img_annotations in annotations_by_img.items():
        label_filename = Path(filename).stem + ".txt"
        label_path = labels_dir / label_filename

        with open(label_path, "w") as f:
            for ann in img_annotations:
                bbox = ann["bbox"]
                class_idx = ann["class_idx"]
                f.write(f"{class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        if copy_images:
            src_image = Path(image_dir) / filename if image_dir else Path(filename)
            if not src_image.exists():
                src_image = Path(coco_json_path).parent / "images" / filename

            if src_image.exists():
                dst_image = images_dir / filename
                dst_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_image, dst_image)

    yaml_path = output_path / "dataset.yaml"
    dataset_yaml = {
        "path": ".",
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": len(class_names),
        "names": class_names,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f)

    logger.info(f"Dataset YAML created at {yaml_path}")
    logger.info(f"Classes: {class_names}")

    return str(yaml_path)


def _find_voc_dir(voc_root: Path) -> Path:
    """Find the actual VOC dataset directory (containing Annotations/, JPEGImages/).

    Handles three common layouts:
      a) voc_root/VOC2007/Annotations/    (standard inside VOCdevkit)
      b) voc_root/VOC2007/Annotations/    (no VOCdevkit wrapper)
      c) voc_root/Annotations/            (flat, no year subdirectory)
    """
    for year in ["VOC2007", "VOC2012", "VOC2010"]:
        candidate = voc_root / year
        if (candidate / "Annotations").exists() and (candidate / "JPEGImages").exists():
            return candidate

    if (voc_root / "Annotations").exists() and (voc_root / "JPEGImages").exists():
        return voc_root

    raise ValueError(
        f"No VOC dataset found in {voc_root}. "
        "Expected Annotations/ and JPEGImages/ at the root "
        "or inside a VOC2007/, VOC2012/, or VOC2010/ subdirectory."
    )


def convert_voc_to_yolo(
    voc_devkit_dir: str,
    output_dir: str,
    copy_images: bool = True,
    split: str = "trainval",
) -> str:
    """Convert VOC format dataset to YOLO format.

    Args:
        voc_devkit_dir: Path to VOC dataset. Supports both VOCdevkit/VOC2007/
                        layout and flat Annotations/+JPEGImages/ layout.
        output_dir: Output directory for YOLO format dataset.
        copy_images: Whether to copy images to output directory.
        split: Dataset split (train, val, test, trainval).

    Returns:
        Path to created dataset YAML file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    voc_dir = _find_voc_dir(Path(voc_devkit_dir))

    logger.info(f"Using VOC dataset from {voc_dir}")

    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    split_file = voc_dir / "ImageSets" / "Main" / f"{split}.txt"
    if split_file.exists():
        with open(split_file) as f:
            image_ids = [line.strip() for line in f.readlines()]
        logger.info("Using ImageSets split file with %d entries", len(image_ids))
    else:
        logger.warning(
            "No ImageSets/Main/%s.txt found — falling back to all .xml files in Annotations/",
            split,
        )
        annotations_dir = voc_dir / "Annotations"
        image_ids = sorted(
            p.stem for p in annotations_dir.glob("*.xml")
        )
        if not image_ids:
            raise ValueError(
                "No annotations found — neither ImageSets/Main/%s.txt "
                "nor any .xml files in %s exist" % (split, annotations_dir)
            )

    all_annotations: List[dict] = []
    all_class_names: Set[str] = set()

    for image_id in image_ids:
        xml_path = voc_dir / "Annotations" / f"{image_id}.xml"
        if not xml_path.exists():
            continue

        annotations, class_names = parse_voc_annotations(str(xml_path))
        all_annotations.extend(annotations)
        all_class_names.update(class_names)

    class_name_to_idx = {name: idx for idx, name in enumerate(sorted(all_class_names))}

    annotations_by_img: Dict[str, List[dict]] = {}
    for ann in all_annotations:
        filename = ann["filename"]
        if filename not in annotations_by_img:
            annotations_by_img[filename] = []
        ann["class_idx"] = class_name_to_idx[ann["class_name"]]
        annotations_by_img[filename].append(ann)

    logger.info(f"Converting {len(annotations_by_img)} images")
    for filename, img_annotations in annotations_by_img.items():
        label_filename = Path(filename).stem + ".txt"
        label_path = labels_dir / label_filename

        with open(label_path, "w") as f:
            for ann in img_annotations:
                bbox = ann["bbox"]
                class_idx = ann["class_idx"]
                f.write(f"{class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        if copy_images:
            src_image = voc_dir / "JPEGImages" / filename
            if src_image.exists():
                dst_image = images_dir / filename
                shutil.copy2(src_image, dst_image)

    yaml_path = output_path / "dataset.yaml"
    class_names_list = sorted(class_name_to_idx.keys())
    dataset_yaml = {
        "path": ".",
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": len(class_names_list),
        "names": class_names_list,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f)

    logger.info(f"Dataset YAML created at {yaml_path}")
    logger.info(f"Classes ({len(class_names_list)}): {class_names_list}")

    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO or VOC format datasets to YOLO format"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["coco", "voc"],
        required=True,
        help="Input format (coco or voc)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input path (COCO JSON file or VOCdevkit directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for YOLO format dataset",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Do not copy images to output directory",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        help="Source image directory (for COCO format)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="trainval",
        help="VOC dataset split (train, val, test, trainval)",
    )

    args = parser.parse_args()

    if args.format == "coco":
        yaml_path = convert_coco_to_yolo(
            args.input,
            args.output,
            copy_images=not args.no_copy_images,
            image_dir=args.image_dir,
        )
    else:
        yaml_path = convert_voc_to_yolo(
            args.input,
            args.output,
            copy_images=not args.no_copy_images,
            split=args.split,
        )

    print("\nConversion complete!")
    print(f"Dataset YAML: {yaml_path}")
    print("\nTo train with this dataset:")
    print(f"  uv run yolo-demo train {yaml_path} --epochs 100")


if __name__ == "__main__":
    main()
