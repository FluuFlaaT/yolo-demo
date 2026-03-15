#!/usr/bin/env python3
"""
Generate calibration dataset for RKNN INT8 quantization.

The calibration dataset is used to collect activation statistics during
quantization. A good calibration dataset should:
- Contain 100-1000 representative images
- Cover various scenarios and lighting conditions
- Include all object classes that need to be detected

Usage:
    # Generate from image directory
    python generate_calib_dataset.py /path/to/images -o calib.txt

    # Generate with specific count
    python generate_calib_dataset.py /path/to/images -o calib.txt -n 500

    # Generate from COCO dataset
    python generate_calib_dataset.py /path/to/coco --coco -o calib.txt
"""

import argparse
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images_from_directory(
    image_dir: str | Path,
    max_count: int = 1000,
    recursive: bool = True,
) -> list[Path]:
    """
    Collect images from a directory.

    Args:
        image_dir: Directory containing images
        max_count: Maximum number of images to collect
        recursive: Search recursively in subdirectories

    Returns:
        List of image paths
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    logger.info(f"Collecting images from: {image_dir}")

    if recursive:
        images = [
            p for p in image_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        images = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    logger.info(f"Found {len(images)} images")

    if len(images) > max_count:
        logger.info(f"Randomly sampling {max_count} images")
        images = random.sample(images, max_count)

    return images


def collect_images_from_coco(
    coco_root: str | Path,
    split: str = "val2017",
    max_count: int = 1000,
) -> list[Path]:
    """
    Collect images from COCO dataset.

    Args:
        coco_root: Root directory of COCO dataset
        split: Dataset split (train2017, val2017, test2017)
        max_count: Maximum number of images to collect

    Returns:
        List of image paths
    """
    coco_root = Path(coco_root)
    images_dir = coco_root / "images" / split

    if not images_dir.exists():
        raise FileNotFoundError(f"COCO images not found: {images_dir}")

    return collect_images_from_directory(images_dir, max_count=max_count)


def generate_dataset_file(
    images: list[Path],
    output_path: str | Path,
    copy_images: bool = False,
    output_dir: Optional[str | Path] = None,
) -> str:
    """
    Generate calibration dataset file.

    The dataset file contains one image path per line.

    Args:
        images: List of image paths
        output_path: Path to output dataset file
        copy_images: Whether to copy images to output directory
        output_dir: Output directory for copied images

    Returns:
        Path to generated dataset file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if copy_images:
        if output_dir is None:
            output_dir = output_path.parent / "calib_images"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Copying {len(images)} images to: {output_dir}")

        for i, img_path in enumerate(images):
            if (i + 1) % 100 == 0:
                logger.info(f"Copying {i + 1}/{len(images)} images...")

            # Generate unique filename
            new_name = f"calib_{i:05d}{img_path.suffix}"
            dest = output_dir / new_name

            try:
                shutil.copy2(img_path, dest)
            except Exception as e:
                logger.warning(f"Failed to copy {img_path}: {e}")

        # Write dataset file with relative paths
        with open(output_path, "w") as f:
            for i in range(len(images)):
                rel_path = output_dir / f"calib_{i:05d}{images[i].suffix}"
                f.write(f"{rel_path}\n")

    else:
        # Write absolute paths
        with open(output_path, "w") as f:
            for img_path in images:
                f.write(f"{img_path.absolute()}\n")

    logger.info(f"Dataset file generated: {output_path}")
    logger.info(f"Total images: {len(images)}")

    return str(output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate calibration dataset for RKNN INT8 quantization"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing images (or COCO root if --coco)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output dataset file path"
    )
    parser.add_argument(
        "-n", "--num-images",
        type=int,
        default=500,
        help="Number of images to collect (default: 500)"
    )
    parser.add_argument(
        "--coco",
        action="store_true",
        help="Input is COCO dataset format"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images to output directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for copied images (default: calib_images/)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Search recursively (default: True)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    try:
        # Collect images
        if args.coco:
            images = collect_images_from_coco(
                args.input_dir,
                max_count=args.num_images,
            )
        else:
            images = collect_images_from_directory(
                args.input_dir,
                max_count=args.num_images,
                recursive=args.recursive,
            )

        if len(images) == 0:
            logger.error("No images found")
            sys.exit(1)

        # Generate dataset file
        dataset_path = generate_dataset_file(
            images=images,
            output_path=args.output,
            copy_images=args.copy,
            output_dir=args.output_dir,
        )

        logger.info("=" * 60)
        logger.info("Calibration dataset generated successfully!")
        logger.info(f"Dataset file: {dataset_path}")
        logger.info(f"Number of images: {len(images)}")
        logger.info("")
        logger.info("Usage with convert_to_rknn.py:")
        logger.info(f"  python convert_to_rknn.py model.onnx --quantize --dataset {dataset_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
