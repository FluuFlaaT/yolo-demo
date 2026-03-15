#!/usr/bin/env python3
"""
RKNN inference script for RK3588 edge devices.

This script runs inference using RKNN models on RK3588 NPU.

Usage:
    # Basic inference
    python inference_rknn.py image.jpg model.rknn

    # With output saving
    python inference_rknn.py image.jpg model.rknn -o output.jpg

    # Adjust thresholds
    python inference_rknn.py image.jpg model.rknn --conf 0.5 --iou 0.5

    # Video inference
    python inference_rknn.py video.mp4 model.rknn --output video_out.mp4

    # Camera inference
    python inference_rknn.py --camera 0 model.rknn
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_class_names(path: Optional[str] = None) -> list[str]:
    """
    Load COCO class names.

    Args:
        path: Optional path to class names file (one per line)

    Returns:
        List of class names
    """
    if path and Path(path).exists():
        with open(path) as f:
            return [line.strip() for line in f.readlines()]

    # Default COCO classes
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ]


def draw_detections(
    image: np.ndarray,
    detections: list,
    class_names: list[str],
) -> np.ndarray:
    """
    Draw detections on image.

    Args:
        image: Input image (BGR format)
        detections: List of Detection objects
        class_names: List of class names

    Returns:
        Image with drawn detections
    """
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)

        # Get color for class
        class_id = det.class_id % len(class_names)
        color = (
            ((class_id * 127) % 255),
            ((class_id * 73) % 255),
            ((class_id * 51) % 255),
        )

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_names[det.class_id]}: {det.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )

        # Label background
        cv2.rectangle(
            image,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1,
        )

        # Label text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    return image


def run_inference(
    image_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_names_path: Optional[str] = None,
) -> None:
    """
    Run inference on a single image.

    Args:
        image_path: Path to input image
        model_path: Path to RKNN model
        output_path: Optional path to save output image
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        class_names_path: Optional path to class names file
    """
    from yolo_demo.inference import create_rknn_engine

    # Load class names
    class_names = load_class_names(class_names_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        sys.exit(1)

    logger.info(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # Create inference engine
    logger.info(f"Loading RKNN model: {model_path}")
    engine = create_rknn_engine(
        model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    engine.load_model()

    device_info = engine.get_device_info()
    logger.info(f"Device: {device_info['backend']}")
    logger.info(f"Input shape: {device_info['input_shape']}")

    # Convert BGR to RGB for inference
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    logger.info("Running inference...")
    result = engine.predict(image_rgb)

    logger.info(f"Inference time: {result.inference_time_ms:.2f}ms")
    logger.info(f"Detected {len(result.detections)} objects:")

    for det in result.detections:
        class_name = class_names[det.class_id] if det.class_id < len(class_names) else str(det.class_id)
        logger.info(f"  - {class_name}: {det.confidence:.2f} [{det.bbox}]")

    # Draw detections
    image_with_dets = draw_detections(image, result.detections, class_names)

    # Save or display result
    if output_path:
        cv2.imwrite(output_path, image_with_dets)
        logger.info(f"Output saved to: {output_path}")
    else:
        # Display in window (requires GUI environment)
        cv2.imshow("Detections", image_with_dets)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video_inference(
    video_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_names_path: Optional[str] = None,
) -> None:
    """
    Run inference on a video file.

    Args:
        video_path: Path to input video
        model_path: Path to RKNN model
        output_path: Optional path to save output video
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        class_names_path: Optional path to class names file
    """
    from yolo_demo.inference import create_rknn_engine

    # Load class names
    class_names = load_class_names(class_names_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Create video writer if output specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Output video: {output_path}")

    # Create inference engine
    logger.info(f"Loading RKNN model: {model_path}")
    engine = create_rknn_engine(
        model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    engine.load_model()

    frame_count = 0
    total_inference_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            logger.info(f"Processing frame {frame_count}/{total_frames}")

        # Run inference
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        result = engine.predict(image_rgb)
        inference_time = (time.time() - start_time) * 1000
        total_inference_time += inference_time

        # Draw detections
        frame_with_dets = draw_detections(frame, result.detections, class_names)

        # Add FPS counter
        avg_fps = 1000 / (total_inference_time / frame_count) if frame_count > 0 else 0
        cv2.putText(
            frame_with_dets,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Write frame
        if writer:
            writer.write(frame_with_dets)
        else:
            # Display
            cv2.imshow("Detection", frame_with_dets)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    logger.info(f"Processed {frame_count} frames")
    logger.info(f"Average inference time: {total_inference_time / frame_count:.2f}ms")
    logger.info(f"Average FPS: {1000 / (total_inference_time / frame_count):.1f}")


def run_camera_inference(
    camera_id: int,
    model_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_names_path: Optional[str] = None,
) -> None:
    """
    Run inference on camera stream.

    Args:
        camera_id: Camera device ID
        model_path: Path to RKNN model
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        class_names_path: Optional path to class names file
    """
    from yolo_demo.inference import create_rknn_engine

    # Load class names
    class_names = load_class_names(class_names_path)

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Could not open camera: {camera_id}")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"Camera {camera_id}: {width}x{height} @ {fps:.1f}fps")

    # Create inference engine
    logger.info(f"Loading RKNN model: {model_path}")
    engine = create_rknn_engine(
        model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    engine.load_model()

    frame_count = 0
    total_inference_time = 0

    logger.info("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            break

        frame_count += 1

        # Run inference
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        result = engine.predict(image_rgb)
        inference_time = (time.time() - start_time) * 1000
        total_inference_time += inference_time

        # Draw detections
        frame_with_dets = draw_detections(frame, result.detections, class_names)

        # Add FPS counter
        avg_fps = 1000 / (total_inference_time / frame_count) if frame_count > 0 else 0
        cv2.putText(
            frame_with_dets,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Display
        cv2.imshow("RKNN Detection", frame_with_dets)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    logger.info(f"Processed {frame_count} frames")
    logger.info(f"Average FPS: {1000 / (total_inference_time / frame_count):.1f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RKNN inference on RK3588 edge devices"
    )
    parser.add_argument(
        "image_or_video",
        nargs="?",
        help="Path to input image or video (omit for camera mode)",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to RKNN model file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path (image or video)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Path to class names file"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=None,
        help="Camera device ID (default: None, use 0 for built-in camera)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.camera and not args.image_or_video:
        parser.error("Please provide an image/video path or use --camera")

    if not Path(args.model).exists():
        parser.error(f"Model not found: {args.model}")

    # Run appropriate mode
    try:
        if args.camera is not None:
            run_camera_inference(
                camera_id=args.camera,
                model_path=args.model,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                class_names_path=args.classes,
            )
        elif args.image_or_video:
            input_path = Path(args.image_or_video)
            suffix = input_path.suffix.lower()

            if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                run_inference(
                    image_path=str(input_path),
                    model_path=args.model,
                    output_path=args.output,
                    conf_threshold=args.conf,
                    iou_threshold=args.iou,
                    class_names_path=args.classes,
                )
            elif suffix in [".mp4", ".avi", ".mov", ".mkv", ".m4v"]:
                run_video_inference(
                    video_path=str(input_path),
                    model_path=args.model,
                    output_path=args.output,
                    conf_threshold=args.conf,
                    iou_threshold=args.iou,
                    class_names_path=args.classes,
                )
            else:
                parser.error(f"Unknown file type: {suffix}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
