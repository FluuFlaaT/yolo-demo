"""Main CLI entry point for YOLO Demo."""

import argparse
import sys
from pathlib import Path

import yolo_demo


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YOLO Demo - Lightweight real-time object detection for edge computing"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {yolo_demo.__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference on an image")
    infer_parser.add_argument("image", type=str, help="Path to input image")
    infer_parser.add_argument(
        "--model", "-m", type=str, default="yolov8n.pt", help="Model path or name"
    )
    infer_parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output image path"
    )
    infer_parser.add_argument(
        "--conf", "-c", type=float, default=0.25, help="Confidence threshold"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a YOLO model")
    train_parser.add_argument("data", type=str, help="Path to dataset YAML")
    train_parser.add_argument(
        "--model", "-m", type=str, default="yolov8n.pt", help="Pretrained model path"
    )
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output directory"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("model", type=str, help="Path to model (.pt file)")
    export_parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output path"
    )
    export_parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    export_parser.add_argument(
        "--rk3588", action="store_true", help="Export for RK3588 (recommended settings)"
    )

    # WebUI command
    webui_parser = subparsers.add_parser("webui", help="Launch Gradio WebUI")
    webui_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    webui_parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    webui_parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link"
    )

    # API command
    api_parser = subparsers.add_parser("api", help="Launch FastAPI server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "infer":
        run_inference(args.image, args.model, args.output, args.conf)
    elif args.command == "train":
        run_training(args.data, args.model, args.epochs, args.batch, args.imgsz, args.output)
    elif args.command == "export":
        run_export(args.model, args.output, args.opset, args.rk3588)
    elif args.command == "webui":
        run_webui(args.host, args.port, args.share)
    elif args.command == "api":
        run_api(args.host, args.port, args.reload)


def run_inference(image: str, model: str, output: str | None, conf: float):
    """Run inference on an image."""
    import cv2
    import numpy as np

    from yolo_demo.inference import create_engine

    # Load image
    img = cv2.imread(image)
    if img is None:
        print(f"Error: Could not read image: {image}")
        sys.exit(1)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    print(f"Loading model: {model}")
    engine = create_engine(model)
    engine.load_model()

    print(f"Running inference on: {image}")
    print(f"Device: {engine.get_device_info()['backend']}")

    result = engine.predict(img_rgb)

    print(f"\nInference time: {result.inference_time_ms:.2f}ms")
    print(f"Detected {len(result.detections)} objects:")
    for det in result.detections:
        print(f"  - {det.class_name}: {det.confidence:.2f} [{det.bbox}]")

    # Draw and save if output specified
    if output:
        for det in result.detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(output, img)
        print(f"\nOutput saved to: {output}")


def run_training(
    data: str, model: str, epochs: int, batch: int, imgsz: int, output: str | None
):
    """Run training."""
    from yolo_demo.training.trainer import TrainingConfig, Trainer

    config = TrainingConfig(
        model=model,
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=output,
    )

    trainer = Trainer(config)
    print(f"Starting training with {model}...")
    print(f"Dataset: {data}")
    print(f"Epochs: {epochs}, Batch: {batch}, Image size: {imgsz}")

    result = trainer.train()

    if result.success:
        print(f"\nTraining completed!")
        print(f"Model saved to: {result.model_path}")
    else:
        print(f"\nTraining failed: {result.error}")
        sys.exit(1)


def run_export(model: str, output: str | None, opset: int, rk3588: bool):
    """Export model to ONNX."""
    from yolo_demo.export.onnx_exporter import ONNXExporter, prepare_for_rk3588

    if rk3588:
        print(f"Exporting {model} for RK3588...")
        onnx_path = prepare_for_rk3588(model, output)
    else:
        print(f"Exporting {model} to ONNX (opset={opset})...")
        exporter = ONNXExporter(model)
        onnx_path = exporter.export(output=output, opset=opset)

    print(f"Exported to: {onnx_path}")


def run_webui(host: str, port: int, share: bool):
    """Launch WebUI."""
    from yolo_demo.ui.webui import create_webui

    app = create_webui()
    print(f"Launching WebUI at http://{host}:{port}")
    app.launch(server_name=host, server_port=port, share=share)


def run_api(host: str, port: int, reload: bool):
    """Launch API server."""
    import uvicorn

    print(f"Launching API at http://{host}:{port}")
    uvicorn.run("yolo_demo.api.app:app", host=host, port=port, reload=reload)


def cli():
    """CLI entry point."""
    main()


if __name__ == "__main__":
    main()
