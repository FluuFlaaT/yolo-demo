"""Gradio WebUI for YOLO Demo."""

import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from ..export.onnx_exporter import ONNXExporter, prepare_for_rk3588
from ..inference import Detection, DetectionResult, create_engine, get_available_backend
from ..training.trainer import TrainingConfig, Trainer


def draw_detections(image, detections: list[Detection]) -> Any:
    """Draw detection boxes on image."""
    try:
        import cv2
    except ImportError:
        return image

    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label
        label = f"{det.class_name}: {det.confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


def create_inference_tab() -> gr.Tab:
    """Create the inference tab."""
    backend = get_available_backend()

    with gr.Tab("Inference", id="inference") as tab:
        gr.Markdown("### Upload an image for object detection")
        gr.Markdown(f"Current backend: **{backend}**")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="numpy")
                detect_btn = gr.Button("Detect Objects", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Detection Result")
                detection_info = gr.JSON(label="Detections")

        def run_inference(image):
            if image is None:
                return None, {"error": "No image provided"}

            # Load model (using default YOLOv8n for demo)
            # In production, this should be configurable
            try:
                engine = create_engine("yolov8n.pt")
                engine.load_model()
                result: DetectionResult = engine.predict(image)

                output_img = draw_detections(image, result.detections)

                detections_dict = {
                    "count": len(result.detections),
                    "inference_time_ms": round(result.inference_time_ms, 2),
                    "device": result.device,
                    "detections": [
                        {
                            "class": det.class_name,
                            "confidence": round(det.confidence, 3),
                            "bbox": det.bbox,
                        }
                        for det in result.detections
                    ],
                }
                return output_img, detections_dict

            except Exception as e:
                return None, {"error": str(e)}

        detect_btn.click(
            fn=run_inference,
            inputs=[input_image],
            outputs=[output_image, detection_info],
        )

    return tab


def create_training_tab() -> gr.Tab:
    """Create the training tab."""
    with gr.Tab("Training", id="training") as tab:
        gr.Markdown("### Incremental Training")
        gr.Markdown("Upload a dataset in YOLO format and train a custom model")

        with gr.Row():
            with gr.Column():
                dataset_yaml = gr.File(label="Dataset YAML (.yaml)")
                pretrained_model = gr.File(label="Pretrained Model (optional)", file_types=[".pt"])

                with gr.Accordion("Training Parameters", open=False):
                    epochs = gr.Slider(10, 500, value=100, step=10, label="Epochs")
                    batch_size = gr.Slider(2, 64, value=16, step=2, label="Batch Size")
                    imgsz = gr.Slider(320, 1280, value=640, step=32, label="Image Size")
                    lr0 = gr.Number(value=0.01, label="Initial Learning Rate")

                train_btn = gr.Button("Start Training", variant="primary")

            with gr.Column():
                training_status = gr.Textbox(label="Status")
                training_output = gr.File(label="Trained Model")

        def run_training(dataset_yaml_file, pretrained_file, epochs, batch_size, imgsz, lr0):
            try:
                # Create training config
                config = TrainingConfig(
                    epochs=int(epochs),
                    batch=int(batch_size),
                    imgsz=int(imgsz),
                    lr0=float(lr0),
                )

                if pretrained_file:
                    config.model = pretrained_file.name

                # Initialize trainer
                trainer = Trainer(config)

                # Start training
                dataset_path = dataset_yaml_file.name if dataset_yaml_file else None
                if not dataset_path:
                    return "Error: Dataset YAML is required", None

                result = trainer.train(data_yaml=dataset_path)

                if result.success:
                    return f"Training completed! Model saved to: {result.model_path}", result.model_path
                else:
                    return f"Training failed: {result.error}", None

            except Exception as e:
                return f"Error: {str(e)}", None

        train_btn.click(
            fn=run_training,
            inputs=[dataset_yaml, pretrained_model, epochs, batch_size, imgsz, lr0],
            outputs=[training_status, training_output],
        )

    return tab


def create_export_tab() -> gr.Tab:
    """Create the export tab."""
    with gr.Tab("Export", id="export") as tab:
        gr.Markdown("### Export Model to ONNX")
        gr.Markdown("Export your YOLO model for deployment on edge devices (RK3588, etc.)")

        with gr.Row():
            with gr.Column():
                model_file = gr.File(label="Model (.pt file)", file_types=[".pt"])
                opset_version = gr.Dropdown(
                    choices=[10, 11, 12, 13, 14, 15],
                    value=11,
                    label="ONNX Opset Version",
                    info="Use 11 or 12 for RK3588 compatibility",
                )
                dynamic_axes = gr.Checkbox(value=True, label="Enable Dynamic Axes")
                simplify = gr.Checkbox(value=True, label="Simplify Model")
                export_btn = gr.Button("Export to ONNX", variant="primary")

            with gr.Column():
                export_status = gr.Textbox(label="Status")
                exported_file = gr.File(label="Exported Model")

        def run_export(model, opset, dynamic, simplify):
            try:
                if not model:
                    return "Error: Please upload a model file", None

                exporter = ONNXExporter(model.name)
                onnx_path = exporter.export(
                    opset=int(opset),
                    dynamic=dynamic,
                    simplify=simplify,
                )

                return f"Exported successfully: {onnx_path}", onnx_path

            except Exception as e:
                return f"Export failed: {str(e)}", None

        export_btn.click(
            fn=run_export,
            inputs=[model_file, opset_version, dynamic_axes, simplify],
            outputs=[export_status, exported_file],
        )

        # RK3588 quick export
        gr.Markdown("### Quick Export for RK3588")
        rk3588_btn = gr.Button("Export for RK3588 (Recommended Settings)")

        def run_rk3588_export(model):
            try:
                if not model:
                    return "Error: Please upload a model file", None

                onnx_path = prepare_for_rk3588(model.name)
                return f"RK3588-ready ONNX exported: {onnx_path}", onnx_path

            except Exception as e:
                return f"Export failed: {str(e)}", None

        rk3588_btn.click(
            fn=run_rk3588_export,
            inputs=[model_file],
            outputs=[export_status, exported_file],
        )

    return tab


def create_webui() -> gr.Blocks:
    """Create the complete Gradio WebUI."""
    with gr.Blocks(title="YOLO Demo - Object Detection") as app:
        gr.Markdown(
            """
            # 🎯 YOLO Demo - Real-time Object Detection
            Lightweight object detection system for edge computing.
            Supports Mac (MPS), NVIDIA (CUDA), and CPU backends.
            """
        )

        with gr.Tabs():
            create_inference_tab()
            create_training_tab()
            create_export_tab()

        gr.Markdown(
            """
            ---
            *Built with Gradio and Ultralytics YOLO*
            """
        )

    return app


def launch(host: str = "0.0.0.0", port: int = 7860, **kwargs: Any) -> None:
    """Launch the WebUI."""
    app = create_webui()
    app.launch(server_name=host, server_port=port, **kwargs)


if __name__ == "__main__":
    launch()
