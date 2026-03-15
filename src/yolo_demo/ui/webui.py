"""Gradio WebUI for YOLO Demo."""

import logging
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Generator

import gradio as gr

from ..export.onnx_exporter import ONNXExporter, prepare_for_rk3588
from ..inference import Detection, DetectionResult, create_engine, get_available_backend
from ..training.trainer import TrainingConfig, Trainer

from .dataset_converter import create_dataset_converter_tab
from .rknn_validator import create_rknn_validation_tab

# Module-level state for the training thread/stop mechanism (single-user demo)
_stop_event: threading.Event = threading.Event()
_train_thread: threading.Thread | None = None

logger = logging.getLogger(__name__)


class _QueueLogHandler(logging.Handler):
    """Logging handler that forwards formatted records into a queue.Queue.

    Log records are put as plain strings.  The training thread signals
    completion by putting a tuple  ("done", model_path)  or
    ("error", message)  into the same queue.
    """

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


# Predefined Ultralytics models
ULTRALYTICS_MODELS = [
    # YOLOv8
    ("YOLOv8n (nano, fastest)", "yolov8n.pt"),
    ("YOLOv8s (small)", "yolov8s.pt"),
    ("YOLOv8m (medium)", "yolov8m.pt"),
    ("YOLOv8l (large)", "yolov8l.pt"),
    ("YOLOv8x (xlarge, most accurate)", "yolov8x.pt"),
    # YOLO11
    ("YOLO11n (nano)", "yolo11n.pt"),
    ("YOLO11s (small)", "yolo11s.pt"),
    ("YOLO11m (medium)", "yolo11m.pt"),
    ("YOLO11l (large)", "yolo11l.pt"),
    ("YOLO11x (xlarge)", "yolo11x.pt"),
    # YOLOv9
    ("YOLOv9c (compact)", "yolov9c.pt"),
    ("YOLOv9e (extended)", "yolov9e.pt"),
    # YOLOv10
    ("YOLOv10n (nano)", "yolov10n.pt"),
    ("YOLOv10s (small)", "yolov10s.pt"),
    ("YOLOv10m (medium)", "yolov10m.pt"),
    ("YOLOv10b (balanced)", "yolov10b.pt"),
    ("YOLOv10l (large)", "yolov10l.pt"),
    ("YOLOv10x (xlarge)", "yolov10x.pt"),
]


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

                # Model selection
                model_type = gr.Radio(
                    choices=["Pretrained Model", "Custom Model"],
                    value="Pretrained Model",
                    label="Model Source",
                )

                pretrained_model = gr.Dropdown(
                    choices=ULTRALYTICS_MODELS,
                    value="yolov8n.pt",
                    label="Select Pretrained Model",
                    visible=True,
                )

                custom_model = gr.File(
                    label="Upload Custom Model (.pt)",
                    visible=False,
                )

                conf_threshold = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.25,
                    step=0.01,
                    label="Confidence Threshold",
                )

                detect_btn = gr.Button("Detect Objects", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Detection Result")
                detection_info = gr.JSON(label="Detections")

        def toggle_model_source(source):
            """Toggle between pretrained and custom model inputs."""
            if source == "Pretrained Model":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        def run_inference(image, model_type, pretrained, custom_model, conf):
            if image is None:
                return None, {"error": "No image provided"}

            # Determine model path
            if model_type == "Pretrained Model":
                model_path = pretrained
            else:
                if custom_model is None:
                    return None, {"error": "Please upload a custom model"}
                model_path = custom_model.name

            logger.info(f"Running inference with model: {model_path}")

            try:
                engine = create_engine(model_path)
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
                        if det.confidence >= conf
                    ],
                }
                return output_img, detections_dict

            except Exception as e:
                logger.error(f"Inference failed: {e}")
                return None, {"error": str(e)}

        detect_btn.click(
            fn=run_inference,
            inputs=[input_image, model_type, pretrained_model, custom_model, conf_threshold],
            outputs=[output_image, detection_info],
        )

        model_type.change(
            fn=toggle_model_source,
            inputs=[model_type],
            outputs=[pretrained_model, custom_model],
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

                # Base model selection — mirrors Inference tab pattern
                base_model_source = gr.Radio(
                    choices=["From List", "Upload File"],
                    value="From List",
                    label="Base Model Source",
                )
                base_model_dropdown = gr.Dropdown(
                    choices=ULTRALYTICS_MODELS,
                    value="yolov8n.pt",
                    label="Select Base Model",
                    visible=True,
                )
                base_model_file = gr.File(
                    label="Upload Base Model (.pt)",
                    visible=False,
                )

                with gr.Accordion("Training Parameters", open=False):
                    epochs = gr.Slider(10, 500, value=100, step=10, label="Epochs")
                    batch_size = gr.Slider(2, 64, value=16, step=2, label="Batch Size")
                    imgsz = gr.Slider(320, 1280, value=640, step=32, label="Image Size")
                    lr0 = gr.Number(value=0.01, label="Initial Learning Rate")

                    # Output directory setting
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        placeholder="Leave empty for default (runs/detect/train)",
                        value="",
                    )

                with gr.Row():
                    train_btn = gr.Button("Start Training", variant="primary")
                    stop_btn = gr.Button("Stop Training", variant="stop")

            with gr.Column():
                training_status = gr.Textbox(
                    label="Status",
                    lines=20,
                    max_lines=40,
                    buttons=["copy"],
                    placeholder="Training log will stream here...",
                )
                training_output = gr.File(label="Trained Model")

        def toggle_base_model_source(source):
            if source == "From List":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        def run_training(
            dataset_yaml_file,
            base_source,
            base_dropdown,
            base_file,
            epochs_val,
            batch_size_val,
            imgsz_val,
            lr0_val,
            output,
        ) -> Generator:
            global _stop_event, _train_thread

            # Validate inputs
            if dataset_yaml_file is None:
                yield "Error: Dataset YAML is required", None
                return

            if base_source == "Upload File" and base_file is None:
                yield "Error: Please upload a base model file", None
                return

            # Determine base model
            if base_source == "From List":
                model_path = base_dropdown
            else:
                model_path = base_file.name

            # Reset stop signal
            _stop_event.clear()

            # Single queue: str → log line, tuple → ("done"|"error", payload)
            msg_queue: queue.Queue = queue.Queue()

            # Attach a log handler to the root logger so every Python logging
            # call made anywhere in the process flows into the textbox.
            _log_handler = _QueueLogHandler(msg_queue)
            _log_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            _root_logger = logging.getLogger()
            _root_logger.addHandler(_log_handler)

            def training_thread_fn():
                try:
                    config = TrainingConfig(
                        model=model_path,
                        epochs=int(epochs_val),
                        batch=int(batch_size_val),
                        imgsz=int(imgsz_val),
                        lr0=float(lr0_val),
                    )
                    if output:
                        config.project = output

                    trainer = Trainer(config)
                    trainer.load_pretrained()
                    yolo_model = trainer.model
                    assert yolo_model is not None

                    # Stop callback: set epoch = total epochs for a clean early exit
                    stop_ev = _stop_event

                    def _on_epoch_end(t):
                        if stop_ev.is_set():
                            t.epoch = t.epochs
                            logger.info("Stop requested — finishing current epoch and exiting.")

                    yolo_model.add_callback("on_train_epoch_end", _on_epoch_end)

                    dataset_path = dataset_yaml_file.name
                    logger.info(f"Starting training with config: {config.to_dict()}")
                    result = trainer.train(data_yaml=dataset_path)

                    if result.success:
                        msg_queue.put(("done", result.model_path or ""))
                    else:
                        msg_queue.put(("error", result.error or "Unknown error"))

                except Exception as e:
                    msg_queue.put(("error", str(e)))

            _train_thread = threading.Thread(target=training_thread_fn, daemon=True)
            _train_thread.start()

            log_lines: list[str] = []

            try:
                while _train_thread.is_alive() or not msg_queue.empty():
                    try:
                        msg = msg_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if isinstance(msg, tuple):
                        kind, payload = msg
                        if kind == "done":
                            final_model_path = payload.strip()
                            serve_path = None
                            src = Path(final_model_path)
                            try:
                                if src.exists():
                                    import shutil

                                    dst = Path(tempfile.gettempdir()) / "yolo_trained_best.pt"
                                    shutil.copy2(src, dst)
                                    serve_path = str(dst)
                                    logger.info(f"Model copied to temp for download: {dst}")
                                else:
                                    logger.warning(
                                        f"Expected model not found at: {final_model_path}"
                                    )
                            except Exception as copy_err:
                                logger.warning(f"Could not copy model: {copy_err}")
                            logger.info(f"Training completed. Model saved to: {final_model_path}")
                            if serve_path is None:
                                logger.warning("File not available for download — see path above.")
                            yield "".join(log_lines), serve_path
                            return
                        else:  # "error"
                            logger.error(f"Training failed: {payload}")
                            yield "".join(log_lines), None
                            return
                    else:
                        log_lines.append(msg)
                        yield "".join(log_lines), None

            finally:
                _root_logger.removeHandler(_log_handler)

            yield "".join(log_lines), None

        def stop_training():
            global _stop_event
            _stop_event.set()

        train_btn.click(
            fn=run_training,
            inputs=[
                dataset_yaml,
                base_model_source,
                base_model_dropdown,
                base_model_file,
                epochs,
                batch_size,
                imgsz,
                lr0,
                output_dir,
            ],
            outputs=[training_status, training_output],
        )

        stop_btn.click(fn=stop_training, inputs=[], outputs=[])

        base_model_source.change(
            fn=toggle_base_model_source,
            inputs=[base_model_source],
            outputs=[base_model_dropdown, base_model_file],
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

                # Custom output filename
                output_filename = gr.Textbox(
                    label="Output Filename (optional)",
                    placeholder="e.g., my-model-rk3588-export.onnx",
                    value="",
                )

                export_btn = gr.Button("Export to ONNX", variant="primary")

            with gr.Column():
                export_status = gr.Textbox(label="Status", lines=3)
                exported_file = gr.File(label="Exported Model")

        def run_export(model, opset, dynamic, simplify, output_filename):
            try:
                if not model:
                    return "Error: Please upload a model file", None

                exporter = ONNXExporter(model.name)

                # Handle custom output filename
                export_kwargs = {
                    "opset": int(opset),
                    "dynamic": dynamic,
                    "simplify": simplify,
                }

                if output_filename:
                    # Ensure .onnx extension
                    if not output_filename.endswith(".onnx"):
                        output_filename += ".onnx"
                    export_kwargs["output_filename"] = output_filename

                onnx_path = exporter.export(**export_kwargs)

                logger.info(f"Model exported to: {onnx_path}")
                return f"Exported successfully!\n\nOutput:\n{onnx_path}", onnx_path

            except Exception as e:
                logger.error(f"Export failed: {e}")
                return f"Export failed: {str(e)}", None

        export_btn.click(
            fn=run_export,
            inputs=[model_file, opset_version, dynamic_axes, simplify, output_filename],
            outputs=[export_status, exported_file],
        )

        # RK3588 quick export
        gr.Markdown("### Quick Export for RK3588")
        gr.Markdown("Exports with recommended settings for RK3588 (opset=11, dynamic, simplified)")

        rk3588_filename = gr.Textbox(
            label="Output Filename",
            placeholder="model-rk3588-export.onnx",
            value="",
        )

        rk3588_btn = gr.Button("Export for RK3588 (Recommended Settings)")

        def run_rk3588_export(model, filename):
            try:
                if not model:
                    return "Error: Please upload a model file", None

                # Generate default filename if not provided
                if not filename:
                    model_name = Path(model.name).stem
                    filename = f"{model_name}-rk3588-export.onnx"
                elif not filename.endswith(".onnx"):
                    filename += ".onnx"

                # Use the prepare_for_rk3588 function but with custom output
                exporter = ONNXExporter(model.name)
                onnx_path = exporter.export(
                    opset=11,
                    dynamic=True,
                    simplify=True,
                    output_filename=filename,
                )

                logger.info(f"RK3588 model exported to: {onnx_path}")
                return f"RK3588-ready ONNX exported!\n\nOutput:\n{onnx_path}", onnx_path

            except Exception as e:
                logger.error(f"RK3588 export failed: {e}")
                return f"Export failed: {str(e)}", None

        rk3588_btn.click(
            fn=run_rk3588_export,
            inputs=[model_file, rk3588_filename],
            outputs=[export_status, exported_file],
        )

    return tab


def create_webui() -> gr.Blocks:
    """Create the complete Gradio WebUI."""
    with gr.Blocks(title="YOLO Demo - Object Detection") as app:
        gr.Markdown(
            """
            # YOLO Demo - Real-time Object Detection
            Lightweight object detection system for edge computing.
            Supports Mac (MPS), NVIDIA (CUDA), and CPU backends.
            """
        )

        with gr.Tabs():
            create_inference_tab()
            create_training_tab()
            create_export_tab()
            create_dataset_converter_tab()
            create_rknn_validation_tab()

        gr.Markdown(
            """
            ---
            *Built with Gradio and Ultralytics YOLO*
            """
        )

    return app


def launch(host: str = "0.0.0.0", port: int = 7860, **kwargs: Any) -> None:
    """Launch the WebUI."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = create_webui()
    logger.info(f"Launching WebUI at http://{host}:{port}")
    # Allow serving files from anywhere under the user's home directory so
    # that trained models written by Ultralytics (e.g. ~/runs/...) can be
    # downloaded via the File output component.
    kwargs.setdefault("allowed_paths", [str(Path.home())])
    app.launch(server_name=host, server_port=port, **kwargs)


if __name__ == "__main__":
    launch()
