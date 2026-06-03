"""Gradio WebUI for YOLO Demo."""

import logging
import queue
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Generator, Optional


# Apply gradio_client compatibility patch for Pydantic v2
def _patch_gradio_client():
    """Patch gradio_client to handle Pydantic v2 schemas with boolean additionalProperties."""
    try:
        import gradio_client.utils as utils

        original_json_schema_to_python_type = utils._json_schema_to_python_type

        def patched_json_schema_to_python_type(schema, defs=None):
            """Patched version that handles boolean additionalProperties."""
            if not isinstance(schema, dict):
                if schema is True:
                    return "Any"
                elif schema is False:
                    return "None"
                return str(schema)
            return original_json_schema_to_python_type(schema, defs)

        utils._json_schema_to_python_type = patched_json_schema_to_python_type
    except Exception:
        pass  # Patching failed, but continue anyway


_patch_gradio_client()

import gradio as gr

from ..export import pt_to_rknn
from ..inference import Detection, DetectionResult, create_engine, get_available_backend
from ..training.trainer import Trainer, TrainingConfig
from .dataset_converter import create_dataset_converter_tab

# Module-level state for the training thread/stop mechanism (single-user demo)
_stop_event: threading.Event = threading.Event()
_train_thread: Optional[threading.Thread] = None

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
                model_name = gr.Textbox(
                    label="Model Name",
                    placeholder="e.g., yolov8n.pt, yolo26s, custom-model",
                    value="yolov8n.pt",
                )

                custom_model = gr.File(
                    label="Or Upload Custom Model (.pt)",
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

        def run_inference(image, model_name, custom_model, conf):
            if image is None:
                return None, {"error": "No image provided"}

            # Determine model path - prefer custom model file, then text input
            if custom_model is not None:
                model_path = custom_model.name
            elif model_name:
                model_path = model_name
            else:
                return None, {"error": "Please enter a model name or upload a custom model"}

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
            inputs=[input_image, model_name, custom_model, conf_threshold],
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

                # Base model selection
                base_model_name = gr.Textbox(
                    label="Base Model Name",
                    placeholder="e.g., yolov8n.pt, yolo26s, custom-model",
                    value="yolov8n.pt",
                )
                base_model_file = gr.File(
                    label="Or Upload Base Model (.pt)",
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
                    placeholder="Training log will stream here...",
                )
                training_output = gr.File(label="Trained Model")

        def run_training(
            dataset_yaml_file,
            base_model_name,
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

            # Determine base model - prefer file upload, then text input
            if base_file is not None:
                model_path = base_file.name
            elif base_model_name:
                model_path = base_model_name
            else:
                yield "Error: Please enter a base model name or upload a model file", None
                return

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
                base_model_name,
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

    return tab


def create_export_tab() -> gr.Tab:
    """Create the export tab."""
    with gr.Tab("Export", id="export") as tab:
        gr.Markdown("### Export Model to RKNN")
        gr.Markdown("Export your YOLO model directly to RKNN format for Rockchip devices")

        with gr.Row():
            with gr.Column():
                pt_model_file = gr.File(label="Model (.pt file)", file_types=[".pt"])
                platform = gr.Dropdown(
                    choices=[
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
                    ],
                    value="rk3588",
                    label="Target Platform",
                )
                imgsz = gr.Slider(320, 1280, value=640, step=32, label="Image Size")
                export_btn = gr.Button("Export to RKNN", variant="primary")

            with gr.Column():
                export_status = gr.Textbox(label="Status", lines=5)
                exported_file = gr.File(label="RKNN Model")

        def run_pt_to_rknn(model, platform, imgsz):
            try:
                if not model:
                    return "Error: Please upload a model file", None

                rknn_path = pt_to_rknn(
                    model.name,
                    target_platform=platform,
                    imgsz=int(imgsz),
                )

                rknn_dir = Path(rknn_path)
                zip_path = str(rknn_dir.with_suffix(".zip"))
                shutil.make_archive(
                    str(rknn_dir.with_suffix("")),
                    "zip",
                    rknn_dir,
                )

                logger.info(f"RKNN model exported to: {rknn_path}")
                return (
                    f"RKNN export successful!\n\nPlatform: {platform}\nDownload: {zip_path}",
                    zip_path,
                )

            except Exception as e:
                logger.error(f"RKNN export failed: {e}")
                return f"Export failed: {str(e)}", None

        export_btn.click(
            fn=run_pt_to_rknn,
            inputs=[pt_model_file, platform, imgsz],
            outputs=[export_status, exported_file],
        )

        gr.Markdown("---")
        gr.Markdown("### Convert ONNX to RKNN")
        gr.Markdown("Convert existing ONNX model to RKNN format (requires rknn-toolkit2)")

        with gr.Row():
            with gr.Column():
                onnx_file = gr.File(
                    label="ONNX Model (.onnx)",
                    file_types=[".onnx"],
                )
                onnx_platform = gr.Dropdown(
                    choices=[
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
                    ],
                    value="rk3588",
                    label="Target Platform",
                )
                onnx_quantize = gr.Checkbox(
                    value=False,
                    label="Enable INT8 Quantization",
                    info="Requires calibration dataset",
                )
                onnx_dataset = gr.File(
                    label="Calibration Dataset (txt file with image paths)",
                    file_types=[".txt"],
                    visible=False,
                )
                onnx_output = gr.Textbox(
                    label="Output Filename (optional)",
                    placeholder="model.rknn",
                    value="",
                )
                onnx_convert_btn = gr.Button("Convert ONNX to RKNN", variant="primary")

            with gr.Column():
                onnx_status = gr.Textbox(label="Status", lines=5)
                onnx_rknn_file = gr.File(label="RKNN Model")

        onnx_quantize.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[onnx_quantize],
            outputs=[onnx_dataset],
        )

        def run_onnx_to_rknn(onnx_input, platform, quantize, dataset, output_filename):
            try:
                if not onnx_input:
                    return "Error: Please upload an ONNX model file", None

                try:
                    from yolo_demo.export import RKNNExporter
                except ImportError:
                    return (
                        "Error: rknn-toolkit2 is not installed.\n"
                        "Install with: pip install 'yolo-demo[rknn]'\n"
                        "Note: Only supports Linux x86_64 and Python 3.8-3.10",
                        None,
                    )

                if not output_filename:
                    model_name = Path(onnx_input.name).stem
                    output_filename = f"{model_name}.rknn"
                elif not output_filename.endswith(".rknn"):
                    output_filename += ".rknn"

                dataset_path = None
                if quantize:
                    if not dataset:
                        return "Error: Calibration dataset required for quantization", None
                    dataset_path = dataset.name

                exporter = RKNNExporter(
                    onnx_input.name,
                    target_platform=platform,
                )
                rknn_path = exporter.export(
                    output=output_filename,
                    quantize=quantize,
                    dataset=dataset_path,
                )

                logger.info(f"RKNN model exported to: {rknn_path}")
                return (
                    f"RKNN conversion successful!\n\nPlatform: {platform}\nOutput: {rknn_path}",
                    rknn_path,
                )

            except ImportError as e:
                return f"Error: {str(e)}", None
            except Exception as e:
                logger.error(f"RKNN conversion failed: {e}")
                return f"Conversion failed: {str(e)}", None

        onnx_convert_btn.click(
            fn=run_onnx_to_rknn,
            inputs=[onnx_file, onnx_platform, onnx_quantize, onnx_dataset, onnx_output],
            outputs=[onnx_status, onnx_rknn_file],
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
