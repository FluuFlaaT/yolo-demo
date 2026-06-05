"""Gradio WebUI for YOLO Demo."""

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Generator, Optional

import gradio as gr  # noqa: E402

from ..export import RKNN_SUPPORTED_PLATFORMS  # noqa: E402
from ..inference import (  # noqa: E402
    Detection,
    create_engine,
    get_available_backend,
)
from .dataset_converter import create_dataset_converter_tab  # noqa: E402
from .services import (
    TrainingJobConfig,
    TrainingSessionManager,
    check_rknn_availability,
    export_onnx_to_rknn,
    export_pt_to_rknn,
    format_detections,
    resolve_model_path,
)


# ── Compatibility patch ──────────────────────────────────────────────────────
# Apply gradio_client compatibility patch for Pydantic v2
def _patch_gradio_client():
    """Patch gradio_client to handle Pydantic v2 schemas with boolean additionalProperties."""
    try:
        import gradio_client.utils as utils

        original = utils._json_schema_to_python_type

        def patched_json_schema_to_python_type(schema, defs=None):
            if not isinstance(schema, dict):
                if schema is True:
                    return "Any"
                elif schema is False:
                    return "None"
                return str(schema)
            return original(schema, defs)

        utils._json_schema_to_python_type = patched_json_schema_to_python_type
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Failed to patch gradio_client for Pydantic v2: %s", e
        )

_patch_gradio_client()

logger = logging.getLogger(__name__)


def draw_detections(image, detections: list[Detection]) -> Any:
    """Draw detection boxes on image."""
    try:
        import cv2
    except ImportError:
        return image

    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name}: {det.confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


def create_inference_tab() -> gr.Tab:
    """Create the inference tab using InferenceService."""
    backend = get_available_backend()

    with gr.Tab("Inference", id="inference") as tab:
        gr.Markdown("### Upload an image for object detection")
        gr.Markdown(f"Current backend: **{backend}**")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="numpy")
                model_name = gr.Textbox(
                    label="Model Name",
                    placeholder="e.g., yolov8n.pt",
                    value="yolov8n.pt",
                )
                custom_model = gr.File(label="Or Upload Custom Model (.pt)")
                conf_threshold = gr.Slider(
                    minimum=0.01, maximum=1.0, value=0.25, step=0.01,
                    label="Confidence Threshold",
                )
                detect_btn = gr.Button("Detect Objects", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Detection Result")
                detection_info = gr.JSON(label="Detections")

        def _on_inference(image, model_name_val, custom_file, conf):
            if image is None:
                return None, {"error": "No image provided"}

            custom_path = custom_file.name if custom_file is not None else None

            try:
                model_path = resolve_model_path(model_name_val, custom_path)
            except ValueError as e:
                return None, {"error": str(e)}

            logger.info("Running inference with model: %s", model_path)

            try:
                engine = create_engine(model_path)
                engine.load_model()
                result = engine.predict(image)
                output_img = draw_detections(image, result.detections)
                info = format_detections(result, conf)
                return output_img, info
            except Exception as e:
                logger.error("Inference failed: %s", e)
                return None, {"error": str(e)}

        detect_btn.click(
            fn=_on_inference,
            inputs=[input_image, model_name, custom_model, conf_threshold],
            outputs=[output_image, detection_info],
        )

    return tab


def create_training_tab() -> gr.Tab:
    """Create the training tab using TrainingService."""
    with gr.Tab("Training", id="training") as tab:
        gr.Markdown("### Incremental Training")
        gr.Markdown("Upload a dataset in YOLO format and train a custom model")

        with gr.Row():
            with gr.Column():
                dataset_yaml = gr.File(label="Dataset YAML (.yaml)")
                base_model_name = gr.Textbox(
                    label="Base Model Name",
                    placeholder="e.g., yolov8n.pt",
                    value="yolov8n.pt",
                )
                base_model_file = gr.File(label="Or Upload Base Model (.pt)")

                with gr.Accordion("Training Parameters", open=False):
                    device_preset = gr.Dropdown(
                        choices=[
                            "Custom", "RK3588", "Jetson Nano", "Jetson Xavier NX",
                            "Jetson Orin", "Desktop GPU", "CPU (slow)",
                        ],
                        value="Custom",
                        label="Edge Device Preset",
                        info="Auto-fills parameters for common edge devices",
                    )
                    epochs = gr.Slider(10, 500, value=100, step=10, label="Epochs")
                    batch_size = gr.Slider(2, 64, value=16, step=2, label="Batch Size")
                    imgsz = gr.Slider(320, 1280, value=640, step=32, label="Image Size")
                    lr0 = gr.Number(value=0.01, label="Initial Learning Rate")
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        placeholder="Leave empty for default (runs/detect/train)",
                        value="",
                    )
                    zip_all = gr.Checkbox(
                        value=False,
                        label="Download ALL files in zip, "
                        "including best.pt, last.pt, results.csv, "
                        "results.png, confusion matrix, labels, and args.yaml",
                    )

                with gr.Row():
                    train_btn = gr.Button("Start Training", variant="primary")
                    stop_btn = gr.Button("Stop Training", variant="stop")

            with gr.Column():
                training_status = gr.Textbox(
                    label="Status", lines=20, max_lines=40,
                    placeholder="Training log will stream here...",
                )
                training_output = gr.File(label="Trained Model")

        session_mgr = TrainingSessionManager(max_concurrent=1)
        active_job_id: Optional[str] = None

        presets = {
            "RK3588":          dict(epochs=100, batch=8,  imgsz=640, lr0=0.01),
            "Jetson Nano":     dict(epochs=50,  batch=4,  imgsz=416, lr0=0.01),
            "Jetson Xavier NX":dict(epochs=100, batch=8,  imgsz=640, lr0=0.01),
            "Jetson Orin":     dict(epochs=100, batch=16, imgsz=640, lr0=0.01),
            "Desktop GPU":     dict(epochs=200, batch=16, imgsz=640, lr0=0.01),
            "CPU (slow)":      dict(epochs=50,  batch=2,  imgsz=320, lr0=0.01),
        }

        def _apply_preset(preset_name):
            if preset_name == "Custom":
                return [gr.update()] * 4
            p = presets.get(preset_name, {})
            return [
                gr.update(value=p.get("epochs", 100)),
                gr.update(value=p.get("batch", 16)),
                gr.update(value=p.get("imgsz", 640)),
                gr.update(value=p.get("lr0", 0.01)),
            ]

        device_preset.change(
            fn=_apply_preset,
            inputs=[device_preset],
            outputs=[epochs, batch_size, imgsz, lr0],
        )

        def _on_training(
            dataset_file, model_name_val, model_file,
            epochs_val, batch_val, imgsz_val, lr0_val, output_val,
            zip_all_val,
        ) -> Generator:
            nonlocal active_job_id

            if dataset_file is None:
                yield "Error: Dataset YAML is required", None
                return

            custom_path = model_file.name if model_file is not None else None
            try:
                model_path = resolve_model_path(model_name_val, custom_path)
            except ValueError as e:
                yield str(e), None
                return

            config = TrainingJobConfig(
                model_path=model_path,
                dataset_path=dataset_file.name,
                epochs=int(epochs_val),
                batch_size=int(batch_val),
                imgsz=int(imgsz_val),
                lr0=float(lr0_val),
                output_dir=output_val or None,
            )

            job_id = session_mgr.submit(config)
            active_job_id = job_id
            session = session_mgr.get(job_id)
            assert session is not None

            log_lines = []

            try:
                while session.is_active() or session.has_messages():
                    msg = session.poll(timeout=0.5)
                    if msg is None:
                        continue

                    if isinstance(msg, str):
                        log_lines.append(msg)
                        yield "".join(log_lines), None
                    elif isinstance(msg, tuple):
                        kind, payload = msg
                        if kind == "done":
                            final_path = payload.strip()
                            if zip_all_val:
                                serve_path = _zip_training_output(final_path)
                            else:
                                serve_path = _copy_model_to_temp(final_path)
                            yield "".join(log_lines), serve_path
                            return
                        else:
                            logger.error("Training failed: %s", payload)
                            yield "".join(log_lines), None
                            return
            finally:
                session_mgr.on_job_finished(job_id)
                if active_job_id == job_id:
                    active_job_id = None

            yield "".join(log_lines), None

        def _on_stop():
            nonlocal active_job_id
            if active_job_id:
                session_mgr.cancel(active_job_id)

        train_btn.click(
            fn=_on_training,
            inputs=[dataset_yaml, base_model_name, base_model_file,
                    epochs, batch_size, imgsz, lr0, output_dir, zip_all],
            outputs=[training_status, training_output],
        )
        stop_btn.click(fn=_on_stop, inputs=[], outputs=[])

    return tab


def _copy_model_to_temp(model_path: str) -> Optional[str]:
    """Copy trained model to a temp directory for Gradio file serving."""
    src = Path(model_path)
    if not src.exists():
        logger.warning("Expected model not found at: %s", model_path)
        return None

    dst = Path(tempfile.gettempdir()) / "yolo_trained_best.pt"
    try:
        shutil.copy2(src, dst)
        logger.info("Model copied to temp for download: %s", dst)
        return str(dst)
    except Exception as e:
        logger.warning("Could not copy model: %s", e)
        return None


def _zip_training_output(model_path: str) -> Optional[str]:
    """Zip the entire training run directory for download.

    The model_path is expected to be `<save_dir>/weights/best.pt`.
    The parent of `weights/` is the training run directory containing all
    Ultralytics auto-generated artifacts (weights, results, plots, etc.).
    """
    best_pt = Path(model_path)
    if not best_pt.exists():
        logger.warning("Expected model not found at: %s", model_path)
        return None

    run_dir = best_pt.parent.parent  # <save_dir>/weights/ → <save_dir>/
    if not run_dir.is_dir():
        logger.warning("Training run directory not found: %s", run_dir)
        return None

    zip_dst = Path(tempfile.gettempdir()) / f"{run_dir.name}.zip"
    try:
        with zipfile.ZipFile(zip_dst, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(run_dir.rglob("*")):
                if file_path.is_file() and not file_path.name.startswith("."):
                    arcname = file_path.relative_to(run_dir.parent)
                    zf.write(file_path, arcname)
        logger.info("Training output zipped to temp for download: %s", zip_dst)
        return str(zip_dst)
    except Exception as e:
        logger.warning("Could not zip training output: %s", e)
        return None


def create_export_tab() -> gr.Tab:
    """Create the export tab using ExportService."""
    with gr.Tab("Export", id="export") as tab:
        gr.Markdown("### Export Model to RKNN")
        gr.Markdown("Export your YOLO model directly to RKNN format for Rockchip devices")

        with gr.Row():
            with gr.Column():
                pt_model_file = gr.File(label="Model (.pt file)", file_types=[".pt"])
                platform = gr.Dropdown(
                    choices=RKNN_SUPPORTED_PLATFORMS, value="rk3588",
                    label="Target Platform",
                )
                imgsz = gr.Slider(320, 1280, value=640, step=32, label="Image Size")
                export_btn = gr.Button("Export to RKNN", variant="primary")

            with gr.Column():
                export_status = gr.Textbox(label="Status", lines=5)
                exported_file = gr.File(label="RKNN Model")

        def _on_pt_export(model, plat, img_sz):
            if not model:
                return "Error: Please upload a model file", None
            try:
                return export_pt_to_rknn(model.name, plat, int(img_sz))
            except Exception as e:
                logger.error("RKNN export failed: %s", e)
                return f"Export failed: {str(e)}", None

        export_btn.click(
            fn=_on_pt_export,
            inputs=[pt_model_file, platform, imgsz],
            outputs=[export_status, exported_file],
        )

        gr.Markdown("---")
        gr.Markdown("### Convert ONNX to RKNN")
        gr.Markdown("Convert existing ONNX model to RKNN format (requires rknn-toolkit2)")

        with gr.Row():
            with gr.Column():
                onnx_file = gr.File(label="ONNX Model (.onnx)", file_types=[".onnx"])
                onnx_platform = gr.Dropdown(
                    choices=RKNN_SUPPORTED_PLATFORMS, value="rk3588",
                    label="Target Platform",
                )
                onnx_quantize = gr.Checkbox(
                    value=False, label="Enable INT8 Quantization",
                    info="Requires calibration dataset",
                )
                onnx_dataset = gr.File(
                    label="Calibration Dataset (txt file with image paths)",
                    file_types=[".txt"], visible=False,
                )
                onnx_output = gr.Textbox(
                    label="Output Filename (optional)",
                    placeholder="model.rknn", value="",
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

        def _on_onnx_export(onnx_input, plat, quantize, dataset, out_filename):
            if not onnx_input:
                return "Error: Please upload an ONNX model file", None

            # Pre-flight check
            available, msg = check_rknn_availability()
            if not available:
                return msg, None

            dataset_path = dataset.name if (dataset and quantize) else None
            try:
                return export_onnx_to_rknn(
                    onnx_input.name, plat,
                    quantize=quantize, dataset_path=dataset_path,
                    output_filename=out_filename or None,
                )
            except Exception as e:
                logger.error("RKNN conversion failed: %s", e)
                return f"Conversion failed: {str(e)}", None

        onnx_convert_btn.click(
            fn=_on_onnx_export,
            inputs=[onnx_file, onnx_platform, onnx_quantize, onnx_dataset, onnx_output],
            outputs=[onnx_status, onnx_rknn_file],
        )

    return tab



def create_webui() -> gr.Blocks:
    """Create the complete Gradio WebUI with health check endpoint."""
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

    # Mount a /health endpoint on the underlying FastAPI app
    _mount_health_endpoint(app)

    return app


def _mount_health_endpoint(blocks: gr.Blocks) -> None:
    """Mount a /health endpoint on Gradio's internal FastAPI-like app."""
    @blocks.app.get("/health")
    async def health():
        return {"status": "healthy", "service": "yolo-demo-webui"}


def launch(host: str = "0.0.0.0", port: int = 7860, **kwargs: Any) -> None:
    """Launch the WebUI with production-grade configuration.

    Environment variables:
        LOG_FORMAT:         Set to "json" for structured JSON-line logging.
        YOLO_ALLOWED_PATHS: Comma-separated paths Gradio can serve files from.
                            Defaults to "/tmp/yolo_outputs".
    """
    from ..utils.logging import setup_logging

    setup_logging(level=logging.INFO)

    app = create_webui()
    logger.info("Launching WebUI at http://%s:%s", host, port)

    allowed = os.environ.get("YOLO_ALLOWED_PATHS", "/tmp/yolo_outputs")
    kwargs.setdefault("allowed_paths", [p.strip() for p in allowed.split(",") if p.strip()])

    app.launch(server_name=host, server_port=port, **kwargs)


if __name__ == "__main__":
    launch()
