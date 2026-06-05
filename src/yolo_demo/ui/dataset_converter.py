"""Dataset conversion tool (COCO/VOC to YOLO format)."""

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import gradio as gr

from .coco2yolo import convert_coco_to_yolo, convert_voc_to_yolo
from .services.dataset_service import extract_voc_zip

logger = logging.getLogger(__name__)


def _extract_coco_zip(zip_path: str) -> str:
    """Extract a COCO zip and return the directory containing the JSON + images.

    Looks for any .json file and an images/ directory in the archive.
    Supports zips that put everything at the root or in a single subdirectory.
    """
    extract_dir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    for f in extract_dir.rglob("*.json"):
        return str(f)

    top_level = sorted(p.name for p in extract_dir.iterdir())
    raise ValueError(
        "No .json file found in COCO zip archive.\n"
        f"Top-level contents: {top_level}\n"
        "Expected: an annotations JSON file (e.g. instances_train2017.json) "
        "and optionally an images/ directory."
    )


def _resolve_voc_root(zip_or_dir_path: str) -> str:
    """Resolve the VOC dataset root from a zip file or local directory.

    If it's a zip, extract it. Then find the directory containing
    Annotations/ and JPEGImages/ (handles VOCdevkit, year subdirs, flat).
    """
    path = Path(zip_or_dir_path)
    if path.suffix.lower() == ".zip":
        return extract_voc_zip(zip_or_dir_path)

    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")

    return str(path)


def create_dataset_converter_tab() -> gr.Tab:
    """Create the dataset converter tab."""
    with gr.Tab("Dataset Converter", id="converter") as tab:
        gr.Markdown("### Convert COCO/VOC to YOLO Format")
        gr.Markdown(
            "Upload a COCO or VOC dataset (zip archive or JSON file) to "
            "convert it to YOLO format for training. Images are copied along "
            "with labels."
        )

        with gr.Row():
            with gr.Column():
                format_type = gr.Radio(
                    choices=["COCO", "VOC"],
                    value="COCO",
                    label="Input Format",
                )

                coco_input = gr.File(
                    label="COCO Dataset (zip or JSON)",
                    file_types=[".zip", ".json", ".jsonl"],
                    visible=True,
                )
                voc_input = gr.File(
                    label="VOC Dataset (zip or directory)",
                    file_types=[".zip"],
                    visible=False,
                )
                voc_split = gr.Dropdown(
                    choices=["train", "val", "test", "trainval"],
                    value="trainval",
                    label="Dataset Split",
                    visible=False,
                )

                copy_images = gr.Checkbox(
                    value=True,
                    label="Copy Images to Output",
                )
                output_name = gr.Textbox(
                    label="Output Dataset Name",
                    placeholder="my-dataset",
                    value="converted-dataset",
                )
                convert_btn = gr.Button("Convert Dataset", variant="primary")

            with gr.Column():
                conversion_status = gr.Textbox(label="Status", lines=5)
                dataset_zip = gr.File(label="Converted Dataset (zip)")
                dataset_preview = gr.JSON(label="Dataset Info")

        def toggle_format(format_val):
            if format_val == "COCO":
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )

        def _on_convert(fmt, coco_file, voc_file, vsplit, copy_imgs, out_name):
            try:
                output_dir = Path(tempfile.mkdtemp()) / out_name
                output_dir.mkdir(parents=True, exist_ok=True)

                if fmt == "COCO":
                    if not coco_file:
                        return "Error: Please upload a COCO zip or JSON file", None, None

                    src = coco_file.name
                    if src.lower().endswith(".zip"):
                        json_path = _extract_coco_zip(src)
                    else:
                        json_path = src

                    yaml_path = convert_coco_to_yolo(
                        coco_json_path=json_path,
                        output_dir=str(output_dir),
                        copy_images=copy_imgs,
                    )
                else:
                    if not voc_file:
                        return "Error: Please upload a VOC zip or directory", None, None

                    voc_root = _resolve_voc_root(voc_file.name)
                    yaml_path = convert_voc_to_yolo(
                        voc_devkit_dir=voc_root,
                        output_dir=str(output_dir),
                        copy_images=copy_imgs,
                        split=vsplit,
                    )

                import yaml

                with open(yaml_path) as f:
                    info = yaml.safe_load(f)

                images_dir = output_dir / "images"
                labels_dir = output_dir / "labels"
                n_images = len(list(images_dir.glob("**/*.*"))) if images_dir.exists() else 0
                n_labels = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

                info["_stats"] = {
                    "num_images": n_images,
                    "num_labels": n_labels,
                    "num_classes": info.get("nc", 0),
                }

                # Zip the entire output directory for download
                zip_path = str(output_dir) + ".zip"
                shutil.make_archive(
                    str(output_dir), "zip", output_dir.parent, out_name
                )

                status = "Conversion successful!\n\n"
                status += f"Classes: {info.get('nc', 0)}\n"
                status += f"Images: {n_images}\n"
                status += f"Labels: {n_labels}\n"
                status += "\nDownload the zip and extract it to use with Training."
                if n_images == 0:
                    status += (
                        "\n\n⚠ No images were copied. "
                        "Check that image paths in your source dataset are accessible."
                    )

                logger.info("Dataset converted: %s → %s", yaml_path, zip_path)
                return status, zip_path, info

            except Exception as e:
                logger.error("Conversion failed: %s", e)
                return f"Conversion failed:\n{str(e)}", None, None

        format_type.change(
            fn=toggle_format,
            inputs=[format_type],
            outputs=[coco_input, voc_input, voc_split],
        )
        convert_btn.click(
            fn=_on_convert,
            inputs=[format_type, coco_input, voc_input, voc_split, copy_images, output_name],
            outputs=[conversion_status, dataset_zip, dataset_preview],
        )

    return tab


def create_converter_ui() -> gr.Blocks:
    """Create standalone dataset converter UI."""
    with gr.Blocks(title="Dataset Converter - YOLO Demo") as app:
        gr.Markdown(
            """
            # Dataset Converter
            Convert COCO or VOC format datasets to YOLO format.
            """
        )
        create_dataset_converter_tab()
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_converter_ui()
    app.launch(server_name="0.0.0.0", server_port=7861)
