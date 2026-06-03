"""Dataset conversion tool (COCO/VOC to YOLO format)."""

import logging
import tempfile
from pathlib import Path

import gradio as gr

try:
    from . import coco2yolo
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
    import coco2yolo  # type: ignore

logger = logging.getLogger(__name__)


def create_dataset_converter_tab() -> gr.Tab:
    """Create the dataset converter tab."""
    with gr.Tab("Dataset Converter", id="converter") as tab:
        gr.Markdown("### Convert COCO/VOC to YOLO Format")
        gr.Markdown(
            "Convert your existing COCO or VOC format datasets to YOLO format for training."
        )

        with gr.Row():
            with gr.Column():
                format_type = gr.Radio(
                    choices=["COCO", "VOC"],
                    value="COCO",
                    label="Input Format",
                )

                # COCO input
                coco_input = gr.File(
                    label="COCO Annotations JSON",
                    file_types=[".json"],
                    visible=True,
                )

                # VOC input
                voc_input = gr.File(
                    label="VOCdevkit Directory (zip or folder)",
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
                dataset_yaml = gr.File(label="Dataset YAML")
                dataset_preview = gr.JSON(label="Dataset Info")

        def toggle_format(format_val):
            """Toggle input fields based on format."""
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

        def convert_dataset(
            format_type,
            coco_file,
            voc_file,
            voc_split,
            copy_images,
            output_name,
        ):
            """Convert dataset to YOLO format."""
            try:
                output_dir = Path(tempfile.mkdtemp()) / output_name

                if format_type == "COCO":
                    if not coco_file:
                        return "Error: Please upload COCO annotations JSON", None, None

                    yaml_path = coco2yolo.convert_coco_to_yolo(
                        coco_json_path=coco_file.name,
                        output_dir=str(output_dir),
                        copy_images=copy_images,
                    )

                else:  # VOC
                    if not voc_file:
                        return "Error: Please upload VOCdevkit directory", None, None

                    # For VOC, we need to extract the zip first
                    if voc_file.name.endswith(".zip"):
                        import zipfile

                        extract_dir = Path(tempfile.mkdtemp())
                        with zipfile.ZipFile(voc_file.name) as zf:
                            zf.extractall(extract_dir)
                        # Find the VOCdevkit directory
                        voc_dir = extract_dir / "VOCdevkit"
                        if not voc_dir.exists():
                            # Try to find any directory containing VOC*
                            for d in extract_dir.iterdir():
                                if d.is_dir() and "VOC" in d.name:
                                    voc_dir = d
                                    break
                    else:
                        voc_dir = Path(voc_file.name)

                    yaml_path = coco2yolo.convert_voc_to_yolo(
                        voc_devkit_dir=str(voc_dir),
                        output_dir=str(output_dir),
                        copy_images=copy_images,
                        split=voc_split,
                    )

                # Load and preview dataset YAML
                import yaml

                with open(yaml_path) as f:
                    dataset_info = yaml.safe_load(f)

                # Count images and labels
                images_dir = output_dir / "images"
                labels_dir = output_dir / "labels"
                num_images = len(list(images_dir.glob("**/*.*"))) if images_dir.exists() else 0
                num_labels = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

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

                logger.info(f"Dataset converted: {yaml_path}")
                return status, yaml_path, dataset_info

            except Exception as e:
                logger.error(f"Conversion failed: {e}")
                return f"Conversion failed:\n{str(e)}", None, None

        format_type.change(
            fn=toggle_format,
            inputs=[format_type],
            outputs=[coco_input, voc_input, voc_split],
        )

        convert_btn.click(
            fn=convert_dataset,
            inputs=[
                format_type,
                coco_input,
                voc_input,
                voc_split,
                copy_images,
                output_name,
            ],
            outputs=[conversion_status, dataset_yaml, dataset_preview],
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
