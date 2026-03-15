"""RKNN Validation tab for the Gradio WebUI.

Runs validate.sh inside a Docker container (linux/amd64) and streams
the output back to the browser in real time.

Also provides a host-side PT vs ONNX comparison (no Docker required).
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import gradio as gr

logger = logging.getLogger(__name__)

# Project root: src/yolo_demo/ui/ → src/yolo_demo/ → src/ → project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_VALIDATE_SH = _PROJECT_ROOT / "docker" / "validate.sh"


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes so the log renders cleanly in Gradio."""
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def _check_docker() -> str:
    """Return empty string if Docker is reachable, else an error message."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return "Docker daemon is not running. Please start Docker and try again."
        return ""
    except FileNotFoundError:
        return "Docker is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return "Docker info timed out. Is Docker running?"


def _run_validation(
    onnx_file: object,
    image_file: object,
    conf: float,
    iou: float,
) -> Generator[Tuple[str, str], None, None]:
    """
    Run validate.sh and stream stdout/stderr into the log textbox.

    Yields (log_text, status_text) tuples; Gradio streams each yield
    to the browser in real time.
    """
    # ── pre-flight checks ────────────────────────────────────────────────────
    if onnx_file is None:
        yield "", "ERROR: Please upload an ONNX model file."
        return

    onnx_path = Path(onnx_file.name).resolve()
    if not onnx_path.exists():
        yield "", f"ERROR: ONNX file not found: {onnx_path}"
        return

    if not _VALIDATE_SH.exists():
        yield "", f"ERROR: validate.sh not found at {_VALIDATE_SH}"
        return

    docker_err = _check_docker()
    if docker_err:
        yield "", docker_err
        return

    # ── build command ────────────────────────────────────────────────────────
    cmd = [
        "bash",
        str(_VALIDATE_SH),
        "--onnx",
        str(onnx_path),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
    ]

    if image_file is not None:
        image_path = Path(image_file.name).resolve()
        if image_path.exists():
            cmd += ["--image", str(image_path)]
        else:
            logger.warning(f"Uploaded image path not found: {image_path}, using synthetic.")

    logger.info("RKNN validation command: %s", " ".join(cmd))

    # ── stream subprocess output ─────────────────────────────────────────────
    log_lines: list = []
    status = "Running..."

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(_PROJECT_ROOT),
        )

        for raw_line in process.stdout:  # type: ignore[union-attr]
            clean = _strip_ansi(raw_line)
            log_lines.append(clean)
            yield "".join(log_lines), status

        process.wait()
        log_text = "".join(log_lines)

        if process.returncode == 0:
            status = "PASS"
        else:
            status = "FAIL"

        yield log_text, status

    except Exception as exc:
        log_lines.append(f"\nFailed to run validation: {exc}\n")
        yield "".join(log_lines), "ERROR"


# ---------------------------------------------------------------------------
# PT vs ONNX host-side comparison helpers
# ---------------------------------------------------------------------------

INPUT_SIZE = 640


def _letterbox(img: np.ndarray, new_shape: int = INPUT_SIZE):
    """Resize with padding to keep aspect ratio."""
    import cv2

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h = new_shape - nh
    pad_w = new_shape - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114)
    return img, r, (left, top)


def _synthetic_image(size: int) -> np.ndarray:
    """Create a simple gradient image with a white rectangle."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        img[i, :, 0] = int(255 * i / size)
        img[:, i, 1] = int(255 * i / size)
    img[100:300, 150:450, :] = [200, 220, 180]
    return img


def _preprocess_image(image_path: str | None) -> tuple[np.ndarray, np.ndarray, float, tuple]:
    """Return (chw_float32 tensor (1,3,H,W), original_bgr, ratio, pad)."""
    import cv2

    if image_path and Path(image_path).exists():
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
    else:
        bgr = _synthetic_image(INPUT_SIZE)

    original = bgr.copy()
    img, ratio, pad = _letterbox(bgr, INPUT_SIZE)
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis]
    return tensor, original, ratio, pad


def _run_pt_inference(pt_path: str, tensor: np.ndarray) -> np.ndarray:
    """Run PT model raw inference and return (1, nc+4, 8400) array."""
    import torch
    from ultralytics import YOLO

    model = YOLO(pt_path)
    model.model.eval()
    with torch.no_grad():
        t = torch.from_numpy(tensor)
        out = model.model.model(t)
    # out is a tuple; first element is the detection head output
    pred = out[0] if isinstance(out, (list, tuple)) else out
    return pred.cpu().numpy()


def _run_onnx_inference(onnx_path: str, tensor: np.ndarray) -> np.ndarray:
    """Run ONNX Runtime inference and return raw output."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess.run([output_name], {input_name: tensor})[0]


def _compare_outputs(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute cosine similarity and mean relative error between two flat arrays."""
    fa = a.astype(np.float32).flatten()
    fb = b.astype(np.float32).flatten()

    if fa.shape != fb.shape:
        min_len = min(len(fa), len(fb))
        fa, fb = fa[:min_len], fb[:min_len]

    abs_err = np.abs(fa - fb)
    denom = np.maximum(np.abs(fa), np.abs(fb))
    denom = np.where(denom < 1e-6, 1e-6, denom)
    rel_err = abs_err / denom

    norm_a, norm_b = np.linalg.norm(fa), np.linalg.norm(fb)
    if norm_a > 0 and norm_b > 0:
        cos_sim = float(np.dot(fa, fb) / (norm_a * norm_b))
    else:
        cos_sim = float("nan")

    return {
        "max_abs_err": float(np.max(abs_err)),
        "mean_abs_err": float(np.mean(abs_err)),
        "max_rel_err": float(np.max(rel_err)),
        "mean_rel_err": float(np.mean(rel_err)),
        "cos_sim": cos_sim,
    }


def _postprocess(
    raw: np.ndarray,
    orig_h: int,
    orig_w: int,
    ratio: float,
    pad: tuple,
    conf_thr: float,
    iou_thr: float,
) -> list[dict]:
    """Decode YOLOv8 output (1, nc+4, 8400) and apply NMS."""
    import cv2

    pred = raw[0]  # (nc+4, 8400) or (8400, nc+4)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # → (8400, nc+4)

    boxes_raw = pred[:, :4]
    scores_raw = pred[:, 4:]
    max_scores = scores_raw.max(axis=1)
    class_ids = scores_raw.argmax(axis=1)

    mask = max_scores > conf_thr
    if not mask.any():
        return []

    boxes_filt = boxes_raw[mask]
    scores_filt = max_scores[mask]
    class_ids_filt = class_ids[mask]

    cx, cy, bw, bh = boxes_filt.T
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    pad_left, pad_top = pad
    xyxy[:, [0, 2]] -= pad_left
    xyxy[:, [1, 3]] -= pad_top
    xyxy /= ratio
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)

    nms_indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores_filt.tolist(), conf_thr, iou_thr)

    detections = []
    if len(nms_indices) > 0:
        for i in np.array(nms_indices).flatten():
            x1_, y1_, x2_, y2_ = xyxy[i]
            detections.append(
                {
                    "class_id": int(class_ids_filt[i]),
                    "confidence": float(scores_filt[i]),
                    "bbox": [float(x1_), float(y1_), float(x2_), float(y2_)],
                }
            )
    return detections


def _run_pt_onnx_comparison(
    pt_file: object,
    onnx_file: object,
    image_file: object,
    conf: float,
    iou: float,
    mean_rel_tol: float,
    cos_sim_thr: float,
) -> tuple[str, str]:
    """
    Compare PT and ONNX model outputs on the host (no Docker).

    Returns (log_text, status).
    """
    lines = []

    def log(msg: str):
        lines.append(msg + "\n")
        logger.info(msg)

    try:
        if pt_file is None:
            return "ERROR: Please upload a PT model file.", "ERROR"
        if onnx_file is None:
            return "ERROR: Please upload an ONNX model file.", "ERROR"

        pt_path = Path(pt_file.name).resolve()
        onnx_path = Path(onnx_file.name).resolve()

        if not pt_path.exists():
            return f"ERROR: PT file not found: {pt_path}", "ERROR"
        if not onnx_path.exists():
            return f"ERROR: ONNX file not found: {onnx_path}", "ERROR"

        image_path = None
        if image_file is not None:
            p = Path(image_file.name).resolve()
            if p.exists():
                image_path = str(p)

        log("=== PT vs ONNX Host-Side Comparison ===")
        log(f"PT   : {pt_path}")
        log(f"ONNX : {onnx_path}")
        log(f"Image: {image_path or '(synthetic)'}")

        # 1. Pre-process
        log("\n--- Pre-processing ---")
        tensor, original, ratio, pad = _preprocess_image(image_path)
        orig_h, orig_w = original.shape[:2]
        log(f"Input tensor shape: {tensor.shape}  dtype: {tensor.dtype}")

        # 2. PT inference
        log("\n--- PT inference ---")
        pt_out = _run_pt_inference(str(pt_path), tensor)
        log(f"PT output shape: {pt_out.shape}  dtype: {pt_out.dtype}")

        # 3. ONNX inference
        log("\n--- ONNX inference ---")
        onnx_out = _run_onnx_inference(str(onnx_path), tensor)
        log(f"ONNX output shape: {onnx_out.shape}  dtype: {onnx_out.dtype}")

        # 4. Numerical comparison
        log("\n--- Numerical comparison ---")
        metrics = _compare_outputs(pt_out, onnx_out)
        log(f"  Max absolute error : {metrics['max_abs_err']:.6f}")
        log(f"  Mean absolute error: {metrics['mean_abs_err']:.6f}")
        log(f"  Max relative error : {metrics['max_rel_err']:.4%}")
        log(f"  Mean relative error: {metrics['mean_rel_err']:.4%}")
        cos_str = f"{metrics['cos_sim']:.6f}" if not np.isnan(metrics["cos_sim"]) else "nan"
        log(f"  Cosine similarity  : {cos_str}")

        passed = metrics["mean_rel_err"] < mean_rel_tol and (
            np.isnan(metrics["cos_sim"]) or metrics["cos_sim"] > cos_sim_thr
        )
        status = "PASS" if passed else "FAIL"
        log(
            f"  Result             : {status}"
            f"  (mean_rel_err<{mean_rel_tol:.1%}, cos_sim>{cos_sim_thr})"
        )

        # 5. Detections — PT
        log("\n--- Detections (PT) ---")
        pt_dets = _postprocess(pt_out, orig_h, orig_w, ratio, pad, conf, iou)
        if pt_dets:
            for i, d in enumerate(pt_dets):
                x1, y1, x2, y2 = d["bbox"]
                log(
                    f"  [{i}] class={d['class_id']}  conf={d['confidence']:.3f}"
                    f"  box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]"
                )
        else:
            log("  No detections found.")

        # 6. Detections — ONNX
        log("\n--- Detections (ONNX) ---")
        onnx_dets = _postprocess(onnx_out, orig_h, orig_w, ratio, pad, conf, iou)
        if onnx_dets:
            for i, d in enumerate(onnx_dets):
                x1, y1, x2, y2 = d["bbox"]
                log(
                    f"  [{i}] class={d['class_id']}  conf={d['confidence']:.3f}"
                    f"  box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]"
                )
        else:
            log("  No detections found.")

        # 7. Summary
        log("\n=== Summary ===")
        log(f"  PT detections  : {len(pt_dets)}")
        log(f"  ONNX detections: {len(onnx_dets)}")
        log(f"  Numerical check: {status}")

        return "".join(lines), status

    except Exception as exc:
        logger.exception("PT vs ONNX comparison failed")
        return f"ERROR: {exc}\n", "ERROR"


# ---------------------------------------------------------------------------
# Gradio tab
# ---------------------------------------------------------------------------


def create_rknn_validation_tab() -> gr.Tab:
    """Create and return the RKNN Validation Gradio tab."""
    with gr.Tab("RKNN Validation", id="rknn_validation") as tab:
        # ── Section 1: Docker-based RKNN validation ──────────────────────────
        gr.Markdown("### RKNN Model Validation (CPU Simulation via Docker)")
        gr.Markdown(
            "Validates that the RKNN conversion pipeline produces output numerically "
            "consistent with the original ONNX model.  \n"
            "Runs inside a Docker container (`linux/amd64`) using **rknn-toolkit2** "
            "CPU simulation — no physical RK3588 required.  \n"
            "**Prerequisites:** Docker must be running on this machine."
        )

        with gr.Row():
            # ── inputs ───────────────────────────────────────────────────────
            with gr.Column(scale=1):
                onnx_file = gr.File(
                    label="ONNX Model (required)",
                    file_types=[".onnx"],
                )
                image_file = gr.File(
                    label="Test Image — optional (synthetic pattern used if omitted)",
                    file_types=[".jpg", ".jpeg", ".png"],
                )

                with gr.Accordion("Detection Thresholds", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.25,
                        step=0.01,
                        label="Confidence Threshold",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.45,
                        step=0.01,
                        label="IoU Threshold (NMS)",
                    )

                run_btn = gr.Button("Run RKNN Validation", variant="primary")

            # ── outputs ──────────────────────────────────────────────────────
            with gr.Column(scale=2):
                status_box = gr.Textbox(
                    label="Result",
                    value="",
                    interactive=False,
                    lines=1,
                    placeholder="PASS / FAIL / ERROR will appear here",
                )
                log_box = gr.Textbox(
                    label="Validation Log (streaming)",
                    value="",
                    interactive=False,
                    lines=28,
                    max_lines=60,
                    buttons=["copy"],
                    placeholder=(
                        "Log output will stream here while validation runs.\n"
                        "Docker image build + model compilation takes ~60 s."
                    ),
                )

        run_btn.click(
            fn=_run_validation,
            inputs=[onnx_file, image_file, conf_slider, iou_slider],
            outputs=[log_box, status_box],
        )

        gr.Markdown("---")

        # ── Section 2: Host-side PT vs ONNX comparison ───────────────────────
        gr.Markdown("### PT vs ONNX Comparison (Host-Side, No Docker)")
        gr.Markdown(
            "Compares raw model outputs between a `.pt` (PyTorch) and `.onnx` model "
            "on the same input.  \n"
            "Runs entirely on this machine using **Ultralytics** and **onnxruntime** — "
            "no Docker required."
        )

        with gr.Row():
            with gr.Column(scale=1):
                cmp_pt_file = gr.File(
                    label="PT Model (required)",
                )
                cmp_onnx_file = gr.File(
                    label="ONNX Model (required)",
                    file_types=[".onnx"],
                )
                cmp_image_file = gr.File(
                    label="Test Image — optional (synthetic pattern used if omitted)",
                    file_types=[".jpg", ".jpeg", ".png"],
                )

                with gr.Accordion("Detection Thresholds", open=False):
                    cmp_conf_slider = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.25,
                        step=0.01,
                        label="Confidence Threshold",
                    )
                    cmp_iou_slider = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.45,
                        step=0.01,
                        label="IoU Threshold (NMS)",
                    )

                with gr.Accordion("Tolerance Settings", open=False):
                    mean_rel_tol_slider = gr.Slider(
                        minimum=0.001,
                        maximum=0.20,
                        value=0.02,
                        step=0.001,
                        label="Mean Relative Error Tolerance",
                        info="Default: 2% (same as RKNN validation)",
                    )
                    cos_sim_thr_slider = gr.Slider(
                        minimum=0.90,
                        maximum=1.0,
                        value=0.99,
                        step=0.001,
                        label="Cosine Similarity Threshold",
                        info="Default: 0.99 (same as RKNN validation)",
                    )

                cmp_btn = gr.Button("Run PT vs ONNX Comparison", variant="primary")

            with gr.Column(scale=2):
                cmp_status_box = gr.Textbox(
                    label="Result",
                    value="",
                    interactive=False,
                    lines=1,
                    placeholder="PASS / FAIL / ERROR will appear here",
                )
                cmp_log_box = gr.Textbox(
                    label="Comparison Log",
                    value="",
                    interactive=False,
                    lines=28,
                    max_lines=60,
                    buttons=["copy"],
                    placeholder="Comparison output will appear here.",
                )

        cmp_btn.click(
            fn=_run_pt_onnx_comparison,
            inputs=[
                cmp_pt_file,
                cmp_onnx_file,
                cmp_image_file,
                cmp_conf_slider,
                cmp_iou_slider,
                mean_rel_tol_slider,
                cos_sim_thr_slider,
            ],
            outputs=[cmp_log_box, cmp_status_box],
        )

    return tab
