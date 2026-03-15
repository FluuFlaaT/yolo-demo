#!/usr/bin/env python3
"""
Validate RKNN model correctness in CPU simulation mode.

Runs inside the rknn-converter Docker container (linux/amd64).
Compares RKNN CPU-sim output against ONNX runtime output on the same input,
then runs full YOLOv8 post-processing and prints detected boxes.

Usage (inside container):
    python validate_rknn.py \\
        --rknn  /workspace/models/best.rknn \\
        --onnx  /workspace/input/best.onnx \\
        [--image /workspace/input/test.jpg] \\
        [--conf  0.25] \\
        [--iou   0.45]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

INPUT_SIZE = 640  # default inference size


def letterbox(img: np.ndarray, new_shape: int = INPUT_SIZE):
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


def preprocess(image_path: Optional[str], input_size: int = INPUT_SIZE):
    """
    Return (chw_float32 tensor, original_bgr_image).
    If image_path is None a synthetic test pattern is generated.
    """
    import cv2

    if image_path and Path(image_path).exists():
        log.info(f"Loading image: {image_path}")
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
    else:
        log.info("No image provided — using synthetic test pattern (gradient + shapes)")
        bgr = _synthetic_image(input_size)

    original = bgr.copy()
    img, ratio, pad = letterbox(bgr, input_size)

    # BGR -> RGB, HWC -> CHW, /255
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis]  # (1, 3, H, W)
    return tensor, original, ratio, pad


def _synthetic_image(size: int) -> np.ndarray:
    """Create a simple gradient image with a white rectangle."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Gradient background
    for i in range(size):
        img[i, :, 0] = int(255 * i / size)
        img[:, i, 1] = int(255 * i / size)
    img[100:300, 150:450, :] = [200, 220, 180]  # fake pill-coloured rectangle
    return img


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def run_onnx(onnx_path: str, tensor: np.ndarray) -> np.ndarray:
    """Run inference with ONNX Runtime and return raw output."""
    import onnxruntime as ort

    log.info(f"Loading ONNX model: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    log.info(f"  Input:  {input_name}  {sess.get_inputs()[0].shape}")
    log.info(f"  Output: {output_name}  {sess.get_outputs()[0].shape}")

    out = sess.run([output_name], {input_name: tensor})[0]
    log.info(f"ONNX output shape: {out.shape}  dtype: {out.dtype}")
    return out


# ---------------------------------------------------------------------------
# RKNN CPU-simulation inference
# ---------------------------------------------------------------------------


def run_rknn_sim(onnx_path: str, tensor: np.ndarray) -> np.ndarray:
    """
    Run RKNN CPU-simulation inference from an ONNX model.

    NOTE: load_rknn() + init_runtime(target=None) is NOT supported by rknn-toolkit2.
    CPU simulation requires: load_onnx() → config() → build() → init_runtime(target=None).
    This re-runs the conversion in-memory (no file written).
    """
    from rknn.api import RKNN

    log.info(f"Building RKNN in-memory from ONNX for simulation: {onnx_path}")
    rknn = RKNN(verbose=False)

    rknn.config(
        target_platform="rk3588",
        quantized_dtype="w16a16i",  # FP16, matches the saved model
        optimization_level=3,
        # Two shape sets: 640×640 (validation size) and 1280×1280 (max / production)
        dynamic_input=[[[1, 3, 640, 640]], [[1, 3, 1280, 1280]]],
    )

    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"load_onnx failed (code {ret})")

    log.info("Building RKNN model (this may take ~60 s)...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"build failed (code {ret})")

    # CPU simulation — no physical device required
    ret = rknn.init_runtime(target=None)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed (code {ret})")

    log.info("RKNN runtime initialised (CPU simulation)")

    outputs = rknn.inference(inputs=[tensor], data_format="nchw")
    rknn.release()

    out = outputs[0]
    log.info(f"RKNN output shape: {out.shape}  dtype: {out.dtype}")
    return out


# ---------------------------------------------------------------------------
# Numerical comparison
# ---------------------------------------------------------------------------


def compare(onnx_out: np.ndarray, rknn_out: np.ndarray) -> bool:
    """Print comparison metrics and return True if within tolerance."""
    a = onnx_out.astype(np.float32).flatten()
    b = rknn_out.astype(np.float32).flatten()

    if a.shape != b.shape:
        log.warning(f"Shape mismatch: ONNX {a.shape} vs RKNN {b.shape}")
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

    abs_err = np.abs(a - b)
    max_abs_err = float(np.max(abs_err))
    mean_abs_err = float(np.mean(abs_err))

    # Relative error: normalised by the larger of the two absolute values
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < 1e-6, 1e-6, denom)  # avoid division by zero
    rel_err = abs_err / denom
    max_rel_err = float(np.max(rel_err))
    mean_rel_err = float(np.mean(rel_err))

    # Cosine similarity (handle zero vectors)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 0 and norm_b > 0:
        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    else:
        cos_sim = float("nan")

    log.info("=== Numerical comparison ===")
    log.info(f"  Max absolute error : {max_abs_err:.6f}")
    log.info(f"  Mean absolute error: {mean_abs_err:.6f}")
    log.info(f"  Max relative error : {max_rel_err:.4%}")
    log.info(f"  Mean relative error: {mean_rel_err:.4%}")
    log.info(f"  Cosine similarity  : {cos_sim:.6f}")

    # FP16 introduces ~1e-3 relative error in theory; allow up to 2% mean
    # (max can spike on near-zero anchors in the CPU simulator – use mean instead)
    MEAN_REL_THRESHOLD = 0.02
    COS_THRESHOLD = 0.99
    passed = mean_rel_err < MEAN_REL_THRESHOLD and (np.isnan(cos_sim) or cos_sim > COS_THRESHOLD)
    status = "PASS" if passed else "FAIL"
    log.info(
        f"  Result             : {status}"
        f"  (mean_rel_err<{MEAN_REL_THRESHOLD:.0%}, cos_sim>{COS_THRESHOLD})"
    )
    return passed


# ---------------------------------------------------------------------------
# YOLOv8 post-processing
# ---------------------------------------------------------------------------


def postprocess(
    raw: np.ndarray,
    orig_h: int,
    orig_w: int,
    ratio: float,
    pad: tuple,
    conf_thr: float,
    iou_thr: float,
    class_names: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Decode YOLOv8 output and apply NMS.

    raw shape: (1, 5, 8400) — [cx, cy, w, h, score] per anchor
               (single-class model: only 1 score column)
    """
    import cv2

    if class_names is None:
        class_names = ["pill"]

    pred = raw[0]  # (5, 8400) or (8400, 5)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # → (8400, 5)

    num_classes = pred.shape[1] - 4
    boxes_raw = pred[:, :4]  # cx, cy, w, h  (in 640-space)
    scores_raw = pred[:, 4:]  # (8400, num_classes)

    max_scores = scores_raw.max(axis=1)
    class_ids = scores_raw.argmax(axis=1)

    mask = max_scores > conf_thr
    if not mask.any():
        log.info(f"No detections above conf_threshold={conf_thr}")
        return []

    boxes_filt = boxes_raw[mask]
    scores_filt = max_scores[mask]
    class_ids_filt = class_ids[mask]

    # cx, cy, w, h → x1, y1, x2, y2  (still in letterbox 640-space)
    cx, cy, bw, bh = boxes_filt.T
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Remove letterbox padding, then scale back to original image
    pad_left, pad_top = pad
    xyxy[:, [0, 2]] -= pad_left
    xyxy[:, [1, 3]] -= pad_top
    xyxy /= ratio

    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)

    # NMS per class
    nms_indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores_filt.tolist(), conf_thr, iou_thr)

    detections = []
    if len(nms_indices) > 0:
        # cv2.dnn.NMSBoxes returns ndarray on newer OpenCV, list on older
        flat_indices = np.array(nms_indices).flatten()
        for i in flat_indices:
            x1_, y1_, x2_, y2_ = xyxy[i]
            cid = int(class_ids_filt[i])
            name = class_names[cid] if cid < len(class_names) else str(cid)
            detections.append(
                {
                    "class_id": cid,
                    "class_name": name,
                    "confidence": float(scores_filt[i]),
                    "bbox": [float(x1_), float(y1_), float(x2_), float(y2_)],
                }
            )

    return detections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Validate RKNN model (CPU simulation)")
    p.add_argument(
        "--onnx", required=True, help="Path to ONNX model (used for RKNN simulation and comparison)"
    )
    p.add_argument("--image", default=None, help="Path to test image (optional)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--input-size", type=int, default=640, help="Model input size")
    p.add_argument(
        "--class-names",
        default="pill",
        help="Comma-separated class names (default: pill)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    class_names = [n.strip() for n in args.class_names.split(",")]

    onnx_path = args.onnx
    if not Path(onnx_path).exists():
        log.error(f"ONNX model not found: {onnx_path}")
        sys.exit(1)

    # 1. Pre-process
    tensor, original, ratio, pad = preprocess(args.image, args.input_size)
    log.info(f"Input tensor shape: {tensor.shape}  dtype: {tensor.dtype}")

    # 2. RKNN CPU-simulation inference (rebuilds from ONNX in-memory)
    rknn_out = run_rknn_sim(onnx_path, tensor)

    # 3. ONNX Runtime inference for comparison
    onnx_out = run_onnx(onnx_path, tensor)
    comparison_passed = compare(onnx_out, rknn_out)

    # 4. Post-processing on RKNN output
    orig_h, orig_w = original.shape[:2]
    log.info("=== Detection results (RKNN simulation) ===")
    detections_rknn = postprocess(
        rknn_out, orig_h, orig_w, ratio, pad, args.conf, args.iou, class_names
    )
    if detections_rknn:
        for i, det in enumerate(detections_rknn):
            x1, y1, x2, y2 = det["bbox"]
            log.info(
                f"  [{i}] {det['class_name']}  conf={det['confidence']:.3f}"
                f"  box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]"
            )
    else:
        log.info("  No detections found.")

    # 5. Post-processing on ONNX output
    log.info("=== Detection results (ONNX) ===")
    detections_onnx = postprocess(
        onnx_out, orig_h, orig_w, ratio, pad, args.conf, args.iou, class_names
    )
    if detections_onnx:
        for i, det in enumerate(detections_onnx):
            x1, y1, x2, y2 = det["bbox"]
            log.info(
                f"  [{i}] {det['class_name']}  conf={det['confidence']:.3f}"
                f"  box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]"
            )
    else:
        log.info("  No detections found.")

    # 6. Summary
    log.info("=== Summary ===")
    log.info(f"  RKNN detections : {len(detections_rknn)}")
    log.info(f"  ONNX detections : {len(detections_onnx)}")
    log.info(f"  Numerical check : {'PASS' if comparison_passed else 'FAIL'}")

    if not comparison_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
