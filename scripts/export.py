#!/usr/bin/env python3
"""Export script for YOLO models."""

import argparse

from yolo_demo.export.onnx_exporter import ONNXExporter, prepare_for_rk3588


def main():
    parser = argparse.ArgumentParser(description="Export YOLO models to ONNX")
    parser.add_argument("model", type=str, help="Path to model (.pt file)")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output path or directory"
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument(
        "--no-dynamic", action="store_true", help="Disable dynamic axes"
    )
    parser.add_argument(
        "--no-simplify", action="store_true", help="Disable model simplification"
    )
    parser.add_argument(
        "--rk3588",
        action="store_true",
        help="Use recommended settings for RK3588",
    )

    args = parser.parse_args()

    if args.rk3588:
        print(f"Exporting {args.model} for RK3588...")
        onnx_path = prepare_for_rk3588(args.model, args.output)
        print(f"✓ Exported to: {onnx_path}")
        print("\nNext steps:")
        print("  1. Transfer the ONNX file to your RK3588 device")
        print("  2. Use rknn-toolkit2 to convert to RKNN format")
        print("  3. Deploy with RKNN runtime")
    else:
        print(f"Exporting {args.model} to ONNX (opset={args.opset})...")

        exporter = ONNXExporter(args.model)
        onnx_path = exporter.export(
            output=args.output,
            opset=args.opset,
            dynamic=not args.no_dynamic,
            simplify=not args.no_simplify,
        )

        print(f"✓ Exported to: {onnx_path}")
        print("\nYou can now use this ONNX model with:")
        print("  - ONNX Runtime")
        print("  - TensorRT")
        print("  - OpenVINO")
        print("  - RKNN (after conversion)")


if __name__ == "__main__":
    main()
