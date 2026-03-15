#!/usr/bin/env python3
"""
Convert ONNX model to RKNN format for RK3588 deployment.

This script handles the complete conversion process:
1. Load ONNX model
2. Create RKNN model
3. Build with optional quantization
4. Export RKNN file

Usage:
    python convert_to_rknn.py model.onnx [-o output.rknn] [--quantize] [--dataset calib.txt]

Requirements:
    - rknn-toolkit2 (Linux x86_64 only, Python 3.6-3.8)
    - onnx (optional, for model validation)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RKNNConverter:
    """Convert ONNX models to RKNN format for RK3588."""

    def __init__(
        self,
        target_platform: str = "rk3588",
        verbose: bool = True,
    ):
        """
        Initialize the RKNN converter.

        Args:
            target_platform: Target RKNN platform (rk3588, rk3568, rk3566, etc.)
            verbose: Enable verbose logging
        """
        self.target_platform = target_platform
        self.verbose = verbose
        self.rknn = None
        self.quantization_enabled = False

    def load_onnx(
        self,
        onnx_path: str | Path,
        check_model: bool = True,
    ) -> bool:
        """
        Load ONNX model for conversion.

        Args:
            onnx_path: Path to ONNX model file
            check_model: Whether to check ONNX model validity

        Returns:
            True if loaded successfully
        """
        try:
            from rknn.api import RKNN
        except ImportError as e:
            logger.error(
                "rknn-toolkit2 not installed. Please install on Linux x86_64:\n"
                "  pip install rknn-toolkit2==1.5.0"
            )
            raise e

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        logger.info(f"Loading ONNX model: {onnx_path}")

        self.rknn = RKNN(verbose=self.verbose)

        # Configure for target platform
        logger.info(f"Configuring for target platform: {self.target_platform}")
        self.rknn.config(
            target_platform=self.target_platform,
            quantization=self.quantization_enabled,
            optimization_level=3,  # Default optimization level
        )

        # Load ONNX model
        load_success = self.rknn.load_onnx(
            model=str(onnx_path),
            check_model=check_model,
        )

        if load_success != 0:
            raise RuntimeError("Failed to load ONNX model")

        logger.info("ONNX model loaded successfully")
        return True

    def build(
        self,
        dataset: Optional[str | Path] = None,
        do_quantization: bool = False,
        calibration_iterations: int = 1000,
    ) -> bool:
        """
        Build RKNN model.

        Args:
            dataset: Path to calibration dataset file (for quantization)
            do_quantization: Enable INT8 quantization
            calibration_iterations: Number of calibration iterations

        Returns:
            True if built successfully
        """
        if self.rknn is None:
            raise RuntimeError("No model loaded. Call load_onnx() first.")

        self.quantization_enabled = do_quantization

        if do_quantization:
            if dataset is None:
                raise ValueError(
                    "Calibration dataset required for quantization. "
                    "Provide path to dataset file (list of image paths)"
                )

            dataset_path = Path(dataset)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Calibration dataset not found: {dataset_path}")

            logger.info(f"Building with INT8 quantization using: {dataset_path}")
            logger.info(f"Calibration iterations: {calibration_iterations}")
        else:
            logger.info("Building with FP16 (no quantization)")

        # Build the model
        build_success = self.rknn.build(
            do_quantization=do_quantization,
            dataset=str(dataset) if dataset else None,
        )

        if build_success != 0:
            raise RuntimeError("Failed to build RKNN model")

        logger.info("RKNN model built successfully")
        return True

    def export_rknn(
        self,
        output_path: str | Path,
        cleanup: bool = True,
    ) -> str:
        """
        Export RKNN model to file.

        Args:
            output_path: Output path for RKNN file
            cleanup: Cleanup temporary files after export

        Returns:
            Path to exported RKNN file
        """
        if self.rknn is None:
            raise RuntimeError("No model to export. Call build() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting RKNN model to: {output_path}")

        export_success = self.rknn.export_rknn(str(output_path))

        if export_success != 0:
            raise RuntimeError("Failed to export RKNN model")

        logger.info(f"RKNN model exported successfully: {output_path}")

        # Report model size
        rknn_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"RKNN model size: {rknn_size:.2f} MB")

        if cleanup and self.rknn:
            self.rknn.release()
            self.rknn = None

        return str(output_path)

    def convert(
        self,
        onnx_path: str | Path,
        output_path: str | Path,
        do_quantization: bool = False,
        dataset: Optional[str | Path] = None,
    ) -> str:
        """
        Complete conversion pipeline.

        Args:
            onnx_path: Input ONNX model path
            output_path: Output RKNN model path
            do_quantization: Enable INT8 quantization
            dataset: Calibration dataset path (required if quantization enabled)

        Returns:
            Path to exported RKNN file
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("RKNN Conversion Pipeline")
        logger.info("=" * 60)

        # Step 1: Load ONNX model
        self.load_onnx(onnx_path)

        # Step 2: Build RKNN model
        self.build(
            dataset=dataset,
            do_quantization=do_quantization,
        )

        # Step 3: Export RKNN model
        rknn_path = self.export_rknn(output_path)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Conversion completed in {elapsed:.2f} seconds")
        logger.info(f"Output: {rknn_path}")
        logger.info("=" * 60)

        return rknn_path

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        if self.rknn:
            self.rknn.release()
            self.rknn = None
            logger.info("RKNN resources released")


def validate_onnx(onnx_path: str | Path) -> bool:
    """
    Validate ONNX model structure.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        True if valid
    """
    try:
        import onnx
    except ImportError:
        logger.warning("onnx not installed, skipping validation")
        return True

    onnx_path = Path(onnx_path)
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info(f"ONNX model validation passed: {onnx_path}")

        # Print model info
        logger.info(f"  Opset: {model.opset_import[0]}")
        logger.info(f"  Inputs: {[i.name for i in model.graph.input]}")
        logger.info(f"  Outputs: {[o.name for o in model.graph.output]}")

        return True
    except Exception as e:
        logger.error(f"ONNX model validation failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to RKNN format for RK3588"
    )
    parser.add_argument(
        "onnx_model",
        type=str,
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output RKNN model path (default: <model_name>.rknn)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization (requires calibration dataset)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to calibration dataset file (required if --quantize)"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="rk3588",
        choices=["rk3588", "rk3568", "rk3566", "rv1126", "rv1109"],
        help="Target RKNN platform"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX model before conversion"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup file logging if specified
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")

    # Validate ONNX model if requested
    if args.validate:
        logger.info("Validating ONNX model...")
        if not validate_onnx(args.onnx_model):
            logger.error("ONNX validation failed, aborting conversion")
            sys.exit(1)

    # Determine output path
    onnx_path = Path(args.onnx_model)
    if args.output:
        output_path = Path(args.output)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".rknn")
    else:
        # Default: same name as ONNX but .rknn extension
        output_path = onnx_path.with_suffix(".rknn")

    # Check if ONNX file exists
    if not onnx_path.exists():
        logger.error(f"ONNX model not found: {onnx_path}")
        sys.exit(1)

    # Validate quantization arguments
    if args.quantize and not args.dataset:
        logger.error("--quantize requires --dataset to be provided")
        sys.exit(1)

    # Run conversion
    try:
        converter = RKNNConverter(
            target_platform=args.platform,
            verbose=args.verbose,
        )

        rknn_path = converter.convert(
            onnx_path=onnx_path,
            output_path=output_path,
            do_quantization=args.quantize,
            dataset=args.dataset if args.quantize else None,
        )

        logger.info(f"SUCCESS: RKNN model saved to {rknn_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
