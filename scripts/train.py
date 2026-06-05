#!/usr/bin/env python3
"""Training script for YOLO models."""

import argparse
import sys

from yolo_demo.training.trainer import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model")
    parser.add_argument("data", type=str, help="Path to dataset YAML")
    parser.add_argument(
        "--model", "-m", type=str, default="yolov8n.pt", help="Pretrained model"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to config YAML"
    )

    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        # Override with command line args
        config.data = args.data
        if args.model != "yolov8n.pt":
            config.model = args.model
        config.epochs = args.epochs
        config.batch = args.batch
        config.imgsz = args.imgsz
        config.lr0 = args.lr0
        if args.output:
            config.project = args.output
    else:
        config = TrainingConfig(
            model=args.model,
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            project=args.output,
        )

    print("Training configuration:")
    print(f"  Model: {config.model}")
    print(f"  Data: {config.data}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch: {config.batch}")
    print(f"  Image size: {config.imgsz}")

    trainer = Trainer(config)
    print("\nStarting training...")

    result = trainer.train()

    if result.success:
        print("\n✓ Training completed!")
        print(f"  Model saved to: {result.model_path}")
        if result.metrics:
            print(f"  Metrics: {result.metrics}")
    else:
        print(f"\n✗ Training failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
