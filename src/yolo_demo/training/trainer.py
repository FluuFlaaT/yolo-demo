"""Incremental training module for YOLO models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from ultralytics import YOLO


@dataclass
class TrainingConfig:
    """Configuration for YOLO training."""

    # Model
    model: str = "yolov8n.pt"  # Model variant or path to weights

    # Data
    data: str | None = None  # Path to dataset YAML

    # Training hyperparameters
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    patience: int = 50
    lr0: float = 0.01
    lrf: float = 0.01

    # Device
    device: str | int | list[int] | None = None  # 'cpu', 0, '0,1', etc.

    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5

    # Other
    workers: int = 8
    optimizer: str = "auto"
    verbose: bool = True
    exist_ok: bool = False
    project: str | None = None
    name: str | None = None

    # Additional kwargs
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for ultralytics."""
        d = {
            "model": self.model,
            "data": self.data,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "patience": self.patience,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "workers": self.workers,
            "optimizer": self.optimizer,
            "verbose": self.verbose,
            "exist_ok": self.exist_ok,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
        }
        if self.device is not None:
            d["device"] = self.device
        if self.project is not None:
            d["project"] = self.project
        if self.name is not None:
            d["name"] = self.name
        d.update(self.extra_kwargs)
        return d


@dataclass
class TrainingResult:
    """Result of a training run."""

    success: bool
    model_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


class Trainer:
    """YOLO trainer supporting incremental training."""

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.model: YOLO | None = None

    def load_pretrained(self, model_path: str | None = None) -> None:
        """
        Load a pretrained model for incremental training.

        Args:
            model_path: Path to pretrained weights. If None, uses config.model.
        """
        path = model_path or self.config.model
        self.model = YOLO(path)

    def train(
        self,
        data_yaml: str | Path | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """
        Run training.

        Args:
            data_yaml: Path to dataset YAML file.
            **kwargs: Additional arguments to override config.

        Returns:
            TrainingResult with training outcome.
        """
        try:
            if self.model is None:
                self.load_pretrained()

            # Merge config with overrides
            train_kwargs = self.config.to_dict()
            train_kwargs.update(kwargs)
            if data_yaml:
                train_kwargs["data"] = str(data_yaml)

            # Remove None values
            train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}

            # Run training
            results = self.model.train(**train_kwargs)

            # Get best model path
            best_model = str(Path(self.model.model_save_dir) / "weights" / "best.pt")

            # Extract metrics
            metrics = {}
            if hasattr(results, "result_dict"):
                metrics = results.result_dict

            return TrainingResult(
                success=True,
                model_path=best_model,
                metrics=metrics,
            )

        except Exception as e:
            return TrainingResult(success=False, error=str(e))

    def export_onnx(self, output: str | Path | None = None, **kwargs: Any) -> str | None:
        """
        Export trained model to ONNX format.

        Args:
            output: Output path for ONNX file.
            **kwargs: Additional arguments for export.

        Returns:
            Path to exported ONNX file, or None if export failed.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_pretrained() or train() first.")

        export_kwargs = {"format": "onnx", "dynamic": True, "opset": 11}
        export_kwargs.update(kwargs)

        if output:
            export_kwargs["save_dir"] = str(Path(output).parent)

        path = self.model.export(**export_kwargs)
        return str(path)
