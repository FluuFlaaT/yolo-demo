"""RKNN conversion module for yolo-demo package."""

import sys
from pathlib import Path

# Add scripts directory to path to import convert_to_rknn
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from convert_to_rknn import main  # noqa: E402

__all__ = ["main"]
