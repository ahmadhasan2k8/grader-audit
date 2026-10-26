"""Minimal image loader for the Tier-2 inference scripts (rescore*.py).

Tier-1 reproduction (reproduce.py) does not use this; it runs on the released logs. This is only needed
to regenerate logs from scratch on a GPU.
"""
from __future__ import annotations
from pathlib import Path

from PIL import Image


def load_image(path, max_size: int | None = None):
    """Open an image as RGB, optionally downscaling so the longest side is <= max_size."""
    img = Image.open(Path(path)).convert("RGB")
    if max_size:
        img.thumbnail((max_size, max_size))  # preserves aspect ratio; longest side <= max_size
    return img
