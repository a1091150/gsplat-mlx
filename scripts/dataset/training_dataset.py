from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class TrainingCamera:
    index: int
    viewmat: np.ndarray
    K: np.ndarray
    position: np.ndarray
    target: np.ndarray
    alpha_mask: np.ndarray | None = None


@dataclass
class TrainingDataset:
    name: str
    cameras: list[TrainingCamera]
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    metadata: dict
    foreground_points: np.ndarray | None = None
    foreground_colors: np.ndarray | None = None
    background_color: np.ndarray | None = None


def write_png(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("write_png expects uint8 RGB image with shape [H, W, 3].")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )


def image_to_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def load_luma(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        luma = image.convert("L")
        if luma.size != (width, height):
            luma = luma.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(luma, dtype=np.float32) / 255.0
