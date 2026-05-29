#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from training_dataset import TrainingCamera, TrainingDataset, image_to_u8, load_rgb, write_png


def make_synthetic_image(width: int, height: int) -> np.ndarray:
    image = np.ones((height, width, 3), dtype=np.float32)
    image[: height // 2, : width // 2] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    image[height // 2 :, width // 2 :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return image


def make_image_fitting_intrinsics(width: int, height: int, fov_x: float = math.pi / 2.0) -> np.ndarray:
    focal = 0.5 * float(width) / math.tan(0.5 * fov_x)
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def make_image_fitting_viewmat(camera_z: float = 8.0) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, camera_z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_image_fitting_dataset(
    out_dir: Path,
    width: int,
    height: int,
    img_path: Path | None = None,
    camera_z: float = 8.0,
    init_xy_extent: float | None = None,
    init_z_extent: float = 0.25,
) -> TrainingDataset:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = load_rgb(img_path, width, height) if img_path is not None else make_synthetic_image(width, height)
    target_path = out_dir / "target.png"
    write_png(target_path, image_to_u8(target))

    K = make_image_fitting_intrinsics(width, height)
    viewmat = make_image_fitting_viewmat(camera_z)
    position = np.linalg.inv(viewmat).astype(np.float32)[:3, 3]
    camera = TrainingCamera(
        index=0,
        viewmat=viewmat,
        K=K,
        position=position,
        target=target.astype(np.float32),
    )

    xy_extent = float(camera_z if init_xy_extent is None else init_xy_extent)
    bbox_min = np.array([-xy_extent, -xy_extent, -float(init_z_extent)], dtype=np.float32)
    bbox_max = np.array([xy_extent, xy_extent, float(init_z_extent)], dtype=np.float32)
    metadata = {
        "source": None if img_path is None else str(img_path),
        "target_path": str(target_path),
        "width": width,
        "height": height,
        "camera_z": camera_z,
        "init_xy_extent": xy_extent,
        "init_z_extent": init_z_extent,
        "fov_x_degrees": 90.0,
        "viewmat": viewmat.astype(float).tolist(),
        "K": K.astype(float).tolist(),
        "bbox": [bbox_min.astype(float).tolist(), bbox_max.astype(float).tolist()],
        "background_color": [0.0, 0.0, 0.0],
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return TrainingDataset(
        name="image_fitting",
        cameras=[camera],
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        metadata=metadata,
        background_color=np.zeros((3,), dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or inspect the single-image fitting dataset.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/image_fitting_dataset"))
    parser.add_argument("--img-path", type=Path, default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera-z", type=float, default=8.0)
    parser.add_argument("--init-xy-extent", type=float, default=None)
    parser.add_argument("--init-z-extent", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_image_fitting_dataset(
        args.out_dir,
        args.width,
        args.height,
        img_path=args.img_path,
        camera_z=args.camera_z,
        init_xy_extent=args.init_xy_extent,
        init_z_extent=args.init_z_extent,
    )
    print(
        "image-fitting dataset ok "
        f"frames={len(dataset.cameras)} size={args.width}x{args.height} "
        f"target={dataset.metadata['target_path']}"
    )


if __name__ == "__main__":
    main()
