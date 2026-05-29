#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training_dataset import TrainingCamera, TrainingDataset, load_luma, load_rgb


def blender_pose_to_viewmat(pose: list[list[float]]) -> np.ndarray:
    c2w_blender = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    w2c_opencv = np.linalg.inv(c2w_blender).astype(np.float32)
    w2c_opencv[1:3, :] *= -1.0
    return w2c_opencv.astype(np.float32)


def camera_position_from_viewmat(viewmat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(viewmat).astype(np.float32)[:3, 3]


def depth_path_for_item(dataset_dir: Path, item: dict) -> Path:
    return dataset_dir / item.get("depth", item["rgb"].replace("_rgb.png", "_depth.png"))


def backproject_foreground_points(
    rgb: np.ndarray,
    depth: np.ndarray,
    alpha: np.ndarray,
    viewmat: np.ndarray,
    K: np.ndarray,
    max_depth: float,
    alpha_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = alpha >= alpha_threshold
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    z = depth[ys, xs] * float(max_depth)
    valid = z > 1.0e-5
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    x = (xs - float(K[0, 2])) * z / max(float(K[0, 0]), 1.0e-6)
    y = (ys - float(K[1, 2])) * z / max(float(K[1, 1]), 1.0e-6)
    points_c = np.stack([x, y, z, np.ones_like(z)], axis=1)
    points_w = (np.linalg.inv(viewmat).astype(np.float32) @ points_c.T).T[:, :3]
    colors = rgb[ys.astype(np.int32), xs.astype(np.int32)]
    return points_w.astype(np.float32), colors.astype(np.float32)


def load_b075x65r3x_dataset(
    dataset_dir: Path,
    width: int,
    height: int,
    max_frames: int = 0,
    frame_step: int = 1,
    start_index: int = 0,
    white_background: bool = True,
) -> TrainingDataset:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"B075X65R3X info.json not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    images = [(idx, item) for idx, item in enumerate(info.get("images", [])) if idx >= start_index]
    if frame_step > 1:
        images = images[::frame_step]
    if max_frames > 0:
        images = images[:max_frames]
    if not images:
        raise RuntimeError(f"No B075X65R3X frames selected from {dataset_dir}")

    cameras = []
    foreground_points = []
    foreground_colors = []
    for source_index, item in images:
        hw = item.get("HW", [height, width])
        raw_h, raw_w = int(hw[0]), int(hw[1])
        sx = float(width) / float(raw_w)
        sy = float(height) / float(raw_h)
        K = np.asarray(item["intrinsic"], dtype=np.float32).reshape(3, 3).copy()
        K[0, :] *= sx
        K[1, :] *= sy

        rgb_path = dataset_dir / item["rgb"]
        alpha_path = dataset_dir / item.get("alpha", item["rgb"].replace("_rgb.png", "_alpha.png"))
        depth_path = depth_path_for_item(dataset_dir, item)
        rgb = load_rgb(rgb_path, width, height)
        target = rgb.copy()
        alpha = np.ones((height, width), dtype=np.float32)
        if alpha_path.exists():
            alpha = load_luma(alpha_path, width, height)
        if white_background:
            target = alpha[..., None] * target + (1.0 - alpha[..., None])

        viewmat = blender_pose_to_viewmat(item["pose"])
        if alpha_path.exists() and depth_path.exists():
            depth = load_luma(depth_path, width, height)
            points, colors = backproject_foreground_points(
                rgb,
                depth,
                alpha,
                viewmat,
                K,
                float(item.get("max_depth", info.get("max_depth", 1.0))),
                alpha_threshold=0.5,
            )
            foreground_points.append(points)
            foreground_colors.append(colors)
        cameras.append(
            TrainingCamera(
                index=source_index,
                viewmat=viewmat,
                K=K.astype(np.float32),
                position=camera_position_from_viewmat(viewmat),
                target=target.astype(np.float32),
            )
        )

    bbox = np.asarray(info.get("bbox", [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]), dtype=np.float32)
    points_np = np.concatenate(foreground_points, axis=0) if foreground_points else np.zeros((0, 3), dtype=np.float32)
    colors_np = np.concatenate(foreground_colors, axis=0) if foreground_colors else np.zeros((0, 3), dtype=np.float32)
    metadata = {
        "source": str(dataset_dir),
        "format_version": info.get("format_version"),
        "width": width,
        "height": height,
        "frames": len(cameras),
        "frame_step": frame_step,
        "start_index": start_index,
        "white_background": white_background,
        "foreground_points": int(points_np.shape[0]),
        "bbox": bbox.astype(float).tolist(),
    }
    return TrainingDataset(
        name="B075X65R3X",
        cameras=cameras,
        bbox_min=bbox[0].astype(np.float32),
        bbox_max=bbox[1].astype(np.float32),
        metadata=metadata,
        foreground_points=points_np,
        foreground_colors=colors_np,
        background_color=np.array([1.0, 1.0, 1.0], dtype=np.float32) if white_background else np.zeros((3,), dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and validate the B075X65R3X dataset.")
    parser.add_argument("--data", type=Path, default=Path("datasets/B075X65R3X"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dark-background", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_b075x65r3x_dataset(
        args.data,
        args.width,
        args.height,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        start_index=args.start_index,
        white_background=not args.dark_background,
    )
    print(
        "B075X65R3X dataset ok "
        f"frames={len(dataset.cameras)} size={args.width}x{args.height} "
        f"foreground_points={0 if dataset.foreground_points is None else len(dataset.foreground_points)} "
        f"bbox_min={dataset.bbox_min.tolist()} bbox_max={dataset.bbox_max.tolist()}"
    )


if __name__ == "__main__":
    main()
