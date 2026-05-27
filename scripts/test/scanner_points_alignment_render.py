#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from render_random_3dgs_png import write_png
from scanner_dataset_random_render_smoke import (
    axis_transform,
    collect_frames,
    load_camera,
    load_target,
    render_random_scene,
)
from train_tiny_3dgs_mlx import image_to_u8, normalize_quats


def load_ply_positions_colors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("Reading points.ply requires the 'plyfile' package.") from exc

    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    names = vertices.data.dtype.names or ()
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.float32)
        if colors.max(initial=0.0) > 1.0:
            colors /= 255.0
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = np.full_like(points, 0.7, dtype=np.float32)
    return points, colors


def prepare_points(
    dataset_dir: Path,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    points, colors = load_ply_positions_colors(dataset_dir / "points.ply")
    raw_count = int(points.shape[0])
    points = (axis_transform()[:3, :3] @ points.T).T.astype(np.float32)
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    return points.astype(np.float32), colors.astype(np.float32), raw_count


def points_to_gaussians(
    points: np.ndarray,
    colors: np.ndarray,
    point_scale: float,
    opacity: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    n = int(points.shape[0])
    means = mx.array(points[None, ...], dtype=mx.float32)
    quats = mx.zeros((1, n, 4), dtype=mx.float32)
    quats = quats + mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32)
    scales = mx.full((1, n, 3), point_scale, dtype=mx.float32)
    rgba = mx.array(colors[None, None, ...], dtype=mx.float32)
    opacities = mx.full((1, n), opacity, dtype=mx.float32)
    return means, normalize_quats(quats), scales, rgba, opacities


def concat_compare(target: np.ndarray, render: np.ndarray) -> np.ndarray:
    gap = np.ones((target.shape[0], 6, 3), dtype=np.float32)
    return np.concatenate([target, gap, render], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/Users/yangdunfu/Downloads/2026_05_04_16_51_29"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scanner_points_alignment"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--point-scale", type=float, default=0.01)
    parser.add_argument("--opacity", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = collect_frames(args.data, args.max_frames, args.frame_step, args.start_index)
    cameras = [load_camera(frame, args.width, args.height) for frame in frames]
    points, colors, raw_point_count = prepare_points(args.data, args.max_points, args.seed)
    means, quats, scales, render_colors, opacities = points_to_gaussians(
        points,
        colors,
        args.point_scale,
        args.opacity,
    )

    debug = {
        "dataset": str(args.data),
        "width": args.width,
        "height": args.height,
        "raw_point_count": raw_point_count,
        "render_point_count": int(points.shape[0]),
        "point_scale": args.point_scale,
        "opacity": args.opacity,
        "frames": [],
    }
    total_visible = 0
    total_intersections = 0
    for view_id, camera in enumerate(cameras):
        target = load_target(camera.image_path, args.width, args.height)
        render = render_random_scene(
            camera,
            means,
            quats,
            scales,
            render_colors,
            opacities,
            args.width,
            args.height,
            args.tile_size,
        )
        mx.eval(*render.values())
        rgb = np.asarray(render["render_colors"][0], dtype=np.float32)
        alpha = np.asarray(render["render_alphas"][0, ..., 0], dtype=np.float32)
        radii = np.asarray(render["radii"])
        isect_ids = np.asarray(render["isect_ids"])
        visible = int(np.count_nonzero(np.any(radii > 0, axis=-1)))
        intersections = int(isect_ids.shape[0])
        total_visible += visible
        total_intersections += intersections

        write_png(args.out_dir / f"target_frame_{camera.index:05d}.png", image_to_u8(target))
        write_png(args.out_dir / f"points_render_frame_{camera.index:05d}.png", image_to_u8(rgb))
        write_png(args.out_dir / f"compare_frame_{camera.index:05d}.png", image_to_u8(concat_compare(target, rgb)))
        debug["frames"].append(
            {
                "view_id": view_id,
                "frame_index": int(camera.index),
                "image_path": str(camera.image_path),
                "K": camera.K.tolist(),
                "viewmat": camera.viewmat.tolist(),
                "visible_gaussians": visible,
                "intersections": intersections,
                "alpha_sum": float(alpha.sum()),
                "alpha_max": float(alpha.max(initial=0.0)),
                "alpha_nonzero_pixels": int(np.count_nonzero(alpha > 1.0e-6)),
            }
        )
        print(
            f"frame={camera.index:05d} visible_gaussians={visible} "
            f"intersections={intersections} alpha_sum={float(alpha.sum()):.6f} "
            f"alpha_max={float(alpha.max(initial=0.0)):.6f}"
        )

    (args.out_dir / "alignment_summary.json").write_text(
        json.dumps(debug, indent=2),
        encoding="utf-8",
    )
    if total_visible <= 0 or total_intersections <= 0:
        raise AssertionError("points alignment render expected visible points")
    print(
        "scanner points alignment render ok "
        f"frames={len(cameras)} raw_points={raw_point_count} "
        f"render_points={points.shape[0]} total_visible={total_visible} "
        f"total_intersections={total_intersections} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
