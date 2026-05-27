#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1.0e-5, 1.0 - 1.0e-5)
    return np.log(values / (1.0 - values)).astype(np.float32)


def axis_transform() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


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
    points = (axis_transform() @ points.T).T.astype(np.float32)
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    return points.astype(np.float32), colors.astype(np.float32), raw_count


def gsplat_to_spz_positions(points: np.ndarray) -> np.ndarray:
    out = np.empty_like(points, dtype=np.float32)
    out[:, 0] = points[:, 0]
    out[:, 1] = -points[:, 2]
    out[:, 2] = points[:, 1]
    return out


def export_spz(
    out_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    point_scale: float,
    opacity: float,
    color_mode: str,
) -> None:
    try:
        import spz
    except ImportError as exc:
        raise ImportError("The 'spz' Python package is required for SPZ export.") from exc

    n = int(points.shape[0])
    cloud = spz.GaussianCloud()
    cloud.antialiased = True
    cloud.positions = gsplat_to_spz_positions(points).reshape(-1).astype(np.float32)
    cloud.scales = np.full((n, 3), np.log(point_scale), dtype=np.float32).reshape(-1)
    cloud.rotations = np.tile(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        (n, 1),
    ).reshape(-1)
    cloud.alphas = np.full((n,), logit(np.array([opacity], dtype=np.float32))[0], dtype=np.float32)
    if color_mode == "rgb":
        cloud.colors = np.clip(colors, 0.0, 1.0).reshape(-1).astype(np.float32)
    elif color_mode == "sh0":
        sh_c0 = 0.28209479177387814
        cloud.colors = ((np.clip(colors, 0.0, 1.0) - 0.5) / sh_c0).reshape(-1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported color mode: {color_mode}")
    cloud.sh_degree = 0
    cloud.sh = np.array([], dtype=np.float32)

    opts = spz.PackOptions()
    ok = spz.save_spz(cloud, opts, str(out_path))
    if not ok:
        raise RuntimeError(f"failed to save spz to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/Users/yangdunfu/Downloads/2026_05_04_16_51_29"))
    parser.add_argument("--out", type=Path, default=Path("outputs/scanner_points.spz"))
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--point-scale", type=float, default=0.01)
    parser.add_argument("--opacity", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--color-mode",
        choices=("rgb", "sh0"),
        default="rgb",
        help="Use raw point RGB or convert RGB to SH degree-0 coefficients.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    points, colors, raw_point_count = prepare_points(args.data, args.max_points, args.seed)
    export_spz(args.out, points, colors, args.point_scale, args.opacity, args.color_mode)
    size = args.out.stat().st_size
    if size <= 0:
        raise AssertionError(f"SPZ output is empty: {args.out}")

    metadata = {
        "dataset": str(args.data),
        "out": str(args.out),
        "raw_point_count": raw_point_count,
        "exported_gaussians": int(points.shape[0]),
        "point_scale": args.point_scale,
        "scale_convention": "log(point_scale) per axis",
        "opacity": args.opacity,
        "opacity_convention": "logit(opacity)",
        "color_mode": args.color_mode,
        "color_convention": (
            "raw point RGB in [0, 1]"
            if args.color_mode == "rgb"
            else "(RGB - 0.5) / SH_C0 degree-0 coefficients"
        ),
        "position_convention": "Task 6.24 gsplat scanner points mapped to SPZ preview as [x, -z, y]",
        "rotation_convention": "identity quaternion wxyz=[1,0,0,0]",
        "file_size_bytes": size,
    }
    metadata_path = args.out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        "scanner points spz export ok "
        f"gaussians={points.shape[0]} file={args.out} bytes={size} "
        f"metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
