#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DATASET = REPO_ROOT / "scripts" / "dataset"
SCRIPTS_TEST = REPO_ROOT / "scripts" / "test"
for path in (SCRIPTS_DATASET, SCRIPTS_TEST):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from b075x65r3x_dataset import load_b075x65r3x_dataset  # noqa: E402
from colmap_360_dataset import load_colmap_scene, select_colmap_cameras  # noqa: E402


@dataclass(frozen=True)
class DiagnosticCamera:
    index: int
    viewmat: np.ndarray
    K: np.ndarray
    width: int
    height: int
    image_path: Path | None = None


@dataclass(frozen=True)
class DiagnosticDataset:
    name: str
    source: Path
    cameras: list[DiagnosticCamera]
    points: np.ndarray
    colors: np.ndarray
    raw_point_count: int
    notes: dict[str, Any]


def axis_transform() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def scanner_pose_to_viewmat(raw_pose: list[float]) -> np.ndarray:
    c2w_src = np.array(raw_pose, dtype=np.float32).reshape(4, 4)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = axis_transform() @ c2w_src[:3, :3]
    c2w[:3, 3] = axis_transform() @ c2w_src[:3, 3]
    r = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    t = c2w[:3, 3:4]

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = r.T
    viewmat[:3, 3:4] = -r.T @ t
    return viewmat


def extract_frame_index(path: Path) -> int | None:
    match = re.search(r"frame_(\d+)\.(?:jpg|json)$", path.name)
    return int(match.group(1)) if match else None


def collect_scanner_frames(dataset_dir: Path, max_frames: int, frame_step: int, start_index: int) -> list[tuple[int, Path, Path]]:
    image_map = {
        idx: path
        for path in sorted(dataset_dir.glob("frame_*.jpg"))
        if (idx := extract_frame_index(path)) is not None
    }
    json_map = {
        idx: path
        for path in sorted(dataset_dir.glob("frame_*.json"))
        if (idx := extract_frame_index(path)) is not None
    }
    common = [idx for idx in sorted(set(image_map) & set(json_map)) if idx >= start_index]
    if frame_step > 1:
        common = common[::frame_step]
    if max_frames > 0:
        common = common[:max_frames]
    if not common:
        raise RuntimeError(f"No scanner frame_*.jpg/json pairs found in {dataset_dir}")
    return [(idx, image_map[idx], json_map[idx]) for idx in common]


def load_scanner_camera(frame: tuple[int, Path, Path], width: int, height: int) -> DiagnosticCamera:
    index, image_path, json_path = frame
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    intrinsics = raw.get("intrinsics")
    pose = raw.get("cameraPoseARFrame")
    if intrinsics is None or len(intrinsics) != 9:
        raise RuntimeError(f"Invalid intrinsics in {json_path}")
    if pose is None or len(pose) != 16:
        raise RuntimeError(f"Invalid cameraPoseARFrame in {json_path}")
    with Image.open(image_path) as image:
        raw_width, raw_height = image.size
    sx = float(width) / float(raw_width)
    sy = float(height) / float(raw_height)
    K = np.array(
        [
            [float(intrinsics[0]) * sx, 0.0, float(intrinsics[2]) * sx],
            [0.0, float(intrinsics[4]) * sy, float(intrinsics[5]) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return DiagnosticCamera(index=index, viewmat=scanner_pose_to_viewmat(pose), K=K, width=width, height=height, image_path=image_path)


def load_ply_positions_colors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("Reading scanner points.ply requires the 'plyfile' package.") from exc

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
    return points, colors.astype(np.float32)


def load_scanner_dataset(args: argparse.Namespace) -> DiagnosticDataset:
    frames = collect_scanner_frames(args.scanner_data, args.scanner_max_frames, args.scanner_frame_step, args.scanner_start_index)
    cameras = [load_scanner_camera(frame, args.width, args.height) for frame in frames]
    raw_points, colors = load_ply_positions_colors(args.scanner_data / "points.ply")
    points = (axis_transform() @ raw_points.T).T.astype(np.float32)
    return DiagnosticDataset(
        name="scanner",
        source=args.scanner_data,
        cameras=cameras,
        points=points,
        colors=colors,
        raw_point_count=int(points.shape[0]),
        notes={
            "loader": "scanner cameraPoseARFrame + axis_transform + OpenCV camera flip",
            "make_target": "codex-scanner-points-train-spz",
            "selected_frames": len(cameras),
        },
    )


def load_b075_dataset(args: argparse.Namespace) -> DiagnosticDataset:
    dataset = load_b075x65r3x_dataset(
        args.b075_data,
        args.width,
        args.height,
        max_frames=args.b075_max_frames,
        frame_step=args.b075_frame_step,
        start_index=args.b075_start_index,
        white_background=True,
    )
    cameras = [
        DiagnosticCamera(
            index=camera.index,
            viewmat=np.asarray(camera.viewmat, dtype=np.float32),
            K=np.asarray(camera.K, dtype=np.float32),
            width=args.width,
            height=args.height,
            image_path=None,
        )
        for camera in dataset.cameras
    ]
    return DiagnosticDataset(
        name="B075X65R3X",
        source=args.b075_data,
        cameras=cameras,
        points=np.asarray(dataset.foreground_points, dtype=np.float32),
        colors=np.asarray(dataset.foreground_colors, dtype=np.float32),
        raw_point_count=int(0 if dataset.foreground_points is None else dataset.foreground_points.shape[0]),
        notes={
            "loader": "info.json Blender pose converted to OpenCV viewmat; foreground points backprojected from alpha/depth",
            "make_target": "codex-fixed-points-train",
            "bbox_min": dataset.bbox_min.astype(float).tolist(),
            "bbox_max": dataset.bbox_max.astype(float).tolist(),
            "selected_frames": len(cameras),
        },
    )


def infer_colmap_size(data_dir: Path, factor: int) -> tuple[int, int]:
    image_dir = data_dir / ("images" if factor <= 1 else f"images_{factor}")
    candidates = sorted([*image_dir.glob("*.jpg"), *image_dir.glob("*.JPG"), *image_dir.glob("*.png"), *image_dir.glob("*.PNG")])
    if not candidates:
        raise FileNotFoundError(f"No images found in {image_dir}")
    with Image.open(candidates[0]) as image:
        return image.size


def load_360_dataset(args: argparse.Namespace) -> DiagnosticDataset:
    width, height = args.colmap_width, args.colmap_height
    if width <= 0 or height <= 0:
        width, height = infer_colmap_size(args.colmap_data, args.colmap_factor)
    scene = load_colmap_scene(
        args.colmap_data,
        args.colmap_factor,
        width,
        height,
        test_every=args.colmap_test_every,
        normalize_world_space=not args.colmap_no_normalize,
    )
    selected = select_colmap_cameras(
        scene.cameras,
        args.colmap_split,
        args.colmap_test_every,
        args.colmap_max_frames,
        args.colmap_frame_step,
        args.colmap_start_index,
    )
    cameras = [
        DiagnosticCamera(
            index=camera.index,
            viewmat=np.asarray(camera.viewmat, dtype=np.float32),
            K=np.asarray(camera.K, dtype=np.float32),
            width=width,
            height=height,
            image_path=camera.image_path,
        )
        for camera in selected
    ]
    return DiagnosticDataset(
        name="360_colmap",
        source=args.colmap_data,
        cameras=cameras,
        points=np.asarray(scene.points, dtype=np.float32),
        colors=np.asarray(scene.colors, dtype=np.float32),
        raw_point_count=int(scene.raw_point_count),
        notes={
            "loader": "COLMAP sparse reconstruction normalized like gsplat examples",
            "make_target": "codex-360-points-train-spz",
            "factor": args.colmap_factor,
            "split": args.colmap_split,
            "scene_scale": scene.scene_scale,
            "normalize_world_space": scene.normalize_world_space,
            "selected_frames": len(cameras),
        },
    )


def as_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": 0.0, "p05": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def camera_center(viewmat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(viewmat).astype(np.float32)[:3, 3]


def rotation_quality(viewmat: np.ndarray) -> tuple[float, float]:
    r = viewmat[:3, :3].astype(np.float64)
    det = float(np.linalg.det(r))
    ortho_error = float(np.linalg.norm((r.T @ r) - np.eye(3), ord="fro"))
    return det, ortho_error


def camera_summary(cameras: list[DiagnosticCamera]) -> dict[str, Any]:
    centers = np.stack([camera_center(camera.viewmat) for camera in cameras], axis=0)
    dets, ortho = zip(*(rotation_quality(camera.viewmat) for camera in cameras), strict=True)
    Ks = np.stack([camera.K for camera in cameras], axis=0)
    widths = np.asarray([camera.width for camera in cameras], dtype=np.float32)
    heights = np.asarray([camera.height for camera in cameras], dtype=np.float32)
    center = np.mean(centers, axis=0)
    radii = np.linalg.norm(centers - center[None, :], axis=1)
    view_dirs = np.stack([np.linalg.inv(c.viewmat).astype(np.float32)[:3, 2] for c in cameras], axis=0)
    to_scene = center[None, :] - centers
    to_scene_norm = np.linalg.norm(to_scene, axis=1)
    view_norm = np.linalg.norm(view_dirs, axis=1)
    cos_to_center = np.sum(view_dirs * to_scene, axis=1) / np.maximum(view_norm * to_scene_norm, 1.0e-8)
    return {
        "count": len(cameras),
        "center_bbox_min": centers.min(axis=0).astype(float).tolist(),
        "center_bbox_max": centers.max(axis=0).astype(float).tolist(),
        "center_bbox_size": (centers.max(axis=0) - centers.min(axis=0)).astype(float).tolist(),
        "center_radius": as_stats(radii),
        "rotation_det": as_stats(np.asarray(dets)),
        "rotation_orthogonality_error": as_stats(np.asarray(ortho)),
        "fx": as_stats(Ks[:, 0, 0]),
        "fy": as_stats(Ks[:, 1, 1]),
        "cx_ratio": as_stats(Ks[:, 0, 2] / widths),
        "cy_ratio": as_stats(Ks[:, 1, 2] / heights),
        "fx_over_width": as_stats(Ks[:, 0, 0] / widths),
        "fy_over_height": as_stats(Ks[:, 1, 1] / heights),
        "view_dir_cos_to_camera_center_mean": as_stats(cos_to_center),
    }


def point_summary(points: np.ndarray, colors: np.ndarray, sample_size: int, seed: int) -> dict[str, Any]:
    if points.size == 0:
        return {"count": 0}
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    center = (bbox_min + bbox_max) * 0.5
    radii = np.linalg.norm(points - center[None, :], axis=1)
    sampled = sample_points(points, min(sample_size, points.shape[0]), seed)
    nn_stats: dict[str, float] | None = None
    if sampled.shape[0] >= 3:
        try:
            from scipy.spatial import cKDTree

            distances, _ = cKDTree(sampled).query(sampled, k=2)
            nn_stats = as_stats(distances[:, 1])
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            nn_stats = {"error": str(exc)}
    color_channels = colors.reshape(-1, 3) if colors.size else np.zeros((0, 3), dtype=np.float32)
    return {
        "count": int(points.shape[0]),
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "bbox_size": bbox_size.astype(float).tolist(),
        "bbox_diagonal": float(np.linalg.norm(bbox_size)),
        "radius_from_bbox_center": as_stats(radii),
        "nearest_neighbor_distance": nn_stats,
        "color": {
            "r": as_stats(color_channels[:, 0]) if color_channels.size else as_stats(np.asarray([])),
            "g": as_stats(color_channels[:, 1]) if color_channels.size else as_stats(np.asarray([])),
            "b": as_stats(color_channels[:, 2]) if color_channels.size else as_stats(np.asarray([])),
        },
    }


def sample_points(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if points.shape[0] <= count:
        return points
    rng = np.random.default_rng(seed)
    ids = rng.choice(points.shape[0], size=count, replace=False)
    return points[ids]


def projection_summary(
    points: np.ndarray,
    cameras: list[DiagnosticCamera],
    sample_size: int,
    max_cameras: int,
    seed: int,
) -> dict[str, Any]:
    sampled = sample_points(points, min(sample_size, points.shape[0]), seed)
    selected = cameras[:max_cameras] if max_cameras > 0 else cameras
    per_camera = []
    positive_fracs = []
    in_frame_fracs = []
    median_depths = []
    for camera in selected:
        points_h = np.concatenate([sampled, np.ones((sampled.shape[0], 1), dtype=np.float32)], axis=1)
        camera_points = (camera.viewmat @ points_h.T).T[:, :3]
        z = camera_points[:, 2]
        positive = z > 1.0e-6
        projected = (camera.K @ camera_points.T).T
        denom = np.where(np.abs(projected[:, 2]) > 1.0e-8, projected[:, 2], np.nan)
        u = projected[:, 0] / denom
        v = projected[:, 1] / denom
        in_frame = positive & (u >= 0.0) & (u < camera.width) & (v >= 0.0) & (v < camera.height)
        positive_frac = float(np.mean(positive))
        in_frame_frac = float(np.mean(in_frame))
        positive_fracs.append(positive_frac)
        in_frame_fracs.append(in_frame_frac)
        median_depth = float(np.median(z[positive])) if np.any(positive) else 0.0
        median_depths.append(median_depth)
        per_camera.append(
            {
                "index": camera.index,
                "positive_depth_fraction": positive_frac,
                "in_frame_fraction": in_frame_frac,
                "median_positive_depth": median_depth,
                "projected_u": as_stats(u[np.isfinite(u) & positive]),
                "projected_v": as_stats(v[np.isfinite(v) & positive]),
            }
        )
    return {
        "sampled_points": int(sampled.shape[0]),
        "camera_count": len(selected),
        "positive_depth_fraction": as_stats(np.asarray(positive_fracs)),
        "in_frame_fraction": as_stats(np.asarray(in_frame_fracs)),
        "median_positive_depth": as_stats(np.asarray(median_depths)),
        "per_camera": per_camera,
    }


def scale_ratios(camera_info: dict[str, Any], point_info: dict[str, Any]) -> dict[str, float]:
    point_diag = float(point_info.get("bbox_diagonal", 0.0))
    cam_radius = float(camera_info["center_radius"]["median"])
    cam_bbox = np.asarray(camera_info["center_bbox_size"], dtype=np.float64)
    return {
        "point_bbox_diagonal_over_camera_radius_median": point_diag / max(cam_radius, 1.0e-8),
        "point_bbox_diagonal_over_camera_bbox_diagonal": point_diag / max(float(np.linalg.norm(cam_bbox)), 1.0e-8),
    }


def summarize_dataset(dataset: DiagnosticDataset, args: argparse.Namespace) -> dict[str, Any]:
    camera_info = camera_summary(dataset.cameras)
    point_info = point_summary(dataset.points, dataset.colors, args.nn_sample_points, args.seed)
    projection_info = projection_summary(
        dataset.points,
        dataset.cameras,
        args.projection_sample_points,
        args.projection_frames,
        args.seed,
    )
    return {
        "name": dataset.name,
        "source": str(dataset.source),
        "raw_point_count": dataset.raw_point_count,
        "notes": dataset.notes,
        "cameras": camera_info,
        "points": point_info,
        "scale_ratios": scale_ratios(camera_info, point_info),
        "projection": projection_info,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare B075X65R3X, scanner, and gsplat 360 dataset geometry.")
    parser.add_argument("--b075-data", type=Path, default=REPO_ROOT / "datasets" / "B075X65R3X")
    parser.add_argument("--scanner-data", type=Path, default=Path("/Users/yangdunfu/Downloads/2026_05_04_16_51_29"))
    parser.add_argument("--colmap-data", type=Path, default=REPO_ROOT / "submodules/gsplat/examples/datasets/data/360_v2/garden")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/dataset_diagnostics/dataset_geometry_diagnostics.json")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--projection-sample-points", type=int, default=50000)
    parser.add_argument("--projection-frames", type=int, default=16)
    parser.add_argument("--nn-sample-points", type=int, default=50000)

    parser.add_argument("--b075-max-frames", type=int, default=0)
    parser.add_argument("--b075-frame-step", type=int, default=1)
    parser.add_argument("--b075-start-index", type=int, default=0)

    parser.add_argument("--scanner-max-frames", type=int, default=999)
    parser.add_argument("--scanner-frame-step", type=int, default=1)
    parser.add_argument("--scanner-start-index", type=int, default=0)

    parser.add_argument("--colmap-factor", type=int, default=4)
    parser.add_argument("--colmap-width", type=int, default=0)
    parser.add_argument("--colmap-height", type=int, default=0)
    parser.add_argument("--colmap-test-every", type=int, default=8)
    parser.add_argument("--colmap-split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--colmap-max-frames", type=int, default=0)
    parser.add_argument("--colmap-frame-step", type=int, default=1)
    parser.add_argument("--colmap-start-index", type=int, default=0)
    parser.add_argument("--colmap-no-normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [load_b075_dataset(args), load_scanner_dataset(args), load_360_dataset(args)]
    report = {
        "purpose": "Dataset geometry, camera, point-cloud, and projection diagnostics for SPZ training quality investigation.",
        "datasets": [summarize_dataset(dataset, args) for dataset in datasets],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {args.out}")
    for item in report["datasets"]:
        in_frame = item["projection"]["in_frame_fraction"]["median"]
        positive = item["projection"]["positive_depth_fraction"]["median"]
        point_count = item["points"]["count"]
        ratio = item["scale_ratios"]["point_bbox_diagonal_over_camera_radius_median"]
        nn = item["points"].get("nearest_neighbor_distance") or {}
        nn_median = nn.get("median", "n/a") if isinstance(nn, dict) else "n/a"
        print(
            f"{item['name']}: cameras={item['cameras']['count']} points={point_count} "
            f"median_positive_depth={positive:.4f} median_in_frame={in_frame:.4f} "
            f"point_diag/camera_radius={ratio:.4f} nn_median={nn_median}"
        )


if __name__ == "__main__":
    main()
