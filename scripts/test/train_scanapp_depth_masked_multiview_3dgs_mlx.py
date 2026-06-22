#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import Adam
from PIL import Image

from render_random_3dgs_png import write_png
from scanner_dataset_utils import load_target
from scanner_points_training_utils import (
    SH_C0,
    FrameBatchSampler,
    ScannerDefaultStrategyConfig,
    ScannerDefaultStrategyRuntime,
    ScannerPointsSHModel,
    append_random_gaussians,
    camera_batch_arrays,
    concat_compare,
    export_trained_spz,
    image_to_u8,
    lr_for_step,
    make_lr_schedule,
    normalize_quats,
    opacity_diagnostics,
    points_extent_diagnostics,
    render_sh_model,
    save_model_parameters_npz,
    sh_coeff_count,
    spz_export_diagnostics,
    ssim_index,
    target_batch_array,
)


MAX_SUPPORTED_SH_DEGREE = 3
MIN_DEPTH_METERS = 0.05
MAX_DEPTH_METERS = 6.0
DEFAULT_MASK_MAX_DEPTH_METERS = 5.0
MIN_CONFIDENCE = 1


@dataclass(frozen=True)
class ScanAppFrame:
    index: int
    name: str
    image_path: Path
    metadata_path: Path
    depth_path: Path
    confidence_path: Path | None
    width: int
    height: int
    depth_width: int
    depth_height: int
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray


@dataclass(frozen=True)
class ScanAppCamera:
    index: int
    viewmat: np.ndarray
    K: np.ndarray
    image_path: Path
    raw_width: int
    raw_height: int
    frame_name: str


@dataclass(frozen=True)
class ScanAppScene:
    frames: list[ScanAppFrame]
    cameras: list[ScanAppCamera]
    points: np.ndarray
    colors: np.ndarray
    raw_point_count: int
    sampled_point_count: int
    retained_point_count: int
    colorized_point_count: int
    confidence_frame_count: int
    confidence_kept_count: int
    confidence_rejected_count: int
    depth_valid_count: int
    depth_rejected_count: int
    sample_step: int
    scene_scale: float


def log(message: str) -> None:
    print(message, flush=True)


def gsplat_active_sh_degree(step: int, target: int, interval: int) -> int:
    return int(min(step // interval, target))


def scanner_axis3x3() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def scanner_axis4x4() -> np.ndarray:
    axis = np.eye(4, dtype=np.float32)
    axis[:3, :3] = scanner_axis3x3()
    return axis


def inverse_rigid4x4(c2w: np.ndarray) -> np.ndarray:
    r = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3:4].astype(np.float32)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = r.T
    viewmat[:3, 3:4] = -r.T @ t
    return viewmat


def scanner_pose_to_viewmat(raw_camera_to_world: np.ndarray) -> np.ndarray:
    c2w = scanner_axis4x4() @ raw_camera_to_world.astype(np.float32)
    c2w[:3, :3] = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    return inverse_rigid4x4(c2w)


def scanner_camera_center(raw_camera_to_world: np.ndarray) -> np.ndarray:
    c2w = scanner_axis4x4() @ raw_camera_to_world.astype(np.float32)
    return c2w[:3, 3].astype(np.float32)


def transform_scanner_points(points: np.ndarray) -> np.ndarray:
    return (scanner_axis3x3() @ points.T).T.astype(np.float32)


def infer_image_size(data_dir: Path) -> tuple[int, int]:
    image_dir = data_dir / "images"
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            with Image.open(path) as image:
                return image.size
    raise RuntimeError(f"No images found in {image_dir}")


def require_array(raw: dict, key: str, count: int, metadata_path: Path) -> np.ndarray:
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise RuntimeError(f"Invalid {key} in {metadata_path}")
    return np.asarray(value, dtype=np.float32)


def load_scanapp_frames(data_dir: Path) -> list[ScanAppFrame]:
    metadata_dir = data_dir / "metadata"
    if not metadata_dir.exists():
        raise FileNotFoundError(f"ScanApp metadata directory not found: {metadata_dir}")
    frames: list[ScanAppFrame] = []
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        image_rel = raw.get("image")
        depth = raw.get("depth")
        if not isinstance(image_rel, str) or not isinstance(depth, dict):
            continue
        depth_rel = depth.get("path")
        if not isinstance(depth_rel, str):
            continue

        width = int(raw.get("width", 0))
        height = int(raw.get("height", 0))
        depth_width = int(depth.get("width", 0))
        depth_height = int(depth.get("height", 0))
        if width <= 0 or height <= 0 or depth_width <= 0 or depth_height <= 0:
            continue

        image_path = data_dir / image_rel
        depth_path = data_dir / depth_rel
        confidence_rel = depth.get("confidence_path")
        confidence_path = data_dir / confidence_rel if isinstance(confidence_rel, str) else None
        if not image_path.exists() or not depth_path.exists():
            continue

        intrinsics = require_array(raw, "intrinsics", 9, metadata_path).reshape(3, 3)
        camera_to_world = require_array(raw, "camera_to_world", 16, metadata_path).reshape(4, 4)
        world_to_camera = require_array(raw, "world_to_camera", 16, metadata_path).reshape(4, 4)
        frame_index = int(raw.get("frame_index", len(frames)))
        frame_name = str(raw.get("frame_name", metadata_path.stem))
        frames.append(
            ScanAppFrame(
                index=frame_index,
                name=frame_name,
                image_path=image_path,
                metadata_path=metadata_path,
                depth_path=depth_path,
                confidence_path=confidence_path if confidence_path and confidence_path.exists() else None,
                width=width,
                height=height,
                depth_width=depth_width,
                depth_height=depth_height,
                intrinsics=intrinsics.astype(np.float32),
                camera_to_world=camera_to_world.astype(np.float32),
                world_to_camera=world_to_camera.astype(np.float32),
            )
        )
    frames.sort(key=lambda frame: (frame.index, frame.metadata_path.name))
    if not frames:
        raise RuntimeError(f"No usable ScanApp frames found in {metadata_dir}")
    return frames


def select_frames(
    frames: list[ScanAppFrame],
    max_frames: int,
    frame_step: int,
    start_index: int,
) -> list[ScanAppFrame]:
    selected = [frame for frame in frames if int(frame.index) >= start_index]
    if frame_step > 1:
        selected = selected[::frame_step]
    if max_frames > 0:
        selected = selected[:max_frames]
    if not selected:
        raise RuntimeError("No ScanApp frames selected")
    return selected


def load_scanapp_cameras(frames: list[ScanAppFrame], width: int, height: int) -> list[ScanAppCamera]:
    cameras: list[ScanAppCamera] = []
    for frame in frames:
        sx = float(width) / float(max(1, frame.width))
        sy = float(height) / float(max(1, frame.height))
        k = frame.intrinsics.copy()
        k[0, :] *= sx
        k[1, :] *= sy
        cameras.append(
            ScanAppCamera(
                index=frame.index,
                viewmat=scanner_pose_to_viewmat(frame.camera_to_world),
                K=k.astype(np.float32),
                image_path=frame.image_path,
                raw_width=frame.width,
                raw_height=frame.height,
                frame_name=frame.name,
            )
        )
    return cameras


def read_depth_float32(path: Path, width: int, height: int) -> np.ndarray:
    expected = width * height
    data = np.fromfile(path, dtype="<f4", count=expected)
    if data.size != expected:
        raise RuntimeError(f"Invalid depth data {path}: expected {expected} float32 values, found {data.size}")
    return data.reshape(height, width)


def read_confidence(path: Path | None, width: int, height: int) -> np.ndarray | None:
    if path is None:
        return None
    expected = width * height
    data = np.fromfile(path, dtype=np.uint8, count=expected)
    if data.size != expected:
        return None
    return data.reshape(height, width)


def load_depth_mask(
    frame: ScanAppFrame,
    width: int,
    height: int,
    min_depth_meters: float,
    max_depth_meters: float,
    min_confidence: int,
) -> np.ndarray:
    depth = read_depth_float32(frame.depth_path, frame.depth_width, frame.depth_height)
    confidence = read_confidence(frame.confidence_path, frame.depth_width, frame.depth_height)
    xs = np.clip(
        ((np.arange(width, dtype=np.float32) + 0.5) * float(frame.depth_width) / float(width)).astype(np.int32),
        0,
        frame.depth_width - 1,
    )
    ys = np.clip(
        ((np.arange(height, dtype=np.float32) + 0.5) * float(frame.depth_height) / float(height)).astype(np.int32),
        0,
        frame.depth_height - 1,
    )
    sampled_depth = depth[ys[:, None], xs[None, :]]
    mask = np.isfinite(sampled_depth) & (sampled_depth >= min_depth_meters) & (sampled_depth <= max_depth_meters)
    if confidence is not None:
        sampled_confidence = confidence[ys[:, None], xs[None, :]]
        mask &= sampled_confidence >= min_confidence
    return mask.astype(np.float32)[..., None]


def make_depth_points_for_frame(
    frame: ScanAppFrame,
    sample_step: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    depth = read_depth_float32(frame.depth_path, frame.depth_width, frame.depth_height)
    confidence = read_confidence(frame.confidence_path, frame.depth_width, frame.depth_height)
    ys = np.arange(0, frame.depth_height, sample_step, dtype=np.int32)
    xs = np.arange(0, frame.depth_width, sample_step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sampled_depth = depth[grid_y, grid_x]
    finite = np.isfinite(sampled_depth)
    depth_mask = finite & (sampled_depth >= MIN_DEPTH_METERS) & (sampled_depth <= MAX_DEPTH_METERS)
    if confidence is not None:
        confidence_values = confidence[grid_y, grid_x]
        confidence_mask = confidence_values >= MIN_CONFIDENCE
    else:
        confidence_mask = np.ones_like(depth_mask, dtype=bool)
    keep = depth_mask & confidence_mask

    kept_x = grid_x[keep].astype(np.float32)
    kept_y = grid_y[keep].astype(np.float32)
    kept_z = sampled_depth[keep].astype(np.float32)
    if kept_z.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), {
            "sampled": int(sampled_depth.size),
            "depth_valid": 0,
            "depth_rejected": int(sampled_depth.size),
            "confidence_kept": 0,
            "confidence_rejected": int(np.count_nonzero(depth_mask)) if confidence is not None else 0,
            "colorized": 0,
        }

    sx = float(frame.depth_width) / float(max(1, frame.width))
    sy = float(frame.depth_height) / float(max(1, frame.height))
    fx = float(frame.intrinsics[0, 0]) * sx
    fy = float(frame.intrinsics[1, 1]) * sy
    cx = float(frame.intrinsics[0, 2]) * sx
    cy = float(frame.intrinsics[1, 2]) * sy
    local_x = (kept_x - cx) * kept_z / fx
    local_y = -(kept_y - cy) * kept_z / fy
    local_z = -kept_z
    local = np.stack([local_x, local_y, local_z, np.ones_like(local_z)], axis=1).astype(np.float32)
    world = (frame.camera_to_world @ local.T).T[:, :3].astype(np.float32)
    world = transform_scanner_points(world)

    with Image.open(frame.image_path) as image:
        rgb = image.convert("RGB")
        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
    image_x = np.clip(((kept_x + 0.5) * float(frame.width) / float(max(1, frame.depth_width))).astype(np.int32), 0, frame.width - 1)
    image_y = np.clip(((kept_y + 0.5) * float(frame.height) / float(max(1, frame.depth_height))).astype(np.int32), 0, frame.height - 1)
    colors = rgb_np[image_y, image_x].astype(np.float32)

    depth_valid = int(np.count_nonzero(depth_mask))
    return world, colors, {
        "sampled": int(sampled_depth.size),
        "depth_valid": depth_valid,
        "depth_rejected": int(sampled_depth.size - depth_valid),
        "confidence_kept": int(np.count_nonzero(depth_mask & confidence_mask)),
        "confidence_rejected": int(np.count_nonzero(depth_mask & ~confidence_mask)) if confidence is not None else 0,
        "colorized": int(colors.shape[0]),
    }


def reservoir_sample_points(
    points: np.ndarray,
    colors: np.ndarray,
    target_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= target_points:
        return points.astype(np.float32), colors.astype(np.float32)
    rng = np.random.default_rng(seed)
    keep = rng.choice(points.shape[0], size=target_points, replace=False)
    return points[keep].astype(np.float32), colors[keep].astype(np.float32)


def scene_scale_from_cameras_and_points(cameras: list[ScanAppCamera], points: np.ndarray) -> float:
    centers = np.stack([np.linalg.inv(camera.viewmat).astype(np.float32)[:3, 3] for camera in cameras], axis=0)
    center = np.mean(centers, axis=0)
    camera_radius = float(np.max(np.linalg.norm(centers - center[None, :], axis=1)))
    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) if points.size else 0.0
    return max(camera_radius, bbox_diag * 0.5, 1.0e-6)


def load_scanapp_scene(
    data_dir: Path,
    width: int,
    height: int,
    max_frames: int,
    frame_step: int,
    start_index: int,
    target_points: int,
    seed: int,
) -> ScanAppScene:
    frames = select_frames(load_scanapp_frames(data_dir), max_frames, frame_step, start_index)
    cameras = load_scanapp_cameras(frames, width, height)
    sample_step = 1
    point_chunks = []
    color_chunks = []
    totals = {
        "sampled": 0,
        "depth_valid": 0,
        "depth_rejected": 0,
        "confidence_kept": 0,
        "confidence_rejected": 0,
        "colorized": 0,
    }
    confidence_frame_count = 0
    for frame in frames:
        if frame.confidence_path is not None:
            confidence_frame_count += 1
        points, colors, stats = make_depth_points_for_frame(frame, sample_step)
        if points.size:
            point_chunks.append(points)
            color_chunks.append(colors)
        for key, value in stats.items():
            totals[key] += int(value)
    if not point_chunks:
        raise RuntimeError("Depth reconstruction produced no points")
    raw_points = np.concatenate(point_chunks, axis=0).astype(np.float32)
    raw_colors = np.concatenate(color_chunks, axis=0).astype(np.float32)
    points, colors = reservoir_sample_points(raw_points, raw_colors, target_points, seed)
    scene_scale = scene_scale_from_cameras_and_points(cameras, points)
    return ScanAppScene(
        frames=frames,
        cameras=cameras,
        points=points,
        colors=colors,
        raw_point_count=int(raw_points.shape[0]),
        sampled_point_count=int(totals["sampled"]),
        retained_point_count=int(points.shape[0]),
        colorized_point_count=int(totals["colorized"]),
        confidence_frame_count=confidence_frame_count,
        confidence_kept_count=int(totals["confidence_kept"]),
        confidence_rejected_count=int(totals["confidence_rejected"]),
        depth_valid_count=int(totals["depth_valid"]),
        depth_rejected_count=int(totals["depth_rejected"]),
        sample_step=int(sample_step),
        scene_scale=scene_scale,
    )


def init_sh_model_from_points(
    points: np.ndarray,
    colors: np.ndarray,
    log_scales: np.ndarray,
    opacity: float,
    max_sh_degree: int,
) -> ScannerPointsSHModel:
    n = int(points.shape[0])
    means = mx.array(points[None, ...], dtype=mx.float32)
    quats = mx.zeros((1, n, 4), dtype=mx.float32) + mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32)
    features_dc = mx.array(((colors[None, ...] - 0.5) / SH_C0).astype(np.float32), dtype=mx.float32)
    rest_count = sh_coeff_count(max_sh_degree) - 1
    features_rest = mx.zeros((1, n, rest_count, 3), dtype=mx.float32)
    opacity_logits = mx.log(mx.full((1, n), opacity, dtype=mx.float32) / (1.0 - opacity))
    return ScannerPointsSHModel.from_arrays(
        means,
        normalize_quats(quats),
        mx.array(log_scales[None, ...], dtype=mx.float32),
        features_dc,
        features_rest,
        opacity_logits,
    )


def knn_log_scales_from_points(points: np.ndarray, init_scale: float = 1.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have shape [N, 3] and be nonempty")
    if points.shape[0] < 4:
        distances = np.linalg.norm(points - points.mean(axis=0, keepdims=True), axis=1)
        fallback = float(max(np.mean(distances), 1.0e-6) * init_scale)
        return np.full((points.shape[0], 3), np.log(fallback), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        if points.shape[0] > 20_000:
            raise ImportError(
                "KNN scale initialization for full ScanApp data requires scipy.spatial.cKDTree; "
                "install scipy or run a reduced --target-points smoke."
            )
        distances = []
        for start in range(0, points.shape[0], 1024):
            chunk = points[start : start + 1024]
            dist2 = np.sum((chunk[:, None, :] - points[None, :, :]) ** 2, axis=-1)
            nearest2 = np.partition(dist2, kth=3, axis=1)[:, :4]
            distances.append(np.sqrt(nearest2))
        distances = np.concatenate(distances, axis=0)
    else:
        distances, _ = cKDTree(points).query(points, k=4)
    dist_avg = np.sqrt(np.mean(np.square(distances[:, 1:]), axis=1))
    scales = np.maximum(dist_avg * float(init_scale), 1.0e-8)
    return np.repeat(np.log(scales)[:, None], 3, axis=1).astype(np.float32)


def loss_components(
    image: mx.array,
    target: mx.array,
    mask: mx.array,
    ssim_lambda: float,
    ssim_window_size: int,
) -> dict[str, mx.array]:
    mask_sum = mx.maximum(mx.sum(mask) * 3.0, 1.0)
    diff = mx.abs(image - target) * mask
    l1 = mx.sum(diff) / mask_sum
    ssim_image = image * mask + target * (1.0 - mask)
    ssim = ssim_index(ssim_image, target, ssim_window_size)
    ssim_loss = 1.0 - ssim
    loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * ssim_loss
    return {"loss": loss, "l1": l1, "ssim": ssim, "ssim_loss": ssim_loss, "mask_pixels": mx.sum(mask)}


def render_loss_stats(
    model: ScannerPointsSHModel,
    camera,
    target: mx.array,
    mask: mx.array,
    width: int,
    height: int,
    tile_size: int,
    sh_degree: int,
    ssim_lambda: float,
    ssim_window_size: int,
) -> dict:
    viewmats, ks = camera_batch_arrays([camera], [0])
    viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)
    render = render_sh_model(model, viewspace_points, viewmats, ks, width, height, tile_size, sh_degree)
    components = loss_components(render["render_colors"], target, mask, ssim_lambda, ssim_window_size)
    diff = (render["render_colors"] - target) * mask
    mse = mx.sum(diff * diff) / mx.maximum(mx.sum(mask) * 3.0, 1.0)
    mx.eval(
        components["loss"],
        components["l1"],
        components["ssim"],
        components["ssim_loss"],
        components["mask_pixels"],
        mse,
        render["render_colors"],
        render["radii"],
        render["flatten_ids"],
    )
    mse_value = float(np.asarray(mse))
    radii = np.asarray(render["radii"])
    flatten_ids = np.asarray(render["flatten_ids"])
    mask_pixels = float(np.asarray(components["mask_pixels"]))
    return {
        "frame_index": int(camera.index),
        "loss": float(np.asarray(components["loss"])),
        "loss_components": {
            "l1": float(np.asarray(components["l1"])),
            "ssim": float(np.asarray(components["ssim"])),
            "ssim_loss": float(np.asarray(components["ssim_loss"])),
            "mask_pixels": mask_pixels,
            "mask_fraction": mask_pixels / float(width * height),
        },
        "psnr": float(-10.0 * np.log10(max(mse_value, 1.0e-12))),
        "visible_gaussians": int(np.count_nonzero(np.any(radii > 0, axis=-1))),
        "intersections": int(flatten_ids.shape[0]),
        "image": np.asarray(render["render_colors"][0], dtype=np.float32),
    }


def evaluate_frames(
    model: ScannerPointsSHModel,
    cameras,
    targets: list[mx.array],
    masks: list[mx.array],
    width: int,
    height: int,
    tile_size: int,
    sh_degree: int,
    ssim_lambda: float,
    ssim_window_size: int,
) -> list[dict]:
    return [
        render_loss_stats(model, camera, target, mask, width, height, tile_size, sh_degree, ssim_lambda, ssim_window_size)
        for camera, target, mask in zip(cameras, targets, masks, strict=True)
    ]


def render_step_grid(
    model: ScannerPointsSHModel,
    cameras,
    width: int,
    height: int,
    tile_size: int,
    sh_degree: int,
) -> np.ndarray:
    tiles = []
    for index in range(16):
        camera = cameras[index % len(cameras)]
        viewmats, ks = camera_batch_arrays([camera], [0])
        viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)
        render = render_sh_model(model, viewspace_points, viewmats, ks, width, height, tile_size, sh_degree)
        mx.eval(render["render_colors"])
        tiles.append(image_to_u8(np.asarray(render["render_colors"][0], dtype=np.float32)))
    rows = [np.concatenate(tiles[start : start + 4], axis=1) for start in range(0, 16, 4)]
    return np.concatenate(rows, axis=0)


def mean_loss(stats: list[dict]) -> float:
    return float(np.mean([item["loss"] for item in stats])) if stats else 0.0


def validate_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/Users/yangdunfu/Documents/iOSProject/ScanProject/20260618_154636"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scanapp_depth_train"))
    parser.add_argument("--out-spz", type=Path, default=None)
    parser.add_argument("--out-model-npz", type=Path, default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--target-points", type=int, default=262_144)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--eval-max-frames", type=int, default=0)
    parser.add_argument("--eval-frame-step", type=int, default=None)
    parser.add_argument("--eval-start-index", type=int, default=0)
    parser.add_argument("--num-random-gaussians", type=int, default=0)
    parser.add_argument("--random-gaussian-bounds-scale", type=float, default=1.05)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--opacity", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-sampling", choices=("sequential", "shuffle"), default="shuffle")
    parser.add_argument("--frame-shuffle-seed", type=int, default=None)
    parser.add_argument("--ssim-lambda", type=float, default=0.2)
    parser.add_argument("--ssim-window-size", type=int, default=11)
    parser.add_argument("--mask-min-depth", type=float, default=MIN_DEPTH_METERS)
    parser.add_argument("--mask-max-depth", type=float, default=DEFAULT_MASK_MAX_DEPTH_METERS)
    parser.add_argument("--mask-min-confidence", type=int, default=MIN_CONFIDENCE)
    parser.add_argument("--means-lr", type=float, default=1.6e-4)
    parser.add_argument("--scales-lr", type=float, default=5.0e-3)
    parser.add_argument("--opacities-lr", type=float, default=5.0e-2)
    parser.add_argument("--quats-lr", type=float, default=1.0e-3)
    parser.add_argument("--sh0-lr", type=float, default=2.5e-3)
    parser.add_argument("--shn-lr", type=float, default=2.5e-3 / 20.0)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--sh-degree-interval", type=int, default=1000)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--step-image-interval", type=int, default=0)
    parser.add_argument("--mlx-cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--refine-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-prune-opa", type=float, default=0.005)
    parser.add_argument("--refine-grow-grad2d", type=float, default=0.0002)
    parser.add_argument("--refine-grow-scale3d", type=float, default=0.01)
    parser.add_argument("--refine-grow-scale2d", type=float, default=0.05)
    parser.add_argument("--refine-prune-scale3d", type=float, default=0.1)
    parser.add_argument("--refine-prune-scale2d", type=float, default=0.15)
    parser.add_argument("--refine-scale2d-stop-iter", type=int, default=0)
    parser.add_argument("--refine-start-iter", type=int, default=500)
    parser.add_argument("--refine-stop-iter", type=int, default=15000)
    parser.add_argument("--refine-reset-every", type=int, default=3000)
    parser.add_argument("--refine-every", type=int, default=100)
    parser.add_argument("--refine-pause-after-reset", type=int, default=0)
    parser.add_argument("--spz-scale-mode", choices=("direct", "scanner_axis"), default="direct")
    parser.add_argument("--spz-rotation-mode", choices=("direct", "position_axis", "fastgs_conjugate", "position_conjugate"), default="position_axis")
    parser.add_argument("--spz-quat-order", choices=("wxyz", "xyzw"), default="xyzw")
    parser.add_argument("--spz-color-mode", choices=("sh", "raw_rgb"), default="sh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mlx_cache_limit_gb < 0.0:
        raise ValueError("--mlx-cache-limit-gb must be nonnegative")
    cache_limit_bytes = int(args.mlx_cache_limit_gb * 1024**3)
    previous_cache_limit = mx.set_cache_limit(cache_limit_bytes)
    log(
        "mlx cache limit configured "
        f"current={cache_limit_bytes} bytes ({args.mlx_cache_limit_gb:.2f} GiB) "
        f"previous={previous_cache_limit} bytes"
    )
    if args.sh_degree < 0 or args.sh_degree > MAX_SUPPORTED_SH_DEGREE:
        raise ValueError(f"--sh-degree must be in [0, {MAX_SUPPORTED_SH_DEGREE}]")
    if args.sh_degree_interval <= 0:
        raise ValueError("--sh-degree-interval must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    if args.eval_frame_step is not None and args.eval_frame_step <= 0:
        raise ValueError("--eval-frame-step must be positive")
    if args.target_points <= 0:
        raise ValueError("--target-points must be positive")
    if args.mask_min_depth < 0.0:
        raise ValueError("--mask-min-depth must be nonnegative")
    if args.mask_max_depth <= args.mask_min_depth:
        raise ValueError("--mask-max-depth must be greater than --mask-min-depth")
    if args.mask_min_confidence < 0:
        raise ValueError("--mask-min-confidence must be nonnegative")
    if args.refine_stop_iter <= args.refine_start_iter:
        raise ValueError("--refine-stop-iter must be greater than --refine-start-iter")
    for name, value in [
        ("--init-scale", args.init_scale),
        ("--opacity", args.opacity),
        ("--means-lr", args.means_lr),
        ("--scales-lr", args.scales_lr),
        ("--opacities-lr", args.opacities_lr),
        ("--quats-lr", args.quats_lr),
        ("--sh0-lr", args.sh0_lr),
        ("--shn-lr", args.shn_lr),
    ]:
        validate_positive(name, value)

    width, height = args.width, args.height
    if width <= 0 or height <= 0:
        log(f"inferring image size data={args.data}")
        inferred_width, inferred_height = infer_image_size(args.data)
        width = inferred_width if width <= 0 else width
        height = inferred_height if height <= 0 else height

    args.out_dir.mkdir(parents=True, exist_ok=True)
    step_image_dir = args.out_dir / "step"
    step_image_count = 0
    if args.step_image_interval > 0:
        step_image_dir.mkdir(parents=True, exist_ok=True)

    log(
        "loading ScanApp depth scene "
        f"data={args.data} size={width}x{height} "
        f"target_points={args.target_points}"
    )
    scene = load_scanapp_scene(
        args.data,
        width=width,
        height=height,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        start_index=args.start_index,
        target_points=args.target_points,
        seed=args.seed,
    )
    cameras = scene.cameras
    eval_frame_step = args.frame_step if args.eval_frame_step is None else args.eval_frame_step
    eval_frames = (
        select_frames(load_scanapp_frames(args.data), args.eval_max_frames, eval_frame_step, args.eval_start_index)
        if args.eval_max_frames > 0
        else []
    )
    eval_cameras = load_scanapp_cameras(eval_frames, width, height) if eval_frames else []
    targets = [mx.array(load_target(camera.image_path, width, height)[None, ...], dtype=mx.float32) for camera in cameras]
    target_masks = [
        mx.array(
            load_depth_mask(frame, width, height, args.mask_min_depth, args.mask_max_depth, args.mask_min_confidence)[None, ...],
            dtype=mx.float32,
        )
        for frame in scene.frames
    ]
    eval_targets = [mx.array(load_target(camera.image_path, width, height)[None, ...], dtype=mx.float32) for camera in eval_cameras]
    eval_masks = [
        mx.array(
            load_depth_mask(frame, width, height, args.mask_min_depth, args.mask_max_depth, args.mask_min_confidence)[None, ...],
            dtype=mx.float32,
        )
        for frame in eval_frames
    ]
    mask_fractions = [float(np.asarray(mx.sum(mask)) / float(width * height)) for mask in target_masks]
    eval_mask_fractions = [float(np.asarray(mx.sum(mask)) / float(width * height)) for mask in eval_masks]
    log(
        "loaded masked targets "
        f"train_frames={len(cameras)} eval_frames={len(eval_cameras)} "
        f"mask_max_depth={args.mask_max_depth:.3f} "
        f"mask_fraction_mean={float(np.mean(mask_fractions)) if mask_fractions else 0.0:.6f} "
        f"scene_scale={scene.scene_scale:.8f}"
    )

    points, colors, raw_point_count = scene.points, scene.colors, scene.raw_point_count
    points, colors = append_random_gaussians(points, colors, args.num_random_gaussians, args.seed + 1009, args.random_gaussian_bounds_scale)
    log(
        "initializing KNN log scales "
        f"points={points.shape[0]} raw_depth_points={raw_point_count} "
        f"target_points={args.target_points} init_scale={args.init_scale}"
    )
    log_scales = knn_log_scales_from_points(points, args.init_scale)
    point_diagnostics = points_extent_diagnostics(points)
    resolved_scene_scale = float(scene.scene_scale * 1.1 * args.global_scale)
    means_lr = float(args.means_lr * resolved_scene_scale)
    means_lr_final = means_lr * 0.01

    log(
        "initializing SH model "
        f"gaussians={points.shape[0]} sh_degree={args.sh_degree} "
        f"opacity={args.opacity}"
    )
    model = init_sh_model_from_points(points, colors, log_scales, args.opacity, args.sh_degree)
    strategy_config = ScannerDefaultStrategyConfig(
        enabled=args.refine_enabled,
        prune_opa=args.refine_prune_opa,
        grow_grad2d=args.refine_grow_grad2d,
        grow_scale3d=args.refine_grow_scale3d,
        grow_scale2d=args.refine_grow_scale2d,
        prune_scale3d=args.refine_prune_scale3d,
        prune_scale2d=args.refine_prune_scale2d,
        refine_scale2d_stop_iter=args.refine_scale2d_stop_iter,
        refine_start_iter=args.refine_start_iter,
        refine_stop_iter=args.refine_stop_iter,
        reset_every=args.refine_reset_every,
        refine_every=args.refine_every,
        pause_refine_after_reset=args.refine_pause_after_reset,
        scene_scale=resolved_scene_scale,
        absgrad=False,
        revised_opacity=False,
    )
    strategy = ScannerDefaultStrategyRuntime(strategy_config, initial_gaussians=model.means.shape[1])
    active_sh_degree = gsplat_active_sh_degree(0, args.sh_degree, args.sh_degree_interval)
    sh_degree_events = [{"step": 0, "active_sh_degree": int(active_sh_degree)}]

    log(f"running initial evaluation frames={len(cameras)} eval_frames={len(eval_cameras)}")
    initial_stats = evaluate_frames(model, cameras, targets, target_masks, width, height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)
    eval_initial_stats = (
        evaluate_frames(model, eval_cameras, eval_targets, eval_masks, width, height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)
        if eval_cameras
        else []
    )
    initial_mean_loss = mean_loss(initial_stats)
    log(f"initial evaluation complete initial_mean_loss={initial_mean_loss:.8f}")

    sampler = FrameBatchSampler(
        frame_count=len(cameras),
        batch_size=args.batch_size,
        mode=args.frame_sampling,
        seed=args.seed + 7919 if args.frame_shuffle_seed is None else args.frame_shuffle_seed,
    )

    def sh_loss_fn(
        means: mx.array,
        quats: mx.array,
        log_scales_: mx.array,
        features_dc: mx.array,
        features_rest: mx.array,
        opacity_logits: mx.array,
        viewspace_points: mx.array,
        viewmats: mx.array,
        ks: mx.array,
        target: mx.array,
        mask: mx.array,
    ) -> mx.array:
        local = ScannerPointsSHModel.from_arrays(means, quats, log_scales_, features_dc, features_rest, opacity_logits)
        losses = []
        radii = []
        batch = int(viewmats.shape[1])
        for idx in range(batch):
            render = render_sh_model(
                local,
                viewspace_points[:, idx : idx + 1],
                viewmats[:, idx : idx + 1],
                ks[:, idx : idx + 1],
                width,
                height,
                args.tile_size,
                active_sh_degree,
            )
            losses.append(
                loss_components(
                    render["render_colors"],
                    target[idx : idx + 1],
                    mask[idx : idx + 1],
                    args.ssim_lambda,
                    args.ssim_window_size,
                )["loss"]
            )
            radii.append(render["radii"])
        return mx.mean(mx.stack(losses)), mx.concatenate(radii, axis=1)

    grad_fn = mx.value_and_grad(sh_loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))
    optimizers = {
        "means": Adam(learning_rate=means_lr),
        "quats": Adam(learning_rate=args.quats_lr),
        "log_scales": Adam(learning_rate=args.scales_lr),
        "features_dc": Adam(learning_rate=args.sh0_lr),
        "features_rest": Adam(learning_rate=args.shn_lr),
        "opacity_logits": Adam(learning_rate=args.opacities_lr),
    }
    lr_schedules = {
        "means": make_lr_schedule(means_lr, means_lr_final, 1.0, args.steps),
        "quats": make_lr_schedule(args.quats_lr, None, 1.0, args.steps),
        "log_scales": make_lr_schedule(args.scales_lr, None, 1.0, args.steps),
        "features_dc": make_lr_schedule(args.sh0_lr, None, 1.0, args.steps),
        "features_rest": make_lr_schedule(args.shn_lr, None, 1.0, args.steps),
        "opacity_logits": make_lr_schedule(args.opacities_lr, None, 1.0, args.steps),
    }

    last_loss = None
    last_viewspace_grad = None
    last_viewspace_grad_norm = None
    log(f"entering training loop steps={args.steps} batch_size={args.batch_size}")
    for step in range(1, args.steps + 1):
        latest_lrs = {}
        for name, schedule in lr_schedules.items():
            lr = lr_for_step(schedule, step)
            optimizers[name].learning_rate = lr
            schedule["latest"] = float(lr)
            latest_lrs[name] = float(lr)
        if step == 1 or step == args.steps or step % args.log_interval == 0:
            for schedule in lr_schedules.values():
                schedule["history"].append({"step": int(step), "lr": float(schedule["latest"])})

        next_active_sh_degree = gsplat_active_sh_degree(step, args.sh_degree, args.sh_degree_interval)
        if next_active_sh_degree != active_sh_degree:
            active_sh_degree = next_active_sh_degree
            sh_degree_events.append({"step": int(step), "active_sh_degree": int(active_sh_degree)})

        batch_ids = sampler.next_batch()
        batch_frame_indices = [int(cameras[idx].index) for idx in batch_ids]
        target = target_batch_array(targets, batch_ids)
        mask = target_batch_array(target_masks, batch_ids)
        viewmats, ks = camera_batch_arrays(cameras, batch_ids)
        viewspace_points = mx.zeros((1, len(batch_ids), model.means.shape[1], 2), dtype=mx.float32)
        (loss, strategy_radii), grads = grad_fn(
            model.means,
            model.quats,
            model.log_scales,
            model.features_dc,
            model.features_rest,
            model.opacity_logits,
            viewspace_points,
            viewmats,
            ks,
            target,
            mask,
        )
        d_means, d_quats, d_log_scales, d_features_dc, d_features_rest, d_opacity_logits, d_viewspace = grads
        mx.eval(loss, d_viewspace)
        last_loss = float(np.asarray(loss))
        last_viewspace_grad = d_viewspace
        last_viewspace_grad_norm = float(np.linalg.norm(np.asarray(d_viewspace)))

        optimizers["means"].update(model, {"means": d_means})
        optimizers["quats"].update(model, {"quats": d_quats})
        optimizers["log_scales"].update(model, {"log_scales": d_log_scales})
        optimizers["features_dc"].update(model, {"features_dc": d_features_dc})
        optimizers["features_rest"].update(model, {"features_rest": d_features_rest})
        optimizers["opacity_logits"].update(model, {"opacity_logits": d_opacity_logits})
        model.quats = normalize_quats(model.quats)
        mx.eval(model.means, model.quats, model.log_scales, model.features_dc, model.features_rest, model.opacity_logits)

        if strategy.config.enabled:
            strategy.update_state(d_viewspace, strategy_radii, width=width, height=height, n_cameras=len(batch_ids))
        strategy.after_optimizer_step(step, model, optimizers, "sh")

        if step == 1 or step == args.steps or step % args.log_interval == 0:
            log(
                f"step={step:04d} frames={batch_frame_indices} "
                f"sh={active_sh_degree} loss={last_loss:.8f} "
                f"means_lr={latest_lrs['means']:.8g} viewspace_grad_norm={last_viewspace_grad_norm:.8f}"
            )
        if args.step_image_interval > 0 and step % args.step_image_interval == 0:
            step_image_count += 1
            image = render_step_grid(model, cameras, width, height, args.tile_size, active_sh_degree)
            out_path = step_image_dir / f"out_{step_image_count:06d}.png"
            write_png(out_path, image)
            log(f"wrote step image step={step} path={out_path}")

    log(f"running final evaluation frames={len(cameras)} eval_frames={len(eval_cameras)}")
    final_stats = evaluate_frames(model, cameras, targets, target_masks, width, height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)
    eval_final_stats = (
        evaluate_frames(model, eval_cameras, eval_targets, eval_masks, width, height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)
        if eval_cameras
        else []
    )
    final_mean_loss = mean_loss(final_stats)

    for initial, final, target in zip(initial_stats, final_stats, targets, strict=True):
        frame_index = final["frame_index"]
        write_png(args.out_dir / f"compare_frame_{frame_index:05d}.png", image_to_u8(concat_compare(np.asarray(target[0]), initial["image"], final["image"])))
    for initial, final, target in zip(eval_initial_stats, eval_final_stats, eval_targets, strict=True):
        frame_index = final["frame_index"]
        write_png(args.out_dir / f"compare_eval_frame_{frame_index:05d}.png", image_to_u8(concat_compare(np.asarray(target[0]), initial["image"], final["image"])))

    if last_loss is None or not np.isfinite(final_mean_loss):
        raise AssertionError("ScanApp masked depth training loss should be finite")
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("ScanApp masked depth training expected nonzero viewspace_points gradient")

    out_spz = args.out_spz if args.out_spz is not None else args.out_dir / "trained_scanapp_depth.spz"
    spz_diagnostics = spz_export_diagnostics(model, "sh", active_sh_degree, args.spz_scale_mode, args.spz_rotation_mode, args.spz_quat_order, args.spz_color_mode)
    exported_gaussians = export_trained_spz(out_spz, model, "sh", active_sh_degree, args.spz_scale_mode, args.spz_rotation_mode, args.spz_quat_order, args.spz_color_mode)
    spz_size = out_spz.stat().st_size
    if spz_size <= 0:
        raise AssertionError(f"SPZ output is empty: {out_spz}")

    frame_summaries = [
        {
            "frame_index": int(final["frame_index"]),
            "initial_loss": float(initial["loss"]),
            "final_loss": float(final["loss"]),
            "initial_loss_components": initial["loss_components"],
            "final_loss_components": final["loss_components"],
            "initial_psnr": float(initial["psnr"]),
            "final_psnr": float(final["psnr"]),
            "initial_visible_gaussians": int(initial["visible_gaussians"]),
            "final_visible_gaussians": int(final["visible_gaussians"]),
            "initial_intersections": int(initial["intersections"]),
            "final_intersections": int(final["intersections"]),
        }
        for initial, final in zip(initial_stats, final_stats, strict=True)
    ]
    eval_frame_summaries = [
        {
            "frame_index": int(final["frame_index"]),
            "initial_loss": float(initial["loss"]),
            "final_loss": float(final["loss"]),
            "initial_loss_components": initial["loss_components"],
            "final_loss_components": final["loss_components"],
            "initial_psnr": float(initial["psnr"]),
            "final_psnr": float(final["psnr"]),
            "initial_visible_gaussians": int(initial["visible_gaussians"]),
            "final_visible_gaussians": int(final["visible_gaussians"]),
            "initial_intersections": int(initial["intersections"]),
            "final_intersections": int(final["intersections"]),
        }
        for initial, final in zip(eval_initial_stats, eval_final_stats, strict=True)
    ]
    eval_final_mean_loss = mean_loss(eval_final_stats) if eval_final_stats else None
    refinement_summary = strategy.summary()
    final_opacity_diagnostics = opacity_diagnostics(model)
    summary = {
        "dataset_type": "scanapp_depth_masked",
        "dataset": str(args.data),
        "width": int(width),
        "height": int(height),
        "raw_point_count": int(raw_point_count),
        "candidate_point_count": int(raw_point_count),
        "sampled_point_count": int(scene.sampled_point_count),
        "target_point_count": int(args.target_points),
        "retained_point_count": int(scene.retained_point_count),
        "exported_gaussians": int(exported_gaussians),
        "point_cloud_gaussians": int(points.shape[0] - args.num_random_gaussians),
        "random_gaussians": int(args.num_random_gaussians),
        "frames": len(cameras),
        "depth_frames": len(scene.frames),
        "eval_frames": len(eval_cameras),
        "confidence_frame_count": int(scene.confidence_frame_count),
        "confidence_kept_count": int(scene.confidence_kept_count),
        "confidence_rejected_count": int(scene.confidence_rejected_count),
        "depth_valid_count": int(scene.depth_valid_count),
        "depth_rejected_count": int(scene.depth_rejected_count),
        "colorized_point_count": int(scene.colorized_point_count),
        "depth_sample_step": int(scene.sample_step),
        "mask_config": {
            "min_depth_meters": float(args.mask_min_depth),
            "max_depth_meters": float(args.mask_max_depth),
            "min_confidence": int(args.mask_min_confidence),
            "mask_fraction_min": float(np.min(mask_fractions)) if mask_fractions else 0.0,
            "mask_fraction_mean": float(np.mean(mask_fractions)) if mask_fractions else 0.0,
            "mask_fraction_max": float(np.max(mask_fractions)) if mask_fractions else 0.0,
            "eval_mask_fraction_min": float(np.min(eval_mask_fractions)) if eval_mask_fractions else None,
            "eval_mask_fraction_mean": float(np.mean(eval_mask_fractions)) if eval_mask_fractions else None,
            "eval_mask_fraction_max": float(np.max(eval_mask_fractions)) if eval_mask_fractions else None,
        },
        "steps": int(args.steps),
        "step_image_interval": int(args.step_image_interval),
        "step_image_count": int(step_image_count),
        "step_image_dir": str(step_image_dir) if args.step_image_interval > 0 else None,
        "mlx_cache_limit_bytes": int(cache_limit_bytes),
        "mlx_cache_limit_gb": float(args.mlx_cache_limit_gb),
        "mlx_previous_cache_limit_bytes": int(previous_cache_limit),
        "scene_scale": float(scene.scene_scale),
        "resolved_scene_scale": float(resolved_scene_scale),
        "initialization": {
            "type": "scanapp_depth_reconstruction",
            "scale_rule": "average distance to 3 nearest neighbors times init_scale",
            "sampling_rule": "all valid depth pixels are reconstructed first, then globally sampled to target_point_count",
            "init_scale": float(args.init_scale),
            "opacity": float(args.opacity),
            "point_extent": point_diagnostics,
            "log_scale_min": float(log_scales.min()),
            "log_scale_mean": float(log_scales.mean()),
            "log_scale_max": float(log_scales.max()),
        },
        "gsplat_default_parity": {
            "max_steps": args.steps == 30000,
            "sh_degree": args.sh_degree == 3,
            "sh_degree_interval": args.sh_degree_interval == 1000,
            "ssim_lambda": args.ssim_lambda == 0.2,
            "init_opacity": args.opacity == 0.1,
            "default_strategy": bool(args.refine_enabled),
        },
        "dataloader": sampler.summary(cameras),
        "loss_config": {
            "mode": "masked_l1_ssim",
            "formula": "(1 - ssim_lambda) * masked_L1 + ssim_lambda * (1 - SSIM(render * mask + target * (1 - mask), target))",
            "ssim_lambda": float(args.ssim_lambda),
            "ssim_window_size": int(args.ssim_window_size),
        },
        "learning_rate_schedule": lr_schedules,
        "initial_mean_loss": float(initial_mean_loss),
        "final_mean_loss": float(final_mean_loss),
        "eval_final_mean_loss": eval_final_mean_loss,
        "last_viewspace_grad_norm": last_viewspace_grad_norm,
        "spz": str(out_spz),
        "spz_file_size_bytes": int(spz_size),
        "spz_export_diagnostics": spz_diagnostics,
        "color_mode": "spherical_harmonics",
        "active_sh_degree_final": int(active_sh_degree),
        "sh_degree_schedule": {
            "start": 0,
            "target": int(args.sh_degree),
            "interval": int(args.sh_degree_interval),
            "formula": "min(step // sh_degree_interval, sh_degree)",
            "events": sh_degree_events,
        },
        "final_opacity_diagnostics": final_opacity_diagnostics,
        "refinement_strategy": refinement_summary,
        "frame_summaries": frame_summaries,
        "eval_frame_summaries": eval_frame_summaries,
    }
    out_model_npz = args.out_model_npz if args.out_model_npz is not None else args.out_dir / "trained_model_params.npz"
    summary["model_npz"] = str(out_model_npz)
    save_model_parameters_npz(out_model_npz, model, "sh", active_sh_degree, args.sh_degree, summary)
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_spz.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(
        "ScanApp masked depth multi-view training ok "
        f"initial_mean_loss={initial_mean_loss:.8f} final_mean_loss={final_mean_loss:.8f} "
        f"last_viewspace_grad_norm={last_viewspace_grad_norm:.8f} "
        f"spz={out_spz} bytes={spz_size} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
