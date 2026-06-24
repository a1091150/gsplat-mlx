#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScanAppFrame:
    index: int
    name: str
    image_rel: str
    image_path: Path
    metadata_path: Path
    width: int
    height: int
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    motion_quality: float | None


def log(message: str) -> None:
    print(message, flush=True)


def require_array(raw: dict, key: str, count: int, metadata_path: Path) -> np.ndarray:
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise RuntimeError(f"Invalid {key} in {metadata_path}")
    return np.asarray(value, dtype=np.float64)


def scanner_axis3x3() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def scanner_axis4x4() -> np.ndarray:
    axis = np.eye(4, dtype=np.float64)
    axis[:3, :3] = scanner_axis3x3()
    return axis


def inverse_rigid4x4(c2w: np.ndarray) -> np.ndarray:
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3:4]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rotation.T
    w2c[:3, 3:4] = -rotation.T @ translation
    return w2c


def scanapp_c2w_to_colmap_w2c(raw_camera_to_world: np.ndarray) -> np.ndarray:
    c2w = scanner_axis4x4() @ raw_camera_to_world.astype(np.float64)
    c2w[:3, :3] = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    return inverse_rigid4x4(c2w)


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    k = np.array(
        [
            [matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0, 0.0, 0.0],
            [matrix[1, 0] + matrix[0, 1], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0, 0.0],
            [matrix[2, 0] + matrix[0, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0],
            [matrix[1, 2] - matrix[2, 1], matrix[2, 0] - matrix[0, 2], matrix[0, 1] - matrix[1, 0], matrix[0, 0] + matrix[1, 1] + matrix[2, 2]],
        ],
        dtype=np.float64,
    )
    k /= 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    if qvec[0] < 0.0:
        qvec *= -1.0
    return qvec / np.linalg.norm(qvec)


def load_scanapp_frames(data_dir: Path) -> list[ScanAppFrame]:
    metadata_dir = data_dir / "metadata"
    if not metadata_dir.exists():
        raise FileNotFoundError(f"ScanApp metadata directory not found: {metadata_dir}")
    frames: list[ScanAppFrame] = []
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        image_rel = raw.get("image")
        if not isinstance(image_rel, str):
            continue
        width = int(raw.get("width", 0))
        height = int(raw.get("height", 0))
        if width <= 0 or height <= 0:
            continue
        image_path = data_dir / image_rel
        if not image_path.exists():
            continue
        intrinsics = require_array(raw, "intrinsics", 9, metadata_path).reshape(3, 3)
        camera_to_world = require_array(raw, "camera_to_world", 16, metadata_path).reshape(4, 4)
        world_to_camera = require_array(raw, "world_to_camera", 16, metadata_path).reshape(4, 4)
        motion_quality_raw = raw.get("motionQuality")
        motion_quality = float(motion_quality_raw) if isinstance(motion_quality_raw, (int, float)) else None
        frames.append(
            ScanAppFrame(
                index=int(raw.get("frame_index", len(frames))),
                name=str(raw.get("frame_name", metadata_path.stem)),
                image_rel=image_rel,
                image_path=image_path,
                metadata_path=metadata_path,
                width=width,
                height=height,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                world_to_camera=world_to_camera,
                motion_quality=motion_quality,
            )
        )
    frames.sort(key=lambda frame: (frame.index, frame.metadata_path.name))
    if not frames:
        raise RuntimeError(f"No usable ScanApp frames found in {metadata_dir}")
    return frames


def select_frames(frames: list[ScanAppFrame], max_frames: int, frame_step: int, start_index: int) -> list[ScanAppFrame]:
    if frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    selected = [frame for frame in frames if frame.index >= start_index]
    if frame_step > 1:
        selected = selected[::frame_step]
    if max_frames > 0:
        selected = selected[:max_frames]
    if not selected:
        raise RuntimeError("No ScanApp frames selected")
    return selected


def median_intrinsics(frames: list[ScanAppFrame]) -> np.ndarray:
    return np.median(np.stack([frame.intrinsics for frame in frames], axis=0), axis=0)


def camera_params_for_frame(frame: ScanAppFrame, shared_intrinsics: np.ndarray | None) -> tuple[float, float, float, float]:
    intrinsics = shared_intrinsics if shared_intrinsics is not None else frame.intrinsics
    return (
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2]),
        float(intrinsics[1, 2]),
    )


def link_or_copy_images(source: Path, target: Path, copy_images: bool) -> str:
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    if copy_images:
        shutil.copytree(source, target)
        return "copy"
    try:
        target.symlink_to(source, target_is_directory=True)
        return "symlink"
    except OSError:
        shutil.copytree(source, target)
        return "copy_fallback"


def write_cameras(path: Path, frames: list[ScanAppFrame], shared_intrinsics: np.ndarray | None) -> dict[int, tuple[ScanAppFrame, tuple[float, float, float, float]]]:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# Number of cameras: {}".format(1 if shared_intrinsics is not None else len(frames)),
    ]
    camera_map: dict[int, tuple[ScanAppFrame, tuple[float, float, float, float]]] = {}
    if shared_intrinsics is not None:
        first = frames[0]
        params = camera_params_for_frame(first, shared_intrinsics)
        camera_map[1] = (first, params)
        lines.append(
            "1 PINHOLE "
            f"{first.width} {first.height} "
            f"{params[0]:.12f} {params[1]:.12f} {params[2]:.12f} {params[3]:.12f}"
        )
    else:
        for camera_id, frame in enumerate(frames, start=1):
            params = camera_params_for_frame(frame, None)
            camera_map[camera_id] = (frame, params)
            lines.append(
                f"{camera_id} PINHOLE "
                f"{frame.width} {frame.height} "
                f"{params[0]:.12f} {params[1]:.12f} {params[2]:.12f} {params[3]:.12f}"
            )
    path.write_text("\n".join(lines) + "\n")
    return camera_map


def write_images(path: Path, frames: list[ScanAppFrame], shared_intrinsics: np.ndarray | None) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(frames)}, mean observations per image: 0",
    ]
    for image_id, frame in enumerate(frames, start=1):
        w2c = scanapp_c2w_to_colmap_w2c(frame.camera_to_world)
        qvec = rotmat_to_qvec(w2c[:3, :3])
        tvec = w2c[:3, 3]
        camera_id = 1 if shared_intrinsics is not None else image_id
        image_name = Path(frame.image_rel).name
        lines.append(
            f"{image_id} "
            f"{qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
            f"{tvec[0]:.12f} {tvec[1]:.12f} {tvec[2]:.12f} "
            f"{camera_id} {image_name}"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def write_empty_points3d(path: Path) -> None:
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        "# Number of points: 0, mean track length: 0",
    ]
    path.write_text("\n".join(lines) + "\n")


def pose_extent(frames: list[ScanAppFrame]) -> dict:
    centers = []
    for frame in frames:
        c2w = scanner_axis4x4() @ frame.camera_to_world.astype(np.float64)
        centers.append(c2w[:3, 3])
    array = np.stack(centers, axis=0)
    center = np.mean(array, axis=0)
    distances = np.linalg.norm(array - center[None, :], axis=1)
    return {
        "bbox_min": np.min(array, axis=0).astype(float).tolist(),
        "bbox_max": np.max(array, axis=0).astype(float).tolist(),
        "center": center.astype(float).tolist(),
        "radius_max": float(np.max(distances)),
        "radius_median": float(np.median(distances)),
    }


def scalar_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a ScanApp recording as a COLMAP text seed model.")
    parser.add_argument("--data", type=Path, required=True, help="ScanApp dataset root containing images/ and metadata/.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shared-intrinsics", choices=("median", "none"), default="median")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    out_dir = args.out_dir
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)
    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    all_frames = load_scanapp_frames(data_dir)
    frames = select_frames(all_frames, args.max_frames, args.frame_step, args.start_index)
    shared_intrinsics = median_intrinsics(frames) if args.shared_intrinsics == "median" else None
    image_mode = link_or_copy_images(data_dir / "images", out_dir / "images", args.copy_images)

    camera_map = write_cameras(sparse_dir / "cameras.txt", frames, shared_intrinsics)
    write_images(sparse_dir / "images.txt", frames, shared_intrinsics)
    write_empty_points3d(sparse_dir / "points3D.txt")

    fx_values = [float(frame.intrinsics[0, 0]) for frame in frames]
    fy_values = [float(frame.intrinsics[1, 1]) for frame in frames]
    cx_values = [float(frame.intrinsics[0, 2]) for frame in frames]
    cy_values = [float(frame.intrinsics[1, 2]) for frame in frames]
    motion_values = [float(frame.motion_quality) for frame in frames if frame.motion_quality is not None and math.isfinite(frame.motion_quality)]
    summary = {
        "dataset_type": "scanapp_colmap_seed_model",
        "data": str(data_dir),
        "out_dir": str(out_dir),
        "sparse_dir": str(sparse_dir),
        "image_path": str(out_dir / "images"),
        "image_mode": image_mode,
        "input_frame_count": len(all_frames),
        "exported_frame_count": len(frames),
        "shared_intrinsics": args.shared_intrinsics,
        "camera_count": len(camera_map),
        "camera_model": "PINHOLE",
        "width": int(frames[0].width),
        "height": int(frames[0].height),
        "median_intrinsics": shared_intrinsics.astype(float).tolist() if shared_intrinsics is not None else None,
        "intrinsics_stats": {
            "fx": scalar_stats(fx_values),
            "fy": scalar_stats(fy_values),
            "cx": scalar_stats(cx_values),
            "cy": scalar_stats(cy_values),
        },
        "motion_quality": scalar_stats(motion_values),
        "pose_extent": pose_extent(frames),
        "frame_indices": [int(frame.index) for frame in frames],
        "image_names": [Path(frame.image_rel).name for frame in frames],
        "points3d_seed_count": 0,
        "notes": [
            "poses use the same ScanApp-to-gsplat axis conversion as existing ScanApp trainers",
            "points3D is intentionally empty; COLMAP point_triangulator should create SfM points after matching",
        ],
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log(
        "ScanApp COLMAP seed export ok "
        f"frames={len(frames)} cameras={len(camera_map)} "
        f"shared_intrinsics={args.shared_intrinsics} out={out_dir}"
    )


if __name__ == "__main__":
    main()
