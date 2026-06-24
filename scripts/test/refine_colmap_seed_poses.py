#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass(frozen=True)
class ColmapPoint3D:
    point_id: int
    xyz: np.ndarray
    color: tuple[int, int, int]
    error: float
    track_length: int


@dataclass(frozen=True)
class ColmapModel:
    cameras: dict[int, ColmapCamera]
    images: dict[str, ColmapImage]
    points3d: dict[int, ColmapPoint3D]


def log(message: str) -> None:
    print(message, flush=True)


def parse_bool_int(value: bool) -> str:
    return "1" if value else "0"


def resolve_model_root(data: Path) -> Path:
    data = data.expanduser().resolve()
    if (data / "sparse" / "0" / "cameras.txt").exists() and (data / "images").exists():
        return data
    candidate = data / "COLMAP_Text_Model"
    if (candidate / "sparse" / "0" / "cameras.txt").exists() and (candidate / "images").exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find a COLMAP text model in {data}. Expected images/ and sparse/0/*.txt."
    )


def sparse0_path(model_root: Path) -> Path:
    path = model_root / "sparse" / "0"
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (path / filename).exists():
            raise FileNotFoundError(f"Missing {path / filename}")
    return path


def read_non_comment_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def read_cameras(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in read_non_comment_lines(path):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid cameras.txt line: {line}")
        camera_id = int(parts[0])
        cameras[camera_id] = ColmapCamera(
            camera_id=camera_id,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=[float(item) for item in parts[4:]],
        )
    return cameras


def read_images(path: Path) -> dict[str, ColmapImage]:
    images: dict[str, ColmapImage] = {}
    raw_lines = path.read_text().splitlines()
    index = 0
    while index < len(raw_lines):
        line = raw_lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Invalid images.txt pose line: {line}")
        image = ColmapImage(
            image_id=int(parts[0]),
            qvec=np.array([float(item) for item in parts[1:5]], dtype=np.float64),
            tvec=np.array([float(item) for item in parts[5:8]], dtype=np.float64),
            camera_id=int(parts[8]),
            name=" ".join(parts[9:]),
        )
        images[image.name] = image
        if index < len(raw_lines):
            index += 1
    return images


def read_points3d(path: Path) -> dict[int, ColmapPoint3D]:
    points: dict[int, ColmapPoint3D] = {}
    for line in read_non_comment_lines(path):
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(f"Invalid points3D.txt line: {line}")
        point_id = int(parts[0])
        track_items = parts[8:]
        points[point_id] = ColmapPoint3D(
            point_id=point_id,
            xyz=np.array([float(item) for item in parts[1:4]], dtype=np.float64),
            color=(int(parts[4]), int(parts[5]), int(parts[6])),
            error=float(parts[7]),
            track_length=len(track_items) // 2,
        )
    return points


def read_model(sparse_dir: Path) -> ColmapModel:
    return ColmapModel(
        cameras=read_cameras(sparse_dir / "cameras.txt"),
        images=read_images(sparse_dir / "images.txt"),
        points3d=read_points3d(sparse_dir / "points3D.txt"),
    )


def read_database_image_ids(database_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(database_path)
    try:
        return {
            str(name): int(image_id)
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }
    finally:
        connection.close()


def write_seed_model_with_database_image_ids(seed_sparse: Path, out_sparse: Path, database_path: Path) -> Path:
    image_ids = read_database_image_ids(database_path)
    seed_images = read_images(seed_sparse / "images.txt")
    missing = sorted(set(seed_images) - set(image_ids))
    if missing:
        raise RuntimeError(f"Seed model images missing from COLMAP database: {missing[:8]}")
    out_sparse.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed_sparse / "cameras.txt", out_sparse / "cameras.txt")
    shutil.copy2(seed_sparse / "points3D.txt", out_sparse / "points3D.txt")
    lines = [
        "# Rewritten by refine_colmap_seed_poses.py to match database image_id values.",
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(seed_images)}",
    ]
    for image in sorted(seed_images.values(), key=lambda item: image_ids[item.name]):
        lines.append(
            f"{image_ids[image.name]} "
            f"{image.qvec[0]:.12f} {image.qvec[1]:.12f} {image.qvec[2]:.12f} {image.qvec[3]:.12f} "
            f"{image.tvec[0]:.12f} {image.tvec[1]:.12f} {image.tvec[2]:.12f} "
            f"{image.camera_id} {image.name}"
        )
        lines.append("")
    (out_sparse / "images.txt").write_text("\n".join(lines) + "\n")
    return out_sparse


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


def write_training_colmap_model(
    refined_sparse_txt: Path,
    image_path: Path,
    out_model_root: Path,
    copy_images: bool,
) -> dict:
    sparse_out = out_model_root / "sparse" / "0"
    if out_model_root.exists():
        shutil.rmtree(out_model_root)
    sparse_out.mkdir(parents=True, exist_ok=True)
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy2(refined_sparse_txt / filename, sparse_out / filename)
    image_mode = link_or_copy_images(image_path, out_model_root / "images", copy_images)
    return {
        "path": str(out_model_root),
        "sparse_path": str(sparse_out),
        "image_path": str(out_model_root / "images"),
        "image_mode": image_mode,
    }


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = qvec.astype(np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("COLMAP qvec has zero norm")
    qw, qx, qy, qz = q / norm
    return np.array(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw],
            [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw],
            [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def camera_center(image: ColmapImage) -> np.ndarray:
    rotation = qvec_to_rotmat(image.qvec)
    return -rotation.T @ image.tvec


def rotation_delta_deg(a: ColmapImage, b: ColmapImage) -> float:
    ra = qvec_to_rotmat(a.qvec)
    rb = qvec_to_rotmat(b.qvec)
    delta = rb @ ra.T
    cos_angle = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "p95": None, "max": None}
    array = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def point_stats(points: dict[int, ColmapPoint3D]) -> dict:
    errors = [point.error for point in points.values() if point.error >= 0.0]
    track_lengths = [point.track_length for point in points.values()]
    tracked = [length for length in track_lengths if length > 0]
    return {
        "point_count": len(points),
        "error": stats(errors),
        "track_length": stats([float(item) for item in track_lengths]),
        "tracked_point_count": len(tracked),
        "untracked_point_count": len(points) - len(tracked),
    }


def camera_summary(cameras: dict[int, ColmapCamera]) -> list[dict]:
    return [
        {
            "camera_id": camera.camera_id,
            "model": camera.model,
            "width": camera.width,
            "height": camera.height,
            "params": camera.params,
        }
        for camera in sorted(cameras.values(), key=lambda item: item.camera_id)
    ]


def pose_diagnostics(seed: ColmapModel, refined: ColmapModel) -> dict:
    common_names = sorted(set(seed.images) & set(refined.images))
    translation_deltas = []
    rotation_deltas = []
    per_image = []
    for name in common_names:
        seed_image = seed.images[name]
        refined_image = refined.images[name]
        seed_center = camera_center(seed_image)
        refined_center = camera_center(refined_image)
        translation_delta = float(np.linalg.norm(refined_center - seed_center))
        rotation_delta = rotation_delta_deg(seed_image, refined_image)
        translation_deltas.append(translation_delta)
        rotation_deltas.append(rotation_delta)
        per_image.append(
            {
                "name": name,
                "seed_image_id": int(seed_image.image_id),
                "refined_image_id": int(refined_image.image_id),
                "translation_delta_m": translation_delta,
                "rotation_delta_deg": rotation_delta,
                "seed_center": seed_center.astype(float).tolist(),
                "refined_center": refined_center.astype(float).tolist(),
            }
        )
    return {
        "common_image_count": len(common_names),
        "seed_only_images": sorted(set(seed.images) - set(refined.images)),
        "refined_only_images": sorted(set(refined.images) - set(seed.images)),
        "translation_delta_m": stats(translation_deltas),
        "rotation_delta_deg": stats(rotation_deltas),
        "largest_translation_deltas": sorted(per_image, key=lambda item: item["translation_delta_m"], reverse=True)[:10],
        "largest_rotation_deltas": sorted(per_image, key=lambda item: item["rotation_delta_deg"], reverse=True)[:10],
    }


def run_command(command: list[str], dry_run: bool) -> dict:
    log("$ " + " ".join(command))
    if dry_run:
        return {"command": command, "returncode": None, "dry_run": True}
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")
    return {"command": command, "returncode": int(completed.returncode), "dry_run": False}


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in [
        out_dir / "database.db",
        out_dir / "sparse_seed_database_ids",
        out_dir / "sparse_triangulated",
        out_dir / "sparse_triangulated_txt",
        out_dir / "sparse_ba",
        out_dir / "sparse_ba_txt",
        out_dir / "refined_colmap_text_model",
    ]:
        if not child.exists():
            continue
        if not overwrite:
            raise FileExistsError(f"{child} already exists. Pass --overwrite to replace generated COLMAP outputs.")
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (out_dir / "sparse_triangulated").mkdir(parents=True, exist_ok=True)
    (out_dir / "sparse_triangulated_txt").mkdir(parents=True, exist_ok=True)
    (out_dir / "sparse_ba").mkdir(parents=True, exist_ok=True)
    (out_dir / "sparse_ba_txt").mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine a COLMAP text seed model with COLMAP feature matching, point triangulation, and bundle adjustment."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to a COLMAP_Text_Model folder or its parent SplatKing folder.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--colmap-bin", type=str, default="colmap")
    parser.add_argument("--matcher", choices=("exhaustive", "sequential"), default="exhaustive")
    parser.add_argument("--sequential-overlap", type=int, default=10)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refine-intrinsics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera-model", type=str, default=None)
    parser.add_argument("--camera-params", type=str, default=None)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = resolve_model_root(args.data)
    image_path = model_root / "images"
    seed_sparse = sparse0_path(model_root)
    seed_model = read_model(seed_sparse)
    if len(seed_model.cameras) != 1 and (args.camera_model is None or args.camera_params is None):
        raise ValueError("Multiple seed cameras found; pass --camera-model and --camera-params explicitly.")
    seed_camera = next(iter(seed_model.cameras.values()))
    camera_model = args.camera_model or seed_camera.model
    camera_params = args.camera_params or ",".join(f"{value:.12g}" for value in seed_camera.params)

    prepare_output(args.out_dir, args.overwrite)
    commands = []
    database_path = args.out_dir / "database.db"
    rewritten_seed_sparse = args.out_dir / "sparse_seed_database_ids"
    triangulation_input_sparse = seed_sparse
    gpu = parse_bool_int(bool(args.use_gpu))

    log(f"model_root={model_root}")
    log(f"seed_sparse={seed_sparse}")
    log(f"images={image_path}")
    log(f"out_dir={args.out_dir}")
    log(f"camera={camera_model} {camera_params}")

    commands.append(
        run_command(
            [
                args.colmap_bin,
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_path),
                "--ImageReader.camera_model",
                camera_model,
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_params",
                camera_params,
                "--FeatureExtraction.use_gpu",
                gpu,
            ],
            args.dry_run,
        )
    )
    if not args.dry_run:
        triangulation_input_sparse = write_seed_model_with_database_image_ids(
            seed_sparse,
            rewritten_seed_sparse,
            database_path,
        )
        log(f"rewrote seed sparse model with database image ids: {triangulation_input_sparse}")
    if args.matcher == "exhaustive":
        match_command = [
            args.colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--FeatureMatching.use_gpu",
            gpu,
        ]
    else:
        match_command = [
            args.colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(database_path),
            "--FeatureMatching.use_gpu",
            gpu,
            "--SequentialMatching.overlap",
            str(args.sequential_overlap),
        ]
    commands.append(run_command(match_command, args.dry_run))
    commands.append(
        run_command(
            [
                args.colmap_bin,
                "point_triangulator",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_path),
                "--input_path",
                str(triangulation_input_sparse),
                "--output_path",
                str(args.out_dir / "sparse_triangulated"),
                "--refine_intrinsics",
                parse_bool_int(bool(args.refine_intrinsics)),
            ],
            args.dry_run,
        )
    )
    commands.append(
        run_command(
            [
                args.colmap_bin,
                "model_converter",
                "--input_path",
                str(args.out_dir / "sparse_triangulated"),
                "--output_path",
                str(args.out_dir / "sparse_triangulated_txt"),
                "--output_type",
                "TXT",
            ],
            args.dry_run,
        )
    )
    commands.append(
        run_command(
            [
                args.colmap_bin,
                "bundle_adjuster",
                "--input_path",
                str(args.out_dir / "sparse_triangulated"),
                "--output_path",
                str(args.out_dir / "sparse_ba"),
                "--BundleAdjustment.refine_focal_length",
                parse_bool_int(bool(args.refine_intrinsics)),
                "--BundleAdjustment.refine_principal_point",
                parse_bool_int(bool(args.refine_intrinsics)),
                "--BundleAdjustment.refine_extra_params",
                "0",
            ],
            args.dry_run,
        )
    )
    commands.append(
        run_command(
            [
                args.colmap_bin,
                "model_converter",
                "--input_path",
                str(args.out_dir / "sparse_ba"),
                "--output_path",
                str(args.out_dir / "sparse_ba_txt"),
                "--output_type",
                "TXT",
            ],
            args.dry_run,
        )
    )

    summary = {
        "dataset_type": "colmap_seed_pose_refinement",
        "model_root": str(model_root),
        "seed_sparse": str(seed_sparse),
        "rewritten_seed_sparse": str(rewritten_seed_sparse) if not args.dry_run else None,
        "triangulation_input_sparse": str(triangulation_input_sparse),
        "image_path": str(image_path),
        "out_dir": str(args.out_dir),
        "database_path": str(database_path),
        "matcher": args.matcher,
        "use_gpu": bool(args.use_gpu),
        "refine_intrinsics": bool(args.refine_intrinsics),
        "camera_model": camera_model,
        "camera_params": camera_params,
        "commands": commands,
        "seed": {
            "camera_count": len(seed_model.cameras),
            "image_count": len(seed_model.images),
            "point_stats": point_stats(seed_model.points3d),
            "cameras": camera_summary(seed_model.cameras),
        },
    }
    if not args.dry_run:
        triangulated_model = read_model(args.out_dir / "sparse_triangulated_txt")
        refined_model = read_model(args.out_dir / "sparse_ba_txt")
        training_model = write_training_colmap_model(
            args.out_dir / "sparse_ba_txt",
            image_path,
            args.out_dir / "refined_colmap_text_model",
            args.copy_images,
        )
        summary["triangulated"] = {
            "camera_count": len(triangulated_model.cameras),
            "image_count": len(triangulated_model.images),
            "point_stats": point_stats(triangulated_model.points3d),
            "cameras": camera_summary(triangulated_model.cameras),
        }
        summary["refined"] = {
            "camera_count": len(refined_model.cameras),
            "image_count": len(refined_model.images),
            "point_stats": point_stats(refined_model.points3d),
            "cameras": camera_summary(refined_model.cameras),
        }
        summary["pose_delta_seed_to_refined"] = pose_diagnostics(seed_model, refined_model)
        summary["refined_training_model"] = training_model
        summary["suggested_training_command"] = (
            "make codex-360-points-train-spz "
            f"COLMAP_360_DATA={training_model['path']} "
            f"COLMAP_360_OUT={args.out_dir / 'refined_3dgs_train'} "
            "COLMAP_360_FACTOR=1 COLMAP_360_MAX_POINTS=0"
        )

    out_json = args.out_dir / "pose_diagnostics.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log(f"wrote diagnostics {out_json}")
    if not args.dry_run:
        point_count = summary["refined"]["point_stats"]["point_count"]
        pose_stats = summary["pose_delta_seed_to_refined"]
        log(
            "COLMAP seed pose refinement ok "
            f"refined_points={point_count} "
            f"translation_delta_mean={pose_stats['translation_delta_m']['mean']} "
            f"rotation_delta_mean_deg={pose_stats['rotation_delta_deg']['mean']} "
            f"training_model={summary['refined_training_model']['path']}"
        )


if __name__ == "__main__":
    main()
