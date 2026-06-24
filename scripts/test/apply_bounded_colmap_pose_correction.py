#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from refine_colmap_seed_poses import (
    ColmapImage,
    ColmapModel,
    camera_center,
    point_stats,
    read_model,
    sparse0_path,
    stats,
)
from export_scanapp_colmap_seed_model import rotmat_to_qvec


def log(message: str) -> None:
    print(message, flush=True)


def resolve_sparse_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir() and (path / "cameras.txt").exists() and (path / "images.txt").exists():
        return path
    return sparse0_path(path)


def resolve_image_path(seed_model_root_or_sparse: Path, explicit_images: Path | None) -> Path | None:
    if explicit_images is not None:
        return explicit_images.expanduser().resolve()
    path = seed_model_root_or_sparse.expanduser().resolve()
    if (path / "images").exists():
        return path / "images"
    if path.name == "0" and path.parent.name == "sparse":
        candidate = path.parent.parent / "images"
        if candidate.exists():
            return candidate
    return None


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = qvec.astype(np.float64)
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw],
            [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw],
            [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def image_w2c(image: ColmapImage) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec_to_rotmat(image.qvec)
    w2c[:3, 3] = image.tvec
    return w2c


def image_c2w(image: ColmapImage) -> np.ndarray:
    return np.linalg.inv(image_w2c(image))


def rotmat_log(rotation: np.ndarray) -> np.ndarray:
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(angle)
    return axis * angle


def rotmat_exp(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1.0e-10:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def clamp_vector(vector: np.ndarray, max_norm: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vector))
    if max_norm <= 0.0:
        return np.zeros_like(vector), norm > 0.0
    if norm <= max_norm:
        return vector, False
    return vector * (max_norm / norm), True


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    smoothed = np.empty_like(values)
    for index in range(values.shape[0]):
        start = max(0, index - radius)
        end = min(values.shape[0], index + radius + 1)
        smoothed[index] = np.mean(values[start:end], axis=0)
    return smoothed


def read_image_points2d_lines(path: Path) -> dict[str, str]:
    lines_by_name: dict[str, str] = {}
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
        name = " ".join(parts[9:])
        points_line = raw_lines[index].strip() if index < len(raw_lines) else ""
        index += 1
        lines_by_name[name] = points_line
    return lines_by_name


def write_images(
    path: Path,
    seed_images: list[ColmapImage],
    output_images_by_name: dict[str, ColmapImage],
    points2d_lines_by_name: dict[str, str],
    corrected_w2c: list[np.ndarray],
) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(seed_images)}, mean observations per image: 0",
    ]
    for image, w2c in zip(seed_images, corrected_w2c, strict=True):
        output_image = output_images_by_name[image.name]
        qvec = rotmat_to_qvec(w2c[:3, :3])
        tvec = w2c[:3, 3]
        lines.append(
            f"{output_image.image_id} "
            f"{qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
            f"{tvec[0]:.12f} {tvec[1]:.12f} {tvec[2]:.12f} "
            f"{image.camera_id} {image.name}"
        )
        lines.append(points2d_lines_by_name.get(image.name, ""))
    path.write_text("\n".join(lines) + "\n")


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


def copy_points(source_model: ColmapModel, source_sparse: Path, points_source: str, target_sparse: Path) -> dict:
    if points_source == "empty":
        (target_sparse / "points3D.txt").write_text(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0, mean track length: 0\n"
        )
        return point_stats({})
    shutil.copy2(source_sparse / "points3D.txt", target_sparse / "points3D.txt")
    return point_stats(source_model.points3d)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply bounded, smoothed COLMAP pose corrections to an ARKit/ScanApp COLMAP seed model."
    )
    parser.add_argument("--seed-model", type=Path, required=True, help="Seed COLMAP model root or sparse/0 directory.")
    parser.add_argument("--teacher-model", type=Path, required=True, help="Refined COLMAP model root or sparse/0 directory.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, default=None, help="Optional image folder for the output model.")
    parser.add_argument("--max-rotation-deg", type=float, default=3.0)
    parser.add_argument("--max-translation-m", type=float, default=0.03)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--points-source", choices=("teacher", "seed", "empty"), default="teacher")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_rotation_deg < 0.0:
        raise ValueError("--max-rotation-deg must be nonnegative")
    if args.max_translation_m < 0.0:
        raise ValueError("--max-translation-m must be nonnegative")

    seed_sparse = resolve_sparse_path(args.seed_model)
    teacher_sparse = resolve_sparse_path(args.teacher_model)
    image_path = resolve_image_path(args.seed_model, args.image_path)
    if image_path is None or not image_path.exists():
        raise FileNotFoundError("Could not resolve images folder; pass --image-path explicitly.")

    out_dir = args.out_dir
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)
    target_sparse = out_dir / "sparse" / "0"
    target_sparse.mkdir(parents=True, exist_ok=True)

    seed_model = read_model(seed_sparse)
    teacher_model = read_model(teacher_sparse)
    missing = sorted(set(seed_model.images) - set(teacher_model.images))
    if missing:
        raise RuntimeError(f"Teacher model is missing seed images: {missing[:8]}")
    seed_images = sorted(seed_model.images.values(), key=lambda image: image.image_id)

    raw_rotvecs = []
    raw_translations = []
    raw_rotation_degrees = []
    raw_translation_norms = []
    rotation_clamped = 0
    translation_clamped = 0
    max_rotation_rad = np.deg2rad(args.max_rotation_deg)
    for seed_image in seed_images:
        teacher_image = teacher_model.images[seed_image.name]
        seed_c2w = image_c2w(seed_image)
        teacher_c2w = image_c2w(teacher_image)
        delta_rotation = teacher_c2w[:3, :3] @ seed_c2w[:3, :3].T
        rotvec = rotmat_log(delta_rotation)
        translation = camera_center(teacher_image) - camera_center(seed_image)
        raw_rotation_degrees.append(float(np.rad2deg(np.linalg.norm(rotvec))))
        raw_translation_norms.append(float(np.linalg.norm(translation)))
        rotvec, did_clamp_rotation = clamp_vector(rotvec, max_rotation_rad)
        translation, did_clamp_translation = clamp_vector(translation, args.max_translation_m)
        rotation_clamped += int(did_clamp_rotation)
        translation_clamped += int(did_clamp_translation)
        raw_rotvecs.append(rotvec)
        raw_translations.append(translation)

    clamped_rotvecs = np.stack(raw_rotvecs, axis=0)
    clamped_translations = np.stack(raw_translations, axis=0)
    smoothed_rotvecs = moving_average(clamped_rotvecs, args.smooth_window)
    smoothed_translations = moving_average(clamped_translations, args.smooth_window)

    corrected_w2c = []
    corrected_rotation_degrees = []
    corrected_translation_norms = []
    for seed_image, rotvec, translation in zip(seed_images, smoothed_rotvecs, smoothed_translations, strict=True):
        seed_c2w = image_c2w(seed_image)
        corrected_c2w = seed_c2w.copy()
        corrected_c2w[:3, :3] = rotmat_exp(rotvec) @ seed_c2w[:3, :3]
        corrected_c2w[:3, 3] = seed_c2w[:3, 3] + translation
        corrected_rotation_degrees.append(float(np.rad2deg(np.linalg.norm(rotvec))))
        corrected_translation_norms.append(float(np.linalg.norm(translation)))
        corrected_w2c.append(np.linalg.inv(corrected_c2w))

    if args.points_source == "teacher":
        output_images_by_name = teacher_model.images
        points2d_lines_by_name = read_image_points2d_lines(teacher_sparse / "images.txt")
        points_stats = copy_points(teacher_model, teacher_sparse, "teacher", target_sparse)
    elif args.points_source == "seed":
        output_images_by_name = seed_model.images
        points2d_lines_by_name = read_image_points2d_lines(seed_sparse / "images.txt")
        points_stats = copy_points(seed_model, seed_sparse, "seed", target_sparse)
    else:
        output_images_by_name = seed_model.images
        points2d_lines_by_name = {}
        points_stats = copy_points(seed_model, seed_sparse, "empty", target_sparse)
    shutil.copy2(seed_sparse / "cameras.txt", target_sparse / "cameras.txt")
    write_images(target_sparse / "images.txt", seed_images, output_images_by_name, points2d_lines_by_name, corrected_w2c)
    image_mode = link_or_copy_images(image_path, out_dir / "images", args.copy_images)

    largest_raw_rotation = sorted(
        [
            {
                "name": image.name,
                "raw_rotation_delta_deg": raw_rotation_degrees[index],
                "raw_translation_delta_m": raw_translation_norms[index],
                "corrected_rotation_delta_deg": corrected_rotation_degrees[index],
                "corrected_translation_delta_m": corrected_translation_norms[index],
            }
            for index, image in enumerate(seed_images)
        ],
        key=lambda item: item["raw_rotation_delta_deg"],
        reverse=True,
    )[:10]
    largest_raw_translation = sorted(
        [
            {
                "name": image.name,
                "raw_rotation_delta_deg": raw_rotation_degrees[index],
                "raw_translation_delta_m": raw_translation_norms[index],
                "corrected_rotation_delta_deg": corrected_rotation_degrees[index],
                "corrected_translation_delta_m": corrected_translation_norms[index],
            }
            for index, image in enumerate(seed_images)
        ],
        key=lambda item: item["raw_translation_delta_m"],
        reverse=True,
    )[:10]
    summary = {
        "dataset_type": "bounded_colmap_pose_correction",
        "seed_sparse": str(seed_sparse),
        "teacher_sparse": str(teacher_sparse),
        "out_dir": str(out_dir),
        "sparse_dir": str(target_sparse),
        "image_path": str(out_dir / "images"),
        "image_mode": image_mode,
        "image_count": len(seed_images),
        "camera_count": len(seed_model.cameras),
        "points_source": args.points_source,
        "point_stats": points_stats,
        "max_rotation_deg": float(args.max_rotation_deg),
        "max_translation_m": float(args.max_translation_m),
        "smooth_window": int(args.smooth_window),
        "raw_teacher_delta": {
            "rotation_delta_deg": stats(raw_rotation_degrees),
            "translation_delta_m": stats(raw_translation_norms),
        },
        "bounded_smoothed_delta": {
            "rotation_delta_deg": stats(corrected_rotation_degrees),
            "translation_delta_m": stats(corrected_translation_norms),
        },
        "clamped": {
            "rotation_count": int(rotation_clamped),
            "translation_count": int(translation_clamped),
        },
        "largest_raw_rotation_deltas": largest_raw_rotation,
        "largest_raw_translation_deltas": largest_raw_translation,
    }
    (out_dir / "pose_correction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log(
        "bounded COLMAP pose correction ok "
        f"images={len(seed_images)} "
        f"raw_rot_mean={summary['raw_teacher_delta']['rotation_delta_deg']['mean']} "
        f"bounded_rot_mean={summary['bounded_smoothed_delta']['rotation_delta_deg']['mean']} "
        f"out={out_dir}"
    )


if __name__ == "__main__":
    main()
