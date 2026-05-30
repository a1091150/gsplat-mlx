#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from gsplat_core import (
    intersect_offset_forward,
    intersect_tile_forward,
    projection_ewa_3dgs_fused_forward,
    rasterize_to_pixels_3dgs_forward,
)
from render_random_3dgs_png import write_png
def image_to_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def normalize_quats(quats: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True))
    return quats / mx.maximum(norm, 1.0e-8)


@dataclass(frozen=True)
class ScannerFrame:
    index: int
    image_path: Path
    json_path: Path


@dataclass(frozen=True)
class ScannerCamera:
    index: int
    viewmat: np.ndarray
    K: np.ndarray
    image_path: Path
    raw_width: int
    raw_height: int


def extract_frame_index(path: Path) -> int | None:
    match = re.search(r"frame_(\d+)\.(?:jpg|json)$", path.name)
    return int(match.group(1)) if match else None


def collect_frames(
    dataset_dir: Path,
    max_frames: int,
    frame_step: int,
    start_index: int,
) -> list[ScannerFrame]:
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
    frames = [ScannerFrame(idx, image_map[idx], json_map[idx]) for idx in common]
    if not frames:
        raise RuntimeError(f"No frame_*.jpg/json pairs found in {dataset_dir}")
    return frames


def axis_transform() -> np.ndarray:
    a4 = np.eye(4, dtype=np.float32)
    a4[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return a4


def scanner_pose_to_viewmat(raw_pose: list[float]) -> np.ndarray:
    c2w_src = np.array(raw_pose, dtype=np.float32).reshape(4, 4)
    c2w = axis_transform() @ c2w_src
    r = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3:4].astype(np.float32)

    # Match the FastGS scanner loader convention before feeding gsplat's
    # row-major world-to-camera viewmat.
    r = r @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    rinv = r.T
    tinv = (-rinv @ t).astype(np.float32)

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = rinv
    viewmat[:3, 3:4] = tinv
    return viewmat


def load_camera(frame: ScannerFrame, width: int, height: int) -> ScannerCamera:
    raw = json.loads(frame.json_path.read_text(encoding="utf-8"))
    intrinsics = raw.get("intrinsics")
    pose = raw.get("cameraPoseARFrame")
    if intrinsics is None or len(intrinsics) != 9:
        raise RuntimeError(f"Invalid intrinsics in {frame.json_path}")
    if pose is None or len(pose) != 16:
        raise RuntimeError(f"Invalid cameraPoseARFrame in {frame.json_path}")

    with Image.open(frame.image_path) as image:
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
    return ScannerCamera(
        index=frame.index,
        viewmat=scanner_pose_to_viewmat(pose),
        K=K,
        image_path=frame.image_path,
        raw_width=raw_width,
        raw_height=raw_height,
    )


def load_target(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def random_gaussians_for_cameras(
    cameras: list[ScannerCamera],
    num_gaussians: int,
    width: int,
    height: int,
    seed: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    per_camera = int(np.ceil(num_gaussians / max(len(cameras), 1)))
    means_world = []
    for camera in cameras:
        inv_view = np.linalg.inv(camera.viewmat).astype(np.float32)
        fx = float(camera.K[0, 0])
        fy = float(camera.K[1, 1])
        z = rng.uniform(1.0, 3.5, size=(per_camera, 1)).astype(np.float32)
        x = rng.uniform(-0.35, 0.35, size=(per_camera, 1)).astype(np.float32)
        y = rng.uniform(-0.35, 0.35, size=(per_camera, 1)).astype(np.float32)
        x = x * (float(width) / max(fx, 1.0)) * z
        y = y * (float(height) / max(fy, 1.0)) * z
        points_c = np.concatenate([x, y, z, np.ones_like(z)], axis=1)
        points_w = (inv_view @ points_c.T).T[:, :3]
        means_world.append(points_w.astype(np.float32))

    means_np = np.concatenate(means_world, axis=0)[:num_gaussians][None, ...]
    quats_np = rng.normal(size=(1, num_gaussians, 4)).astype(np.float32)
    quats_np[..., 0] += 2.0
    scales_np = rng.uniform(0.015, 0.06, size=(1, num_gaussians, 3)).astype(np.float32)
    colors_np = rng.uniform(0.08, 0.95, size=(1, 1, num_gaussians, 3)).astype(np.float32)
    opacities_np = rng.uniform(0.18, 0.65, size=(1, num_gaussians)).astype(np.float32)
    return (
        mx.array(means_np),
        normalize_quats(mx.array(quats_np)),
        mx.array(scales_np),
        mx.array(colors_np),
        mx.array(opacities_np),
    )


def render_random_scene(
    camera: ScannerCamera,
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    colors: mx.array,
    opacities: mx.array,
    width: int,
    height: int,
    tile_size: int,
) -> dict[str, mx.array]:
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    viewmats = mx.array(camera.viewmat[None, None, ...], dtype=mx.float32)
    Ks = mx.array(camera.K[None, None, ...], dtype=mx.float32)
    projection = projection_ewa_3dgs_fused_forward(
        {
            "means": means,
            "quats": quats,
            "scales": scales,
            "viewmats": viewmats,
            "Ks": Ks,
            "viewspace_points": mx.zeros((1, 1, means.shape[1], 2), dtype=mx.float32),
        },
        image_width=width,
        image_height=height,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=0.0,
        calc_compensations=False,
        camera_model=0,
    )
    intersections = intersect_tile_forward(
        {
            "means2d": projection["means2d"],
            "radii": projection["radii"],
            "depths": projection["depths"],
        },
        I=1,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
    )
    tile_offsets = intersect_offset_forward(
        intersections["isect_ids"],
        I=1,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    render = rasterize_to_pixels_3dgs_forward(
        {
            "means2d": projection["means2d"],
            "conics": projection["conics"],
            "colors": colors,
            "opacities": mx.expand_dims(opacities, axis=1),
            "backgrounds": mx.array([[0.02, 0.02, 0.025]], dtype=mx.float32),
            "tile_offsets": tile_offsets,
            "flatten_ids": intersections["flatten_ids"],
        },
        image_width=width,
        image_height=height,
        tile_size=tile_size,
    )
    return {
        **render,
        "radii": projection["radii"],
        "tiles_per_gauss": intersections["tiles_per_gauss"],
        "isect_ids": intersections["isect_ids"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/Users/yangdunfu/Downloads/2026_05_04_16_51_29"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scanner_dataset_random_render"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=120)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-gaussians", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = collect_frames(args.data, args.max_frames, args.frame_step, args.start_index)
    cameras = [load_camera(frame, args.width, args.height) for frame in frames]
    means, quats, scales, colors, opacities = random_gaussians_for_cameras(
        cameras,
        args.num_gaussians,
        args.width,
        args.height,
        args.seed,
    )

    debug = {
        "dataset": str(args.data),
        "width": args.width,
        "height": args.height,
        "num_gaussians": args.num_gaussians,
        "frames": [],
    }
    total_visible = 0
    total_intersections = 0
    for view_id, camera in enumerate(cameras):
        target = load_target(camera.image_path, args.width, args.height)
        write_png(args.out_dir / f"target_frame_{camera.index:05d}.png", image_to_u8(target))
        render = render_random_scene(
            camera,
            means,
            quats,
            scales,
            colors,
            opacities,
            args.width,
            args.height,
            args.tile_size,
        )
        mx.eval(*render.values())
        rgb = np.asarray(render["render_colors"][0], dtype=np.float32)
        radii = np.asarray(render["radii"])
        isect_ids = np.asarray(render["isect_ids"])
        visible = int(np.count_nonzero(np.any(radii > 0, axis=-1)))
        intersections = int(isect_ids.shape[0])
        total_visible += visible
        total_intersections += intersections
        write_png(args.out_dir / f"random_render_frame_{camera.index:05d}.png", image_to_u8(rgb))
        debug["frames"].append(
            {
                "view_id": view_id,
                "frame_index": camera.index,
                "image_path": str(camera.image_path),
                "raw_width": camera.raw_width,
                "raw_height": camera.raw_height,
                "K": camera.K.tolist(),
                "viewmat": camera.viewmat.tolist(),
                "visible_gaussians": visible,
                "intersections": intersections,
            }
        )
        print(
            f"frame={camera.index:05d} visible_gaussians={visible} "
            f"intersections={intersections}"
        )

    (args.out_dir / "debug_camera_metadata.json").write_text(
        json.dumps(debug, indent=2),
        encoding="utf-8",
    )
    if total_visible <= 0 or total_intersections <= 0:
        raise AssertionError("scanner dataset random render expected visible gaussians")
    print(
        "scanner dataset random render smoke ok "
        f"frames={len(cameras)} total_visible={total_visible} "
        f"total_intersections={total_intersections} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
