#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BG = np.array([0.025, 0.025, 0.025], dtype=np.float32)
SH_C0 = 0.28209479177387814
mx = None
Adam = None
Tiny3DGSModel = None
ScannerPointsSHModel = None
active_sh_degree_for_step = None
normalize_quats = None
render_model = None
render_sh_model = None
sh_coeff_count = None


@dataclass
class FixedCamera:
    index: int
    viewmat: np.ndarray
    K: np.ndarray
    position: np.ndarray
    target: np.ndarray


def write_png(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("write_png expects uint8 RGB image with shape [H, W, 3].")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )


def image_to_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1.0e-5, 1.0 - 1.0e-5)
    return np.log(values / (1.0 - values)).astype(np.float32)


def load_training_deps() -> None:
    global Adam, ScannerPointsSHModel, active_sh_degree_for_step, mx, normalize_quats, render_sh_model, sh_coeff_count
    import mlx.core as mlx_core
    from mlx.optimizers import Adam as MlxAdam
    from train_tiny_3dgs_mlx import normalize_quats as tiny_normalize_quats
    from train_scanner_points_multiview_3dgs_mlx import ScannerPointsSHModel as ScannerSHModel
    from train_scanner_points_multiview_3dgs_mlx import active_sh_degree_for_step as scanner_active_sh_degree_for_step
    from train_scanner_points_multiview_3dgs_mlx import render_sh_model as scanner_render_sh_model
    from train_scanner_points_multiview_3dgs_mlx import sh_coeff_count as scanner_sh_coeff_count

    mx = mlx_core
    Adam = MlxAdam
    ScannerPointsSHModel = ScannerSHModel
    active_sh_degree_for_step = scanner_active_sh_degree_for_step
    normalize_quats = tiny_normalize_quats
    render_sh_model = scanner_render_sh_model
    sh_coeff_count = scanner_sh_coeff_count


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def dodecahedron_geometry() -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    normals = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            normals.append([0.0, s1, s2 * phi])
            normals.append([s1, s2 * phi, 0.0])
            normals.append([s1 * phi, 0.0, s2])
    normals = np.asarray(normals, dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    vertices = []
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            for k in range(j + 1, len(normals)):
                mat = np.stack([normals[i], normals[j], normals[k]], axis=0)
                if abs(np.linalg.det(mat)) < 1.0e-8:
                    continue
                point = np.linalg.solve(mat, np.ones((3,), dtype=np.float64))
                if np.all(normals @ point <= 1.0 + 1.0e-6):
                    vertices.append(point)
    verts = []
    for point in vertices:
        if not any(np.linalg.norm(point - prev) < 1.0e-5 for prev in verts):
            verts.append(point)
    vertices_np = np.asarray(verts, dtype=np.float64)
    vertices_np /= np.max(np.linalg.norm(vertices_np, axis=1))

    faces: list[list[int]] = []
    for normal in normals:
        face = np.where(np.abs(vertices_np @ normal - np.max(vertices_np @ normal)) < 1.0e-5)[0]
        center = vertices_np[face].mean(axis=0)
        axis_u = vertices_np[face[0]] - center
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        angles = np.arctan2((vertices_np[face] - center) @ axis_v, (vertices_np[face] - center) @ axis_u)
        faces.append(face[np.argsort(angles)].astype(int).tolist())
    return vertices_np.astype(np.float32), faces, normals.astype(np.float32)


def face_colors(count: int) -> np.ndarray:
    start = np.array([1.0, 0.04, 0.02], dtype=np.float32)
    mid = np.array([0.06, 0.74, 0.95], dtype=np.float32)
    end = np.array([0.62, 0.08, 0.90], dtype=np.float32)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    first = (1.0 - np.minimum(t * 2.0, 1.0)) * start + np.minimum(t * 2.0, 1.0) * mid
    second = (1.0 - np.maximum((t - 0.5) * 2.0, 0.0)) * mid + np.maximum((t - 0.5) * 2.0, 0.0) * end
    return np.where(t <= 0.5, first, second).astype(np.float32)


def look_at_viewmat(position: np.ndarray, target: np.ndarray = np.zeros(3, dtype=np.float32)) -> np.ndarray:
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_guess))) > 0.95:
        up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(up_guess, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    rot = np.stack([right, up, forward], axis=0)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rot
    view[:3, 3] = -rot @ position
    return view


def make_camera_positions(count: int, radius: float) -> np.ndarray:
    positions = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(count):
        y = 1.0 - 2.0 * (i + 0.5) / float(count)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        positions.append([radius * r * np.cos(theta), radius * y, radius * r * np.sin(theta)])
    return np.asarray(positions, dtype=np.float32)


def make_intrinsics(width: int, height: int, focal_scale: float) -> np.ndarray:
    focal = focal_scale * float(min(width, height))
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def project_vertices(vertices: np.ndarray, viewmat: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (viewmat @ verts_h.T).T[:, :3]
    z = np.clip(cam[:, 2], 1.0e-5, None)
    uv = np.empty((vertices.shape[0], 2), dtype=np.float32)
    uv[:, 0] = K[0, 0] * cam[:, 0] / z + K[0, 2]
    uv[:, 1] = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return uv, cam


def render_dodecahedron(
    vertices: np.ndarray,
    faces: list[list[int]],
    normals: np.ndarray,
    colors: np.ndarray,
    viewmat: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    uv, cam = project_vertices(vertices, viewmat, K)
    image = Image.new("RGB", (width, height), tuple((BG * 255).astype(np.uint8).tolist()))
    draw = ImageDraw.Draw(image)
    rot = viewmat[:3, :3]
    visible_faces = []
    for face_id, face in enumerate(faces):
        if np.any(cam[face, 2] <= 0.01):
            continue
        normal_cam = rot @ normals[face_id]
        center_cam = cam[face].mean(axis=0)
        if float(np.dot(normal_cam, center_cam)) < 0.0:
            visible_faces.append(face_id)
    face_order = sorted(visible_faces, key=lambda idx: float(np.mean(cam[faces[idx], 2])), reverse=True)
    for face_id in face_order:
        face = faces[face_id]
        if np.any(cam[face, 2] <= 0.01):
            continue
        pts = [(float(uv[idx, 0]), float(uv[idx, 1])) for idx in face]
        color = tuple((np.clip(colors[face_id], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).tolist())
        draw.polygon(pts, fill=color, outline=(255, 255, 255))
    line_width = max(1, int(round(min(width, height) / 192.0)))
    for face_id in face_order:
        face = faces[face_id]
        if np.any(cam[face, 2] <= 0.01):
            continue
        pts = [(float(uv[idx, 0]), float(uv[idx, 1])) for idx in face]
        draw.line(pts + [pts[0]], fill=(255, 255, 255), width=line_width)
    return np.asarray(image, dtype=np.float32) / 255.0


def generate_dataset(out_dir: Path, width: int, height: int, camera_count: int, radius: float, focal_scale: float) -> list[FixedCamera]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces, normals = dodecahedron_geometry()
    colors = face_colors(len(faces))
    K = make_intrinsics(width, height, focal_scale)
    cameras = []
    metadata = {
        "width": width,
        "height": height,
        "camera_count": camera_count,
        "camera_radius": radius,
        "focal_scale": focal_scale,
        "vertices": vertices.astype(float).tolist(),
        "faces": faces,
        "face_colors": colors.astype(float).tolist(),
        "frames": [],
    }
    for idx, position in enumerate(make_camera_positions(camera_count, radius)):
        viewmat = look_at_viewmat(position)
        target = render_dodecahedron(vertices, faces, normals, colors, viewmat, K, width, height)
        image_path = out_dir / f"frame_{idx:03d}.png"
        write_png(image_path, image_to_u8(target))
        cameras.append(FixedCamera(idx, viewmat, K.copy(), position, target))
        metadata["frames"].append(
            {
                "index": idx,
                "image_path": str(image_path),
                "position": position.astype(float).tolist(),
                "viewmat": viewmat.astype(float).tolist(),
                "K": K.astype(float).tolist(),
            }
        )
    (out_dir / "dataset_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return cameras


def sample_gaussians(
    num_gaussians: int,
    seed: int,
    vertices: np.ndarray,
    faces: list[list[int]],
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not is_power_of_two(num_gaussians):
        raise ValueError(f"--num-gaussians must be a power of two, got {num_gaussians}")
    rng = np.random.default_rng(seed)
    del faces, colors
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    pad = 0.08 * float(np.max(bbox_max - bbox_min))
    means = rng.uniform(
        bbox_min - pad,
        bbox_max + pad,
        size=(1, num_gaussians, 3),
    ).astype(np.float32)
    random_colors = rng.uniform(0.02, 0.98, size=(1, 1, num_gaussians, 3)).astype(np.float32)
    color_logits = logit(random_colors)
    quats = np.zeros((1, num_gaussians, 4), dtype=np.float32)
    quats[0, :, 0] = 1.0
    quats += rng.normal(0.0, 0.03, size=quats.shape).astype(np.float32)
    scales = rng.uniform(0.035, 0.075, size=(1, num_gaussians, 3)).astype(np.float32)
    opacities = rng.uniform(0.35, 0.75, size=(1, num_gaussians)).astype(np.float32)
    return means, quats, np.log(scales).astype(np.float32), color_logits, logit(opacities)


def camera_arrays(camera: FixedCamera) -> tuple[mx.array, mx.array]:
    return (
        mx.array(camera.viewmat[None, None, ...], dtype=mx.float32),
        mx.array(camera.K[None, None, ...], dtype=mx.float32),
    )


def render_np(
    model,
    camera: FixedCamera,
    width: int,
    height: int,
    tile_size: int,
    sh_degree: int,
) -> tuple[np.ndarray, dict]:
    viewmats, Ks = camera_arrays(camera)
    viewspace = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)
    render = render_sh_model(model, viewspace, viewmats, Ks, width, height, tile_size, sh_degree)
    mx.eval(render["render_colors"], render["radii"], render["flatten_ids"])
    return np.asarray(render["render_colors"][0], dtype=np.float32), render


def save_grid(
    path: Path,
    model,
    cameras: list[FixedCamera],
    view_ids: list[int],
    width: int,
    height: int,
    tile_size: int,
    sh_degree: int,
) -> None:
    tile_w = 192
    tile_h = 96
    tiles = []
    for view_id in view_ids:
        camera = cameras[view_id]
        render, _ = render_np(model, camera, width, height, tile_size, sh_degree)
        pair = np.concatenate([camera.target, render], axis=1)
        image = Image.fromarray(image_to_u8(pair)).resize((tile_w, tile_h), Image.Resampling.BILINEAR)
        tiles.append(np.asarray(image, dtype=np.uint8))
    grid_cols = 4
    grid_rows = int(np.ceil(len(tiles) / grid_cols))
    grid = np.full((grid_rows * tile_h, grid_cols * tile_w, 3), 255, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // grid_cols
        col = idx % grid_cols
        grid[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    write_png(path, grid)


def save_model_npz(path: Path, model, active_sh_degree: int, max_sh_degree: int, summary: dict) -> None:
    mx.eval(model.means, model.normalized_quats, model.log_scales, model.features_dc, model.features_rest, model.opacity_logits)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        color_mode=np.array("sh", dtype=np.str_),
        active_sh_degree=np.array(active_sh_degree, dtype=np.int32),
        max_sh_degree=np.array(max_sh_degree, dtype=np.int32),
        means=np.asarray(model.means, dtype=np.float32),
        quats_wxyz=np.asarray(model.normalized_quats, dtype=np.float32),
        log_scales=np.asarray(model.log_scales, dtype=np.float32),
        features_dc=np.asarray(model.features_dc, dtype=np.float32),
        features_rest=np.asarray(model.features_rest, dtype=np.float32),
        opacity_logits=np.asarray(model.opacity_logits, dtype=np.float32),
        summary_json=np.array(json.dumps(summary), dtype=np.str_),
    )


def export_spz(path: Path, model, active_sh_degree: int) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_spz_variants_from_model_npz import positions_to_spz, quats_to_storage, scales_to_spz, transform_quats_wxyz

    try:
        import spz
    except ImportError as exc:
        raise ImportError("The 'spz' Python package is required for fixed-points SPZ export.") from exc

    mx.eval(model.means, model.normalized_quats, model.log_scales, model.features_dc, model.features_rest, model.opacity_logits)
    means = np.asarray(model.means[0], dtype=np.float32)
    quats = np.asarray(model.normalized_quats[0], dtype=np.float32)
    log_scales = np.asarray(model.log_scales[0], dtype=np.float32)
    features_dc = np.asarray(model.features_dc[0], dtype=np.float32)
    active_rest = sh_coeff_count(active_sh_degree) - 1
    features_rest = np.asarray(model.features_rest[0, :, :active_rest, :], dtype=np.float32)
    opacity_logits = np.asarray(model.opacity_logits[0], dtype=np.float32)

    cloud = spz.GaussianCloud()
    cloud.antialiased = True
    cloud.positions = positions_to_spz(means, "scanner").reshape(-1).astype(np.float32)
    cloud.scales = scales_to_spz(log_scales, "direct").reshape(-1).astype(np.float32)
    cloud.rotations = quats_to_storage(transform_quats_wxyz(quats, "position_axis"), "xyzw").reshape(-1).astype(np.float32)
    cloud.alphas = opacity_logits.reshape(-1).astype(np.float32)
    cloud.colors = features_dc.reshape(-1).astype(np.float32)
    cloud.sh_degree = int(active_sh_degree)
    cloud.sh = features_rest.reshape(-1).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not spz.save_spz(cloud, spz.PackOptions(), str(path)):
        raise RuntimeError(f"failed to save spz to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-out", type=Path, default=Path("outputs/fixed_points_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fixed_points_train"))
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--num-cameras", type=int, default=48)
    parser.add_argument("--num-gaussians", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--grid-interval", type=int, default=200)
    parser.add_argument("--grid-tiles", type=int, default=16)
    parser.add_argument("--sh-degree-start", type=int, default=0)
    parser.add_argument("--sh-degree-target", type=int, default=3)
    parser.add_argument("--sh-degree-schedule-interval", type=int, default=1000)
    parser.add_argument("--camera-radius", type=float, default=3.2)
    parser.add_argument("--focal-scale", type=float, default=0.92)
    parser.add_argument("--lr-means", type=float, default=8.0e-3)
    parser.add_argument("--lr-colors", type=float, default=3.0e-2)
    parser.add_argument("--lr-opacity", type=float, default=8.0e-3)
    parser.add_argument("--lr-scales", type=float, default=3.0e-3)
    parser.add_argument("--lr-quats", type=float, default=2.0e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = generate_dataset(args.dataset_out, args.width, args.height, args.num_cameras, args.camera_radius, args.focal_scale)
    if args.dataset_only:
        print(f"fixed-points dataset ok frames={len(cameras)} output_dir={args.dataset_out}")
        return

    load_training_deps()
    if args.sh_degree_start < 0 or args.sh_degree_target < args.sh_degree_start or args.sh_degree_target > 3:
        raise ValueError("--sh-degree-start/target must satisfy 0 <= start <= target <= 3")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces, _ = dodecahedron_geometry()
    colors = face_colors(len(faces))
    means, quats, log_scales, color_logits, opacity_logits = sample_gaussians(
        args.num_gaussians,
        args.seed,
        vertices,
        faces,
        colors,
    )
    random_colors = 1.0 / (1.0 + np.exp(-color_logits[:, 0]))
    features_dc = ((random_colors - 0.5) / SH_C0).astype(np.float32)
    features_rest = np.zeros((1, args.num_gaussians, sh_coeff_count(args.sh_degree_target) - 1, 3), dtype=np.float32)
    model = ScannerPointsSHModel.from_arrays(
        mx.array(means, dtype=mx.float32),
        normalize_quats(mx.array(quats, dtype=mx.float32)),
        mx.array(log_scales, dtype=mx.float32),
        mx.array(features_dc, dtype=mx.float32),
        mx.array(features_rest, dtype=mx.float32),
        mx.array(opacity_logits, dtype=mx.float32),
    )
    targets = [mx.array(camera.target[None, ...], dtype=mx.float32) for camera in cameras]
    grid_ids = np.linspace(0, len(cameras) - 1, min(args.grid_tiles, len(cameras)), dtype=np.int32).astype(int).tolist()

    def loss_fn(
        means_: mx.array,
        quats_: mx.array,
        log_scales_: mx.array,
        features_dc_: mx.array,
        features_rest_: mx.array,
        opacity_logits_: mx.array,
        viewspace_points_: mx.array,
        viewmats_: mx.array,
        Ks_: mx.array,
        target_: mx.array,
        sh_degree_: int,
    ) -> mx.array:
        local = ScannerPointsSHModel.from_arrays(means_, quats_, log_scales_, features_dc_, features_rest_, opacity_logits_)
        render = render_sh_model(local, viewspace_points_, viewmats_, Ks_, args.width, args.height, args.tile_size, sh_degree_)
        diff = render["render_colors"] - target_
        return mx.mean(mx.abs(diff))

    def evaluate_loss(sh_degree: int) -> float:
        losses = []
        for camera, target in zip(cameras, targets, strict=True):
            viewmats, Ks = camera_arrays(camera)
            viewspace = mx.zeros((1, 1, args.num_gaussians, 2), dtype=mx.float32)
            loss = loss_fn(
                model.means,
                model.quats,
                model.log_scales,
                model.features_dc,
                model.features_rest,
                model.opacity_logits,
                viewspace,
                viewmats,
                Ks,
                target,
                sh_degree,
            )
            mx.eval(loss)
            losses.append(float(np.asarray(loss)))
        return float(np.mean(losses))

    initial_active_sh_degree = int(args.sh_degree_start)
    initial_mean_loss = evaluate_loss(initial_active_sh_degree)
    save_grid(args.out_dir / "grid_step_0000.png", model, cameras, grid_ids, args.width, args.height, args.tile_size, initial_active_sh_degree)
    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))
    optimizers = {
        "means": Adam(learning_rate=args.lr_means),
        "quats": Adam(learning_rate=args.lr_quats),
        "log_scales": Adam(learning_rate=args.lr_scales),
        "features_dc": Adam(learning_rate=args.lr_colors),
        "features_rest": Adam(learning_rate=args.lr_colors),
        "opacity_logits": Adam(learning_rate=args.lr_opacity),
    }

    last_loss = None
    last_viewspace_grad = None
    active_sh_degree = initial_active_sh_degree
    sh_degree_events = [{"step": 0, "active_sh_degree": active_sh_degree}]
    for step in range(1, args.steps + 1):
        next_sh_degree = active_sh_degree_for_step(
            args.sh_degree_start,
            args.sh_degree_target,
            args.sh_degree_schedule_interval,
            step,
        )
        if next_sh_degree != active_sh_degree:
            active_sh_degree = int(next_sh_degree)
            sh_degree_events.append({"step": int(step), "active_sh_degree": active_sh_degree})
        view_id = (step - 1) % len(cameras)
        camera = cameras[view_id]
        target = targets[view_id]
        viewmats, Ks = camera_arrays(camera)
        viewspace = mx.zeros((1, 1, args.num_gaussians, 2), dtype=mx.float32)
        loss, grads = grad_fn(
            model.means,
            model.quats,
            model.log_scales,
            model.features_dc,
            model.features_rest,
            model.opacity_logits,
            viewspace,
            viewmats,
            Ks,
            target,
            active_sh_degree,
        )
        d_means, d_quats, d_log_scales, d_features_dc, d_features_rest, d_opacity_logits, d_viewspace = grads
        mx.eval(loss, d_viewspace)
        last_loss = float(np.asarray(loss))
        last_viewspace_grad = d_viewspace

        optimizers["means"].update(model, {"means": d_means})
        optimizers["quats"].update(model, {"quats": d_quats})
        optimizers["log_scales"].update(model, {"log_scales": d_log_scales})
        optimizers["features_dc"].update(model, {"features_dc": d_features_dc})
        optimizers["features_rest"].update(model, {"features_rest": d_features_rest})
        optimizers["opacity_logits"].update(model, {"opacity_logits": d_opacity_logits})
        model.quats = normalize_quats(model.quats)
        mx.eval(model.means, model.quats, model.log_scales, model.features_dc, model.features_rest, model.opacity_logits)

        if step == 1 or step == args.steps or step % args.grid_interval == 0:
            print(f"step={step:04d} view={view_id:02d} sh_degree={active_sh_degree} loss={last_loss:.8f}")
        if step % args.grid_interval == 0:
            save_grid(args.out_dir / f"grid_step_{step:04d}.png", model, cameras, grid_ids, args.width, args.height, args.tile_size, active_sh_degree)

    final_mean_loss = evaluate_loss(active_sh_degree)
    if last_loss is None or not np.isfinite(final_mean_loss):
        raise AssertionError("fixed-points training loss should be finite")
    if final_mean_loss > initial_mean_loss * 1.05:
        raise AssertionError(f"fixed-points training diverged: initial={initial_mean_loss:.8f} final={final_mean_loss:.8f}")
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("fixed-points training expected nonzero viewspace_points gradient")

    summary = {
        "dataset": str(args.dataset_out),
        "out_dir": str(args.out_dir),
        "width": args.width,
        "height": args.height,
        "cameras": len(cameras),
        "gaussians": args.num_gaussians,
        "steps": args.steps,
        "grid_interval": args.grid_interval,
        "grid_tiles": len(grid_ids),
        "loss_function": "l1",
        "initial_active_sh_degree": initial_active_sh_degree,
        "active_sh_degree_final": active_sh_degree,
        "sh_degree_schedule": {
            "start": int(args.sh_degree_start),
            "target": int(args.sh_degree_target),
            "interval": int(args.sh_degree_schedule_interval),
            "events": sh_degree_events,
        },
        "initial_mean_loss": initial_mean_loss,
        "final_mean_loss": final_mean_loss,
        "spz_convention": {
            "position": "scanner",
            "scale": "direct",
            "rotation": "position_axis",
            "quat_order": "xyzw",
            "color": "sh",
        },
        "model_npz": str(args.out_dir / "trained_model_params.npz"),
        "spz": str(args.out_dir / "trained_fixed_points.spz"),
    }
    save_model_npz(args.out_dir / "trained_model_params.npz", model, active_sh_degree, args.sh_degree_target, summary)
    export_spz(args.out_dir / "trained_fixed_points.spz", model, active_sh_degree)
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "fixed-points 3dgs mlx training ok "
        f"initial_mean_loss={initial_mean_loss:.8f} final_mean_loss={final_mean_loss:.8f} "
        f"gaussians={args.num_gaussians} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
