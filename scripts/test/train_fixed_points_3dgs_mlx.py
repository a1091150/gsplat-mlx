#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from b075x65r3x_dataset import load_b075x65r3x_dataset
from dodecahedron_dataset import load_dodecahedron_dataset
from training_dataset import TrainingCamera, TrainingDataset, image_to_u8, write_png


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
sh_colors_for_camera = None
projection_ewa_3dgs_fused_forward = None
intersect_tile_forward = None
intersect_offset_forward = None
rasterize_to_pixels_3dgs_forward = None


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1.0e-5, 1.0 - 1.0e-5)
    return np.log(values / (1.0 - values)).astype(np.float32)


def load_training_deps() -> None:
    global Adam, ScannerPointsSHModel, active_sh_degree_for_step, intersect_offset_forward, intersect_tile_forward, mx, normalize_quats
    global projection_ewa_3dgs_fused_forward, rasterize_to_pixels_3dgs_forward, render_sh_model, sh_coeff_count, sh_colors_for_camera
    import mlx.core as mlx_core
    from mlx.optimizers import Adam as MlxAdam
    from gsplat_core import intersect_offset_forward as gs_intersect_offset_forward
    from gsplat_core import intersect_tile_forward as gs_intersect_tile_forward
    from gsplat_core import projection_ewa_3dgs_fused_forward as gs_projection_ewa_3dgs_fused_forward
    from gsplat_core import rasterize_to_pixels_3dgs_forward as gs_rasterize_to_pixels_3dgs_forward
    from train_tiny_3dgs_mlx import normalize_quats as tiny_normalize_quats
    from train_scanner_points_multiview_3dgs_mlx import ScannerPointsSHModel as ScannerSHModel
    from train_scanner_points_multiview_3dgs_mlx import active_sh_degree_for_step as scanner_active_sh_degree_for_step
    from train_scanner_points_multiview_3dgs_mlx import render_sh_model as scanner_render_sh_model
    from train_scanner_points_multiview_3dgs_mlx import sh_coeff_count as scanner_sh_coeff_count
    from train_scanner_points_multiview_3dgs_mlx import sh_colors_for_camera as scanner_sh_colors_for_camera

    mx = mlx_core
    Adam = MlxAdam
    ScannerPointsSHModel = ScannerSHModel
    active_sh_degree_for_step = scanner_active_sh_degree_for_step
    normalize_quats = tiny_normalize_quats
    render_sh_model = scanner_render_sh_model
    sh_coeff_count = scanner_sh_coeff_count
    sh_colors_for_camera = scanner_sh_colors_for_camera
    projection_ewa_3dgs_fused_forward = gs_projection_ewa_3dgs_fused_forward
    intersect_tile_forward = gs_intersect_tile_forward
    intersect_offset_forward = gs_intersect_offset_forward
    rasterize_to_pixels_3dgs_forward = gs_rasterize_to_pixels_3dgs_forward


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def sample_bbox_gaussians(
    num_gaussians: int,
    seed: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not is_power_of_two(num_gaussians):
        raise ValueError(f"--num-gaussians must be a power of two, got {num_gaussians}")
    rng = np.random.default_rng(seed)
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


def sample_foreground_gaussians(
    num_gaussians: int,
    seed: int,
    dataset: TrainingDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not is_power_of_two(num_gaussians):
        raise ValueError(f"--num-gaussians must be a power of two, got {num_gaussians}")
    points = dataset.foreground_points
    colors = dataset.foreground_colors
    if points is None or colors is None or len(points) == 0:
        raise ValueError(f"{dataset.name} does not provide foreground points; use --init-mode bbox instead.")

    rng = np.random.default_rng(seed)
    replace = len(points) < num_gaussians
    ids = rng.choice(len(points), size=num_gaussians, replace=replace)
    means = points[ids][None, ...].astype(np.float32)
    sampled_colors = np.clip(colors[ids], 0.02, 0.98).astype(np.float32)
    color_logits = logit(sampled_colors[None, None, ...])

    quats = np.zeros((1, num_gaussians, 4), dtype=np.float32)
    quats[0, :, 0] = 1.0
    quats += rng.normal(0.0, 0.01, size=quats.shape).astype(np.float32)

    extent = float(np.max(dataset.bbox_max - dataset.bbox_min))
    base_scale = max(extent * 0.01, 1.0e-4)
    scales = rng.uniform(base_scale * 0.5, base_scale * 1.5, size=(1, num_gaussians, 3)).astype(np.float32)
    opacities = np.full((1, num_gaussians), 0.65, dtype=np.float32)
    return means, quats, np.log(scales).astype(np.float32), color_logits, logit(opacities)


def sample_initial_gaussians(
    init_mode: str,
    num_gaussians: int,
    seed: int,
    dataset: TrainingDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if init_mode == "bbox":
        return sample_bbox_gaussians(num_gaussians, seed, dataset.bbox_min, dataset.bbox_max)
    return sample_foreground_gaussians(num_gaussians, seed, dataset)


def camera_arrays(camera: TrainingCamera) -> tuple[mx.array, mx.array]:
    return (
        mx.array(camera.viewmat[None, None, ...], dtype=mx.float32),
        mx.array(camera.K[None, None, ...], dtype=mx.float32),
    )


def render_np(
    model,
    camera: TrainingCamera,
    width: int,
    height: int,
    tile_size: int,
    sh_degree: int,
    background_color: np.ndarray,
) -> tuple[np.ndarray, dict]:
    viewmats, Ks = camera_arrays(camera)
    viewspace = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)
    render = render_sh_model_background(model, viewspace, viewmats, Ks, width, height, tile_size, sh_degree, background_color)
    mx.eval(render["render_colors"], render["radii"], render["flatten_ids"])
    return np.asarray(render["render_colors"][0], dtype=np.float32), render


def render_sh_model_background(
    model,
    viewspace_points,
    viewmats,
    Ks,
    width: int,
    height: int,
    tile_size: int,
    sh_degree: int,
    background_color: np.ndarray,
) -> dict:
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    projection = projection_ewa_3dgs_fused_forward(
        {
            "means": model.means,
            "quats": model.normalized_quats,
            "scales": model.scales,
            "viewmats": viewmats,
            "Ks": Ks,
            "viewspace_points": viewspace_points,
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
    tile_offsets = mx.stop_gradient(
        intersect_offset_forward(
            intersections["isect_ids"],
            I=1,
            tile_width=tile_width,
            tile_height=tile_height,
        )
    )
    flatten_ids = mx.stop_gradient(intersections["flatten_ids"])
    render = rasterize_to_pixels_3dgs_forward(
        {
            "means2d": projection["means2d"],
            "conics": projection["conics"],
            "colors": sh_colors_for_camera(model, viewmats, sh_degree),
            "opacities": mx.expand_dims(model.opacities, axis=1),
            "backgrounds": mx.array(np.asarray(background_color, dtype=np.float32)[None, :], dtype=mx.float32),
            "tile_offsets": tile_offsets,
            "flatten_ids": flatten_ids,
        },
        image_width=width,
        image_height=height,
        tile_size=tile_size,
    )
    return {
        **render,
        "radii": projection["radii"],
        "tiles_per_gauss": mx.stop_gradient(intersections["tiles_per_gauss"]),
        "flatten_ids": flatten_ids,
    }


def save_grid(
    path: Path,
    model,
    cameras: list[TrainingCamera],
    view_ids: list[int],
    width: int,
    height: int,
    tile_size: int,
    sh_degree: int,
    background_color: np.ndarray,
) -> None:
    tile_w = 192
    tile_h = 96
    tiles = []
    for view_id in view_ids:
        camera = cameras[view_id]
        render, _ = render_np(model, camera, width, height, tile_size, sh_degree, background_color)
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


def select_grid_ids(camera_count: int, requested_tiles: int) -> list[int]:
    if camera_count <= 0:
        raise ValueError("Cannot select grid views from an empty dataset.")
    tile_count = min(max(int(requested_tiles), 1), camera_count)
    if tile_count == 1:
        return [0]
    ids = np.rint(np.linspace(0, camera_count - 1, tile_count)).astype(int)
    return np.clip(ids, 0, camera_count - 1).tolist()


def should_save_grid(step: int, total_steps: int, grid_interval: int) -> bool:
    return step == 1 or step == total_steps or (grid_interval > 0 and step % grid_interval == 0)


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
    parser.add_argument("--dataset", choices=("b075x65r3x", "dodecahedron"), default="b075x65r3x")
    parser.add_argument("--data", type=Path, default=Path("datasets/B075X65R3X"))
    parser.add_argument("--dataset-out", type=Path, default=Path("outputs/fixed_points_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fixed_points_train"))
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--num-cameras", type=int, default=48)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-gaussians", type=int, default=1024)
    parser.add_argument("--init-mode", choices=("foreground", "bbox"), default="foreground")
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


def load_dataset(args: argparse.Namespace) -> TrainingDataset:
    if args.dataset == "dodecahedron":
        return load_dodecahedron_dataset(
            args.dataset_out,
            args.width,
            args.height,
            args.num_cameras,
            args.camera_radius,
            args.focal_scale,
        )
    return load_b075x65r3x_dataset(
        args.data,
        args.width,
        args.height,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        start_index=args.start_index,
    )


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args)
    cameras = dataset.cameras
    if args.dataset_only:
        print(f"{dataset.name} dataset ok frames={len(cameras)}")
        return

    load_training_deps()
    if args.sh_degree_start < 0 or args.sh_degree_target < args.sh_degree_start or args.sh_degree_target > 3:
        raise ValueError("--sh-degree-start/target must satisfy 0 <= start <= target <= 3")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    background_color = (
        dataset.background_color.astype(np.float32)
        if dataset.background_color is not None
        else np.array([0.025, 0.025, 0.025], dtype=np.float32)
    )
    means, quats, log_scales, color_logits, opacity_logits = sample_initial_gaussians(
        args.init_mode,
        args.num_gaussians,
        args.seed,
        dataset,
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
    grid_ids = select_grid_ids(len(cameras), args.grid_tiles)

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
        render = render_sh_model_background(
            local,
            viewspace_points_,
            viewmats_,
            Ks_,
            args.width,
            args.height,
            args.tile_size,
            sh_degree_,
            background_color,
        )
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
    save_grid(
        args.out_dir / "grid_step_0000.png",
        model,
        cameras,
        grid_ids,
        args.width,
        args.height,
        args.tile_size,
        initial_active_sh_degree,
        background_color,
    )
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

        if should_save_grid(step, args.steps, args.grid_interval):
            print(f"step={step:04d} view={view_id:02d} sh_degree={active_sh_degree} loss={last_loss:.8f}")
        if step == args.steps or (args.grid_interval > 0 and step % args.grid_interval == 0):
            save_grid(
                args.out_dir / f"grid_step_{step:04d}.png",
                model,
                cameras,
                grid_ids,
                args.width,
                args.height,
                args.tile_size,
                active_sh_degree,
                background_color,
            )

    final_mean_loss = evaluate_loss(active_sh_degree)
    if last_loss is None or not np.isfinite(final_mean_loss):
        raise AssertionError("fixed-points training loss should be finite")
    if final_mean_loss > initial_mean_loss * 1.05:
        raise AssertionError(f"fixed-points training diverged: initial={initial_mean_loss:.8f} final={final_mean_loss:.8f}")
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("fixed-points training expected nonzero viewspace_points gradient")

    summary = {
        "dataset": args.dataset,
        "dataset_path": str(args.data if args.dataset == "b075x65r3x" else args.dataset_out),
        "dataset_metadata": dataset.metadata,
        "out_dir": str(args.out_dir),
        "width": args.width,
        "height": args.height,
        "cameras": len(cameras),
        "gaussians": args.num_gaussians,
        "init_mode": args.init_mode,
        "foreground_points": 0 if dataset.foreground_points is None else int(dataset.foreground_points.shape[0]),
        "background_color": background_color.astype(float).tolist(),
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
