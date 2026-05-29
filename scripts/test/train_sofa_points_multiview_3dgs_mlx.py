#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.optimizers import Adam

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from b075x65r3x_dataset import load_b075x65r3x_dataset  # noqa: E402
from dodecahedron_dataset import load_dodecahedron_dataset  # noqa: E402
from train_360_points_multiview_3dgs_mlx import (  # noqa: E402
    MAX_SUPPORTED_SH_DEGREE,
    gsplat_active_sh_degree,
    init_sh_model_from_points,
    knn_log_scales_from_points,
    log,
    mean_loss,
    validate_positive,
)
from train_scanner_points_multiview_3dgs_mlx import (  # noqa: E402
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
    local_average,
    make_lr_schedule,
    normalize_quats,
    opacity_diagnostics,
    points_extent_diagnostics,
    render_sh_model,
    save_model_parameters_npz,
    spz_export_diagnostics,
    target_batch_array,
)
from training_dataset import TrainingDataset, write_png  # noqa: E402


def camera_center_from_viewmat(viewmat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(viewmat).astype(np.float32)[:3, 3]


def scene_scale_from_dataset(dataset: TrainingDataset) -> float:
    centers = np.stack(
        [
            camera.position.astype(np.float32)
            if getattr(camera, "position", None) is not None
            else camera_center_from_viewmat(camera.viewmat)
            for camera in dataset.cameras
        ],
        axis=0,
    )
    center = np.mean(centers, axis=0)
    camera_radius = float(np.max(np.linalg.norm(centers - center[None, :], axis=1)))
    bbox_diag = float(np.linalg.norm(dataset.bbox_max - dataset.bbox_min))
    return max(camera_radius, bbox_diag * 0.5, 1.0e-6)


def select_cameras(cameras: list, max_frames: int, frame_step: int, start_index: int) -> list:
    selected = [camera for camera in cameras if int(camera.index) >= start_index]
    if frame_step > 1:
        selected = selected[::frame_step]
    if max_frames > 0:
        selected = selected[:max_frames]
    if not selected:
        raise RuntimeError("No training cameras selected")
    return selected


def sample_points(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    raw_count = int(points.shape[0])
    if raw_count == 0:
        raise RuntimeError("Dataset does not provide foreground points for point-based initialization")
    if max_points > 0 and raw_count > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(raw_count, size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    return points.astype(np.float32), colors.astype(np.float32), raw_count


def camera_alpha_mask(camera, width: int, height: int) -> np.ndarray:
    mask = getattr(camera, "alpha_mask", None)
    if mask is None:
        return np.ones((height, width, 1), dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.shape != (height, width, 1):
        raise ValueError(f"alpha mask for camera {camera.index} has shape {mask.shape}, expected {(height, width, 1)}")
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def mask_batch_array(masks: list[mx.array], batch_ids: list[int]) -> mx.array:
    return mx.concatenate([masks[idx] for idx in batch_ids], axis=0)


def masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    weights = mx.broadcast_to(mask, values.shape)
    return mx.sum(values * weights) / mx.maximum(mx.sum(weights), 1.0e-8)


def masked_ssim_index(image: mx.array, target: mx.array, mask: mx.array, window_size: int) -> mx.array:
    mu_x = local_average(image, window_size)
    mu_y = local_average(target, window_size)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = local_average(image * image, window_size) - mu_x2
    sigma_y2 = local_average(target * target, window_size) - mu_y2
    sigma_xy = local_average(image * target, window_size) - mu_xy
    c1 = 0.01 * 0.01
    c2 = 0.03 * 0.03
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / mx.maximum(denominator, 1.0e-12)
    return masked_mean(ssim_map, mask)


def masked_loss_components(
    image: mx.array,
    target: mx.array,
    mask: mx.array,
    ssim_lambda: float,
    ssim_window_size: int,
) -> dict[str, mx.array]:
    l1 = masked_mean(mx.abs(image - target), mask)
    ssim = masked_ssim_index(image, target, mask, ssim_window_size)
    ssim_loss = 1.0 - ssim
    loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * ssim_loss
    mask_coverage = mx.mean(mask)
    return {"loss": loss, "l1": l1, "ssim": ssim, "ssim_loss": ssim_loss, "mask_coverage": mask_coverage}


def render_loss_stats_masked(
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
    components = masked_loss_components(render["render_colors"], target, mask, ssim_lambda, ssim_window_size)
    diff = render["render_colors"] - target
    mse = masked_mean(diff * diff, mask)
    mx.eval(
        components["loss"],
        components["l1"],
        components["ssim"],
        components["ssim_loss"],
        components["mask_coverage"],
        mse,
        render["render_colors"],
        render["radii"],
        render["flatten_ids"],
    )
    mse_value = float(np.asarray(mse))
    radii = np.asarray(render["radii"])
    flatten_ids = np.asarray(render["flatten_ids"])
    return {
        "frame_index": int(camera.index),
        "loss": float(np.asarray(components["loss"])),
        "loss_components": {
            "l1": float(np.asarray(components["l1"])),
            "ssim": float(np.asarray(components["ssim"])),
            "ssim_loss": float(np.asarray(components["ssim_loss"])),
            "mask_coverage": float(np.asarray(components["mask_coverage"])),
        },
        "psnr": float(-10.0 * np.log10(max(mse_value, 1.0e-12))),
        "visible_gaussians": int(np.count_nonzero(np.any(radii > 0, axis=-1))),
        "intersections": int(flatten_ids.shape[0]),
        "image": np.asarray(render["render_colors"][0], dtype=np.float32),
    }


def evaluate_frames_masked(
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
        render_loss_stats_masked(model, camera, target, mask, width, height, tile_size, sh_degree, ssim_lambda, ssim_window_size)
        for camera, target, mask in zip(cameras, targets, masks, strict=True)
    ]


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
        white_background=not args.dark_background,
    )


def parse_args(default_dataset: str = "b075x65r3x") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("b075x65r3x", "dodecahedron"), default=default_dataset)
    parser.add_argument("--data", type=Path, default=Path("datasets/B075X65R3X"))
    parser.add_argument("--dataset-out", type=Path, default=Path("outputs/dodecahedron_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sofa_train"))
    parser.add_argument("--out-spz", type=Path, default=None)
    parser.add_argument("--out-model-npz", type=Path, default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--num-cameras", type=int, default=48)
    parser.add_argument("--camera-radius", type=float, default=3.2)
    parser.add_argument("--focal-scale", type=float, default=0.92)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=50000)
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
    parser.add_argument("--dark-background", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log(f"loading dataset type={args.dataset} size={args.width}x{args.height}")
    dataset = load_dataset(args)
    cameras = (
        dataset.cameras
        if args.dataset == "b075x65r3x"
        else select_cameras(dataset.cameras, args.max_frames, args.frame_step, args.start_index)
    )
    targets = [mx.array(camera.target[None, ...], dtype=mx.float32) for camera in cameras]
    masks = [mx.array(camera_alpha_mask(camera, args.width, args.height)[None, ...], dtype=mx.float32) for camera in cameras]
    mask_coverages = [float(np.mean(camera_alpha_mask(camera, args.width, args.height))) for camera in cameras]
    scene_scale = scene_scale_from_dataset(dataset)
    resolved_scene_scale = float(scene_scale * 1.1 * args.global_scale)
    log(f"loaded targets train_frames={len(cameras)} scene_scale={scene_scale:.8f}")

    log("preparing foreground points")
    points, colors, raw_point_count = sample_points(dataset.foreground_points, dataset.foreground_colors, args.max_points, args.seed)
    points, colors = append_random_gaussians(points, colors, args.num_random_gaussians, args.seed + 1009, args.random_gaussian_bounds_scale)
    log(f"initializing KNN log scales points={points.shape[0]} init_scale={args.init_scale}")
    log_scales = knn_log_scales_from_points(points, args.init_scale)
    point_diagnostics = points_extent_diagnostics(points)
    means_lr = float(args.means_lr * resolved_scene_scale)
    means_lr_final = means_lr * 0.01

    log(
        "initializing SH model "
        f"gaussians={points.shape[0]} sh_degree={args.sh_degree} opacity={args.opacity}"
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

    log(f"running initial evaluation frames={len(cameras)}")
    initial_stats = evaluate_frames_masked(model, cameras, targets, masks, args.width, args.height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)
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
                args.width,
                args.height,
                args.tile_size,
                active_sh_degree,
            )
            losses.append(masked_loss_components(render["render_colors"], target[idx : idx + 1], mask[idx : idx + 1], args.ssim_lambda, args.ssim_window_size)["loss"])
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
        mask = mask_batch_array(masks, batch_ids)
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
            strategy.update_state(d_viewspace, strategy_radii, width=args.width, height=args.height, n_cameras=len(batch_ids))
        strategy.after_optimizer_step(step, model, optimizers, "sh")

        if step == 1 or step == args.steps or step % args.log_interval == 0:
            log(
                f"step={step:04d} frames={batch_frame_indices} sh={active_sh_degree} "
                f"loss={last_loss:.8f} means_lr={latest_lrs['means']:.8g} "
                f"viewspace_grad_norm={last_viewspace_grad_norm:.8f}"
            )

    log(f"running final evaluation frames={len(cameras)}")
    final_stats = evaluate_frames_masked(model, cameras, targets, masks, args.width, args.height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)
    final_mean_loss = mean_loss(final_stats)
    for initial, final, target in zip(initial_stats, final_stats, targets, strict=True):
        frame_index = final["frame_index"]
        write_png(args.out_dir / f"compare_frame_{frame_index:05d}.png", image_to_u8(concat_compare(np.asarray(target[0]), initial["image"], final["image"])))

    if last_loss is None or not np.isfinite(final_mean_loss):
        raise AssertionError("training loss should be finite")
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("training expected nonzero viewspace_points gradient")

    out_spz = args.out_spz if args.out_spz is not None else args.out_dir / f"trained_{args.dataset}.spz"
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
    refinement_summary = strategy.summary()
    final_opacity_diagnostics = opacity_diagnostics(model)
    summary = {
        "dataset_type": args.dataset,
        "dataset": str(args.data if args.dataset == "b075x65r3x" else args.dataset_out),
        "width": int(args.width),
        "height": int(args.height),
        "raw_point_count": int(raw_point_count),
        "exported_gaussians": int(exported_gaussians),
        "max_points": int(args.max_points),
        "point_cloud_gaussians": int(points.shape[0] - args.num_random_gaussians),
        "random_gaussians": int(args.num_random_gaussians),
        "frames": len(cameras),
        "steps": int(args.steps),
        "mlx_cache_limit_bytes": int(cache_limit_bytes),
        "mlx_cache_limit_gb": float(args.mlx_cache_limit_gb),
        "mlx_previous_cache_limit_bytes": int(previous_cache_limit),
        "scene_scale": float(scene_scale),
        "resolved_scene_scale": float(resolved_scene_scale),
        "initialization": {
            "type": "foreground_points",
            "scale_rule": "average distance to 3 nearest neighbors times init_scale",
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
            "mode": "l1_ssim",
            "formula": "(1 - ssim_lambda) * L1 + ssim_lambda * (1 - SSIM)",
            "ssim_lambda": float(args.ssim_lambda),
            "ssim_window_size": int(args.ssim_window_size),
            "alpha_masked": True,
            "mask_coverage_min": float(np.min(mask_coverages)),
            "mask_coverage_mean": float(np.mean(mask_coverages)),
            "mask_coverage_max": float(np.max(mask_coverages)),
        },
        "learning_rate_schedule": lr_schedules,
        "initial_mean_loss": float(initial_mean_loss),
        "final_mean_loss": float(final_mean_loss),
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
    }
    out_model_npz = args.out_model_npz if args.out_model_npz is not None else args.out_dir / "trained_model_params.npz"
    summary["model_npz"] = str(out_model_npz)
    save_model_parameters_npz(out_model_npz, model, "sh", active_sh_degree, args.sh_degree, summary)
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_spz.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(
        f"{args.dataset} points multi-view training ok "
        f"initial_mean_loss={initial_mean_loss:.8f} final_mean_loss={final_mean_loss:.8f} "
        f"last_viewspace_grad_norm={last_viewspace_grad_norm:.8f} "
        f"spz={out_spz} bytes={spz_size} output_dir={args.out_dir}"
    )


def main(default_dataset: str = "b075x65r3x") -> None:
    train(parse_args(default_dataset))


if __name__ == "__main__":
    main()
