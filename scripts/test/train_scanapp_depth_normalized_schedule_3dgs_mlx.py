#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.optimizers import Adam

from colmap_360_dataset import normalize_scene, transform_cameras
from scanner_dataset_utils import load_target
from scanner_points_training_utils import (
    FrameBatchSampler,
    ScannerDefaultStrategyConfig,
    ScannerDefaultStrategyRuntime,
    camera_batch_arrays,
    concat_compare,
    export_trained_spz,
    image_to_u8,
    lr_for_step,
    make_lr_schedule,
    normalize_quats,
    opacity_diagnostics,
    points_extent_diagnostics,
    save_model_parameters_npz,
    spz_export_diagnostics,
    target_batch_array,
)
import train_scanapp_depth_consistency_multiview_3dgs_mlx as base


MAX_SUPPORTED_SH_DEGREE = 3


def log(message: str) -> None:
    print(message, flush=True)


def parse_factor_schedule(raw: str) -> list[float]:
    factors = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not factors:
        raise ValueError("--image-scale-factors must contain at least one factor")
    if any(factor <= 0.0 for factor in factors):
        raise ValueError("--image-scale-factors must be positive")
    return factors


def split_stage_steps(total_steps: int, stage_count: int, raw: str | None) -> list[int]:
    if raw:
        steps = [int(item.strip()) for item in raw.split(",") if item.strip()]
        if len(steps) != stage_count:
            raise ValueError("--stage-steps must have the same count as --image-scale-factors")
        if any(step <= 0 for step in steps):
            raise ValueError("--stage-steps entries must be positive")
        return steps
    base_steps = total_steps // stage_count
    remainder = total_steps % stage_count
    return [base_steps + (1 if index < remainder else 0) for index in range(stage_count)]


def parse_int_schedule(raw: str, count: int, name: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if len(values) == 1:
        values = values * count
    if len(values) != count:
        raise ValueError(f"{name} must contain either 1 value or the same count as --image-scale-factors")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} entries must be positive")
    return values


def stage_size(reference_width: int, reference_height: int, factor: float) -> tuple[int, int]:
    width = int(round(float(reference_width) / factor))
    height = int(round(float(reference_height) / factor))
    return max(1, width), max(1, height)


def camera_c2ws(cameras) -> np.ndarray:
    return np.stack([np.linalg.inv(camera.viewmat).astype(np.float32) for camera in cameras], axis=0)


def apply_world_transform_to_cameras(cameras, transform: np.ndarray):
    if not cameras:
        return []
    transformed_c2ws = transform_cameras(transform.astype(np.float32), camera_c2ws(cameras))
    return [
        replace(camera, viewmat=np.linalg.inv(c2w).astype(np.float32))
        for camera, c2w in zip(cameras, transformed_c2ws, strict=True)
    ]


def box_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return image.astype(np.float32, copy=True)
    if kernel_size % 2 == 0:
        raise ValueError("Mean blur kernel size must be odd")
    pad = kernel_size // 2
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    integral = np.pad(padded.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)), mode="constant")
    total = integral[kernel_size:, kernel_size:] - integral[:-kernel_size, kernel_size:] - integral[kernel_size:, :-kernel_size] + integral[:-kernel_size, :-kernel_size]
    return (total / float(kernel_size * kernel_size)).astype(np.float32)


def apply_mask_aware_mean_blur(target: np.ndarray, mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return target.astype(np.float32, copy=True)
    numerator = box_filter(target * mask, kernel_size)
    denominator = np.maximum(box_filter(mask, kernel_size), 1.0e-6)
    blurred = np.clip(numerator / denominator, 0.0, 1.0).astype(np.float32)
    return (blurred * mask + target * (1.0 - mask)).astype(np.float32)


def load_stage_arrays(frames, cameras, width: int, height: int, args: argparse.Namespace, blur_kernel: int):
    target_arrays = [load_target(camera.image_path, width, height).astype(np.float32) for camera in cameras]
    mask_arrays = [
        base.load_depth_mask(frame, width, height, args.mask_min_depth, args.mask_max_depth, args.mask_min_confidence).astype(np.float32)
        for frame in frames
    ]
    if args.target_blur_mode == "mean" and blur_kernel > 1:
        target_arrays = [
            apply_mask_aware_mean_blur(target, mask, blur_kernel)
            for target, mask in zip(target_arrays, mask_arrays, strict=True)
        ]
    targets = [mx.array(target[None, ...], dtype=mx.float32) for target in target_arrays]
    masks = [mx.array(mask[None, ...], dtype=mx.float32) for mask in mask_arrays]
    mask_fractions = [float(np.asarray(mx.sum(mask)) / float(width * height)) for mask in masks]
    return targets, masks, mask_fractions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/Users/yangdunfu/Documents/iOSProject/ScanProject/20260618_154636"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scanapp_depth_normalized_schedule_train"))
    parser.add_argument("--out-spz", type=Path, default=None)
    parser.add_argument("--out-model-npz", type=Path, default=None)
    parser.add_argument("--reference-width", type=int, default=1920)
    parser.add_argument("--reference-height", type=int, default=1440)
    parser.add_argument("--image-scale-factors", type=str, default="4,2")
    parser.add_argument("--stage-steps", type=str, default=None)
    parser.add_argument("--target-blur-mode", choices=("none", "mean"), default="mean")
    parser.add_argument("--target-blur-kernels", type=str, default="7,3")
    parser.add_argument("--target-points", type=int, default=262_144)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--eval-max-frames", type=int, default=0)
    parser.add_argument("--eval-frame-step", type=int, default=None)
    parser.add_argument("--eval-start-index", type=int, default=0)
    parser.add_argument("--normalize-world-space", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-random-gaussians", type=int, default=0)
    parser.add_argument("--random-gaussian-bounds-scale", type=float, default=1.05)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--opacity", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-sampling", choices=("sequential", "shuffle"), default="shuffle")
    parser.add_argument("--frame-shuffle-seed", type=int, default=None)
    parser.add_argument("--ssim-lambda", type=float, default=0.2)
    parser.add_argument("--ssim-window-size", type=int, default=11)
    parser.add_argument("--mask-min-depth", type=float, default=base.MIN_DEPTH_METERS)
    parser.add_argument("--mask-max-depth", type=float, default=base.DEFAULT_MASK_MAX_DEPTH_METERS)
    parser.add_argument("--mask-min-confidence", type=int, default=base.MIN_CONFIDENCE)
    parser.add_argument("--means-lr", type=float, default=1.6e-4)
    parser.add_argument("--scales-lr", type=float, default=5.0e-4)
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
    if args.target_points <= 0:
        raise ValueError("--target-points must be positive")
    if args.refine_stop_iter <= args.refine_start_iter:
        raise ValueError("--refine-stop-iter must be greater than --refine-start-iter")
    for name, value in [
        ("--reference-width", args.reference_width),
        ("--reference-height", args.reference_height),
        ("--init-scale", args.init_scale),
        ("--opacity", args.opacity),
        ("--means-lr", args.means_lr),
        ("--scales-lr", args.scales_lr),
        ("--opacities-lr", args.opacities_lr),
        ("--quats-lr", args.quats_lr),
        ("--sh0-lr", args.sh0_lr),
        ("--shn-lr", args.shn_lr),
    ]:
        base.validate_positive(name, value)

    scale_factors = parse_factor_schedule(args.image_scale_factors)
    stage_steps = split_stage_steps(args.steps, len(scale_factors), args.stage_steps)
    blur_kernels = parse_int_schedule(args.target_blur_kernels, len(scale_factors), "--target-blur-kernels")
    if args.target_blur_mode == "mean" and any(kernel % 2 == 0 for kernel in blur_kernels):
        raise ValueError("--target-blur-kernels entries must be odd for mean blur")
    stage_specs = [
        {
            "stage_index": index,
            "scale_factor": float(factor),
            "width": stage_size(args.reference_width, args.reference_height, factor)[0],
            "height": stage_size(args.reference_width, args.reference_height, factor)[1],
            "steps": int(steps),
            "target_blur_kernel": int(kernel if args.target_blur_mode == "mean" else 1),
        }
        for index, (factor, steps, kernel) in enumerate(zip(scale_factors, stage_steps, blur_kernels, strict=True))
    ]

    cache_limit_bytes = int(args.mlx_cache_limit_gb * 1024**3)
    previous_cache_limit = mx.set_cache_limit(cache_limit_bytes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    step_image_dir = args.out_dir / "step"
    step_image_count = 0
    if args.step_image_interval > 0:
        step_image_dir.mkdir(parents=True, exist_ok=True)

    first_stage = stage_specs[0]
    log(
        "loading ScanApp scene for normalized schedule "
        f"data={args.data} first_size={first_stage['width']}x{first_stage['height']} "
        f"target_points={args.target_points}"
    )
    scene = base.load_scanapp_scene(
        args.data,
        width=int(first_stage["width"]),
        height=int(first_stage["height"]),
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        start_index=args.start_index,
        target_points=args.target_points,
        seed=args.seed,
        keyframe_filter_enabled=False,
        keyframe_min_translation=0.0,
        keyframe_window=1,
        keyframe_sharpness_stride=8,
        keyframe_min_frames=1,
        min_motion_quality=0.0,
        shared_intrinsics_mode="median",
        per_frame_point_samples=0,
        consistency_filter_enabled=False,
        consistency_neighbor_window=1,
        consistency_min_views=1,
        consistency_abs_depth_tol=0.0,
        consistency_rel_depth_tol=0.0,
        consistency_keep_unobserved=True,
    )
    shared_intrinsics = base.shared_median_intrinsics(scene.frames)
    points = scene.points
    colors = scene.colors
    raw_point_count = scene.raw_point_count
    original_point_diagnostics = points_extent_diagnostics(points)

    world_transform = np.eye(4, dtype=np.float32)
    if args.normalize_world_space:
        c2ws = camera_c2ws(scene.cameras)
        normalized_c2ws, points, world_transform = normalize_scene(c2ws, points)
        first_cameras = [
            replace(camera, viewmat=np.linalg.inv(c2w).astype(np.float32))
            for camera, c2w in zip(scene.cameras, normalized_c2ws, strict=True)
        ]
    else:
        first_cameras = scene.cameras
    points, colors = base.append_random_gaussians(points, colors, args.num_random_gaussians, args.seed + 1009, args.random_gaussian_bounds_scale)
    normalized_point_diagnostics = points_extent_diagnostics(points)
    resolved_scene_scale = float(base.scene_scale_from_cameras_and_points(first_cameras, points) * 1.1 * args.global_scale)
    means_lr = float(args.means_lr * resolved_scene_scale)
    means_lr_final = means_lr * 0.01

    log(
        "initializing normalized schedule model "
        f"gaussians={points.shape[0]} raw_depth_points={raw_point_count} "
        f"scene_scale={resolved_scene_scale:.8f}"
    )
    log_scales = base.knn_log_scales_from_points(points, args.init_scale)
    model = base.init_sh_model_from_points(points, colors, log_scales, args.opacity, args.sh_degree, None)
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
    active_sh_degree = base.gsplat_active_sh_degree(0, args.sh_degree, args.sh_degree_interval)
    sh_degree_events = [{"step": 0, "active_sh_degree": int(active_sh_degree)}]

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
        width: int,
        height: int,
    ) -> mx.array:
        local = base.ScannerPointsSHModel.from_arrays(means, quats, log_scales_, features_dc, features_rest, opacity_logits)
        losses = []
        radii = []
        batch = int(viewmats.shape[1])
        for idx in range(batch):
            render = base.render_sh_model(
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
                base.loss_components(
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
    eval_frame_step = args.frame_step if args.eval_frame_step is None else args.eval_frame_step
    eval_frames = (
        base.select_frames(base.load_scanapp_frames(args.data), args.eval_max_frames, eval_frame_step, args.eval_start_index)
        if args.eval_max_frames > 0
        else []
    )

    sampler = FrameBatchSampler(
        frame_count=len(scene.frames),
        batch_size=args.batch_size,
        mode=args.frame_sampling,
        seed=args.seed + 7919 if args.frame_shuffle_seed is None else args.frame_shuffle_seed,
    )

    stage_summaries = []
    final_stats = []
    eval_final_stats = []
    final_targets = []
    final_cameras = []
    final_stage_initial_stats = []
    last_loss = None
    last_viewspace_grad = None
    last_viewspace_grad_norm = None
    global_step = 0

    for stage in stage_specs:
        stage_index = int(stage["stage_index"])
        width = int(stage["width"])
        height = int(stage["height"])
        stage_start_step = global_step + 1
        stage_end_step = global_step + int(stage["steps"])
        log(
            "starting image-scale stage "
            f"stage={stage_index} factor={stage['scale_factor']} size={width}x{height} "
            f"steps={stage['steps']} blur={stage['target_blur_kernel']} "
            f"global_steps={stage_start_step}-{stage_end_step}"
        )
        stage_cameras = base.load_scanapp_cameras(scene.frames, width, height, shared_intrinsics)
        stage_cameras = apply_world_transform_to_cameras(stage_cameras, world_transform) if args.normalize_world_space else stage_cameras
        stage_targets, stage_masks, stage_mask_fractions = load_stage_arrays(scene.frames, stage_cameras, width, height, args, int(stage["target_blur_kernel"]))
        eval_cameras = base.load_scanapp_cameras(eval_frames, width, height, shared_intrinsics) if eval_frames else []
        eval_cameras = apply_world_transform_to_cameras(eval_cameras, world_transform) if args.normalize_world_space else eval_cameras
        eval_targets, eval_masks, eval_mask_fractions = load_stage_arrays(eval_frames, eval_cameras, width, height, args, int(stage["target_blur_kernel"])) if eval_frames else ([], [], [])

        if stage_index > 0:
            strategy.reset_running_state(int(model.means.shape[1]))

        stage_initial_stats = base.evaluate_frames(
            model,
            stage_cameras,
            stage_targets,
            stage_masks,
            width,
            height,
            args.tile_size,
            active_sh_degree,
            args.ssim_lambda,
            args.ssim_window_size,
        )
        for _ in range(int(stage["steps"])):
            global_step += 1
            latest_lrs = {}
            for name, schedule in lr_schedules.items():
                lr = lr_for_step(schedule, global_step)
                optimizers[name].learning_rate = lr
                schedule["latest"] = float(lr)
                latest_lrs[name] = float(lr)
            if global_step == 1 or global_step == args.steps or global_step % args.log_interval == 0:
                for schedule in lr_schedules.values():
                    schedule["history"].append({"step": int(global_step), "lr": float(schedule["latest"])})

            next_active_sh_degree = base.gsplat_active_sh_degree(global_step, args.sh_degree, args.sh_degree_interval)
            if next_active_sh_degree != active_sh_degree:
                active_sh_degree = next_active_sh_degree
                sh_degree_events.append({"step": int(global_step), "active_sh_degree": int(active_sh_degree)})

            batch_ids = sampler.next_batch()
            batch_frame_indices = [int(stage_cameras[idx].index) for idx in batch_ids]
            target = target_batch_array(stage_targets, batch_ids)
            mask = target_batch_array(stage_masks, batch_ids)
            viewmats, ks = camera_batch_arrays(stage_cameras, batch_ids)
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
                width,
                height,
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
            strategy.after_optimizer_step(global_step, model, optimizers, "sh")

            if global_step == 1 or global_step == args.steps or global_step % args.log_interval == 0:
                log(
                    f"step={global_step:04d} stage={stage_index} size={width}x{height} frames={batch_frame_indices} "
                    f"sh={active_sh_degree} loss={last_loss:.8f} "
                    f"means_lr={latest_lrs['means']:.8g} scales_lr={latest_lrs['log_scales']:.8g} "
                    f"viewspace_grad_norm={last_viewspace_grad_norm:.8f}"
                )
            if args.step_image_interval > 0 and global_step % args.step_image_interval == 0:
                step_image_count += 1
                image = base.render_step_grid(model, stage_cameras, width, height, args.tile_size, active_sh_degree)
                out_path = step_image_dir / f"out_{step_image_count:06d}.png"
                base.write_png(out_path, image)
                log(f"wrote step image step={global_step} path={out_path}")

        stage_final_stats = base.evaluate_frames(
            model,
            stage_cameras,
            stage_targets,
            stage_masks,
            width,
            height,
            args.tile_size,
            active_sh_degree,
            args.ssim_lambda,
            args.ssim_window_size,
        )
        stage_eval_final_stats = (
            base.evaluate_frames(model, eval_cameras, eval_targets, eval_masks, width, height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)
            if eval_cameras
            else []
        )
        stage_summaries.append(
            {
                "stage_index": stage_index,
                "scale_factor": float(stage["scale_factor"]),
                "width": width,
                "height": height,
                "steps": int(stage["steps"]),
                "target_blur_kernel": int(stage["target_blur_kernel"]),
                "global_step_start": int(stage_start_step),
                "global_step_end": int(stage_end_step),
                "initial_mean_loss": base.mean_loss(stage_initial_stats),
                "final_mean_loss": base.mean_loss(stage_final_stats),
                "eval_final_mean_loss": base.mean_loss(stage_eval_final_stats) if stage_eval_final_stats else None,
                "mask_fraction_min": float(np.min(stage_mask_fractions)) if stage_mask_fractions else 0.0,
                "mask_fraction_mean": float(np.mean(stage_mask_fractions)) if stage_mask_fractions else 0.0,
                "mask_fraction_max": float(np.max(stage_mask_fractions)) if stage_mask_fractions else 0.0,
                "eval_mask_fraction_mean": float(np.mean(eval_mask_fractions)) if eval_mask_fractions else None,
                "strategy_events_so_far": int(len(strategy.events)),
            }
        )
        final_stats = stage_final_stats
        eval_final_stats = stage_eval_final_stats
        final_targets = stage_targets
        final_cameras = stage_cameras
        final_stage_initial_stats = stage_initial_stats

    if last_loss is None or not np.isfinite(base.mean_loss(final_stats)):
        raise AssertionError("ScanApp normalized schedule training loss should be finite")
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("ScanApp normalized schedule training expected nonzero viewspace_points gradient")

    for initial, final, target in zip(final_stage_initial_stats, final_stats, final_targets, strict=True):
        frame_index = final["frame_index"]
        base.write_png(args.out_dir / f"compare_frame_{frame_index:05d}.png", image_to_u8(concat_compare(np.asarray(target[0]), initial["image"], final["image"])))

    out_spz = args.out_spz if args.out_spz is not None else args.out_dir / "trained_scanapp_depth_normalized_schedule.spz"
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
        for initial, final in zip(final_stage_initial_stats, final_stats, strict=True)
    ]
    eval_frame_summaries = [
        {
            "frame_index": int(final["frame_index"]),
            "final_loss": float(final["loss"]),
            "final_loss_components": final["loss_components"],
            "final_psnr": float(final["psnr"]),
            "final_visible_gaussians": int(final["visible_gaussians"]),
            "final_intersections": int(final["intersections"]),
        }
        for final in eval_final_stats
    ]
    summary = {
        "dataset_type": "scanapp_depth_normalized_image_scale_schedule",
        "experiment": "ScanApp depth with median K, gsplat default-like strategy, normalized world space, and staged image scale factors.",
        "dataset": str(args.data),
        "reference_width": int(args.reference_width),
        "reference_height": int(args.reference_height),
        "image_scale_factors": [float(item) for item in scale_factors],
        "target_blur": {
            "mode": args.target_blur_mode,
            "kernels": [int(item["target_blur_kernel"]) for item in stage_specs],
            "mask_aware": True,
            "formula": "box_filter(rgb * mask) / max(box_filter(mask), eps), blended back only inside mask",
        },
        "stage_specs": stage_specs,
        "stage_summaries": stage_summaries,
        "steps": int(args.steps),
        "raw_point_count": int(raw_point_count),
        "candidate_point_count": int(raw_point_count),
        "sampled_point_count": int(scene.sampled_point_count),
        "target_point_count": int(args.target_points),
        "retained_point_count": int(scene.retained_point_count),
        "exported_gaussians": int(exported_gaussians),
        "point_cloud_gaussians": int(points.shape[0] - args.num_random_gaussians),
        "random_gaussians": int(args.num_random_gaussians),
        "frames": len(final_cameras),
        "depth_frames": len(scene.frames),
        "eval_frames": len(eval_final_stats),
        "confidence_frame_count": int(scene.confidence_frame_count),
        "confidence_kept_count": int(scene.confidence_kept_count),
        "confidence_rejected_count": int(scene.confidence_rejected_count),
        "depth_valid_count": int(scene.depth_valid_count),
        "depth_rejected_count": int(scene.depth_rejected_count),
        "colorized_point_count": int(scene.colorized_point_count),
        "mask_config": {
            "min_depth_meters": float(args.mask_min_depth),
            "max_depth_meters": float(args.mask_max_depth),
            "min_confidence": int(args.mask_min_confidence),
        },
        "shared_intrinsics_mode": "median",
        "shared_intrinsics": shared_intrinsics.astype(float).tolist(),
        "world_normalization": {
            "enabled": bool(args.normalize_world_space),
            "spz_world_space": "normalized" if args.normalize_world_space else "scanapp_transformed_meter",
            "transform": world_transform.astype(float).tolist(),
            "original_point_extent": original_point_diagnostics,
            "trained_point_extent": normalized_point_diagnostics,
        },
        "scene_scale": float(resolved_scene_scale),
        "resolved_scene_scale": float(resolved_scene_scale),
        "initialization": {
            "type": "scanapp_depth_knn_reconstruction",
            "scale_rule": "average distance to 3 nearest neighbors times init_scale",
            "sampling_rule": "all valid depth pixels are reconstructed first, then globally sampled to target_point_count",
            "init_scale": float(args.init_scale),
            "opacity": float(args.opacity),
            "log_scale_min": float(log_scales.min()),
            "log_scale_mean": float(log_scales.mean()),
            "log_scale_max": float(log_scales.max()),
        },
        "loss_config": {
            "mode": "masked_l1_ssim",
            "formula": "(1 - ssim_lambda) * masked_L1 + ssim_lambda * (1 - SSIM(render * mask + target * (1 - mask), target))",
            "ssim_lambda": float(args.ssim_lambda),
            "ssim_window_size": int(args.ssim_window_size),
        },
        "learning_rate_schedule": lr_schedules,
        "final_mean_loss": float(base.mean_loss(final_stats)),
        "eval_final_mean_loss": base.mean_loss(eval_final_stats) if eval_final_stats else None,
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
            "formula": "min(global_step // sh_degree_interval, sh_degree)",
            "events": sh_degree_events,
        },
        "final_opacity_diagnostics": opacity_diagnostics(model),
        "refinement_strategy": strategy.summary(),
        "dataloader": sampler.summary(final_cameras),
        "frame_summaries": frame_summaries,
        "eval_frame_summaries": eval_frame_summaries,
        "mlx_cache_limit_bytes": int(cache_limit_bytes),
        "mlx_cache_limit_gb": float(args.mlx_cache_limit_gb),
        "mlx_previous_cache_limit_bytes": int(previous_cache_limit),
        "step_image_interval": int(args.step_image_interval),
        "step_image_count": int(step_image_count),
        "step_image_dir": str(step_image_dir) if args.step_image_interval > 0 else None,
    }
    out_model_npz = args.out_model_npz if args.out_model_npz is not None else args.out_dir / "trained_model_params.npz"
    summary["model_npz"] = str(out_model_npz)
    save_model_parameters_npz(out_model_npz, model, "sh", active_sh_degree, args.sh_degree, summary)
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_spz.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(
        "ScanApp normalized schedule training ok "
        f"final_mean_loss={summary['final_mean_loss']:.8f} "
        f"last_viewspace_grad_norm={last_viewspace_grad_norm:.8f} "
        f"spz={out_spz} bytes={spz_size} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
