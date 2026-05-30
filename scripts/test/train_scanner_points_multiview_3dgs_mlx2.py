#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.optimizers import Adam

from render_random_3dgs_png import write_png
from scanner_dataset_utils import collect_frames, load_camera, load_target
from train_360_points_multiview_3dgs_mlx import (
    MAX_SUPPORTED_SH_DEGREE,
    evaluate_frames,
    gsplat_active_sh_degree,
    init_sh_model_from_points,
    knn_log_scales_from_points,
    log,
    loss_components,
    mean_loss,
    validate_positive,
)
from scanner_points_training_utils import (
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
    prepare_points,
    points_extent_diagnostics,
    render_sh_model,
    save_model_parameters_npz,
    spz_export_diagnostics,
    target_batch_array,
)


def camera_center_from_viewmat(viewmat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(viewmat).astype(np.float32)[:3, 3]


def scene_scale_from_cameras(cameras: list) -> float:
    centers = np.stack([camera_center_from_viewmat(camera.viewmat) for camera in cameras], axis=0)
    center = np.mean(centers, axis=0)
    return max(float(np.max(np.linalg.norm(centers - center[None, :], axis=1))), 1.0e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/Users/yangdunfu/Downloads/2026_05_04_16_51_29"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scanner_points_multiview_train_spz2"))
    parser.add_argument("--out-spz", type=Path, default=None)
    parser.add_argument("--out-model-npz", type=Path, default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=0)
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
    log(f"loading scanner dataset data={args.data} size={args.width}x{args.height}")
    frames = collect_frames(args.data, args.max_frames, args.frame_step, args.start_index)
    cameras = [load_camera(frame, args.width, args.height) for frame in frames]
    scene_frames = collect_frames(args.data, 0, 1, 0)
    scene_cameras = [load_camera(frame, args.width, args.height) for frame in scene_frames]
    targets = [mx.array(load_target(camera.image_path, args.width, args.height)[None, ...], dtype=mx.float32) for camera in cameras]
    scene_scale = scene_scale_from_cameras(scene_cameras)
    resolved_scene_scale = float(scene_scale * 1.1 * args.global_scale)
    log(f"loaded targets train_frames={len(cameras)} scene_scale={scene_scale:.8f}")

    log("preparing scanner points.ply")
    points, colors, raw_point_count = prepare_points(args.data, args.max_points, args.seed)
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
    initial_stats = evaluate_frames(model, cameras, targets, args.width, args.height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)
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
            losses.append(loss_components(render["render_colors"], target[idx : idx + 1], args.ssim_lambda, args.ssim_window_size)["loss"])
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
    final_stats = evaluate_frames(model, cameras, targets, args.width, args.height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)
    final_mean_loss = mean_loss(final_stats)

    for initial, final, target in zip(initial_stats, final_stats, targets, strict=True):
        frame_index = final["frame_index"]
        write_png(args.out_dir / f"compare_frame_{frame_index:05d}.png", image_to_u8(concat_compare(np.asarray(target[0]), initial["image"], final["image"])))

    if last_loss is None or not np.isfinite(final_mean_loss):
        raise AssertionError("scanner gsplat-style training loss should be finite")
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("scanner gsplat-style training expected nonzero viewspace_points gradient")

    out_spz = args.out_spz if args.out_spz is not None else args.out_dir / "trained_scanner_points_spz2.spz"
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
        "dataset_type": "scanner_points_gsplat_style",
        "dataset": str(args.data),
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
            "type": "scanner_points_ply",
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
            "knn_scale_init": True,
        },
        "dataloader": sampler.summary(cameras),
        "loss_config": {
            "mode": "l1_ssim",
            "formula": "(1 - ssim_lambda) * L1 + ssim_lambda * (1 - SSIM)",
            "ssim_lambda": float(args.ssim_lambda),
            "ssim_window_size": int(args.ssim_window_size),
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
        "scanner gsplat-style points multi-view training ok "
        f"initial_mean_loss={initial_mean_loss:.8f} final_mean_loss={final_mean_loss:.8f} "
        f"last_viewspace_grad_norm={last_viewspace_grad_norm:.8f} "
        f"spz={out_spz} bytes={spz_size} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
