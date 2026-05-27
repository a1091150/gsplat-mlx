#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.optimizers import Adam

from render_random_3dgs_png import write_png
from scanner_dataset_random_render_smoke import collect_frames, load_camera, load_target
from scanner_points_alignment_render import prepare_points
from train_scanner_random_3dgs_mlx import (
    camera_arrays,
    concat_compare,
    evaluate_frames,
    mean_loss,
    mx_logit,
    save_frame_targets,
)
from train_tiny_3dgs_mlx import Tiny3DGSModel, image_to_u8, normalize_quats, render_model


def init_model_from_points(
    points: np.ndarray,
    colors: np.ndarray,
    point_scale: float,
    opacity: float,
) -> Tiny3DGSModel:
    n = int(points.shape[0])
    means = mx.array(points[None, ...], dtype=mx.float32)
    quats = mx.zeros((1, n, 4), dtype=mx.float32) + mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32)
    log_scales = mx.full((1, n, 3), np.log(point_scale), dtype=mx.float32)
    color_logits = mx_logit(mx.array(colors[None, None, ...], dtype=mx.float32))
    opacity_logits = mx_logit(mx.full((1, n), opacity, dtype=mx.float32))
    return Tiny3DGSModel.from_arrays(
        means,
        normalize_quats(quats),
        log_scales,
        color_logits,
        opacity_logits,
    )


def quat_wxyz_to_rotmat(quats: np.ndarray) -> np.ndarray:
    q = quats / np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1.0e-8, None)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    q = np.empty((rot.shape[0], 4), dtype=np.float32)
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    mask = trace > 0.0
    if np.any(mask):
        s = np.sqrt(trace[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (rot[mask, 2, 1] - rot[mask, 1, 2]) / s
        q[mask, 2] = (rot[mask, 0, 2] - rot[mask, 2, 0]) / s
        q[mask, 3] = (rot[mask, 1, 0] - rot[mask, 0, 1]) / s
    mask_x = (~mask) & (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2])
    if np.any(mask_x):
        s = np.sqrt(1.0 + rot[mask_x, 0, 0] - rot[mask_x, 1, 1] - rot[mask_x, 2, 2]) * 2.0
        q[mask_x, 0] = (rot[mask_x, 2, 1] - rot[mask_x, 1, 2]) / s
        q[mask_x, 1] = 0.25 * s
        q[mask_x, 2] = (rot[mask_x, 0, 1] + rot[mask_x, 1, 0]) / s
        q[mask_x, 3] = (rot[mask_x, 0, 2] + rot[mask_x, 2, 0]) / s
    mask_y = (~mask) & (~mask_x) & (rot[:, 1, 1] > rot[:, 2, 2])
    if np.any(mask_y):
        s = np.sqrt(1.0 + rot[mask_y, 1, 1] - rot[mask_y, 0, 0] - rot[mask_y, 2, 2]) * 2.0
        q[mask_y, 0] = (rot[mask_y, 0, 2] - rot[mask_y, 2, 0]) / s
        q[mask_y, 1] = (rot[mask_y, 0, 1] + rot[mask_y, 1, 0]) / s
        q[mask_y, 2] = 0.25 * s
        q[mask_y, 3] = (rot[mask_y, 1, 2] + rot[mask_y, 2, 1]) / s
    mask_z = (~mask) & (~mask_x) & (~mask_y)
    if np.any(mask_z):
        s = np.sqrt(1.0 + rot[mask_z, 2, 2] - rot[mask_z, 0, 0] - rot[mask_z, 1, 1]) * 2.0
        q[mask_z, 0] = (rot[mask_z, 1, 0] - rot[mask_z, 0, 1]) / s
        q[mask_z, 1] = (rot[mask_z, 0, 2] + rot[mask_z, 2, 0]) / s
        q[mask_z, 2] = (rot[mask_z, 1, 2] + rot[mask_z, 2, 1]) / s
        q[mask_z, 3] = 0.25 * s
    return q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1.0e-8, None)


def positions_to_spz(means: np.ndarray) -> np.ndarray:
    out = np.empty_like(means, dtype=np.float32)
    out[:, 0] = means[:, 0]
    out[:, 1] = -means[:, 2]
    out[:, 2] = means[:, 1]
    return out


def quats_to_spz(quats: np.ndarray) -> np.ndarray:
    axis3 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot = quat_wxyz_to_rotmat(quats)
    return rotmat_to_quat_wxyz(axis3 @ rot @ axis3.T)


def export_trained_spz(
    path: Path,
    model: Tiny3DGSModel,
    color_mode: str,
) -> int:
    try:
        import spz
    except ImportError as exc:
        raise ImportError("The 'spz' Python package is required for SPZ export.") from exc

    mx.eval(model.means, model.log_scales, model.normalized_quats, model.color_logits, model.opacity_logits)
    means = np.asarray(model.means[0], dtype=np.float32)
    log_scales = np.asarray(model.log_scales[0], dtype=np.float32)
    quats = np.asarray(model.normalized_quats[0], dtype=np.float32)
    colors = np.asarray(model.colors[0, 0], dtype=np.float32)
    opacity_logits = np.asarray(model.opacity_logits[0], dtype=np.float32)

    cloud = spz.GaussianCloud()
    cloud.antialiased = True
    cloud.positions = positions_to_spz(means).reshape(-1).astype(np.float32)
    cloud.scales = log_scales.reshape(-1).astype(np.float32)
    cloud.rotations = quats_to_spz(quats).reshape(-1).astype(np.float32)
    cloud.alphas = opacity_logits.reshape(-1).astype(np.float32)
    if color_mode == "rgb":
        cloud.colors = np.clip(colors, 0.0, 1.0).reshape(-1).astype(np.float32)
    elif color_mode == "sh0":
        sh_c0 = 0.28209479177387814
        cloud.colors = ((np.clip(colors, 0.0, 1.0) - 0.5) / sh_c0).reshape(-1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported color mode: {color_mode}")
    cloud.sh_degree = 0
    cloud.sh = np.array([], dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    opts = spz.PackOptions()
    ok = spz.save_spz(cloud, opts, str(path))
    if not ok:
        raise RuntimeError(f"failed to save spz to {path}")
    return int(means.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/Users/yangdunfu/Downloads/2026_05_04_16_51_29"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scanner_points_multiview_train"))
    parser.add_argument("--out-spz", type=Path, default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--point-scale", type=float, default=0.01)
    parser.add_argument("--opacity", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr-means", type=float, default=2.0e-3)
    parser.add_argument("--lr-colors", type=float, default=2.0e-2)
    parser.add_argument("--lr-opacity", type=float, default=5.0e-3)
    parser.add_argument("--lr-scales", type=float, default=1.0e-3)
    parser.add_argument("--lr-quats", type=float, default=1.0e-3)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--spz-color-mode", choices=("rgb", "sh0"), default="rgb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = collect_frames(args.data, args.max_frames, args.frame_step, args.start_index)
    cameras = [load_camera(frame, args.width, args.height) for frame in frames]
    targets = [
        mx.array(load_target(camera.image_path, args.width, args.height)[None, ...], dtype=mx.float32)
        for camera in cameras
    ]
    points, colors, raw_point_count = prepare_points(args.data, args.max_points, args.seed)
    model = init_model_from_points(points, colors, args.point_scale, args.opacity)
    save_frame_targets(args.out_dir, cameras, targets)

    initial_stats = evaluate_frames(model, cameras, targets, args.width, args.height, args.tile_size)
    initial_mean_loss = mean_loss(initial_stats)
    for item in initial_stats:
        write_png(
            args.out_dir / f"step_0000_frame_{item['frame_index']:05d}.png",
            image_to_u8(item["image"]),
        )

    def loss_fn(
        means: mx.array,
        quats: mx.array,
        log_scales: mx.array,
        color_logits: mx.array,
        opacity_logits: mx.array,
        viewspace_points: mx.array,
        viewmats: mx.array,
        Ks: mx.array,
        target: mx.array,
    ) -> mx.array:
        local = Tiny3DGSModel.from_arrays(means, quats, log_scales, color_logits, opacity_logits)
        render = render_model(local, viewspace_points, viewmats, Ks, args.width, args.height, args.tile_size)
        diff = render["render_colors"] - target
        return mx.mean(diff * diff)

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))
    optimizers = {
        "means": Adam(learning_rate=args.lr_means),
        "quats": Adam(learning_rate=args.lr_quats),
        "log_scales": Adam(learning_rate=args.lr_scales),
        "color_logits": Adam(learning_rate=args.lr_colors),
        "opacity_logits": Adam(learning_rate=args.lr_opacity),
    }

    last_loss = None
    last_viewspace_grad = None
    last_viewspace_grad_norm = None
    for step in range(1, args.steps + 1):
        view_id = (step - 1) % len(cameras)
        camera = cameras[view_id]
        target = targets[view_id]
        viewmats, Ks = camera_arrays(camera)
        viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)
        loss, grads = grad_fn(
            model.means,
            model.quats,
            model.log_scales,
            model.color_logits,
            model.opacity_logits,
            viewspace_points,
            viewmats,
            Ks,
            target,
        )
        d_means, d_quats, d_log_scales, d_color_logits, d_opacity_logits, d_viewspace = grads
        mx.eval(loss, d_viewspace)
        last_loss = float(np.asarray(loss))
        last_viewspace_grad = d_viewspace
        last_viewspace_grad_norm = float(np.linalg.norm(np.asarray(d_viewspace)))

        optimizers["means"].update(model, {"means": d_means})
        optimizers["quats"].update(model, {"quats": d_quats})
        optimizers["log_scales"].update(model, {"log_scales": d_log_scales})
        optimizers["color_logits"].update(model, {"color_logits": d_color_logits})
        optimizers["opacity_logits"].update(model, {"opacity_logits": d_opacity_logits})
        model.quats = normalize_quats(model.quats)
        mx.eval(model.means, model.quats, model.log_scales, model.color_logits, model.opacity_logits)

        if step == 1 or step == args.steps or step % args.log_interval == 0:
            print(
                f"step={step:04d} frame={camera.index:05d} "
                f"loss={last_loss:.8f} viewspace_grad_norm={last_viewspace_grad_norm:.8f}"
            )

    final_stats = evaluate_frames(model, cameras, targets, args.width, args.height, args.tile_size)
    final_mean_loss = mean_loss(final_stats)
    target_images = [np.asarray(target[0], dtype=np.float32) for target in targets]
    for initial, final, target_image in zip(initial_stats, final_stats, target_images, strict=True):
        frame_index = final["frame_index"]
        write_png(args.out_dir / f"step_{args.steps:04d}_frame_{frame_index:05d}.png", image_to_u8(final["image"]))
        write_png(
            args.out_dir / f"compare_frame_{frame_index:05d}.png",
            image_to_u8(concat_compare(target_image, initial["image"], final["image"])),
        )

    if last_loss is None or not np.isfinite(final_mean_loss):
        raise AssertionError("scanner points multi-view training loss should be finite")
    if final_mean_loss > initial_mean_loss * 1.05:
        raise AssertionError(
            "scanner points multi-view training loss should not diverge: "
            f"initial_mean={initial_mean_loss:.8f} final_mean={final_mean_loss:.8f}"
        )
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("scanner points multi-view training expected nonzero viewspace_points gradient")

    out_spz = args.out_spz if args.out_spz is not None else args.out_dir / "trained_scanner_points.spz"
    exported_gaussians = export_trained_spz(out_spz, model, args.spz_color_mode)
    spz_size = out_spz.stat().st_size
    if spz_size <= 0:
        raise AssertionError(f"SPZ output is empty: {out_spz}")

    frame_summaries = []
    for initial, final in zip(initial_stats, final_stats, strict=True):
        frame_summaries.append(
            {
                "frame_index": int(final["frame_index"]),
                "initial_loss": float(initial["loss"]),
                "final_loss": float(final["loss"]),
                "initial_psnr": float(initial["psnr"]),
                "final_psnr": float(final["psnr"]),
                "initial_visible_gaussians": int(initial["visible_gaussians"]),
                "final_visible_gaussians": int(final["visible_gaussians"]),
                "initial_intersections": int(initial["intersections"]),
                "final_intersections": int(final["intersections"]),
            }
        )
    summary = {
        "dataset": str(args.data),
        "width": args.width,
        "height": args.height,
        "raw_point_count": raw_point_count,
        "exported_gaussians": exported_gaussians,
        "max_points": args.max_points,
        "frames": len(cameras),
        "steps": args.steps,
        "initial_mean_loss": initial_mean_loss,
        "final_mean_loss": final_mean_loss,
        "last_viewspace_grad_norm": last_viewspace_grad_norm,
        "spz": str(out_spz),
        "spz_file_size_bytes": spz_size,
        "spz_position_convention": "[x, -z, y]",
        "spz_scale_convention": "trained log_scales",
        "spz_opacity_convention": "trained opacity logits",
        "spz_rotation_convention": "trained wxyz quats transformed by scanner axis3",
        "spz_color_mode": args.spz_color_mode,
        "frame_summaries": frame_summaries,
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_spz.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for item in frame_summaries:
        print(
            f"frame={item['frame_index']:05d} "
            f"loss={item['initial_loss']:.8f}->{item['final_loss']:.8f} "
            f"psnr={item['initial_psnr']:.2f}->{item['final_psnr']:.2f} "
            f"visible={item['initial_visible_gaussians']}->{item['final_visible_gaussians']} "
            f"intersections={item['initial_intersections']}->{item['final_intersections']}"
        )
    print(
        "scanner points multi-view training ok "
        f"initial_mean_loss={initial_mean_loss:.8f} "
        f"final_mean_loss={final_mean_loss:.8f} "
        f"last_viewspace_grad_norm={last_viewspace_grad_norm:.8f} "
        f"spz={out_spz} bytes={spz_size} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
