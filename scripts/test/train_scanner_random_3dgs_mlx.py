#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.optimizers import Adam

from render_random_3dgs_png import write_png
from scanner_dataset_random_render_smoke import (
    collect_frames,
    load_camera,
    load_target,
    random_gaussians_for_cameras,
)
from train_tiny_3dgs_mlx import (
    Tiny3DGSModel,
    image_to_u8,
    normalize_quats,
    render_model,
    save_render,
)


def mx_logit(values: mx.array) -> mx.array:
    clipped = mx.minimum(mx.maximum(values, 1.0e-5), 1.0 - 1.0e-5)
    return mx.log(clipped / (1.0 - clipped))


def camera_arrays(camera) -> tuple[mx.array, mx.array]:
    return (
        mx.array(camera.viewmat[None, None, ...], dtype=mx.float32),
        mx.array(camera.K[None, None, ...], dtype=mx.float32),
    )


def init_model_from_scanner_cameras(
    cameras,
    num_gaussians: int,
    width: int,
    height: int,
    seed: int,
) -> Tiny3DGSModel:
    means, quats, scales, colors, opacities = random_gaussians_for_cameras(
        cameras,
        num_gaussians,
        width,
        height,
        seed,
    )
    return Tiny3DGSModel.from_arrays(
        means,
        quats,
        mx.log(mx.maximum(scales, 1.0e-5)),
        mx_logit(colors),
        mx_logit(opacities),
    )


def save_frame_targets(out_dir: Path, cameras, targets: list[mx.array]) -> None:
    for camera, target in zip(cameras, targets, strict=True):
        mx.eval(target)
        write_png(
            out_dir / f"target_frame_{camera.index:05d}.png",
            image_to_u8(np.asarray(target[0], dtype=np.float32)),
        )


def save_frame_renders(
    out_dir: Path,
    prefix: str,
    model: Tiny3DGSModel,
    cameras,
    width: int,
    height: int,
    tile_size: int,
) -> None:
    for camera in cameras:
        viewmats, Ks = camera_arrays(camera)
        viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)
        render = render_model(model, viewspace_points, viewmats, Ks, width, height, tile_size)
        save_render(out_dir / f"{prefix}_frame_{camera.index:05d}.png", render["render_colors"])


def mean_loss(
    model: Tiny3DGSModel,
    cameras,
    targets: list[mx.array],
    width: int,
    height: int,
    tile_size: int,
) -> float:
    losses = []
    for camera, target in zip(cameras, targets, strict=True):
        viewmats, Ks = camera_arrays(camera)
        viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)
        render = render_model(model, viewspace_points, viewmats, Ks, width, height, tile_size)
        diff = render["render_colors"] - target
        loss = mx.mean(diff * diff)
        mx.eval(loss)
        losses.append(float(np.asarray(loss)))
    return float(np.mean(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/Users/yangdunfu/Downloads/2026_05_04_16_51_29"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scanner_random_3dgs_train"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-gaussians", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--lr-means", type=float, default=1.0e-2)
    parser.add_argument("--lr-colors", type=float, default=5.0e-2)
    parser.add_argument("--lr-opacity", type=float, default=1.0e-2)
    parser.add_argument("--lr-scales", type=float, default=4.0e-3)
    parser.add_argument("--lr-quats", type=float, default=3.0e-3)
    parser.add_argument("--log-interval", type=int, default=10)
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
    model = init_model_from_scanner_cameras(
        cameras,
        args.num_gaussians,
        args.width,
        args.height,
        args.seed,
    )
    save_frame_targets(args.out_dir, cameras, targets)
    save_frame_renders(args.out_dir, "step_0000", model, cameras, args.width, args.height, args.tile_size)

    initial_mean_loss = mean_loss(model, cameras, targets, args.width, args.height, args.tile_size)

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
        local = Tiny3DGSModel.from_arrays(
            means,
            quats,
            log_scales,
            color_logits,
            opacity_logits,
        )
        render = render_model(
            local,
            viewspace_points,
            viewmats,
            Ks,
            args.width,
            args.height,
            args.tile_size,
        )
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
    for step in range(1, args.steps + 1):
        view_id = (step - 1) % len(cameras)
        camera = cameras[view_id]
        target = targets[view_id]
        viewmats, Ks = camera_arrays(camera)
        viewspace_points = mx.zeros((1, 1, args.num_gaussians, 2), dtype=mx.float32)
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

        optimizers["means"].update(model, {"means": d_means})
        optimizers["quats"].update(model, {"quats": d_quats})
        optimizers["log_scales"].update(model, {"log_scales": d_log_scales})
        optimizers["color_logits"].update(model, {"color_logits": d_color_logits})
        optimizers["opacity_logits"].update(model, {"opacity_logits": d_opacity_logits})
        model.quats = normalize_quats(model.quats)
        mx.eval(
            model.means,
            model.quats,
            model.log_scales,
            model.color_logits,
            model.opacity_logits,
        )

        if step == 1 or step == args.steps or step % args.log_interval == 0:
            print(f"step={step:04d} frame={camera.index:05d} loss={last_loss:.8f}")

    final_mean_loss = mean_loss(model, cameras, targets, args.width, args.height, args.tile_size)
    save_frame_renders(
        args.out_dir,
        f"step_{args.steps:04d}",
        model,
        cameras,
        args.width,
        args.height,
        args.tile_size,
    )

    if last_loss is None or not np.isfinite(final_mean_loss):
        raise AssertionError("scanner random training loss should be finite")
    if final_mean_loss > initial_mean_loss * 1.05:
        raise AssertionError(
            "scanner random training loss should not diverge: "
            f"initial_mean={initial_mean_loss:.8f} final_mean={final_mean_loss:.8f}"
        )
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("scanner random training expected nonzero viewspace_points gradient")

    print(
        "scanner random 3dgs mlx training ok "
        f"initial_mean_loss={initial_mean_loss:.8f} "
        f"final_mean_loss={final_mean_loss:.8f} "
        f"frames={len(cameras)} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
