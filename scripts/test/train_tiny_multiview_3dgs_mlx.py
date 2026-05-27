#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.optimizers import Adam

from train_tiny_3dgs_mlx import (
    Tiny3DGSModel,
    image_to_u8,
    normalize_quats,
    render_model,
    save_render,
)
from render_random_3dgs_png import write_png


def make_camera(
    width: int,
    height: int,
    x_shift: float,
    focal_scale: float = 0.72,
) -> tuple[mx.array, mx.array]:
    viewmats = mx.array(
        [[[[1.0, 0.0, 0.0, x_shift],
           [0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )
    focal = focal_scale * float(min(width, height))
    Ks = mx.array(
        [[[[focal, 0.0, width * 0.5],
           [0.0, focal, height * 0.5],
           [0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )
    return viewmats, Ks


def make_cameras(width: int, height: int, num_views: int) -> list[tuple[mx.array, mx.array]]:
    if num_views <= 1:
        return [make_camera(width, height, 0.0)]
    shifts = np.linspace(-0.18, 0.18, num_views, dtype=np.float32)
    return [make_camera(width, height, float(shift)) for shift in shifts]


def render_targets(
    target_model: Tiny3DGSModel,
    cameras: list[tuple[mx.array, mx.array]],
    width: int,
    height: int,
    tile_size: int,
) -> list[mx.array]:
    targets = []
    for viewmats, Ks in cameras:
        viewspace_points = mx.zeros((1, 1, target_model.means.shape[1], 2), dtype=mx.float32)
        render = render_model(target_model, viewspace_points, viewmats, Ks, width, height, tile_size)
        target = mx.stop_gradient(render["render_colors"])
        mx.eval(target, render["tiles_per_gauss"])
        if int(np.sum(np.asarray(render["tiles_per_gauss"]))) <= 0:
            raise AssertionError("multi-view target expected nonzero tile intersections")
        targets.append(target)
    return targets


def save_target(path: Path, image: mx.array) -> None:
    mx.eval(image)
    write_png(path, image_to_u8(np.asarray(image[0], dtype=np.float32)))


def save_views(
    out_dir: Path,
    prefix: str,
    model: Tiny3DGSModel,
    cameras: list[tuple[mx.array, mx.array]],
    width: int,
    height: int,
    tile_size: int,
) -> None:
    for view_id, (viewmats, Ks) in enumerate(cameras):
        viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)
        render = render_model(model, viewspace_points, viewmats, Ks, width, height, tile_size)
        save_render(out_dir / f"{prefix}_view_{view_id:02d}.png", render["render_colors"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tiny_3dgs_multiview_train"))
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--target-seed", type=int, default=119)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--num-gaussians", type=int, default=1024)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--lr-means", type=float, default=2.0e-2)
    parser.add_argument("--lr-colors", type=float, default=6.0e-2)
    parser.add_argument("--lr-opacity", type=float, default=2.0e-2)
    parser.add_argument("--lr-scales", type=float, default=8.0e-3)
    parser.add_argument("--lr-quats", type=float, default=5.0e-3)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)

    model = Tiny3DGSModel(args.seed, args.num_gaussians)
    target_model = Tiny3DGSModel(args.target_seed, args.num_gaussians)
    cameras = make_cameras(args.width, args.height, args.num_views)
    targets = render_targets(target_model, cameras, args.width, args.height, args.tile_size)
    for view_id, target in enumerate(targets):
        save_target(args.out_dir / f"target_view_{view_id:02d}.png", target)
    save_views(args.out_dir, "step_0000", model, cameras, args.width, args.height, args.tile_size)

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

    initial_losses = []
    for viewmats, Ks, target in zip(
        [camera[0] for camera in cameras],
        [camera[1] for camera in cameras],
        targets,
        strict=True,
    ):
        viewspace_points = mx.zeros((1, 1, args.num_gaussians, 2), dtype=mx.float32)
        loss = loss_fn(
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
        mx.eval(loss)
        initial_losses.append(float(np.asarray(loss)))
    initial_mean_loss = float(np.mean(initial_losses))

    last_loss = None
    last_viewspace_grad = None
    for step in range(1, args.steps + 1):
        view_id = (step - 1) % len(cameras)
        viewmats, Ks = cameras[view_id]
        target = targets[view_id]
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
            print(f"step={step:04d} view={view_id:02d} loss={last_loss:.8f}")

    final_losses = []
    for viewmats, Ks, target in zip(
        [camera[0] for camera in cameras],
        [camera[1] for camera in cameras],
        targets,
        strict=True,
    ):
        viewspace_points = mx.zeros((1, 1, args.num_gaussians, 2), dtype=mx.float32)
        loss = loss_fn(
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
        mx.eval(loss)
        final_losses.append(float(np.asarray(loss)))
    final_mean_loss = float(np.mean(final_losses))
    save_views(
        args.out_dir,
        f"step_{args.steps:04d}",
        model,
        cameras,
        args.width,
        args.height,
        args.tile_size,
    )

    if last_loss is None or not np.isfinite(final_mean_loss):
        raise AssertionError("multi-view training loss should be finite")
    if final_mean_loss > initial_mean_loss * 1.05:
        raise AssertionError(
            "multi-view training loss should not diverge: "
            f"initial_mean={initial_mean_loss:.8f} final_mean={final_mean_loss:.8f}"
        )
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("multi-view training expected nonzero viewspace_points gradient")

    print(
        "tiny multi-view 3dgs mlx training ok "
        f"initial_mean_loss={initial_mean_loss:.8f} "
        f"final_mean_loss={final_mean_loss:.8f} "
        f"views={len(cameras)} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
