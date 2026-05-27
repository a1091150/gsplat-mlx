#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import Adam

from gsplat_core import (
    intersect_offset_forward,
    intersect_tile_forward,
    projection_ewa_3dgs_fused_forward,
    rasterize_to_pixels_3dgs_forward,
)
from render_random_3dgs_png import write_png


def normalize_quats(quats: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True))
    return quats / mx.maximum(norm, 1.0e-8)


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1.0e-5, 1.0 - 1.0e-5)
    return np.log(values / (1.0 - values)).astype(np.float32)


def make_camera(width: int, height: int) -> tuple[mx.array, mx.array]:
    viewmats = mx.array(
        [[[[1.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )
    focal = 0.72 * float(min(width, height))
    Ks = mx.array(
        [[[[focal, 0.0, width * 0.5],
           [0.0, focal, height * 0.5],
           [0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )
    return viewmats, Ks


def make_target_image(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    u = (x + 0.5) / float(width)
    v = (y + 0.5) / float(height)

    image = np.full((height, width, 3), 0.025, dtype=np.float32)
    red = ((u - 0.32) ** 2 + (v - 0.34) ** 2) < 0.13 ** 2
    blue = ((u - 0.67) ** 2 + (v - 0.66) ** 2) < 0.16 ** 2
    green = (np.abs(u - 0.54) + np.abs(v - 0.30)) < 0.16
    image[red] = np.array([0.95, 0.08, 0.05], dtype=np.float32)
    image[blue] = np.array([0.10, 0.18, 0.92], dtype=np.float32)
    image[green] = np.array([0.08, 0.82, 0.22], dtype=np.float32)
    return image


def image_to_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


class Tiny3DGSModel(nn.Module):
    def __init__(self, seed: int, num_gaussians: int):
        super().__init__()
        rng = np.random.default_rng(seed)
        n = num_gaussians

        grid_cols = int(np.ceil(np.sqrt(n)))
        grid_rows = int(np.ceil(n / grid_cols))
        xs = np.linspace(-0.9, 0.9, grid_cols, dtype=np.float32)
        ys = np.linspace(-0.9, 0.9, grid_rows, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        xy = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=-1)[:n]
        xy += rng.normal(0.0, 0.035, size=xy.shape).astype(np.float32)

        means = np.zeros((1, n, 3), dtype=np.float32)
        means[0, :, :2] = xy
        means[0, :, 2] = rng.uniform(2.0, 2.8, size=(n,)).astype(np.float32)

        quats = np.zeros((1, n, 4), dtype=np.float32)
        quats[0, :, 0] = 1.0
        quats += rng.normal(0.0, 0.04, size=quats.shape).astype(np.float32)

        scales = rng.uniform(0.07, 0.17, size=(1, n, 3)).astype(np.float32)
        colors = rng.uniform(0.18, 0.72, size=(1, 1, n, 3)).astype(np.float32)
        opacities = rng.uniform(0.25, 0.72, size=(1, n)).astype(np.float32)

        self.means = mx.array(means)
        self.quats = normalize_quats(mx.array(quats))
        self.log_scales = mx.array(np.log(scales).astype(np.float32))
        self.color_logits = mx.array(logit(colors))
        self.opacity_logits = mx.array(logit(opacities))

    @classmethod
    def from_arrays(
        cls,
        means: mx.array,
        quats: mx.array,
        log_scales: mx.array,
        color_logits: mx.array,
        opacity_logits: mx.array,
    ) -> "Tiny3DGSModel":
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.means = means
        model.quats = quats
        model.log_scales = log_scales
        model.color_logits = color_logits
        model.opacity_logits = opacity_logits
        return model

    @property
    def scales(self) -> mx.array:
        return mx.exp(self.log_scales)

    @property
    def colors(self) -> mx.array:
        return mx.sigmoid(self.color_logits)

    @property
    def opacities(self) -> mx.array:
        return mx.sigmoid(self.opacity_logits)

    @property
    def normalized_quats(self) -> mx.array:
        return normalize_quats(self.quats)


def render_model(
    model: Tiny3DGSModel,
    viewspace_points: mx.array,
    viewmats: mx.array,
    Ks: mx.array,
    width: int,
    height: int,
    tile_size: int,
) -> dict[str, mx.array]:
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
    tile_offsets = mx.stop_gradient(intersect_offset_forward(
        intersections["isect_ids"],
        I=1,
        tile_width=tile_width,
        tile_height=tile_height,
    ))
    flatten_ids = mx.stop_gradient(intersections["flatten_ids"])
    render = rasterize_to_pixels_3dgs_forward(
        {
            "means2d": projection["means2d"],
            "conics": projection["conics"],
            "colors": model.colors,
            "opacities": mx.expand_dims(model.opacities, axis=1),
            "backgrounds": mx.array([[0.025, 0.025, 0.025]], dtype=mx.float32),
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


def save_render(path: Path, image: mx.array) -> None:
    mx.eval(image)
    write_png(path, image_to_u8(np.asarray(image[0], dtype=np.float32)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tiny_3dgs_train"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--num-gaussians", type=int, default=1024)
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
    viewmats, Ks = make_camera(args.width, args.height)
    target_np = make_target_image(args.width, args.height)
    target = mx.array(target_np[None, ...], dtype=mx.float32)
    write_png(args.out_dir / "target.png", image_to_u8(target_np))

    def loss_fn(
        means: mx.array,
        quats: mx.array,
        log_scales: mx.array,
        color_logits: mx.array,
        opacity_logits: mx.array,
        viewspace_points: mx.array,
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

    initial_viewspace = mx.zeros((1, 1, args.num_gaussians, 2), dtype=mx.float32)
    initial_render = render_model(
        model,
        initial_viewspace,
        viewmats,
        Ks,
        args.width,
        args.height,
        args.tile_size,
    )
    mx.eval(initial_render["render_colors"], initial_render["tiles_per_gauss"])
    if int(np.sum(np.asarray(initial_render["tiles_per_gauss"]))) <= 0:
        raise AssertionError("tiny training expected nonzero tile intersections")
    save_render(args.out_dir / "step_0000.png", initial_render["render_colors"])

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))
    optimizers = {
        "means": Adam(learning_rate=args.lr_means),
        "quats": Adam(learning_rate=args.lr_quats),
        "log_scales": Adam(learning_rate=args.lr_scales),
        "color_logits": Adam(learning_rate=args.lr_colors),
        "opacity_logits": Adam(learning_rate=args.lr_opacity),
    }

    first_loss = None
    last_loss = None
    last_viewspace_grad = None
    for step in range(1, args.steps + 1):
        viewspace_points = mx.zeros((1, 1, args.num_gaussians, 2), dtype=mx.float32)
        loss, grads = grad_fn(
            model.means,
            model.quats,
            model.log_scales,
            model.color_logits,
            model.opacity_logits,
            viewspace_points,
        )
        d_means, d_quats, d_log_scales, d_color_logits, d_opacity_logits, d_viewspace = grads
        mx.eval(loss, d_viewspace)
        curr_loss = float(np.asarray(loss))
        if first_loss is None:
            first_loss = curr_loss
        last_loss = curr_loss
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
            print(f"step={step:04d} loss={curr_loss:.8f}")

    final_render = render_model(
        model,
        mx.zeros((1, 1, args.num_gaussians, 2), dtype=mx.float32),
        viewmats,
        Ks,
        args.width,
        args.height,
        args.tile_size,
    )
    mx.eval(final_render["render_colors"], final_render["radii"], final_render["flatten_ids"])
    save_render(args.out_dir / f"step_{args.steps:04d}.png", final_render["render_colors"])

    if first_loss is None or last_loss is None or not np.isfinite(last_loss):
        raise AssertionError("tiny training loss should be finite")
    if last_loss > first_loss * 1.05:
        raise AssertionError(
            f"tiny training loss should not diverge: initial={first_loss:.8f} final={last_loss:.8f}"
        )
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("tiny training expected nonzero viewspace_points gradient")

    visible = int(np.count_nonzero(np.any(np.asarray(final_render["radii"]) > 0, axis=-1)))
    print(
        "tiny 3dgs mlx training ok "
        f"initial_loss={first_loss:.8f} final_loss={last_loss:.8f} "
        f"visible_gaussians={visible} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
