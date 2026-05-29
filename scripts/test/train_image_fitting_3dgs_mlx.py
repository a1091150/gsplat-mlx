#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from image_fitting_dataset import load_image_fitting_dataset
from training_dataset import TrainingCamera, image_to_u8, write_png


SH_C0 = 0.28209479177387814


def normalize_quats(quats: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True))
    return quats / mx.maximum(norm, 1.0e-8)


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1.0e-5, 1.0 - 1.0e-5)
    return np.log(values / (1.0 - values)).astype(np.float32)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def random_unit_quats_wxyz(rng: np.random.Generator, count: int) -> np.ndarray:
    u = rng.random((count, 1), dtype=np.float32)
    v = rng.random((count, 1), dtype=np.float32)
    w = rng.random((count, 1), dtype=np.float32)
    quats_xyzw = np.concatenate(
        [
            np.sqrt(1.0 - u) * np.sin(2.0 * np.pi * v),
            np.sqrt(1.0 - u) * np.cos(2.0 * np.pi * v),
            np.sqrt(u) * np.sin(2.0 * np.pi * w),
            np.sqrt(u) * np.cos(2.0 * np.pi * w),
        ],
        axis=-1,
    ).astype(np.float32)
    return quats_xyzw[:, [3, 0, 1, 2]]


class ImageFitting3DGSModel(nn.Module):
    def __init__(
        self,
        seed: int,
        num_gaussians: int,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        init_scale: float,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)
        means = rng.uniform(bbox_min, bbox_max, size=(1, num_gaussians, 3)).astype(np.float32)
        scales = np.full((1, num_gaussians, 3), init_scale, dtype=np.float32)
        colors = rng.uniform(0.02, 0.98, size=(1, 1, num_gaussians, 3)).astype(np.float32)
        opacities = np.full((1, num_gaussians), 0.73, dtype=np.float32)
        quats = random_unit_quats_wxyz(rng, num_gaussians)[None, ...]

        self.means = mx.array(means, dtype=mx.float32)
        self.quats = normalize_quats(mx.array(quats, dtype=mx.float32))
        self.log_scales = mx.array(np.log(scales), dtype=mx.float32)
        self.color_logits = mx.array(logit(colors), dtype=mx.float32)
        self.opacity_logits = mx.array(logit(opacities), dtype=mx.float32)

    @classmethod
    def from_arrays(
        cls,
        means: mx.array,
        quats: mx.array,
        log_scales: mx.array,
        color_logits: mx.array,
        opacity_logits: mx.array,
    ) -> "ImageFitting3DGSModel":
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


def camera_arrays(camera: TrainingCamera) -> tuple[mx.array, mx.array]:
    return (
        mx.array(camera.viewmat[None, None, ...], dtype=mx.float32),
        mx.array(camera.K[None, None, ...], dtype=mx.float32),
    )


def render_model(
    model: ImageFitting3DGSModel,
    viewspace_points: mx.array,
    viewmats: mx.array,
    Ks: mx.array,
    width: int,
    height: int,
    tile_size: int,
    background_color: np.ndarray,
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
            "colors": model.colors,
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


def save_render_pair(path: Path, target: np.ndarray, render: mx.array) -> None:
    mx.eval(render)
    rendered = np.asarray(render[0], dtype=np.float32)
    pair = np.concatenate([target, rendered], axis=1)
    write_png(path, image_to_u8(pair))


def save_model_npz(path: Path, model: ImageFitting3DGSModel, summary: dict) -> None:
    mx.eval(model.means, model.normalized_quats, model.log_scales, model.color_logits, model.opacity_logits)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        color_mode=np.array("rgb", dtype=np.str_),
        means=np.asarray(model.means, dtype=np.float32),
        quats_wxyz=np.asarray(model.normalized_quats, dtype=np.float32),
        log_scales=np.asarray(model.log_scales, dtype=np.float32),
        color_logits=np.asarray(model.color_logits, dtype=np.float32),
        opacity_logits=np.asarray(model.opacity_logits, dtype=np.float32),
        summary_json=np.array(json.dumps(summary), dtype=np.str_),
    )


def export_spz(path: Path, model: ImageFitting3DGSModel) -> int:
    try:
        import spz
    except ImportError as exc:
        raise ImportError("The 'spz' Python package is required for image-fitting SPZ export.") from exc

    mx.eval(model.means, model.normalized_quats, model.log_scales, model.color_logits, model.opacity_logits)
    means = np.asarray(model.means[0], dtype=np.float32)
    log_scales = np.asarray(model.log_scales[0], dtype=np.float32)
    quats_wxyz = np.asarray(model.normalized_quats[0], dtype=np.float32)
    colors = np.asarray(model.colors[0, 0], dtype=np.float32)
    opacity_logits = np.asarray(model.opacity_logits[0], dtype=np.float32)

    cloud = spz.GaussianCloud()
    cloud.antialiased = True
    cloud.positions = means.reshape(-1).astype(np.float32)
    cloud.scales = log_scales.reshape(-1).astype(np.float32)
    cloud.rotations = quats_wxyz[:, [1, 2, 3, 0]].reshape(-1).astype(np.float32)
    cloud.alphas = opacity_logits.reshape(-1).astype(np.float32)
    cloud.colors = ((np.clip(colors, 0.0, 1.0) - 0.5) / SH_C0).reshape(-1).astype(np.float32)
    cloud.sh_degree = 0
    cloud.sh = np.array([], dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not spz.save_spz(cloud, spz.PackOptions(), str(path)):
        raise RuntimeError(f"failed to save spz to {path}")
    return int(means.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MLX 3D Gaussian splats to a single image.")
    parser.add_argument("--img-path", type=Path, default=None)
    parser.add_argument("--dataset-out", type=Path, default=Path("outputs/image_fitting_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/image_fitting_train"))
    parser.add_argument("--spz-out", type=Path, default=None)
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-gaussians", type=int, default=1024)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--camera-z", type=float, default=8.0)
    parser.add_argument("--init-xy-extent", type=float, default=None)
    parser.add_argument("--init-z-extent", type=float, default=0.25)
    parser.add_argument("--init-scale", type=float, default=0.02)
    parser.add_argument("--lr-means", type=float, default=1.0e-2)
    parser.add_argument("--lr-colors", type=float, default=1.0e-2)
    parser.add_argument("--lr-opacity", type=float, default=1.0e-2)
    parser.add_argument("--lr-scales", type=float, default=1.0e-2)
    parser.add_argument("--lr-quats", type=float, default=1.0e-2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not is_power_of_two(args.num_gaussians):
        raise ValueError(f"--num-gaussians must be a power of two, got {args.num_gaussians}")

    mx.random.seed(args.seed)
    dataset = load_image_fitting_dataset(
        args.dataset_out,
        args.width,
        args.height,
        img_path=args.img_path,
        camera_z=args.camera_z,
        init_xy_extent=args.init_xy_extent,
        init_z_extent=args.init_z_extent,
    )
    camera = dataset.cameras[0]
    if args.dataset_only:
        print(f"{dataset.name} dataset ok frames=1 target={dataset.metadata['target_path']}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    background_color = np.zeros((3,), dtype=np.float32) if dataset.background_color is None else dataset.background_color
    model = ImageFitting3DGSModel(args.seed, args.num_gaussians, dataset.bbox_min, dataset.bbox_max, args.init_scale)
    viewmats, Ks = camera_arrays(camera)
    target = mx.array(camera.target[None, ...], dtype=mx.float32)

    def loss_fn(
        means: mx.array,
        quats: mx.array,
        log_scales: mx.array,
        color_logits: mx.array,
        opacity_logits: mx.array,
        viewspace_points: mx.array,
    ) -> mx.array:
        local = ImageFitting3DGSModel.from_arrays(means, quats, log_scales, color_logits, opacity_logits)
        render = render_model(
            local,
            viewspace_points,
            viewmats,
            Ks,
            args.width,
            args.height,
            args.tile_size,
            background_color,
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
        background_color,
    )
    mx.eval(initial_render["render_colors"], initial_render["tiles_per_gauss"])
    if int(np.sum(np.asarray(initial_render["tiles_per_gauss"]))) <= 0:
        raise AssertionError("image-fitting training expected nonzero tile intersections")
    save_render_pair(args.out_dir / "step_0000.png", camera.target, initial_render["render_colors"])

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
        mx.eval(model.means, model.quats, model.log_scales, model.color_logits, model.opacity_logits)

        if step == 1 or step == args.steps or (args.log_interval > 0 and step % args.log_interval == 0):
            print(f"step={step:04d} loss={curr_loss:.8f}")
        if step == args.steps or (args.save_interval > 0 and step % args.save_interval == 0):
            render = render_model(
                model,
                mx.zeros((1, 1, args.num_gaussians, 2), dtype=mx.float32),
                viewmats,
                Ks,
                args.width,
                args.height,
                args.tile_size,
                background_color,
            )
            save_render_pair(args.out_dir / f"step_{step:04d}.png", camera.target, render["render_colors"])

    if first_loss is None or last_loss is None or not np.isfinite(last_loss):
        raise AssertionError("image-fitting training loss should be finite")
    if last_loss > first_loss * 1.05:
        raise AssertionError(
            f"image-fitting training loss should not diverge: initial={first_loss:.8f} final={last_loss:.8f}"
        )
    if last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8):
        raise AssertionError("image-fitting training expected nonzero viewspace_points gradient")

    final_render = render_model(
        model,
        mx.zeros((1, 1, args.num_gaussians, 2), dtype=mx.float32),
        viewmats,
        Ks,
        args.width,
        args.height,
        args.tile_size,
        background_color,
    )
    mx.eval(final_render["radii"])
    visible = int(np.count_nonzero(np.any(np.asarray(final_render["radii"]) > 0, axis=-1)))
    spz_path = args.spz_out if args.spz_out is not None else args.out_dir / "trained_image_fitting.spz"
    exported_points = export_spz(spz_path, model)
    summary = {
        "dataset": dataset.name,
        "dataset_metadata": dataset.metadata,
        "out_dir": str(args.out_dir),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "gaussians": args.num_gaussians,
        "loss_function": "mse",
        "initial_loss": first_loss,
        "final_loss": last_loss,
        "visible_gaussians": visible,
        "model_npz": str(args.out_dir / "trained_model_params.npz"),
        "spz": str(spz_path),
        "spz_convention": {
            "position": "direct",
            "scale": "direct",
            "rotation": "direct",
            "quat_order": "xyzw",
            "color": "sh_degree_0",
        },
        "spz_points": exported_points,
    }
    save_model_npz(args.out_dir / "trained_model_params.npz", model, summary)
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "image-fitting 3dgs mlx training ok "
        f"initial_loss={first_loss:.8f} final_loss={last_loss:.8f} "
        f"visible_gaussians={visible} spz={spz_path} output_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
