#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
    spherical_harmonics_forward,
)
from render_random_3dgs_png import write_png
from scanner_dataset_utils import axis_transform, collect_frames, load_camera, load_target


SH_C0 = 0.28209479177387814
MAX_SUPPORTED_SH_DEGREE = 3
GAUSSIAN_AXES = {
    "means": 1,
    "quats": 1,
    "log_scales": 1,
    "opacity_logits": 1,
    "color_logits": 2,
    "features_dc": 1,
    "features_rest": 1,
}


def normalize_quats(quats: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True))
    return quats / mx.maximum(norm, 1.0e-8)


def mx_logit(values: mx.array) -> mx.array:
    clipped = mx.minimum(mx.maximum(values, 1.0e-5), 1.0 - 1.0e-5)
    return mx.log(clipped / (1.0 - clipped))


def image_to_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def camera_arrays(camera) -> tuple[mx.array, mx.array]:
    return (
        mx.array(camera.viewmat[None, None, ...], dtype=mx.float32),
        mx.array(camera.K[None, None, ...], dtype=mx.float32),
    )


def concat_compare(target: np.ndarray, initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    gap = np.ones((target.shape[0], 6, 3), dtype=np.float32)
    return np.concatenate([target, gap, initial, gap, final], axis=1)


def mean_loss(stats: list[dict]) -> float:
    return float(np.mean([item["loss"] for item in stats])) if stats else 0.0


class Tiny3DGSModel(nn.Module):
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


def load_ply_positions_colors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("Reading points.ply requires the 'plyfile' package.") from exc

    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    names = vertices.data.dtype.names or ()
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.float32)
        if colors.max(initial=0.0) > 1.0:
            colors /= 255.0
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = np.full_like(points, 0.7, dtype=np.float32)
    return points, colors


def prepare_points(
    dataset_dir: Path,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    points, colors = load_ply_positions_colors(dataset_dir / "points.ply")
    raw_count = int(points.shape[0])
    points = (axis_transform()[:3, :3] @ points.T).T.astype(np.float32)
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    return points.astype(np.float32), colors.astype(np.float32), raw_count


class FrameBatchSampler:
    def __init__(self, frame_count: int, batch_size: int, mode: str, seed: int):
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if mode not in ("sequential", "shuffle", "pingpong"):
            raise ValueError(f"Unsupported frame sampling mode: {mode}")
        self.frame_count = int(frame_count)
        self.batch_size = int(batch_size)
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.order = self._make_order()
        self.position = 0
        self.epoch = 0
        self.usage_counts = np.zeros((self.frame_count,), dtype=np.int64)
        self.history: list[list[int]] = []
        if self.mode == "shuffle":
            self.rng.shuffle(self.order)

    def _make_order(self) -> np.ndarray:
        forward = np.arange(self.frame_count, dtype=np.int32)
        if self.mode != "pingpong" or self.frame_count <= 1:
            return forward
        backward = np.arange(self.frame_count - 2, 0, -1, dtype=np.int32)
        return np.concatenate([forward, backward])

    def next_batch(self) -> list[int]:
        batch = []
        while len(batch) < self.batch_size:
            if self.position >= len(self.order):
                self.epoch += 1
                self.position = 0
                self.order = self._make_order()
                if self.mode == "shuffle":
                    self.rng.shuffle(self.order)
            take = min(self.batch_size - len(batch), len(self.order) - self.position)
            batch.extend(self.order[self.position : self.position + take].astype(int).tolist())
            self.position += take
        for idx in batch:
            self.usage_counts[idx] += 1
        if len(self.history) < 16:
            self.history.append([int(idx) for idx in batch])
        return batch

    def summary(self, cameras) -> dict:
        return {
            "mode": self.mode,
            "batch_size": self.batch_size,
            "frame_count": self.frame_count,
            "order_length": int(len(self.order)),
            "completed_epochs": int(self.epoch),
            "usage_counts": self.usage_counts.astype(int).tolist(),
            "usage_count_min": int(self.usage_counts.min()) if self.usage_counts.size else 0,
            "usage_count_max": int(self.usage_counts.max()) if self.usage_counts.size else 0,
            "sampled_batches": self.history,
            "sampled_frame_indices": [
                [int(cameras[idx].index) for idx in batch]
                for batch in self.history
            ],
        }


class ScannerPointsSHModel(nn.Module):
    def __init__(
        self,
        means: mx.array,
        quats: mx.array,
        log_scales: mx.array,
        features_dc: mx.array,
        features_rest: mx.array,
        opacity_logits: mx.array,
    ):
        super().__init__()
        self.means = means
        self.quats = quats
        self.log_scales = log_scales
        self.features_dc = features_dc
        self.features_rest = features_rest
        self.opacity_logits = opacity_logits

    @classmethod
    def from_arrays(
        cls,
        means: mx.array,
        quats: mx.array,
        log_scales: mx.array,
        features_dc: mx.array,
        features_rest: mx.array,
        opacity_logits: mx.array,
    ) -> "ScannerPointsSHModel":
        return cls(
            means,
            quats,
            log_scales,
            features_dc,
            features_rest,
            opacity_logits,
        )

    @property
    def scales(self) -> mx.array:
        return mx.exp(self.log_scales)

    @property
    def opacities(self) -> mx.array:
        return mx.sigmoid(self.opacity_logits)

    @property
    def normalized_quats(self) -> mx.array:
        return normalize_quats(self.quats)


def sh_coeff_count(degree: int) -> int:
    return (degree + 1) * (degree + 1)


@dataclass
class ScannerDefaultStrategyConfig:
    enabled: bool = False
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    reset_every: int = 3000
    opacity_reset_steps: list[int] | None = None
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    scene_scale: float = 1.0
    absgrad: bool = False
    revised_opacity: bool = False

    def should_refine(self, step: int) -> bool:
        return (
            self.enabled
            and step > self.refine_start_iter
            and step < self.refine_stop_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        )

    def should_reset_opacity(self, step: int) -> bool:
        if not self.enabled or step <= 0:
            return False
        if self.opacity_reset_steps is not None:
            return int(step) in set(int(item) for item in self.opacity_reset_steps)
        return self.enabled and step > 0 and step % self.reset_every == 0


class ScannerDefaultStrategyRuntime:
    def __init__(self, config: ScannerDefaultStrategyConfig, initial_gaussians: int):
        self.config = config
        self.initial_gaussians = int(initial_gaussians)
        self.last_gaussians = int(initial_gaussians)
        self.grad2d = np.zeros((initial_gaussians,), dtype=np.float32)
        self.count = np.zeros((initial_gaussians,), dtype=np.float32)
        self.radii = (
            np.zeros((initial_gaussians,), dtype=np.float32)
            if config.refine_scale2d_stop_iter > 0
            else None
        )
        self.events: list[dict] = []
        self.totals = {
            "n_clone": 0,
            "n_split": 0,
            "n_prune": 0,
            "n_prune_opacity": 0,
            "n_prune_scale3d": 0,
            "n_prune_scale2d": 0,
            "n_opacity_reset": 0,
        }
        self.last_grad2d_stats = self._grad2d_stats()
        self.last_grad2d_mode = "signed_grad_norm"
        self.absgrad_fallback_count = 0
        self.rng = np.random.default_rng(20280628)

    def _ensure_size(self, gaussian_count: int) -> None:
        gaussian_count = int(gaussian_count)
        if self.grad2d.shape[0] == gaussian_count:
            return
        old_count = self.grad2d.shape[0]
        if gaussian_count < old_count:
            self.grad2d = self.grad2d[:gaussian_count]
            self.count = self.count[:gaussian_count]
            if self.radii is not None:
                self.radii = self.radii[:gaussian_count]
            return
        pad = gaussian_count - old_count
        self.grad2d = np.concatenate([self.grad2d, np.zeros((pad,), dtype=np.float32)])
        self.count = np.concatenate([self.count, np.zeros((pad,), dtype=np.float32)])
        if self.radii is not None:
            self.radii = np.concatenate([self.radii, np.zeros((pad,), dtype=np.float32)])

    def reset_running_state(self, gaussian_count: int | None = None) -> None:
        if gaussian_count is None:
            gaussian_count = self.grad2d.shape[0]
        gaussian_count = int(gaussian_count)
        self.grad2d = np.zeros((gaussian_count,), dtype=np.float32)
        self.count = np.zeros((gaussian_count,), dtype=np.float32)
        if self.radii is not None:
            self.radii = np.zeros((gaussian_count,), dtype=np.float32)
        self.last_grad2d_stats = self._grad2d_stats()

    def update_state(
        self,
        d_viewspace: mx.array,
        radii: mx.array,
        width: int,
        height: int,
        n_cameras: int,
        d_viewspace_abs: mx.array | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        if self.config.absgrad and d_viewspace_abs is not None:
            mx.eval(d_viewspace_abs, radii)
            grads = np.asarray(d_viewspace_abs, dtype=np.float32)
            self.last_grad2d_mode = "absgrad_norm"
        else:
            mx.eval(d_viewspace, radii)
            grads = np.asarray(d_viewspace, dtype=np.float32)
            if self.config.absgrad:
                self.absgrad_fallback_count += 1
                self.last_grad2d_mode = "absgrad_requested_signed_grad_fallback"
            else:
                self.last_grad2d_mode = "signed_grad_norm"
        radii_np = np.asarray(radii, dtype=np.float32)
        self._ensure_size(grads.shape[-2])

        grads[..., 0] *= float(width) / 2.0 * float(n_cameras)
        grads[..., 1] *= float(height) / 2.0 * float(n_cameras)
        visible = np.all(radii_np > 0.0, axis=-1)
        if not np.any(visible):
            self.last_grad2d_stats = self._grad2d_stats()
            return

        gaussian_ids = np.where(visible)[-1]
        grad_norms = np.linalg.norm(grads[visible], axis=-1)
        np.add.at(self.grad2d, gaussian_ids, grad_norms)
        np.add.at(self.count, gaussian_ids, np.ones_like(grad_norms, dtype=np.float32))
        if self.radii is not None:
            normalized_radii = radii_np[visible].max(axis=-1) / float(max(width, height))
            self.radii[gaussian_ids] = np.maximum(self.radii[gaussian_ids], normalized_radii)
        self.last_grad2d_stats = self._grad2d_stats()

    def _grad2d_stats(self) -> dict:
        visible = self.count > 0.0
        avg = np.zeros_like(self.grad2d)
        avg[visible] = self.grad2d[visible] / np.maximum(self.count[visible], 1.0)
        return {
            "visible_gaussians": int(np.count_nonzero(visible)),
            "total_observations": int(self.count.sum()),
            "grad2d_mean": float(avg[visible].mean()) if np.any(visible) else 0.0,
            "grad2d_max": float(avg[visible].max()) if np.any(visible) else 0.0,
            "count_max": float(self.count.max()) if self.count.size else 0.0,
            "radii_max": float(self.radii.max()) if self.radii is not None and self.radii.size else None,
        }

    def _apply_keep_to_state(self, keep: np.ndarray) -> None:
        self.grad2d = self.grad2d[keep]
        self.count = self.count[keep]
        if self.radii is not None:
            self.radii = self.radii[keep]
        self.last_grad2d_stats = self._grad2d_stats()

    def _take_gaussians(self, array: mx.array, keep: np.ndarray, axis: int) -> mx.array:
        return mx.take(array, mx.array(keep.astype(np.int32), dtype=mx.int32), axis=axis)

    def _select_gaussians(self, array: mx.array, selected: np.ndarray, axis: int) -> mx.array:
        return mx.take(array, mx.array(selected.astype(np.int32), dtype=mx.int32), axis=axis)

    def _append_gaussians(self, array: mx.array, extra: mx.array, axis: int) -> mx.array:
        return mx.concatenate([array, extra], axis=axis)

    def _apply_keep_to_param_and_optimizer(
        self,
        model: Tiny3DGSModel | ScannerPointsSHModel,
        optimizers: dict[str, Adam],
        name: str,
        keep: np.ndarray,
    ) -> None:
        if not hasattr(model, name):
            return
        axis = GAUSSIAN_AXES[name]
        setattr(model, name, self._take_gaussians(getattr(model, name), keep, axis))
        optimizer = optimizers.get(name)
        if optimizer is None or name not in optimizer.state:
            return
        param_state = optimizer.state[name]
        for state_name in ("m", "v"):
            if state_name in param_state:
                param_state[state_name] = self._take_gaussians(param_state[state_name], keep, axis)

    def _apply_duplicate_to_state(self, selected: np.ndarray) -> None:
        self.grad2d = np.concatenate([self.grad2d, self.grad2d[selected]])
        self.count = np.concatenate([self.count, self.count[selected]])
        if self.radii is not None:
            self.radii = np.concatenate([self.radii, self.radii[selected]])
        self.last_grad2d_stats = self._grad2d_stats()

    def _apply_duplicate_to_param_and_optimizer(
        self,
        model: Tiny3DGSModel | ScannerPointsSHModel,
        optimizers: dict[str, Adam],
        name: str,
        selected: np.ndarray,
    ) -> None:
        if not hasattr(model, name):
            return
        axis = GAUSSIAN_AXES[name]
        param = getattr(model, name)
        extra = self._select_gaussians(param, selected, axis)
        setattr(model, name, self._append_gaussians(param, extra, axis))

        optimizer = optimizers.get(name)
        if optimizer is None or name not in optimizer.state:
            return
        param_state = optimizer.state[name]
        for state_name in ("m", "v"):
            if state_name not in param_state:
                continue
            state_extra = self._select_gaussians(param_state[state_name], selected, axis)
            param_state[state_name] = self._append_gaussians(
                param_state[state_name],
                mx.zeros_like(state_extra),
                axis,
            )

    def _duplicate_high_grad_small(
        self,
        model: Tiny3DGSModel | ScannerPointsSHModel,
        optimizers: dict[str, Adam],
        color_mode: str,
    ) -> int:
        visible = self.count > 0.0
        if not np.any(visible):
            return 0

        avg_grad2d = np.zeros_like(self.grad2d)
        avg_grad2d[visible] = self.grad2d[visible] / np.maximum(self.count[visible], 1.0)
        mx.eval(model.log_scales)
        scales = np.exp(np.asarray(model.log_scales[0], dtype=np.float32))
        max_scale = scales.max(axis=-1)
        small = max_scale <= self.config.grow_scale3d * self.config.scene_scale
        selected = np.where((avg_grad2d > self.config.grow_grad2d) & small)[0].astype(np.int32)
        if selected.size == 0:
            return 0

        names = ["means", "quats", "log_scales", "opacity_logits"]
        if color_mode == "rgb":
            names.append("color_logits")
        else:
            names.extend(["features_dc", "features_rest"])
        for name in names:
            self._apply_duplicate_to_param_and_optimizer(model, optimizers, name, selected)
        self._apply_duplicate_to_state(selected)
        mx.eval(model.means, model.quats, model.log_scales, model.opacity_logits)
        return int(selected.size)

    def _apply_split_to_state(self, split_mask: np.ndarray) -> None:
        rest = np.where(~split_mask)[0]
        selected = np.where(split_mask)[0]
        self.grad2d = np.concatenate([self.grad2d[rest], self.grad2d[selected], self.grad2d[selected]])
        self.count = np.concatenate([self.count[rest], self.count[selected], self.count[selected]])
        if self.radii is not None:
            self.radii = np.concatenate([self.radii[rest], self.radii[selected], self.radii[selected]])
        self.last_grad2d_stats = self._grad2d_stats()

    def _split_extra_for_param(
        self,
        name: str,
        param: mx.array,
        selected: np.ndarray,
        child_means: np.ndarray,
        child_log_scales: np.ndarray,
        child_opacity_logits: np.ndarray | None,
    ) -> mx.array:
        axis = GAUSSIAN_AXES[name]
        if name == "means":
            return mx.array(child_means[None, ...], dtype=param.dtype)
        if name == "log_scales":
            return mx.array(child_log_scales[None, ...], dtype=param.dtype)
        if name == "opacity_logits" and child_opacity_logits is not None:
            return mx.array(child_opacity_logits[None, ...], dtype=param.dtype)
        selected_param = self._select_gaussians(param, selected, axis)
        return mx.concatenate([selected_param, selected_param], axis=axis)

    def _apply_split_to_param_and_optimizer(
        self,
        model: Tiny3DGSModel | ScannerPointsSHModel,
        optimizers: dict[str, Adam],
        name: str,
        split_mask: np.ndarray,
        child_means: np.ndarray,
        child_log_scales: np.ndarray,
        child_opacity_logits: np.ndarray | None,
    ) -> None:
        if not hasattr(model, name):
            return
        axis = GAUSSIAN_AXES[name]
        rest = np.where(~split_mask)[0].astype(np.int32)
        selected = np.where(split_mask)[0].astype(np.int32)
        param = getattr(model, name)
        rest_param = self._take_gaussians(param, rest, axis)
        split_param = self._split_extra_for_param(
            name,
            param,
            selected,
            child_means,
            child_log_scales,
            child_opacity_logits,
        )
        setattr(model, name, mx.concatenate([rest_param, split_param], axis=axis))

        optimizer = optimizers.get(name)
        if optimizer is None or name not in optimizer.state:
            return
        param_state = optimizer.state[name]
        for state_name in ("m", "v"):
            if state_name not in param_state:
                continue
            rest_state = self._take_gaussians(param_state[state_name], rest, axis)
            selected_state = self._select_gaussians(param_state[state_name], selected, axis)
            split_state = mx.zeros_like(mx.concatenate([selected_state, selected_state], axis=axis))
            param_state[state_name] = mx.concatenate([rest_state, split_state], axis=axis)

    def _split_high_grad_large(
        self,
        step: int,
        original_gaussian_count: int,
        model: Tiny3DGSModel | ScannerPointsSHModel,
        optimizers: dict[str, Adam],
        color_mode: str,
    ) -> int:
        visible = self.count > 0.0
        if not np.any(visible):
            return 0

        avg_grad2d = np.zeros_like(self.grad2d)
        avg_grad2d[visible] = self.grad2d[visible] / np.maximum(self.count[visible], 1.0)
        mx.eval(model.means, model.log_scales, model.normalized_quats, model.opacity_logits)
        means = np.asarray(model.means[0], dtype=np.float32)
        log_scales = np.asarray(model.log_scales[0], dtype=np.float32)
        quats = np.asarray(model.normalized_quats[0], dtype=np.float32)
        opacity_logits = np.asarray(model.opacity_logits[0], dtype=np.float32)

        scales = np.exp(log_scales)
        max_scale = scales.max(axis=-1)
        large = max_scale > self.config.grow_scale3d * self.config.scene_scale
        split_mask = (avg_grad2d > self.config.grow_grad2d) & large
        if self.radii is not None and step < self.config.refine_scale2d_stop_iter:
            split_mask |= self.radii > self.config.grow_scale2d
        split_mask[int(original_gaussian_count) :] = False
        selected = np.where(split_mask)[0].astype(np.int32)
        if selected.size == 0:
            return 0

        selected_scales = scales[selected]
        selected_quats = quats[selected]
        rotmats = quat_wxyz_to_rotmat(selected_quats)
        local_noise = self.rng.standard_normal((2, selected.size, 3)).astype(np.float32)
        offsets = np.einsum("nij,nj,bnj->bni", rotmats, selected_scales, local_noise)
        child_means = (means[selected][None, :, :] + offsets).reshape(-1, 3).astype(np.float32)
        child_log_scales = np.concatenate(
            [
                np.log(np.clip(selected_scales / 1.6, 1.0e-12, None)),
                np.log(np.clip(selected_scales / 1.6, 1.0e-12, None)),
            ],
            axis=0,
        ).astype(np.float32)
        child_opacity_logits = None
        if self.config.revised_opacity:
            selected_opacity = 1.0 / (1.0 + np.exp(-opacity_logits[selected]))
            child_opacity = 1.0 - np.sqrt(np.clip(1.0 - selected_opacity, 0.0, 1.0))
            child_opacity = np.clip(child_opacity, 1.0e-6, 1.0 - 1.0e-6)
            child_opacity_logits = np.concatenate(
                [
                    np.log(child_opacity / (1.0 - child_opacity)),
                    np.log(child_opacity / (1.0 - child_opacity)),
                ],
                axis=0,
            ).astype(np.float32)

        names = ["means", "quats", "log_scales", "opacity_logits"]
        if color_mode == "rgb":
            names.append("color_logits")
        else:
            names.extend(["features_dc", "features_rest"])
        for name in names:
            self._apply_split_to_param_and_optimizer(
                model,
                optimizers,
                name,
                split_mask,
                child_means,
                child_log_scales,
                child_opacity_logits,
            )
        self._apply_split_to_state(split_mask)
        mx.eval(model.means, model.quats, model.log_scales, model.opacity_logits)
        return int(selected.size)

    def _prune_gaussians(
        self,
        step: int,
        model: Tiny3DGSModel | ScannerPointsSHModel,
        optimizers: dict[str, Adam],
        color_mode: str,
    ) -> tuple[int, dict]:
        mx.eval(model.opacity_logits, model.log_scales)
        opacities = 1.0 / (1.0 + np.exp(-np.asarray(model.opacity_logits[0], dtype=np.float32)))
        prune_opacity = opacities < self.config.prune_opa
        prune_scale3d = np.zeros_like(prune_opacity, dtype=bool)
        prune_scale2d = np.zeros_like(prune_opacity, dtype=bool)
        if step > self.config.reset_every:
            scales = np.exp(np.asarray(model.log_scales[0], dtype=np.float32))
            prune_scale3d = scales.max(axis=-1) > self.config.prune_scale3d * self.config.scene_scale
            if self.radii is not None and step < self.config.refine_scale2d_stop_iter:
                prune_scale2d = self.radii > self.config.prune_scale2d

        prune = prune_opacity | prune_scale3d | prune_scale2d
        prune_breakdown = {
            "opacity": int(np.count_nonzero(prune_opacity)),
            "scale3d": int(np.count_nonzero(prune_scale3d)),
            "scale2d": int(np.count_nonzero(prune_scale2d)),
            "total_unique": int(np.count_nonzero(prune)),
        }
        if not np.any(prune):
            return 0, prune_breakdown
        keep = np.where(~prune)[0].astype(np.int32)
        if keep.size == 0:
            keep = np.array([int(np.argmax(opacities))], dtype=np.int32)
        n_prune = int(opacities.shape[0] - keep.shape[0])

        names = ["means", "quats", "log_scales", "opacity_logits"]
        if color_mode == "rgb":
            names.append("color_logits")
        else:
            names.extend(["features_dc", "features_rest"])
        for name in names:
            self._apply_keep_to_param_and_optimizer(model, optimizers, name, keep)
        self._apply_keep_to_state(keep)
        mx.eval(model.means, model.quats, model.log_scales, model.opacity_logits)
        prune_breakdown["actual_removed"] = n_prune
        return n_prune, prune_breakdown

    def _reset_opacity(
        self,
        model: Tiny3DGSModel | ScannerPointsSHModel,
        optimizers: dict[str, Adam],
    ) -> int:
        value = self.opacity_reset_target()
        max_logit = self.opacity_reset_target_logit()
        model.opacity_logits = mx.minimum(model.opacity_logits, mx.array(max_logit, dtype=model.opacity_logits.dtype))

        optimizer = optimizers.get("opacity_logits")
        if optimizer is not None and "opacity_logits" in optimizer.state:
            param_state = optimizer.state["opacity_logits"]
            for state_name in ("m", "v"):
                if state_name in param_state:
                    param_state[state_name] = mx.zeros_like(param_state[state_name])
        mx.eval(model.opacity_logits)
        return int(model.opacity_logits.shape[1])

    def opacity_reset_target(self) -> float:
        return float(np.clip(self.config.prune_opa * 2.0, 1.0e-6, 1.0 - 1.0e-6))

    def opacity_reset_target_logit(self) -> float:
        value = self.opacity_reset_target()
        return float(np.log(value / (1.0 - value)))

    def after_optimizer_step(
        self,
        step: int,
        model: Tiny3DGSModel | ScannerPointsSHModel,
        optimizers: dict[str, Adam],
        color_mode: str,
    ) -> None:
        gaussian_count = int(model.means.shape[1])
        self.last_gaussians = gaussian_count
        self._ensure_size(gaussian_count)
        scheduled_refine = self.config.should_refine(step)
        scheduled_reset = self.config.should_reset_opacity(step)
        if not scheduled_refine and not scheduled_reset:
            return
        n_clone = self._duplicate_high_grad_small(model, optimizers, color_mode) if scheduled_refine else 0
        n_split = (
            self._split_high_grad_large(step, gaussian_count, model, optimizers, color_mode)
            if scheduled_refine
            else 0
        )
        n_prune, prune_breakdown = (
            self._prune_gaussians(step, model, optimizers, color_mode)
            if scheduled_refine
            else (0, {"opacity": 0, "scale3d": 0, "scale2d": 0, "total_unique": 0, "actual_removed": 0})
        )
        n_opacity_reset = self._reset_opacity(model, optimizers) if scheduled_reset else 0
        self.totals["n_clone"] += n_clone
        self.totals["n_split"] += n_split
        self.totals["n_prune"] += n_prune
        self.totals["n_prune_opacity"] += prune_breakdown["opacity"]
        self.totals["n_prune_scale3d"] += prune_breakdown["scale3d"]
        self.totals["n_prune_scale2d"] += prune_breakdown["scale2d"]
        self.totals["n_opacity_reset"] += n_opacity_reset
        after_count = int(model.means.shape[1])
        self.last_gaussians = after_count
        grad2d_stats_before_reset = self._grad2d_stats()
        stats_reset_after_refine = bool(scheduled_refine)
        if stats_reset_after_refine:
            self.reset_running_state(after_count)
        self.events.append(
            {
                "step": int(step),
                "scheduled_refine": bool(scheduled_refine),
                "scheduled_opacity_reset": bool(scheduled_reset),
                "num_gaussians_before": int(gaussian_count),
                "num_gaussians_after": after_count,
                "n_clone": n_clone,
                "n_split": n_split,
                "n_prune": n_prune,
                "prune_breakdown": prune_breakdown,
                "n_opacity_reset": n_opacity_reset,
                "stats_reset_after_refine": stats_reset_after_refine,
                "grad2d_stats_before_reset": grad2d_stats_before_reset,
                "grad2d_stats_after_reset": self.last_grad2d_stats,
                "grad2d_stats": self.last_grad2d_stats,
                "grad2d_mode": self.last_grad2d_mode,
                "absgrad_fallback_count": self.absgrad_fallback_count,
                "status": "clone_split_scale_prune_reset_task_6_30a",
            }
        )

    def summary(self) -> dict:
        latest_event = self.events[-1] if self.events else None
        topology_event_count = int(
            sum(
                1
                for event in self.events
                if event["n_clone"] or event["n_split"] or event["n_prune"] or event["n_opacity_reset"]
            )
        )
        return {
            "implementation_phase": "task_6_30a_refine_state_reset",
            "enabled": self.config.enabled,
            "config": asdict(self.config),
            "initial_gaussians": self.initial_gaussians,
            "final_gaussians": self.last_gaussians,
            "gaussian_delta": int(self.last_gaussians - self.initial_gaussians),
            "events": self.events,
            "latest_event": latest_event,
            "event_count": len(self.events),
            "topology_event_count": topology_event_count,
            "totals": self.totals,
            "operation_totals": {
                "clone": self.totals["n_clone"],
                "split": self.totals["n_split"],
                "prune": self.totals["n_prune"],
                "prune_by_reason": {
                    "opacity": self.totals["n_prune_opacity"],
                    "scale3d": self.totals["n_prune_scale3d"],
                    "scale2d": self.totals["n_prune_scale2d"],
                },
                "opacity_reset": self.totals["n_opacity_reset"],
            },
            "grad2d_accumulation": "dense_viewspace_points_gradient",
            "grad2d_mode": self.last_grad2d_mode,
            "absgrad_fallback_count": self.absgrad_fallback_count,
            "grad2d_stats": self.last_grad2d_stats,
            "preview_diagnostics": {
                "visible_gaussians": self.last_grad2d_stats["visible_gaussians"],
                "total_observations": self.last_grad2d_stats["total_observations"],
                "grad2d_mean": self.last_grad2d_stats["grad2d_mean"],
                "grad2d_max": self.last_grad2d_stats["grad2d_max"],
                "radii_max": self.last_grad2d_stats["radii_max"],
                "grad2d_mode": self.last_grad2d_mode,
                "absgrad_fallback_count": self.absgrad_fallback_count,
            },
            "opacity_reset_target": self.opacity_reset_target(),
            "opacity_reset_target_logit": self.opacity_reset_target_logit(),
            "topology_changes": "clone_split_opacity_scale_prune_reset_task_6_30a",
        }


def init_rgb_model_from_points(
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


def init_sh_model_from_points(
    points: np.ndarray,
    colors: np.ndarray,
    point_scale: float,
    opacity: float,
    max_sh_degree: int,
) -> ScannerPointsSHModel:
    n = int(points.shape[0])
    means = mx.array(points[None, ...], dtype=mx.float32)
    quats = mx.zeros((1, n, 4), dtype=mx.float32) + mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32)
    log_scales = mx.full((1, n, 3), np.log(point_scale), dtype=mx.float32)
    features_dc = mx.array(((colors[None, ...] - 0.5) / SH_C0).astype(np.float32), dtype=mx.float32)
    rest_count = sh_coeff_count(max_sh_degree) - 1
    features_rest = mx.zeros((1, n, rest_count, 3), dtype=mx.float32)
    opacity_logits = mx_logit(mx.full((1, n), opacity, dtype=mx.float32))
    return ScannerPointsSHModel.from_arrays(
        means,
        normalize_quats(quats),
        log_scales,
        features_dc,
        features_rest,
        opacity_logits,
    )


def append_random_gaussians(
    points: np.ndarray,
    colors: np.ndarray,
    count: int,
    seed: int,
    bounds_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return points, colors
    if points.size == 0:
        raise ValueError("Cannot append random Gaussians without existing point cloud bounds")

    rng = np.random.default_rng(seed)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    half_extent = np.maximum((maxs - mins) * 0.5 * bounds_scale, 1.0e-3)
    random_points = rng.uniform(center - half_extent, center + half_extent, size=(count, 3)).astype(np.float32)
    random_colors = rng.uniform(0.08, 0.95, size=(count, 3)).astype(np.float32)
    return (
        np.concatenate([points, random_points], axis=0).astype(np.float32),
        np.concatenate([colors, random_colors], axis=0).astype(np.float32),
    )


def points_extent_diagnostics(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have shape [N, 3] and be nonempty")
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    half_extent = (bbox_max - bbox_min) * 0.5
    distances = np.linalg.norm(points - center[None, :], axis=1)
    return {
        "point_count": int(points.shape[0]),
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "center": center.astype(float).tolist(),
        "bbox_size": (bbox_max - bbox_min).astype(float).tolist(),
        "bbox_diagonal": float(np.linalg.norm(bbox_max - bbox_min)),
        "max_radius": float(distances.max()),
        "p90_radius": float(np.percentile(distances, 90.0)),
        "p95_radius": float(np.percentile(distances, 95.0)),
        "p99_radius": float(np.percentile(distances, 99.0)),
    }


def select_scene_scale(diagnostics: dict, mode: str, fixed_scale: float) -> float:
    if mode == "fixed":
        return float(fixed_scale)
    if mode == "points_extent":
        return float(max(diagnostics["p95_radius"], 1.0e-6))
    raise ValueError(f"Unsupported scene scale mode: {mode}")


def select_point_scale(
    diagnostics: dict,
    mode: str,
    fixed_scale: float,
    fraction: float,
) -> float:
    if mode == "fixed":
        return float(fixed_scale)
    if mode == "scene_fraction":
        scene_scale = float(max(diagnostics["p95_radius"], 1.0e-6))
        return float(max(scene_scale * fraction, 1.0e-8))
    raise ValueError(f"Unsupported point scale mode: {mode}")


def camera_center_from_viewmat(viewmats: mx.array) -> mx.array:
    rot = viewmats[:, :, :3, :3]
    trans = viewmats[:, :, :3, 3]
    return -mx.matmul(mx.swapaxes(rot, -1, -2), trans[..., None])[..., 0]


def sh_colors_for_camera(
    model: ScannerPointsSHModel,
    viewmats: mx.array,
    sh_degree: int,
) -> mx.array:
    active_coeffs = sh_coeff_count(sh_degree)
    dirs = model.means[:, None, :, :] - camera_center_from_viewmat(viewmats)[:, :, None, :]
    dirs = dirs / mx.maximum(mx.sqrt(mx.sum(dirs * dirs, axis=-1, keepdims=True)), 1.0e-8)
    coeffs = mx.concatenate(
        [
            mx.expand_dims(model.features_dc, axis=2),
            model.features_rest[:, :, : active_coeffs - 1, :],
        ],
        axis=2,
    )
    coeffs = mx.broadcast_to(coeffs[:, None, :, :, :], (*dirs.shape[:-1], active_coeffs, 3))
    colors = spherical_harmonics_forward(
        sh_degree,
        {
            "dirs": mx.reshape(dirs, (-1, 3)),
            "coeffs": mx.reshape(coeffs, (-1, active_coeffs, 3)),
        },
    )
    return mx.clip(mx.reshape(colors + 0.5, (*dirs.shape[:-1], 3)), 0.0, 1.0)


def camera_batch_arrays(cameras, batch_ids: list[int]) -> tuple[mx.array, mx.array]:
    viewmats = np.stack([cameras[idx].viewmat for idx in batch_ids], axis=0)
    Ks = np.stack([cameras[idx].K for idx in batch_ids], axis=0)
    return (
        mx.array(viewmats[None, ...], dtype=mx.float32),
        mx.array(Ks[None, ...], dtype=mx.float32),
    )


def target_batch_array(targets: list[mx.array], batch_ids: list[int]) -> mx.array:
    return mx.concatenate([targets[idx] for idx in batch_ids], axis=0)


def render_sh_model(
    model: ScannerPointsSHModel,
    viewspace_points: mx.array,
    viewmats: mx.array,
    Ks: mx.array,
    width: int,
    height: int,
    tile_size: int,
    sh_degree: int,
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
            "colors": sh_colors_for_camera(model, viewmats, sh_degree),
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


def local_average(image: mx.array, window_size: int) -> mx.array:
    channels = int(image.shape[-1])
    weight = mx.ones((channels, window_size, window_size, 1), dtype=image.dtype)
    weight = weight / float(window_size * window_size)
    return mx.conv2d(image, weight, padding=window_size // 2, groups=channels)


def ssim_index(image: mx.array, target: mx.array, window_size: int) -> mx.array:
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
    return mx.mean(numerator / mx.maximum(denominator, 1.0e-12))


def image_loss_components(
    image: mx.array,
    target: mx.array,
    loss_mode: str,
    ssim_lambda: float,
    ssim_window_size: int,
) -> dict[str, mx.array]:
    l1 = nn.losses.l1_loss(image, target)
    if loss_mode == "l1":
        return {
            "loss": l1,
            "l1": l1,
            "ssim": mx.array(1.0, dtype=image.dtype),
            "dssim": mx.array(0.0, dtype=image.dtype),
        }
    if loss_mode != "l1_dssim":
        raise ValueError(f"Unsupported loss mode: {loss_mode}")
    ssim = ssim_index(image, target, ssim_window_size)
    dssim = (1.0 - ssim) * 0.5
    loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * dssim
    return {
        "loss": loss,
        "l1": l1,
        "ssim": ssim,
        "dssim": dssim,
    }


def image_training_loss(
    image: mx.array,
    target: mx.array,
    loss_mode: str,
    ssim_lambda: float,
    ssim_window_size: int,
) -> mx.array:
    return image_loss_components(image, target, loss_mode, ssim_lambda, ssim_window_size)["loss"]


def render_loss_stats(
    render: dict,
    target: mx.array,
    loss_mode: str,
    ssim_lambda: float,
    ssim_window_size: int,
) -> tuple[float, float, dict]:
    components = image_loss_components(render["render_colors"], target, loss_mode, ssim_lambda, ssim_window_size)
    diff = render["render_colors"] - target
    mse = mx.mean(diff * diff)
    mx.eval(
        components["loss"],
        components["l1"],
        components["ssim"],
        components["dssim"],
        mse,
        render["render_colors"],
        render["radii"],
        render["flatten_ids"],
    )
    loss_value = float(np.asarray(components["loss"]))
    mse_value = float(np.asarray(mse))
    psnr = float(-10.0 * np.log10(max(mse_value, 1.0e-12)))
    return loss_value, psnr, {
        "l1": float(np.asarray(components["l1"])),
        "ssim": float(np.asarray(components["ssim"])),
        "dssim": float(np.asarray(components["dssim"])),
    }


def evaluate_rgb_frames(
    model: Tiny3DGSModel,
    cameras,
    targets: list[mx.array],
    width: int,
    height: int,
    tile_size: int,
    loss_mode: str,
    ssim_lambda: float,
    ssim_window_size: int,
) -> list[dict]:
    stats = []
    for camera, target in zip(cameras, targets, strict=True):
        viewmats, Ks = camera_arrays(camera)
        viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)
        render = render_model(model, viewspace_points, viewmats, Ks, width, height, tile_size)
        loss, psnr, loss_components = render_loss_stats(render, target, loss_mode, ssim_lambda, ssim_window_size)
        radii = np.asarray(render["radii"])
        flatten_ids = np.asarray(render["flatten_ids"])
        stats.append(
            {
                "frame_index": int(camera.index),
                "loss": loss,
                "loss_components": loss_components,
                "psnr": psnr,
                "visible_gaussians": int(np.count_nonzero(np.any(radii > 0, axis=-1))),
                "intersections": int(flatten_ids.shape[0]),
                "image": np.asarray(render["render_colors"][0], dtype=np.float32),
            }
        )
    return stats


def evaluate_sh_frames(
    model: ScannerPointsSHModel,
    cameras,
    targets: list[mx.array],
    width: int,
    height: int,
    tile_size: int,
    sh_degree: int,
    loss_mode: str,
    ssim_lambda: float,
    ssim_window_size: int,
) -> list[dict]:
    stats = []
    for camera, target in zip(cameras, targets, strict=True):
        viewmats, Ks = camera_arrays(camera)
        viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)
        render = render_sh_model(model, viewspace_points, viewmats, Ks, width, height, tile_size, sh_degree)
        loss, psnr, loss_components = render_loss_stats(render, target, loss_mode, ssim_lambda, ssim_window_size)
        radii = np.asarray(render["radii"])
        flatten_ids = np.asarray(render["flatten_ids"])
        stats.append(
            {
                "frame_index": int(camera.index),
                "loss": loss,
                "loss_components": loss_components,
                "psnr": psnr,
                "visible_gaussians": int(np.count_nonzero(np.any(radii > 0, axis=-1))),
                "intersections": int(flatten_ids.shape[0]),
                "image": np.asarray(render["render_colors"][0], dtype=np.float32),
            }
        )
    return stats


def evaluate_current_frames(
    model: Tiny3DGSModel | ScannerPointsSHModel,
    cameras,
    targets: list[mx.array],
    width: int,
    height: int,
    tile_size: int,
    color_mode: str,
    sh_degree: int,
    loss_mode: str,
    ssim_lambda: float,
    ssim_window_size: int,
) -> list[dict]:
    if color_mode == "rgb":
        return evaluate_rgb_frames(model, cameras, targets, width, height, tile_size, loss_mode, ssim_lambda, ssim_window_size)
    return evaluate_sh_frames(
        model,
        cameras,
        targets,
        width,
        height,
        tile_size,
        sh_degree,
        loss_mode,
        ssim_lambda,
        ssim_window_size,
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


def gsplat_to_spz_axis() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def scales_to_spz(log_scales: np.ndarray, mode: str) -> np.ndarray:
    if mode == "direct":
        return log_scales.astype(np.float32)
    if mode == "scanner_axis":
        out = np.empty_like(log_scales, dtype=np.float32)
        out[:, 0] = log_scales[:, 0]
        out[:, 1] = log_scales[:, 2]
        out[:, 2] = log_scales[:, 1]
        return out
    raise ValueError(f"Unsupported SPZ scale mode: {mode}")


def quats_to_spz(quats: np.ndarray, mode: str) -> np.ndarray:
    if mode == "direct":
        return quats / np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1.0e-8, None)

    position_axis = gsplat_to_spz_axis()
    fastgs_axis = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot = quat_wxyz_to_rotmat(quats)
    if mode == "position_axis":
        return rotmat_to_quat_wxyz(position_axis @ rot)
    if mode == "fastgs_conjugate":
        return rotmat_to_quat_wxyz(fastgs_axis @ rot @ fastgs_axis.T)
    if mode == "position_conjugate":
        return rotmat_to_quat_wxyz(position_axis @ rot @ position_axis.T)
    raise ValueError(f"Unsupported SPZ rotation mode: {mode}")


def quats_to_spz_storage(quats_wxyz: np.ndarray, order: str) -> np.ndarray:
    q = quats_wxyz / np.clip(np.linalg.norm(quats_wxyz, axis=1, keepdims=True), 1.0e-8, None)
    if order == "wxyz":
        return q.astype(np.float32)
    if order == "xyzw":
        return q[:, [1, 2, 3, 0]].astype(np.float32)
    raise ValueError(f"Unsupported SPZ quaternion order: {order}")


def export_trained_spz(
    path: Path,
    model: Tiny3DGSModel | ScannerPointsSHModel,
    color_mode: str,
    sh_degree: int,
    spz_scale_mode: str,
    spz_rotation_mode: str,
    spz_quat_order: str,
    spz_color_mode: str,
) -> int:
    try:
        import spz
    except ImportError as exc:
        raise ImportError("The 'spz' Python package is required for SPZ export.") from exc

    if color_mode == "rgb":
        mx.eval(model.means, model.log_scales, model.normalized_quats, model.color_logits, model.opacity_logits)
    else:
        mx.eval(
            model.means,
            model.log_scales,
            model.normalized_quats,
            model.features_dc,
            model.features_rest,
            model.opacity_logits,
        )
    means = np.asarray(model.means[0], dtype=np.float32)
    log_scales = np.asarray(model.log_scales[0], dtype=np.float32)
    quats = np.asarray(model.normalized_quats[0], dtype=np.float32)
    opacity_logits = np.asarray(model.opacity_logits[0], dtype=np.float32)

    cloud = spz.GaussianCloud()
    cloud.antialiased = True
    cloud.positions = positions_to_spz(means).reshape(-1).astype(np.float32)
    cloud.scales = scales_to_spz(log_scales, spz_scale_mode).reshape(-1).astype(np.float32)
    cloud.rotations = quats_to_spz_storage(quats_to_spz(quats, spz_rotation_mode), spz_quat_order).reshape(-1)
    cloud.alphas = opacity_logits.reshape(-1).astype(np.float32)
    if color_mode == "rgb":
        colors = np.asarray(model.colors[0, 0], dtype=np.float32)
        if spz_color_mode == "sh":
            cloud.colors = ((np.clip(colors, 0.0, 1.0) - 0.5) / SH_C0).reshape(-1).astype(np.float32)
        elif spz_color_mode == "raw_rgb":
            cloud.colors = np.clip(colors, 0.0, 1.0).reshape(-1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported SPZ color mode: {spz_color_mode}")
        cloud.sh_degree = 0
        cloud.sh = np.array([], dtype=np.float32)
    elif color_mode == "sh":
        features_dc = np.asarray(model.features_dc[0], dtype=np.float32)
        cloud.colors = features_dc.reshape(-1).astype(np.float32)
        if spz_color_mode == "sh":
            active_rest = sh_coeff_count(sh_degree) - 1
            features_rest = np.asarray(model.features_rest[0, :, :active_rest, :], dtype=np.float32)
            cloud.sh_degree = int(sh_degree)
            cloud.sh = features_rest.reshape(-1).astype(np.float32)
        elif spz_color_mode == "raw_rgb":
            cloud.colors = np.clip(0.5 + SH_C0 * features_dc, 0.0, 1.0).reshape(-1).astype(np.float32)
            cloud.sh_degree = 0
            cloud.sh = np.array([], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported SPZ color mode: {spz_color_mode}")
    else:
        raise ValueError(f"Unsupported color mode: {color_mode}")

    path.parent.mkdir(parents=True, exist_ok=True)
    opts = spz.PackOptions()
    ok = spz.save_spz(cloud, opts, str(path))
    if not ok:
        raise RuntimeError(f"failed to save spz to {path}")
    return int(means.shape[0])


def array_min_mean_max(values: np.ndarray) -> dict:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": float(flat.min()),
        "mean": float(flat.mean()),
        "max": float(flat.max()),
    }


def vector_axis_min_max(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"min": [], "max": []}
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
    }


def spz_export_diagnostics(
    model: Tiny3DGSModel | ScannerPointsSHModel,
    color_mode: str,
    sh_degree: int,
    spz_scale_mode: str,
    spz_rotation_mode: str,
    spz_quat_order: str,
    spz_color_mode: str,
) -> dict:
    if color_mode == "rgb":
        mx.eval(model.means, model.log_scales, model.normalized_quats, model.color_logits, model.opacity_logits)
    else:
        mx.eval(
            model.means,
            model.log_scales,
            model.normalized_quats,
            model.features_dc,
            model.features_rest,
            model.opacity_logits,
        )

    means = np.asarray(model.means[0], dtype=np.float32)
    spz_positions = positions_to_spz(means)
    log_scales = np.asarray(model.log_scales[0], dtype=np.float32)
    spz_log_scales = scales_to_spz(log_scales, spz_scale_mode)
    quats = np.asarray(model.normalized_quats[0], dtype=np.float32)
    spz_quats = quats_to_spz(quats, spz_rotation_mode)
    spz_stored_quats = quats_to_spz_storage(spz_quats, spz_quat_order)
    opacity_logits = np.asarray(model.opacity_logits[0], dtype=np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacity_logits))

    diagnostics = {
        "axis_mode": "scanner",
        "position_convention": "[x, -z, y]",
        "rotation_convention": {
            "direct": "trained wxyz quats",
            "position_axis": "trained wxyz quats converted with position axis: R_spz = A @ R",
            "fastgs_conjugate": "trained wxyz quats transformed with legacy FastGS conjugation",
            "position_conjugate": "trained wxyz quats transformed with position-axis conjugation",
        }[spz_rotation_mode],
        "rotation_mode": spz_rotation_mode,
        "quat_order": spz_quat_order,
        "color_export_mode": spz_color_mode,
        "scale_convention": (
            "trained log_scales"
            if spz_scale_mode == "direct"
            else "trained log_scales permuted from [sx, sy, sz] to [sx, sz, sy]"
        ),
        "scale_mode": spz_scale_mode,
        "opacity_convention": "trained opacity logits",
        "gaussian_count": int(means.shape[0]),
        "trained_position_min_max": vector_axis_min_max(means),
        "spz_position_min_max": vector_axis_min_max(spz_positions),
        "log_scale": array_min_mean_max(log_scales),
        "scale": array_min_mean_max(np.exp(log_scales)),
        "trained_log_scale_axis_min_max": vector_axis_min_max(log_scales),
        "spz_log_scale_axis_min_max": vector_axis_min_max(spz_log_scales),
        "spz_scale_axis_min_max": vector_axis_min_max(np.exp(spz_log_scales)),
        "opacity_logit": array_min_mean_max(opacity_logits),
        "opacity": array_min_mean_max(opacities),
        "trained_quat_norm": array_min_mean_max(np.linalg.norm(quats, axis=-1)),
        "spz_quat_norm": array_min_mean_max(np.linalg.norm(spz_quats, axis=-1)),
        "spz_stored_quat_norm": array_min_mean_max(np.linalg.norm(spz_stored_quats, axis=-1)),
        "color_mode": color_mode,
        "sh_degree": int(sh_degree) if color_mode == "sh" and spz_color_mode == "sh" else 0,
    }
    if color_mode == "rgb":
        colors = np.asarray(model.colors[0, 0], dtype=np.float32)
        diagnostics.update(
            {
                "colors_shape": list(colors.shape),
                "colors_min_max": vector_axis_min_max(np.clip(colors, 0.0, 1.0)),
                "sh_shape": [0],
            }
        )
    else:
        features_dc = np.asarray(model.features_dc[0], dtype=np.float32)
        active_rest = sh_coeff_count(sh_degree) - 1 if spz_color_mode == "sh" else 0
        features_rest = np.asarray(model.features_rest[0, :, :active_rest, :], dtype=np.float32)
        diagnostics.update(
            {
                "features_dc_shape": list(features_dc.shape),
                "features_dc_min_max": vector_axis_min_max(features_dc),
                "features_rest_shape": list(features_rest.shape),
                "features_rest": array_min_mean_max(features_rest),
                "active_rest_coefficients": int(active_rest),
            }
        )
    return diagnostics


def save_model_parameters_npz(
    path: Path,
    model: Tiny3DGSModel | ScannerPointsSHModel,
    color_mode: str,
    active_sh_degree: int,
    max_sh_degree: int,
    summary: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if color_mode == "rgb":
        mx.eval(model.means, model.normalized_quats, model.log_scales, model.color_logits, model.opacity_logits)
        payload = {
            "color_mode": np.array("rgb", dtype=np.str_),
            "active_sh_degree": np.array(active_sh_degree, dtype=np.int32),
            "max_sh_degree": np.array(max_sh_degree, dtype=np.int32),
            "means": np.asarray(model.means, dtype=np.float32),
            "quats_wxyz": np.asarray(model.normalized_quats, dtype=np.float32),
            "log_scales": np.asarray(model.log_scales, dtype=np.float32),
            "color_logits": np.asarray(model.color_logits, dtype=np.float32),
            "opacity_logits": np.asarray(model.opacity_logits, dtype=np.float32),
            "summary_json": np.array(json.dumps(summary), dtype=np.str_),
        }
    else:
        mx.eval(
            model.means,
            model.normalized_quats,
            model.log_scales,
            model.features_dc,
            model.features_rest,
            model.opacity_logits,
        )
        payload = {
            "color_mode": np.array("sh", dtype=np.str_),
            "active_sh_degree": np.array(active_sh_degree, dtype=np.int32),
            "max_sh_degree": np.array(max_sh_degree, dtype=np.int32),
            "means": np.asarray(model.means, dtype=np.float32),
            "quats_wxyz": np.asarray(model.normalized_quats, dtype=np.float32),
            "log_scales": np.asarray(model.log_scales, dtype=np.float32),
            "features_dc": np.asarray(model.features_dc, dtype=np.float32),
            "features_rest": np.asarray(model.features_rest, dtype=np.float32),
            "opacity_logits": np.asarray(model.opacity_logits, dtype=np.float32),
            "summary_json": np.array(json.dumps(summary), dtype=np.str_),
        }
    np.savez_compressed(path, **payload)


def opacity_diagnostics(model: Tiny3DGSModel | ScannerPointsSHModel) -> dict:
    mx.eval(model.opacity_logits)
    logits = np.asarray(model.opacity_logits[0], dtype=np.float32)
    opacities = 1.0 / (1.0 + np.exp(-logits))
    return {
        "logit_min": float(logits.min()) if logits.size else 0.0,
        "logit_mean": float(logits.mean()) if logits.size else 0.0,
        "logit_max": float(logits.max()) if logits.size else 0.0,
        "opacity_min": float(opacities.min()) if opacities.size else 0.0,
        "opacity_mean": float(opacities.mean()) if opacities.size else 0.0,
        "opacity_max": float(opacities.max()) if opacities.size else 0.0,
    }


def active_sh_degree_for_step(start: int, target: int, interval: int, step: int) -> int:
    if interval <= 0:
        return int(target)
    return int(min(target, start + max(step - 1, 0) // interval))


def scheduled_lr(
    step: int,
    initial_lr: float,
    final_lr: float | None,
    max_steps: int,
    delay_mult: float,
) -> float:
    if final_lr is None:
        return float(initial_lr)
    if initial_lr <= 0.0 or final_lr <= 0.0:
        raise ValueError("scheduled learning rates must be positive")
    if max_steps <= 0:
        return float(final_lr)
    t = float(np.clip(step / max_steps, 0.0, 1.0))
    lr = float(np.exp(np.log(initial_lr) * (1.0 - t) + np.log(final_lr) * t))
    if delay_mult < 1.0:
        delay_rate = delay_mult + (1.0 - delay_mult) * np.sin(0.5 * np.pi * t)
        lr *= float(delay_rate)
    return lr


def validate_lr_schedule_args(
    name: str,
    initial_lr: float,
    final_lr: float | None,
    delay_mult: float,
    max_steps: int,
) -> None:
    if initial_lr <= 0.0:
        raise ValueError(f"--lr-{name} must be positive")
    if final_lr is not None and final_lr <= 0.0:
        raise ValueError(f"--lr-{name}-final must be positive")
    if delay_mult <= 0.0 or delay_mult > 1.0:
        raise ValueError(f"--lr-{name}-delay-mult must be in (0, 1]")
    if max_steps <= 0:
        raise ValueError(f"--lr-{name}-max-steps must be positive")


def make_lr_schedule(
    initial_lr: float,
    final_lr: float | None,
    delay_mult: float,
    max_steps: int,
) -> dict:
    return {
        "initial": float(initial_lr),
        "final": None if final_lr is None else float(final_lr),
        "delay_mult": float(delay_mult),
        "max_steps": int(max_steps),
        "latest": float(initial_lr),
        "history": [],
    }


def lr_for_step(schedule: dict, step: int) -> float:
    return scheduled_lr(
        step,
        schedule["initial"],
        schedule["final"],
        schedule["max_steps"],
        schedule["delay_mult"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/Users/yangdunfu/Downloads/2026_05_04_16_51_29"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scanner_points_multiview_train"))
    parser.add_argument("--out-spz", type=Path, default=None)
    parser.add_argument("--out-model-npz", type=Path, default=None)
    parser.add_argument("--spz-scale-mode", choices=("direct", "scanner_axis"), default="direct")
    parser.add_argument(
        "--spz-rotation-mode",
        choices=("direct", "position_axis", "fastgs_conjugate", "position_conjugate"),
        default="position_axis",
    )
    parser.add_argument("--spz-quat-order", choices=("wxyz", "xyzw"), default="xyzw")
    parser.add_argument("--spz-color-mode", choices=("sh", "raw_rgb"), default="sh")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--eval-max-frames", type=int, default=0)
    parser.add_argument("--eval-frame-step", type=int, default=None)
    parser.add_argument("--eval-start-index", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--num-random-gaussians", type=int, default=0)
    parser.add_argument("--random-gaussian-bounds-scale", type=float, default=1.05)
    parser.add_argument("--point-scale", type=float, default=0.01)
    parser.add_argument("--point-scale-mode", choices=("fixed", "scene_fraction"), default="scene_fraction")
    parser.add_argument("--point-scale-fraction", type=float, default=0.005)
    parser.add_argument("--opacity", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--frame-sampling", choices=("sequential", "shuffle"), default="shuffle")
    parser.add_argument("--frame-shuffle-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--loss-mode", choices=("l1", "l1_dssim"), default="l1_dssim")
    parser.add_argument("--ssim-lambda", type=float, default=0.2)
    parser.add_argument("--ssim-window-size", type=int, default=11)
    parser.add_argument("--lr-means", type=float, default=2.0e-3)
    parser.add_argument("--lr-means-final", type=float, default=None)
    parser.add_argument("--lr-means-delay-mult", type=float, default=1.0)
    parser.add_argument("--lr-means-max-steps", type=int, default=None)
    parser.add_argument("--lr-colors", type=float, default=2.0e-2)
    parser.add_argument("--lr-colors-final", type=float, default=None)
    parser.add_argument("--lr-colors-delay-mult", type=float, default=1.0)
    parser.add_argument("--lr-colors-max-steps", type=int, default=None)
    parser.add_argument("--lr-sh-rest", type=float, default=None)
    parser.add_argument("--lr-sh-rest-final", type=float, default=None)
    parser.add_argument("--lr-sh-rest-delay-mult", type=float, default=1.0)
    parser.add_argument("--lr-sh-rest-max-steps", type=int, default=None)
    parser.add_argument("--lr-opacity", type=float, default=5.0e-3)
    parser.add_argument("--lr-opacity-final", type=float, default=None)
    parser.add_argument("--lr-opacity-delay-mult", type=float, default=1.0)
    parser.add_argument("--lr-opacity-max-steps", type=int, default=None)
    parser.add_argument("--lr-scales", type=float, default=1.0e-3)
    parser.add_argument("--lr-scales-final", type=float, default=None)
    parser.add_argument("--lr-scales-delay-mult", type=float, default=1.0)
    parser.add_argument("--lr-scales-max-steps", type=int, default=None)
    parser.add_argument("--lr-quats", type=float, default=1.0e-3)
    parser.add_argument("--lr-quats-final", type=float, default=None)
    parser.add_argument("--lr-quats-delay-mult", type=float, default=1.0)
    parser.add_argument("--lr-quats-max-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--color-mode", choices=("rgb", "sh"), default="rgb")
    parser.add_argument("--sh-degree", type=int, default=0)
    parser.add_argument("--max-sh-degree", type=int, default=1)
    parser.add_argument("--sh-degree-start", type=int, default=None)
    parser.add_argument("--sh-degree-target", type=int, default=None)
    parser.add_argument("--sh-degree-schedule-interval", type=int, default=0)
    parser.add_argument("--refine-enabled", action="store_true")
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
    parser.add_argument("--refine-scene-scale", type=float, default=1.0)
    parser.add_argument("--refine-scene-scale-mode", choices=("fixed", "points_extent"), default="points_extent")
    parser.add_argument("--refine-absgrad", action="store_true")
    parser.add_argument("--refine-revised-opacity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sh_degree < 0 or args.max_sh_degree < 0:
        raise ValueError("SH degree values must be nonnegative")
    if args.sh_degree > args.max_sh_degree:
        raise ValueError("--sh-degree must be <= --max-sh-degree")
    if args.max_sh_degree > MAX_SUPPORTED_SH_DEGREE:
        raise ValueError(f"--max-sh-degree currently supports up to {MAX_SUPPORTED_SH_DEGREE}")
    sh_degree_start = args.sh_degree if args.sh_degree_start is None else args.sh_degree_start
    sh_degree_target = args.sh_degree if args.sh_degree_target is None else args.sh_degree_target
    if sh_degree_start < 0 or sh_degree_target < 0:
        raise ValueError("--sh-degree-start and --sh-degree-target must be nonnegative")
    if sh_degree_start > sh_degree_target:
        raise ValueError("--sh-degree-start must be <= --sh-degree-target")
    if sh_degree_target > args.max_sh_degree:
        raise ValueError("--sh-degree-target must be <= --max-sh-degree")
    if args.sh_degree_schedule_interval < 0:
        raise ValueError("--sh-degree-schedule-interval must be nonnegative")
    if args.color_mode == "rgb" and (
        args.sh_degree_start is not None or args.sh_degree_target is not None or args.sh_degree_schedule_interval > 0
    ):
        raise ValueError("SH degree schedule arguments require --color-mode sh")
    initial_active_sh_degree = active_sh_degree_for_step(
        sh_degree_start,
        sh_degree_target,
        args.sh_degree_schedule_interval,
        1,
    )
    if args.num_random_gaussians < 0:
        raise ValueError("--num-random-gaussians must be nonnegative")
    if args.random_gaussian_bounds_scale <= 0.0:
        raise ValueError("--random-gaussian-bounds-scale must be positive")
    if args.point_scale <= 0.0:
        raise ValueError("--point-scale must be positive")
    if args.point_scale_fraction <= 0.0:
        raise ValueError("--point-scale-fraction must be positive")
    if args.eval_max_frames < 0:
        raise ValueError("--eval-max-frames must be nonnegative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    eval_frame_step = args.frame_step if args.eval_frame_step is None else args.eval_frame_step
    if eval_frame_step <= 0:
        raise ValueError("--eval-frame-step must be positive")
    if args.refine_every <= 0:
        raise ValueError("--refine-every must be positive")
    if args.refine_reset_every <= 0:
        raise ValueError("--refine-reset-every must be positive")
    if args.refine_stop_iter <= args.refine_start_iter:
        raise ValueError("--refine-stop-iter must be greater than --refine-start-iter")
    if args.refine_scene_scale <= 0.0:
        raise ValueError("--refine-scene-scale must be positive")
    if args.ssim_lambda < 0.0 or args.ssim_lambda > 1.0:
        raise ValueError("--ssim-lambda must be in [0, 1]")
    if args.ssim_window_size <= 0 or args.ssim_window_size % 2 == 0:
        raise ValueError("--ssim-window-size must be a positive odd integer")
    lr_means_max_steps = args.steps if args.lr_means_max_steps is None else args.lr_means_max_steps
    lr_colors_max_steps = args.steps if args.lr_colors_max_steps is None else args.lr_colors_max_steps
    lr_sh_rest = args.lr_colors if args.lr_sh_rest is None else args.lr_sh_rest
    lr_sh_rest_max_steps = args.steps if args.lr_sh_rest_max_steps is None else args.lr_sh_rest_max_steps
    lr_opacity_max_steps = args.steps if args.lr_opacity_max_steps is None else args.lr_opacity_max_steps
    lr_scales_max_steps = args.steps if args.lr_scales_max_steps is None else args.lr_scales_max_steps
    lr_quats_max_steps = args.steps if args.lr_quats_max_steps is None else args.lr_quats_max_steps
    validate_lr_schedule_args(
        "means",
        args.lr_means,
        args.lr_means_final,
        args.lr_means_delay_mult,
        lr_means_max_steps,
    )
    validate_lr_schedule_args(
        "colors",
        args.lr_colors,
        args.lr_colors_final,
        args.lr_colors_delay_mult,
        lr_colors_max_steps,
    )
    validate_lr_schedule_args(
        "sh-rest",
        lr_sh_rest,
        args.lr_sh_rest_final,
        args.lr_sh_rest_delay_mult,
        lr_sh_rest_max_steps,
    )
    validate_lr_schedule_args(
        "opacity",
        args.lr_opacity,
        args.lr_opacity_final,
        args.lr_opacity_delay_mult,
        lr_opacity_max_steps,
    )
    validate_lr_schedule_args(
        "scales",
        args.lr_scales,
        args.lr_scales_final,
        args.lr_scales_delay_mult,
        lr_scales_max_steps,
    )
    validate_lr_schedule_args(
        "quats",
        args.lr_quats,
        args.lr_quats_final,
        args.lr_quats_delay_mult,
        lr_quats_max_steps,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = collect_frames(args.data, args.max_frames, args.frame_step, args.start_index)
    eval_start_index = args.eval_start_index if args.eval_start_index is not None else frames[-1].index + 1
    cameras = [load_camera(frame, args.width, args.height) for frame in frames]
    targets = [
        mx.array(load_target(camera.image_path, args.width, args.height)[None, ...], dtype=mx.float32)
        for camera in cameras
    ]
    eval_frames = collect_frames(args.data, args.eval_max_frames, eval_frame_step, eval_start_index) if args.eval_max_frames > 0 else []
    eval_cameras = [load_camera(frame, args.width, args.height) for frame in eval_frames]
    eval_targets = [
        mx.array(load_target(camera.image_path, args.width, args.height)[None, ...], dtype=mx.float32)
        for camera in eval_cameras
    ]
    points, colors, raw_point_count = prepare_points(args.data, args.max_points, args.seed)
    points, colors = append_random_gaussians(
        points,
        colors,
        args.num_random_gaussians,
        args.seed + 1009,
        args.random_gaussian_bounds_scale,
    )
    point_diagnostics = points_extent_diagnostics(points)
    resolved_point_scale = select_point_scale(
        point_diagnostics,
        args.point_scale_mode,
        args.point_scale,
        args.point_scale_fraction,
    )
    resolved_refine_scene_scale = select_scene_scale(
        point_diagnostics,
        args.refine_scene_scale_mode,
        args.refine_scene_scale,
    )
    initialization_diagnostics = {
        "point_extent": point_diagnostics,
        "point_scale_mode": args.point_scale_mode,
        "point_scale_fixed": float(args.point_scale),
        "point_scale_fraction": float(args.point_scale_fraction),
        "resolved_point_scale": float(resolved_point_scale),
        "refine_scene_scale_mode": args.refine_scene_scale_mode,
        "refine_scene_scale_fixed": float(args.refine_scene_scale),
        "resolved_refine_scene_scale": float(resolved_refine_scene_scale),
    }
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
        scene_scale=resolved_refine_scene_scale,
        absgrad=args.refine_absgrad,
        revised_opacity=args.refine_revised_opacity,
    )
    if args.color_mode == "rgb":
        model = init_rgb_model_from_points(points, colors, resolved_point_scale, args.opacity)
    else:
        model = init_sh_model_from_points(points, colors, resolved_point_scale, args.opacity, args.max_sh_degree)
    strategy = ScannerDefaultStrategyRuntime(strategy_config, initial_gaussians=model.means.shape[1])

    initial_stats = evaluate_current_frames(
        model,
        cameras,
        targets,
        args.width,
        args.height,
        args.tile_size,
        args.color_mode,
        initial_active_sh_degree,
        args.loss_mode,
        args.ssim_lambda,
        args.ssim_window_size,
    )
    eval_initial_stats = evaluate_current_frames(
        model,
        eval_cameras,
        eval_targets,
        args.width,
        args.height,
        args.tile_size,
        args.color_mode,
        initial_active_sh_degree,
        args.loss_mode,
        args.ssim_lambda,
        args.ssim_window_size,
    ) if eval_cameras else []
    initial_mean_loss = mean_loss(initial_stats)
    active_sh_degree = initial_active_sh_degree
    sampler = FrameBatchSampler(
        frame_count=len(cameras),
        batch_size=args.batch_size,
        mode=args.frame_sampling,
        seed=args.seed + 7919 if args.frame_shuffle_seed is None else args.frame_shuffle_seed,
    )
    sh_degree_events = (
        [{"step": 1, "active_sh_degree": int(initial_active_sh_degree)}]
        if args.color_mode == "sh"
        else []
    )

    def rgb_loss_fn(
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
        losses = []
        radii = []
        batch = int(viewmats.shape[1])
        for idx in range(batch):
            render = render_model(
                local,
                viewspace_points[:, idx : idx + 1],
                viewmats[:, idx : idx + 1],
                Ks[:, idx : idx + 1],
                args.width,
                args.height,
                args.tile_size,
            )
            losses.append(
                image_training_loss(
                    render["render_colors"],
                    target[idx : idx + 1],
                    args.loss_mode,
                    args.ssim_lambda,
                    args.ssim_window_size,
                )
            )
            radii.append(render["radii"])
        return mx.mean(mx.stack(losses)), mx.concatenate(radii, axis=1)

    def sh_loss_fn(
        means: mx.array,
        quats: mx.array,
        log_scales: mx.array,
        features_dc: mx.array,
        features_rest: mx.array,
        opacity_logits: mx.array,
        viewspace_points: mx.array,
        viewmats: mx.array,
        Ks: mx.array,
        target: mx.array,
    ) -> mx.array:
        local = ScannerPointsSHModel.from_arrays(
            means,
            quats,
            log_scales,
            features_dc,
            features_rest,
            opacity_logits,
        )
        losses = []
        radii = []
        batch = int(viewmats.shape[1])
        for idx in range(batch):
            render = render_sh_model(
                local,
                viewspace_points[:, idx : idx + 1],
                viewmats[:, idx : idx + 1],
                Ks[:, idx : idx + 1],
                args.width,
                args.height,
                args.tile_size,
                active_sh_degree,
            )
            losses.append(
                image_training_loss(
                    render["render_colors"],
                    target[idx : idx + 1],
                    args.loss_mode,
                    args.ssim_lambda,
                    args.ssim_window_size,
                )
            )
            radii.append(render["radii"])
        return mx.mean(mx.stack(losses)), mx.concatenate(radii, axis=1)

    rgb_grad_fn = mx.value_and_grad(rgb_loss_fn, argnums=(0, 1, 2, 3, 4, 5))
    sh_grad_fn = mx.value_and_grad(sh_loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))
    optimizers = {
        "means": Adam(learning_rate=args.lr_means),
        "quats": Adam(learning_rate=args.lr_quats),
        "log_scales": Adam(learning_rate=args.lr_scales),
        "color_logits": Adam(learning_rate=args.lr_colors),
        "opacity_logits": Adam(learning_rate=args.lr_opacity),
    }
    if args.color_mode == "sh":
        optimizers["features_dc"] = Adam(learning_rate=args.lr_colors)
        optimizers["features_rest"] = Adam(learning_rate=lr_sh_rest)

    lr_schedules = {
        "means": make_lr_schedule(
            args.lr_means,
            args.lr_means_final,
            args.lr_means_delay_mult,
            lr_means_max_steps,
        ),
        "quats": make_lr_schedule(
            args.lr_quats,
            args.lr_quats_final,
            args.lr_quats_delay_mult,
            lr_quats_max_steps,
        ),
        "log_scales": make_lr_schedule(
            args.lr_scales,
            args.lr_scales_final,
            args.lr_scales_delay_mult,
            lr_scales_max_steps,
        ),
        "opacity_logits": make_lr_schedule(
            args.lr_opacity,
            args.lr_opacity_final,
            args.lr_opacity_delay_mult,
            lr_opacity_max_steps,
        ),
    }
    if args.color_mode == "rgb":
        lr_schedules["color_logits"] = make_lr_schedule(
            args.lr_colors,
            args.lr_colors_final,
            args.lr_colors_delay_mult,
            lr_colors_max_steps,
        )
    else:
        lr_schedules["features_dc"] = make_lr_schedule(
            args.lr_colors,
            args.lr_colors_final,
            args.lr_colors_delay_mult,
            lr_colors_max_steps,
        )
        lr_schedules["features_rest"] = make_lr_schedule(
            lr_sh_rest,
            args.lr_sh_rest_final,
            args.lr_sh_rest_delay_mult,
            lr_sh_rest_max_steps,
        )

    last_loss = None
    last_viewspace_grad = None
    last_viewspace_grad_norm = None
    for step in range(1, args.steps + 1):
        latest_lrs = {}
        for name, schedule in lr_schedules.items():
            lr = lr_for_step(schedule, step)
            optimizers[name].learning_rate = lr
            schedule["latest"] = float(lr)
            latest_lrs[name] = float(lr)
        if step == 1 or step == args.steps or step % args.log_interval == 0:
            for name, schedule in lr_schedules.items():
                schedule["history"].append({"step": int(step), "lr": float(schedule["latest"])})
        batch_ids = sampler.next_batch()
        batch_frame_indices = [int(cameras[idx].index) for idx in batch_ids]
        target = target_batch_array(targets, batch_ids)
        viewmats, Ks = camera_batch_arrays(cameras, batch_ids)
        viewspace_points = mx.zeros((1, len(batch_ids), model.means.shape[1], 2), dtype=mx.float32)
        if args.color_mode == "rgb":
            (loss, strategy_radii), grads = rgb_grad_fn(
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
        else:
            next_active_sh_degree = active_sh_degree_for_step(
                sh_degree_start,
                sh_degree_target,
                args.sh_degree_schedule_interval,
                step,
            )
            if next_active_sh_degree != active_sh_degree:
                active_sh_degree = next_active_sh_degree
                sh_degree_events.append({"step": int(step), "active_sh_degree": int(active_sh_degree)})
            (loss, strategy_radii), grads = sh_grad_fn(
                model.means,
                model.quats,
                model.log_scales,
                model.features_dc,
                model.features_rest,
                model.opacity_logits,
                viewspace_points,
                viewmats,
                Ks,
                target,
            )
            (
                d_means,
                d_quats,
                d_log_scales,
                d_features_dc,
                d_features_rest,
                d_opacity_logits,
                d_viewspace,
            ) = grads
        mx.eval(loss, d_viewspace)
        last_loss = float(np.asarray(loss))
        last_viewspace_grad = d_viewspace
        last_viewspace_grad_norm = float(np.linalg.norm(np.asarray(d_viewspace)))

        optimizers["means"].update(model, {"means": d_means})
        optimizers["quats"].update(model, {"quats": d_quats})
        optimizers["log_scales"].update(model, {"log_scales": d_log_scales})
        if args.color_mode == "rgb":
            optimizers["color_logits"].update(model, {"color_logits": d_color_logits})
        else:
            optimizers["features_dc"].update(model, {"features_dc": d_features_dc})
            optimizers["features_rest"].update(model, {"features_rest": d_features_rest})
        optimizers["opacity_logits"].update(model, {"opacity_logits": d_opacity_logits})
        model.quats = normalize_quats(model.quats)
        if args.color_mode == "rgb":
            mx.eval(model.means, model.quats, model.log_scales, model.color_logits, model.opacity_logits)
        else:
            mx.eval(
                model.means,
                model.quats,
                model.log_scales,
                model.features_dc,
                model.features_rest,
                model.opacity_logits,
            )
        if strategy.config.enabled:
            strategy.update_state(
                d_viewspace,
                strategy_radii,
                width=args.width,
                height=args.height,
                n_cameras=len(batch_ids),
            )
        strategy.after_optimizer_step(step, model, optimizers, args.color_mode)

        if step == 1 or step == args.steps or step % args.log_interval == 0:
            print(
                f"step={step:04d} frames={batch_frame_indices} "
                f"loss={last_loss:.8f} means_lr={latest_lrs['means']:.8g} "
                f"opacity_lr={latest_lrs['opacity_logits']:.8g} "
                f"viewspace_grad_norm={last_viewspace_grad_norm:.8f}"
            )

    final_stats = evaluate_current_frames(
        model,
        cameras,
        targets,
        args.width,
        args.height,
        args.tile_size,
        args.color_mode,
        active_sh_degree,
        args.loss_mode,
        args.ssim_lambda,
        args.ssim_window_size,
    )
    eval_final_stats = evaluate_current_frames(
        model,
        eval_cameras,
        eval_targets,
        args.width,
        args.height,
        args.tile_size,
        args.color_mode,
        active_sh_degree,
        args.loss_mode,
        args.ssim_lambda,
        args.ssim_window_size,
    ) if eval_cameras else []
    final_mean_loss = mean_loss(final_stats)
    target_images = [np.asarray(target[0], dtype=np.float32) for target in targets]
    for initial, final, target_image in zip(initial_stats, final_stats, target_images, strict=True):
        frame_index = final["frame_index"]
        write_png(
            args.out_dir / f"compare_frame_{frame_index:05d}.png",
            image_to_u8(concat_compare(target_image, initial["image"], final["image"])),
        )
    eval_target_images = [np.asarray(target[0], dtype=np.float32) for target in eval_targets]
    for initial, final, target_image in zip(eval_initial_stats, eval_final_stats, eval_target_images, strict=True):
        frame_index = final["frame_index"]
        write_png(
            args.out_dir / f"compare_eval_frame_{frame_index:05d}.png",
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
    export_sh_degree = active_sh_degree if args.color_mode == "sh" else args.sh_degree
    spz_diagnostics = spz_export_diagnostics(
        model,
        args.color_mode,
        export_sh_degree,
        args.spz_scale_mode,
        args.spz_rotation_mode,
        args.spz_quat_order,
        args.spz_color_mode,
    )
    exported_gaussians = export_trained_spz(
        out_spz,
        model,
        args.color_mode,
        export_sh_degree,
        args.spz_scale_mode,
        args.spz_rotation_mode,
        args.spz_quat_order,
        args.spz_color_mode,
    )
    spz_size = out_spz.stat().st_size
    if spz_size <= 0:
        raise AssertionError(f"SPZ output is empty: {out_spz}")

    final_opacity_diagnostics = opacity_diagnostics(model)
    refinement_summary = strategy.summary()
    frame_summaries = []
    for initial, final in zip(initial_stats, final_stats, strict=True):
        frame_summaries.append(
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
        )
    eval_frame_summaries = []
    for initial, final in zip(eval_initial_stats, eval_final_stats, strict=True):
        eval_frame_summaries.append(
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
        )
    eval_initial_mean_loss = mean_loss(eval_initial_stats) if eval_initial_stats else None
    eval_final_mean_loss = mean_loss(eval_final_stats) if eval_final_stats else None
    eval_final_mean_psnr = (
        float(np.mean([item["final_psnr"] for item in eval_frame_summaries]))
        if eval_frame_summaries
        else None
    )
    train_eval_loss_gap = (
        float(eval_final_mean_loss - final_mean_loss)
        if eval_final_mean_loss is not None
        else None
    )
    preview_diagnostics = {
        "loss_delta": float(final_mean_loss - initial_mean_loss),
        "loss_ratio": float(final_mean_loss / initial_mean_loss) if initial_mean_loss != 0.0 else None,
        "last_viewspace_grad_norm": last_viewspace_grad_norm,
        "initial_visible_gaussians_total": int(sum(item["initial_visible_gaussians"] for item in frame_summaries)),
        "final_visible_gaussians_total": int(sum(item["final_visible_gaussians"] for item in frame_summaries)),
        "initial_intersections_total": int(sum(item["initial_intersections"] for item in frame_summaries)),
        "final_intersections_total": int(sum(item["final_intersections"] for item in frame_summaries)),
        "eval_final_mean_loss": eval_final_mean_loss,
        "train_eval_loss_gap": train_eval_loss_gap,
        "refinement_operation_totals": refinement_summary["operation_totals"],
        "refinement_latest_event": refinement_summary["latest_event"],
        "final_opacity": final_opacity_diagnostics,
    }
    summary = {
        "dataset": str(args.data),
        "width": args.width,
        "height": args.height,
        "raw_point_count": raw_point_count,
        "exported_gaussians": exported_gaussians,
        "max_points": args.max_points,
        "point_cloud_gaussians": int(points.shape[0] - args.num_random_gaussians),
        "random_gaussians": args.num_random_gaussians,
        "random_gaussian_bounds_scale": args.random_gaussian_bounds_scale,
        "frames": len(cameras),
        "eval_frames": len(eval_cameras),
        "eval_start_index": eval_start_index if eval_cameras else None,
        "eval_frame_step": eval_frame_step if eval_cameras else None,
        "steps": args.steps,
        "initialization": initialization_diagnostics,
        "dataloader": sampler.summary(cameras),
        "loss_function": args.loss_mode,
        "loss_config": {
            "mode": args.loss_mode,
            "ssim_lambda": float(args.ssim_lambda),
            "ssim_window_size": int(args.ssim_window_size),
            "dssim_formula": "(1 - ssim) / 2",
        },
        "psnr_metric": "computed from render-target MSE for image-quality diagnostics",
        "image_outputs": "compare_frame_*.png and optional compare_eval_frame_*.png",
        "learning_rate_schedule": lr_schedules,
        "initial_mean_loss": initial_mean_loss,
        "final_mean_loss": final_mean_loss,
        "last_viewspace_grad_norm": last_viewspace_grad_norm,
        "spz": str(out_spz),
        "spz_file_size_bytes": spz_size,
        "spz_position_convention": "[x, -z, y]",
        "spz_scale_mode": args.spz_scale_mode,
        "spz_scale_convention": spz_diagnostics["scale_convention"],
        "spz_opacity_convention": "trained opacity logits",
        "spz_rotation_mode": args.spz_rotation_mode,
        "spz_rotation_convention": spz_diagnostics["rotation_convention"],
        "spz_quat_order": args.spz_quat_order,
        "spz_color_mode": args.spz_color_mode,
        "spz_export_diagnostics": spz_diagnostics,
        "color_mode": args.color_mode,
        "color_path": "rgb_logits" if args.color_mode == "rgb" else "spherical_harmonics",
        "sh_degree": None if args.color_mode == "rgb" else args.sh_degree,
        "active_sh_degree_final": None if args.color_mode == "rgb" else active_sh_degree,
        "export_sh_degree": None if args.color_mode == "rgb" else export_sh_degree,
        "sh_degree_schedule": None
        if args.color_mode == "rgb"
        else {
            "start": int(sh_degree_start),
            "target": int(sh_degree_target),
            "interval": int(args.sh_degree_schedule_interval),
            "initial_active_degree": int(initial_active_sh_degree),
            "final_active_degree": int(active_sh_degree),
            "events": sh_degree_events,
        },
        "max_sh_degree": None if args.color_mode == "rgb" else args.max_sh_degree,
        "sh_coeff_count": None if args.color_mode == "rgb" else sh_coeff_count(active_sh_degree),
        "max_sh_coeff_count": None if args.color_mode == "rgb" else sh_coeff_count(args.max_sh_degree),
        "final_opacity_diagnostics": final_opacity_diagnostics,
        "preview_diagnostics": preview_diagnostics,
        "eval_diagnostics": {
            "enabled": bool(eval_cameras),
            "initial_mean_loss": eval_initial_mean_loss,
            "final_mean_loss": eval_final_mean_loss,
            "final_mean_psnr": eval_final_mean_psnr,
            "train_eval_loss_gap": train_eval_loss_gap,
            "final_visible_gaussians_total": int(sum(item["final_visible_gaussians"] for item in eval_frame_summaries)),
            "final_intersections_total": int(sum(item["final_intersections"] for item in eval_frame_summaries)),
        },
        "spz_color_convention": (
            "colors stores clipped RGB values; sh is empty"
            if args.color_mode == "rgb"
            else "colors stores SH degree-0 coefficients; sh stores higher-order coefficients"
        ),
        "refinement_strategy": refinement_summary,
        "frame_summaries": frame_summaries,
        "eval_frame_summaries": eval_frame_summaries,
    }
    out_model_npz = args.out_model_npz if args.out_model_npz is not None else args.out_dir / "trained_model_params.npz"
    summary["model_npz"] = str(out_model_npz)
    save_model_parameters_npz(
        out_model_npz,
        model,
        args.color_mode,
        active_sh_degree,
        args.max_sh_degree,
        summary,
    )
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
