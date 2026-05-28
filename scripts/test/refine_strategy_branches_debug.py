#!/usr/bin/env python3

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
from mlx.optimizers import Adam

from train_scanner_points_multiview_3dgs_mlx import (
    ScannerDefaultStrategyConfig,
    ScannerDefaultStrategyRuntime,
    init_rgb_model_from_points,
    mx_logit,
)


def print_json(label: str, value) -> None:
    print(f"{label}:")
    print(json.dumps(value, indent=2, sort_keys=True))


def make_model(n: int = 3, point_scale: float = 0.01, opacity: float = 0.65):
    points = np.zeros((n, 3), dtype=np.float32)
    points[:, 0] = np.linspace(-0.1, 0.1, n, dtype=np.float32)
    points[:, 2] = np.linspace(2.0, 2.2, n, dtype=np.float32)
    colors = np.tile(np.array([[0.7, 0.3, 0.2]], dtype=np.float32), (n, 1))
    return init_rgb_model_from_points(points, colors, point_scale=point_scale, opacity=opacity)


def set_log_scales(model, scales: list[float]) -> None:
    arr = np.repeat(np.array(scales, dtype=np.float32)[:, None], 3, axis=1)
    model.log_scales = mx.array(np.log(arr)[None, ...], dtype=mx.float32)


def set_opacity(model, values: list[float]) -> None:
    model.opacity_logits = mx_logit(mx.array(np.array(values, dtype=np.float32)[None, ...], dtype=mx.float32))


def run_scale2d_split_case() -> dict:
    model = make_model(n=3, point_scale=0.001)
    strategy = ScannerDefaultStrategyRuntime(
        ScannerDefaultStrategyConfig(
            enabled=True,
            grow_grad2d=10.0,
            grow_scale3d=0.01,
            grow_scale2d=0.05,
            refine_scale2d_stop_iter=10,
            refine_start_iter=0,
            refine_stop_iter=10,
            refine_every=1,
            reset_every=100,
        ),
        initial_gaussians=3,
    )
    strategy.grad2d[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    strategy.count[:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    strategy.radii[:] = np.array([0.08, 0.01, 0.01], dtype=np.float32)
    strategy.after_optimizer_step(step=1, model=model, optimizers={}, color_mode="rgb")
    event = strategy.events[-1]
    out = {
        "event": event,
        "final_gaussians": int(model.means.shape[1]),
    }
    if event["n_split"] != 1:
        raise AssertionError(f"expected one scale2d-triggered split, got {event['n_split']}")
    return out


def run_revised_opacity_split_case() -> dict:
    parent_opacity = 0.64
    expected_child_opacity = 1.0 - np.sqrt(1.0 - parent_opacity)
    model = make_model(n=2, point_scale=0.1, opacity=parent_opacity)
    strategy = ScannerDefaultStrategyRuntime(
        ScannerDefaultStrategyConfig(
            enabled=True,
            grow_grad2d=0.1,
            grow_scale3d=0.01,
            refine_start_iter=0,
            refine_stop_iter=10,
            refine_every=1,
            reset_every=100,
            revised_opacity=True,
        ),
        initial_gaussians=2,
    )
    strategy.grad2d[:] = np.array([1.0, 0.0], dtype=np.float32)
    strategy.count[:] = np.array([1.0, 1.0], dtype=np.float32)
    strategy.after_optimizer_step(step=1, model=model, optimizers={}, color_mode="rgb")
    event = strategy.events[-1]
    mx.eval(model.opacity_logits)
    opacities = 1.0 / (1.0 + np.exp(-np.asarray(model.opacity_logits[0], dtype=np.float32)))
    child_opacities = opacities[-2:]
    out = {
        "event": event,
        "expected_child_opacity": float(expected_child_opacity),
        "child_opacities": child_opacities.astype(float).tolist(),
        "final_opacities": opacities.astype(float).tolist(),
    }
    if event["n_split"] != 1:
        raise AssertionError(f"expected one revised-opacity split, got {event['n_split']}")
    if not np.allclose(child_opacities, expected_child_opacity, atol=1.0e-5):
        raise AssertionError(f"unexpected child opacities: {child_opacities}")
    return out


def run_scale3d_prune_case() -> dict:
    model = make_model(n=3, point_scale=0.01)
    set_log_scales(model, [0.5, 0.001, 0.001])
    strategy = ScannerDefaultStrategyRuntime(
        ScannerDefaultStrategyConfig(
            enabled=True,
            grow_grad2d=10.0,
            prune_scale3d=0.1,
            refine_start_iter=0,
            refine_stop_iter=10,
            refine_every=1,
            reset_every=1,
            scene_scale=1.0,
        ),
        initial_gaussians=3,
    )
    strategy.grad2d[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    strategy.count[:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    strategy.after_optimizer_step(step=2, model=model, optimizers={}, color_mode="rgb")
    event = strategy.events[-1]
    out = {
        "event": event,
        "final_gaussians": int(model.means.shape[1]),
    }
    if event["prune_breakdown"]["scale3d"] != 1 or event["n_prune"] != 1:
        raise AssertionError(f"expected one scale3d prune, got {event['prune_breakdown']}")
    return out


def run_opacity_reset_timing_case() -> dict:
    model = make_model(n=3, point_scale=0.01, opacity=0.8)
    optimizer = Adam(learning_rate=0.1)
    optimizer.update(model, {"opacity_logits": mx.ones_like(model.opacity_logits)})
    mx.eval(model.opacity_logits)
    before_m = float(np.asarray(mx.sum(optimizer.state["opacity_logits"]["m"])))
    before_events = 0
    strategy = ScannerDefaultStrategyRuntime(
        ScannerDefaultStrategyConfig(
            enabled=True,
            refine_start_iter=100,
            refine_stop_iter=200,
            refine_every=10,
            reset_every=3,
            prune_opa=0.005,
        ),
        initial_gaussians=3,
    )
    strategy.after_optimizer_step(step=2, model=model, optimizers={"opacity_logits": optimizer}, color_mode="rgb")
    before_events = len(strategy.events)
    strategy.after_optimizer_step(step=3, model=model, optimizers={"opacity_logits": optimizer}, color_mode="rgb")
    event = strategy.events[-1]
    mx.eval(model.opacity_logits, optimizer.state["opacity_logits"]["m"], optimizer.state["opacity_logits"]["v"])
    logits = np.asarray(model.opacity_logits[0], dtype=np.float32)
    after_m = float(np.asarray(mx.sum(optimizer.state["opacity_logits"]["m"])))
    after_v = float(np.asarray(mx.sum(optimizer.state["opacity_logits"]["v"])))
    out = {
        "events_before_reset_step": before_events,
        "event": event,
        "opacity_reset_target_logit": strategy.opacity_reset_target_logit(),
        "logits_after_reset": logits.astype(float).tolist(),
        "optimizer_m_sum_before": before_m,
        "optimizer_m_sum_after": after_m,
        "optimizer_v_sum_after": after_v,
    }
    if before_events != 0:
        raise AssertionError("step before reset_every should not create a strategy event")
    if event["n_opacity_reset"] != 3:
        raise AssertionError(f"expected opacity reset for 3 Gaussians, got {event['n_opacity_reset']}")
    if np.any(logits > strategy.opacity_reset_target_logit() + 1.0e-5):
        raise AssertionError("opacity reset should clamp logits to reset target")
    if after_m != 0.0 or after_v != 0.0:
        raise AssertionError("opacity reset should clear Adam m/v state for opacity logits")
    return out


def main() -> None:
    report = {
        "scale2d_split": run_scale2d_split_case(),
        "revised_opacity_split": run_revised_opacity_split_case(),
        "scale3d_prune": run_scale3d_prune_case(),
        "opacity_reset_timing": run_opacity_reset_timing_case(),
    }
    print("=== refine strategy branches debug ===")
    print_json("report", report)
    print("refine strategy branches debug ok")


if __name__ == "__main__":
    main()
