#!/usr/bin/env python3

from __future__ import annotations

import json

import numpy as np

from scanner_points_training_utils import (
    ScannerDefaultStrategyConfig,
    ScannerDefaultStrategyRuntime,
    init_rgb_model_from_points,
)


def print_json(label: str, value) -> None:
    print(f"{label}:")
    print(json.dumps(value, indent=2, sort_keys=True))


def seed_strategy_stats(strategy: ScannerDefaultStrategyRuntime) -> None:
    strategy.grad2d[:] = np.array([0.8, 0.5, 0.0], dtype=np.float32)
    strategy.count[:] = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    if strategy.radii is not None:
        strategy.radii[:] = np.array([0.12, 0.07, 0.0], dtype=np.float32)
    strategy.last_grad2d_stats = strategy._grad2d_stats()


def make_model():
    points = np.array(
        [
            [-0.1, -0.1, 2.0],
            [0.1, 0.0, 2.1],
            [0.0, 0.12, 2.2],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [0.8, 0.2, 0.2],
            [0.2, 0.8, 0.2],
            [0.2, 0.2, 0.8],
        ],
        dtype=np.float32,
    )
    return init_rgb_model_from_points(points, colors, point_scale=0.001, opacity=0.5)


def run_refine_reset_case() -> None:
    model = make_model()
    strategy = ScannerDefaultStrategyRuntime(
        ScannerDefaultStrategyConfig(
            enabled=True,
            grow_grad2d=0.1,
            grow_scale3d=1.0,
            refine_scale2d_stop_iter=10,
            refine_start_iter=0,
            refine_stop_iter=10,
            refine_every=1,
            reset_every=100,
            scene_scale=1.0,
        ),
        initial_gaussians=3,
    )
    seed_strategy_stats(strategy)
    before = strategy._grad2d_stats()
    strategy.after_optimizer_step(step=1, model=model, optimizers={}, color_mode="rgb")
    event = strategy.events[-1]
    after = strategy._grad2d_stats()

    print("=== refine reset case ===")
    print(f"gaussians: before={event['num_gaussians_before']} after={event['num_gaussians_after']}")
    print(f"n_clone={event['n_clone']} n_split={event['n_split']} n_prune={event['n_prune']}")
    print_json("stats before after_optimizer_step", before)
    print_json("event stats before reset", event["grad2d_stats_before_reset"])
    print_json("event stats after reset", event["grad2d_stats_after_reset"])
    print_json("current stats", after)

    if not event["stats_reset_after_refine"]:
        raise AssertionError("expected refine step to reset running stats")
    if event["num_gaussians_after"] <= event["num_gaussians_before"]:
        raise AssertionError("debug setup expected clone to increase Gaussian count")
    if after["total_observations"] != 0:
        raise AssertionError("expected grad/count observations to be zero after refine reset")
    if after["visible_gaussians"] != 0:
        raise AssertionError("expected visible Gaussian count to be zero after refine reset")
    if after["radii_max"] != 0.0:
        raise AssertionError("expected radii stats to be zero after refine reset")


def run_opacity_reset_only_case() -> None:
    model = make_model()
    strategy = ScannerDefaultStrategyRuntime(
        ScannerDefaultStrategyConfig(
            enabled=True,
            refine_start_iter=100,
            refine_stop_iter=200,
            refine_every=10,
            reset_every=2,
        ),
        initial_gaussians=3,
    )
    seed_strategy_stats(strategy)
    before = strategy._grad2d_stats()
    strategy.after_optimizer_step(step=2, model=model, optimizers={}, color_mode="rgb")
    event = strategy.events[-1]
    after = strategy._grad2d_stats()

    print("=== opacity reset only case ===")
    print(f"n_opacity_reset={event['n_opacity_reset']} stats_reset_after_refine={event['stats_reset_after_refine']}")
    print_json("stats before after_optimizer_step", before)
    print_json("current stats", after)

    if event["stats_reset_after_refine"]:
        raise AssertionError("opacity-only reset should not clear refine running stats")
    if after["total_observations"] != before["total_observations"]:
        raise AssertionError("opacity-only reset should preserve grad/count observations")


def main() -> None:
    run_refine_reset_case()
    run_opacity_reset_only_case()
    print("refine state reset debug ok")


if __name__ == "__main__":
    main()
