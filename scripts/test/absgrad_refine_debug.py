#!/usr/bin/env python3

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from train_scanner_points_multiview_3dgs_mlx import (
    ScannerDefaultStrategyConfig,
    ScannerDefaultStrategyRuntime,
)


def stats_for(absgrad: bool, d_viewspace_abs: mx.array | None) -> dict:
    strategy = ScannerDefaultStrategyRuntime(
        ScannerDefaultStrategyConfig(
            enabled=True,
            absgrad=absgrad,
            refine_scale2d_stop_iter=5,
        ),
        initial_gaussians=2,
    )
    d_viewspace = mx.array(
        [[[[0.3, -0.4], [-0.1, 0.2]]]],
        dtype=mx.float32,
    )
    radii = mx.array([[[[3.0, 2.0], [4.0, 1.0]]]], dtype=mx.float32)
    strategy.update_state(
        d_viewspace,
        radii,
        width=100,
        height=80,
        n_cameras=1,
        d_viewspace_abs=d_viewspace_abs,
    )
    return {
        "grad2d": strategy.grad2d.astype(float).tolist(),
        "count": strategy.count.astype(float).tolist(),
        "radii": None if strategy.radii is None else strategy.radii.astype(float).tolist(),
        "stats": strategy.last_grad2d_stats,
        "mode": strategy.last_grad2d_mode,
        "fallback_count": strategy.absgrad_fallback_count,
    }


def main() -> None:
    true_absgrad = mx.array(
        [[[[0.9, 0.8], [0.5, 0.6]]]],
        dtype=mx.float32,
    )
    signed = stats_for(absgrad=False, d_viewspace_abs=None)
    abs_mode = stats_for(absgrad=True, d_viewspace_abs=true_absgrad)
    fallback = stats_for(absgrad=True, d_viewspace_abs=None)

    report = {
        "signed": signed,
        "absgrad": abs_mode,
        "fallback": fallback,
    }
    print("=== absgrad refine debug ===")
    print(json.dumps(report, indent=2, sort_keys=True))

    if signed["mode"] != "signed_grad_norm":
        raise AssertionError(f"unexpected signed mode: {signed['mode']}")
    if abs_mode["mode"] != "absgrad_norm":
        raise AssertionError(f"unexpected absgrad mode: {abs_mode['mode']}")
    if fallback["mode"] != "absgrad_requested_signed_grad_fallback":
        raise AssertionError(f"unexpected fallback mode: {fallback['mode']}")
    if fallback["fallback_count"] != 1:
        raise AssertionError(f"expected one absgrad fallback, got {fallback['fallback_count']}")
    if not abs_mode["grad2d"][0] > signed["grad2d"][0]:
        raise AssertionError("provided absgrad should drive a different/larger grad2d accumulation")
    if abs_mode["grad2d"] == fallback["grad2d"]:
        raise AssertionError("real absgrad path should differ from signed-gradient fallback in this fixture")

    print("absgrad refine debug ok")


if __name__ == "__main__":
    main()
