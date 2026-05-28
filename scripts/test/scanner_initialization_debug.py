#!/usr/bin/env python3

from __future__ import annotations

import json

import numpy as np

from train_scanner_points_multiview_3dgs_mlx import (
    points_extent_diagnostics,
    select_point_scale,
    select_scene_scale,
)


def main() -> None:
    points = np.array(
        [
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
            [0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    )
    diagnostics = points_extent_diagnostics(points)
    fixed_point_scale = select_point_scale(diagnostics, "fixed", fixed_scale=0.01, fraction=0.005)
    fraction_point_scale = select_point_scale(diagnostics, "scene_fraction", fixed_scale=0.01, fraction=0.005)
    fixed_scene_scale = select_scene_scale(diagnostics, "fixed", fixed_scale=1.25)
    extent_scene_scale = select_scene_scale(diagnostics, "points_extent", fixed_scale=1.25)

    report = {
        "diagnostics": diagnostics,
        "fixed_point_scale": fixed_point_scale,
        "fraction_point_scale": fraction_point_scale,
        "fixed_scene_scale": fixed_scene_scale,
        "extent_scene_scale": extent_scene_scale,
    }
    print("=== scanner initialization debug ===")
    print(json.dumps(report, indent=2, sort_keys=True))

    if diagnostics["point_count"] != points.shape[0]:
        raise AssertionError("point count diagnostic mismatch")
    if not np.allclose(diagnostics["center"], [0.0, 0.0, 0.0], atol=1.0e-6):
        raise AssertionError(f"unexpected center: {diagnostics['center']}")
    if not np.isclose(fixed_point_scale, 0.01):
        raise AssertionError(f"fixed point scale changed unexpectedly: {fixed_point_scale}")
    if not fraction_point_scale > 0.0:
        raise AssertionError("scene-fraction point scale should be positive")
    if not np.isclose(fixed_scene_scale, 1.25):
        raise AssertionError(f"fixed scene scale changed unexpectedly: {fixed_scene_scale}")
    if not extent_scene_scale > 0.0:
        raise AssertionError("points-extent scene scale should be positive")
    if not np.isclose(fraction_point_scale, extent_scene_scale * 0.005):
        raise AssertionError("point scale fraction should be relative to extent scene scale")

    print("scanner initialization debug ok")


if __name__ == "__main__":
    main()
