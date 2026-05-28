#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def axis_min_max(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"shape": list(values.shape), "min": [], "max": []}
    return {
        "shape": list(values.shape),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
    }


def scalar_min_mean_max(values: np.ndarray) -> dict:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {"shape": list(values.shape), "min": None, "mean": None, "max": None}
    return {
        "shape": list(values.shape),
        "min": float(flat.min()),
        "mean": float(flat.mean()),
        "max": float(flat.max()),
    }


def reshape_or_empty(values, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, width)
    return arr.reshape(-1, width)


def load_spz(path: Path):
    import spz

    try:
        return spz.load_spz(str(path), spz.UnpackOptions())
    except TypeError:
        return spz.load_spz(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spz",
        type=Path,
        default=Path("outputs/scanner_points_multiview_train/trained_scanner_points.spz"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/scanner_points_multiview_train/training_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cloud = load_spz(args.spz)
    positions = reshape_or_empty(cloud.positions, 3)
    scales = reshape_or_empty(cloud.scales, 3)
    rotations = reshape_or_empty(cloud.rotations, 4)
    colors = reshape_or_empty(cloud.colors, 3)
    sh = np.asarray(cloud.sh, dtype=np.float32)
    alphas = np.asarray(cloud.alphas, dtype=np.float32)

    summary = None
    if args.summary.exists():
        summary = json.loads(args.summary.read_text(encoding="utf-8"))

    rotation_norms = np.linalg.norm(rotations, axis=1) if rotations.size else np.array([], dtype=np.float32)
    report = {
        "spz": str(args.spz),
        "file_size_bytes": args.spz.stat().st_size if args.spz.exists() else None,
        "num_points": int(getattr(cloud, "num_points", positions.shape[0])),
        "antialiased": bool(getattr(cloud, "antialiased", False)),
        "sh_degree": int(getattr(cloud, "sh_degree", 0)),
        "positions": axis_min_max(positions),
        "scales_log": axis_min_max(scales),
        "scales_exp": axis_min_max(np.exp(scales)) if scales.size else axis_min_max(scales),
        "alphas": scalar_min_mean_max(alphas),
        "rotations": axis_min_max(rotations),
        "rotation_norms": scalar_min_mean_max(rotation_norms),
        "colors": axis_min_max(colors),
        "sh": scalar_min_mean_max(sh),
        "summary_spz_scale_mode": None if summary is None else summary.get("spz_scale_mode"),
        "summary_spz_scale_convention": None if summary is None else summary.get("spz_scale_convention"),
        "summary_spz_rotation_mode": None if summary is None else summary.get("spz_rotation_mode"),
        "summary_spz_rotation_convention": None if summary is None else summary.get("spz_rotation_convention"),
        "summary_spz_export_diagnostics": None if summary is None else summary.get("spz_export_diagnostics"),
    }
    print("=== spz convention debug ===")
    print(json.dumps(report, indent=2, sort_keys=True))

    if positions.shape[0] == 0:
        raise AssertionError("SPZ contains no positions")
    if rotations.size and not np.all(np.isfinite(rotation_norms)):
        raise AssertionError("SPZ rotation norms contain non-finite values")
    if rotations.size and np.max(np.abs(rotation_norms - 1.0)) > 2.0e-3:
        raise AssertionError("SPZ rotations are not normalized")

    print("spz convention debug ok")


if __name__ == "__main__":
    main()
