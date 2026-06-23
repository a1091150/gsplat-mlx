#!/usr/bin/env python3

from __future__ import annotations

import sys

import train_scanapp_depth_consistency_multiview_3dgs_mlx as scanapp_trainer


DEFAULT_ARGS = [
    "--width",
    "960",
    "--height",
    "720",
    "--shared-intrinsics",
    "median",
    "--prior-init-mode",
    "knn",
    "--no-keyframe-filter-enabled",
    "--no-consistency-filter-enabled",
    "--refine-reset-every",
    "3000",
    "--refine-start-iter",
    "500",
    "--refine-stop-iter",
    "15000",
    "--refine-every",
    "100",
    "--refine-prune-opa",
    "0.005",
    "--refine-grow-grad2d",
    "0.0002",
    "--refine-grow-scale3d",
    "0.01",
    "--refine-grow-scale2d",
    "0.05",
    "--refine-prune-scale3d",
    "0.1",
    "--refine-prune-scale2d",
    "0.15",
    "--refine-scale2d-stop-iter",
    "0",
    "--refine-pause-after-reset",
    "0",
    "--summary-dataset-type",
    "scanapp_depth_gsplat_default_medianK",
    "--summary-experiment",
    "ScanApp depth baseline using resized 960x720 images, median K, KNN scale initialization, and gsplat DefaultStrategy parameters.",
]


def main() -> None:
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    scanapp_trainer.main()


if __name__ == "__main__":
    main()
