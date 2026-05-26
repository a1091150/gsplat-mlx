#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(default_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("refs") / default_name,
        help="Output .npz path.",
    )
    return parser.parse_args()


def load_torch_cuda() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available.")
    return torch


def load_gsplat_wrapper() -> Any:
    from gsplat.cuda import _wrapper

    return _wrapper


def to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy()


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_arrays = {
        key: to_numpy(value) if hasattr(value, "detach") else np.asarray(value)
        for key, value in arrays.items()
    }
    np.savez(path, **numpy_arrays)
    print(f"wrote {path}")
    for key, value in numpy_arrays.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
