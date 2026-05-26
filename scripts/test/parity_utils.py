#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GSPALT_SUBMODULE = ROOT / "submodules" / "gsplat"


class SkipParity(RuntimeError):
    pass


def load_torch_cuda() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        raise SkipParity(f"PyTorch is not available: {exc}") from exc
    if not torch.cuda.is_available():
        raise SkipParity("PyTorch CUDA is not available.")
    return torch


def load_gsplat_wrapper() -> Any:
    if not GSPALT_SUBMODULE.exists():
        raise SkipParity(f"gsplat submodule not found: {GSPALT_SUBMODULE}")
    sys.path.insert(0, str(GSPALT_SUBMODULE))
    try:
        from gsplat.cuda import _wrapper
    except Exception as exc:  # pragma: no cover - depends on local env
        raise SkipParity(f"gsplat CUDA wrapper is not available: {exc}") from exc
    return _wrapper


def torch_to_mx(tensor: Any) -> mx.array:
    return mx.array(tensor.detach().cpu().numpy())


def mx_to_numpy(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.array(array)


def torch_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def compare_array(
    name: str,
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> bool:
    print(
        f"{name}: ref_shape={reference.shape}, actual_shape={actual.shape}, "
        f"ref_dtype={reference.dtype}, actual_dtype={actual.dtype}"
    )
    if reference.shape != actual.shape:
        print(f"{name}: FAIL shape mismatch")
        return False

    if reference.size == 0 and actual.size == 0:
        print(f"{name}: PASS empty arrays")
        return True

    diff = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    max_abs = float(diff.max()) if diff.size else 0.0
    ok = np.allclose(reference, actual, atol=atol, rtol=rtol)
    print(f"{name}: max_abs_diff={max_abs:.8g}, atol={atol}, rtol={rtol}, ok={ok}")
    return bool(ok)


def finish(results: list[bool]) -> None:
    if not all(results):
        raise SystemExit(1)
    print("parity ok")
