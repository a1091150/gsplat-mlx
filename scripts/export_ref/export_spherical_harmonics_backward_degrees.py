#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from export_utils import load_gsplat_wrapper, load_torch_cuda, save_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("refs"),
        help="Directory for output .npz fixtures.",
    )
    return parser.parse_args()


def call_spherical_harmonics_bwd(
    gsplat, degrees_to_use, dirs, coeffs, masks, v_colors, compute_v_dirs
):
    spherical_harmonics_bwd = gsplat._make_lazy_cuda_func("spherical_harmonics_bwd")
    try:
        return spherical_harmonics_bwd(
            degrees_to_use,
            dirs,
            coeffs,
            masks,
            v_colors.contiguous(),
            compute_v_dirs,
        )
    except TypeError:
        return spherical_harmonics_bwd(
            degrees_to_use,
            coeffs.shape[-2],
            dirs,
            coeffs,
            masks,
            v_colors.contiguous(),
            compute_v_dirs,
        )


def main() -> None:
    args = parse_args()
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    dirs = torch.tensor(
        [
            [0.25, -0.5, 1.0],
            [-0.75, 0.1, 0.6],
            [0.2, 0.3, -0.4],
            [0.9, -0.2, 0.35],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    coeffs = (
        torch.arange(1, 4 * 25 * 3 + 1, device="cuda", dtype=torch.float32)
        .reshape(4, 25, 3)
        .mul_(0.01)
    )
    masks = torch.tensor([True, False, True, True], device="cuda", dtype=torch.bool)
    v_colors = torch.tensor(
        [
            [0.3, -0.2, 0.7],
            [1.0, 2.0, 3.0],
            [-0.4, 0.5, 0.25],
            [0.8, -0.6, 0.1],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    compute_v_dirs = True

    for degrees_to_use in range(5):
        v_coeffs, v_dirs = call_spherical_harmonics_bwd(
            gsplat,
            degrees_to_use,
            dirs,
            coeffs,
            masks,
            v_colors,
            compute_v_dirs,
        )
        save_npz(
            args.out_dir
            / f"spherical_harmonics_backward_degree{degrees_to_use}_masks.npz",
            input__degrees_to_use=degrees_to_use,
            input__dirs=dirs,
            input__coeffs=coeffs,
            input__masks=masks,
            cotangent__v_colors=v_colors,
            meta__compute_v_dirs=compute_v_dirs,
            ref__v_dirs=v_dirs,
            ref__v_coeffs=v_coeffs,
        )


if __name__ == "__main__":
    main()
