#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import spherical_harmonics_forward
from parity_utils import (
    SkipParity,
    compare_array,
    finish,
    load_gsplat_wrapper,
    load_torch_cuda,
    mx_to_numpy,
    torch_to_mx,
    torch_to_numpy,
)


def main() -> None:
    try:
        torch = load_torch_cuda()
        gsplat = load_gsplat_wrapper()
    except SkipParity as exc:
        print(f"SKIP: {exc}")
        return

    dirs_t = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    coeffs_t = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    masks_t = torch.tensor([True, False], device="cuda", dtype=torch.bool)

    ref = gsplat.spherical_harmonics(1, dirs_t, coeffs_t, masks_t)
    actual = spherical_harmonics_forward(
        1,
        {
            "dirs": torch_to_mx(dirs_t),
            "coeffs": torch_to_mx(coeffs_t),
            "masks": torch_to_mx(masks_t),
        },
    )
    mx.eval(actual)

    finish([compare_array("colors", torch_to_numpy(ref), mx_to_numpy(actual))])


if __name__ == "__main__":
    main()
