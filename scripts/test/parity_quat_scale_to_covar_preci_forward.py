#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import quat_scale_to_covar_preci_forward
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

    quats_t = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    scales_t = torch.tensor(
        [[2.0, 3.0, 4.0], [0.5, 2.0, 4.0]],
        device="cuda",
        dtype=torch.float32,
    )

    covars_t, precis_t = gsplat.quat_scale_to_covar_preci(
        quats_t, scales_t, compute_covar=True, compute_preci=True, triu=True
    )
    outputs = quat_scale_to_covar_preci_forward(
        {"quats": torch_to_mx(quats_t), "scales": torch_to_mx(scales_t)},
        compute_covar=True,
        compute_preci=True,
        triu=True,
    )
    mx.eval(*outputs.values())

    finish(
        [
            compare_array("covars", torch_to_numpy(covars_t), mx_to_numpy(outputs["covars"])),
            compare_array("precis", torch_to_numpy(precis_t), mx_to_numpy(outputs["precis"])),
        ]
    )


if __name__ == "__main__":
    main()
