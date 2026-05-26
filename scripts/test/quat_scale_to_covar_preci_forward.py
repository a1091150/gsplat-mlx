#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import quat_scale_to_covar_preci_forward


def main() -> None:
    quats = mx.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ],
        dtype=mx.float32,
    )
    scales = mx.array(
        [
            [2.0, 3.0, 4.0],
            [0.5, 2.0, 4.0],
        ],
        dtype=mx.float32,
    )

    outputs = quat_scale_to_covar_preci_forward(
        {
            "quats": quats,
            "scales": scales,
        },
        compute_covar=True,
        compute_preci=True,
        triu=True,
    )

    mx.eval(*outputs.values())
    for key, value in outputs.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}, values={value}")


if __name__ == "__main__":
    main()
