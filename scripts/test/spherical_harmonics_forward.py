#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import spherical_harmonics_forward


def main() -> None:
    dirs = mx.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=mx.float32,
    )
    coeffs = mx.array(
        [
            [
                [1.0, 2.0, 3.0],
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            [
                [4.0, 5.0, 6.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ],
        dtype=mx.float32,
    )
    masks = mx.array([True, False])

    colors = spherical_harmonics_forward(
        1,
        {
            "dirs": dirs,
            "coeffs": coeffs,
            "masks": masks,
        },
    )

    mx.eval(colors)
    print(f"colors: shape={colors.shape}, dtype={colors.dtype}, values={colors}")


if __name__ == "__main__":
    main()
