#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import intersect_offset_forward, intersect_tile_forward


def main() -> None:
    means2d = mx.array(
        [[[20.0, 20.0], [50.0, 50.0], [8.0, 8.0]]],
        dtype=mx.float32,
    )
    radii = mx.array([[[10, 10], [5, 5], [0, 0]]], dtype=mx.int32)
    depths = mx.array([[1.0, 0.5, 2.0]], dtype=mx.float32)

    outputs = intersect_tile_forward(
        {
            "means2d": means2d,
            "radii": radii,
            "depths": depths,
        },
        I=1,
        tile_size=16,
        tile_width=4,
        tile_height=4,
        sort=True,
        segmented=False,
    )
    offsets = intersect_offset_forward(
        outputs["isect_ids"],
        I=1,
        tile_width=4,
        tile_height=4,
    )

    mx.eval(*outputs.values(), offsets)
    for key, value in outputs.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}, values={value}")
    print(f"offsets: shape={offsets.shape}, dtype={offsets.dtype}, values={offsets}")


if __name__ == "__main__":
    main()
