#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import rasterize_to_pixels_3dgs_forward


def main() -> None:
    means2d = mx.array([[[1.0, 1.0]]], dtype=mx.float32)
    conics = mx.array([[[1.0, 0.0, 1.0]]], dtype=mx.float32)
    colors = mx.array([[[1.0, 0.0, 0.0]]], dtype=mx.float32)
    opacities = mx.array([[0.5]], dtype=mx.float32)
    backgrounds = mx.array([[0.1, 0.2, 0.3]], dtype=mx.float32)
    tile_offsets = mx.array([[[0]]], dtype=mx.int32)
    flatten_ids = mx.array([0], dtype=mx.int32)

    outputs = rasterize_to_pixels_3dgs_forward(
        {
            "means2d": means2d,
            "conics": conics,
            "colors": colors,
            "opacities": opacities,
            "backgrounds": backgrounds,
            "tile_offsets": tile_offsets,
            "flatten_ids": flatten_ids,
        },
        image_width=2,
        image_height=2,
        tile_size=2,
    )

    mx.eval(*outputs.values())
    for key, value in outputs.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}, values={value}")


if __name__ == "__main__":
    main()
