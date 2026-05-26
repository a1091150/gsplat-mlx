#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import (
    intersect_offset_forward,
    intersect_tile_forward,
    projection_ewa_3dgs_fused_forward,
    rasterize_to_pixels_3dgs_forward,
    spherical_harmonics_forward,
)


def main() -> None:
    image_width = 16
    image_height = 16
    tile_size = 8
    tile_width = 2
    tile_height = 2
    c0 = 0.2820947917738781

    means = mx.array([[[0.0, 0.0, 2.0]]], dtype=mx.float32)
    quats = mx.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=mx.float32)
    scales = mx.array([[[0.1, 0.1, 0.1]]], dtype=mx.float32)
    projection_opacities = mx.array([[0.8]], dtype=mx.float32)
    raster_opacities = mx.array([[[0.8]]], dtype=mx.float32)
    viewmats = mx.array(
        [[
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]],
        dtype=mx.float32,
    )
    Ks = mx.array(
        [[
            [
                [20.0, 0.0, 8.0],
                [0.0, 20.0, 8.0],
                [0.0, 0.0, 1.0],
            ]
        ]],
        dtype=mx.float32,
    )

    projection = projection_ewa_3dgs_fused_forward(
        {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": projection_opacities,
            "viewmats": viewmats,
            "Ks": Ks,
        },
        image_width=image_width,
        image_height=image_height,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=0.0,
        calc_compensations=False,
        camera_model=0,
    )

    intersections = intersect_tile_forward(
        {
            "means2d": projection["means2d"],
            "radii": projection["radii"],
            "depths": projection["depths"],
        },
        I=1,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
    )
    tile_offsets = intersect_offset_forward(
        intersections["isect_ids"],
        I=1,
        tile_width=tile_width,
        tile_height=tile_height,
    )

    dirs = mx.array([[[[0.0, 0.0, 1.0]]]], dtype=mx.float32)
    coeffs = mx.array([[[[[1.0 / c0, 0.0, 0.0]]]]], dtype=mx.float32)
    colors = spherical_harmonics_forward(0, {"dirs": dirs, "coeffs": coeffs})

    render = rasterize_to_pixels_3dgs_forward(
        {
            "means2d": projection["means2d"],
            "conics": projection["conics"],
            "colors": colors,
            "opacities": raster_opacities,
            "backgrounds": mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32),
            "tile_offsets": tile_offsets,
            "flatten_ids": intersections["flatten_ids"],
        },
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )

    mx.eval(*projection.values(), *intersections.values(), tile_offsets, colors, *render.values())
    for key, value in projection.items():
        print(f"projection.{key}: shape={value.shape}, dtype={value.dtype}")
    for key, value in intersections.items():
        print(f"intersections.{key}: shape={value.shape}, dtype={value.dtype}")
    print(f"tile_offsets: shape={tile_offsets.shape}, dtype={tile_offsets.dtype}")
    print(f"colors: shape={colors.shape}, dtype={colors.dtype}")
    for key, value in render.items():
        print(f"render.{key}: shape={value.shape}, dtype={value.dtype}")

    alpha_sum = mx.sum(render["render_alphas"])
    red_sum = mx.sum(render["render_colors"][..., 0])
    mx.eval(alpha_sum, red_sum)
    print(f"alpha_sum={alpha_sum.item():.6f}, red_sum={red_sum.item():.6f}")


if __name__ == "__main__":
    main()
