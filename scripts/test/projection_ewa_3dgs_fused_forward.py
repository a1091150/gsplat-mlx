#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import projection_ewa_3dgs_fused_forward


def main() -> None:
    means = mx.array([[[0.0, 0.0, 2.0], [0.25, 0.0, 2.5]]], dtype=mx.float32)
    quats = mx.array([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=mx.float32)
    scales = mx.array([[[0.05, 0.05, 0.05], [0.08, 0.06, 0.05]]], dtype=mx.float32)
    opacities = mx.array([[0.8, 0.6]], dtype=mx.float32)
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
                [120.0, 0.0, 32.0],
                [0.0, 120.0, 32.0],
                [0.0, 0.0, 1.0],
            ]
        ]],
        dtype=mx.float32,
    )

    outputs = projection_ewa_3dgs_fused_forward(
        {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "viewmats": viewmats,
            "Ks": Ks,
        },
        image_width=64,
        image_height=64,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=0.0,
        calc_compensations=True,
        camera_model=0,
    )
    mx.eval(*outputs.values())
    for key, value in outputs.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
