import numpy as np
import mlx.core as mx

from gsplat_core import (
    projection_ewa_3dgs_fused_forward,
    rasterize_to_pixels_3dgs_forward,
)


def to_numpy(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.array(array)


def assert_shape(name: str, array: mx.array, shape: tuple[int, ...]) -> None:
    if tuple(array.shape) != shape:
        raise AssertionError(f"{name}: expected shape {shape}, got {array.shape}")


def assert_nonzero(name: str, array: mx.array) -> None:
    values = to_numpy(array)
    if not np.any(np.abs(values) > 1.0e-8):
        raise AssertionError(f"{name}: expected a nonzero gradient")


def main() -> None:
    means3d = mx.array(
        [[[0.0, 0.0, 2.0], [0.18, 0.1, 2.4], [-0.16, 0.12, 2.2]]],
        dtype=mx.float32,
    )
    covars = mx.array(
        [
            [
                [0.025, 0.0, 0.0, 0.025, 0.0, 0.03],
                [0.02, 0.001, 0.0, 0.024, 0.0, 0.028],
                [0.018, -0.001, 0.0, 0.022, 0.0, 0.026],
            ]
        ],
        dtype=mx.float32,
    )
    viewspace_points = mx.zeros((1, 1, 3, 2), dtype=mx.float32)
    colors = mx.array(
        [[[[0.9, 0.2, 0.1], [0.1, 0.8, 0.2], [0.2, 0.3, 0.9]]]],
        dtype=mx.float32,
    )
    opacities = mx.array([[[0.75, 0.55, 0.65]]], dtype=mx.float32)
    viewmats = mx.array(
        [[[[1.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )
    Ks = mx.array(
        [[[[16.0, 0.0, 8.0],
           [0.0, 16.0, 8.0],
           [0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )
    tile_offsets = mx.array([[[0]]], dtype=mx.int32)
    flatten_ids = mx.array([0, 1, 2], dtype=mx.int32)
    target = mx.zeros((1, 16, 16, 3), dtype=mx.float32)

    def loss_fn(
        means3d_arg: mx.array,
        covars_arg: mx.array,
        colors_arg: mx.array,
        opacities_arg: mx.array,
        viewspace_points_arg: mx.array,
    ) -> mx.array:
        projection = projection_ewa_3dgs_fused_forward(
            {
                "means": means3d_arg,
                "covars": covars_arg,
                "viewmats": viewmats,
                "Ks": Ks,
                "viewspace_points": viewspace_points_arg,
            },
            image_width=16,
            image_height=16,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=0.0,
            calc_compensations=False,
            camera_model=0,
        )
        render = rasterize_to_pixels_3dgs_forward(
            {
                "means2d": projection["means2d"],
                "conics": projection["conics"],
                "colors": colors_arg,
                "opacities": opacities_arg,
                "tile_offsets": tile_offsets,
                "flatten_ids": flatten_ids,
            },
            image_width=16,
            image_height=16,
            tile_size=16,
        )
        diff = render["render_colors"] - target
        return (
            mx.sum(diff * diff)
            + 0.1 * mx.sum(render["render_alphas"])
        )

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(
        means3d,
        covars,
        colors,
        opacities,
        viewspace_points,
    )
    mx.eval(loss, *grads)

    v_means3d, v_covars, v_colors, v_opacities, v_viewspace_points = grads
    assert_shape("v_means3d", v_means3d, tuple(means3d.shape))
    assert_shape("v_covars", v_covars, tuple(covars.shape))
    assert_shape("v_colors", v_colors, tuple(colors.shape))
    assert_shape("v_opacities", v_opacities, tuple(opacities.shape))
    assert_shape("v_viewspace_points", v_viewspace_points, tuple(viewspace_points.shape))
    assert_nonzero("v_means3d", v_means3d)
    assert_nonzero("v_covars", v_covars)
    assert_nonzero("v_viewspace_points", v_viewspace_points)

    step = 1.0e-3
    next_loss = loss_fn(
        means3d - step * v_means3d,
        covars - step * v_covars,
        colors - step * v_colors,
        opacities - step * v_opacities,
        viewspace_points - step * v_viewspace_points,
    )
    mx.eval(next_loss)
    if not np.isfinite(float(to_numpy(next_loss))):
        raise AssertionError("updated projection training smoke loss should be finite")

    print(
        "training projection viewspace proxy smoke ok "
        f"loss={float(to_numpy(loss)):.6f} "
        f"next_loss={float(to_numpy(next_loss)):.6f}"
    )


if __name__ == "__main__":
    main()
