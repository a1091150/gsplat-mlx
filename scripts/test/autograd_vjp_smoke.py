import mlx.core as mx

from gsplat_core import (
    quat_scale_to_covar_preci_forward,
    rasterize_to_pixels_3dgs_forward,
    spherical_harmonics_forward,
)


def assert_shape(name, array, shape):
    if tuple(array.shape) != tuple(shape):
        raise AssertionError(f"{name}: expected shape {shape}, got {array.shape}")


def test_spherical_harmonics_vjp():
    dirs = mx.array(
        [[0.25, -0.5, 1.0], [-0.75, 0.1, 0.6]],
        dtype=mx.float32,
    )
    coeffs = mx.ones((2, 4, 3), dtype=mx.float32) * 0.1

    def loss_fn(dirs_arg, coeffs_arg):
        colors = spherical_harmonics_forward(
            1,
            {
                "dirs": dirs_arg,
                "coeffs": coeffs_arg,
            },
        )
        return mx.sum(colors)

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(dirs, coeffs)
    mx.eval(loss, *grads)
    assert_shape("sh v_dirs", grads[0], dirs.shape)
    assert_shape("sh v_coeffs", grads[1], coeffs.shape)


def test_quat_scale_vjp():
    quats = mx.array(
        [[1.0, 0.1, 0.2, -0.1], [0.9, -0.2, 0.1, 0.3]],
        dtype=mx.float32,
    )
    scales = mx.array(
        [[0.4, 0.5, 0.6], [0.3, 0.45, 0.55]],
        dtype=mx.float32,
    )

    def loss_fn(quats_arg, scales_arg):
        outputs = quat_scale_to_covar_preci_forward(
            {
                "quats": quats_arg,
                "scales": scales_arg,
            },
            compute_covar=True,
            compute_preci=True,
            triu=True,
        )
        return mx.sum(outputs["covars"]) + mx.sum(outputs["precis"])

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(quats, scales)
    mx.eval(loss, *grads)
    assert_shape("quat_scale v_quats", grads[0], quats.shape)
    assert_shape("quat_scale v_scales", grads[1], scales.shape)


def test_rasterize_vjp():
    means2d = mx.array([[[4.0, 4.0], [9.0, 9.0]]], dtype=mx.float32)
    conics = mx.array([[[0.3, 0.0, 0.3], [0.25, 0.0, 0.25]]], dtype=mx.float32)
    colors = mx.array([[[0.8, 0.1, 0.2], [0.1, 0.7, 0.3]]], dtype=mx.float32)
    opacities = mx.array([[0.7, 0.6]], dtype=mx.float32)
    tile_offsets = mx.array([[[0]]], dtype=mx.int32)
    flatten_ids = mx.array([0, 1], dtype=mx.int32)

    def loss_fn(means2d_arg, conics_arg, colors_arg, opacities_arg):
        outputs = rasterize_to_pixels_3dgs_forward(
            {
                "means2d": means2d_arg,
                "conics": conics_arg,
                "colors": colors_arg,
                "opacities": opacities_arg,
                "tile_offsets": tile_offsets,
                "flatten_ids": flatten_ids,
            },
            image_width=16,
            image_height=16,
            tile_size=16,
        )
        return mx.sum(outputs["render_colors"]) + mx.sum(outputs["render_alphas"])

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(
        means2d,
        conics,
        colors,
        opacities,
    )
    mx.eval(loss, *grads)
    assert_shape("raster v_means2d", grads[0], means2d.shape)
    assert_shape("raster v_conics", grads[1], conics.shape)
    assert_shape("raster v_colors", grads[2], colors.shape)
    assert_shape("raster v_opacities", grads[3], opacities.shape)


def main():
    test_spherical_harmonics_vjp()
    test_quat_scale_vjp()
    test_rasterize_vjp()
    print("autograd vjp smoke ok")


if __name__ == "__main__":
    main()
