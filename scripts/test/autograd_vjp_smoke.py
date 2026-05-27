import numpy as np
import mlx.core as mx

from gsplat_core import (
    projection_ewa_3dgs_fused_forward,
    quat_scale_to_covar_preci_forward,
    rasterize_to_pixels_3dgs_forward,
    spherical_harmonics_forward,
)


def assert_shape(name, array, shape):
    if tuple(array.shape) != tuple(shape):
        raise AssertionError(f"{name}: expected shape {shape}, got {array.shape}")


def assert_nonzero(name, array):
    mx.eval(array)
    if not np.any(np.abs(np.array(array)) > 1.0e-8):
        raise AssertionError(f"{name}: expected a nonzero gradient")


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


def test_projection_vjp():
    means = mx.array(
        [[[0.1, -0.05, 2.0], [0.25, 0.15, 3.0]]],
        dtype=mx.float32,
    )
    covars = mx.array(
        [
            [
                [0.04, 0.002, 0.001, 0.05, -0.003, 0.06],
                [0.03, -0.001, 0.002, 0.045, 0.004, 0.055],
            ]
        ],
        dtype=mx.float32,
    )
    viewspace_points = mx.zeros((1, 1, 2, 2), dtype=mx.float32)
    viewmats = mx.array(
        [[[[1.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )
    Ks = mx.array(
        [[[[90.0, 0.0, 32.0],
           [0.0, 88.0, 24.0],
           [0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )

    def loss_fn(means_arg, covars_arg, viewspace_points_arg):
        outputs = projection_ewa_3dgs_fused_forward(
            {
                "means": means_arg,
                "covars": covars_arg,
                "viewmats": viewmats,
                "Ks": Ks,
                "viewspace_points": viewspace_points_arg,
            },
            image_width=64,
            image_height=48,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=0.0,
            calc_compensations=True,
            camera_model=0,
        )
        return (
            mx.sum(outputs["means2d"])
            + 0.25 * mx.sum(outputs["depths"])
            + 0.1 * mx.sum(outputs["conics"])
            + 0.5 * mx.sum(outputs["compensations"])
        )

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(
        means,
        covars,
        viewspace_points,
    )
    mx.eval(loss, *grads)
    assert_shape("projection v_means", grads[0], means.shape)
    assert_shape("projection v_covars", grads[1], covars.shape)
    assert_shape("projection v_viewspace_points", grads[2], viewspace_points.shape)
    assert_nonzero("projection v_means", grads[0])
    assert_nonzero("projection v_covars", grads[1])
    assert_nonzero("projection v_viewspace_points", grads[2])


def test_projection_quat_scale_vjp():
    means = mx.array(
        [[[0.1, -0.05, 2.0], [0.25, 0.15, 3.0]]],
        dtype=mx.float32,
    )
    quats = mx.array(
        [[[1.0, 0.1, 0.2, -0.1], [0.9, -0.2, 0.1, 0.3]]],
        dtype=mx.float32,
    )
    scales = mx.array(
        [[[0.22, 0.26, 0.3], [0.18, 0.24, 0.29]]],
        dtype=mx.float32,
    )
    viewspace_points = mx.zeros((1, 1, 2, 2), dtype=mx.float32)
    viewmats = mx.array(
        [[[[1.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )
    Ks = mx.array(
        [[[[90.0, 0.0, 32.0],
           [0.0, 88.0, 24.0],
           [0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )

    def loss_fn(means_arg, quats_arg, scales_arg, viewmats_arg, viewspace_points_arg):
        outputs = projection_ewa_3dgs_fused_forward(
            {
                "means": means_arg,
                "quats": quats_arg,
                "scales": scales_arg,
                "viewmats": viewmats_arg,
                "Ks": Ks,
                "viewspace_points": viewspace_points_arg,
            },
            image_width=64,
            image_height=48,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=0.0,
            calc_compensations=True,
            camera_model=0,
        )
        return (
            mx.sum(outputs["means2d"])
            + 0.25 * mx.sum(outputs["depths"])
            + 0.1 * mx.sum(outputs["conics"])
            + 0.5 * mx.sum(outputs["compensations"])
        )

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(
        means,
        quats,
        scales,
        viewmats,
        viewspace_points,
    )
    mx.eval(loss, *grads)
    assert_shape("projection quat path v_means", grads[0], means.shape)
    assert_shape("projection quat path v_quats", grads[1], quats.shape)
    assert_shape("projection quat path v_scales", grads[2], scales.shape)
    assert_shape("projection quat path v_viewmats", grads[3], viewmats.shape)
    assert_shape(
        "projection quat path v_viewspace_points",
        grads[4],
        viewspace_points.shape,
    )
    assert_nonzero("projection quat path v_means", grads[0])
    assert_nonzero("projection quat path v_quats", grads[1])
    assert_nonzero("projection quat path v_scales", grads[2])
    assert_nonzero("projection quat path v_viewmats", grads[3])
    assert_nonzero("projection quat path v_viewspace_points", grads[4])


def main():
    test_spherical_harmonics_vjp()
    test_quat_scale_vjp()
    test_rasterize_vjp()
    test_projection_vjp()
    test_projection_quat_scale_vjp()
    print("autograd vjp smoke ok")


if __name__ == "__main__":
    main()
