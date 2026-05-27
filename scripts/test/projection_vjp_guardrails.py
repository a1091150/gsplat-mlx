import autograd_vjp_smoke as smoke

mx = smoke.mx
np = smoke.np
projection_ewa_3dgs_fused_forward = smoke.projection_ewa_3dgs_fused_forward


EXPECTED_FULL_GPU_SUPPORT = (
    "dense covars input",
    "quat/scale input",
    "pinhole camera_model=0",
    "v_means, v_covars, v_viewmats, and viewspace_points gradients",
    "v_quats and v_scales gradients",
)

EXPECTED_LIMITATIONS = (
    "non-pinhole cameras are not supported by projection backward",
    "packed projection is not implemented in gsplat_core yet",
    "Ks and opacity gradients are intentionally out of scope",
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


def base_inputs() -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
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
    return means, covars, viewspace_points, viewmats, Ks


def projection_loss(
    means: mx.array,
    covars: mx.array,
    viewspace_points: mx.array,
    viewmats: mx.array,
    Ks: mx.array,
) -> mx.array:
    outputs = projection_ewa_3dgs_fused_forward(
        {
            "means": means,
            "covars": covars,
            "viewmats": viewmats,
            "Ks": Ks,
            "viewspace_points": viewspace_points,
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


def check_supported_full_gpu_vjp() -> None:
    means, covars, viewspace_points, viewmats, Ks = base_inputs()

    def loss_fn(
        means_arg: mx.array,
        covars_arg: mx.array,
        viewspace_points_arg: mx.array,
    ) -> mx.array:
        return projection_loss(
            means_arg,
            covars_arg,
            viewspace_points_arg,
            viewmats,
            Ks,
        )

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(
        means,
        covars,
        viewspace_points,
    )
    mx.eval(loss, *grads)
    assert_shape("supported v_means", grads[0], tuple(means.shape))
    assert_shape("supported v_covars", grads[1], tuple(covars.shape))
    assert_shape(
        "supported v_viewspace_points",
        grads[2],
        tuple(viewspace_points.shape),
    )
    assert_nonzero("supported v_means", grads[0])
    assert_nonzero("supported v_covars", grads[1])
    assert_nonzero("supported v_viewspace_points", grads[2])


def check_supported_full_gpu_viewmats_vjp() -> None:
    means, covars, viewspace_points, viewmats, Ks = base_inputs()

    def loss_fn(
        means_arg: mx.array,
        covars_arg: mx.array,
        viewmats_arg: mx.array,
        viewspace_points_arg: mx.array,
    ) -> mx.array:
        return projection_loss(
            means_arg,
            covars_arg,
            viewspace_points_arg,
            viewmats_arg,
            Ks,
        )

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(
        means,
        covars,
        viewmats,
        viewspace_points,
    )
    mx.eval(loss, *grads)
    assert_shape("supported v_means with v_viewmats", grads[0], tuple(means.shape))
    assert_shape("supported v_covars with v_viewmats", grads[1], tuple(covars.shape))
    assert_shape("supported v_viewmats", grads[2], tuple(viewmats.shape))
    assert_shape(
        "supported v_viewspace_points with v_viewmats",
        grads[3],
        tuple(viewspace_points.shape),
    )
    assert_nonzero("supported v_viewmats", grads[2])


def main() -> None:
    check_supported_full_gpu_vjp()
    check_supported_full_gpu_viewmats_vjp()
    smoke.test_projection_quat_scale_vjp()

    print("projection vjp guardrails ok")
    print("full GPU support:")
    for item in EXPECTED_FULL_GPU_SUPPORT:
        print(f"  - {item}")
    print("expected limitations:")
    for item in EXPECTED_LIMITATIONS:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
