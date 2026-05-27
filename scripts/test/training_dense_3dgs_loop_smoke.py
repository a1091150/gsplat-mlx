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


def clamp(array: mx.array, low: float, high: float) -> mx.array:
    return mx.minimum(mx.maximum(array, low), high)


def normalize_quats(quats: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True))
    return quats / mx.maximum(norm, 1.0e-8)


def make_camera() -> tuple[mx.array, mx.array]:
    viewmats = mx.array(
        [[[[1.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )
    Ks = mx.array(
        [[[[18.0, 0.0, 8.0],
           [0.0, 18.0, 8.0],
           [0.0, 0.0, 1.0]]]],
        dtype=mx.float32,
    )
    return viewmats, Ks


def render_scene(
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    colors: mx.array,
    opacities: mx.array,
    viewspace_points: mx.array,
    viewmats: mx.array,
    Ks: mx.array,
) -> dict[str, mx.array]:
    projection = projection_ewa_3dgs_fused_forward(
        {
            "means": means,
            "quats": quats,
            "scales": scales,
            "viewmats": viewmats,
            "Ks": Ks,
            "viewspace_points": viewspace_points,
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
    return rasterize_to_pixels_3dgs_forward(
        {
            "means2d": projection["means2d"],
            "conics": projection["conics"],
            "colors": colors,
            "opacities": opacities,
            "tile_offsets": mx.array([[[0]]], dtype=mx.int32),
            "flatten_ids": mx.array([0, 1, 2, 3], dtype=mx.int32),
        },
        image_width=16,
        image_height=16,
        tile_size=16,
    )


def main() -> None:
    viewmats, Ks = make_camera()
    viewspace_shape = (1, 1, 4, 2)

    target_means = mx.array(
        [[[-0.18, -0.10, 2.0],
          [0.05, 0.12, 2.3],
          [0.22, -0.04, 2.15],
          [-0.02, 0.02, 2.6]]],
        dtype=mx.float32,
    )
    target_quats = normalize_quats(mx.array(
        [[[1.0, 0.05, 0.0, 0.0],
          [0.98, 0.0, 0.12, 0.05],
          [0.96, -0.08, 0.04, 0.1],
          [1.0, 0.02, -0.04, 0.08]]],
        dtype=mx.float32,
    ))
    target_scales = mx.array(
        [[[0.18, 0.12, 0.15],
          [0.14, 0.18, 0.13],
          [0.16, 0.12, 0.20],
          [0.12, 0.15, 0.14]]],
        dtype=mx.float32,
    )
    target_colors = mx.array(
        [[[[0.95, 0.18, 0.10],
           [0.12, 0.78, 0.24],
           [0.18, 0.28, 0.92],
           [0.85, 0.82, 0.20]]]],
        dtype=mx.float32,
    )
    target_opacities = mx.array([[[0.78, 0.62, 0.70, 0.55]]], dtype=mx.float32)
    target_render = render_scene(
        target_means,
        target_quats,
        target_scales,
        target_colors,
        target_opacities,
        mx.zeros(viewspace_shape, dtype=mx.float32),
        viewmats,
        Ks,
    )
    target_colors_img = target_render["render_colors"]
    target_alphas_img = target_render["render_alphas"]
    mx.eval(target_colors_img, target_alphas_img)

    means = target_means + mx.array(
        [[[0.04, -0.03, 0.03],
          [-0.03, 0.02, -0.04],
          [0.03, 0.04, 0.02],
          [0.02, -0.03, -0.02]]],
        dtype=mx.float32,
    )
    quats = normalize_quats(target_quats + mx.array(
        [[[0.0, 0.02, -0.01, 0.01],
          [0.01, -0.02, 0.01, 0.0],
          [0.0, 0.01, 0.02, -0.01],
          [0.02, 0.0, -0.01, 0.01]]],
        dtype=mx.float32,
    ))
    scales = target_scales * 1.12
    colors = clamp(target_colors + 0.08, 0.0, 1.0)
    opacities = clamp(target_opacities - 0.08, 0.05, 0.95)

    def loss_fn(
        means_arg: mx.array,
        quats_arg: mx.array,
        scales_arg: mx.array,
        colors_arg: mx.array,
        opacities_arg: mx.array,
        viewspace_points_arg: mx.array,
    ) -> mx.array:
        render = render_scene(
            means_arg,
            normalize_quats(quats_arg),
            mx.maximum(scales_arg, 0.03),
            clamp(colors_arg, 0.0, 1.0),
            clamp(opacities_arg, 0.01, 0.99),
            viewspace_points_arg,
            viewmats,
            Ks,
        )
        color_diff = render["render_colors"] - target_colors_img
        alpha_diff = render["render_alphas"] - target_alphas_img
        return mx.mean(color_diff * color_diff) + 0.25 * mx.mean(alpha_diff * alpha_diff)

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))
    initial_loss = None
    first_grads = None

    for _ in range(8):
        viewspace_points = mx.zeros(viewspace_shape, dtype=mx.float32)
        loss, grads = grad_fn(
            means,
            quats,
            scales,
            colors,
            opacities,
            viewspace_points,
        )
        mx.eval(loss, *grads)
        if initial_loss is None:
            initial_loss = float(to_numpy(loss))
            first_grads = grads
        means = means - 2.0e-2 * grads[0]
        quats = normalize_quats(quats - 5.0e-3 * grads[1])
        scales = mx.maximum(scales - 2.0e-3 * grads[2], 0.03)
        colors = clamp(colors - 5.0e-2 * grads[3], 0.0, 1.0)
        opacities = clamp(opacities - 1.0e-2 * grads[4], 0.01, 0.99)
        mx.eval(means, quats, scales, colors, opacities)

    final_loss = float(to_numpy(loss_fn(
        means,
        quats,
        scales,
        colors,
        opacities,
        mx.zeros(viewspace_shape, dtype=mx.float32),
    )))
    if initial_loss is None or first_grads is None:
        raise AssertionError("training loop did not run")
    if not np.isfinite(initial_loss) or not np.isfinite(final_loss):
        raise AssertionError("dense training loop loss should be finite")
    if final_loss > initial_loss * 1.05:
        raise AssertionError(
            f"dense training loop loss should not increase badly: "
            f"initial={initial_loss:.8f} final={final_loss:.8f}"
        )

    names = (
        "v_means",
        "v_quats",
        "v_scales",
        "v_colors",
        "v_opacities",
        "v_viewspace_points",
    )
    shapes = (
        tuple(means.shape),
        tuple(quats.shape),
        tuple(scales.shape),
        tuple(colors.shape),
        tuple(opacities.shape),
        viewspace_shape,
    )
    for name, grad, shape in zip(names, first_grads, shapes, strict=True):
        assert_shape(name, grad, shape)
        assert_nonzero(name, grad)

    print(
        "dense 3dgs training loop smoke ok "
        f"initial_loss={initial_loss:.8f} final_loss={final_loss:.8f}"
    )


if __name__ == "__main__":
    main()
