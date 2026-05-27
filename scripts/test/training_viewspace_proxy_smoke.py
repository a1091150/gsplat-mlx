import numpy as np
import mlx.core as mx

from gsplat_core import rasterize_to_pixels_3dgs_forward


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
    means2d = mx.array(
        [[[4.0, 4.0], [10.0, 9.0], [7.0, 12.0]]],
        dtype=mx.float32,
    )
    viewspace_points = mx.zeros_like(means2d)
    conics = mx.array(
        [[[0.32, 0.0, 0.32], [0.28, 0.02, 0.25], [0.2, -0.01, 0.24]]],
        dtype=mx.float32,
    )
    colors = mx.array(
        [[[0.9, 0.2, 0.1], [0.1, 0.8, 0.2], [0.2, 0.3, 0.9]]],
        dtype=mx.float32,
    )
    opacities = mx.array([[0.75, 0.55, 0.65]], dtype=mx.float32)
    target = mx.zeros((1, 16, 16, 3), dtype=mx.float32)
    tile_offsets = mx.array([[[0]]], dtype=mx.int32)
    flatten_ids = mx.array([0, 1, 2], dtype=mx.int32)

    def loss_fn(
        means2d_arg: mx.array,
        colors_arg: mx.array,
        opacities_arg: mx.array,
        viewspace_points_arg: mx.array,
    ) -> mx.array:
        screen_means = means2d_arg + viewspace_points_arg
        outputs = rasterize_to_pixels_3dgs_forward(
            {
                "means2d": screen_means,
                "conics": conics,
                "colors": colors_arg,
                "opacities": opacities_arg,
                "tile_offsets": tile_offsets,
                "flatten_ids": flatten_ids,
            },
            image_width=16,
            image_height=16,
            tile_size=16,
        )
        diff = outputs["render_colors"] - target
        return mx.sum(diff * diff) + 0.1 * mx.sum(outputs["render_alphas"])

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(
        means2d,
        colors,
        opacities,
        viewspace_points,
    )
    mx.eval(loss, *grads)

    v_means2d, v_colors, v_opacities, v_viewspace_points = grads
    assert_shape("v_means2d", v_means2d, tuple(means2d.shape))
    assert_shape("v_colors", v_colors, tuple(colors.shape))
    assert_shape("v_opacities", v_opacities, tuple(opacities.shape))
    assert_shape("v_viewspace_points", v_viewspace_points, tuple(viewspace_points.shape))
    assert_nonzero("v_viewspace_points", v_viewspace_points)

    proxy_delta = np.max(np.abs(to_numpy(v_viewspace_points) - to_numpy(v_means2d)))
    if proxy_delta > 1.0e-6:
        raise AssertionError(
            "viewspace_points gradient should match means2d gradient when "
            f"screen_means = means2d + viewspace_points; max diff={proxy_delta}"
        )

    step = 1.0e-2
    next_loss = loss_fn(
        means2d - step * v_means2d,
        colors - step * v_colors,
        opacities - step * v_opacities,
        viewspace_points - step * v_viewspace_points,
    )
    mx.eval(next_loss)
    if not np.isfinite(float(to_numpy(next_loss))):
        raise AssertionError("updated training smoke loss should be finite")

    print(
        "training viewspace proxy smoke ok "
        f"loss={float(to_numpy(loss)):.6f} "
        f"next_loss={float(to_numpy(next_loss)):.6f}"
    )


if __name__ == "__main__":
    main()
