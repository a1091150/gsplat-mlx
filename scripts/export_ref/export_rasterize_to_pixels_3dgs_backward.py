#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("rasterize_to_pixels_3dgs_backward.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    image_width = 2
    image_height = 2
    tile_size = 2
    absgrad = True
    means2d = torch.tensor(
        [[[0.75, 0.75], [1.35, 1.15]]],
        device="cuda",
        dtype=torch.float32,
    )
    conics = torch.tensor(
        [[[0.35, 0.02, 0.45], [0.25, -0.03, 0.3]]],
        device="cuda",
        dtype=torch.float32,
    )
    colors = torch.tensor(
        [[[0.8, 0.2, 0.1], [0.1, 0.7, 0.4]]],
        device="cuda",
        dtype=torch.float32,
    )
    opacities = torch.tensor([[0.6, 0.45]], device="cuda", dtype=torch.float32)
    backgrounds = torch.tensor([[0.05, 0.1, 0.2]], device="cuda", dtype=torch.float32)
    masks = None
    tile_offsets = torch.tensor([[[0]]], device="cuda", dtype=torch.int32)
    flatten_ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    render_colors, render_alphas, last_ids = gsplat._make_lazy_cuda_func(
        "rasterize_to_pixels_3dgs_fwd"
    )(
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
    )
    v_render_colors = torch.tensor(
        [
            [
                [[0.2, -0.1, 0.3], [0.4, 0.05, -0.2]],
                [[-0.3, 0.25, 0.15], [0.1, -0.4, 0.35]],
            ]
        ],
        device="cuda",
        dtype=torch.float32,
    )
    v_render_alphas = torch.tensor(
        [[[[0.2], [-0.1]], [[0.05], [0.3]]]],
        device="cuda",
        dtype=torch.float32,
    )

    (
        v_means2d_abs,
        v_means2d,
        v_conics,
        v_colors,
        v_opacities,
    ) = gsplat._make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
        render_alphas,
        last_ids,
        v_render_colors.contiguous(),
        v_render_alphas.contiguous(),
        absgrad,
    )
    v_backgrounds = (v_render_colors * (1.0 - render_alphas.float())).sum(dim=(-3, -2))

    save_npz(
        args.out,
        input__means2d=means2d,
        input__conics=conics,
        input__colors=colors,
        input__opacities=opacities,
        input__backgrounds=backgrounds,
        input__image_width=image_width,
        input__image_height=image_height,
        input__tile_size=tile_size,
        input__tile_offsets=tile_offsets,
        input__flatten_ids=flatten_ids,
        fwd__render_colors=render_colors,
        fwd__render_alphas=render_alphas,
        fwd__last_ids=last_ids,
        cotangent__v_render_colors=v_render_colors,
        cotangent__v_render_alphas=v_render_alphas,
        meta__absgrad=absgrad,
        ref__v_means2d_abs=v_means2d_abs,
        ref__v_means2d=v_means2d,
        ref__v_conics=v_conics,
        ref__v_colors=v_colors,
        ref__v_opacities=v_opacities,
        ref__v_backgrounds=v_backgrounds,
    )


if __name__ == "__main__":
    main()
