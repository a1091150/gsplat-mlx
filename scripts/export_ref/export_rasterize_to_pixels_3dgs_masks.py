#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("rasterize_to_pixels_3dgs_masks.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    means2d = torch.tensor([[[1.0, 1.0]]], device="cuda", dtype=torch.float32)
    conics = torch.tensor([[[1.0, 0.0, 1.0]]], device="cuda", dtype=torch.float32)
    colors = torch.tensor([[[1.0, 0.0, 0.0]]], device="cuda", dtype=torch.float32)
    opacities = torch.tensor([[0.5]], device="cuda", dtype=torch.float32)
    backgrounds = torch.tensor([[0.1, 0.2, 0.3]], device="cuda", dtype=torch.float32)
    masks = torch.tensor([[[True, False]]], device="cuda", dtype=torch.bool)
    tile_offsets = torch.tensor([[[0, 1]]], device="cuda", dtype=torch.int32)
    flatten_ids = torch.tensor([0], device="cuda", dtype=torch.int32)

    render_colors, render_alphas = gsplat.rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        image_width=4,
        image_height=2,
        tile_size=2,
        isect_offsets=tile_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
        packed=False,
        absgrad=False,
    )

    save_npz(
        args.out,
        input__image_width=4,
        input__image_height=2,
        input__tile_size=2,
        input__means2d=means2d,
        input__conics=conics,
        input__colors=colors,
        input__opacities=opacities,
        input__backgrounds=backgrounds,
        input__masks=masks,
        input__tile_offsets=tile_offsets,
        input__flatten_ids=flatten_ids,
        ref__render_colors=render_colors,
        ref__render_alphas=render_alphas,
    )


if __name__ == "__main__":
    main()
