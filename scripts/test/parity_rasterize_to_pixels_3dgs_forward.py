#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import rasterize_to_pixels_3dgs_forward
from parity_utils import (
    SkipParity,
    compare_array,
    finish,
    load_gsplat_wrapper,
    load_torch_cuda,
    mx_to_numpy,
    torch_to_mx,
    torch_to_numpy,
)


def main() -> None:
    try:
        torch = load_torch_cuda()
        gsplat = load_gsplat_wrapper()
    except SkipParity as exc:
        print(f"SKIP: {exc}")
        return

    means2d_t = torch.tensor([[[1.0, 1.0]]], device="cuda", dtype=torch.float32)
    conics_t = torch.tensor([[[1.0, 0.0, 1.0]]], device="cuda", dtype=torch.float32)
    colors_t = torch.tensor([[[1.0, 0.0, 0.0]]], device="cuda", dtype=torch.float32)
    opacities_t = torch.tensor([[0.5]], device="cuda", dtype=torch.float32)
    backgrounds_t = torch.tensor([[0.1, 0.2, 0.3]], device="cuda", dtype=torch.float32)
    offsets_t = torch.tensor([[[0]]], device="cuda", dtype=torch.int32)
    flatten_t = torch.tensor([0], device="cuda", dtype=torch.int32)

    colors_ref, alphas_ref = gsplat.rasterize_to_pixels(
        means2d_t,
        conics_t,
        colors_t,
        opacities_t,
        image_width=2,
        image_height=2,
        tile_size=2,
        isect_offsets=offsets_t,
        flatten_ids=flatten_t,
        backgrounds=backgrounds_t,
        masks=None,
        packed=False,
        absgrad=False,
    )

    actual = rasterize_to_pixels_3dgs_forward(
        {
            "means2d": torch_to_mx(means2d_t),
            "conics": torch_to_mx(conics_t),
            "colors": torch_to_mx(colors_t),
            "opacities": torch_to_mx(opacities_t),
            "backgrounds": torch_to_mx(backgrounds_t),
            "tile_offsets": torch_to_mx(offsets_t),
            "flatten_ids": torch_to_mx(flatten_t),
        },
        image_width=2,
        image_height=2,
        tile_size=2,
    )
    mx.eval(*actual.values())

    finish(
        [
            compare_array("render_colors", torch_to_numpy(colors_ref), mx_to_numpy(actual["render_colors"])),
            compare_array("render_alphas", torch_to_numpy(alphas_ref), mx_to_numpy(actual["render_alphas"])),
        ]
    )


if __name__ == "__main__":
    main()
