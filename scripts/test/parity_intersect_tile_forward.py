#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import intersect_offset_forward, intersect_tile_forward
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

    means2d_t = torch.tensor(
        [[[20.0, 20.0], [50.0, 50.0], [8.0, 8.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    radii_t = torch.tensor([[[10, 10], [5, 5], [0, 0]]], device="cuda", dtype=torch.int32)
    depths_t = torch.tensor([[1.0, 0.5, 2.0]], device="cuda", dtype=torch.float32)

    tiles_ref, isect_ref, flatten_ref = gsplat.isect_tiles(
        means2d_t,
        radii_t,
        depths_t,
        tile_size=16,
        tile_width=4,
        tile_height=4,
        sort=True,
        segmented=False,
        packed=False,
    )
    offsets_ref = gsplat.isect_offset_encode(isect_ref, 1, 4, 4)

    actual = intersect_tile_forward(
        {
            "means2d": torch_to_mx(means2d_t),
            "radii": torch_to_mx(radii_t),
            "depths": torch_to_mx(depths_t),
        },
        I=1,
        tile_size=16,
        tile_width=4,
        tile_height=4,
        sort=True,
        segmented=False,
    )
    offsets = intersect_offset_forward(actual["isect_ids"], I=1, tile_width=4, tile_height=4)
    mx.eval(*actual.values(), offsets)

    finish(
        [
            compare_array("tiles_per_gauss", torch_to_numpy(tiles_ref), mx_to_numpy(actual["tiles_per_gauss"])),
            compare_array("isect_ids", torch_to_numpy(isect_ref), mx_to_numpy(actual["isect_ids"])),
            compare_array("flatten_ids", torch_to_numpy(flatten_ref), mx_to_numpy(actual["flatten_ids"])),
            compare_array("offsets", torch_to_numpy(offsets_ref), mx_to_numpy(offsets)),
        ]
    )


if __name__ == "__main__":
    main()
