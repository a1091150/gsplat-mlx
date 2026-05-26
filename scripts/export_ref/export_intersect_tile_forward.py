#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("intersect_tile_forward.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    means2d = torch.tensor(
        [[[20.0, 20.0], [50.0, 50.0], [8.0, 8.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    radii = torch.tensor([[[10, 10], [5, 5], [0, 0]]], device="cuda", dtype=torch.int32)
    depths = torch.tensor([[1.0, 0.5, 2.0]], device="cuda", dtype=torch.float32)

    tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
        means2d,
        radii,
        depths,
        tile_size=16,
        tile_width=4,
        tile_height=4,
        sort=True,
        segmented=False,
        packed=False,
    )
    offsets = gsplat.isect_offset_encode(isect_ids, 1, 4, 4)

    save_npz(
        args.out,
        input__I=1,
        input__tile_size=16,
        input__tile_width=4,
        input__tile_height=4,
        input__sort=True,
        input__segmented=False,
        input__means2d=means2d,
        input__radii=radii,
        input__depths=depths,
        ref__tiles_per_gauss=tiles_per_gauss,
        ref__isect_ids=isect_ids,
        ref__flatten_ids=flatten_ids,
        ref__offsets=offsets,
    )


if __name__ == "__main__":
    main()
