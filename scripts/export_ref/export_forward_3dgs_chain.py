#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("forward_3dgs_chain.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    image_width = 16
    image_height = 16
    tile_size = 8
    tile_width = 2
    tile_height = 2
    c0 = 0.2820947917738781

    means = torch.tensor([[[0.0, 0.0, 2.0]]], device="cuda", dtype=torch.float32)
    quats = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device="cuda", dtype=torch.float32)
    scales = torch.tensor([[[0.1, 0.1, 0.1]]], device="cuda", dtype=torch.float32)
    projection_opacities = torch.tensor([[0.8]], device="cuda", dtype=torch.float32)
    raster_opacities = torch.tensor([[[0.8]]], device="cuda", dtype=torch.float32)
    viewmats = torch.eye(4, device="cuda", dtype=torch.float32).reshape(1, 1, 4, 4)
    Ks = torch.tensor(
        [[[[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]]]],
        device="cuda",
        dtype=torch.float32,
    )

    radii, means2d, depths, conics, compensations = gsplat.fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width=image_width,
        height=image_height,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=0.0,
        packed=False,
        calc_compensations=False,
        camera_model="pinhole",
        opacities=projection_opacities,
    )
    tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
        means2d,
        radii,
        depths,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
        packed=False,
    )
    tile_offsets = gsplat.isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    dirs = torch.tensor([[[[0.0, 0.0, 1.0]]]], device="cuda", dtype=torch.float32)
    coeffs = torch.tensor([[[[[1.0 / c0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32)
    colors = gsplat.spherical_harmonics(0, dirs, coeffs)
    backgrounds = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    render_colors, render_alphas = gsplat.rasterize_to_pixels(
        means2d,
        conics,
        colors,
        raster_opacities,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=tile_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
        masks=None,
        packed=False,
        absgrad=False,
    )

    save_npz(
        args.out,
        input__image_width=image_width,
        input__image_height=image_height,
        input__tile_size=tile_size,
        input__tile_width=tile_width,
        input__tile_height=tile_height,
        input__means=means,
        input__quats=quats,
        input__scales=scales,
        input__projection_opacities=projection_opacities,
        input__raster_opacities=raster_opacities,
        input__viewmats=viewmats,
        input__Ks=Ks,
        input__dirs=dirs,
        input__coeffs=coeffs,
        input__backgrounds=backgrounds,
        ref__radii=radii,
        ref__means2d=means2d,
        ref__depths=depths,
        ref__conics=conics,
        ref__compensations=compensations,
        ref__tiles_per_gauss=tiles_per_gauss,
        ref__isect_ids=isect_ids,
        ref__flatten_ids=flatten_ids,
        ref__tile_offsets=tile_offsets,
        ref__colors=colors,
        ref__render_colors=render_colors,
        ref__render_alphas=render_alphas,
    )


if __name__ == "__main__":
    main()
