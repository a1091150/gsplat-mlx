#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import (
    intersect_offset_forward,
    intersect_tile_forward,
    projection_ewa_3dgs_fused_forward,
    rasterize_to_pixels_3dgs_forward,
    spherical_harmonics_forward,
)
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

    image_width = 16
    image_height = 16
    tile_size = 8
    tile_width = 2
    tile_height = 2
    c0 = 0.2820947917738781

    means_t = torch.tensor([[[0.0, 0.0, 2.0]]], device="cuda", dtype=torch.float32)
    quats_t = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device="cuda", dtype=torch.float32)
    scales_t = torch.tensor([[[0.1, 0.1, 0.1]]], device="cuda", dtype=torch.float32)
    projection_opacities_t = torch.tensor([[0.8]], device="cuda", dtype=torch.float32)
    raster_opacities_t = torch.tensor([[[0.8]]], device="cuda", dtype=torch.float32)
    viewmats_t = torch.eye(4, device="cuda", dtype=torch.float32).reshape(1, 1, 4, 4)
    Ks_t = torch.tensor(
        [[[[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]]]],
        device="cuda",
        dtype=torch.float32,
    )

    radii_t, means2d_t, depths_t, conics_t, _ = gsplat.fully_fused_projection(
        means_t,
        None,
        quats_t,
        scales_t,
        viewmats_t,
        Ks_t,
        width=image_width,
        height=image_height,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=0.0,
        packed=False,
        calc_compensations=False,
        camera_model="pinhole",
        opacities=projection_opacities_t,
    )
    tiles_t, isect_ids_t, flatten_ids_t = gsplat.isect_tiles(
        means2d_t,
        radii_t,
        depths_t,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
        packed=False,
    )
    offsets_t = gsplat.isect_offset_encode(isect_ids_t, 1, tile_width, tile_height)
    dirs_t = torch.tensor([[[[0.0, 0.0, 1.0]]]], device="cuda", dtype=torch.float32)
    coeffs_t = torch.tensor([[[[[1.0 / c0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32)
    colors_t = gsplat.spherical_harmonics(0, dirs_t, coeffs_t)
    backgrounds_t = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    gsplat_backgrounds_t = backgrounds_t[:, None, :]
    colors_ref, alphas_ref = gsplat.rasterize_to_pixels(
        means2d_t,
        conics_t,
        colors_t,
        raster_opacities_t,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=offsets_t,
        flatten_ids=flatten_ids_t,
        backgrounds=gsplat_backgrounds_t,
        masks=None,
        packed=False,
        absgrad=False,
    )

    projection = projection_ewa_3dgs_fused_forward(
        {
            "means": torch_to_mx(means_t),
            "quats": torch_to_mx(quats_t),
            "scales": torch_to_mx(scales_t),
            "opacities": torch_to_mx(projection_opacities_t),
            "viewmats": torch_to_mx(viewmats_t),
            "Ks": torch_to_mx(Ks_t),
        },
        image_width=image_width,
        image_height=image_height,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=0.0,
        calc_compensations=False,
        camera_model=0,
    )
    intersections = intersect_tile_forward(
        {
            "means2d": projection["means2d"],
            "radii": projection["radii"],
            "depths": projection["depths"],
        },
        I=1,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
    )
    offsets = intersect_offset_forward(
        intersections["isect_ids"], I=1, tile_width=tile_width, tile_height=tile_height
    )
    colors = spherical_harmonics_forward(
        0,
        {"dirs": torch_to_mx(dirs_t), "coeffs": torch_to_mx(coeffs_t)},
    )
    render = rasterize_to_pixels_3dgs_forward(
        {
            "means2d": projection["means2d"],
            "conics": projection["conics"],
            "colors": colors,
            "opacities": torch_to_mx(raster_opacities_t),
            "backgrounds": torch_to_mx(backgrounds_t),
            "tile_offsets": offsets,
            "flatten_ids": intersections["flatten_ids"],
        },
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )
    mx.eval(*projection.values(), *intersections.values(), offsets, colors, *render.values())

    finish(
        [
            compare_array("radii", torch_to_numpy(radii_t), mx_to_numpy(projection["radii"])),
            compare_array("means2d", torch_to_numpy(means2d_t), mx_to_numpy(projection["means2d"]), atol=1.0e-4, rtol=1.0e-4),
            compare_array("depths", torch_to_numpy(depths_t), mx_to_numpy(projection["depths"]), atol=1.0e-4, rtol=1.0e-4),
            compare_array("conics", torch_to_numpy(conics_t), mx_to_numpy(projection["conics"]), atol=1.0e-4, rtol=1.0e-4),
            compare_array("tiles_per_gauss", torch_to_numpy(tiles_t), mx_to_numpy(intersections["tiles_per_gauss"])),
            compare_array("isect_ids", torch_to_numpy(isect_ids_t), mx_to_numpy(intersections["isect_ids"])),
            compare_array("flatten_ids", torch_to_numpy(flatten_ids_t), mx_to_numpy(intersections["flatten_ids"])),
            compare_array("tile_offsets", torch_to_numpy(offsets_t), mx_to_numpy(offsets)),
            compare_array("colors", torch_to_numpy(colors_t), mx_to_numpy(colors)),
            compare_array("render_colors", torch_to_numpy(colors_ref), mx_to_numpy(render["render_colors"]), atol=1.0e-4, rtol=1.0e-4),
            compare_array("render_alphas", torch_to_numpy(alphas_ref), mx_to_numpy(render["render_alphas"]), atol=1.0e-4, rtol=1.0e-4),
        ]
    )


if __name__ == "__main__":
    main()
