#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("projection_ewa_3dgs_fused_edge_cases.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    means = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.001],
                [0.0, 0.0, 200.0],
                [0.25, -0.25, 2.0],
            ]
        ],
        device="cuda",
        dtype=torch.float32,
    )
    covars = torch.tensor(
        [
            [
                [0.01, 0.0, 0.0, 0.01, 0.0, 0.01],
                [0.01, 0.0, 0.0, 0.01, 0.0, 0.01],
                [0.01, 0.0, 0.0, 0.01, 0.0, 0.01],
                [0.000001, 0.0, 0.0, 0.000001, 0.0, 0.000001],
            ]
        ],
        device="cuda",
        dtype=torch.float32,
    )
    viewmats = torch.eye(4, device="cuda", dtype=torch.float32).reshape(1, 1, 4, 4)
    Ks = torch.tensor(
        [[[[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]]],
        device="cuda",
        dtype=torch.float32,
    )

    radii, _, _, _, _ = gsplat.fully_fused_projection(
        means,
        covars,
        None,
        None,
        viewmats,
        Ks,
        width=64,
        height=64,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=2.0,
        packed=False,
        calc_compensations=False,
        camera_model="pinhole",
        opacities=None,
    )

    save_npz(
        args.out,
        input__image_width=64,
        input__image_height=64,
        input__eps2d=0.3,
        input__near_plane=0.01,
        input__far_plane=100.0,
        input__radius_clip=2.0,
        input__calc_compensations=False,
        input__camera_model=0,
        input__means=means,
        input__covars=covars,
        input__viewmats=viewmats,
        input__Ks=Ks,
        ref__radii=radii,
    )


if __name__ == "__main__":
    main()
