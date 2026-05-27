#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("projection_ewa_3dgs_fused_backward.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    image_width = 64
    image_height = 48
    eps2d = 0.3
    near_plane = 0.01
    far_plane = 100.0
    radius_clip = 0.0
    calc_compensations = True
    camera_model = 0
    camera_model_obj = gsplat._make_lazy_cuda_obj("CameraModelType.PINHOLE")
    viewmats_requires_grad = False

    means = torch.tensor([[[0.1, -0.05, 2.0], [0.25, 0.15, 3.0]]], device="cuda", dtype=torch.float32)
    covars = torch.tensor(
        [[[0.04, 0.002, 0.001, 0.05, -0.003, 0.06],
          [0.03, -0.001, 0.002, 0.045, 0.004, 0.055]]],
        device="cuda",
        dtype=torch.float32,
    )
    viewmats = torch.eye(4, device="cuda", dtype=torch.float32).reshape(1, 1, 4, 4)
    Ks = torch.tensor(
        [[[[90.0, 0.0, 32.0], [0.0, 88.0, 24.0], [0.0, 0.0, 1.0]]]],
        device="cuda",
        dtype=torch.float32,
    )
    opacities = None
    quats = None
    scales = None

    projection_fwd = gsplat._make_lazy_cuda_func("projection_ewa_3dgs_fused_fwd")
    try:
        radii, means2d, depths, conics, compensations = projection_fwd(
            means,
            covars,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model_obj,
        )
    except TypeError:
        radii, means2d, depths, conics, compensations = projection_fwd(
            means,
            covars,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model,
        )
    v_means2d = torch.tensor([[[[0.2, -0.1], [0.05, 0.3]]]], device="cuda", dtype=torch.float32)
    v_depths = torch.tensor([[[0.4, -0.2]]], device="cuda", dtype=torch.float32)
    v_conics = torch.tensor(
        [[[[0.1, -0.05, 0.2], [-0.15, 0.25, -0.1]]]],
        device="cuda",
        dtype=torch.float32,
    )
    v_compensations = torch.tensor([[[0.3, -0.25]]], device="cuda", dtype=torch.float32)

    projection_bwd = gsplat._make_lazy_cuda_func("projection_ewa_3dgs_fused_bwd")
    try:
        v_means, v_covars, v_quats, v_scales, v_viewmats = projection_bwd(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            image_width,
            image_height,
            eps2d,
            camera_model_obj,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations.contiguous(),
            viewmats_requires_grad,
        )
    except TypeError:
        v_means, v_covars, v_quats, v_scales, v_viewmats = projection_bwd(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            image_width,
            image_height,
            eps2d,
            camera_model,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations.contiguous(),
            viewmats_requires_grad,
        )

    save_npz(
        args.out,
        input__means=means,
        input__covars=covars,
        input__viewmats=viewmats,
        input__Ks=Ks,
        input__image_width=image_width,
        input__image_height=image_height,
        input__eps2d=eps2d,
        input__camera_model=camera_model,
        fwd__radii=radii,
        fwd__means2d=means2d,
        fwd__depths=depths,
        fwd__conics=conics,
        fwd__compensations=compensations,
        cotangent__v_means2d=v_means2d,
        cotangent__v_depths=v_depths,
        cotangent__v_conics=v_conics,
        cotangent__v_compensations=v_compensations,
        meta__viewmats_requires_grad=viewmats_requires_grad,
        ref__v_means=v_means,
        ref__v_covars=v_covars,
    )


if __name__ == "__main__":
    main()
