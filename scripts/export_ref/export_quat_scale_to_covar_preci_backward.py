#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("quat_scale_to_covar_preci_backward.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9238795, 0.3826834, 0.0, 0.0],
            [0.35, 0.2, -0.4, 0.8],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    scales = torch.tensor(
        [
            [2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5],
            [2.0, 0.75, 1.25],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    v_covars = torch.tensor(
        [
            [0.1, 0.0, 0.0, 0.2, 0.0, 0.3],
            [-0.2, 0.05, 0.1, 0.4, -0.15, 0.25],
            [0.3, -0.1, 0.2, -0.4, 0.15, 0.05],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    v_precis = torch.tensor(
        [
            [0.4, 0.0, 0.0, 0.5, 0.0, 0.6],
            [0.3, -0.25, 0.35, -0.1, 0.2, 0.45],
            [-0.15, 0.2, -0.25, 0.5, -0.35, 0.1],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    triu = True

    quat_scale_bwd = gsplat._make_lazy_cuda_func("quat_scale_to_covar_preci_bwd")
    v_quats, v_scales = quat_scale_bwd(
        quats,
        scales,
        triu,
        v_covars.contiguous(),
        v_precis.contiguous(),
    )

    save_npz(
        args.out,
        input__quats=quats,
        input__scales=scales,
        input__triu=triu,
        cotangent__v_covars=v_covars,
        cotangent__v_precis=v_precis,
        ref__v_quats=v_quats,
        ref__v_scales=v_scales,
    )


if __name__ == "__main__":
    main()
