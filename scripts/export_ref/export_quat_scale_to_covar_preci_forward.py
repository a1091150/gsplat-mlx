#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("quat_scale_to_covar_preci_forward.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    scales = torch.tensor(
        [[2.0, 3.0, 4.0], [0.5, 2.0, 4.0]],
        device="cuda",
        dtype=torch.float32,
    )
    covars, precis = gsplat.quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=True, triu=True
    )

    save_npz(
        args.out,
        input__quats=quats,
        input__scales=scales,
        ref__covars=covars,
        ref__precis=precis,
    )


if __name__ == "__main__":
    main()
