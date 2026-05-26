#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("quat_scale_to_covar_preci_edge_cases.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    quats = torch.tensor(
        [
            [0.35, 0.2, -0.4, 0.8],
            [2.0, -1.0, 0.5, 0.25],
            [1.0, 0.0, 0.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    scales = torch.tensor(
        [
            [0.5, 1.5, 2.5],
            [2.0, 0.75, 1.25],
            [0.25, 3.0, 4.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    _, precis = gsplat.quat_scale_to_covar_preci(
        quats,
        scales,
        compute_covar=False,
        compute_preci=True,
        triu=False,
    )

    save_npz(
        args.out,
        input__compute_covar=False,
        input__compute_preci=True,
        input__triu=False,
        input__quats=quats,
        input__scales=scales,
        ref__precis=precis,
    )


if __name__ == "__main__":
    main()
