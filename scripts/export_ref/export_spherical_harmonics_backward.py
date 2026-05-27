#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("spherical_harmonics_backward.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    degrees_to_use = 4
    dirs = torch.tensor(
        [
            [0.25, -0.5, 1.0],
            [-0.75, 0.1, 0.6],
            [0.2, 0.3, -0.4],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    coeffs = (
        torch.arange(1, 3 * 25 * 3 + 1, device="cuda", dtype=torch.float32)
        .reshape(3, 25, 3)
        .mul_(0.01)
    )
    masks = torch.tensor([True, False, True], device="cuda", dtype=torch.bool)
    v_colors = torch.tensor(
        [
            [0.3, -0.2, 0.7],
            [1.0, 2.0, 3.0],
            [-0.4, 0.5, 0.25],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    compute_v_dirs = True

    spherical_harmonics_bwd = gsplat._make_lazy_cuda_func("spherical_harmonics_bwd")
    try:
        v_coeffs, v_dirs = spherical_harmonics_bwd(
            degrees_to_use,
            coeffs.shape[-2],
            dirs,
            coeffs,
            masks,
            v_colors.contiguous(),
            compute_v_dirs,
        )
    except TypeError:
        v_coeffs, v_dirs = spherical_harmonics_bwd(
            degrees_to_use,
            dirs,
            coeffs,
            masks,
            v_colors.contiguous(),
            compute_v_dirs,
        )

    save_npz(
        args.out,
        input__degrees_to_use=degrees_to_use,
        input__dirs=dirs,
        input__coeffs=coeffs,
        input__masks=masks,
        cotangent__v_colors=v_colors,
        meta__compute_v_dirs=compute_v_dirs,
        ref__v_dirs=v_dirs,
        ref__v_coeffs=v_coeffs,
    )


if __name__ == "__main__":
    main()
