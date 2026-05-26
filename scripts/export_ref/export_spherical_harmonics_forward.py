#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("spherical_harmonics_forward.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    dirs = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    coeffs = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    masks = torch.tensor([True, False], device="cuda", dtype=torch.bool)
    colors = gsplat.spherical_harmonics(1, dirs, coeffs, masks)

    save_npz(
        args.out,
        input__degrees_to_use=1,
        input__dirs=dirs,
        input__coeffs=coeffs,
        input__masks=masks,
        ref__colors=colors,
    )


if __name__ == "__main__":
    main()
