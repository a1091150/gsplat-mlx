#!/usr/bin/env python3

from export_utils import load_gsplat_wrapper, load_torch_cuda, parse_args, save_npz


def main() -> None:
    args = parse_args("spherical_harmonics_degree4_masks.npz")
    torch = load_torch_cuda()
    gsplat = load_gsplat_wrapper()

    dirs = torch.tensor(
        [
            [0.25, -0.5, 1.0],
            [-0.75, 0.1, 0.6],
            [0.0, 0.0, 0.0],
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
    colors = gsplat.spherical_harmonics(4, dirs, coeffs, masks)

    save_npz(
        args.out,
        input__degrees_to_use=4,
        input__dirs=dirs,
        input__coeffs=coeffs,
        input__masks=masks,
        ref__colors=colors,
    )


if __name__ == "__main__":
    main()
