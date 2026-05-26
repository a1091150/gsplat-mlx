#!/usr/bin/env python3

import mlx.core as mx

from gsplat_core import projection_ewa_3dgs_fused_forward
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

    means_t = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.25, -0.25, 2.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    quats_t = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    scales_t = torch.tensor(
        [[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]],
        device="cuda",
        dtype=torch.float32,
    )
    opacities_t = torch.tensor([[0.8, 0.6]], device="cuda", dtype=torch.float32)
    viewmats_t = torch.eye(4, device="cuda", dtype=torch.float32).reshape(1, 1, 4, 4)
    Ks_t = torch.tensor(
        [[[[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]]],
        device="cuda",
        dtype=torch.float32,
    )

    ref = gsplat.fully_fused_projection(
        means_t,
        None,
        quats_t,
        scales_t,
        viewmats_t,
        Ks_t,
        width=64,
        height=64,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=0.0,
        packed=False,
        calc_compensations=True,
        camera_model="pinhole",
        opacities=opacities_t,
    )
    actual = projection_ewa_3dgs_fused_forward(
        {
            "means": torch_to_mx(means_t),
            "quats": torch_to_mx(quats_t),
            "scales": torch_to_mx(scales_t),
            "opacities": torch_to_mx(opacities_t),
            "viewmats": torch_to_mx(viewmats_t),
            "Ks": torch_to_mx(Ks_t),
        },
        image_width=64,
        image_height=64,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=0.0,
        calc_compensations=True,
        camera_model=0,
    )
    mx.eval(*actual.values())

    ref_names = ["radii", "means2d", "depths", "conics", "compensations"]
    results = []
    for name, ref_value in zip(ref_names, ref):
        results.append(
            compare_array(
                name,
                torch_to_numpy(ref_value),
                mx_to_numpy(actual[name]),
                atol=1.0e-4,
                rtol=1.0e-4,
            )
        )
    finish(results)


if __name__ == "__main__":
    main()
