#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from gsplat_core import (
    intersect_offset_forward,
    intersect_tile_forward,
    projection_ewa_3dgs_fused_forward,
    quat_scale_to_covar_preci_backward,
    quat_scale_to_covar_preci_forward,
    rasterize_to_pixels_3dgs_backward,
    rasterize_to_pixels_3dgs_forward,
    spherical_harmonics_backward,
    spherical_harmonics_forward,
)
from parity_utils import compare_array, finish, mx_to_numpy


ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / "refs"


def mx_array(data: np.lib.npyio.NpzFile, name: str) -> mx.array:
    return mx.array(data[name])


def scalar(data: np.lib.npyio.NpzFile, name: str) -> int | float | bool:
    return data[name].item()


def ref(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    return data[f"ref__{name}"]


def compare_projection(data: np.lib.npyio.NpzFile) -> list[bool]:
    inputs = {
        "means": mx_array(data, "input__means"),
        "viewmats": mx_array(data, "input__viewmats"),
        "Ks": mx_array(data, "input__Ks"),
    }
    if "input__covars" in data.files:
        inputs["covars"] = mx_array(data, "input__covars")
    else:
        inputs["quats"] = mx_array(data, "input__quats")
        inputs["scales"] = mx_array(data, "input__scales")
    if "input__opacities" in data.files:
        inputs["opacities"] = mx_array(data, "input__opacities")

    actual = projection_ewa_3dgs_fused_forward(
        inputs,
        image_width=int(scalar(data, "input__image_width")),
        image_height=int(scalar(data, "input__image_height")),
        eps2d=float(scalar(data, "input__eps2d")),
        near_plane=float(scalar(data, "input__near_plane")),
        far_plane=float(scalar(data, "input__far_plane")),
        radius_clip=float(scalar(data, "input__radius_clip")),
        calc_compensations=bool(scalar(data, "input__calc_compensations")),
        camera_model=int(scalar(data, "input__camera_model")),
    )
    mx.eval(*actual.values())
    results = [
        compare_array("radii", ref(data, "radii"), mx_to_numpy(actual["radii"])),
    ]
    if "ref__means2d" in data.files:
        results.append(compare_array("means2d", ref(data, "means2d"), mx_to_numpy(actual["means2d"]), atol=1.0e-4, rtol=1.0e-4))
    if "ref__depths" in data.files:
        results.append(compare_array("depths", ref(data, "depths"), mx_to_numpy(actual["depths"]), atol=1.0e-4, rtol=1.0e-4))
    if "ref__conics" in data.files:
        results.append(compare_array("conics", ref(data, "conics"), mx_to_numpy(actual["conics"]), atol=1.0e-4, rtol=1.0e-4))
    if "ref__compensations" in data.files:
        results.append(compare_array("compensations", ref(data, "compensations"), mx_to_numpy(actual["compensations"]), atol=1.0e-4, rtol=1.0e-4))
    return results


def compare_intersect(data: np.lib.npyio.NpzFile) -> list[bool]:
    actual = intersect_tile_forward(
        {
            "means2d": mx_array(data, "input__means2d"),
            "radii": mx_array(data, "input__radii"),
            "depths": mx_array(data, "input__depths"),
        },
        I=int(scalar(data, "input__I")),
        tile_size=int(scalar(data, "input__tile_size")),
        tile_width=int(scalar(data, "input__tile_width")),
        tile_height=int(scalar(data, "input__tile_height")),
        sort=bool(scalar(data, "input__sort")),
        segmented=bool(scalar(data, "input__segmented")),
    )
    offsets = intersect_offset_forward(
        actual["isect_ids"],
        I=int(scalar(data, "input__I")),
        tile_width=int(scalar(data, "input__tile_width")),
        tile_height=int(scalar(data, "input__tile_height")),
    )
    mx.eval(*actual.values(), offsets)
    return [
        compare_array("tiles_per_gauss", ref(data, "tiles_per_gauss"), mx_to_numpy(actual["tiles_per_gauss"])),
        compare_array("isect_ids", ref(data, "isect_ids"), mx_to_numpy(actual["isect_ids"])),
        compare_array("flatten_ids", ref(data, "flatten_ids"), mx_to_numpy(actual["flatten_ids"])),
        compare_array("offsets", ref(data, "offsets"), mx_to_numpy(offsets)),
    ]


def compare_rasterize(data: np.lib.npyio.NpzFile) -> list[bool]:
    inputs = {
        "means2d": mx_array(data, "input__means2d"),
        "conics": mx_array(data, "input__conics"),
        "colors": mx_array(data, "input__colors"),
        "opacities": mx_array(data, "input__opacities"),
        "backgrounds": mx_array(data, "input__backgrounds"),
        "tile_offsets": mx_array(data, "input__tile_offsets"),
        "flatten_ids": mx_array(data, "input__flatten_ids"),
    }
    if "input__masks" in data.files:
        inputs["masks"] = mx_array(data, "input__masks")

    actual = rasterize_to_pixels_3dgs_forward(
        inputs,
        image_width=int(scalar(data, "input__image_width")),
        image_height=int(scalar(data, "input__image_height")),
        tile_size=int(scalar(data, "input__tile_size")),
    )
    mx.eval(*actual.values())
    return [
        compare_array("render_colors", ref(data, "render_colors"), mx_to_numpy(actual["render_colors"]), atol=1.0e-4, rtol=1.0e-4),
        compare_array("render_alphas", ref(data, "render_alphas"), mx_to_numpy(actual["render_alphas"]), atol=1.0e-4, rtol=1.0e-4),
    ]


def compare_rasterize_backward(data: np.lib.npyio.NpzFile) -> list[bool]:
    inputs = {
        "means2d": mx_array(data, "input__means2d"),
        "conics": mx_array(data, "input__conics"),
        "colors": mx_array(data, "input__colors"),
        "opacities": mx_array(data, "input__opacities"),
        "tile_offsets": mx_array(data, "input__tile_offsets"),
        "flatten_ids": mx_array(data, "input__flatten_ids"),
    }
    if "input__backgrounds" in data.files:
        inputs["backgrounds"] = mx_array(data, "input__backgrounds")
    if "input__masks" in data.files:
        inputs["masks"] = mx_array(data, "input__masks")

    actual = rasterize_to_pixels_3dgs_backward(
        inputs,
        {
            "render_alphas": mx_array(data, "fwd__render_alphas"),
            "last_ids": mx_array(data, "fwd__last_ids"),
        },
        {
            "v_render_colors": mx_array(data, "cotangent__v_render_colors"),
            "v_render_alphas": mx_array(data, "cotangent__v_render_alphas"),
        },
        image_width=int(scalar(data, "input__image_width")),
        image_height=int(scalar(data, "input__image_height")),
        tile_size=int(scalar(data, "input__tile_size")),
        absgrad=bool(scalar(data, "meta__absgrad")) if "meta__absgrad" in data.files else False,
    )
    mx.eval(*actual.values())
    results = [
        compare_array("v_means2d", ref(data, "v_means2d"), mx_to_numpy(actual["v_means2d"]), atol=1.0e-4, rtol=1.0e-4),
        compare_array("v_conics", ref(data, "v_conics"), mx_to_numpy(actual["v_conics"]), atol=1.0e-4, rtol=1.0e-4),
        compare_array("v_colors", ref(data, "v_colors"), mx_to_numpy(actual["v_colors"]), atol=1.0e-4, rtol=1.0e-4),
        compare_array("v_opacities", ref(data, "v_opacities"), mx_to_numpy(actual["v_opacities"]), atol=1.0e-4, rtol=1.0e-4),
    ]
    if "ref__v_backgrounds" in data.files:
        results.append(compare_array("v_backgrounds", ref(data, "v_backgrounds"), mx_to_numpy(actual["v_backgrounds"]), atol=1.0e-4, rtol=1.0e-4))
    if "ref__v_means2d_abs" in data.files:
        results.append(compare_array("v_means2d_abs", ref(data, "v_means2d_abs"), mx_to_numpy(actual["v_means2d_abs"]), atol=1.0e-4, rtol=1.0e-4))
    return results


def compare_spherical_harmonics(data: np.lib.npyio.NpzFile) -> list[bool]:
    inputs = {
        "dirs": mx_array(data, "input__dirs"),
        "coeffs": mx_array(data, "input__coeffs"),
    }
    if "input__masks" in data.files:
        inputs["masks"] = mx_array(data, "input__masks")

    actual = spherical_harmonics_forward(
        int(scalar(data, "input__degrees_to_use")),
        inputs,
    )
    mx.eval(actual)
    return [compare_array("colors", ref(data, "colors"), mx_to_numpy(actual), atol=1.0e-4, rtol=1.0e-4)]


def compare_spherical_harmonics_backward(data: np.lib.npyio.NpzFile) -> list[bool]:
    inputs = {
        "dirs": mx_array(data, "input__dirs"),
        "coeffs": mx_array(data, "input__coeffs"),
    }
    if "input__masks" in data.files:
        inputs["masks"] = mx_array(data, "input__masks")
    actual = spherical_harmonics_backward(
        int(scalar(data, "input__degrees_to_use")),
        inputs,
        {"v_colors": mx_array(data, "cotangent__v_colors")},
        compute_v_dirs=bool(scalar(data, "meta__compute_v_dirs")),
    )
    mx.eval(*actual.values())
    results = [
        compare_array("v_coeffs", ref(data, "v_coeffs"), mx_to_numpy(actual["v_coeffs"]), atol=1.0e-4, rtol=1.0e-4),
    ]
    if "ref__v_dirs" in data.files:
        results.append(compare_array("v_dirs", ref(data, "v_dirs"), mx_to_numpy(actual["v_dirs"]), atol=1.0e-3, rtol=1.0e-3))
    return results


def compare_quat_scale(data: np.lib.npyio.NpzFile) -> list[bool]:
    compute_covar = bool(scalar(data, "input__compute_covar")) if "input__compute_covar" in data.files else True
    compute_preci = bool(scalar(data, "input__compute_preci")) if "input__compute_preci" in data.files else True
    triu = bool(scalar(data, "input__triu")) if "input__triu" in data.files else True
    actual = quat_scale_to_covar_preci_forward(
        {
            "quats": mx_array(data, "input__quats"),
            "scales": mx_array(data, "input__scales"),
        },
        compute_covar=compute_covar,
        compute_preci=compute_preci,
        triu=triu,
    )
    mx.eval(*actual.values())
    results = []
    if "ref__covars" in data.files:
        results.append(compare_array("covars", ref(data, "covars"), mx_to_numpy(actual["covars"]), atol=1.0e-4, rtol=1.0e-4))
    if "ref__precis" in data.files:
        results.append(compare_array("precis", ref(data, "precis"), mx_to_numpy(actual["precis"]), atol=1.0e-4, rtol=1.0e-4))
    return results


def compare_quat_scale_backward(data: np.lib.npyio.NpzFile) -> list[bool]:
    inputs = {
        "quats": mx_array(data, "input__quats"),
        "scales": mx_array(data, "input__scales"),
    }
    cotangents = {}
    if "cotangent__v_covars" in data.files:
        cotangents["v_covars"] = mx_array(data, "cotangent__v_covars")
    if "cotangent__v_precis" in data.files:
        cotangents["v_precis"] = mx_array(data, "cotangent__v_precis")
    triu = bool(scalar(data, "input__triu")) if "input__triu" in data.files else True
    actual = quat_scale_to_covar_preci_backward(inputs, cotangents, triu=triu)
    mx.eval(*actual.values())
    return [
        compare_array("v_quats", ref(data, "v_quats"), mx_to_numpy(actual["v_quats"]), atol=3.0e-3, rtol=3.0e-3),
        compare_array("v_scales", ref(data, "v_scales"), mx_to_numpy(actual["v_scales"]), atol=3.0e-3, rtol=3.0e-3),
    ]


def compare_chain(data: np.lib.npyio.NpzFile) -> list[bool]:
    projection = projection_ewa_3dgs_fused_forward(
        {
            "means": mx_array(data, "input__means"),
            "quats": mx_array(data, "input__quats"),
            "scales": mx_array(data, "input__scales"),
            "opacities": mx_array(data, "input__projection_opacities"),
            "viewmats": mx_array(data, "input__viewmats"),
            "Ks": mx_array(data, "input__Ks"),
        },
        image_width=int(scalar(data, "input__image_width")),
        image_height=int(scalar(data, "input__image_height")),
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
        tile_size=int(scalar(data, "input__tile_size")),
        tile_width=int(scalar(data, "input__tile_width")),
        tile_height=int(scalar(data, "input__tile_height")),
        sort=True,
        segmented=False,
    )
    offsets = intersect_offset_forward(
        intersections["isect_ids"],
        I=1,
        tile_width=int(scalar(data, "input__tile_width")),
        tile_height=int(scalar(data, "input__tile_height")),
    )
    colors = spherical_harmonics_forward(
        0,
        {"dirs": mx_array(data, "input__dirs"), "coeffs": mx_array(data, "input__coeffs")},
    )
    render = rasterize_to_pixels_3dgs_forward(
        {
            "means2d": projection["means2d"],
            "conics": projection["conics"],
            "colors": colors,
            "opacities": mx_array(data, "input__raster_opacities"),
            "backgrounds": mx_array(data, "input__backgrounds"),
            "tile_offsets": offsets,
            "flatten_ids": intersections["flatten_ids"],
        },
        image_width=int(scalar(data, "input__image_width")),
        image_height=int(scalar(data, "input__image_height")),
        tile_size=int(scalar(data, "input__tile_size")),
    )
    mx.eval(*projection.values(), *intersections.values(), offsets, colors, *render.values())
    return [
        compare_array("radii", ref(data, "radii"), mx_to_numpy(projection["radii"])),
        compare_array("means2d", ref(data, "means2d"), mx_to_numpy(projection["means2d"]), atol=1.0e-4, rtol=1.0e-4),
        compare_array("depths", ref(data, "depths"), mx_to_numpy(projection["depths"]), atol=1.0e-4, rtol=1.0e-4),
        compare_array("conics", ref(data, "conics"), mx_to_numpy(projection["conics"]), atol=1.0e-4, rtol=1.0e-4),
        compare_array("tiles_per_gauss", ref(data, "tiles_per_gauss"), mx_to_numpy(intersections["tiles_per_gauss"])),
        compare_array("isect_ids", ref(data, "isect_ids"), mx_to_numpy(intersections["isect_ids"])),
        compare_array("flatten_ids", ref(data, "flatten_ids"), mx_to_numpy(intersections["flatten_ids"])),
        compare_array("tile_offsets", ref(data, "tile_offsets"), mx_to_numpy(offsets)),
        compare_array("colors", ref(data, "colors"), mx_to_numpy(colors), atol=1.0e-4, rtol=1.0e-4),
        compare_array("render_colors", ref(data, "render_colors"), mx_to_numpy(render["render_colors"]), atol=1.0e-4, rtol=1.0e-4),
        compare_array("render_alphas", ref(data, "render_alphas"), mx_to_numpy(render["render_alphas"]), atol=1.0e-4, rtol=1.0e-4),
    ]


COMPARERS: dict[str, Callable[[np.lib.npyio.NpzFile], list[bool]]] = {
    "forward_3dgs_chain.npz": compare_chain,
    "intersect_tile_forward.npz": compare_intersect,
    "projection_ewa_3dgs_fused_forward.npz": compare_projection,
    "quat_scale_to_covar_preci_backward.npz": compare_quat_scale_backward,
    "quat_scale_to_covar_preci_forward.npz": compare_quat_scale,
    "rasterize_to_pixels_3dgs_backward.npz": compare_rasterize_backward,
    "rasterize_to_pixels_3dgs_forward.npz": compare_rasterize,
    "spherical_harmonics_backward.npz": compare_spherical_harmonics_backward,
    "spherical_harmonics_forward.npz": compare_spherical_harmonics,
}

EXTRA_COMPARERS: dict[str, Callable[[np.lib.npyio.NpzFile], list[bool]]] = {
    "projection_ewa_3dgs_fused_edge_cases.npz": compare_projection,
    "quat_scale_to_covar_preci_edge_cases.npz": compare_quat_scale,
    "rasterize_to_pixels_3dgs_masks.npz": compare_rasterize,
    "spherical_harmonics_degree4_masks.npz": compare_spherical_harmonics,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific exported .npz files to compare. Defaults to all refs/*.npz fixtures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.paths or sorted(
        path
        for path in REFS.glob("*.npz")
        if path.name in COMPARERS or path.name in EXTRA_COMPARERS
    )
    all_results: list[bool] = []
    for path in paths:
        path = path if path.is_absolute() else ROOT / path
        comparer = COMPARERS.get(path.name) or EXTRA_COMPARERS.get(path.name)
        if comparer is None:
            print(f"{path}: SKIP no comparer")
            continue
        if not path.exists():
            print(f"{path}: FAIL missing fixture")
            all_results.append(False)
            continue
        print(f"== {path.relative_to(ROOT)} ==")
        with np.load(path, allow_pickle=True) as data:
            all_results.extend(comparer(data))
    finish(all_results)


if __name__ == "__main__":
    main()
