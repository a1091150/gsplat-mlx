#!/usr/bin/env python3

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import mlx.core as mx
import numpy as np

from gsplat_core import (
    intersect_offset_forward,
    intersect_tile_forward,
    projection_ewa_3dgs_fused_forward,
    rasterize_to_pixels_3dgs_forward,
)


def write_png(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("write_png expects uint8 RGB image with shape [H, W, 3].")

    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )


def normalize_quats(quats: mx.array) -> mx.array:
    norms = mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True))
    return quats / mx.maximum(norms, mx.array(1.0e-8, dtype=mx.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/random_3dgs.png"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-gaussians", type=int, default=1024)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = args.num_gaussians
    image_width = args.width
    image_height = args.height
    tile_size = args.tile_size
    tile_width = (image_width + tile_size - 1) // tile_size
    tile_height = (image_height + tile_size - 1) // tile_size

    np_rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)

    means_np = np.empty((1, n, 3), dtype=np.float32)
    means_np[..., 0] = np_rng.uniform(-1.2, 1.2, size=(1, n))
    means_np[..., 1] = np_rng.uniform(-1.2, 1.2, size=(1, n))
    means_np[..., 2] = np_rng.uniform(2.0, 6.0, size=(1, n))

    scales_np = np_rng.uniform(0.015, 0.055, size=(1, n, 3)).astype(np.float32)
    quats_np = np_rng.normal(size=(1, n, 4)).astype(np.float32)
    colors_np = np_rng.uniform(0.05, 1.0, size=(1, 1, n, 3)).astype(np.float32)
    opacities_np = np_rng.uniform(0.18, 0.72, size=(1, n)).astype(np.float32)

    means = mx.array(means_np)
    quats = normalize_quats(mx.array(quats_np))
    scales = mx.array(scales_np)
    projection_opacities = mx.array(opacities_np)
    raster_opacities = mx.array(opacities_np.reshape(1, 1, n))
    colors = mx.array(colors_np)
    backgrounds = mx.array([[0.015, 0.018, 0.025]], dtype=mx.float32)

    viewmats = mx.array(
        [[
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]],
        dtype=mx.float32,
    )
    focal = 0.86 * float(min(image_width, image_height))
    Ks = mx.array(
        [[
            [
                [focal, 0.0, image_width * 0.5],
                [0.0, focal, image_height * 0.5],
                [0.0, 0.0, 1.0],
            ]
        ]],
        dtype=mx.float32,
    )

    projection = projection_ewa_3dgs_fused_forward(
        {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": projection_opacities,
            "viewmats": viewmats,
            "Ks": Ks,
        },
        image_width=image_width,
        image_height=image_height,
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
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
    )
    tile_offsets = intersect_offset_forward(
        intersections["isect_ids"],
        I=1,
        tile_width=tile_width,
        tile_height=tile_height,
    )

    render = rasterize_to_pixels_3dgs_forward(
        {
            "means2d": projection["means2d"],
            "conics": projection["conics"],
            "colors": colors,
            "opacities": raster_opacities,
            "backgrounds": backgrounds,
            "tile_offsets": tile_offsets,
            "flatten_ids": intersections["flatten_ids"],
        },
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )

    mx.eval(*projection.values(), *intersections.values(), tile_offsets, *render.values())
    rgb = np.asarray(render["render_colors"][0])
    alpha = np.asarray(render["render_alphas"][0, ..., 0])
    radii = np.asarray(projection["radii"])
    isect_ids = np.asarray(intersections["isect_ids"])

    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    write_png(args.out, rgb_u8)

    visible = np.count_nonzero(np.any(radii > 0, axis=-1))
    print(f"wrote {args.out}")
    print(f"image={image_width}x{image_height}, gaussians={n}, seed={args.seed}")
    print(f"tiles={tile_width}x{tile_height}, tile_size={tile_size}")
    print(f"visible_gaussians={visible}, intersections={isect_ids.shape[0]}")
    print(
        "alpha: "
        f"sum={float(alpha.sum()):.6f}, "
        f"max={float(alpha.max(initial=0.0)):.6f}, "
        f"nonzero_pixels={int(np.count_nonzero(alpha > 1.0e-6))}"
    )
    print(
        "rgb: "
        f"min={float(rgb.min(initial=0.0)):.6f}, "
        f"max={float(rgb.max(initial=0.0)):.6f}, "
        f"mean={float(rgb.mean()):.6f}"
    )


if __name__ == "__main__":
    main()
