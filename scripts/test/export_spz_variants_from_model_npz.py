#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814


def sh_coeff_count(degree: int) -> int:
    return (degree + 1) * (degree + 1)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def normalize_quats(quats: np.ndarray) -> np.ndarray:
    return quats / np.clip(np.linalg.norm(quats, axis=-1, keepdims=True), 1.0e-8, None)


def quat_wxyz_to_rotmat(quats: np.ndarray) -> np.ndarray:
    q = normalize_quats(quats)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    q = np.empty((rot.shape[0], 4), dtype=np.float32)
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    mask = trace > 0.0
    if np.any(mask):
        s = np.sqrt(trace[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (rot[mask, 2, 1] - rot[mask, 1, 2]) / s
        q[mask, 2] = (rot[mask, 0, 2] - rot[mask, 2, 0]) / s
        q[mask, 3] = (rot[mask, 1, 0] - rot[mask, 0, 1]) / s
    mask_x = (~mask) & (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2])
    if np.any(mask_x):
        s = np.sqrt(1.0 + rot[mask_x, 0, 0] - rot[mask_x, 1, 1] - rot[mask_x, 2, 2]) * 2.0
        q[mask_x, 0] = (rot[mask_x, 2, 1] - rot[mask_x, 1, 2]) / s
        q[mask_x, 1] = 0.25 * s
        q[mask_x, 2] = (rot[mask_x, 0, 1] + rot[mask_x, 1, 0]) / s
        q[mask_x, 3] = (rot[mask_x, 0, 2] + rot[mask_x, 2, 0]) / s
    mask_y = (~mask) & (~mask_x) & (rot[:, 1, 1] > rot[:, 2, 2])
    if np.any(mask_y):
        s = np.sqrt(1.0 + rot[mask_y, 1, 1] - rot[mask_y, 0, 0] - rot[mask_y, 2, 2]) * 2.0
        q[mask_y, 0] = (rot[mask_y, 0, 2] - rot[mask_y, 2, 0]) / s
        q[mask_y, 1] = (rot[mask_y, 0, 1] + rot[mask_y, 1, 0]) / s
        q[mask_y, 2] = 0.25 * s
        q[mask_y, 3] = (rot[mask_y, 1, 2] + rot[mask_y, 2, 1]) / s
    mask_z = (~mask) & (~mask_x) & (~mask_y)
    if np.any(mask_z):
        s = np.sqrt(1.0 + rot[mask_z, 2, 2] - rot[mask_z, 0, 0] - rot[mask_z, 1, 1]) * 2.0
        q[mask_z, 0] = (rot[mask_z, 1, 0] - rot[mask_z, 0, 1]) / s
        q[mask_z, 1] = (rot[mask_z, 0, 2] + rot[mask_z, 2, 0]) / s
        q[mask_z, 2] = (rot[mask_z, 1, 2] + rot[mask_z, 2, 1]) / s
        q[mask_z, 3] = 0.25 * s
    return normalize_quats(q)


def gsplat_to_spz_axis() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def positions_to_spz(means: np.ndarray, mode: str) -> np.ndarray:
    if mode == "direct":
        return means.astype(np.float32)
    if mode == "scanner":
        out = np.empty_like(means, dtype=np.float32)
        out[:, 0] = means[:, 0]
        out[:, 1] = -means[:, 2]
        out[:, 2] = means[:, 1]
        return out
    raise ValueError(f"Unsupported position mode: {mode}")


def scales_to_spz(log_scales: np.ndarray, mode: str) -> np.ndarray:
    if mode == "direct":
        return log_scales.astype(np.float32)
    if mode == "scanner_axis":
        out = np.empty_like(log_scales, dtype=np.float32)
        out[:, 0] = log_scales[:, 0]
        out[:, 1] = log_scales[:, 2]
        out[:, 2] = log_scales[:, 1]
        return out
    raise ValueError(f"Unsupported scale mode: {mode}")


def transform_quats_wxyz(quats: np.ndarray, mode: str) -> np.ndarray:
    if mode == "direct":
        return normalize_quats(quats).astype(np.float32)

    position_axis = gsplat_to_spz_axis()
    fastgs_axis = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot = quat_wxyz_to_rotmat(quats)
    if mode == "position_axis":
        return rotmat_to_quat_wxyz(position_axis @ rot)
    if mode == "fastgs_conjugate":
        return rotmat_to_quat_wxyz(fastgs_axis @ rot @ fastgs_axis.T)
    if mode == "position_conjugate":
        return rotmat_to_quat_wxyz(position_axis @ rot @ position_axis.T)
    raise ValueError(f"Unsupported rotation mode: {mode}")


def quats_to_storage(quats_wxyz: np.ndarray, order: str) -> np.ndarray:
    q = normalize_quats(quats_wxyz).astype(np.float32)
    if order == "wxyz":
        return q
    if order == "xyzw":
        return q[:, [1, 2, 3, 0]]
    raise ValueError(f"Unsupported quaternion storage order: {order}")


def colors_to_spz(model: dict, mode: str) -> tuple[np.ndarray, int, np.ndarray]:
    source_mode = model["color_mode"]
    active_sh_degree = int(model["active_sh_degree"])
    if mode == "sh":
        if source_mode == "sh":
            active_rest = max(0, sh_coeff_count(active_sh_degree) - 1)
            return (
                model["features_dc"].astype(np.float32),
                active_sh_degree,
                model["features_rest"][:, :active_rest, :].astype(np.float32),
            )
        colors = sigmoid(model["color_logits"])
        return ((colors - 0.5) / SH_C0).astype(np.float32), 0, np.array([], dtype=np.float32)
    if mode == "raw_rgb":
        if source_mode == "sh":
            colors = np.clip(0.5 + SH_C0 * model["features_dc"], 0.0, 1.0)
        else:
            colors = sigmoid(model["color_logits"])
        return colors.astype(np.float32), 0, np.array([], dtype=np.float32)
    raise ValueError(f"Unsupported color mode: {mode}")


def load_model_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    model = {
        "color_mode": str(data["color_mode"]) if "color_mode" in data else "rgb",
        "active_sh_degree": int(data["active_sh_degree"]) if "active_sh_degree" in data else 0,
        "max_sh_degree": int(data["max_sh_degree"]) if "max_sh_degree" in data else 0,
        "means": np.asarray(data["means"], dtype=np.float32).reshape(-1, 3),
        "quats_wxyz": np.asarray(data["quats_wxyz"], dtype=np.float32).reshape(-1, 4),
        "log_scales": np.asarray(data["log_scales"], dtype=np.float32).reshape(-1, 3),
        "opacity_logits": np.asarray(data["opacity_logits"], dtype=np.float32).reshape(-1),
        "summary": json.loads(str(data["summary_json"])) if "summary_json" in data else {},
    }
    if "color_logits" in data:
        model["color_logits"] = np.asarray(data["color_logits"], dtype=np.float32).reshape(-1, 3)
    if "features_dc" in data:
        model["features_dc"] = np.asarray(data["features_dc"], dtype=np.float32).reshape(-1, 3)
    if "features_rest" in data:
        model["features_rest"] = np.asarray(data["features_rest"], dtype=np.float32).reshape(
            model["means"].shape[0],
            -1,
            3,
        )
    return model


def export_one(
    path: Path,
    model: dict,
    position_mode: str,
    scale_mode: str,
    rotation_mode: str,
    quat_order: str,
    color_mode: str,
) -> dict:
    import spz

    cloud = spz.GaussianCloud()
    cloud.antialiased = True
    cloud.positions = positions_to_spz(model["means"], position_mode).reshape(-1).astype(np.float32)
    cloud.scales = scales_to_spz(model["log_scales"], scale_mode).reshape(-1).astype(np.float32)
    quats_wxyz = transform_quats_wxyz(model["quats_wxyz"], rotation_mode)
    cloud.rotations = quats_to_storage(quats_wxyz, quat_order).reshape(-1).astype(np.float32)
    cloud.alphas = model["opacity_logits"].reshape(-1).astype(np.float32)
    colors, sh_degree, sh = colors_to_spz(model, color_mode)
    cloud.colors = colors.reshape(-1).astype(np.float32)
    cloud.sh_degree = int(sh_degree)
    cloud.sh = sh.reshape(-1).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    opts = spz.PackOptions()
    ok = spz.save_spz(cloud, opts, str(path))
    if not ok:
        raise RuntimeError(f"failed to save spz to {path}")

    loaded = spz.load_spz(str(path), spz.UnpackOptions())
    loaded_rot = np.asarray(loaded.rotations, dtype=np.float32).reshape(-1, 4)
    return {
        "path": str(path),
        "position_mode": position_mode,
        "scale_mode": scale_mode,
        "rotation_mode": rotation_mode,
        "quat_order": quat_order,
        "color_mode": color_mode,
        "source_color_mode": model["color_mode"],
        "sh_degree": int(sh_degree),
        "file_size_bytes": path.stat().st_size,
        "num_points": int(model["means"].shape[0]),
        "loaded_rotation_norm_min": float(np.linalg.norm(loaded_rot, axis=1).min()),
        "loaded_rotation_norm_max": float(np.linalg.norm(loaded_rot, axis=1).max()),
    }


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-npz", type=Path, default=Path("outputs/scanner_points_multiview_train/trained_model_params.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/spz_variants"))
    parser.add_argument("--prefix", default="scanner_points")
    parser.add_argument("--position-modes", default="scanner")
    parser.add_argument("--scale-modes", default="direct,scanner_axis")
    parser.add_argument("--rotation-modes", default="direct,position_axis,fastgs_conjugate,position_conjugate")
    parser.add_argument("--quat-orders", default="wxyz,xyzw")
    parser.add_argument("--color-modes", default="sh,raw_rgb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model_npz(args.model_npz)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_npz": str(args.model_npz),
        "model_summary": model["summary"],
        "spz_source_notes": {
            "rotation_storage_conflict": (
                "submodules/spz documents GaussianCloud rotations as xyzw, "
                "while packQuaternionSmallestThree reads the float buffer as Quat4f."
            ),
            "color_storage": "SPZ colors are SH DC coefficients; raw_rgb variants are for debugging only.",
        },
        "variants": [],
    }

    for position_mode in split_csv(args.position_modes):
        for scale_mode in split_csv(args.scale_modes):
            for rotation_mode in split_csv(args.rotation_modes):
                for quat_order in split_csv(args.quat_orders):
                    for color_mode in split_csv(args.color_modes):
                        name = (
                            f"{args.prefix}"
                            f"_pos-{position_mode}"
                            f"_scale-{scale_mode}"
                            f"_rot-{rotation_mode}"
                            f"_q-{quat_order}"
                            f"_color-{color_mode}.spz"
                        )
                        info = export_one(
                            args.out_dir / name,
                            model,
                            position_mode,
                            scale_mode,
                            rotation_mode,
                            quat_order,
                            color_mode,
                        )
                        manifest["variants"].append(info)
                        print(
                            "saved "
                            f"{info['path']} "
                            f"scale={scale_mode} rot={rotation_mode} "
                            f"q={quat_order} color={color_mode}"
                        )

    manifest_path = args.out_dir / f"{args.prefix}_spz_variants_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
