#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameInfo:
    index: int
    metadata_path: Path


def log(message: str) -> None:
    print(message, flush=True)


def load_frame_infos(data_dir: Path) -> list[FrameInfo]:
    metadata_dir = data_dir / "metadata"
    if not metadata_dir.exists():
        raise FileNotFoundError(f"ScanApp metadata directory not found: {metadata_dir}")
    frames: list[FrameInfo] = []
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        image_rel = raw.get("image")
        depth = raw.get("depth")
        if not isinstance(image_rel, str) or not isinstance(depth, dict):
            continue
        depth_rel = depth.get("path")
        if not isinstance(depth_rel, str):
            continue
        if not (data_dir / image_rel).exists() or not (data_dir / depth_rel).exists():
            continue
        frames.append(FrameInfo(index=int(raw.get("frame_index", len(frames))), metadata_path=metadata_path))
    frames.sort(key=lambda item: (item.index, item.metadata_path.name))
    if not frames:
        raise RuntimeError(f"No usable ScanApp frames found in {data_dir}")
    return frames


def select_frames(frames: list[FrameInfo], max_frames: int, frame_step: int, start_index: int) -> list[FrameInfo]:
    selected = [frame for frame in frames if int(frame.index) >= int(start_index)]
    if frame_step > 1:
        selected = selected[::frame_step]
    if max_frames > 0:
        selected = selected[:max_frames]
    if not selected:
        raise RuntimeError("No ScanApp frames selected")
    return selected


def chunk_frames(frames: list[FrameInfo], chunk_size: int, chunk_stride: int, max_chunks: int) -> list[list[FrameInfo]]:
    chunks: list[list[FrameInfo]] = []
    for start in range(0, len(frames), chunk_stride):
        chunk = frames[start : start + chunk_size]
        if len(chunk) < chunk_size:
            break
        chunks.append(chunk)
        if max_chunks > 0 and len(chunks) >= max_chunks:
            break
    if not chunks:
        raise RuntimeError("No full chunks produced; reduce --chunk-size or selected frame count")
    return chunks


def add_bool_flag(command: list[str], enabled: bool, name: str) -> None:
    command.append(name if enabled else f"--no-{name[2:]}")


def child_command(args: argparse.Namespace, chunk: list[FrameInfo], chunk_dir: Path) -> list[str]:
    first = chunk[0].index
    chunk_spz = chunk_dir / "trained_scanapp_depth_consistency.spz"
    chunk_npz = chunk_dir / "trained_model_params.npz"
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_scanapp_depth_consistency_multiview_3dgs_mlx.py")),
        "--data",
        str(args.data),
        "--out-dir",
        str(chunk_dir),
        "--out-spz",
        str(chunk_spz),
        "--out-model-npz",
        str(chunk_npz),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--target-points",
        str(args.target_points),
        "--max-frames",
        str(args.chunk_size),
        "--frame-step",
        "1",
        "--start-index",
        str(first),
        "--eval-max-frames",
        "0",
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--log-interval",
        str(args.log_interval),
        "--step-image-interval",
        str(args.step_image_interval),
        "--mlx-cache-limit-gb",
        str(args.mlx_cache_limit_gb),
        "--global-scale",
        str(args.global_scale),
        "--mask-min-depth",
        str(args.mask_min_depth),
        "--mask-max-depth",
        str(args.mask_max_depth),
        "--mask-min-confidence",
        str(args.mask_min_confidence),
        "--no-keyframe-filter-enabled",
        "--prior-init-mode",
        args.prior_init_mode,
        "--prior-normal-knn",
        str(args.prior_normal_knn),
        "--prior-tangent-knn",
        str(args.prior_tangent_knn),
        "--prior-normal-scale-ratio",
        str(args.prior_normal_scale_ratio),
        "--prior-tangent-scale-multiplier",
        str(args.prior_tangent_scale_multiplier),
        "--consistency-neighbor-window",
        str(args.consistency_neighbor_window),
        "--consistency-min-views",
        str(args.consistency_min_views),
        "--consistency-abs-depth-tol",
        str(args.consistency_abs_depth_tol),
        "--consistency-rel-depth-tol",
        str(args.consistency_rel_depth_tol),
        "--spz-scale-mode",
        args.spz_scale_mode,
        "--spz-rotation-mode",
        args.spz_rotation_mode,
        "--spz-quat-order",
        args.spz_quat_order,
        "--spz-color-mode",
        args.spz_color_mode,
        "--refine-reset-every",
        str(args.refine_reset_every),
    ]
    add_bool_flag(command, args.consistency_filter_enabled, "--consistency-filter-enabled")
    add_bool_flag(command, args.consistency_keep_unobserved, "--consistency-keep-unobserved")
    add_bool_flag(command, args.refine_enabled, "--refine-enabled")
    return command


def read_child_summary(chunk_dir: Path) -> dict:
    summary_path = chunk_dir / "training_summary.json"
    if not summary_path.exists():
        return {"summary_path": str(summary_path), "summary_found": False}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "summary_path": str(summary_path),
        "summary_found": True,
        "frames": int(summary.get("frames", 0)),
        "steps": int(summary.get("steps", 0)),
        "initial_mean_loss": summary.get("initial_mean_loss"),
        "final_mean_loss": summary.get("final_mean_loss"),
        "exported_gaussians": summary.get("exported_gaussians"),
        "spz": summary.get("spz"),
        "consistency_filter": summary.get("consistency_filter"),
        "frame_indices": summary.get("keyframe_filter", {}).get("selected_frame_indices"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/scanapp_depth_chunked_consistency_train"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--target-points", type=int, default=262_144)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--chunk-stride", type=int, default=4)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--step-image-interval", type=int, default=0)
    parser.add_argument("--mlx-cache-limit-gb", type=float, default=24.0)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--mask-min-depth", type=float, default=0.05)
    parser.add_argument("--mask-max-depth", type=float, default=5.0)
    parser.add_argument("--mask-min-confidence", type=int, default=1)
    parser.add_argument("--prior-init-mode", choices=("disc", "knn"), default="disc")
    parser.add_argument("--prior-normal-knn", type=int, default=16)
    parser.add_argument("--prior-tangent-knn", type=int, default=3)
    parser.add_argument("--prior-normal-scale-ratio", type=float, default=0.2)
    parser.add_argument("--prior-tangent-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--consistency-filter-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--consistency-neighbor-window", type=int, default=2)
    parser.add_argument("--consistency-min-views", type=int, default=2)
    parser.add_argument("--consistency-abs-depth-tol", type=float, default=0.04)
    parser.add_argument("--consistency-rel-depth-tol", type=float, default=0.015)
    parser.add_argument("--consistency-keep-unobserved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spz-scale-mode", choices=("direct", "scanner_axis"), default="direct")
    parser.add_argument("--spz-rotation-mode", choices=("direct", "position_axis", "fastgs_conjugate", "position_conjugate"), default="position_axis")
    parser.add_argument("--spz-quat-order", choices=("wxyz", "xyzw"), default="xyzw")
    parser.add_argument("--spz-color-mode", choices=("sh", "raw_rgb"), default="sh")
    parser.add_argument("--refine-reset-every", type=int, default=999999)
    parser.add_argument("--refine-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.chunk_stride <= 0:
        raise ValueError("--chunk-stride must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    selected = select_frames(load_frame_infos(args.data), args.max_frames, args.frame_step, args.start_index)
    chunks = chunk_frames(selected, args.chunk_size, args.chunk_stride, args.max_chunks)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunk_results = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_name = f"chunk_{chunk_index:03d}_{chunk[0].index:05d}_{chunk[-1].index:05d}"
        chunk_dir = args.out_dir / chunk_name
        command = child_command(args, chunk, chunk_dir)
        chunk_result = {
            "chunk_index": int(chunk_index),
            "chunk_dir": str(chunk_dir),
            "frame_indices": [int(frame.index) for frame in chunk],
            "command": command,
        }
        log(f"chunk={chunk_index:03d} frames={chunk_result['frame_indices']} out={chunk_dir}")
        if args.dry_run:
            log("dry-run command: " + " ".join(command))
            chunk_result["returncode"] = None
        else:
            completed = subprocess.run(command, check=False)
            chunk_result["returncode"] = int(completed.returncode)
            chunk_result.update(read_child_summary(chunk_dir))
            if completed.returncode != 0 and not args.keep_going:
                chunk_results.append(chunk_result)
                break
        chunk_results.append(chunk_result)

    final_losses = [
        item["final_mean_loss"]
        for item in chunk_results
        if item.get("returncode") == 0 and isinstance(item.get("final_mean_loss"), (int, float))
    ]
    summary = {
        "dataset_type": "scanapp_depth_chunked_consistency",
        "dataset": str(args.data),
        "selected_frame_count": int(len(selected)),
        "selected_frame_indices": [int(frame.index) for frame in selected],
        "chunk_size": int(args.chunk_size),
        "chunk_stride": int(args.chunk_stride),
        "chunk_count": int(len(chunks)),
        "completed_chunk_count": int(sum(1 for item in chunk_results if item.get("returncode") == 0)),
        "failed_chunk_count": int(sum(1 for item in chunk_results if item.get("returncode") not in (0, None))),
        "final_loss_min": min(final_losses) if final_losses else None,
        "final_loss_mean": sum(final_losses) / len(final_losses) if final_losses else None,
        "final_loss_max": max(final_losses) if final_losses else None,
        "chunks": chunk_results,
    }
    summary_path = args.out_dir / "chunked_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote chunked summary path={summary_path}")
    if summary["failed_chunk_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
