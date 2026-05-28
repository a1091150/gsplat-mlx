#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("refs") / "training_reference_config.json",
        help="Output JSON config path for CUDA/gsplat training reference runs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Users/yangdunfu/Downloads/2026_05_04_16_51_29",
        help="Scanner dataset path to record in the reference config.",
    )
    return parser.parse_args()


def reference_config(dataset: str) -> dict:
    return {
        "schema": "gsplat_core_training_reference_config_v1",
        "purpose": (
            "Fixed-seed CUDA gsplat training reference contract for comparing "
            "longer training behavior against scripts/test/"
            "train_scanner_points_multiview_3dgs_mlx.py summaries."
        ),
        "dataset": dataset,
        "seed": 37,
        "image": {
            "width": 512,
            "height": 512,
        },
        "frames": {
            "max_frames": 66,
            "frame_step": 1,
            "start_index": 0,
            "eval_max_frames": 0,
            "eval_frame_step": 1,
            "eval_start_index": None,
        },
        "dataloader": {
            "frame_sampling": "shuffle",
            "frame_shuffle_seed": 7956,
            "batch_size": 1,
            "summary_fields": [
                "sampled_batches",
                "sampled_frame_indices",
                "usage_counts",
                "completed_epochs",
            ],
        },
        "model": {
            "max_points": 700000,
            "num_random_gaussians": 0,
            "random_gaussian_bounds_scale": 1.05,
            "color_mode": "sh",
            "sh_degree": 0,
            "max_sh_degree": 3,
            "sh_degree_start": 0,
            "sh_degree_target": 3,
            "sh_degree_schedule_interval": 1000,
            "opacity": 0.65,
            "point_scale_mode": "scene_fraction",
            "point_scale_fraction": 0.005,
            "refine_scene_scale_mode": "points_extent",
        },
        "training": {
            "steps": 10000,
            "loss_mode": "l1_dssim",
            "ssim_lambda": 0.2,
            "ssim_window_size": 11,
        },
        "learning_rate_schedule": {
            "means": {
                "initial": 0.002,
                "final": 0.0002,
                "delay_mult": 1.0,
                "max_steps": 10000,
            },
            "features_dc": {
                "initial": 0.02,
                "final": 0.005,
                "delay_mult": 1.0,
                "max_steps": 10000,
            },
            "features_rest": {
                "initial": 0.001,
                "final": 0.0001,
                "delay_mult": 1.0,
                "max_steps": 10000,
            },
            "opacity_logits": {
                "initial": 0.005,
                "final": 0.001,
                "delay_mult": 1.0,
                "max_steps": 10000,
            },
            "log_scales": {
                "initial": 0.001,
                "final": 0.0005,
                "delay_mult": 1.0,
                "max_steps": 10000,
            },
            "quats": {
                "initial": 0.001,
                "final": 0.0001,
                "delay_mult": 1.0,
                "max_steps": 10000,
            },
        },
        "refinement": {
            "enabled": False,
            "prune_opa": 0.005,
            "grow_grad2d": 0.0002,
            "grow_scale3d": 0.01,
            "grow_scale2d": 0.05,
            "prune_scale3d": 0.1,
            "prune_scale2d": 0.15,
            "refine_scale2d_stop_iter": 0,
            "refine_start_iter": 500,
            "refine_stop_iter": 15000,
            "reset_every": 3000,
            "refine_every": 100,
            "pause_refine_after_reset": 0,
            "absgrad": False,
            "revised_opacity": False,
        },
        "summary_contract": {
            "required_common_fields": [
                "dataset",
                "width",
                "height",
                "frames",
                "steps",
                "loss_function",
                "initial_mean_loss",
                "final_mean_loss",
                "dataloader",
                "learning_rate_schedule",
                "refinement_strategy",
                "frame_summaries",
            ],
            "recommended_cuda_summary_path": "refs/training_reference_summary_cuda.json",
            "recommended_mlx_summary_path": (
                "outputs/scanner_points_multiview_train/training_summary.json"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    config = reference_config(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
