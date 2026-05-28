#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass

from train_scanner_points_multiview_3dgs_mlx import FrameBatchSampler


@dataclass
class FakeCamera:
    index: int


def collect_batches(frame_count: int, batch_size: int, mode: str, seed: int, steps: int) -> tuple[list[list[int]], dict]:
    cameras = [FakeCamera(index=100 + idx) for idx in range(frame_count)]
    sampler = FrameBatchSampler(frame_count, batch_size, mode, seed)
    batches = [sampler.next_batch() for _ in range(steps)]
    return batches, sampler.summary(cameras)


def main() -> None:
    sequential_batches, sequential_summary = collect_batches(
        frame_count=5,
        batch_size=2,
        mode="sequential",
        seed=7,
        steps=4,
    )
    shuffle_a, shuffle_summary_a = collect_batches(
        frame_count=5,
        batch_size=2,
        mode="shuffle",
        seed=11,
        steps=4,
    )
    shuffle_b, shuffle_summary_b = collect_batches(
        frame_count=5,
        batch_size=2,
        mode="shuffle",
        seed=11,
        steps=4,
    )
    shuffle_c, _ = collect_batches(
        frame_count=5,
        batch_size=2,
        mode="shuffle",
        seed=12,
        steps=4,
    )

    report = {
        "sequential_batches": sequential_batches,
        "sequential_summary": sequential_summary,
        "shuffle_batches_seed_11": shuffle_a,
        "shuffle_summary_seed_11": shuffle_summary_a,
        "shuffle_batches_seed_12": shuffle_c,
    }
    print("=== scanner dataloader debug ===")
    print(json.dumps(report, indent=2, sort_keys=True))

    if sequential_batches != [[0, 1], [2, 3], [4, 0], [1, 2]]:
        raise AssertionError(f"unexpected sequential batches: {sequential_batches}")
    if shuffle_a != shuffle_b or shuffle_summary_a["sampled_batches"] != shuffle_summary_b["sampled_batches"]:
        raise AssertionError("shuffle sampler should be deterministic for the same seed")
    if shuffle_a == shuffle_c:
        raise AssertionError("different shuffle seeds should change the sampled order in this fixture")
    if min(sequential_summary["usage_counts"]) <= 0:
        raise AssertionError("sequential sampler should cover every frame in this fixture")
    if sequential_summary["usage_count_max"] - sequential_summary["usage_count_min"] > 1:
        raise AssertionError("sequential sampler usage should stay balanced")
    if shuffle_summary_a["batch_size"] != 2 or shuffle_summary_a["mode"] != "shuffle":
        raise AssertionError("shuffle summary should record mode and batch size")

    print("scanner dataloader debug ok")


if __name__ == "__main__":
    main()
