#!/usr/bin/env python3

from __future__ import annotations

import json

from scanner_points_training_utils import (
    lr_for_step,
    make_lr_schedule,
    validate_lr_schedule_args,
)


def build_schedule(name: str, initial: float, final: float | None, delay: float, max_steps: int) -> dict:
    validate_lr_schedule_args(name, initial, final, delay, max_steps)
    return make_lr_schedule(initial, final, delay, max_steps)


def sample_schedule(schedule: dict, steps: tuple[int, ...]) -> list[dict]:
    out = []
    for step in steps:
        lr = lr_for_step(schedule, step)
        schedule["latest"] = lr
        schedule["history"].append({"step": step, "lr": lr})
        out.append({"step": step, "lr": lr})
    return out


def main() -> None:
    steps = (1, 50, 100)
    schedules = {
        "means": build_schedule("means", 2.0e-3, 2.0e-4, 1.0, 100),
        "features_dc": build_schedule("colors", 2.0e-2, 5.0e-3, 1.0, 100),
        "features_rest": build_schedule("sh-rest", 1.0e-3, 1.0e-4, 0.5, 100),
        "opacity_logits": build_schedule("opacity", 5.0e-3, 1.0e-3, 1.0, 100),
        "log_scales": build_schedule("scales", 1.0e-3, 5.0e-4, 1.0, 100),
        "quats": build_schedule("quats", 1.0e-3, None, 1.0, 100),
    }

    samples = {name: sample_schedule(schedule, steps) for name, schedule in schedules.items()}
    print("=== optimizer LR schedule debug ===")
    print(json.dumps(samples, indent=2, sort_keys=True))

    for name in ("means", "features_dc", "features_rest", "opacity_logits", "log_scales"):
        first = samples[name][0]["lr"]
        last = samples[name][-1]["lr"]
        if not last < first:
            raise AssertionError(f"{name} expected decayed LR: first={first} last={last}")

    quat_lrs = [item["lr"] for item in samples["quats"]]
    if len(set(quat_lrs)) != 1:
        raise AssertionError(f"quats expected fixed LR when final is None: {quat_lrs}")

    if samples["features_rest"][0]["lr"] >= schedules["features_rest"]["initial"]:
        raise AssertionError("features_rest delay_mult should lower the early LR sample")

    print("optimizer LR schedule debug ok")


if __name__ == "__main__":
    main()
