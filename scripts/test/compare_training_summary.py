#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlx",
        type=Path,
        default=ROOT / "outputs" / "scanner_points_multiview_train" / "training_summary.json",
        help="MLX training_summary.json path.",
    )
    parser.add_argument(
        "--cuda",
        type=Path,
        default=ROOT / "refs" / "training_reference_summary_cuda.json",
        help="CUDA gsplat training summary JSON path.",
    )
    parser.add_argument(
        "--loss-rtol",
        type=float,
        default=0.25,
        help="Relative tolerance for initial/final loss comparisons.",
    )
    parser.add_argument(
        "--loss-atol",
        type=float,
        default=1.0e-3,
        help="Absolute tolerance for initial/final loss comparisons.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"skip: missing {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_path(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def approx_equal(a: Any, b: Any, atol: float, rtol: float) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


def compare_scalar(name: str, mlx: dict[str, Any], cuda: dict[str, Any], atol: float = 0.0, rtol: float = 0.0) -> bool:
    a = get_path(mlx, name)
    b = get_path(cuda, name)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        ok = approx_equal(a, b, atol, rtol) if (atol or rtol) else a == b
    else:
        ok = a == b
    print(f"{'ok' if ok else 'mismatch'}: {name}: mlx={a!r} cuda={b!r}")
    return ok


def compare_list_prefix(name: str, mlx: dict[str, Any], cuda: dict[str, Any], count: int = 8) -> bool:
    a = get_path(mlx, name, [])
    b = get_path(cuda, name, [])
    a_prefix = a[:count] if isinstance(a, list) else []
    b_prefix = b[:count] if isinstance(b, list) else []
    ok = a_prefix == b_prefix
    print(f"{'ok' if ok else 'mismatch'}: {name} prefix: mlx={a_prefix!r} cuda={b_prefix!r}")
    return ok


def compare_lr_history(mlx: dict[str, Any], cuda: dict[str, Any]) -> list[bool]:
    results = []
    mlx_lr = get_path(mlx, "learning_rate_schedule", {})
    cuda_lr = get_path(cuda, "learning_rate_schedule", {})
    names = sorted(set(mlx_lr) & set(cuda_lr))
    if not names:
        print("skip: no shared learning_rate_schedule entries")
        return results
    for name in names:
        for field in ("initial", "final", "delay_mult", "max_steps", "latest"):
            a = get_path(mlx_lr[name], field)
            b = get_path(cuda_lr[name], field)
            if a is None and b is None:
                ok = True
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                ok = approx_equal(a, b, 1.0e-8, 1.0e-6)
            else:
                ok = a == b
            print(f"{'ok' if ok else 'mismatch'}: lr.{name}.{field}: mlx={a!r} cuda={b!r}")
            results.append(ok)
    return results


def compare_summaries(mlx: dict[str, Any], cuda: dict[str, Any], loss_atol: float, loss_rtol: float) -> list[bool]:
    results = [
        compare_scalar("width", mlx, cuda),
        compare_scalar("height", mlx, cuda),
        compare_scalar("frames", mlx, cuda),
        compare_scalar("steps", mlx, cuda),
        compare_scalar("loss_function", mlx, cuda),
        compare_scalar("color_mode", mlx, cuda),
        compare_scalar("initial_mean_loss", mlx, cuda, atol=loss_atol, rtol=loss_rtol),
        compare_scalar("final_mean_loss", mlx, cuda, atol=loss_atol, rtol=loss_rtol),
        compare_scalar("refinement_strategy.gaussian_delta", mlx, cuda),
        compare_scalar("refinement_strategy.operation_totals.clone", mlx, cuda),
        compare_scalar("refinement_strategy.operation_totals.split", mlx, cuda),
        compare_scalar("refinement_strategy.operation_totals.prune", mlx, cuda),
        compare_scalar("refinement_strategy.operation_totals.opacity_reset", mlx, cuda),
        compare_scalar("dataloader.mode", mlx, cuda),
        compare_scalar("dataloader.batch_size", mlx, cuda),
        compare_list_prefix("dataloader.sampled_batches", mlx, cuda),
        compare_list_prefix("dataloader.sampled_frame_indices", mlx, cuda),
    ]
    results.extend(compare_lr_history(mlx, cuda))
    return results


def main() -> None:
    args = parse_args()
    mlx = load_json(args.mlx)
    cuda = load_json(args.cuda)
    if mlx is None or cuda is None:
        return
    results = compare_summaries(mlx, cuda, args.loss_atol, args.loss_rtol)
    passed = sum(bool(item) for item in results)
    total = len(results)
    print(f"training summary comparison: passed={passed} total={total}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
