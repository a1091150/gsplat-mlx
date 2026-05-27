#!/usr/bin/env python3

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from train_scanner_points_multiview_3dgs_mlx import image_loss_components


def scalar(value: mx.array) -> float:
    mx.eval(value)
    return float(np.asarray(value))


def summarize_components(image: mx.array, target: mx.array, mode: str) -> dict:
    components = image_loss_components(
        image,
        target,
        loss_mode=mode,
        ssim_lambda=0.2,
        ssim_window_size=11,
    )
    mx.eval(components["loss"], components["l1"], components["ssim"], components["dssim"])
    return {name: float(np.asarray(value)) for name, value in components.items()}


def main() -> None:
    rng = np.random.default_rng(20280528)
    base = rng.uniform(0.05, 0.95, size=(1, 32, 32, 3)).astype(np.float32)
    shifted = np.roll(base, shift=2, axis=2).copy()
    noisy = np.clip(base + rng.normal(0.0, 0.08, size=base.shape).astype(np.float32), 0.0, 1.0)

    image = mx.array(base)
    same = mx.array(base)
    shifted_target = mx.array(shifted)
    noisy_target = mx.array(noisy)

    same_l1 = summarize_components(image, same, "l1")
    same_mixed = summarize_components(image, same, "l1_dssim")
    shifted_mixed = summarize_components(image, shifted_target, "l1_dssim")
    noisy_mixed = summarize_components(image, noisy_target, "l1_dssim")

    def loss_fn(x: mx.array) -> mx.array:
        return image_loss_components(
            x,
            noisy_target,
            loss_mode="l1_dssim",
            ssim_lambda=0.2,
            ssim_window_size=11,
        )["loss"]

    loss, grad = mx.value_and_grad(loss_fn)(image)
    mx.eval(loss, grad)
    grad_norm = float(np.linalg.norm(np.asarray(grad)))

    report = {
        "same_l1": same_l1,
        "same_l1_dssim": same_mixed,
        "shifted_l1_dssim": shifted_mixed,
        "noisy_l1_dssim": noisy_mixed,
        "grad_loss": scalar(loss),
        "grad_norm": grad_norm,
    }
    print("=== image loss debug ===")
    print(json.dumps(report, indent=2, sort_keys=True))

    if abs(same_l1["loss"]) > 1.0e-6:
        raise AssertionError(f"identical l1 loss should be near zero: {same_l1['loss']}")
    if abs(same_mixed["loss"]) > 1.0e-5:
        raise AssertionError(f"identical l1_dssim loss should be near zero: {same_mixed['loss']}")
    if shifted_mixed["loss"] <= same_mixed["loss"]:
        raise AssertionError("shifted image should have larger mixed loss than identical image")
    if noisy_mixed["loss"] <= same_mixed["loss"]:
        raise AssertionError("noisy image should have larger mixed loss than identical image")
    if grad_norm <= 0.0:
        raise AssertionError("l1_dssim loss should produce a nonzero gradient for noisy target")

    print("image loss debug ok")


if __name__ == "__main__":
    main()
