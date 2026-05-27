# Projection backward full GPU path note

## Context

Projection EWA 3DGS forward has a Metal path, but projection backward is still a
CPU reference path.

The current explicit backward API is useful for parity and debugging:

```text
projection_ewa_3dgs_fused_backward(...)
```

However, it should not be treated as the final training graph when the forward
primitive runs on GPU.

## Current limitation

`GSPlatProjectionEWA3DGSFused::vjp(...)` builds a backward primitive from the
forward primitive outputs and cotangents.

The problematic path is:

```text
GPU projection forward primitive
  -> vjp(...)
  -> CPU projection backward primitive
  -> v_means / v_covars / v_quats / v_scales
```

This crosses device/stream boundaries inside MLX autograd. With the projection
`vjp(...)` materialization removed, Python autograd smoke currently produces
zero `v_means` / `v_means3d` even though explicit C++ backward tests pass.

This indicates the issue is not the analytic CPU backward math itself. The
fragile part is the lazy graph dependency between GPU forward outputs and the
CPU backward primitive created inside `vjp(...)`.

## Why pure GPU graph is the target

Other backward-capable primitives in `gsplat_core` keep the backward primitive
on the same stream as the forward primitive:

```text
GSPlatSphericalHarmonics::vjp(...)       -> .s = stream()
GSPlatQuatScaleToCovarPreci::vjp(...)   -> .s = stream()
GSPlatRasterizeToPixels3DGS::vjp(...)   -> .s = stream()
```

That matches the MLX extension style: transformation rules should build follow-
up ops on the primitive stream. A pure GPU path should therefore keep projection
forward and projection backward on the same GPU stream and avoid explicit
`mx::eval(...)` in `vjp(...)`.

## Full GPU gap

The first-pass implementation now exists for the dense pinhole covars path:

```text
GSPlatProjectionEWA3DGSFusedBackward::eval_gpu(...)
gsplat_projection_ewa_3dgs_fused_backward_kernel
```

It supports:

```text
use_covars = true
camera_model = pinhole
calc_compensations = true/false
v_means
v_covars
```

The remaining gap before projection autograd can be treated as a pure GPU
training path is wiring `GSPlatProjectionEWA3DGSFused::vjp(...)` to run the
backward primitive on `stream()` and validating that no `vjp(...)`
materialization is needed.

Until that is validated, projection training still has two imperfect choices:

```text
1. CPU reference backward
   - Good for explicit parity.
   - Not the final GPU training graph.
   - Cross-device autograd is fragile.

2. Force materialization in projection vjp
   - Can make the current smoke pass.
   - Adds synchronization in a place that should ideally only wire graph nodes.
   - Should be treated as a temporary workaround, not the final design.
```

## Recommended implementation slice

Start with dense pinhole covars path only:

```text
camera_model = pinhole
use_covars = true
calc_compensations = true/false
packed = false
```

First GPU outputs:

```text
v_means
v_covars
```

Next outputs after covars GPU path parity is stable:

```text
v_viewmats
v_quats
v_scales
```

Keep these out of the first GPU slice:

```text
Ks gradients
opacity gradients
packed path
ortho / fisheye / ftheta camera models
external distortion
```

## Acceptance checks

The full GPU projection backward path is ready when:

```text
make codex-xcode-test
make pip-install
conda run -n fastgs_core python scripts/test/compare_exported_npz.py
conda run -n fastgs_core python scripts/test/autograd_vjp_smoke.py
make codex-training-smoke
```

all pass with projection `vjp(...)` using `stream()` and without extra
`mx::eval(...)` materialization in the `vjp(...)` layer.
