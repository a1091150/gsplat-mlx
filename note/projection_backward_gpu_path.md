# Projection backward full GPU path note

## Context

Projection EWA 3DGS forward has a Metal path, and projection backward now has a
first-pass Metal path for dense covars + pinhole.

The current explicit backward API is useful for parity and debugging:

```text
projection_ewa_3dgs_fused_backward(...)
```

However, it should not be treated as the final training graph when the forward
primitive runs on GPU.

## Resolved routing

`GSPlatProjectionEWA3DGSFused::vjp(...)` builds a backward primitive from the
forward primitive outputs and cotangents.

The supported full-GPU path is now:

```text
GPU projection forward primitive
  -> vjp(...)
  -> GPU projection backward primitive on stream()
  -> v_means / v_covars / v_viewmats
```

This path is selected only when the current Metal backward implementation
supports the request:

```text
use_covars = true
or quats/scales input
camera_model = pinhole
```

Unsupported requests still use the CPU reference path for now. The CPU
reference path remains useful for parity and debugging, but it is not the final
GPU training graph because it can cross device/stream boundaries inside MLX
autograd.

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

## Full GPU status

The current implementation now covers the dense fused pinhole path:

```text
GSPlatProjectionEWA3DGSFusedBackward::eval_gpu(...)
gsplat_projection_ewa_3dgs_fused_backward_kernel
```

It supports:

```text
use_covars = true
or quats/scales input
camera_model = pinhole
calc_compensations = true/false
v_means
v_covars
v_quats
v_scales
v_viewmats
```

The CUDA implementation uses warp-level reduction followed by `gpuAtomicAdd`
for `v_viewmats`, because all gaussians for the same `(batch, camera)` add into
the same view matrix. The first MLX/Metal implementation keeps the same
reduction semantics but assigns one thread per `(batch, camera)` and loops over
gaussians, which avoids relying on floating-point atomics while parity is being
locked down.

`GSPlatProjectionEWA3DGSFused::vjp(...)` now routes the supported dense fused
pinhole path to `stream()`. The projection autograd and dense training smoke
checks pass without extra `mx::eval(...)` materialization in the `vjp(...)`
layer.

Unsupported projection training requests still have two imperfect choices:

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

## Support matrix

Full GPU projection VJP currently means:

```text
projection forward GPU
  -> projection vjp(...)
  -> projection backward GPU on stream()
```

Supported:

```text
dense covars input
quat/scale input
pinhole camera_model = 0
calc_compensations = true/false
v_means
v_covars
v_quats
v_scales
v_viewmats
viewspace_points gradient proxy
```

Fallback or not implemented:

```text
non-pinhole cameras
packed projection
Ks gradients
opacity gradients
external distortion paths
```

Use `scripts/test/projection_vjp_guardrails.py` or
`make codex-projection-guardrails` to verify the supported boundary and print
the expected limitations.

## Completed implementation slice

Dense fused pinhole projection backward currently supports:

```text
camera_model = pinhole
use_covars = true
or quats/scales input
calc_compensations = true/false
packed = false
```

GPU outputs:

```text
v_means
v_covars
v_quats
v_scales
```

Next outputs after dense fused GPU path parity is stable:

```text
packed projection
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
