# Temporary Metal/CUDA Low-Level Parity Tasks

## Purpose
- Track Metal/CUDA low-level operator parity separately from `Task.md`.
- Keep scanner after-training strategy work from absorbing Metal kernel scope.
- This file is temporary; merge or fold it back into `Task.md` manually when
  the workstreams stabilize.

## Scope Boundary
- This file owns low-level CUDA-to-Metal semantic gaps in `gsplat_core/metal`
  and related C++ primitive guardrails.
- `Task.md` owns trainer/strategy behavior such as densify, clone, split,
  prune, optimizer schedules, dataloader behavior, and scanner diagnostics.
- If training needs a new low-level behavior, record the need in `Task.md`, then
  add the Metal/CUDA implementation task here before changing kernels.

## Notes On Packed And Segmented Paths
- `packed` in gsplat CUDA means using a compact `[nnz, ...]` layout instead of
  dense `[I, N, ...]` or `[B, C, N, ...]` tensors. The packed path carries
  `image_ids` and sometimes `gaussian_ids` so kernels process only active or
  sparse Gaussian entries.
- For the current dense MLX training path, packed support is mainly a memory /
  performance and sparse-workflow feature, not a correctness blocker.
- Do not port packed paths unless dense-only training becomes insufficient or a
  future high-level API requires packed inputs.
- Segmented sort is also not a current priority because the MLX path already
  has a replacement sort/reorder implementation for the dense staged path.

## Priority Order
1. Spherical harmonics backward analytic `v_dirs`.
2. Quat/scale covariance/precision backward analytic VJP.
3. Projection backward non-pinhole coverage.
4. Intersect AccuTile/SNUGBOX coverage.

## Task M1 - Spherical Harmonics Backward Analytic VJP

### Finding
- CUDA uses analytic `sh_coeffs_to_color_fast_vjp` in
  `submodules/gsplat/gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu` around line
  488.
- Metal previously computed `v_dirs` with centered finite differences using
  `eps = 1e-3` in
  `gsplat_core/metal/gsplat_spherical_harmonics.metal` around line 244.
- This can introduce small gradient differences during training, even though
  existing smoke and fixture tests cover the current approximation.

### Tasks
- [x] Port CUDA-style analytic SH direction VJP to Metal.
- [x] Keep `compute_v_dirs=false` behavior unchanged.
- [ ] Add or update CUDA `.npz` fixtures for degree 0 through degree 4,
  including masks and `compute_v_dirs=true`.
- [x] Tighten comparer tolerances only after analytic parity is validated.

## Task M2 - Quat/Scale Covariance/Precision Analytic Backward

### Finding
- CUDA uses analytic covariance and precision VJPs in
  `submodules/gsplat/gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu` around line 181
  and helper functions in `submodules/gsplat/gsplat/cuda/include/Utils.cuh`.
- Metal currently uses centered finite differences with `eps = 1e-3` in
  `gsplat_core/metal/gsplat_quat_scale_to_covar.metal` around line 190.
- This appears to be an intentional first-version simplification, but it is not
  CUDA-equivalent and may be slower or less stable in long training runs.

### Tasks
- [x] Port CUDA-style analytic `quat_scale_to_covar_vjp` to Metal.
- [x] Port CUDA-style analytic `quat_scale_to_preci_vjp` to Metal.
- [x] Preserve `triu=true` and `triu=false` cotangent layout semantics.
- [x] Preserve optional `v_covars` and `v_precis` behavior.
- [x] Add edge-case fixtures for non-unit quaternions, precision-only output,
  and full `3x3` cotangents.

## Task M3 - Projection Backward Non-Pinhole Coverage

### Finding
- CUDA projection backward dispatches by `camera_model` and calls
  `persp_proj_vjp`, `ortho_proj_vjp`, or `fisheye_proj_vjp` in
  `submodules/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu` around line
  429.
- Metal backward currently uses `persp_proj_vjp` in
  `gsplat_core/metal/gsplat_projection.metal` around line 538.
- C++ validation rejects non-pinhole backward around
  `gsplat_core/gsplat_projection.cpp` line 157, so this is an explicit scope
  limit rather than a silent bug.
- Projection forward already has pinhole, ortho, and fisheye branches and is
  closer to CUDA forward behavior.

### Tasks
- [ ] Keep non-pinhole projection backward guarded until Metal VJPs are ported.
- [ ] Port `ortho_proj_vjp` to Metal if non-pinhole training becomes required.
- [ ] Port `fisheye_proj_vjp` to Metal if fisheye training becomes required.
- [ ] Add CUDA `.npz` fixtures for ortho and fisheye projection backward before
  removing guardrails.

## Task M4 - Intersect AccuTile/SNUGBOX Coverage

### Finding
- CUDA supports `conics + opacities` AccuTile / SNUGBOX ellipse tile
  intersection in `submodules/gsplat/gsplat/cuda/csrc/IntersectTile.cu` around
  line 195.
- Metal currently implements only the radius AABB fallback in
  `gsplat_core/metal/gsplat_intersect.metal` around line 50.
- C++ validation explicitly rejects `use_conics`, `use_opacities`, `packed`,
  and `segmented` around `gsplat_core/gsplat_intersect.cpp` line 114, so this is
  not a silent fallback.

### Tasks
- [ ] Do not port AccuTile unless dense AABB tile coverage becomes insufficient
  for quality or performance.
- [ ] If needed, port the CUDA ellipse tile intersection path using `conics`
  and `opacities`.
- [ ] Add CUDA `.npz` fixtures that compare tile counts, encoded `isect_ids`,
  `flatten_ids`, and offsets for AccuTile.
- [ ] Keep packed and segmented behavior out of scope for this task unless a
  separate requirement promotes them.
