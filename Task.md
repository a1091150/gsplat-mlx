# gsplat 3DGS MLX Migration Tasks

## Project Scope
- Source project: `submodules/gsplat`.
- Source CUDA code: `submodules/gsplat/gsplat/cuda`.
- Target package: `gsplat_core`.
- Target direction: gsplat CUDA/PyTorch extension to MLX Metal extension.
- Primary milestone: 3DGS forward low-level ops.
- Conda environment: `fastgs_core`.

## Explicitly Out of Scope
- 2DGS operators and rendering paths.
- Backward/autograd kernels.
- UT / rolling-shutter / world-ray rasterization paths.
- Lidar operators.
- Adam, relocation, MCMC perturb, camera wrappers, and external distortion.
- Python test automation targets in Makefile. Python tests remain manual scripts.

## Directory and Naming Rules
- C++ headers: `gsplat_core/include/gsplat_<op>.h`.
- C++ implementation: `gsplat_core/gsplat_<op>.cpp`.
- Metal kernels: `gsplat_core/metal/gsplat_<op>.metal`.
- Binding entry: `gsplat_core/binding/binding.cpp`.
- Python package: `python_package/gsplat_core`.
- Manual Python scripts: `scripts/test`.
- C++ namespace: `gsplat_core`.
- Python extension module: `_gsplat_core`.
- Do not introduce `fastgs` names for new gsplat migration files, symbols, or APIs.

## Build and Install Entry Points
- `make env-check`: validate conda Python, CMake, MLX, nanobind, and MLX CMake package path.
- `make xcode-build`: configure and build `_gsplat_core` with Xcode using the Python `mlx` package CMake config.
- `make pip-install`: install package with `pip install . --no-build-isolation`.
- `make pip-develop`: editable install with `pip install -e . --no-build-isolation`.
- `make clean`: remove local build and Python packaging artifacts.

## MLX Training Gradient Rule
- PyTorch can use `retain_grad()` on intermediate tensors such as `means2d`.
- MLX gradients are produced by `mx.value_and_grad(fn, argnums=...)`, so only selected function arguments receive gradients.
- If gsplat training or after-training logic needs intermediate gradients such as screen-space `means2d` gradients, use an explicit dummy trainable parameter as the gradient carrier.
- Prefer the FastGS MLX pattern and name `viewspace_points` when the behavior mirrors FastGS.
- The dummy gradient proxy must be a visible argument to the loss function and must be included in `value_and_grad(..., argnums=...)`.
- Related note: `note/mlx_training_gradient_proxy.md`.

---

# Task 1 - Project/Build Skeleton

## Scope
- Establish the minimal gsplat_core build skeleton before CUDA op migration.
- Keep the root build entry style aligned with the FastGS MLX migration project.
- Use only the Python `mlx` package for MLX CMake configuration.

## Completed
- [x] Add root `CMakeLists.txt`.
- [x] Add root `Makefile`.
- [x] Add `setup.py` and `pyproject.toml`.
- [x] Add C++ source layout:
  - [x] `gsplat_core/include/`
  - [x] `gsplat_core/metal/`
  - [x] `gsplat_core/binding/`
  - [x] `gsplat_core/test/`
- [x] Add dummy C++ source/header/test.
- [x] Add nanobind module `_gsplat_core`.
- [x] Add Python package directory `python_package/gsplat_core`.
- [x] Add manual dummy script under `scripts/test`.
- [x] Remove FastGS naming from project, target, module, and package identifiers.

## Validation Completed
- [x] `make env-check`.
- [x] `make xcode-build`.
- [x] C++ dummy target builds.
- [x] `_gsplat_core` builds through Xcode.

## Notes
- The current implementation is intentionally dummy-only.
- Python MLX runtime checks are left to manual local testing.

---

# Task 2 - CUDA Source Map

## Scope
- Identify the gsplat CUDA 3DGS forward operators to migrate.
- Record the source files and exported torch ops before implementing any op.

## Completed
- [x] Add detailed source map: `note/gsplat_cuda_source_map.md`.
- [x] Add MLX training gradient proxy rule: `note/mlx_training_gradient_proxy.md`.
- [x] Map Python wrapper, torch op schema, C++ launcher, CUDA kernel, and target MLX files.
- [x] Confirm the next implementation slice should continue projection parity before new ops.

## Source Files
- `submodules/gsplat/gsplat/cuda/ext.cpp`
- `submodules/gsplat/gsplat/cuda/_wrapper.py`
- `submodules/gsplat/gsplat/cuda/_torch_impl.py`
- `submodules/gsplat/gsplat/cuda/csrc/QuatScaleToCovar.cpp`
- `submodules/gsplat/gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu`
- `submodules/gsplat/gsplat/cuda/csrc/SphericalHarmonics.cpp`
- `submodules/gsplat/gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu`
- `submodules/gsplat/gsplat/cuda/csrc/Projection.cpp`
- `submodules/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu`
- `submodules/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu`
- `submodules/gsplat/gsplat/cuda/csrc/Intersect.cpp`
- `submodules/gsplat/gsplat/cuda/csrc/IntersectTile.cu`
- `submodules/gsplat/gsplat/cuda/csrc/Rasterization.cpp`
- `submodules/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu`
- `submodules/gsplat/gsplat/cuda/csrc/RasterizeToIndices3DGS.cu`

## 3DGS Forward Ops
- [x] `quat_scale_to_covar_preci_fwd`
- [x] `spherical_harmonics_fwd`
- [x] `projection_ewa_3dgs_fused_fwd`
- [ ] `projection_ewa_3dgs_packed_fwd`
- [x] `intersect_tile`
- [x] `intersect_offset`
- [x] `rasterize_to_pixels_3dgs_fwd`
- [ ] `rasterize_to_indices_3dgs`

## Excluded CUDA Ops
- [ ] `projection_2dgs_fused_fwd`
- [ ] `projection_2dgs_packed_fwd`
- [ ] `rasterize_to_pixels_2dgs_fwd`
- [ ] `rasterize_to_indices_2dgs`
- [ ] `projection_ut_3dgs_fused`
- [ ] `rasterize_to_pixels_from_world_3dgs_fwd`
- [ ] `intersect_tile_lidar`

---

# Task 3 - Forward Low-Level Ops

## Scope
- Migrate 3DGS forward low-level ops one at a time.
- Use MLX Primitive C++ wrappers and Metal kernels, following the FastGS MLX migration style.
- Keep APIs close to gsplat CUDA op semantics while using MLX arrays.

## Planned Subtasks
- [ ] Task 3.1: Projection 3DGS fused forward numeric parity.
- [x] Task 3.2: Intersect tile / intersect offset forward.
- [x] Task 3.3: Rasterize to pixels 3DGS forward.
- [x] Task 3.4: Spherical harmonics forward.
- [x] Task 3.5: Quat/scale to covariance/precision forward.
- [x] Task 3.6: End-to-end 3DGS forward smoke chain.
- [x] Task 3.7: CUDA/PyTorch parity reference scripts.
- [x] Task 3.8: CUDA reference `.npz` export scripts.
- [x] Task 3.9: Spherical harmonics MLX Primitive + Metal kernel.
- [x] Task 3.10: Quat/scale covariance/precision MLX Primitive + Metal kernel.
- [x] Task 3.11A: Intersect tile count MLX Primitive + Metal kernel.
- [x] Task 3.11B: Intersect offset MLX Primitive + Metal kernel.

## Implementation Rules
- Each op gets a header, C++ implementation, and Metal kernel file.
- Each op exposes one migration-friendly C++ function in namespace `gsplat_core`.
- Binding functions should use clear low-level names based on gsplat CUDA op names.
- CPU fallback may throw or return zero-filled placeholders until a CPU path is explicitly needed.
- GPU path is the source of truth for migrated behavior.
- SH degree forward path and Task 3.11A relationship are recorded in `note/gsplat_3dgs_forward_sh_path.md`.

## Task 3.1 - Projection 3DGS Fused Forward
- [x] Add `gsplat_core/include/gsplat_projection.h`.
- [x] Add `gsplat_core/gsplat_projection.cpp`.
- [x] Add `gsplat_core/metal/gsplat_projection.metal`.
- [x] Expose `projection_ewa_3dgs_fused_forward(...)` from `_gsplat_core`.
- [x] Support covars path or quats/scales path.
- [x] Support optional opacities and optional compensations output.
- [x] Build through `make xcode-build`.
- [x] Add C++ smoke coverage through `make codex-xcode-test`.
- [x] Add fixed C++/Metal numeric smoke for pinhole quats/scales path.
- [x] Add fixed C++/Metal numeric smoke for pinhole covars path.
- [ ] CUDA/PyTorch numeric parity.
- [ ] Packed projection forward.

## Task 3.2 - Intersect Tile / Intersect Offset Forward
- [x] Add `gsplat_core/include/gsplat_intersect.h`.
- [x] Add `gsplat_core/gsplat_intersect.cpp`.
- [x] Add `gsplat_core/metal/gsplat_intersect.metal`.
- [x] Expose `intersect_tile_forward(...)` from `_gsplat_core`.
- [x] Expose `intersect_offset_forward(...)` from `_gsplat_core`.
- [x] Support dense AABB fallback path for `[I, N, ...]` style inputs.
- [x] Support optional sort for dense AABB fallback path.
- [x] Add C++ smoke coverage for tile counts, encoded tile ids, flatten ids, and offsets.
- [x] Add `GSPlatIntersectTileCount` MLX Primitive.
- [x] Add Metal kernel `gsplat_intersect_tile_count_kernel`.
- [x] Validate dense AABB tile count on GPU.
- [x] Add `GSPlatIntersectOffset` MLX Primitive.
- [x] Add Metal kernel `gsplat_intersect_offset_kernel`.
- [x] Validate dense sorted `isect_ids` offset generation on GPU.
- [ ] Move intersect tile counting/encoding from C++ reference path to Metal kernels.
- [ ] Support packed path with `image_ids` and `gaussian_ids`.
- [ ] Support AccuTile/SNUGBOX path with `conics` and `opacities`.
- [ ] Support segmented sort.
- [ ] CUDA/PyTorch numeric parity.

## Task 3.3 - Rasterize To Pixels 3DGS Forward
- [x] Add `gsplat_core/include/gsplat_rasterize.h`.
- [x] Add `gsplat_core/gsplat_rasterize.cpp`.
- [x] Add `gsplat_core/metal/gsplat_rasterize.metal`.
- [x] Expose `rasterize_to_pixels_3dgs_forward(...)` from `_gsplat_core`.
- [x] Support dense first-version C++ reference path.
- [x] Support front-to-back alpha compositing.
- [x] Support optional backgrounds.
- [x] Add C++ smoke coverage for render colors, render alphas, and last ids.
- [ ] Move rasterization from C++ reference path to Metal kernels.
- [ ] Support masks.
- [ ] Support packed path.
- [ ] CUDA/PyTorch numeric parity.

## Task 3.4 - Spherical Harmonics Forward
- [x] Add `gsplat_core/include/gsplat_spherical_harmonics.h`.
- [x] Add `gsplat_core/gsplat_spherical_harmonics.cpp`.
- [x] Add `gsplat_core/metal/gsplat_spherical_harmonics.metal`.
- [x] Expose `spherical_harmonics_forward(...)` from `_gsplat_core`.
- [x] Support dense first-version C++ reference path.
- [x] Support degree 0 through degree 4 SH basis evaluation.
- [x] Support optional masks with deterministic zero output for masked entries.
- [x] Add C++ smoke coverage for degree 0, degree 1, and masks.
- [x] Move spherical harmonics from C++ reference path to MLX Primitive + Metal kernel.
- [x] Keep CPU reference path as fallback.
- [x] Add C++/Metal GPU smoke coverage for degree 1 and masks.
- [ ] CUDA/PyTorch numeric parity.

## Task 3.5 - Quat/Scale To Covariance/Precision Forward
- [x] Add `gsplat_core/include/gsplat_quat_scale_to_covar.h`.
- [x] Add `gsplat_core/gsplat_quat_scale_to_covar.cpp`.
- [x] Add `gsplat_core/metal/gsplat_quat_scale_to_covar.metal`.
- [x] Expose `quat_scale_to_covar_preci_forward(...)` from `_gsplat_core`.
- [x] Support dense first-version C++ reference path.
- [x] Support `compute_covar` and `compute_preci`.
- [x] Support `triu=true` output order `[00, 01, 02, 11, 12, 22]`.
- [x] Support `triu=false` full `3x3` row-major output.
- [x] Add C++ smoke coverage for identity quats, non-unit quats, covariance, precision, and empty optional outputs.
- [x] Move quat/scale covariance from C++ reference path to MLX Primitive + Metal kernel.
- [x] Keep CPU reference path as fallback.
- [x] Add C++/Metal GPU smoke coverage for triu, full matrix, covariance, precision, and empty optional outputs.
- [ ] CUDA/PyTorch numeric parity.

## Task 3.6 - End-to-End 3DGS Forward Smoke Chain
- [x] Add C++ smoke coverage that chains projection, intersect tile, intersect offset, spherical harmonics, and rasterize.
- [x] Validate the first-version dense data flow from projected 3D Gaussian attributes to non-empty rendered pixels.
- [x] Verify render output shapes for colors, alphas, and last ids.
- [x] Verify the smoke scene produces nonzero alpha and expected red-only color energy.
- [x] Add manual Python script `scripts/test/forward_3dgs_chain.py`.
- [ ] CUDA/PyTorch numeric parity.
- [ ] High-level Python rasterization compatibility wrapper.
- [ ] Full Metal implementations for the current C++ reference-path ops.

## Task 3.7 - CUDA/PyTorch Parity Reference Scripts
- [x] Add shared parity helper `scripts/test/parity_utils.py`.
- [x] Add projection fused parity script against `gsplat.cuda._wrapper.fully_fused_projection`.
- [x] Add intersect tile / offset parity script against `gsplat.cuda._wrapper.isect_tiles` and `isect_offset_encode`.
- [x] Add rasterize 3DGS parity script against `gsplat.cuda._wrapper.rasterize_to_pixels`.
- [x] Add spherical harmonics parity script against `gsplat.cuda._wrapper.spherical_harmonics`.
- [x] Add quat/scale covariance/precision parity script against `gsplat.cuda._wrapper.quat_scale_to_covar_preci`.
- [x] Add end-to-end forward chain parity script.
- [x] Scripts skip clearly when PyTorch CUDA or gsplat CUDA wrapper is unavailable.
- [ ] Run and record numeric parity on a CUDA machine.
- [ ] Tune tolerances after first CUDA reference run.

## Task 3.8 - CUDA Reference NPZ Export Scripts
- [x] Add `scripts/export_ref` for CUDA/Colab-only gsplat reference exports.
- [x] Add shared export helper `scripts/export_ref/export_utils.py`.
- [x] Add `.npz` export script for projection fused forward.
- [x] Add `.npz` export script for intersect tile / offset.
- [x] Add `.npz` export script for rasterize to pixels 3DGS.
- [x] Add `.npz` export script for spherical harmonics.
- [x] Add `.npz` export script for quat/scale covariance/precision.
- [x] Add `.npz` export script for the end-to-end 3DGS forward chain.
- [x] Document export usage in `scripts/export_ref/README.md`.
- [ ] Add Mac/MLX compare scripts that consume exported `.npz` files.

## Task 3.9 - Spherical Harmonics MLX Primitive + Metal Kernel
- [x] Add `GSPlatSphericalHarmonics` Primitive.
- [x] Add Metal kernel `gsplat_spherical_harmonics_forward_kernel`.
- [x] Preserve the CPU reference implementation in `eval_cpu`.
- [x] Support degree 0 through degree 4 on GPU.
- [x] Support optional masks on GPU.
- [x] Route Python binding `spherical_harmonics_forward` through GPU.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [ ] Run Python manual script after local package reinstall.
- [ ] Compare against exported CUDA `.npz` reference.

## Task 3.10 - Quat/Scale Covariance/Precision MLX Primitive + Metal Kernel
- [x] Add `GSPlatQuatScaleToCovarPreci` Primitive.
- [x] Add Metal kernel `gsplat_quat_scale_to_covar_preci_forward_kernel`.
- [x] Preserve the CPU reference implementation in `eval_cpu`.
- [x] Support covariance and precision outputs on GPU.
- [x] Support `triu=true` and `triu=false` output layouts on GPU.
- [x] Support empty optional outputs when `compute_covar` or `compute_preci` is disabled.
- [x] Route Python binding `quat_scale_to_covar_preci_forward` through GPU.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [ ] Run Python manual script after local package reinstall.
- [ ] Compare against exported CUDA `.npz` reference.

## Task 3.11A - Intersect Tile Count MLX Primitive + Metal Kernel
- [x] Add `gsplat_intersect_tile_count(...)` C++ entry point.
- [x] Add `GSPlatIntersectTileCount` Primitive.
- [x] Add Metal kernel `gsplat_intersect_tile_count_kernel`.
- [x] Support dense AABB count path for `means2d [..., N, 2]`, `radii [..., N, 2]`, and `depths [..., N]`.
- [x] Keep full `intersect_tile_forward` on the C++ reference path for `isect_ids` and `flatten_ids`.
- [x] Add C++/Metal smoke coverage for dense AABB tile counts.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [ ] Use tile count primitive inside a future full GPU intersect path.
- [ ] Add encode/prefix/sort GPU path.

## Task 3.11B - Intersect Offset MLX Primitive + Metal Kernel
- [x] Add `GSPlatIntersectOffset` Primitive.
- [x] Add Metal kernel `gsplat_intersect_offset_kernel`.
- [x] Preserve CPU fallback with a lower-bound reference implementation.
- [x] Route Python binding `intersect_offset_forward` through GPU.
- [x] Add C++/Metal smoke coverage for sorted dense AABB `isect_ids`.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [ ] Compare against exported CUDA `.npz` reference.

---

# Task 4 - Binding and Python-Facing API

## Scope
- Expose migrated low-level ops through `gsplat_core`.
- Keep Python API names close to gsplat CUDA op names, with `_forward` suffixes where useful for clarity.

## Planned APIs
- [x] `quat_scale_to_covar_preci_forward(...)`
- [x] `spherical_harmonics_forward(...)`
- [x] `projection_ewa_3dgs_fused_forward(...)`
- [ ] `projection_ewa_3dgs_packed_forward(...)`
- [x] `intersect_tile_forward(...)`
- [x] `intersect_offset_forward(...)`
- [x] `rasterize_to_pixels_3dgs_forward(...)`
- [ ] `rasterize_to_indices_3dgs_forward(...)`

## Notes
- High-level `gsplat.rendering.rasterization` compatibility is not part of the first milestone.
- End-to-end rendering can be added after the low-level op chain is stable.

---

# Task 5 - Parity and Smoke Validation

## Scope
- Add manual Python scripts under `scripts/test`.
- Validate shape and dtype parity before numeric parity.
- Compare against gsplat CUDA/PyTorch references when the local environment supports it.

## Planned Scripts
- [x] `scripts/test/projection_ewa_3dgs_fused_forward.py`
- [x] `scripts/test/intersect_tile_forward.py`
- [x] `scripts/test/rasterize_to_pixels_3dgs_forward.py`
- [x] `scripts/test/spherical_harmonics_forward.py`
- [x] `scripts/test/quat_scale_to_covar_preci_forward.py`
- [x] `scripts/test/forward_3dgs_chain.py`
- [x] `scripts/test/parity_projection_ewa_3dgs_fused_forward.py`
- [x] `scripts/test/parity_intersect_tile_forward.py`
- [x] `scripts/test/parity_rasterize_to_pixels_3dgs_forward.py`
- [x] `scripts/test/parity_spherical_harmonics_forward.py`
- [x] `scripts/test/parity_quat_scale_to_covar_preci_forward.py`
- [x] `scripts/test/parity_forward_3dgs_chain.py`
- [x] `scripts/export_ref/export_projection_ewa_3dgs_fused_forward.py`
- [x] `scripts/export_ref/export_intersect_tile_forward.py`
- [x] `scripts/export_ref/export_rasterize_to_pixels_3dgs_forward.py`
- [x] `scripts/export_ref/export_spherical_harmonics_forward.py`
- [x] `scripts/export_ref/export_quat_scale_to_covar_preci_forward.py`
- [x] `scripts/export_ref/export_forward_3dgs_chain.py`

## Acceptance Criteria
- [ ] `make env-check` passes.
- [ ] `make xcode-build` passes.
- [ ] `make pip-develop` succeeds in the conda environment.
- [ ] Each migrated op imports from `gsplat_core`.
- [ ] Each migrated op has a manual script that reports input shapes, output shapes, dtypes, and parity status.
- [x] 3DGS forward low-level chain can render a small fixed scene once projection, intersect, and rasterize are migrated.
- [x] Spherical harmonics C++/Metal smoke validates GPU degree 1 and masks.
- [x] Quat/scale C++/Metal smoke validates GPU covariance and precision outputs.
- [x] Intersect tile count C++/Metal smoke validates dense AABB GPU counts.
- [x] Intersect offset C++/Metal smoke validates GPU offsets from sorted `isect_ids`.
- [ ] CUDA/PyTorch parity scripts pass on a CUDA machine.
