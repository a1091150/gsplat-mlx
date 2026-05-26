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
- [ ] `quat_scale_to_covar_preci_fwd`
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
- [ ] Task 3.5: Quat/scale to covariance/precision forward.

## Implementation Rules
- Each op gets a header, C++ implementation, and Metal kernel file.
- Each op exposes one migration-friendly C++ function in namespace `gsplat_core`.
- Binding functions should use clear low-level names based on gsplat CUDA op names.
- CPU fallback may throw or return zero-filled placeholders until a CPU path is explicitly needed.
- GPU path is the source of truth for migrated behavior.

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
- [ ] Move spherical harmonics from C++ reference path to Metal kernels.
- [ ] CUDA/PyTorch numeric parity.

---

# Task 4 - Binding and Python-Facing API

## Scope
- Expose migrated low-level ops through `gsplat_core`.
- Keep Python API names close to gsplat CUDA op names, with `_forward` suffixes where useful for clarity.

## Planned APIs
- [ ] `quat_scale_to_covar_preci_forward(...)`
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
- [ ] `scripts/test/quat_scale_to_covar_preci_forward.py`

## Acceptance Criteria
- [ ] `make env-check` passes.
- [ ] `make xcode-build` passes.
- [ ] `make pip-develop` succeeds in the conda environment.
- [ ] Each migrated op imports from `gsplat_core`.
- [ ] Each migrated op has a manual script that reports input shapes, output shapes, dtypes, and parity status.
- [ ] 3DGS forward low-level chain can render a small fixed scene once projection, intersect, and rasterize are migrated.
