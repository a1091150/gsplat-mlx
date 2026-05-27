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
- The initial skeleton started dummy-only; the current project now includes the
  dense 3DGS forward low-level path.
- Python MLX runtime checks are covered by manual scripts and exported `.npz`
  comparison scripts.

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
- [x] `projection_ewa_3dgs_fused_fwd` dense path
- [ ] `projection_ewa_3dgs_packed_fwd`
- [x] `intersect_tile` dense path
- [x] `intersect_offset`
- [x] `rasterize_to_pixels_3dgs_fwd` dense path
- [ ] `rasterize_to_indices_3dgs`

## Forward Milestone Status
- Dense low-level 3DGS forward render path is implemented and validated:
  projection fused dense -> intersect tile/offset dense -> spherical harmonics
  -> rasterize pixels dense.
- Dense rasterize supports optional backgrounds and tile masks.
- Existing exported CUDA `.npz` fixtures pass on the Mac/MLX side for the dense
  low-level chain.
- The remaining work is for fuller gsplat CUDA forward coverage, not for the
  current dense render smoke path.

## Remaining Forward Coverage
- Packed projection:
  `projection_ewa_3dgs_packed_fwd`,
  `projection_ewa_3dgs_packed_forward(...)`.
- Packed intersect/rasterize paths:
  support `[nnz, ...]` arrays plus `image_ids` and `gaussian_ids`.
- Intersect advanced modes:
  segmented sort and AccuTile/SNUGBOX path using `conics` and `opacities`.
- Rasterize indices:
  `rasterize_to_indices_3dgs`,
  `rasterize_to_indices_3dgs_forward(...)`.
- High-level Python compatibility wrapper for a gsplat-style rasterization API.
- Additional optional CUDA fixtures:
  rasterize masks, SH degree 4 masks, and quat/scale edge cases.

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
- [x] Task 3.1: Projection 3DGS fused forward numeric parity.
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
- [x] Task 3.11C: Intersect tile encode MLX Primitive + Metal kernel first pass.
- [x] Task 3.11D: Intersect tile GPU prefix/sort/reorder helper path.
- [x] Task 3.11E: Replace dense intersect tile forward with staged GPU path.
- [x] Task 3.12: Rasterize to pixels 3DGS MLX Primitive + Metal kernel.
- [x] Task 3.13: Projection 3DGS fused forward parity cleanup.
- [x] Task 3.14: Spherical harmonics exported parity cleanup.
- [x] Task 3.15: Quat/scale covariance exported parity cleanup.
- [x] Task 3.16: Rasterize to pixels 3DGS masks path.

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
- [x] Add fixed C++/Metal smoke for near/far culling, radius clipping, and empty compensations.
- [x] Add CUDA reference export script for projection covars/culling edge cases.
- [x] Extend exported `.npz` compare support for covars projection fixtures.
- [x] CUDA/PyTorch numeric parity for existing exported dense pinhole fixture.
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
- [x] Add `GSPlatIntersectTileEncode` MLX Primitive.
- [x] Add Metal kernel `gsplat_intersect_tile_encode_kernel`.
- [x] Validate dense AABB unsorted `isect_ids` and `flatten_ids` generation on GPU.
- [x] Add GPU exclusive prefix helper for dense `tiles_per_gauss`.
- [x] Add GPU sort/reorder helper for `isect_ids` and `flatten_ids`.
- [x] Validate count -> prefix -> encode -> sort/reorder staged GPU path.
- [x] Move dense intersect tile counting/encoding from C++ reference path to staged GPU path.
- [ ] Support packed path with `image_ids` and `gaussian_ids`.
- [ ] Support AccuTile/SNUGBOX path with `conics` and `opacities`.
- [ ] Support segmented sort.
- [x] CUDA/PyTorch numeric parity for exported dense fixture.

## Task 3.3 - Rasterize To Pixels 3DGS Forward
- [x] Add `gsplat_core/include/gsplat_rasterize.h`.
- [x] Add `gsplat_core/gsplat_rasterize.cpp`.
- [x] Add `gsplat_core/metal/gsplat_rasterize.metal`.
- [x] Expose `rasterize_to_pixels_3dgs_forward(...)` from `_gsplat_core`.
- [x] Support dense first-version C++ reference path.
- [x] Support front-to-back alpha compositing.
- [x] Support optional backgrounds.
- [x] Add C++ smoke coverage for render colors, render alphas, and last ids.
- [x] Move rasterization from C++ reference path to Metal kernels.
- [x] Keep CPU reference path as fallback.
- [x] Add C++/Metal GPU smoke coverage for render colors, render alphas, and last ids.
- [x] Validate Python-facing rasterize output against exported CUDA `.npz`.
- [x] Support masks.
- [ ] Support packed path.
- [x] CUDA/PyTorch numeric parity for exported dense fixture.

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
- [x] Add C++/Metal GPU smoke coverage for degree 4 and masks.
- [x] Add CUDA reference export script for degree 4 and masks.
- [x] Extend exported `.npz` compare support for optional SH masks fixtures.
- [x] CUDA/PyTorch numeric parity for existing exported degree 1 masks fixture.

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
- [x] Add C++/Metal GPU smoke coverage for `compute_covar=false`, `compute_preci=true`, and `triu=false`.
- [x] Add CUDA reference export script for full precision-only edge cases.
- [x] Extend exported `.npz` compare support for optional covars/precis and compute flags.
- [x] CUDA/PyTorch numeric parity for existing exported triu covar+precision fixture.

## Task 3.6 - End-to-End 3DGS Forward Smoke Chain
- [x] Add C++ smoke coverage that chains projection, intersect tile, intersect offset, spherical harmonics, and rasterize.
- [x] Validate the first-version dense data flow from projected 3D Gaussian attributes to non-empty rendered pixels.
- [x] Verify render output shapes for colors, alphas, and last ids.
- [x] Verify the smoke scene produces nonzero alpha and expected red-only color energy.
- [x] Add manual Python script `scripts/test/forward_3dgs_chain.py`.
- [x] CUDA/PyTorch numeric parity for exported dense chain fixture.
- [ ] High-level Python rasterization compatibility wrapper.
- [x] Full Metal implementations for the current dense low-level chain.

## Task 3.7 - CUDA/PyTorch Parity Reference Scripts
- [x] Add shared parity helper `scripts/test/parity_utils.py`.
- [x] Add projection fused parity script against `gsplat.cuda._wrapper.fully_fused_projection`.
- [x] Add intersect tile / offset parity script against `gsplat.cuda._wrapper.isect_tiles` and `isect_offset_encode`.
- [x] Add rasterize 3DGS parity script against `gsplat.cuda._wrapper.rasterize_to_pixels`.
- [x] Add spherical harmonics parity script against `gsplat.cuda._wrapper.spherical_harmonics`.
- [x] Add quat/scale covariance/precision parity script against `gsplat.cuda._wrapper.quat_scale_to_covar_preci`.
- [x] Add end-to-end forward chain parity script.
- [x] Scripts skip clearly when PyTorch CUDA or gsplat CUDA wrapper is unavailable.
- [x] Run and record numeric parity from exported CUDA `.npz` fixtures on the Mac/MLX side.
- [x] Tune tolerances for current exported dense fixtures.
- [ ] Run direct parity scripts on a CUDA machine.

## Task 3.8 - CUDA Reference NPZ Export Scripts
- [x] Add `scripts/export_ref` for CUDA/Colab-only gsplat reference exports.
- [x] Add shared export helper `scripts/export_ref/export_utils.py`.
- [x] Add `.npz` export script for projection fused forward.
- [x] Add `.npz` export script for intersect tile / offset.
- [x] Add `.npz` export script for rasterize to pixels 3DGS.
- [x] Add `.npz` export script for spherical harmonics.
- [x] Add `.npz` export script for quat/scale covariance/precision.
- [x] Add `.npz` export script for the end-to-end 3DGS forward chain.
- [x] Add optional edge-case `.npz` export scripts for projection, SH masks, quat/scale, and rasterize masks.
- [x] Document export usage in `scripts/export_ref/README.md`.
- [x] Add Mac/MLX compare scripts that consume exported `.npz` files.

## Task 3.9 - Spherical Harmonics MLX Primitive + Metal Kernel
- [x] Add `GSPlatSphericalHarmonics` Primitive.
- [x] Add Metal kernel `gsplat_spherical_harmonics_forward_kernel`.
- [x] Preserve the CPU reference implementation in `eval_cpu`.
- [x] Support degree 0 through degree 4 on GPU.
- [x] Support optional masks on GPU.
- [x] Route Python binding `spherical_harmonics_forward` through GPU.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [x] Run Python-facing exported `.npz` comparison after local package reinstall.
- [x] Compare against exported CUDA `.npz` reference for current available fixture.

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
- [x] Run Python-facing exported `.npz` comparison after local package reinstall.
- [x] Compare against exported CUDA `.npz` reference for current available fixture.

## Task 3.11A - Intersect Tile Count MLX Primitive + Metal Kernel
- [x] Add `gsplat_intersect_tile_count(...)` C++ entry point.
- [x] Add `GSPlatIntersectTileCount` Primitive.
- [x] Add Metal kernel `gsplat_intersect_tile_count_kernel`.
- [x] Support dense AABB count path for `means2d [..., N, 2]`, `radii [..., N, 2]`, and `depths [..., N]`.
- [x] First keep full `intersect_tile_forward` on the C++ reference path for `isect_ids` and `flatten_ids`.
- [x] Add C++/Metal smoke coverage for dense AABB tile counts.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [x] Use tile count primitive inside the dense staged GPU intersect path.
- [x] Add encode/prefix/sort GPU path.

## Task 3.11B - Intersect Offset MLX Primitive + Metal Kernel
- [x] Add `GSPlatIntersectOffset` Primitive.
- [x] Add Metal kernel `gsplat_intersect_offset_kernel`.
- [x] Preserve CPU fallback with a lower-bound reference implementation.
- [x] Route Python binding `intersect_offset_forward` through GPU.
- [x] Add C++/Metal smoke coverage for sorted dense AABB `isect_ids`.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [x] Compare against exported CUDA `.npz` reference.

## Task 3.11C - Intersect Tile Encode MLX Primitive + Metal Kernel First Pass
- [x] Add `gsplat_intersect_tile_encode(...)` C++ entry point.
- [x] Add `GSPlatIntersectTileEncode` multi-output Primitive.
- [x] Add Metal kernel `gsplat_intersect_tile_encode_kernel`.
- [x] Preserve CPU fallback with dense AABB reference implementation.
- [x] Accept caller-provided dense exclusive `tile_offsets` and `total_isects`.
- [x] Generate unsorted `isect_ids` and `flatten_ids` on GPU.
- [x] Initially keep Python `intersect_tile_forward` on the existing reference path.
- [x] Add C++/Metal smoke coverage for dense AABB encode output.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [x] Confirm existing exported `.npz` parity remains green.
- [x] Add GPU prefix sum path for `tile_offsets`.
- [x] Add GPU sort/reorder path for `isect_ids` and `flatten_ids`.
- [x] Replace full dense `intersect_tile_forward` with GPU path after prefix/sort are available.
- [x] Compare full dense GPU intersect path against exported CUDA `.npz` reference.

## Task 3.11D - Intersect Tile GPU Prefix/Sort/Reorder Helper Path
- [x] Add `gsplat_intersect_tile_offsets(...)` helper using MLX `cumsum(..., inclusive=false)`.
- [x] Add `gsplat_intersect_tile_sort(...)` helper using MLX `argsort` and `take`.
- [x] Validate dense GPU exclusive prefix offsets from `tiles_per_gauss`.
- [x] Validate staged GPU path: count -> prefix -> encode -> sort/reorder.
- [x] Initially keep Python `intersect_tile_forward` on the existing reference path.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [x] Confirm existing exported `.npz` parity remains green.
- [x] Resolve dynamic `total_isects` output sizing with an eval/read-last staged helper.
- [x] Replace full dense `intersect_tile_forward` after dynamic sizing is handled.
- [x] Compare full GPU intersect path against exported CUDA `.npz` reference.

## Task 3.11E - Replace Dense Intersect Tile Forward With Staged GPU Path
- [x] Route `gsplat_intersect_tile(...)` through `gsplat_intersect_tile_gpu_staged(...)`.
- [x] Route Python binding `intersect_tile_forward` through GPU.
- [x] Keep dense validation and unsupported packed/segmented/AccuTile checks.
- [x] Preserve `sort=true` and `sort=false` behavior through the staged path.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [x] Reinstall package with `make pip-install`.
- [x] Validate exported `.npz` parity for Python-facing `intersect_tile_forward`.

## Task 3.12 - Rasterize To Pixels 3DGS MLX Primitive + Metal Kernel
- [x] Add `GSPlatRasterizeToPixels3DGS` MLX Primitive.
- [x] Route dense rasterize GPU stream through Metal kernel.
- [x] Keep dense CPU reference path as fallback.
- [x] Route Python `rasterize_to_pixels_3dgs_forward` through GPU.
- [x] Support dense inputs: `means2d`, `conics`, `colors`, `opacities`, `backgrounds`, `tile_offsets`, and `flatten_ids`.
- [x] Produce `render_colors`, `render_alphas`, and `last_ids`.
- [x] Add C++/Metal numeric smoke coverage.
- [x] Update end-to-end C++ forward chain to use GPU rasterize.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate `_gsplat_core` target with `make xcode-build`.
- [x] Reinstall package with `make pip-install`.
- [x] Validate exported `.npz` parity for Python-facing `rasterize_to_pixels_3dgs_forward` and the forward chain.

## Task 3.13 - Projection 3DGS Fused Forward Parity Cleanup
- [x] Re-check dense pinhole projection parity against exported CUDA `.npz`.
- [x] Keep packed projection out of scope for this cleanup.
- [x] Add C++/Metal smoke for `calc_compensations=false` empty output.
- [x] Add C++/Metal smoke for near-plane, far-plane, and radius-clip culling.
- [x] Add CUDA export script `scripts/export_ref/export_projection_ewa_3dgs_fused_edge_cases.py`.
- [x] Update `scripts/test/compare_exported_npz.py` to support projection fixtures using `covars` instead of `quats`/`scales`.
- [x] Keep edge-case CUDA fixture optional until exported from a CUDA machine.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate exported `.npz` parity for the existing projection fixture.

## Task 3.14 - Spherical Harmonics Exported Parity Cleanup
- [x] Add C++/Metal smoke coverage for degree 4 SH with masks.
- [x] Compare degree 4 GPU output against CPU reference fallback.
- [x] Add CUDA export script `scripts/export_ref/export_spherical_harmonics_degree4_masks.py`.
- [x] Update `scripts/test/compare_exported_npz.py` to support SH fixtures without masks.
- [x] Keep degree 4 CUDA fixture optional until exported from a CUDA machine.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate exported `.npz` parity for the existing SH fixture.

## Task 3.15 - Quat/Scale Covariance Exported Parity Cleanup
- [x] Add C++/Metal smoke coverage for full `3x3` precision-only output.
- [x] Compare full precision-only GPU output against CPU reference fallback.
- [x] Add CUDA export script `scripts/export_ref/export_quat_scale_to_covar_preci_edge_cases.py`.
- [x] Update `scripts/test/compare_exported_npz.py` to support `input__compute_covar`, `input__compute_preci`, and `input__triu`.
- [x] Update exported `.npz` comparison to support optional `ref__covars` and `ref__precis`.
- [x] Keep full precision-only CUDA fixture optional until exported from a CUDA machine.
- [x] Validate with `make codex-xcode-test`.
- [x] Validate exported `.npz` parity for the existing quat/scale fixture.

## Task 3.16 - Rasterize To Pixels 3DGS Masks Path
- [x] Match gsplat CUDA mask semantics for dense rasterize tile masks.
- [x] Accept optional `masks` with the same shape as `tile_offsets`.
- [x] Skip masked-out tiles before compositing and write background color with zero alpha.
- [x] Add C++ CPU reference support for rasterize masks.
- [x] Add Metal support for rasterize masks.
- [x] Add C++/Metal smoke coverage for a two-tile masked render.
- [x] Add CUDA export script `scripts/export_ref/export_rasterize_to_pixels_3dgs_masks.py`.
- [x] Update exported `.npz` compare support for optional rasterize masks fixtures.
- [x] Keep masks CUDA fixture optional until exported from a CUDA machine.

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
- High-level `gsplat.rendering.rasterization` compatibility is not part of the first dense low-level milestone.
- End-to-end dense low-level rendering is available; high-level API compatibility remains future work.

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
- [x] `scripts/export_ref/export_rasterize_to_pixels_3dgs_masks.py`
- [x] `scripts/export_ref/export_spherical_harmonics_forward.py`
- [x] `scripts/export_ref/export_spherical_harmonics_degree4_masks.py`
- [x] `scripts/export_ref/export_quat_scale_to_covar_preci_forward.py`
- [x] `scripts/export_ref/export_quat_scale_to_covar_preci_edge_cases.py`
- [x] `scripts/export_ref/export_forward_3dgs_chain.py`
- [x] `scripts/test/compare_exported_npz.py`

## Acceptance Criteria
- [x] `make env-check` passes.
- [x] `make xcode-build` passes.
- [ ] `make pip-develop` succeeds in the conda environment.
- [x] `make pip-install` succeeds in the conda environment.
- [x] Each migrated op imports from `gsplat_core`.
- [x] Each migrated op has a manual script or exported `.npz` comparer for shape, dtype, and parity status.
- [x] 3DGS forward low-level chain can render a small fixed scene once projection, intersect, and rasterize are migrated.
- [x] Spherical harmonics C++/Metal smoke validates GPU degree 1 and masks.
- [x] Spherical harmonics C++/Metal smoke validates GPU degree 4 and masks.
- [x] Quat/scale C++/Metal smoke validates GPU covariance and precision outputs.
- [x] Quat/scale C++/Metal smoke validates full precision-only output.
- [x] Intersect tile count C++/Metal smoke validates dense AABB GPU counts.
- [x] Intersect offset C++/Metal smoke validates GPU offsets from sorted `isect_ids`.
- [x] Intersect tile encode C++/Metal smoke validates dense AABB unsorted GPU ids.
- [x] Intersect tile staged GPU smoke validates prefix, encode, sort, and reorder.
- [x] Python-facing dense `intersect_tile_forward` matches exported CUDA `.npz` through staged GPU path.
- [x] Rasterize to pixels C++/Metal smoke validates dense GPU compositing.
- [x] Rasterize to pixels C++/Metal smoke validates dense GPU tile masks.
- [x] Python-facing dense `rasterize_to_pixels_3dgs_forward` matches exported CUDA `.npz` through Metal path.
- [x] Projection C++/Metal smoke validates dense culling and empty compensation behavior.
- [x] Existing Python-facing dense `projection_ewa_3dgs_fused_forward` matches exported CUDA `.npz`.
- [x] Existing Python-facing dense `spherical_harmonics_forward` matches exported CUDA `.npz`.
- [x] Existing Python-facing dense `quat_scale_to_covar_preci_forward` matches exported CUDA `.npz`.
- [x] Current available exported CUDA `.npz` fixtures pass on the Mac/MLX side.
- [ ] Direct CUDA/PyTorch parity scripts pass on a CUDA machine.

---

# Task 6 - 3DGS Backward Migration Plan

## Scope
- Plan the gsplat 3DGS dense backward migration before implementing any single
  backward kernel.
- Keep the first backward milestone aligned with the completed dense forward
  low-level path.
- Prefer explicit low-level backward APIs first, then connect MLX Primitive
  `vjp(...)` after parity and calling conventions are stable.

## In Scope For First Backward Milestone
- Dense 3DGS backward for the current low-level forward chain:
  projection fused dense, rasterize to pixels 3DGS dense, spherical harmonics,
  and quat/scale covariance/precision.
- MLX arrays and Metal kernels under the existing `gsplat_core` naming scheme.
- Python-facing manual scripts and exported `.npz` comparison scripts.
- C++/Metal smoke tests through `make codex-xcode-test`.
- Training-gradient design that supports `viewspace_points` as a dummy
  trainable gradient carrier when screen-space `means2d` gradients are needed.

## Out Of Scope For First Backward Milestone
- 2DGS backward.
- Packed backward paths.
- UT / rolling-shutter / world-ray backward.
- Lidar backward.
- Adam, relocation, MCMC perturb, camera wrappers, and external distortion.
- A high-level gsplat-compatible training wrapper before low-level backward
  parity is established.

## CUDA Backward Source Map
- Python autograd and saved tensors:
  - `submodules/gsplat/gsplat/cuda/_wrapper.py`
  - `submodules/gsplat/gsplat/cuda/ext.cpp`
- Quat/scale backward:
  - `submodules/gsplat/gsplat/cuda/csrc/QuatScaleToCovar.cpp`
  - `submodules/gsplat/gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu`
  - `submodules/gsplat/gsplat/cuda/csrc/QuatScaleToCovar.h`
- Spherical harmonics backward:
  - `submodules/gsplat/gsplat/cuda/csrc/SphericalHarmonics.cpp`
  - `submodules/gsplat/gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu`
  - `submodules/gsplat/gsplat/cuda/csrc/SphericalHarmonics.h`
- Projection 3DGS fused backward:
  - `submodules/gsplat/gsplat/cuda/csrc/Projection.cpp`
  - `submodules/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu`
- Rasterize to pixels 3DGS backward:
  - `submodules/gsplat/gsplat/cuda/csrc/Rasterization.cpp`
  - `submodules/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu`
  - `submodules/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu`

## API Strategy
- First expose explicit backward functions from `_gsplat_core`, for example:
  - `quat_scale_to_covar_preci_backward(...)`
  - `spherical_harmonics_backward(...)`
  - `projection_ewa_3dgs_fused_backward(...)`
  - `rasterize_to_pixels_3dgs_backward(...)`
- Keep explicit backward APIs close to the CUDA custom-op backward signatures,
  using MLX arrays instead of torch tensors.
- Use explicit backward APIs for CUDA `.npz` parity first because they are
  easier to test and debug than autograd-integrated primitive `vjp(...)`.
- After each explicit backward op is numerically stable, wire the matching MLX
  Primitive `vjp(...)` implementation to call the explicit backward path.
- Keep `jvp(...)` unsupported until there is a concrete forward-mode need.

## MLX Gradient Proxy Rule
- PyTorch can retain intermediate gradients with `retain_grad()`, but MLX
  gradients are selected through `mx.value_and_grad(fn, argnums=...)`.
- If training or after-training logic needs `means2d` gradient information,
  introduce a dummy trainable parameter, preferably `viewspace_points`, as a
  visible loss-function argument.
- The dummy gradient proxy must be included in `argnums` so MLX returns its
  gradient.
- Backward API design must preserve enough information for densify, clone,
  split, and related after-training logic to consume the screen-space gradient
  path.

## Proposed Implementation Order
- [x] Task 6.1: Backward CUDA source map and saved-tensor contract.
- [x] Task 6.2: Backward API design doc and `.npz` fixture schema.
- [x] Task 6.3: Spherical harmonics explicit backward.
- [x] Task 6.4: Quat/scale covariance/precision explicit backward.
- [x] Task 6.5: Rasterize to pixels 3DGS explicit backward.
- [x] Task 6.6: Projection EWA 3DGS fused explicit backward.
- [x] Task 6.7: Wire MLX Primitive `vjp(...)` for stable backward ops.
- [x] Task 6.8: Dense training smoke with `viewspace_points` gradient proxy.
- [x] Task 6.9: Projection EWA 3DGS analytic backward.
- [x] Task 6.10: Projection forward `vjp(...)` wiring.
- [x] Task 6.11: Projection backward full GPU path plan and scaffold.

## Validation Plan
- Add CUDA/Colab export scripts under `scripts/export_ref` for each backward op.
- Store exported backward fixtures under `refs/*.npz`.
- Extend `scripts/test/compare_exported_npz.py` with backward comparers.
- Add C++/Metal smoke tests for small deterministic scenes.
- Run:
  - `make codex-xcode-test`
  - `make xcode-build`
  - `make pip-install`
  - `conda run -n fastgs_core python scripts/test/compare_exported_npz.py`
- Use direct CUDA/PyTorch parity scripts only on a CUDA machine.

## Open Design Questions
- Whether rasterize backward should output both signed `means2d` gradient and
  absolute-gradient equivalents for `absgrad` behavior.
- Whether `last_ids` and transmittance handling should exactly mirror CUDA
  saved forward state or recompute small pieces in the backward kernel.
- How much of the high-level training wrapper should be delayed until dense
  low-level backward parity is stable.
- Whether projection forward `vjp(...)` should remain documented as a
  cross-device CPU-reference limitation until projection backward has a Metal
  implementation and can run on the same stream as forward.

## Task 6.1 - Backward CUDA Source Map And Saved-Tensor Contract
- [x] Add `note/gsplat_backward_source_map.md`.
- [x] Map dense 3DGS backward custom op schemas from `ext.cpp`.
- [x] Map `_wrapper.py` saved tensors and backward cotangents.
- [x] Document explicit MLX backward API candidates.
- [x] Document non-differentiable inputs and scalar params.
- [x] Mark packed, 2DGS, UT/world-ray, and lidar backward as out of scope.
- [x] Recommend `spherical_harmonics_backward` as the first implementation slice.

## Task 6.2 - Backward API Design Doc And Fixture Schema
- [x] Add `note/gsplat_backward_api_and_fixture_schema.md`.
- [x] Define explicit backward API naming rules.
- [x] Define Python binding dictionary groups: `inputs`, `forward_outputs`, and `cotangents`.
- [x] Define `.npz` key prefixes: `input__*`, `fwd__*`, `cotangent__*`, `ref__v_*`, and `meta__*`.
- [x] Define optional input and optional gradient key behavior.
- [x] Define first CUDA export script names and expected fixture names.
- [x] Define required fixture keys for SH, quat/scale, rasterize, and projection backward.
- [x] Lock Task 6.3 to `spherical_harmonics_backward(...)` as the first implementation slice.

## Task 6.3 - Spherical Harmonics Explicit Backward
- [x] Add `SphericalHarmonicsBackwardInput`.
- [x] Add explicit C++ entry `gsplat_spherical_harmonics_backward(...)`.
- [x] Add `GSPlatSphericalHarmonicsBackward` MLX Primitive.
- [x] Add Metal kernel `gsplat_spherical_harmonics_backward_kernel`.
- [x] Support degree 0 through 4.
- [x] Support optional masks.
- [x] Support `compute_v_dirs=true/false`.
- [x] Expose Python binding `spherical_harmonics_backward(...)`.
- [x] Add C++ CPU smoke for degree 1 gradients.
- [x] Add C++/Metal smoke for degree 4 with masks.
- [x] Add CUDA export script `scripts/export_ref/export_spherical_harmonics_backward.py`.
- [x] Extend exported `.npz` compare support for `spherical_harmonics_backward.npz`.
- [x] Validate exported CUDA fixture `refs/spherical_harmonics_backward.npz`.
- [ ] Replace first-version finite-difference `v_dirs` with CUDA-style analytic VJP if tighter parity is required.

## Task 6.4 - Quat/Scale Covariance/Precision Explicit Backward
- [x] Add `QuatScaleToCovarPreciBackwardInput`.
- [x] Add explicit C++ entry `gsplat_quat_scale_to_covar_preci_backward(...)`.
- [x] Add `GSPlatQuatScaleToCovarPreciBackward` MLX Primitive.
- [x] Add Metal kernel `gsplat_quat_scale_to_covar_preci_backward_kernel`.
- [x] Support optional `v_covars` and `v_precis` cotangents.
- [x] Support `triu=true` and `triu=false` output layouts.
- [x] Expose Python binding `quat_scale_to_covar_preci_backward(...)`.
- [x] Add C++ CPU smoke with identity-quaternion scale-gradient reference.
- [x] Add C++/Metal smoke comparing GPU backward against CPU backward.
- [x] Add CUDA export script `scripts/export_ref/export_quat_scale_to_covar_preci_backward.py`.
- [x] Extend exported `.npz` compare support for `quat_scale_to_covar_preci_backward.npz`.
- [x] Validate exported CUDA fixture `refs/quat_scale_to_covar_preci_backward.npz`.
- [ ] Replace first-version finite-difference VJP with CUDA-style analytic VJP if tighter parity or speed is required.

## Task 6.5 - Rasterize To Pixels 3DGS Explicit Backward
- [x] Add `RasterizeToPixels3DGSBackwardInput`.
- [x] Add explicit C++ entry `gsplat_rasterize_to_pixels_3dgs_backward(...)`.
- [x] Add `GSPlatRasterizeToPixels3DGSBackward` MLX Primitive.
- [x] Add Metal kernel `gsplat_rasterize_to_pixels_3dgs_backward_kernel`.
- [x] Implement dense, unpacked 3DGS backward path.
- [x] Support optional backgrounds and masks inputs.
- [x] Support `absgrad` output for `v_means2d_abs`.
- [x] Expose Python binding `rasterize_to_pixels_3dgs_backward(...)`.
- [x] Add C++/Metal smoke comparing GPU backward against CPU backward.
- [x] Add CUDA export script `scripts/export_ref/export_rasterize_to_pixels_3dgs_backward.py`.
- [x] Extend exported `.npz` compare support for `rasterize_to_pixels_3dgs_backward.npz`.
- [x] Validate exported CUDA fixture `refs/rasterize_to_pixels_3dgs_backward.npz`.
- [ ] Add packed rasterize backward support if packed training path becomes in scope.

## Task 6.6 - Projection EWA 3DGS Fused Explicit Backward
- [x] Add `ProjectionEWA3DGSFusedBackwardInput`.
- [x] Add explicit C++ entry `gsplat_projection_ewa_3dgs_fused_backward(...)`.
- [x] Add `GSPlatProjectionEWA3DGSFusedBackward` MLX Primitive.
- [x] Implement first-pass dense pinhole backward with finite-difference VJP.
- [x] Support covars path outputs: `v_means`, `v_covars`.
- [x] Support quats/scales path outputs: `v_means`, `v_quats`, `v_scales`.
- [x] Support optional `v_compensations`.
- [x] Expose Python binding `projection_ewa_3dgs_fused_backward(...)`.
- [x] Add C++ smoke for covars path.
- [x] Add CUDA export script `scripts/export_ref/export_projection_ewa_3dgs_fused_backward.py`.
- [x] Extend exported `.npz` compare support for `projection_ewa_3dgs_fused_backward.npz`.
- [x] Validate exported CUDA fixture `refs/projection_ewa_3dgs_fused_backward.npz`.
- [x] Replace first-version finite-difference VJP with CUDA-style analytic VJP for dense pinhole covars path.
- [x] Add analytic quats/scales covariance-to-parameter VJP for projection backward.
- [ ] Add Metal projection backward kernel after analytic VJP is locked.
- [ ] Add packed projection backward support if packed training path becomes in scope.

## Task 6.7 - Wire MLX Primitive `vjp(...)` For Stable Backward Ops
- [x] Wire `GSPlatSphericalHarmonics::vjp(...)` to `gsplat_spherical_harmonics_backward(...)`.
- [x] Wire `GSPlatQuatScaleToCovarPreci::vjp(...)` to `gsplat_quat_scale_to_covar_preci_backward(...)`.
- [x] Wire `GSPlatRasterizeToPixels3DGS::vjp(...)` to `gsplat_rasterize_to_pixels_3dgs_backward(...)`.
- [x] Keep projection forward `vjp(...)` unimplemented until projection backward is ready for full training.
- [x] Add `scripts/test/autograd_vjp_smoke.py` for `mx.value_and_grad(..., argnums=...)` smoke coverage.
- [x] Validate C++/Xcode smoke with `make codex-xcode-test`.
- [x] Validate installed Python extension with `make pip-install`.
- [x] Validate Python autograd smoke with `conda run -n fastgs_core python scripts/test/autograd_vjp_smoke.py`.
- [x] Confirm exported `.npz` parity remains green.

## Task 6.8 - Dense Training Smoke With `viewspace_points` Gradient Proxy
- [x] Add `scripts/test/training_viewspace_proxy_smoke.py`.
- [x] Use rasterize dense forward/backward as the first stable training-gradient path.
- [x] Keep projection out of this smoke until projection forward `vjp(...)` is backed by analytic/Metal backward.
- [x] Model the FastGS MLX proxy pattern with `screen_means = means2d + viewspace_points`.
- [x] Include `viewspace_points` in `mx.value_and_grad(..., argnums=...)`.
- [x] Validate `viewspace_points` gradient shape and nonzero values.
- [x] Validate proxy gradient matches the direct `means2d` gradient for the additive proxy path.
- [x] Add `make codex-training-smoke`.

## Task 6.9 - Projection EWA 3DGS Analytic Backward
- [x] Replace CPU finite-difference projection VJP for dense pinhole covars path.
- [x] Add closed-form VJP pieces for inverse conic, antialiasing compensation, perspective projection, world-to-camera position, and covariance transform.
- [x] Add analytic quats/scales covariance-to-parameter VJP.
- [x] Preserve explicit Python API `projection_ewa_3dgs_fused_backward(...)`.
- [x] Keep projection forward `vjp(...)` unimplemented until the projection backward contract is ready for full training.
- [x] Validate `refs/projection_ewa_3dgs_fused_backward.npz` against CUDA reference.
- [x] Confirm full exported `.npz` parity remains green.
- [ ] Add Metal projection backward kernel.
- [ ] Add packed projection backward support if packed training path becomes in scope.

## Task 6.10 - Projection Forward `vjp(...)` Wiring
- [x] Wire `GSPlatProjectionEWA3DGSFused::vjp(...)` to `gsplat_projection_ewa_3dgs_fused_backward(...)`.
- [x] Support gradients for `means` and dense covars path.
- [x] Support gradients for quats/scales path.
- [x] Support optional `viewmats` gradients when requested through `argnums`.
- [x] Add `viewspace_points` as a projection primitive input for the MLX retain-grad proxy path.
- [x] Return `means2d` cotangent to `viewspace_points` when its primitive argnum is requested.
- [x] Keep `Ks`, opacities, packed, ortho, and fisheye gradients out of scope.
- [x] Extend `scripts/test/autograd_vjp_smoke.py` with projection `value_and_grad` smoke.
- [x] Add `scripts/test/training_projection_viewspace_proxy_smoke.py` for projection -> rasterize -> viewspace proxy smoke.
- [x] Update `make codex-training-smoke` to run both viewspace proxy smoke scripts.
- [x] Dense covars + pinhole projection `vjp(...)` now routes to the same-stream
  Metal backward path. Unsupported projection VJP cases remain CPU/reference or
  out of scope and should not be treated as full-GPU training coverage.

## Task 6.11 - Projection Backward Full GPU Path Plan And Scaffold
- [x] Add `note/projection_backward_gpu_path.md`.
- [x] Document why the current projection `vjp(...)` is a CPU-reference path
  and not a pure GPU training graph.
- [x] Document that pure GPU graph wiring should use `stream()` instead of
  forcing `.s = mx::Device::cpu`.
- [x] Document current full-GPU gap: `GSPlatProjectionEWA3DGSFusedBackward::eval_gpu(...)`.
- [x] Implement first-pass `gsplat_projection_ewa_3dgs_fused_backward_kernel`.
- [x] Support dense pinhole covars path first: `v_means`, `v_covars`.
- [x] Support `calc_compensations=true/false` in the dense covars GPU kernel.
- [x] Add C++/Xcode smoke comparing projection GPU backward against CPU reference.
- [ ] Add optional `v_viewmats`.
- [ ] Add quats/scales GPU backward after covars path parity is stable.
- [x] Change projection `vjp(...)` backward input from CPU to `stream()` after
  Metal backward is available.
- [x] Validate no projection `vjp(...)` materialization is needed when forward
  and backward run on the same GPU stream.
- [x] Re-enable projection autograd and training smoke as required full-GPU
  acceptance checks.

## Task 6.12 - Projection VJP Full GPU Path Routing
- [x] Route projection `vjp(...)` to the current primitive stream when the
  request is supported by the first-pass Metal backward path:
  dense covars and pinhole camera.
- [x] Keep CPU/reference routing for unsupported projection backward cases:
  non-pinhole cameras, packed paths, and future distortion variants.
- [x] Validate projection autograd smoke with `v_means`, `v_covars`, and
  `viewspace_points` gradients.
- [x] Validate dense training smoke without adding extra `mx::eval(...)` inside
  projection `vjp(...)`.

## Task 6.13 - Projection Backward Unsupported Path Guardrails
- [x] Add `scripts/test/projection_vjp_guardrails.py`.
- [x] Add `make codex-projection-guardrails`.
- [x] Verify the supported full-GPU projection VJP boundary:
  dense covars, pinhole camera, and nonzero `v_means`, `v_covars`,
  `v_quats`, `v_scales`, `v_viewmats`, and `viewspace_points` gradients.
- [x] Report unsupported or fallback projection VJP cases explicitly:
  packed projection, non-pinhole cameras, `Ks`, opacities, and distortion
  paths.
- [x] Keep unsupported fallback diagnostics separate from full-GPU acceptance.

## Task 6.14 - Projection Backward `v_viewmats` GPU Path
- [x] Add a Metal `v_viewmats` reduction kernel for dense covars + pinhole
  projection backward.
- [x] Keep the first Metal version deterministic by assigning one thread per
  `(batch, camera)` and reducing over all gaussians, instead of relying on
  floating-point atomics.
- [x] Route projection `vjp(...)` through the same-stream GPU backward path
  even when `v_viewmats` is requested.
- [x] Extend C++/Xcode smoke to compare GPU `v_viewmats` against CPU reference.
- [x] Update projection VJP guardrails so `v_viewmats` is part of the supported
  dense covars + pinhole full-GPU boundary.

## Task 6.15 - Projection Backward Quat/Scale GPU Path
- [x] Add analytic quaternion/scale covariance VJP helpers to projection Metal.
- [x] Support `use_covars=false` in
  `GSPlatProjectionEWA3DGSFusedBackward::eval_gpu(...)`.
- [x] Route projection `vjp(...)` through same-stream GPU backward for both
  dense covars and quats/scales when `camera_model == pinhole`.
- [x] Extend C++/Xcode smoke to compare GPU `v_quats` and `v_scales` against
  CPU reference.
- [x] Extend Python autograd smoke with projection quats/scales gradients.
- [x] Update projection VJP guardrails so quats/scales are part of supported
  dense fused pinhole full-GPU coverage.

## Task 6.16 - Dense 3DGS Training Loop Smoke
- [x] Add `scripts/test/training_dense_3dgs_loop_smoke.py`.
- [x] Exercise the dense quats/scales projection path followed by dense
  rasterize to pixels in one `mx.value_and_grad(...)` loss.
- [x] Include `viewspace_points` as a dummy trainable argument so the MLX path
  mirrors the FastGS retain-grad proxy design.
- [x] Request gradients for means, quats, scales, colors, opacities, and
  `viewspace_points` through sorted `argnums=(0, 1, 2, 3, 4, 5)`.
- [x] Validate finite loss, nonzero gradients, expected gradient shapes, and a
  short SGD-style update loop that does not diverge.
- [x] Add `make codex-dense-training-smoke` and include it in the broader
  `make codex-training-smoke` / `make codex-projection-guardrails` checks.

## Task 6.17 - Full Forward Training Path Smoke
- [x] Replace the fixed one-tile `flatten_ids` setup in
  `scripts/test/training_dense_3dgs_loop_smoke.py` with the full dense forward
  routing path:
  projection -> intersect tile -> intersect offset -> rasterize.
- [x] Use a small `32x32` image with `16x16` tiles so the smoke exercises a
  `2x2` tile grid instead of a trivial one-tile render.
- [x] Keep dense quats/scales, pinhole camera, and `viewspace_points` as the
  trainable-proxy coverage.
- [x] Apply `mx.stop_gradient(...)` to `tile_offsets` and `flatten_ids` before
  rasterize, because tile assignment/sorting is discrete routing and should not
  be differentiated through in this training smoke.
- [x] Validate nonzero tile intersections and expected tile offset shape before
  running the short training loop.
- [x] Validate `make codex-dense-training-smoke`,
  `make codex-training-smoke`, and `make codex-projection-guardrails`.
- [ ] Add a larger training smoke that uses backgrounds, masks, or multi-camera
  batches after the dense single-camera path remains stable.

## Task 6.18 - Minimal Python MLX Training Example
- [x] Add `scripts/test/train_tiny_3dgs_mlx.py`.
- [x] Use `mlx.nn.Module` to hold tiny trainable 3DGS parameters:
  means, quats, log-scales, color logits, and opacity logits.
- [x] Use separate `mlx.optimizers.Adam` instances for the parameter groups so
  future trainer code can tune learning rates per group.
- [x] Use `mx.value_and_grad(..., argnums=(0, 1, 2, 3, 4, 5))` and pass
  `viewspace_points` as the dummy trainable proxy argument.
- [x] Render through the current full dense forward path:
  projection -> intersect tile -> intersect offset -> rasterize.
- [x] Stop gradients through discrete `tile_offsets` and `flatten_ids`.
- [x] Generate a small synthetic RGB target image with numpy and save target,
  initial render, and final render PNGs.
- [x] Add `make codex-tiny-train` for the manual tiny trainer.
- [ ] Extend the tiny trainer with optional external image input after the
  generated-image path stays stable.

## Task 6.19 - Tiny Multi-View Training Smoke
- [x] Add `scripts/test/train_tiny_multiview_3dgs_mlx.py` as a separate
  multi-view training script.
- [x] Generate multiple synthetic pinhole cameras that share one Gaussian set
  but use different `viewmats`, `Ks`, and target images.
- [x] Keep the first version dense, single-scene, pinhole-only, and
  quats/scales-based.
- [x] Generate multi-view targets by rendering a fixed target Gaussian model so
  the target views are 3D-consistent.
- [x] Train by cycling one view per step while all views update the
  same `Tiny3DGSModel` parameters.
- [x] Keep `viewspace_points` as an explicit dummy trainable proxy argument for
  each selected view.
- [x] Render every view through:
  projection -> intersect tile -> intersect offset -> rasterize.
- [x] Continue to stop gradients through discrete `tile_offsets` and
  `flatten_ids`.
- [x] Save target, initial, and final preview PNGs per view, such as
  `target_view_00.png`, `step_0000_view_00.png`, and
  `step_0040_view_00.png`.
- [x] Add `make codex-tiny-multiview-train` and related Makefile variables.
- [x] Validate that multi-view loss is finite and does not diverge on a short
  run.
- [ ] Add external image or dataset-backed multi-view targets after the
  generated target-model path stays stable.

## Task 6.20 - Tiny Trainer SPZ Export
- [ ] Add optional SPZ export after the tiny trainer has a stable single-view
  and multi-view path.
- [ ] Keep SPZ export separate from Task 6.19 so camera/loss behavior and file
  format behavior can be debugged independently.
- [ ] Detect whether the `spz` Python package is available in the
  `fastgs_core` conda environment and report a clear skip/error if missing.
- [ ] Export trained Gaussian attributes:
  means, log-scales or scales in the expected SPZ convention, normalized
  quaternions, sigmoid opacities, and color data.
- [ ] Decide and document the color convention for the first exporter:
  direct RGB-to-SH degree 0 or an explicitly marked RGB approximation.
- [ ] Decide and document quaternion order and coordinate-system assumptions
  before treating the exported file as viewer-compatible.
- [ ] Add a `--save-spz` or similar CLI flag to
  `scripts/test/train_tiny_3dgs_mlx.py`.
- [ ] Add a Makefile variable for optional SPZ export after the CLI flag is
  validated.
- [ ] Validate that the exporter writes a nonempty `.spz` file and that the
  training path still works when export is disabled.
- [ ] Defer scanner-scene SPZ export until Task 6.24 confirms `points.ply` and
  scanner camera convention render in the expected positions.

## Task 6.21 - Scanner Dataset Camera/Image Loader Smoke
- [x] Add `scripts/test/scanner_dataset_random_render_smoke.py`.
- [x] Read scanner dataset `frame_*.jpg` / `frame_*.json` pairs from a dataset
  directory such as `/Users/yangdunfu/Downloads/2026_05_04_16_51_29`.
- [x] Do not read `points.ply` in this step; initialize random Gaussians
  instead.
- [x] Convert scanner JSON `intrinsics` to gsplat-style `Ks` with resize scale
  applied.
- [x] Convert scanner JSON `cameraPoseARFrame` to gsplat-style row-major
  world-to-camera `viewmats`, following the FastGS scanner axis transform and
  camera-axis flip as the first compatibility assumption.
- [x] Render random Gaussians through the current gsplat_core forward path:
  projection -> intersect tile -> intersect offset -> rasterize.
- [x] Save resized target images, random render previews, and
  `debug_camera_metadata.json` with `K`, `viewmat`, visible Gaussian count, and
  intersection count.
- [x] Add `make codex-scanner-dataset-smoke` and related Makefile variables.
- [x] Validate a short smoke against the scanner dataset with visible Gaussians
  and nonzero intersections.
- [ ] Compare camera convention against a known-good gsplat/PyTorch or viewer
  render before using this loader for real training.

## Task 6.22 - Scanner Dataset Random Gaussian Training Smoke
- [x] Add `scripts/test/train_scanner_random_3dgs_mlx.py`.
- [x] Read scanner dataset `frame_*.jpg` / `frame_*.json` pairs through the
  Task 6.21 camera/image loader.
- [x] Do not read `points.ply`; initialize random Gaussians around the selected
  scanner cameras.
- [x] Train one shared `Tiny3DGSModel` against real scanner target images by
  cycling selected frames.
- [x] Use `mx.value_and_grad(..., argnums=(0, 1, 2, 3, 4, 5))` with
  `viewspace_points` as the dummy trainable proxy argument.
- [x] Update means, quats, log-scales, color logits, and opacity logits with
  Adam optimizers.
- [x] Render through projection -> intersect tile -> intersect offset ->
  rasterize for every training step.
- [x] Save target images plus initial/final render previews per scanner frame.
- [x] Add `make codex-scanner-random-train` and scanner training Makefile
  variables.
- [x] Validate a short training run with finite loss, nonzero
  `viewspace_points` gradient, and non-diverging mean frame loss.
- [ ] Revisit camera convention and initialization quality before treating this
  as a real scanner training pipeline.

## Task 6.23 - Scanner Training Preview Diagnostics
- [x] Extend `scripts/test/train_scanner_random_3dgs_mlx.py` with per-frame
  diagnostic evaluation.
- [x] Report initial/final loss and PSNR for every selected scanner frame.
- [x] Report visible Gaussian counts and intersection counts before and after
  the training smoke.
- [x] Report the latest `viewspace_points` gradient norm during training.
- [x] Save side-by-side comparison PNGs per frame:
  target | initial render | final render.
- [x] Save `training_summary.json` with run settings, mean losses, latest
  viewspace gradient norm, and per-frame summaries.
- [x] Validate diagnostics on a short scanner random training run.
- [ ] Use these diagnostics to judge camera convention before adding
  `points.ply` initialization.

## Task 6.24 - Scanner `points.ply` Camera Alignment Render
- [x] Add `scripts/test/scanner_points_alignment_render.py`.
- [x] Add a scanner dataset alignment script that reads both scanner cameras
  and `points.ply`.
- [x] Use `/Users/yangdunfu/Documents/GitHub/fastgs_core/scripts/train_scanner_fastgs2.py`
  as a reference for `points.ply` parsing, axis transform, color loading, and
  camera/image pairing, but do not assume its convention is automatically
  correct for gsplat_core.
- [x] Convert `points.ply` positions and RGB colors into gsplat_core Gaussian
  parameters without training.
- [x] Initialize simple render-only Gaussian attributes:
  means from points, small linear scales, identity/random quats, sigmoid
  opacities, and RGB colors.
- [x] Render selected scanner frames through:
  projection -> intersect tile -> intersect offset -> rasterize.
- [x] Save target image, point-cloud render, and side-by-side comparison PNGs
  per selected frame.
- [x] Save debug metadata with camera `K`, `viewmat`, point count, visible
  point/Gaussian count, intersection count, and render alpha stats.
- [x] Add `make codex-scanner-points-align` and related Makefile variables.
- [x] Validate a short alignment render with visible points and nonzero alpha.
- [ ] Use the rendered alignment previews to decide whether the FastGS scanner
  axis transform/camera flip matches gsplat_core expectations.
- [ ] If alignment is wrong, document candidate convention fixes instead of
  proceeding to training or SPZ export.

## Task 6.25 - Scanner Scene SPZ Export
- [x] Add `scripts/test/export_scanner_points_spz.py`.
- [x] Add SPZ export for scanner-scene Gaussians after Task 6.24 verifies
  `points.ply` and camera alignment.
- [x] Keep this separate from Task 6.24 so render alignment and file export can
  be debugged independently.
- [x] Detect whether the `spz` Python package is available in the
  `fastgs_core` conda environment.
- [x] Export scanner-initialized Gaussian attributes from `points.ply`:
  positions, log-scales, identity quaternions, logit opacities, and colors.
- [x] Document coordinate system assumptions, quaternion ordering, scale
  convention, opacity convention, and color/SH convention before treating the
  output as viewer-compatible.
- [x] Add `make codex-scanner-points-spz` and related Makefile variables.
- [x] Validate that export stays separate from training/render smoke.
- [x] Validate that export enabled writes a nonempty `.spz` file and reports
  the exported Gaussian count.
- [ ] Inspect exported `.spz` in the intended viewer and adjust coordinate or
  color conventions if needed.

## Task 6.26 - Scanner Points Multi-View Training And SPZ Export
- [x] Add `scripts/test/train_scanner_points_multiview_3dgs_mlx.py`.
- [x] Read scanner `frame_*.jpg` / `frame_*.json` camera-image pairs through
  the Task 6.21 loader.
- [x] Read `points.ply` through the Task 6.24 path and initialize Gaussians from
  the point cloud instead of random points.
- [x] Use the Task 6.24 scanner axis transform and camera convention as the
  first implementation assumption, because the visual alignment looked
  reasonable.
- [x] Train one shared Gaussian model across multiple scanner frames/views by
  cycling or sampling views per step.
- [x] Train means, log-scales, color logits/RGB, opacity logits, and
  quats with separate Adam optimizers or per-parameter learning rates.
- [x] Keep `viewspace_points` as an explicit dummy trainable proxy argument in
  `mx.value_and_grad(..., argnums=...)` so screen-space gradient diagnostics
  remain available.
- [x] Render every training step through:
  projection -> intersect tile -> intersect offset -> rasterize.
- [x] Stop gradients through discrete `tile_offsets` and `flatten_ids`.
- [x] Save target, initial render, final render, and side-by-side comparison
  PNGs for selected scanner frames.
- [x] Save `training_summary.json` with per-frame loss/PSNR, visible Gaussian
  counts, intersection counts, viewspace gradient norm, exported point count,
  and training settings.
- [x] Export trained Gaussians to `.spz` at the end of training using the Task
  6.25 SPZ convention, and write matching metadata.
- [x] Add `make codex-scanner-points-train-spz` and related variables for
  scanner points multi-view training and SPZ export.
- [x] Validate a short smoke run with a small point subset, finite loss,
  nonzero `viewspace_points` gradient, preview PNGs, and nonempty `.spz`
  output.
- [ ] Keep this as a smoke/minimal trainer first; defer densify, split, clone,
  pruning, opacity reset, and full scanner optimizer policy until this path is
  stable.

## Task 6.27 - Scanner Points Trainer SH Degree 0/1 Color Path
- [x] Replace the scanner points multi-view trainer's RGB color-logit path with
  SH coefficients split into `features_dc` and `features_rest`.
- [x] Initialize `features_dc` from `points.ply` RGB using the gsplat/SPZ
  degree-0 SH convention `(rgb - 0.5) / C0`.
- [x] Add `--sh-degree` and `--max-sh-degree` so the trainer can run degree 0
  or degree 1 now while keeping coefficient storage available for higher
  degrees.
- [x] Render per-view colors through `spherical_harmonics_forward` using the
  current camera center and Gaussian-to-camera directions before rasterization.
- [x] Train `features_dc` and `features_rest` with separate Adam updates while
  preserving means, quats, log-scales, opacity logits, and the explicit
  `viewspace_points` gradient proxy.
- [x] Export trained SPZ files with SH metadata: `cloud.colors` stores degree-0
  coefficients and `cloud.sh` stores the active higher-order coefficients.
- [x] Add `--color-mode rgb|sh`, keeping RGB logits available as the default
  path and using SH coefficients only when `--color-mode sh` is selected.
- [x] Add `SCANNER_POINTS_TRAIN_COLOR_MODE`,
  `SCANNER_POINTS_TRAIN_SH_DEGREE`, and `SCANNER_POINTS_TRAIN_MAX_SH_DEGREE`
  to `make codex-scanner-points-train-spz`.
- [x] Treat degree 2/3 as supported by the same coefficient path and
  smoke-test the SPZ exportable higher-degree path; current `spz` package
  validation rejects degree 4.

## Task 6.28 - 3DGS After-Training Densify/Split/Clone/Prune Plan
- [ ] Map gsplat source behavior from:
  - `submodules/gsplat/gsplat/strategy/default.py`
  - `submodules/gsplat/gsplat/strategy/ops.py`
  - `submodules/gsplat/examples/simple_trainer.py`
- [ ] Implement the MLX equivalent of gsplat `DefaultStrategy` for the scanner
  points trainer, scoped to 3DGS only.
- [ ] Keep MCMC relocation/noise, 2DGS, UT, lidar, distributed packed paths,
  sparse gradients, and visible Adam outside the first implementation.

### Source Behavior To Port
- `DefaultStrategy` keeps running state:
  - `grad2d`: accumulated norm of image-plane gradients per Gaussian.
  - `count`: visible-hit count per Gaussian.
  - optional `radii`: max normalized 2D radius per Gaussian when
    `refine_scale2d_stop_iter > 0`.
  - `scene_scale`.
- gsplat uses `info["means2d"].retain_grad()` before backward. In MLX this
  must stay as an explicit `viewspace_points` dummy trainable input passed to
  `mx.value_and_grad(..., argnums=...)`.
- After backward and optimizer update, gsplat updates strategy state from:
  - `viewspace_points` gradient in MLX, standing in for `means2d.grad`.
  - `radii`, `width`, `height`, and camera count.
  - visible Gaussian ids from dense radii masks first; packed ids can be
    deferred.
- Gradient normalization follows gsplat:
  - `grad[..., 0] *= width / 2 * n_cameras`
  - `grad[..., 1] *= height / 2 * n_cameras`
  - accumulate `norm(grad)` into `grad2d`.

### Default Thresholds From gsplat
- `prune_opa = 0.005`
- `grow_grad2d = 0.0002`
- `grow_scale3d = 0.01`
- `grow_scale2d = 0.05`
- `prune_scale3d = 0.1`
- `prune_scale2d = 0.15`
- `refine_start_iter = 500`
- `refine_stop_iter = 15000`
- `reset_every = 3000`
- `refine_every = 100`
- `pause_refine_after_reset = 0`
- `refine_scale2d_stop_iter = 0` by default, so screen-size split/prune can be
  implemented after the first 3D scale path.

### Operation Semantics
- Clone/duplicate:
  - Select `grad2d / count > grow_grad2d`.
  - Require `max(exp(log_scales)) <= grow_scale3d * scene_scale`.
  - Append copies of every trainable Gaussian parameter.
  - Append zero optimizer state for new rows.
  - Append duplicated running state entries.
- Split:
  - Select high-gradient Gaussians that are not small by the 3D scale test.
  - Optionally include large 2D radius when `refine_scale2d_stop_iter` is
    enabled.
  - Replace selected Gaussians with two children.
  - Sample child offsets from Gaussian local covariance:
    `rotmat(normalized_quat) @ scale @ randn`.
  - Child scales use `log(exp(parent_scale) / 1.6)`.
  - Copy color/SH/opacity/quaternion parameters unless revised opacity is
    explicitly added later.
  - Reset optimizer state for children.
- Prune:
  - Always remove `sigmoid(opacity_logits) < prune_opa`.
  - After `reset_every`, also remove Gaussians with
    `max(exp(log_scales)) > prune_scale3d * scene_scale`.
  - Optional 2D radius pruning can follow the source behavior later.
- Opacity reset:
  - Every `reset_every` steps, clamp opacity logits to at most
    `logit(prune_opa * 2)`.
  - Reset optimizer state for opacity parameters.

### MLX Implementation Notes
- Since MLX `Adam` state is tied to `nn.Module` arrays, every topology-changing
  operation must rebuild model arrays and either:
  - rebuild optimizers, or
  - update optimizer state arrays with matching append/remove rows.
- First implementation may rebuild optimizers after clone/split/prune for
  correctness, then preserve optimizer state in a later performance pass.
- Must support both current color modes:
  - RGB logits: update `color_logits`.
  - SH coefficients: update `features_dc` and `features_rest`.
- `viewspace_points` is not a stored model parameter; it is recreated per step
  with the current Gaussian count and only used to collect screen-space
  gradients.
- Topology changes must happen after optimizer updates for the step, matching
  gsplat `simple_trainer.py`.

### Implementation Slices
- [x] Task 6.28A: Add a small Python strategy/state object to
  `train_scanner_points_multiview_3dgs_mlx.py` with thresholds, counters,
  schedule flags, and summary logging only.
- [x] Task 6.28B: Accumulate dense 3DGS `viewspace_points` gradients into
  `grad2d/count` and validate nonzero counts on smoke data.
- [x] Task 6.28C: Implement prune by opacity first because it only removes rows
  and is easiest to validate.
- [x] Task 6.28D: Implement clone/duplicate for high-gradient small Gaussians.
- [x] Task 6.28E: Implement split for high-gradient large Gaussians with
  quaternion-derived local random offsets and scale shrink.
- [x] Task 6.28F: Add opacity reset.
- [x] Task 6.28G: Add summary fields and preview diagnostics:
  `num_gaussians_before/after`, `n_clone`, `n_split`, `n_prune`,
  `n_opacity_reset`, `grad2d_mean/max`, and visible counts.
- [x] Task 6.28H: Add optional Makefile variables for strategy thresholds and
  disabled-by-default refinement in smoke runs.
- [x] Task 6.28I: Add gsplat-style scale prune after `reset_every`, including
  3D scale pruning and optional 2D radius pruning when radius tracking is
  enabled.

### Validation Plan
- Start with tiny scanner runs and force low thresholds so each operation can be
  triggered deterministically.
- Validate shapes after every topology change for:
  - `means`, `quats`, `log_scales`, `opacity_logits`
  - RGB `color_logits`
  - SH `features_dc/features_rest`
- Validate `viewspace_points` shape follows the updated Gaussian count on the
  next step.
- Validate SPZ export after topology changes for both RGB and SH modes.
- Keep image quality expectations weak at first; the primary check is finite
  loss, nonzero gradient stats, correct counts, and nonempty SPZ output.

## Task 6.29 - Scanner Trainer Loss and Remaining gsplat Alignment

### Completed
- [x] Switch `train_scanner_points_multiview_3dgs_mlx.py` training loss to
  `mlx.nn.losses.l1_loss` for both RGB and SH color paths.
- [x] Report initial/final frame losses with the same L1 metric used for
  training.
- [x] Keep PSNR as an MSE-derived image-quality diagnostic and record the active
  loss function in `training_summary.json`.

### Remaining After-Training Strategy Alignment
- [ ] Validate clone/split/prune/reset thresholds on long scanner training runs.
- [ ] Implement real `absgrad` behavior instead of only preserving the CLI/config
  option.
- [ ] Add a dedicated smoke for `refine_scale2d_stop_iter > 0` so 2D radius
  split/prune is exercised.
- [ ] Validate `revised_opacity` split on longer training runs and decide whether
  it should be exposed as a recommended scanner setting.
- [ ] Keep MCMC relocation/noise, distributed packed paths, sparse gradients, and
  visible Adam outside the current first scanner trainer implementation unless a
  later task explicitly brings them in.

### Remaining Training Behavior Alignment
- [x] Add progressive SH degree scheduling so SH training can increase the
  active degree during the run while keeping `--max-sh-degree` as model
  capacity.
- [x] Add first optimizer learning-rate schedule for `means` / positions,
  including Makefile controls and training summary diagnostics.
- [x] Add held-out frame evaluation diagnostics with optional
  `compare_eval_frame_*.png` outputs and train/eval loss gap summaries.
- [ ] Add optimizer learning-rate schedules for SH/features, opacity, scales,
  and quats. Current non-position optimizers still use fixed Adam learning
  rates.
- [ ] Decide whether to add an SSIM/DSSIM component or keep the scanner trainer
  intentionally L1-only for this MLX smoke path.
- [ ] Add optional background/mask handling if scanner frames require it.
- [ ] Keep camera optimization and appearance embeddings out of scope until the
  low-level gsplat_core path is stable.

### Remaining Low-Level gsplat API Parity
- [ ] Implement packed projection/rasterize forward paths if dense-only training
  becomes insufficient.
- [ ] Implement `rasterize_to_indices_3dgs` if an index-only render/debug path is
  needed.
- [ ] Finish projection backward GPU/analytic parity and packed backward support
  only after dense scanner training is stable.
- [ ] Continue CUDA/PyTorch parity through exported `.npz` references for forward
  and backward edge cases.

### Remaining SPZ and Viewer Alignment
- [ ] Inspect exported RGB and SH `.spz` files in the target viewer after longer
  scanner training runs.
- [ ] Confirm coordinate, quaternion, opacity-logit, and SH coefficient
  conventions against the viewer and, where possible, gsplat/PyTorch output.
- [ ] Decide whether SPZ export should store trained SH degree exactly or pad /
  clamp to viewer-supported degrees.
