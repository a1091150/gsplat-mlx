# gsplat 3DGS backward API and fixture schema

## Purpose

This note fixes the first-version explicit backward API style and CUDA `.npz`
fixture layout before any backward kernel implementation starts.

The API is intentionally explicit first:

```text
Python binding -> explicit C++ backward entry -> MLX Primitive + Metal kernel
```

After explicit backward parity is stable, the matching forward Primitive
`vjp(...)` can call the explicit backward path.

## Naming

Python-facing names should mirror gsplat CUDA op names but use the project
style already established in `gsplat_core`:

```text
spherical_harmonics_backward(...)
quat_scale_to_covar_preci_backward(...)
rasterize_to_pixels_3dgs_backward(...)
projection_ewa_3dgs_fused_backward(...)
```

C++ entry points should use `gsplat_` prefixes:

```text
gsplat_spherical_harmonics_backward(...)
gsplat_quat_scale_to_covar_preci_backward(...)
gsplat_rasterize_to_pixels_3dgs_backward(...)
gsplat_projection_ewa_3dgs_fused_backward(...)
```

Metal kernel names should include `_backward_kernel`:

```text
gsplat_spherical_harmonics_backward_kernel
gsplat_quat_scale_to_covar_preci_backward_kernel
gsplat_rasterize_to_pixels_3dgs_backward_kernel
gsplat_projection_ewa_3dgs_fused_backward_kernel
```

## Python binding shape

Use dictionaries for array groups and keyword arguments for scalars:

```python
outputs = some_backward(
    inputs={...},
    cotangents={...},
    forward_outputs={...},  # only when needed
    scalar_flag=...,
)
```

Rules:

- `inputs` contains differentiable forward inputs and any optional forward
  inputs needed by CUDA backward.
- `forward_outputs` contains saved forward outputs required by backward, such
  as `render_alphas`, `last_ids`, `radii`, `conics`, and `compensations`.
- `cotangents` contains incoming gradients from later graph nodes.
- Optional arrays are omitted when absent.
- Optional outputs are omitted from the returned dict when not requested or not
  applicable.
- Empty MLX arrays may be used internally in C++ for optional inputs, but Python
  scripts and `.npz` fixtures should prefer omitted keys.

## `.npz` key convention

Backward fixtures should use these prefixes:

```text
input__*       forward inputs or non-differentiable saved inputs
fwd__*         forward outputs saved for backward
cotangent__*   incoming gradient tensors
ref__v_*       CUDA reference gradients
meta__*        scalar metadata when the value is not naturally an input
```

Examples:

```text
input__dirs
input__coeffs
input__masks
cotangent__v_colors
ref__v_dirs
ref__v_coeffs
meta__compute_v_dirs
```

Scalar params may use either `input__*` or `meta__*`. Use `input__*` when the
scalar corresponds directly to the CUDA op argument, and `meta__*` for fixture
control flags.

Optional reference gradients:

- If an optional gradient is not requested, omit `ref__v_*`.
- If CUDA returns an empty optional tensor because a path is disabled, omit the
  key rather than storing a zero-length array.
- Comparers must check key existence before comparing optional gradients.

## Export script names

First-version CUDA export scripts:

```text
scripts/export_ref/export_spherical_harmonics_backward.py
scripts/export_ref/export_quat_scale_to_covar_preci_backward.py
scripts/export_ref/export_rasterize_to_pixels_3dgs_backward.py
scripts/export_ref/export_projection_ewa_3dgs_fused_backward.py
```

Expected fixture names:

```text
refs/spherical_harmonics_backward.npz
refs/quat_scale_to_covar_preci_backward.npz
refs/rasterize_to_pixels_3dgs_backward.npz
refs/projection_ewa_3dgs_fused_backward.npz
```

## Comparer names

Add comparers to `scripts/test/compare_exported_npz.py`:

```text
compare_spherical_harmonics_backward
compare_quat_scale_backward
compare_rasterize_backward
compare_projection_backward
```

The forward and backward rasterize comparers should have distinct names in code
to avoid confusion with the existing forward comparer.

## Op schemas

### `spherical_harmonics_backward`

Python API:

```python
spherical_harmonics_backward(
    degrees_to_use: int,
    inputs: dict[str, mx.array],
    cotangents: dict[str, mx.array],
    compute_v_dirs: bool = True,
) -> dict[str, mx.array]
```

Required keys:

```text
inputs:
  dirs
  coeffs

cotangents:
  v_colors
```

Optional keys:

```text
inputs:
  masks
```

Returned keys:

```text
v_coeffs
v_dirs   optional, present only when compute_v_dirs=true
```

Fixture keys:

```text
input__degrees_to_use
input__dirs
input__coeffs
input__masks optional
cotangent__v_colors
meta__compute_v_dirs
ref__v_coeffs
ref__v_dirs optional
```

### `quat_scale_to_covar_preci_backward`

Python API:

```python
quat_scale_to_covar_preci_backward(
    inputs: dict[str, mx.array],
    cotangents: dict[str, mx.array],
    triu: bool = True,
) -> dict[str, mx.array]
```

Required keys:

```text
inputs:
  quats
  scales
```

Optional keys:

```text
cotangents:
  v_covars
  v_precis
```

Returned keys:

```text
v_quats
v_scales
```

Fixture keys:

```text
input__quats
input__scales
input__triu
cotangent__v_covars optional
cotangent__v_precis optional
ref__v_quats
ref__v_scales
```

### `rasterize_to_pixels_3dgs_backward`

Python API:

```python
rasterize_to_pixels_3dgs_backward(
    inputs: dict[str, mx.array],
    forward_outputs: dict[str, mx.array],
    cotangents: dict[str, mx.array],
    image_width: int,
    image_height: int,
    tile_size: int,
    absgrad: bool = False,
) -> dict[str, mx.array]
```

Required keys:

```text
inputs:
  means2d
  conics
  colors
  opacities
  tile_offsets
  flatten_ids

forward_outputs:
  render_alphas
  last_ids

cotangents:
  v_render_colors
  v_render_alphas
```

Optional keys:

```text
inputs:
  backgrounds
  masks
```

Returned keys:

```text
v_means2d
v_conics
v_colors
v_opacities
v_backgrounds optional, present only when backgrounds exists
v_means2d_abs optional, present only when absgrad=true
```

Fixture keys:

```text
input__image_width
input__image_height
input__tile_size
input__absgrad
input__means2d
input__conics
input__colors
input__opacities
input__backgrounds optional
input__masks optional
input__tile_offsets
input__flatten_ids
fwd__render_alphas
fwd__last_ids
cotangent__v_render_colors
cotangent__v_render_alphas
ref__v_means2d
ref__v_conics
ref__v_colors
ref__v_opacities
ref__v_backgrounds optional
ref__v_means2d_abs optional
```

Notes:

- CUDA computes `v_backgrounds` in the Python wrapper, not inside
  `rasterize_to_pixels_3dgs_bwd`. The MLX explicit backward API should still
  return it for Python-facing parity.
- `last_ids` and `render_alphas` are saved forward outputs and should be stored
  under `fwd__*`.

### `projection_ewa_3dgs_fused_backward`

Python API:

```python
projection_ewa_3dgs_fused_backward(
    inputs: dict[str, mx.array],
    forward_outputs: dict[str, mx.array],
    cotangents: dict[str, mx.array],
    image_width: int,
    image_height: int,
    eps2d: float = 0.3,
    camera_model: int = 0,
    viewmats_requires_grad: bool = False,
) -> dict[str, mx.array]
```

Required keys:

```text
inputs:
  means
  viewmats
  Ks

forward_outputs:
  radii
  conics

cotangents:
  v_means2d
  v_depths
  v_conics
```

Mutually exclusive input paths:

```text
inputs:
  covars

or:
  quats
  scales
```

Optional keys:

```text
forward_outputs:
  compensations

cotangents:
  v_compensations
```

Returned keys:

```text
v_means
v_covars optional, present only for covars input path
v_quats optional, present only for quats/scales input path
v_scales optional, present only for quats/scales input path
v_viewmats optional, present only when viewmats_requires_grad=true
```

Fixture keys:

```text
input__image_width
input__image_height
input__eps2d
input__camera_model
input__viewmats_requires_grad
input__means
input__covars optional
input__quats optional
input__scales optional
input__viewmats
input__Ks
fwd__radii
fwd__conics
fwd__compensations optional
cotangent__v_means2d
cotangent__v_depths
cotangent__v_conics
cotangent__v_compensations optional
ref__v_means
ref__v_covars optional
ref__v_quats optional
ref__v_scales optional
ref__v_viewmats optional
```

Notes:

- `v_radii` is intentionally absent because radii are integer outputs.
- `near_plane`, `far_plane`, `radius_clip`, and `calc_compensations` affect
  forward saved outputs but are not direct CUDA backward arguments.
- First fixtures should cover the quats/scales path. Covars path can be a
  second fixture once the first backward op is stable.

## First implementation slice contract

Task 6.3 should implement only:

```text
spherical_harmonics_backward(...)
```

Minimum fixture:

```text
refs/spherical_harmonics_backward.npz
```

The fixture should include:

```text
input__degrees_to_use
input__dirs
input__coeffs
input__masks
cotangent__v_colors
meta__compute_v_dirs = true
ref__v_dirs
ref__v_coeffs
```

Validation target:

```text
max_abs_diff <= 1.0e-4 for v_dirs and v_coeffs
```
