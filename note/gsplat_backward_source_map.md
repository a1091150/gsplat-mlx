# gsplat 3DGS backward source map

## Scope

This note maps the gsplat CUDA 3DGS dense backward operators that should back
the first MLX Metal backward milestone.

The first milestone follows the already-migrated dense forward chain:

```text
quat/scale -> projection fused dense -> intersect dense -> SH -> rasterize dense
```

Packed backward, 2DGS backward, UT/world-ray backward, lidar, Adam, relocation,
MCMC perturb, camera wrappers, and external distortion stay out of scope.

## MLX strategy

Use explicit low-level backward functions first. They are easier to compare
against CUDA `.npz` fixtures than autograd-integrated primitive `vjp(...)`.
After an explicit backward op is stable, wire the corresponding MLX Primitive
`vjp(...)` to call it.

Keep `jvp(...)` unsupported until there is a concrete forward-mode need.

## Training gradient rule

PyTorch can use `retain_grad()` to keep intermediate gradients such as
`means2d.grad`. MLX gradients are selected through
`mx.value_and_grad(fn, argnums=...)`.

If training or after-training logic needs screen-space `means2d` gradients,
use a visible dummy trainable parameter such as `viewspace_points` and include
it in `argnums`. Backward APIs should preserve this path for densify, clone,
split, and related after-training logic.

## Shared CUDA registration references

Python wrapper and autograd saved tensors:

```text
submodules/gsplat/gsplat/cuda/_wrapper.py
```

Torch custom op schema and C++ bindings:

```text
submodules/gsplat/gsplat/cuda/ext.cpp
```

## Backward ops

### `spherical_harmonics_bwd`

Python autograd function:

```text
_wrapper.py::_SphericalHarmonics.backward
```

CUDA/C++ sources:

```text
gsplat/cuda/csrc/SphericalHarmonics.cpp
gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu
gsplat/cuda/csrc/SphericalHarmonics.h
```

CUDA schema:

```text
spherical_harmonics_bwd(
  int degrees_to_use,
  Tensor dirs,
  Tensor coeffs,
  Tensor? masks,
  Tensor v_colors,
  bool compute_v_dirs
) -> (Tensor, Tensor)
```

Forward saved tensors:

```text
dirs   [..., 3]
coeffs [..., K, 3]
masks  [...] optional
```

Incoming cotangent:

```text
v_colors [..., 3]
```

Returned gradients:

```text
v_dirs   [..., 3] optional, only if dirs needs grad
v_coeffs [..., K, 3]
```

First MLX API candidate:

```text
spherical_harmonics_backward(
  degrees_to_use,
  inputs={dirs, coeffs, masks?},
  cotangents={v_colors},
  compute_v_dirs
) -> {v_dirs?, v_coeffs}
```

Notes:

- This is the smallest backward op and is a good first implementation slice.
- Masks are non-differentiable and return no gradient.

### `quat_scale_to_covar_preci_bwd`

Python autograd function:

```text
_wrapper.py::_QuatScaleToCovarPreci.backward
```

CUDA/C++ sources:

```text
gsplat/cuda/csrc/QuatScaleToCovar.cpp
gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu
gsplat/cuda/csrc/QuatScaleToCovar.h
```

CUDA schema:

```text
quat_scale_to_covar_preci_bwd(
  Tensor quats,
  Tensor scales,
  bool triu,
  Tensor? v_covars,
  Tensor? v_precis
) -> (Tensor, Tensor)
```

Forward saved tensors:

```text
quats  [..., 4]
scales [..., 3]
```

Incoming cotangents:

```text
v_covars [..., 3, 3] or [..., 6] optional
v_precis [..., 3, 3] or [..., 6] optional
```

Returned gradients:

```text
v_quats  [..., 4]
v_scales [..., 3]
```

First MLX API candidate:

```text
quat_scale_to_covar_preci_backward(
  inputs={quats, scales},
  cotangents={v_covars?, v_precis?},
  triu
) -> {v_quats, v_scales}
```

Notes:

- If both cotangents are absent, CUDA returns zero gradients.
- This is also a compact backward slice, but SH is simpler.

### `projection_ewa_3dgs_fused_bwd`

Python autograd function:

```text
_wrapper.py::_FullyFusedProjection.backward
```

CUDA/C++ sources:

```text
gsplat/cuda/csrc/Projection.cpp
gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu
```

CUDA schema:

```text
projection_ewa_3dgs_fused_bwd(
  Tensor means,
  Tensor? covars,
  Tensor? quats,
  Tensor? scales,
  Tensor viewmats,
  Tensor Ks,
  int image_width,
  int image_height,
  float eps2d,
  int camera_model,
  Tensor radii,
  Tensor conics,
  Tensor? compensations,
  Tensor v_means2d,
  Tensor v_depths,
  Tensor v_conics,
  Tensor? v_compensations,
  bool viewmats_requires_grad
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

Forward saved tensors:

```text
means         [..., N, 3]
covars        [..., N, 6] optional
quats         [..., N, 4] optional
scales        [..., N, 3] optional
viewmats      [..., C, 4, 4]
Ks            [..., C, 3, 3]
radii         [..., C, N, 2]
conics        [..., C, N, 3]
compensations [..., C, N] optional
```

Saved scalar params:

```text
image_width
image_height
eps2d
camera_model
```

Incoming cotangents:

```text
v_means2d       [..., C, N, 2]
v_depths        [..., C, N]
v_conics        [..., C, N, 3]
v_compensations [..., C, N] optional
```

Returned gradients:

```text
v_means    [..., N, 3]
v_covars   [..., N, 6] optional
v_quats    [..., N, 4] optional
v_scales   [..., N, 3] optional
v_viewmats [..., C, 4, 4] optional
```

No gradients are returned for:

```text
Ks
image_width
image_height
eps2d
near_plane
far_plane
radius_clip
calc_compensations
camera_model
```

First MLX API candidate:

```text
projection_ewa_3dgs_fused_backward(
  inputs={means, covars?, quats?, scales?, viewmats, Ks},
  forward_outputs={radii, conics, compensations?},
  cotangents={v_means2d, v_depths, v_conics, v_compensations?},
  image_width,
  image_height,
  eps2d,
  camera_model,
  viewmats_requires_grad
) -> {v_means, v_covars?, v_quats?, v_scales?, v_viewmats?}
```

Notes:

- `covars` and `quats/scales` are mutually exclusive, same as forward.
- `v_radii` is ignored by CUDA because radii are integer/non-differentiable.
- Projection backward is important for training but should come after smaller
  backward API and fixture patterns are established.

### `rasterize_to_pixels_3dgs_bwd`

Python autograd function:

```text
_wrapper.py::_RasterizeToPixels.backward
```

CUDA/C++ sources:

```text
gsplat/cuda/csrc/Rasterization.cpp
gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu
gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu
```

CUDA schema:

```text
rasterize_to_pixels_3dgs_bwd(
  Tensor means2d,
  Tensor conics,
  Tensor colors,
  Tensor opacities,
  Tensor? backgrounds,
  Tensor? masks,
  int image_width,
  int image_height,
  int tile_size,
  Tensor tile_offsets,
  Tensor flatten_ids,
  Tensor render_alphas,
  Tensor last_ids,
  Tensor v_render_colors,
  Tensor v_render_alphas,
  bool absgrad
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

Forward saved tensors:

```text
means2d       [..., N, 2]
conics        [..., N, 3]
colors        [..., N, channels]
opacities     [..., N]
backgrounds   [..., channels] optional
masks         [..., tile_height, tile_width] optional
tile_offsets  [..., tile_height, tile_width]
flatten_ids   [n_isects]
render_alphas [..., image_height, image_width, 1]
last_ids      [..., image_height, image_width]
```

Saved scalar params:

```text
image_width
image_height
tile_size
absgrad
```

Incoming cotangents:

```text
v_render_colors [..., image_height, image_width, channels]
v_render_alphas [..., image_height, image_width, 1]
```

Returned gradients from CUDA op:

```text
v_means2d_abs [..., N, 2] optional, only when absgrad=true
v_means2d     [..., N, 2]
v_conics      [..., N, 3]
v_colors      [..., N, channels]
v_opacities   [..., N]
```

Additional Python-wrapper gradient:

```text
v_backgrounds = sum(v_render_colors * (1 - render_alphas), dim=(-3, -2))
```

No gradients are returned for:

```text
masks
image_width
image_height
tile_size
tile_offsets
flatten_ids
absgrad
```

First MLX API candidate:

```text
rasterize_to_pixels_3dgs_backward(
  inputs={means2d, conics, colors, opacities, backgrounds?, masks?,
          tile_offsets, flatten_ids},
  forward_outputs={render_alphas, last_ids},
  cotangents={v_render_colors, v_render_alphas},
  image_width,
  image_height,
  tile_size,
  absgrad
) -> {v_means2d_abs?, v_means2d, v_conics, v_colors,
      v_opacities, v_backgrounds?}
```

Notes:

- CUDA forward stores `render_alphas` and `last_ids` specifically for backward.
- The CUDA comments mention transmittance precision tradeoffs; MLX should keep
  the saved-state contract first and optimize later.
- `absgrad` is relevant to gsplat's screen-space gradient workflows. The first
  implementation should decide whether to expose it immediately or return a
  deterministic empty optional output when disabled.
- This op is central for training and should be implemented after the smaller
  SH/quat backward patterns are proven.

## First implementation recommendation

Start with `spherical_harmonics_backward`:

- It has one cotangent and two differentiable input groups.
- Optional masks are non-differentiable and already supported in forward.
- It provides a clean path to test explicit backward APIs, CUDA `.npz` export,
  C++/Metal smoke, and later Primitive `vjp(...)` wiring.

Then implement `quat_scale_to_covar_preci_backward`, followed by rasterize and
projection.
