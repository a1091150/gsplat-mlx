# gsplat CUDA 3DGS forward source map

## Scope

This note maps the gsplat CUDA 3DGS forward operators that are candidates for
the `gsplat_core` MLX Metal migration.

Source root:

```text
submodules/gsplat/gsplat/cuda
```

Target root:

```text
gsplat_core
```

This map intentionally focuses on low-level 3DGS forward ops. It excludes 2DGS,
backward/autograd kernels, UT/3DGUT, lidar, Adam, relocation, camera wrappers,
and external distortion.

Training/backward note:

```text
note/mlx_training_gradient_proxy.md
```

When a later training path needs intermediate gradients such as `means2d.grad`,
do not rely on PyTorch-style `retain_grad()`. Use an explicit dummy trainable
gradient proxy, such as `viewspace_points`, and make sure it is selected by
`mx.value_and_grad(..., argnums=...)`.

Current milestone status:

```text
Done:
  dense 3DGS forward low-level render path
  quat/scale -> projection fused dense -> intersect dense -> SH -> rasterize dense
  rasterize backgrounds and tile masks
  exported CUDA .npz comparison for current available dense fixtures

Remaining:
  packed projection
  packed intersect/rasterize with image_ids and gaussian_ids
  intersect segmented sort
  intersect AccuTile/SNUGBOX with conics/opacities
  rasterize_to_indices_3dgs
  high-level gsplat-style Python rasterization wrapper
```

## Dispatch layers

The gsplat CUDA path has four useful layers:

```text
Python public wrapper
  -> Python autograd Function
  -> torch custom op registered in ext.cpp
  -> C++ launcher / CUDA kernel
```

Important files:

```text
gsplat/cuda/_wrapper.py
gsplat/cuda/_torch_impl.py
gsplat/cuda/ext.cpp
gsplat/cuda/csrc/*.cpp
gsplat/cuda/csrc/*.h
gsplat/cuda/csrc/*.cu
```

For MLX migration, `gsplat_core` should usually mirror the custom op level:

```text
gsplat_core/include/gsplat_<op>.h
gsplat_core/gsplat_<op>.cpp
gsplat_core/metal/gsplat_<op>.metal
gsplat_core/binding/binding.cpp
```

## Forward operator map

### `quat_scale_to_covar_preci_fwd`

Python wrapper:

```text
gsplat/cuda/_wrapper.py::quat_scale_to_covar_preci
gsplat/cuda/_wrapper.py::_QuatScaleToCovarPreci.forward
```

Torch op:

```text
quat_scale_to_covar_preci_fwd(
  Tensor quats,
  Tensor scales,
  bool compute_covar,
  bool compute_preci,
  bool triu
) -> (Tensor, Tensor)
```

CUDA source:

```text
gsplat/cuda/csrc/QuatScaleToCovar.cpp
gsplat/cuda/csrc/QuatScaleToCovar.h
gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu
```

Inputs:

```text
quats  [..., 4]
scales [..., 3]
```

Outputs:

```text
covars [..., 3, 3] or [..., 6] if triu
precis [..., 3, 3] or [..., 6] if triu
```

Target:

```text
gsplat_core/include/gsplat_quat_scale_to_covar.h
gsplat_core/gsplat_quat_scale_to_covar.cpp
gsplat_core/metal/gsplat_quat_scale_to_covar.metal
Python: quat_scale_to_covar_preci_forward
```

Status: migrated for the dense low-level forward path. The MLX/Metal path
supports covariance and precision outputs, `triu=true`, `triu=false`, and
empty optional outputs. Existing exported CUDA `.npz` fixture parity passes.

### `spherical_harmonics_fwd`

Python wrapper:

```text
gsplat/cuda/_wrapper.py::spherical_harmonics
gsplat/cuda/_wrapper.py::_SphericalHarmonics.forward
```

Torch op:

```text
spherical_harmonics_fwd(
  int degrees_to_use,
  Tensor dirs,
  Tensor coeffs,
  Tensor? masks
) -> Tensor
```

CUDA source:

```text
gsplat/cuda/csrc/SphericalHarmonics.cpp
gsplat/cuda/csrc/SphericalHarmonics.h
gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu
```

Inputs:

```text
degrees_to_use int
dirs           [..., 3]
coeffs         [..., K, 3]
masks          [...] optional
```

Outputs:

```text
colors [..., 3]
```

Target:

```text
gsplat_core/include/gsplat_spherical_harmonics.h
gsplat_core/gsplat_spherical_harmonics.cpp
gsplat_core/metal/gsplat_spherical_harmonics.metal
Python: spherical_harmonics_forward
```

Status: migrated for the dense low-level forward path. The MLX/Metal path
supports degrees 0 through 4 and optional masks. Existing exported CUDA `.npz`
fixture parity passes; the degree-4 masks export script is available for an
additional CUDA fixture.

### `projection_ewa_3dgs_fused_fwd`

Python wrapper:

```text
gsplat/cuda/_wrapper.py::fully_fused_projection(packed=False)
gsplat/cuda/_wrapper.py::_FullyFusedProjection.forward
```

Torch op:

```text
projection_ewa_3dgs_fused_fwd(
  Tensor means,
  Tensor? covars,
  Tensor? quats,
  Tensor? scales,
  Tensor? opacities,
  Tensor viewmats,
  Tensor Ks,
  int image_width,
  int image_height,
  float eps2d,
  float near_plane,
  float far_plane,
  float radius_clip,
  bool calc_compensations,
  int camera_model
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

CUDA source:

```text
gsplat/cuda/csrc/Projection.cpp
gsplat/cuda/csrc/Projection.h
gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu
```

Inputs:

```text
means     [..., N, 3]
covars    [..., N, 6] optional
quats     [..., N, 4] optional
scales    [..., N, 3] optional
opacities [..., N] optional
viewmats  [..., C, 4, 4]
Ks        [..., C, 3, 3]
```

Outputs:

```text
radii         [..., C, N, 2] int32
means2d       [..., C, N, 2]
depths        [..., C, N]
conics        [..., C, N, 3]
compensations [..., C, N]
```

Target:

```text
gsplat_core/include/gsplat_projection.h
gsplat_core/gsplat_projection.cpp
gsplat_core/metal/gsplat_projection.metal
Python: projection_ewa_3dgs_fused_forward
```

Status: migrated for the dense fused path. Shape, dtype, C++/Metal smoke,
edge-case culling checks, and existing exported CUDA `.npz` parity pass.
The packed projection path remains separate and is not migrated yet.

### `projection_ewa_3dgs_packed_fwd`

Python wrapper:

```text
gsplat/cuda/_wrapper.py::fully_fused_projection(packed=True)
gsplat/cuda/_wrapper.py::_FullyFusedProjectionPacked.forward
```

Torch op:

```text
projection_ewa_3dgs_packed_fwd(...) -> (
  batch_ids,
  camera_ids,
  gaussian_ids,
  indptr,
  radii,
  means2d,
  depths,
  conics,
  compensations
)
```

CUDA source:

```text
gsplat/cuda/csrc/Projection.cpp
gsplat/cuda/csrc/Projection.h
gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu
```

Target:

```text
gsplat_core/include/gsplat_projection.h
gsplat_core/gsplat_projection.cpp
gsplat_core/metal/gsplat_projection_packed.metal
Python: projection_ewa_3dgs_packed_forward
```

Status: not migrated. This is part of the remaining packed-path coverage.

### `intersect_tile`

Python wrapper:

```text
gsplat/cuda/_wrapper.py::isect_tiles
```

Torch op:

```text
intersect_tile(
  Tensor means2d,
  Tensor radii,
  Tensor depths,
  Tensor? conics,
  Tensor? opacities,
  Tensor? image_ids,
  Tensor? gaussian_ids,
  int I,
  int tile_size,
  int tile_width,
  int tile_height,
  bool sort,
  bool segmented
) -> (Tensor, Tensor, Tensor)
```

CUDA source:

```text
gsplat/cuda/csrc/Intersect.cpp
gsplat/cuda/csrc/Intersect.h
gsplat/cuda/csrc/IntersectTile.cu
```

Inputs:

```text
means2d      [..., N, 2] or [nnz, 2]
radii        [..., N, 2] or [nnz, 2]
depths       [..., N] or [nnz]
conics       optional, enables tighter AccuTile/SNUGBOX path
opacities    optional, enables tighter AccuTile/SNUGBOX path
image_ids    optional packed path
gaussian_ids optional packed path
```

Outputs:

```text
tiles_per_gauss int32, [..., N] or [nnz]
isect_ids       int64, [n_isects]
flatten_ids     int32, [n_isects]
```

Target:

```text
gsplat_core/include/gsplat_intersect.h
gsplat_core/gsplat_intersect.cpp
gsplat_core/metal/gsplat_intersect.metal
Python: intersect_tile_forward
```

Status: migrated for the dense AABB path. The implementation uses MLX/Metal
stages for count, prefix, encode, sort, and reorder, and Python-facing
`intersect_tile_forward` matches the existing exported CUDA `.npz` fixture.

Remaining coverage: packed inputs with `image_ids`/`gaussian_ids`, segmented
sort, and the AccuTile/SNUGBOX path that uses `conics` and `opacities`.

### `intersect_offset`

Python wrapper:

```text
gsplat/cuda/_wrapper.py::isect_offset_encode
```

Torch op:

```text
intersect_offset(
  Tensor isect_ids,
  int I,
  int tile_width,
  int tile_height
) -> Tensor
```

CUDA source:

```text
gsplat/cuda/csrc/Intersect.cpp
gsplat/cuda/csrc/Intersect.h
gsplat/cuda/csrc/IntersectTile.cu
```

Outputs:

```text
offsets [I, tile_height, tile_width]
```

Target:

```text
gsplat_core/include/gsplat_intersect.h
gsplat_core/gsplat_intersect.cpp
gsplat_core/metal/gsplat_intersect.metal
Python: intersect_offset_forward
```

Status: migrated. The MLX/Metal path is used by Python-facing
`intersect_offset_forward` and by the dense forward chain.

### `rasterize_to_pixels_3dgs_fwd`

Python wrapper:

```text
gsplat/cuda/_wrapper.py::rasterize_to_pixels
gsplat/cuda/_wrapper.py::_RasterizeToPixels.forward
```

Torch op:

```text
rasterize_to_pixels_3dgs_fwd(
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
  Tensor flatten_ids
) -> (Tensor, Tensor, Tensor)
```

CUDA source:

```text
gsplat/cuda/csrc/Rasterization.cpp
gsplat/cuda/csrc/Rasterization.h
gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu
```

Outputs:

```text
render_colors [..., image_height, image_width, channels]
render_alphas [..., image_height, image_width, 1]
last_ids      [..., image_height, image_width]
```

Target:

```text
gsplat_core/include/gsplat_rasterize.h
gsplat_core/gsplat_rasterize.cpp
gsplat_core/metal/gsplat_rasterize.metal
Python: rasterize_to_pixels_3dgs_forward
```

Status: migrated for the dense low-level forward path. The MLX/Metal path
supports front-to-back compositing, optional backgrounds, and optional tile
masks. Existing exported CUDA `.npz` fixture parity passes for the base dense
case; the rasterize masks export script is available for an additional CUDA
fixture.

Remaining coverage: packed rasterize input layout.

### `rasterize_to_indices_3dgs`

Python wrapper:

```text
gsplat/cuda/_wrapper.py::rasterize_to_indices_in_range
```

Torch op:

```text
rasterize_to_indices_3dgs(
  int range_start,
  int range_end,
  Tensor transmittances,
  Tensor means2d,
  Tensor conics,
  Tensor opacities,
  int image_width,
  int image_height,
  int tile_size,
  Tensor tile_offsets,
  Tensor flatten_ids
) -> (Tensor, Tensor)
```

CUDA source:

```text
gsplat/cuda/csrc/Rasterization.cpp
gsplat/cuda/csrc/Rasterization.h
gsplat/cuda/csrc/RasterizeToIndices3DGS.cu
```

Outputs:

```text
out_gauss_ids [M]
out_indices   [M]
```

Target:

```text
gsplat_core/include/gsplat_rasterize.h
gsplat_core/gsplat_rasterize.cpp
gsplat_core/metal/gsplat_rasterize_indices.metal
Python: rasterize_to_indices_3dgs_forward
```

Status: not migrated. This is outside the current dense render-pixels forward
milestone and remains part of fuller gsplat forward coverage.

## Excluded source map

These are explicitly out of scope for the first 3DGS forward milestone:

```text
2DGS:
  Projection2DGSFused.cu
  Projection2DGSPacked.cu
  RasterizeToPixels2DGSFwd.cu
  RasterizeToPixels2DGSBwd.cu
  RasterizeToIndices2DGS.cu

Backward:
  *_bwd torch ops
  RasterizeToPixels3DGSBwd.cu
  ProjectionEWA3DGSFused.cu backward section

UT / 3DGUT / rolling shutter / world-ray:
  ProjectionUT3DGSFused.cu
  RasterizeToPixelsFromWorld3DGSFwd.cu
  RasterizeToPixelsFromWorld3DGSBwd.cu

Lidar:
  Lidar.cpp
  IntersectTileLidar.cu
  _lidar.py
  _torch_impl_lidar.py

Other systems:
  Adam.cpp
  AdamCUDA.cu
  Relocation.cpp
  RelocationCUDA.cu
  MCMCPerturb.cpp
  MCMCPerturbCUDA.cu
  CameraWrappers.cu
  ExternalDistortionWrappers.cu
```

## Recommended next slice

The next implementation slice should be projection parity, not a new operator.

Reason:

```text
projection_ewa_3dgs_fused_fwd already has gsplat_core files, binding, Metal,
and C++ smoke tests. It needs numeric parity before intersect/rasterize depend
on its outputs.
```

Suggested next checks:

```text
1. Add fixed C++ projection fixtures for pinhole + identity viewmat.
2. Check numeric means2d/depths/radii/conics for quats/scales path.
3. Add covars path fixture.
4. Only after projection parity, move to intersect_tile/intersect_offset.
```
