# gsplat 3DGS Forward Path with SH Degree

This note records the planned forward path for the gsplat 3DGS renderer when
colors are produced from spherical harmonics coefficients.

The scope is 3DGS forward only. It excludes 2DGS, backward kernels, UT, lidar,
Adam, relocation, and high-level camera wrapper systems.

## Forward Path

```text
input 3D Gaussians
  means3d
  quats / scales or covars
  opacities
  SH coefficients
  viewmats
  Ks
  image/camera params

-> projection
-> view directions for SH
-> spherical harmonics
-> optional opacity compensation
-> intersect tile
-> intersect offset
-> rasterize to pixels
-> render_colors / render_alphas / last_ids
```

## 1. Projection

Source CUDA path:

```text
submodules/gsplat/gsplat/cuda/csrc/Projection.cpp
submodules/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu
```

Current gsplat_core path:

```text
Primitive: GSPlatProjectionEWA3DGSFused
Metal:     gsplat_core/metal/gsplat_projection.metal
API:       projection_ewa_3dgs_fused_forward
```

Outputs:

```text
radii
means2d
depths
conics
compensations optional
```

Status:

```text
Metal path implemented for fused 3DGS projection.
Packed projection remains out of scope for the current first milestone.
```

## 2. View Directions for SH

The SH path needs view directions with shape compatible with projected
Gaussians:

```text
dirs [..., C, N, 3]
```

The high-level wrapper is not implemented yet. For now, scripts and callers
prepare `dirs` manually.

Future high-level wrapper should compute directions from camera centers and
Gaussian means, matching gsplat CUDA/PyTorch semantics.

## 3. Spherical Harmonics

Source CUDA path:

```text
submodules/gsplat/gsplat/cuda/csrc/SphericalHarmonics.cpp
submodules/gsplat/gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu
```

Current gsplat_core path:

```text
Primitive: GSPlatSphericalHarmonics
Metal:     gsplat_core/metal/gsplat_spherical_harmonics.metal
API:       spherical_harmonics_forward
```

Inputs:

```text
degrees_to_use
dirs
coeffs
masks optional
```

Output:

```text
colors [..., C, N, 3]
```

Status:

```text
Metal path implemented for degree 0 through degree 4.
CPU reference fallback remains available.
CUDA numeric parity still needs exported reference comparison.
```

## 4. Optional Opacity Compensation

If projection is called with:

```text
calc_compensations=True
```

then the high-level path should apply:

```text
effective_opacities = opacities * compensations
```

This is currently caller-side logic. It is not yet wrapped in a high-level
`gsplat_core` rasterization API.

## 5. Intersect Tile

Source CUDA path:

```text
submodules/gsplat/gsplat/cuda/csrc/Intersect.cpp
submodules/gsplat/gsplat/cuda/csrc/IntersectTile.cu
```

Current gsplat_core path:

```text
API: intersect_tile_forward
Current implementation: C++ reference path
```

Planned internal decomposition:

```text
GSPlatIntersectTileCount
  -> fixed-shape tile count output

prefix sum / offset preparation
  -> needed for dynamic intersection output length

GSPlatIntersectTileEncode
  -> isect_ids and flatten_ids

sort
  -> sort isect_ids and reorder flatten_ids
```

Outputs:

```text
tiles_per_gauss
isect_ids
flatten_ids
```

## 6. Intersect Offset

Source CUDA path:

```text
submodules/gsplat/gsplat/cuda/csrc/Intersect.cpp
```

Current gsplat_core path:

```text
API: intersect_offset_forward
Current implementation: C++ reference path
```

Planned primitive:

```text
Primitive: GSPlatIntersectOffset
Metal:     gsplat_core/metal/gsplat_intersect.metal
```

Input:

```text
sorted isect_ids
```

Output:

```text
tile_offsets [I, tile_height, tile_width]
```

## 7. Rasterize

Source CUDA path:

```text
submodules/gsplat/gsplat/cuda/csrc/Rasterization.cpp
submodules/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu
```

Current gsplat_core path:

```text
API: rasterize_to_pixels_3dgs_forward
Current implementation: C++ reference path
```

Inputs:

```text
means2d
conics
colors from SH
opacities or compensated opacities
tile_offsets
flatten_ids
backgrounds optional
masks optional future
```

Outputs:

```text
render_colors
render_alphas
last_ids
```

## Relationship to Task 3.11A

The full SH degree forward path is:

```text
projection -> dirs -> spherical_harmonics -> opacity compensation
-> intersect_tile -> intersect_offset -> rasterize
```

Task 3.11A is not the full forward path. It is one sub-step inside
`intersect_tile`:

```text
intersect_tile
  -> tile count       Task 3.11A
  -> prefix sum       later
  -> encode ids       later
  -> sort             later
```

Task 3.11A is still the same overall direction because intersect tile is a
required stage of the 3DGS forward renderer. The purpose of 3.11A is to move the
first fixed-shape piece of intersect from C++ reference code to Metal before
tackling dynamic outputs.

## Planned Task Order

```text
Task 3.11A - Intersect Tile Count Metal
Task 3.11B - Intersect Offset Metal
Task 3.11C - Intersect Tile Encode Dense First Version
Task 3.11D - Intersect Full Dense AABB GPU Path
Task 3.12  - Rasterize 3DGS Metal First Version
Task 3.13  - High-Level SH Degree Forward Wrapper
```

## Current Metal Status

Implemented Metal paths:

```text
GSPlatProjectionEWA3DGSFused
GSPlatSphericalHarmonics
GSPlatQuatScaleToCovarPreci
```

Still C++ reference paths:

```text
intersect_tile_forward
intersect_offset_forward
rasterize_to_pixels_3dgs_forward
```

The next recommended implementation slice is:

```text
Task 3.11A - Intersect Tile Count Metal
```
