# gsplat CUDA Reference Export Scripts

Run these scripts on a CUDA machine or Colab environment with PyTorch and
`gsplat` installed. They export deterministic `.npz` files that contain both
inputs and gsplat CUDA reference outputs.

Example:

```bash
python scripts/export_ref/export_projection_ewa_3dgs_fused_forward.py \
  --out refs/projection_ewa_3dgs_fused_forward.npz
```

Available scripts:

```text
export_projection_ewa_3dgs_fused_forward.py
export_projection_ewa_3dgs_fused_backward.py
export_projection_ewa_3dgs_fused_edge_cases.py
export_intersect_tile_forward.py
export_rasterize_to_pixels_3dgs_backward.py
export_rasterize_to_pixels_3dgs_forward.py
export_rasterize_to_pixels_3dgs_masks.py
export_spherical_harmonics_forward.py
export_spherical_harmonics_degree4_masks.py
export_spherical_harmonics_backward.py
export_quat_scale_to_covar_preci_backward.py
export_quat_scale_to_covar_preci_forward.py
export_quat_scale_to_covar_preci_edge_cases.py
export_forward_3dgs_chain.py
export_training_reference_config.py
```

The `.npz` keys use:

```text
input__*
ref__*
```

The Mac/MLX side should load the same `input__*` arrays, run `gsplat_core`,
and compare against `ref__*`.

After copying the exported fixtures into `refs/`, compare them on the Mac/MLX
side with:

```bash
conda run -n fastgs_core python scripts/test/compare_exported_npz.py
```

Pass one or more paths to compare a subset:

```bash
conda run -n fastgs_core python scripts/test/compare_exported_npz.py \
  refs/forward_3dgs_chain.npz
```

Training reference planning:

```bash
python scripts/export_ref/export_training_reference_config.py \
  --out refs/training_reference_config.json
```

This writes a fixed-seed JSON contract for longer CUDA gsplat training
comparisons. After a CUDA-side trainer writes
`refs/training_reference_summary_cuda.json`, compare it with the MLX scanner
training summary using:

```bash
conda run -n fastgs_core python scripts/test/compare_training_summary.py \
  --mlx outputs/scanner_points_multiview_train/training_summary.json \
  --cuda refs/training_reference_summary_cuda.json
```
