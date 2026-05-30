# Task 360: Train Mip-NeRF 360 / COLMAP Scenes

## Understanding

The downloaded dataset lives at:

```text
submodules/gsplat/examples/datasets/data/360_v2
```

Each scene, for example `garden`, `stump`, `bicycle`, follows the `gsplat` example layout:

```text
<scene>/
  images/
  images_2/
  images_4/
  images_8/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

`submodules/gsplat/examples/datasets/colmap.py` is the reference reader. It uses `pycolmap.SceneManager` to load COLMAP cameras, images, and sparse points; sorts images by filename; maps the original `images/` names to the selected factor folder such as `images_4/`; returns camera intrinsics `K`, camera-to-world poses, RGB targets, sparse points, and sparse point colors. The train/val split is index based: training images are `index % test_every != 0`, validation images are `index % test_every == 0`.

The current Makefile training targets:

```text
make codex-scanner-points-train-spz
make codex-scanner-points-train-spz-refine
```

currently call `scripts/test/train_scanner_points_multiview_3dgs_mlx.py`. That script expects the scanner dataset format:

```text
frame_*.jpg
frame_*.json
points.ply
```

So the 360 task is not just changing `SCANNER_DATASET`. We need a COLMAP/360 loader that provides the same training ingredients the scanner trainer already needs: frame list, `viewmat` world-to-camera matrices, scaled `K`, RGB target images, and initial sparse points/colors.

## Goal

Add support for training one Mip-NeRF 360 scene from `submodules/gsplat/examples/datasets/data/360_v2/<scene>` with the existing MLX 3DGS training path, then export:

```text
trained_scanner_points.spz
trained_model_params.npz
debug/training metadata and preview renders
```

The implementation should stay close to `gsplat`'s COLMAP parser behavior so that camera transforms, image factor handling, and sparse point initialization match the upstream example.

## Training Process And Parameters

Training behavior and defaults should follow `submodules/gsplat/examples/simple_trainer.py`.

Reference command shape from `gsplat`:

```sh
python simple_trainer.py default \
  --eval_steps -1 \
  --disable_viewer \
  --data_factor <2-or-4> \
  --render_traj_path ellipse \
  --data_dir data/360_v2/<scene>/ \
  --result_dir results/benchmark/<scene>/
```

The MLX target can expose its own Makefile variables, but the official target's resolved training values should match these `gsplat` defaults:

| Setting | gsplat default / benchmark value | MLX task target |
| --- | --- | --- |
| Dataset type | `colmap` | `--data-type colmap` |
| Data factor | `4` default; benchmark uses `2` for `bonsai`, `counter`, `kitchen`, `room`; `4` for `garden`, `bicycle`, `stump` | `COLMAP_360_FACTOR` follows the same scene rule |
| Train/val split | `test_every=8` | `COLMAP_360_TEST_EVERY=8` |
| World normalization | `normalize_world_space=True` | enable by default |
| Batch size | `1` | `COLMAP_360_BATCH_SIZE=1` |
| Max steps | `30000` | use `30000` for full 360 runs |
| Eval/save steps | `7000`, `30000` in upstream checkpoint/eval flow | current MLX entrypoint writes final SPZ/NPZ/summary; milestone save/eval remains to add if needed |
| Initialization | `init_type=sfm` | initialize from COLMAP `points3D.bin` |
| Initial opacity | `0.1` for `default` strategy | `opacity=0.1` |
| Initial scale | average distance to 3 nearest neighbors times `init_scale=1.0` | implement KNN-based scale instead of fixed `point_scale` for 360 |
| SH degree | `3` | `max_sh_degree=3`, final active degree 3 |
| SH schedule | `min(step // 1000, 3)` | active SH degree increases every 1000 steps |
| Loss | `lerp(L1, 1 - SSIM, 0.2)` | `loss_mode=l1_dssim`, `ssim_lambda=0.2` |
| Means LR | `1.6e-4 * scene_scale`, exponential decay to 1% over max steps | match scaled LR and decay |
| Scales LR | `5e-3` | match |
| Opacity LR | `5e-2` | match |
| Quat LR | `1e-3` | match |
| SH DC LR | `2.5e-3` | match |
| SH rest LR | `2.5e-3 / 20 = 1.25e-4` | match |
| Densification | `DefaultStrategy` | refine/densify enabled for full runs |
| Refine start/stop/every | `500` / `15000` / `100` | match |
| Opacity reset | every `3000` steps | match |
| Prune opacity | `0.005` | match |
| Grow grad2d | `0.0002` | match |
| Grow scale3d / scale2d | `0.01` / `0.05` | match |
| Prune scale3d / scale2d | `0.1` / `0.15` | match |
| MLX cache limit | local runtime setting | `32 GiB` via `mx.set_cache_limit` |

`gsplat` also has an `mcmc` benchmark mode. The first implementation should target `default` because the current MLX trainer already has a gsplat-style default refine path. Add `mcmc` only after the COLMAP/default pipeline is correct.

## Proposed Implementation

1. Add a reusable COLMAP dataset loader.
   - Create a small module under `scripts/test/`, for example `colmap_360_dataset.py`.
   - Reference `submodules/gsplat/examples/datasets/colmap.py` for behavior.
   - Load `sparse/0` first and fall back to `sparse`.
   - Read `cameras.bin`, `images.bin`, and `points3D.bin` through `pycolmap.Reconstruction` in this environment.
   - Sort images by filename to match gsplat metrics/splits.
   - Select image directory by factor: `images` for factor `1`, `images_2`, `images_4`, or `images_8` for downsampled training.
   - Scale `K` by the selected image size, matching the reference parser's actual-image-size correction.
   - Convert COLMAP world-to-camera data to the row-major `viewmat` shape currently used by the MLX renderer.
   - Return sparse points and RGB colors from `points3D.bin`.

2. Generalize the scanner trainer input path.
   - Keep `train_scanner_points_multiview_3dgs_mlx.py` as the training engine.
   - Add arguments such as:

```text
--data-type scanner|colmap
--data-factor 4
--test-every 8
--normalize-world
```

   - For `scanner`, preserve current behavior.
   - For `colmap`, call the new 360/COLMAP loader instead of `collect_frames`, `load_camera`, `load_target`, and `prepare_points`.
   - For `colmap`, initialize Gaussian scale with gsplat's SFM rule: average distance to the 3 nearest neighbors, converted to log scale.
   - For `colmap`, compute `scene_scale` as gsplat does: parser camera extent times `1.1 * global_scale`.

3. Add Makefile variables and target aliases.
   - Suggested defaults:

```make
COLMAP_360_ROOT ?= submodules/gsplat/examples/datasets/data/360_v2
COLMAP_360_SCENE ?= garden
COLMAP_360_DATA ?= $(COLMAP_360_ROOT)/$(COLMAP_360_SCENE)
COLMAP_360_FACTOR ?= $(if $(filter bonsai counter kitchen room,$(COLMAP_360_SCENE)),2,4)
COLMAP_360_TEST_EVERY ?= 8
COLMAP_360_STEPS ?= 30000
COLMAP_360_TRAIN_OUT ?= outputs/360_$(COLMAP_360_SCENE)_train
COLMAP_360_TRAIN_SPZ ?= $(COLMAP_360_TRAIN_OUT)/trained_360_$(COLMAP_360_SCENE).spz
COLMAP_360_TRAIN_MODEL_NPZ ?= $(COLMAP_360_TRAIN_OUT)/trained_model_params.npz
```

   - Add:

```text
make codex-360-points-train-spz
make codex-360-points-train-spz-refine
```

   - These should pass `--data "$(COLMAP_360_DATA)"` and `--data-factor "$(COLMAP_360_FACTOR)"`.
   - The full-run target should resolve training values from the gsplat table above.
   - Defaults must mean full gsplat-style training: all train frames, all COLMAP sparse points, and 30000 steps.

4. Official full training command.

```sh
make codex-360-points-train-spz \
  COLMAP_360_SCENE=garden
```

5. Match gsplat's scene factors for benchmark-style runs.
   - Use factor `4` for `garden`, `bicycle`, and `stump`.
   - Use factor `2` for `bonsai`, `counter`, `kitchen`, and `room`.
   - Keep `render_traj_path=ellipse` in the planned output/eval path if trajectory rendering is added.

## Validation Plan

1. Dataset parser validation:
   - Confirm scene folder exists.
   - Confirm `images`, selected factor folder, and `sparse/0` exist.
   - Print image count, train count, eval count, point count, first image path, first `K`, and first `viewmat`.

2. Render alignment validation:
   - Render initial sparse-point Gaussians for a few cameras.
   - Save target/render/compare PNGs.
   - Check that visible Gaussian count is nonzero and projections land in the image.

3. Parameter parity check:
   - Dump a resolved training config next to outputs.
   - Confirm 360 full-run values match `gsplat` defaults: all train frames, all COLMAP sparse points, 30000 steps, SH degree 3, SH interval 1000, SFM initialization, opacity 0.1, L1+SSIM 0.2, and DefaultStrategy refine thresholds.

4. SPZ export:
   - Confirm `.spz` exists and is non-empty.
   - Confirm exported Gaussian count matches the final model count after optional refine/densify/prune.

5. Full run:
   - Run enough steps to produce visibly improving comparisons.
   - Compare train/eval losses from the saved summary JSON.

## Risks / Notes

- `pycolmap` is required by the gsplat parser. The conda env must have it installed or we need a fallback COLMAP binary reader.
- Camera convention is the highest-risk part. The loader should explicitly verify that COLMAP `w2c` becomes the same row-major world-to-camera `viewmat` expected by `projection_ewa_3dgs_fused_forward`.
- The current Makefile variable names are scanner-specific. Reusing them is okay for minimal change, but adding `COLMAP_360_*` variables will make commands clearer and reduce accidental deviation from gsplat defaults.
- Mip-NeRF 360 scenes can have many sparse points and high-resolution images. Official full runs must use the gsplat benchmark factors, all train frames, all COLMAP sparse points, and 30000 steps.
- The current MLX trainer uses fixed point-scale options. For 360 parity, SFM initialization should use KNN-derived scales like `gsplat` before we trust quality comparisons.

## Definition of Done

- A 360 scene can be loaded without converting it to scanner `frame_*.json` format.
- The Makefile has a clear target for 360 training.
- Full-run defaults match `gsplat`'s `simple_trainer.py default` settings.
- The official command runs through parsing, rendering/training, and SPZ export.
- Output artifacts include compare images, summary JSON, model NPZ, and SPZ.

## Execution Status

- Added `scripts/test/colmap_360_dataset.py`.
  - Uses the installed `pycolmap.Reconstruction` API.
  - Loads COLMAP `sparse/0`, sorted image names, selected factor image folder, normalized world space, sparse points, RGB point colors, camera `K`, and row-major world-to-camera `viewmat`.
- Added `scripts/test/train_360_points_multiview_3dgs_mlx.py`.
  - Dedicated 360/COLMAP trainer entrypoint.
  - Uses gsplat-style defaults: SFM point initialization, KNN scale initialization, opacity `0.1`, SH degree `3`, SH interval `1000`, `L1 + (1 - SSIM)` with lambda `0.2`, 30000-step default, and DefaultStrategy-style refine enabled by default.
  - Uses gsplat's active SH schedule formula: `min(step // sh_degree_interval, sh_degree)`.
  - Configures MLX cache limit to `32 GiB` by default and flushes stage/step logs.
  - Exports compare PNGs, summary JSON, model NPZ, and SPZ.
- Added Makefile targets:
  - `make codex-360-points-train-spz`
  - `make codex-360-points-train-spz-refine`
- Development-only reduced validation command verified directly through `conda run`:

```sh
conda run -n gsplat_core python scripts/test/train_360_points_multiview_3dgs_mlx.py \
  --data submodules/gsplat/examples/datasets/data/360_v2/garden \
  --data-factor 4 \
  --out-dir outputs/360_garden_refine_smoke \
  --out-spz outputs/360_garden_refine_smoke/trained_360_garden.spz \
  --out-model-npz outputs/360_garden_refine_smoke/trained_model_params.npz \
  --width 128 \
  --height 128 \
  --max-frames 2 \
  --eval-max-frames 1 \
  --max-points 1024 \
  --steps 2 \
  --log-interval 1 \
  --refine-enabled
```

Reduced validation result:

```text
initial_mean_loss=0.28105298
final_mean_loss=0.24811158
last_viewspace_grad_norm=0.00050152
spz=outputs/360_garden_refine_smoke/trained_360_garden.spz
```

Note: `pycolmap` is available, but this environment's `pycolmap` package does not expose `SceneManager`; the new loader uses `pycolmap.Reconstruction` instead. `scipy` is installed and `scipy.spatial.cKDTree` has been verified for practical KNN scale initialization.

Additional scipy-backed reduced run:

```text
max_points=5000
initial_mean_loss=0.25997254
final_mean_loss=0.21672288
last_viewspace_grad_norm=0.00071158
spz=outputs/360_garden_scipy_smoke/trained_360_garden.spz
```

MLX cache-limit validation:

```text
mlx cache limit configured current=34359738368 bytes (32.00 GiB)
entering training loop steps=2 batch_size=1
step=0001 ...
step=0002 ...
```

SH schedule validation:

```text
step=1000 ... sh=1
```
