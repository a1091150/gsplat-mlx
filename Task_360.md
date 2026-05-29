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

Training behavior and defaults should follow `submodules/gsplat/examples/simple_trainer.py`, not the current scanner smoke defaults.

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

The MLX target can expose its own Makefile variables, but the resolved training values should match these `gsplat` defaults unless we explicitly mark a run as smoke/debug:

| Setting | gsplat default / benchmark value | MLX task target |
| --- | --- | --- |
| Dataset type | `colmap` | `--data-type colmap` |
| Data factor | `4` default; benchmark uses `2` for `bonsai`, `counter`, `kitchen`, `room`; `4` for `garden`, `bicycle`, `stump` | `COLMAP_360_FACTOR` should follow the same scene rule |
| Train/val split | `test_every=8` | `COLMAP_360_TEST_EVERY=8` |
| World normalization | `normalize_world_space=True` | enable by default |
| Batch size | `1` | `SCANNER_POINTS_TRAIN_BATCH_SIZE=1` or renamed 360 equivalent |
| Max steps | `30000` | use `30000` for full 360 runs |
| Eval/save steps | `7000`, `30000` | save/eval previews at equivalent milestones |
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

`gsplat` also has an `mcmc` benchmark mode. The first implementation should target `default` because the current MLX trainer already has a gsplat-style default refine path. Add `mcmc` only after the COLMAP/default pipeline is correct.

## Proposed Implementation

1. Add a reusable COLMAP dataset loader.
   - Create a small module under `scripts/test/`, for example `colmap_360_dataset.py`.
   - Reference `submodules/gsplat/examples/datasets/colmap.py` for behavior.
   - Load `sparse/0` first and fall back to `sparse`.
   - Read `cameras.bin`, `images.bin`, and `points3D.bin` through `pycolmap.SceneManager`.
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
COLMAP_360_FACTOR ?= 4
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

   - These should pass `--data-type colmap`, `--data "$(COLMAP_360_DATA)"`, and `--data-factor "$(COLMAP_360_FACTOR)"`.
   - The full-run target should resolve training values from the gsplat table above.
   - Keep any reduced values under explicit smoke variables or a `codex-360-points-smoke` target.

4. Start with a small smoke run.
   - Use one scene, probably `garden` or `stump`.
   - Use factor `4` or `8` first.
   - Use a small frame count and point count before full training.

```sh
make codex-360-points-train-spz \
  COLMAP_360_SCENE=garden \
  COLMAP_360_FACTOR=4 \
  SCANNER_POINTS_TRAIN_FRAMES=8 \
  SCANNER_POINTS_TRAIN_MAX_POINTS=20000 \
  SCANNER_POINTS_TRAIN_STEPS=20
```

   Smoke runs may reduce frames, points, and steps, but must not redefine the intended full-training defaults.

5. Then run a real training pass.

```sh
make codex-360-points-train-spz-refine \
  COLMAP_360_SCENE=garden \
  COLMAP_360_FACTOR=4 \
  SCANNER_POINTS_TRAIN_FRAMES=999 \
  SCANNER_POINTS_TRAIN_STEPS=30000 \
  SCANNER_POINTS_EVAL_FRAMES=8
```

6. Match gsplat's scene factors for benchmark-style runs.
   - Use factor `4` for `garden`, `bicycle`, and `stump`.
   - Use factor `2` for `bonsai`, `counter`, `kitchen`, and `room`.
   - Keep `render_traj_path=ellipse` in the planned output/eval path if trajectory rendering is added.

## Validation Plan

1. Dataset parser smoke:
   - Confirm scene folder exists.
   - Confirm `images`, selected factor folder, and `sparse/0` exist.
   - Print image count, train count, eval count, point count, first image path, first `K`, and first `viewmat`.

2. Render alignment smoke:
   - Render initial sparse-point Gaussians for a few cameras.
   - Save target/render/compare PNGs.
   - Check that visible Gaussian count is nonzero and projections land in the image.

3. Training smoke:
   - Run 10-20 steps.
   - Confirm loss is finite.
   - Confirm model NPZ and preview PNGs are written.

4. Parameter parity check:
   - Dump a resolved training config next to outputs.
   - Confirm 360 full-run values match `gsplat` defaults: 30000 steps, SH degree 3, SH interval 1000, SFM initialization, opacity 0.1, L1+DSSIM 0.2, and DefaultStrategy refine thresholds.

5. SPZ export:
   - Confirm `.spz` exists and is non-empty.
   - Confirm exported Gaussian count matches the final model count after optional refine/densify/prune.

6. Full run:
   - Run enough steps to produce visibly improving comparisons.
   - Compare train/eval losses from the saved summary JSON.

## Risks / Notes

- `pycolmap` is required by the gsplat parser. The conda env must have it installed or we need a fallback COLMAP binary reader.
- Camera convention is the highest-risk part. The loader should explicitly verify that COLMAP `w2c` becomes the same row-major world-to-camera `viewmat` expected by `projection_ewa_3dgs_fused_forward`.
- The current Makefile variable names are scanner-specific. Reusing them is okay for minimal change, but adding `COLMAP_360_*` variables will make commands clearer and reduce accidental deviation from gsplat defaults.
- Mip-NeRF 360 scenes can have many sparse points and high-resolution images. Smoke runs may use `COLMAP_360_FACTOR=4` or `8`, smaller point counts, and fewer frames, but full runs should use the gsplat benchmark factors and 30000 steps.
- The current MLX trainer uses fixed point-scale options. For 360 parity, SFM initialization should use KNN-derived scales like `gsplat` before we trust quality comparisons.

## Definition of Done

- A 360 scene can be loaded without converting it to scanner `frame_*.json` format.
- The Makefile has a clear target for 360 training.
- Full-run defaults match `gsplat`'s `simple_trainer.py default` settings unless a command is explicitly marked smoke/debug.
- A smoke command runs through parsing, rendering/training, and SPZ export.
- Output artifacts include compare images, summary JSON, model NPZ, and SPZ.
