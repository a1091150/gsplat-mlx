# Fixed Point 3DGS Training Tasks

## Goal
Create a synthetic, fully controlled fixed-point 3DGS training test before
continuing scanner-scene training. The test dataset is generated locally and is
intended to make camera convention, color learning, SH export, refinement, and
training stability easier to inspect than real scanner data.

## Dataset
- Generate a regular dodecahedron mesh centered at the origin.
- Use 12 planar faces.
- Assign each face a distinct color on a deterministic gradient:
  - face 0 starts red.
  - the final face ends purple.
  - intermediate faces interpolate through visually distinct hues.
- Draw white outlines on dodecahedron face boundaries so edge alignment is easy
  to inspect.
- Render 48 camera views around the object.
- Each generated frame must include:
  - RGB image.
  - camera intrinsics `K`.
  - row-major world-to-camera `viewmat`.
  - frame index.
  - camera position / look-at metadata.
- Store the generated dataset under `outputs/fixed_points_dataset` by default.

## Camera Plan
- Use 48 deterministic cameras on a sphere around the dodecahedron.
- Cameras look at the origin.
- Keep focal length, image size, near/far assumptions, and camera convention
  explicit in metadata.
- Use the same camera tensor convention as the current gsplat_core Python
  training path:
  - `viewmats` shape `(B, C, 4, 4)`.
  - `Ks` shape `(B, C, 3, 3)`.
  - pinhole camera model.

## Training Plan
- Add a fixed-point training script under `scripts/test`.
- Do not read scanner `points.ply`.
- Initialize trainable Gaussians from deterministic random positions inside the
  dodecahedron bounding box, not from face samples.
- Gaussian count must be a power of two.
  - Makefile default: `4096`.
  - Allowed examples: `256`, `512`, `1024`, `2048`, `4096`.
- Train for `8000` steps by default through the Makefile target.
- Use MLX `nn.Module`, Adam optimizer, and `mx.value_and_grad(...)`.
- Use L1 image loss.
- Use SH color training with progressive degree scheduling.
  - Default starts at degree 0.
  - Increase one SH degree every 1000 steps.
  - Default target degree is 3.
- Use `viewspace_points` dummy trainable argument for 2D gradient/refine
  compatibility, matching the project training design.
- Start without refine/densify in the first fixed-point smoke unless explicitly
  enabled later.

## Output Plan
- Output directory default: `outputs/fixed_points_train`.
- Save generated dataset preview images.
- Save training compare grids every `200` steps.
- Each grid contains 16 tiles.
  - Use selected camera views from the 48-view set.
  - Each tile should show a compact comparison suitable for human inspection.
  - Preferred first version: target/render pairs for 8 views, or 16 final
    render views if the pair layout becomes too small.
- Always save:
  - `training_summary.json`.
  - final SPZ using the selected project convention:
    - scanner position mode.
    - direct scale mode.
    - position-axis rotation mode.
    - `xyzw` quaternion storage.
    - active SH degree color export.
  - final model parameter `.npz` so SPZ variants can be regenerated without
    retraining.

## Proposed Make Targets
- `make codex-fixed-points-dataset`
  - Generate the synthetic dodecahedron dataset only.
- `make codex-fixed-points-train`
  - Generate or reuse the dataset, train for the configured step count, and
    save preview grids.
- Default variables:
  - `FIXED_POINTS_DATASET_OUT ?= outputs/fixed_points_dataset`
  - `FIXED_POINTS_TRAIN_OUT ?= outputs/fixed_points_train`
  - `FIXED_POINTS_WIDTH ?= 512`
  - `FIXED_POINTS_HEIGHT ?= 512`
  - `FIXED_POINTS_CAMERAS ?= 48`
  - `FIXED_POINTS_GAUSSIANS ?= 4096`
  - `FIXED_POINTS_STEPS ?= 8000`
  - `FIXED_POINTS_GRID_INTERVAL ?= 200`
  - `FIXED_POINTS_GRID_TILES ?= 16`
  - `FIXED_POINTS_SH_DEGREE_START ?= 0`
  - `FIXED_POINTS_SH_DEGREE_TARGET ?= 3`
  - `FIXED_POINTS_SH_DEGREE_SCHEDULE_INTERVAL ?= 1000`

## Task Breakdown

### Task FP.1 - Synthetic Dataset Generator
- [x] Create a deterministic dodecahedron mesh generator.
- [x] Assign 12 face colors from red to purple.
- [x] Generate 48 camera poses and intrinsics.
- [x] Rasterize the colored dodecahedron into RGB PNG targets.
- [x] Draw white face-boundary lines in the generated target images.
- [x] Save camera metadata and dataset summary.

### Task FP.2 - Fixed Gaussian Initialization
- [x] Sample a power-of-two number of Gaussian centers uniformly inside the
  dodecahedron bounding box.
- [x] Initialize quats, log-scales, opacity logits, and random SH color
  parameters.
- [x] Validate Gaussian count is a power of two.

### Task FP.3 - Training Script
- [x] Train against the 48 generated views for the configured step count.
- [x] Use the existing gsplat_core forward path.
- [x] Use `viewspace_points` dummy trainable input.
- [x] Use L1 image loss.
- [x] Use progressive SH degree scheduling with one degree increase every 1000
  steps by default.
- [x] Log loss and selected training metadata in `training_summary.json`.

### Task FP.4 - 16-Grid Preview Output
- [x] Save a preview grid every 200 steps.
- [x] Use 16 deterministic view slots from the 48 cameras.
- [x] Include target/render pairs for human inspection.

### Task FP.5 - Export
- [x] Save final SPZ using the selected SPZ convention.
- [x] Save final model parameter `.npz`.
- [x] Save `training_summary.json` with dataset, camera, training, and export
  settings.

## Validation
- The generated target images should clearly show the dodecahedron with
  different colored faces.
- The first grid should show poor or partial reconstruction.
- Later grids should visibly approach the target colors and shape.
- Final loss should be finite and lower than initial loss.
- Final SPZ should load in the viewer with the same selected convention as the
  scanner points SPZ path.

## Assumptions
- This task is synthetic and does not depend on scanner datasets.
- The dodecahedron image renderer may be CPU/numpy based for dataset generation.
- Training should use the project MLX/gsplat_core path, not a separate renderer.
- The first implementation can prioritize human-inspection quality over perfect
  physically based rendering.
