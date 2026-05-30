# Scanner Points MLX vs MLX2 Training Notes

## Background

This note compares:

- `scripts/test/train_scanner_points_multiview_3dgs_mlx.py`
- `scripts/test/train_scanner_points_multiview_3dgs_mlx2.py`

The practical question is why `mlx2.py` can produce visibly better results than
`mlx.py` for scanner points training.

One important Makefile detail:

- `make codex-scanner-points-train-spz-refine` runs `train_scanner_points_multiview_3dgs_mlx.py`.
- `make codex-scanner-points-train-spz2` runs `train_scanner_points_multiview_3dgs_mlx2.py`.

So better `mlx2.py` results are not from the refine target directly unless the
`spz2` target, or an equivalent direct command, is used.

## High-Level Difference

`mlx.py` is the older, more general scanner trainer. It grew incrementally from
fixed point/Gaussian training into a flexible trainer with RGB mode, SH mode,
progressive SH, optional refine, configurable loss, configurable scale
initialization, configurable LR schedules, and SPZ export.

`mlx2.py` is a narrower gsplat-style scanner trainer. It reuses many building
blocks from `mlx.py`, including renderer, sampler, refine runtime, diagnostics,
and SPZ export, but its training defaults are much closer to gsplat's default
3DGS recipe.

In short:

- `mlx.py`: flexible scanner training engine.
- `mlx2.py`: scanner entrypoint tuned toward gsplat default behavior.

## Most Likely Reasons MLX2 Looks Better

### 1. KNN Scale Initialization

`mlx2.py` initializes Gaussian log scales from KNN distances:

```text
average distance to nearest 3 neighbors * init_scale
```

This gives every Gaussian a local scale based on point-cloud density.

`mlx.py` currently uses fixed scale or scene-fraction scale. With the Makefile
defaults, it uses:

```text
--point-scale-mode scene_fraction
--point-scale-fraction 0.005
```

That can be too coarse in dense areas and too small in sparse areas. Since
scale controls visibility, raster footprint, gradients, and refine decisions,
this is probably the biggest quality difference.

Likelihood: very high.

### 2. Initial Opacity

`mlx2.py` defaults to:

```text
opacity = 0.1
```

This matches gsplat default-style initialization.

`mlx.py` defaults to:

```text
opacity = 0.65
```

The `codex-scanner-points-train-spz-refine` target does not currently override
opacity, so it uses the higher script default.

High initial opacity can make early training over-occlude. Many Gaussians may
block each other, alpha can saturate early, and the gradient/refine signal can
become less useful. This can also affect prune/reset behavior.

Likelihood: very high.

### 3. Learning Rate Recipe

`mlx2.py` uses gsplat-style learning rates:

```text
means:        1.6e-4 * scene_scale, decay to 1%
scales:       5.0e-3
opacity:      5.0e-2
quats:        1.0e-3
SH DC:        2.5e-3
SH rest:      2.5e-3 / 20 = 1.25e-4
```

The scanner Makefile defaults for `mlx.py` are currently closer to:

```text
means:        0.002 -> 0.0002
colors/DC:    0.02  -> 0.005
SH rest:      0.001 -> 0.0001
opacity:      0.005 -> 0.001
scales:       0.001 -> 0.0005
quats:        0.001 -> 0.0001
```

Compared with `mlx2.py`, `mlx.py` tends to move color/SH more aggressively but
adjust opacity and scale more slowly. For 3DGS, geometry, scale, and opacity
need to settle well; otherwise color can overfit around bad Gaussian shapes.

Likelihood: high.

### 4. SSIM Loss Formula

`mlx2.py` uses:

```text
(1 - lambda) * L1 + lambda * (1 - SSIM)
```

with `lambda = 0.2`.

`mlx.py` in `l1_dssim` mode uses:

```text
(1 - lambda) * L1 + lambda * ((1 - SSIM) / 2)
```

So with `lambda = 0.2`, the structural loss contribution is effectively half
as strong as in `mlx2.py`.

This can matter visually because SSIM helps preserve structure and edges.

Likelihood: medium to high.

### 5. Scene Scale Used By Refine

`mlx2.py` computes scene scale from camera centers:

```text
scene_scale = camera extent
resolved_scene_scale = scene_scale * 1.1 * global_scale
```

This is closer to gsplat's scene-scale convention.

`mlx.py` defaults refine scene scale to point extent:

```text
--refine-scene-scale-mode points_extent
```

If scanner point clouds contain outliers, uneven density, or a point extent that
does not match camera coverage, the clone/split/prune thresholds can be biased.

Likelihood: medium.

## Refine Is Not After-Training

Both trainers call the refine runtime inside the training loop, after optimizer
updates:

```text
strategy.update_state(...)
strategy.after_optimizer_step(...)
```

So clone, split, prune, and opacity reset happen during training. They are not a
separate post-training process.

## Practical Ranking Of Suspected Causes

Most likely quality drivers:

1. KNN scale initialization.
2. Initial opacity `0.1` instead of `0.65`.
3. gsplat-style learning rates.
4. stronger SSIM term in the loss.
5. camera-based scene scale for refine thresholds.

The first two are the fastest to test and are likely to explain a large part of
the visual gap.

## Suggested Experiments

### Experiment A: Make MLX Opacity Match MLX2

Run `mlx.py` with:

```text
--opacity 0.1
```

Expected result:

- less early opacity saturation
- cleaner gradients
- more useful prune/split behavior

### Experiment B: Add KNN Scale Initialization To MLX

Add or expose a scale mode equivalent to `mlx2.py`:

```text
--point-scale-mode knn
--init-scale 1.0
```

Expected result:

- better initial raster footprint
- more stable visibility
- more reliable densification signal

### Experiment C: Match MLX2 Learning Rates

Use gsplat-style LR values:

```text
means:        1.6e-4 * scene_scale, final = 1%
scales:       5.0e-3
opacity:      5.0e-2
quats:        1.0e-3
SH DC:        2.5e-3
SH rest:      1.25e-4
```

Expected result:

- faster opacity/scale correction
- less color overfitting to bad geometry

### Experiment D: Match MLX2 Loss

Change `mlx.py` loss from DSSIM to full `1 - SSIM`:

```text
(1 - lambda) * L1 + lambda * (1 - SSIM)
```

Expected result:

- stronger structure-preserving optimization
- possibly sharper visual alignment

### Experiment E: Match Refine Scene Scale

Use camera-center extent instead of point extent for refine scene scale.

Expected result:

- clone/split/prune thresholds better match camera coverage
- fewer incorrect topology decisions from point-cloud extent bias

## Working Hypothesis

`mlx2.py` looks better mainly because it starts from a better-conditioned 3DGS
state and uses a gsplat-like optimization recipe. The renderer/refine core is
largely shared with `mlx.py`; the visible improvement is therefore more likely
from initialization, opacity, LR, loss, and scene-scale choices than from a
fundamentally different rasterization path.

