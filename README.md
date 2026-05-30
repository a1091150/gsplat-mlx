# gsplat-mlx
## gsplat_core

`gsplat_core` brings core 3D Gaussian Splatting operators from CUDA `gsplat` to
Apple Silicon through MLX primitives and custom Metal kernels, with
fixture-based parity checks against the original CUDA implementation.

The project focuses on low-level CUDA-to-Metal operator parity for dense 3DGS
rendering and training workflows. It provides MLX primitives, Metal kernels,
C++ implementations, and Python bindings for the main forward rendering path.

## Features

- MLX + Metal implementation of core 3DGS operators.
- Python extension module powered by nanobind.
- Dense 3DGS forward rendering pipeline:
  - quaternion / scale to covariance and precision
  - spherical harmonics evaluation
  - fused 3DGS projection
  - tile intersection and offset generation
  - rasterization to pixels
- Selected backward / VJP support for training-oriented workflows.
- CUDA reference export scripts for parity validation.
- `.npz` fixture-based comparison between CUDA `gsplat` and MLX / Metal
  outputs.
- Xcode, CMake, and Makefile build entry points.
- MLX training scripts for image fitting, scanner data, COLMAP /
  Mip-NeRF 360-style scenes, and SPZ export.

## Project Status

This project is an active migration of the `gsplat` CUDA / PyTorch low-level
3DGS path to Apple MLX and Metal.

Implemented and validated:

- Dense low-level 3DGS forward chain.
- Projection, intersection, spherical harmonics, covariance / precision, and
  rasterization operators.
- CUDA reference fixture comparison for key operators.
- CUDA-style analytic spherical harmonics direction VJP in Metal.
- CUDA-style analytic quaternion / scale covariance and precision VJPs in
  Metal, including `triu=true`, `triu=false`, optional cotangents, and edge-case
  fixtures.
- Basic training smoke tests using MLX autograd.

Current guardrails and known gaps:

- Packed sparse `gsplat` paths.
- Segmented sort paths.
- Projection backward currently supports the pinhole path; ortho and fisheye
  backward VJPs remain guarded until Metal parity fixtures are added.
- Intersect tile currently uses the dense radius AABB path; CUDA AccuTile /
  SNUGBOX ellipse intersection with `conics` and `opacities` is not ported.
- Additional CUDA `.npz` fixtures are still needed for spherical harmonics
  backward degree 0 through degree 4 with masks and `compute_v_dirs=true`.
- 2DGS operators.
- LiDAR operators.
- Rolling shutter / world-ray rasterization paths.
- Full high-level `gsplat` Python API compatibility.

Packed and segmented support are not current correctness blockers for the dense
MLX training path. They are mainly memory, performance, and sparse-workflow
features, and should be promoted only if a future high-level API or training
workflow needs them.

## Requirements

- macOS with Apple Silicon.
- Python 3.11.
- Conda environment, default: `gsplat_core`.
- MLX.
- nanobind.
- CMake 3.27+.
- Xcode command line tools.

## Environment Setup

- Install Xcode from the App Store.
- In Xcode, open Settings -> Components -> Other Components and make sure the
  Metal Toolchain is installed.
- Install CMake:

```bash
brew install cmake
```

- Install Conda. The `Makefile` assumes a Conda environment by default.

Create the default Conda environment:

```bash
conda create -n gsplat_core python=3.11
conda activate gsplat_core
```

Install the required Python packages:

```bash
pip install mlx==0.30.0 nanobind==2.4.0 cmake opencv-python plyfile pillow scipy pycolmap tyro
```

Install `gsplat_core`:

```bash
pip install . --no-build-isolation
```

Install `spz` for SPZ export workflows:

```bash
git submodule update --init --recursive
cd submodules/spz
git checkout ef094fd1a96ca6ff414d72d7904ee4f4f6d97be9
pip install .
```

Notes:

- MLX releases may require matching nanobind versions. mlx 0.30.0 uses nanobind 2.4.0.
- Some `spz` versions have known SPZ export issues.

## Build

Check the local environment:

```bash
make env-check
```

Build the Python extension with Xcode:

```bash
make xcode-build
```

Install the package:

```bash
make pip-install
```

For editable development install:

```bash
make pip-develop
```

## Tests and Validation

Run the C++ / Metal smoke test:

```bash
make codex-xcode-test
```

Run a dense 3DGS training smoke test:

```bash
make codex-dense-training-smoke
```

Render a random 3DGS PNG for manual inspection:

```bash
make codex-random-png
```

Compare exported CUDA reference fixtures against the MLX / Metal
implementation:

```bash
conda run -n gsplat_core python scripts/test/compare_exported_npz.py
```

See [scripts/export_ref/README.md](scripts/export_ref/README.md) for CUDA-side
fixture export instructions.

## CUDA Reference Fixtures

The `scripts/export_ref` directory contains scripts intended to run on a CUDA
machine with PyTorch and `gsplat` installed. These scripts export deterministic
`.npz` files containing both inputs and CUDA reference outputs.

Example:

```bash
python scripts/export_ref/export_forward_3dgs_chain.py \
  --out refs/forward_3dgs_chain.npz
```

The Mac / MLX side then loads the same inputs, runs `gsplat_core`, and compares
against the reference outputs.

## Datasets

Training scripts expect datasets under `datasets/` by default, except scanner
captures where `SCANNER_DATASET` can point at an exported folder anywhere on
disk.

### Mip-NeRF 360 / COLMAP

The 360 training target uses `datasets/360_v2/<scene>`. Use gsplat's
[`download_dataset.py`](https://github.com/nerfstudio-project/gsplat/blob/main/examples/datasets/download_dataset.py)
as the reference downloader; its default dataset downloads `360_v2.zip`.

After downloading and extracting the archive, place the dataset at
`datasets/360_v2`.

The default Makefile scene is `garden`:

```bash
make codex-360-points-train-spz
```

Override the scene or root as needed:

```bash
make codex-360-points-train-spz \
  COLMAP_360_ROOT=datasets/360_v2 \
  COLMAP_360_SCENE=bonsai
```

### B075X65R3X(Brown Sofa)

The `B075X65R3X` dataset is available from
[hbb1/torch-splatting](https://github.com/hbb1/torch-splatting). Download
`B075X65R3X.zip`, unzip it, and place the extracted folder at:

```text
datasets/B075X65R3X
```

Then run:

```bash
make codex-sofa-train-spz
```

### 3D Scanner App Captures

Scanner training currently targets datasets exported from the
[3D Scanner iPhone App](https://3dscannerapp.com/).

1. Use an iPhone with LiDAR support.
2. Install the 3D Scanner App.
3. Capture in landscape orientation, with the front camera on the left side.
4. Choose the Point Cloud option for scanning.
5. Export using the "All Data" option and transfer it to your Mac with AirDrop.

Run scanner training by pointing `SCANNER_DATASET` at the exported folder:

```bash
make codex-scanner-points-train-spz2 \
  SCANNER_DATASET=/path/to/3d-scanner-app-export
```

## Training Experiments

The repository includes experimental MLX training scripts for several
workflows:

```bash
make codex-image-fitting-train
make codex-360-points-train-spz
make codex-sofa-train-spz
make codex-dodecahedron-train-spz
```

These scripts are development and validation tools for the MLX / Metal 3DGS
pipeline rather than a stable high-level training API.

## Repository Layout

```text
gsplat_core/
  C++ operator implementations
gsplat_core/include/
  Public C++ headers
gsplat_core/metal/
  Metal kernels
gsplat_core/binding/
  nanobind Python bindings
python_package/gsplat_core/
  Python package wrapper
scripts/test/
  Local MLX tests, parity checks, rendering, and training experiments
scripts/export_ref/
  CUDA-side reference export scripts
refs/
  Exported reference fixtures
note/
  Migration notes and source mapping
```

## Motivation

`gsplat` provides high-performance CUDA kernels for 3D Gaussian Splatting. This
project explores how much of that low-level rendering and training path can be
brought to Apple Silicon using MLX primitives and custom Metal kernels, while
keeping behavior close to the CUDA reference implementation.
