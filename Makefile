CONDA_ENV ?= fastgs_core
XCODE_BUILD_DIR ?= build_xcode
XCODE_DERIVED_DATA_DIR ?= /private/tmp/gsplat_core_xcode_derived
CONFIG ?= Release
RANDOM_3DGS_PNG ?= outputs/random_3dgs.png
RANDOM_3DGS_SEED ?= 7
RANDOM_3DGS_N ?= 1024
RANDOM_3DGS_WIDTH ?= 512
RANDOM_3DGS_HEIGHT ?= 512
TINY_TRAIN_OUT ?= outputs/tiny_3dgs_train
TINY_TRAIN_STEPS ?= 40
TINY_TRAIN_N ?= 1024
TINY_TRAIN_WIDTH ?= 512
TINY_TRAIN_HEIGHT ?= 512
TINY_MULTIVIEW_OUT ?= outputs/tiny_3dgs_multiview_train
TINY_MULTIVIEW_STEPS ?= 40
TINY_MULTIVIEW_N ?= 1024
TINY_MULTIVIEW_WIDTH ?= 512
TINY_MULTIVIEW_HEIGHT ?= 512
TINY_MULTIVIEW_VIEWS ?= 3
SCANNER_DATASET ?= /Users/yangdunfu/Downloads/2026_05_04_16_51_29
SCANNER_SMOKE_OUT ?= outputs/scanner_dataset_random_render
SCANNER_SMOKE_WIDTH ?= 512
SCANNER_SMOKE_HEIGHT ?= 512
SCANNER_SMOKE_N ?= 1024
SCANNER_SMOKE_FRAMES ?= 3
SCANNER_SMOKE_FRAME_STEP ?= 1
SCANNER_TRAIN_OUT ?= outputs/scanner_random_3dgs_train
SCANNER_TRAIN_WIDTH ?= 512
SCANNER_TRAIN_HEIGHT ?= 512
SCANNER_TRAIN_N ?= 1024
SCANNER_TRAIN_FRAMES ?= 3
SCANNER_TRAIN_FRAME_STEP ?= 1
SCANNER_TRAIN_STEPS ?= 200
SCANNER_ALIGN_OUT ?= outputs/scanner_points_alignment
SCANNER_ALIGN_WIDTH ?= 512
SCANNER_ALIGN_HEIGHT ?= 512
SCANNER_ALIGN_FRAMES ?= 3
SCANNER_ALIGN_FRAME_STEP ?= 1
SCANNER_ALIGN_MAX_POINTS ?= 50000
SCANNER_ALIGN_POINT_SCALE ?= 0.01
SCANNER_SPZ_OUT ?= outputs/scanner_points.spz
SCANNER_SPZ_MAX_POINTS ?= 50000
SCANNER_SPZ_POINT_SCALE ?= 0.01
SCANNER_SPZ_OPACITY ?= 0.65
SCANNER_SPZ_COLOR_MODE ?= rgb
SCANNER_POINTS_TRAIN_OUT ?= outputs/scanner_points_multiview_train
SCANNER_POINTS_TRAIN_SPZ ?= outputs/scanner_points_multiview_train/trained_scanner_points.spz
SCANNER_POINTS_TRAIN_WIDTH ?= 512
SCANNER_POINTS_TRAIN_HEIGHT ?= 512
SCANNER_POINTS_TRAIN_MAX_POINTS ?= 50000
SCANNER_POINTS_TRAIN_FRAMES ?= 3
SCANNER_POINTS_TRAIN_FRAME_STEP ?= 1
SCANNER_POINTS_TRAIN_STEPS ?= 200
SCANNER_POINTS_TRAIN_POINT_SCALE ?= 0.01
SCANNER_POINTS_TRAIN_SH_DEGREE ?= 0
SCANNER_POINTS_TRAIN_MAX_SH_DEGREE ?= 1
CONDA_BASE := $(shell conda info --base 2>/dev/null)

.PHONY: help env-check xcode-build pip-install pip-develop codex-xcode-test codex-random-png codex-training-smoke codex-dense-training-smoke codex-tiny-train codex-tiny-multiview-train codex-scanner-dataset-smoke codex-scanner-random-train codex-scanner-points-align codex-scanner-points-spz codex-scanner-points-train-spz codex-projection-guardrails clean

help:
	@printf "Targets:\n"
	@printf "  make env-check    Print conda python/cmake paths and mlx/nanobind versions.\n"
	@printf "  make xcode-build  Configure and build _gsplat_core with Xcode using Python mlx package CMake config.\n"
	@printf "  make pip-install  pip install . --no-build-isolation in the conda env.\n"
	@printf "  make pip-develop  pip install -e . --no-build-isolation in the conda env.\n"
	@printf "  make codex-xcode-test  Build and run the C++ smoke test through the Xcode project.\n"
	@printf "  make codex-random-png  Render a random 3DGS PNG smoke image for manual inspection.\n"
	@printf "  make codex-training-smoke  Run MLX value_and_grad viewspace proxy smoke test.\n"
	@printf "  make codex-dense-training-smoke  Run dense projection+rasterize training loop smoke test.\n"
	@printf "  make codex-tiny-train  Run a tiny MLX nn.Module+Adam 3DGS training example.\n"
	@printf "  make codex-tiny-multiview-train  Run a tiny multi-view MLX 3DGS training example.\n"
	@printf "  make codex-scanner-dataset-smoke  Render random Gaussians with scanner dataset cameras.\n"
	@printf "  make codex-scanner-random-train  Train random Gaussians against scanner dataset frames.\n"
	@printf "  make codex-scanner-points-align  Render points.ply with scanner dataset cameras.\n"
	@printf "  make codex-scanner-points-spz  Export scanner points.ply to SPZ.\n"
	@printf "  make codex-scanner-points-train-spz  Train points.ply Gaussians and export SPZ.\n"
	@printf "  make codex-projection-guardrails  Check projection VJP full-GPU support boundaries.\n"
	@printf "  make clean        Remove Xcode/build folders and Python packaging artifacts.\n"

env-check:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && \
	echo "CONDA_ENV=$(CONDA_ENV)" && \
	echo "python=$$(which python)" && \
	echo "cmake=$$(which cmake)" && \
	python -c "import importlib.metadata as md, sys; print(\"python_version=\"+sys.version.split()[0]); print(\"mlx=\"+md.version(\"mlx\")); print(\"nanobind=\"+md.version(\"nanobind\"))" && \
	MLX_CMAKE_DIR=$$(python -c "import importlib.util, pathlib; spec = importlib.util.find_spec(\"mlx\"); print(pathlib.Path(spec.submodule_search_locations[0]) / \"share\" / \"cmake\" / \"MLX\")") && \
	echo "mlx_cmake_dir=$$MLX_CMAKE_DIR"'

xcode-build:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && \
	cmake -S . -B $(XCODE_BUILD_DIR) -G Xcode \
		-DCMAKE_CXX_COMPILER="$$(xcrun --find clang++)" \
		-DPython_EXECUTABLE="$$(which python)" \
		-DGSPLAT_BUILD_PYTHON=ON \
		-DGSPLAT_BUILD_TEST=ON \
		-DGSPLAT_BUILD_METAL=ON \
		-DGSPLAT_XCODE_DEV_MODE=ON \
		-DGSPLAT_USE_MLX_MODULE_CMAKE_DIR=ON && \
	cmake --build $(XCODE_BUILD_DIR) --config $(CONFIG) --target _gsplat_core'

pip-install:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && pip install . --no-build-isolation'

pip-develop:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && pip install -e . --no-build-isolation'

codex-xcode-test:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && \
	cmake -S . -B $(XCODE_BUILD_DIR) -G Xcode \
		-DCMAKE_CXX_COMPILER="$$(xcrun --find clang++)" \
		-DPython_EXECUTABLE="$$(which python)" \
		-DGSPLAT_BUILD_PYTHON=ON \
		-DGSPLAT_BUILD_TEST=ON \
		-DGSPLAT_BUILD_METAL=ON \
		-DGSPLAT_XCODE_DEV_MODE=ON \
		-DGSPLAT_USE_MLX_MODULE_CMAKE_DIR=ON && \
	xcodebuild -project $(XCODE_BUILD_DIR)/gsplat_core.xcodeproj \
		-scheme gsplat_core_dummy_test \
		-configuration $(CONFIG) \
		-derivedDataPath $(XCODE_DERIVED_DATA_DIR) \
		build && \
	./$(XCODE_BUILD_DIR)/$(CONFIG)/gsplat_core_dummy_test'

codex-random-png:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && \
	python scripts/test/render_random_3dgs_png.py \
		--out "$(RANDOM_3DGS_PNG)" \
		--seed $(RANDOM_3DGS_SEED) \
		--num-gaussians $(RANDOM_3DGS_N) \
		--width $(RANDOM_3DGS_WIDTH) \
		--height $(RANDOM_3DGS_HEIGHT)'

codex-training-smoke:
	conda run -n $(CONDA_ENV) python scripts/test/training_viewspace_proxy_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/training_projection_viewspace_proxy_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/training_dense_3dgs_loop_smoke.py

codex-dense-training-smoke:
	conda run -n $(CONDA_ENV) python scripts/test/training_dense_3dgs_loop_smoke.py

codex-tiny-train:
	conda run -n $(CONDA_ENV) python scripts/test/train_tiny_3dgs_mlx.py \
		--out-dir "$(TINY_TRAIN_OUT)" \
		--steps $(TINY_TRAIN_STEPS) \
		--num-gaussians $(TINY_TRAIN_N) \
		--width $(TINY_TRAIN_WIDTH) \
		--height $(TINY_TRAIN_HEIGHT)

codex-tiny-multiview-train:
	conda run -n $(CONDA_ENV) python scripts/test/train_tiny_multiview_3dgs_mlx.py \
		--out-dir "$(TINY_MULTIVIEW_OUT)" \
		--steps $(TINY_MULTIVIEW_STEPS) \
		--num-gaussians $(TINY_MULTIVIEW_N) \
		--num-views $(TINY_MULTIVIEW_VIEWS) \
		--width $(TINY_MULTIVIEW_WIDTH) \
		--height $(TINY_MULTIVIEW_HEIGHT)

codex-scanner-dataset-smoke:
	conda run -n $(CONDA_ENV) python scripts/test/scanner_dataset_random_render_smoke.py \
		--data "$(SCANNER_DATASET)" \
		--out-dir "$(SCANNER_SMOKE_OUT)" \
		--width $(SCANNER_SMOKE_WIDTH) \
		--height $(SCANNER_SMOKE_HEIGHT) \
		--num-gaussians $(SCANNER_SMOKE_N) \
		--max-frames $(SCANNER_SMOKE_FRAMES) \
		--frame-step $(SCANNER_SMOKE_FRAME_STEP)

codex-scanner-random-train:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanner_random_3dgs_mlx.py \
		--data "$(SCANNER_DATASET)" \
		--out-dir "$(SCANNER_TRAIN_OUT)" \
		--width $(SCANNER_TRAIN_WIDTH) \
		--height $(SCANNER_TRAIN_HEIGHT) \
		--num-gaussians $(SCANNER_TRAIN_N) \
		--max-frames $(SCANNER_TRAIN_FRAMES) \
		--frame-step $(SCANNER_TRAIN_FRAME_STEP) \
		--steps $(SCANNER_TRAIN_STEPS)

codex-scanner-points-align:
	conda run -n $(CONDA_ENV) python scripts/test/scanner_points_alignment_render.py \
		--data "$(SCANNER_DATASET)" \
		--out-dir "$(SCANNER_ALIGN_OUT)" \
		--width $(SCANNER_ALIGN_WIDTH) \
		--height $(SCANNER_ALIGN_HEIGHT) \
		--max-frames $(SCANNER_ALIGN_FRAMES) \
		--frame-step $(SCANNER_ALIGN_FRAME_STEP) \
		--max-points $(SCANNER_ALIGN_MAX_POINTS) \
		--point-scale $(SCANNER_ALIGN_POINT_SCALE)

codex-scanner-points-spz:
	conda run -n $(CONDA_ENV) python scripts/test/export_scanner_points_spz.py \
		--data "$(SCANNER_DATASET)" \
		--out "$(SCANNER_SPZ_OUT)" \
		--max-points $(SCANNER_SPZ_MAX_POINTS) \
		--point-scale $(SCANNER_SPZ_POINT_SCALE) \
		--opacity $(SCANNER_SPZ_OPACITY) \
		--color-mode $(SCANNER_SPZ_COLOR_MODE)

codex-scanner-points-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanner_points_multiview_3dgs_mlx.py \
		--data "$(SCANNER_DATASET)" \
		--out-dir "$(SCANNER_POINTS_TRAIN_OUT)" \
		--out-spz "$(SCANNER_POINTS_TRAIN_SPZ)" \
		--width $(SCANNER_POINTS_TRAIN_WIDTH) \
		--height $(SCANNER_POINTS_TRAIN_HEIGHT) \
		--max-frames $(SCANNER_POINTS_TRAIN_FRAMES) \
		--frame-step $(SCANNER_POINTS_TRAIN_FRAME_STEP) \
		--max-points $(SCANNER_POINTS_TRAIN_MAX_POINTS) \
		--steps $(SCANNER_POINTS_TRAIN_STEPS) \
		--point-scale $(SCANNER_POINTS_TRAIN_POINT_SCALE) \
		--sh-degree $(SCANNER_POINTS_TRAIN_SH_DEGREE) \
		--max-sh-degree $(SCANNER_POINTS_TRAIN_MAX_SH_DEGREE)

codex-projection-guardrails:
	conda run -n $(CONDA_ENV) python scripts/test/training_viewspace_proxy_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/training_projection_viewspace_proxy_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/training_dense_3dgs_loop_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/projection_vjp_guardrails.py

clean:
	rm -rf $(XCODE_BUILD_DIR) build build-* dist *.egg-info python_package/*.egg-info
