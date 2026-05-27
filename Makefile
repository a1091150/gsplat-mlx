CONDA_ENV ?= fastgs_core
XCODE_BUILD_DIR ?= build_xcode
XCODE_DERIVED_DATA_DIR ?= /private/tmp/gsplat_core_xcode_derived
CONFIG ?= Release
RANDOM_3DGS_PNG ?= outputs/random_3dgs.png
RANDOM_3DGS_SEED ?= 7
RANDOM_3DGS_N ?= 1024
RANDOM_3DGS_WIDTH ?= 512
RANDOM_3DGS_HEIGHT ?= 512
CONDA_BASE := $(shell conda info --base 2>/dev/null)

.PHONY: help env-check xcode-build pip-install pip-develop codex-xcode-test codex-random-png codex-training-smoke clean

help:
	@printf "Targets:\n"
	@printf "  make env-check    Print conda python/cmake paths and mlx/nanobind versions.\n"
	@printf "  make xcode-build  Configure and build _gsplat_core with Xcode using Python mlx package CMake config.\n"
	@printf "  make pip-install  pip install . --no-build-isolation in the conda env.\n"
	@printf "  make pip-develop  pip install -e . --no-build-isolation in the conda env.\n"
	@printf "  make codex-xcode-test  Build and run the C++ smoke test through the Xcode project.\n"
	@printf "  make codex-random-png  Render a random 3DGS PNG smoke image for manual inspection.\n"
	@printf "  make codex-training-smoke  Run MLX value_and_grad viewspace proxy smoke test.\n"
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

clean:
	rm -rf $(XCODE_BUILD_DIR) build build-* dist *.egg-info python_package/*.egg-info
