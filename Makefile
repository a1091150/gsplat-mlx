CONDA_ENV ?= gsplat_core
XCODE_BUILD_DIR ?= build_xcode
XCODE_DERIVED_DATA_DIR ?= /private/tmp/gsplat_core_xcode_derived
CONFIG ?= Release
RANDOM_3DGS_PNG ?= outputs/random_3dgs.png
RANDOM_3DGS_SEED ?= 7
RANDOM_3DGS_N ?= 1024
RANDOM_3DGS_WIDTH ?= 512
RANDOM_3DGS_HEIGHT ?= 512

IMAGE_FITTING_IMG_PATH ?=
IMAGE_FITTING_DATASET_OUT ?= outputs/image_fitting_dataset
IMAGE_FITTING_TRAIN_OUT ?= outputs/image_fitting_train
IMAGE_FITTING_SPZ_OUT ?= outputs/image_fitting_train/trained_image_fitting.spz
IMAGE_FITTING_STEPS ?= 150
IMAGE_FITTING_N ?= 1024
IMAGE_FITTING_WIDTH ?= 256
IMAGE_FITTING_HEIGHT ?= 256
IMAGE_FITTING_SEED ?= 11
IMAGE_FITTING_CAMERA_Z ?= 8.0
IMAGE_FITTING_INIT_XY_EXTENT ?= 8.0
IMAGE_FITTING_INIT_Z_EXTENT ?= 0.25
IMAGE_FITTING_INIT_SCALE ?= 0.02
IMAGE_FITTING_LOG_INTERVAL ?= 50
IMAGE_FITTING_SAVE_INTERVAL ?= 1

SCANNER_DATASET ?= /Users/yangdunfu/Downloads/2026_05_04_16_51_29
SCANNER_POINTS_TRAIN2_OUT ?= outputs/scanner_points_multiview_train_spz2
SCANNER_POINTS_TRAIN2_SPZ ?= $(SCANNER_POINTS_TRAIN2_OUT)/trained_scanner_points_spz2.spz
SCANNER_POINTS_TRAIN2_MODEL_NPZ ?= $(SCANNER_POINTS_TRAIN2_OUT)/trained_model_params.npz
SCANNER_POINTS_TRAIN2_WIDTH ?= 512
SCANNER_POINTS_TRAIN2_HEIGHT ?= 512
SCANNER_POINTS_TRAIN2_MAX_POINTS ?= 0
SCANNER_POINTS_TRAIN2_FRAMES ?= 0
SCANNER_POINTS_TRAIN2_FRAME_STEP ?= 1
SCANNER_POINTS_TRAIN2_START_INDEX ?= 0
SCANNER_POINTS_TRAIN2_STEPS ?= 4000
SCANNER_POINTS_TRAIN2_BATCH_SIZE ?= 1
SCANNER_POINTS_TRAIN2_LOG_INTERVAL ?= 100
SCANNER_POINTS_TRAIN2_MLX_CACHE_LIMIT_GB ?= 32
SCANNER_POINTS_TRAIN2_SPZ_SCALE_MODE ?= direct
SCANNER_POINTS_TRAIN2_SPZ_ROTATION_MODE ?= position_axis
SCANNER_POINTS_TRAIN2_SPZ_QUAT_ORDER ?= xyzw
SCANNER_POINTS_TRAIN2_SPZ_COLOR_MODE ?= sh

COLMAP_360_ROOT ?= datasets/360_v2
COLMAP_360_SCENE ?= garden
COLMAP_360_DATA ?= $(COLMAP_360_ROOT)/$(COLMAP_360_SCENE)
COLMAP_360_FACTOR ?= $(if $(filter bonsai counter kitchen room,$(COLMAP_360_SCENE)),2,4)
COLMAP_360_TEST_EVERY ?= 8
COLMAP_360_WIDTH ?= 0
COLMAP_360_HEIGHT ?= 0
COLMAP_360_MAX_FRAMES ?= 0
COLMAP_360_FRAME_STEP ?= 1
COLMAP_360_START_INDEX ?= 0
COLMAP_360_EVAL_FRAMES ?= 0
COLMAP_360_EVAL_FRAME_STEP ?= $(COLMAP_360_FRAME_STEP)
COLMAP_360_EVAL_START_INDEX ?= 0
COLMAP_360_MAX_POINTS ?= 0
COLMAP_360_STEPS ?= 4000
COLMAP_360_BATCH_SIZE ?= 1
COLMAP_360_OUT ?= outputs/360_$(COLMAP_360_SCENE)_train
COLMAP_360_SPZ ?= $(COLMAP_360_OUT)/trained_360_$(COLMAP_360_SCENE).spz
COLMAP_360_MODEL_NPZ ?= $(COLMAP_360_OUT)/trained_model_params.npz
COLMAP_360_LOG_INTERVAL ?= 100
COLMAP_360_STEP_IMAGE_INTERVAL ?= 0
COLMAP_360_MLX_CACHE_LIMIT_GB ?= 32

SOFA_DATA ?= datasets/B075X65R3X
SOFA_WIDTH ?= 512
SOFA_HEIGHT ?= 512
SOFA_MAX_FRAMES ?= 0
SOFA_FRAME_STEP ?= 1
SOFA_START_INDEX ?= 0
SOFA_MAX_POINTS ?= 4096
SOFA_STEPS ?= $(COLMAP_360_STEPS)
SOFA_BATCH_SIZE ?= $(COLMAP_360_BATCH_SIZE)
SOFA_OUT ?= outputs/sofa_train
SOFA_SPZ ?= $(SOFA_OUT)/trained_sofa.spz
SOFA_MODEL_NPZ ?= $(SOFA_OUT)/trained_model_params.npz
SOFA_LOG_INTERVAL ?= $(COLMAP_360_LOG_INTERVAL)
SOFA_STEP_IMAGE_INTERVAL ?= 2
SOFA_MLX_CACHE_LIMIT_GB ?= $(COLMAP_360_MLX_CACHE_LIMIT_GB)

DODECAHEDRON_DATASET_OUT ?= outputs/dodecahedron_dataset
DODECAHEDRON_WIDTH ?= 512
DODECAHEDRON_HEIGHT ?= 512
DODECAHEDRON_CAMERAS ?= 48
DODECAHEDRON_CAMERA_RADIUS ?= 3.2
DODECAHEDRON_FOCAL_SCALE ?= 0.92
DODECAHEDRON_MAX_FRAMES ?= 0
DODECAHEDRON_FRAME_STEP ?= 1
DODECAHEDRON_START_INDEX ?= 0
DODECAHEDRON_MAX_POINTS ?= 0
DODECAHEDRON_STEPS ?= $(COLMAP_360_STEPS)
DODECAHEDRON_BATCH_SIZE ?= $(COLMAP_360_BATCH_SIZE)
DODECAHEDRON_OUT ?= outputs/dodecahedron_train
DODECAHEDRON_SPZ ?= $(DODECAHEDRON_OUT)/trained_dodecahedron.spz
DODECAHEDRON_MODEL_NPZ ?= $(DODECAHEDRON_OUT)/trained_model_params.npz
DODECAHEDRON_LOG_INTERVAL ?= $(COLMAP_360_LOG_INTERVAL)
DODECAHEDRON_STEP_IMAGE_INTERVAL ?= 0
DODECAHEDRON_MLX_CACHE_LIMIT_GB ?= $(COLMAP_360_MLX_CACHE_LIMIT_GB)
CONDA_BASE := $(shell conda info --base 2>/dev/null)

IMAGE_FITTING_IMAGE_FLAGS =
ifneq ($(strip $(IMAGE_FITTING_IMG_PATH)),)
IMAGE_FITTING_IMAGE_FLAGS += --img-path "$(IMAGE_FITTING_IMG_PATH)"
endif

.PHONY: help env-check xcode-build pip-install pip-develop codex-xcode-test codex-random-png codex-training-smoke codex-dense-training-smoke codex-image-fitting-train codex-scanner-points-train-spz2 codex-360-points-train-spz codex-360-points-train-spz-refine codex-sofa-train-spz codex-dodecahedron-train-spz codex-projection-guardrails clean

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
	@printf "  make codex-image-fitting-train  Train MLX 3DGS against one image or the synthetic image-fitting target.\n"
	@printf "  make codex-scanner-points-train-spz2  Train scanner points.ply with 360/gsplat-style settings and export SPZ.\n"
	@printf "  make codex-360-points-train-spz  Train a Mip-NeRF 360/COLMAP scene with gsplat default-style settings and export SPZ.\n"
	@printf "  make codex-360-points-train-spz-refine  Alias for the gsplat-default 360 refine/densify training target.\n"
	@printf "  make codex-sofa-train-spz  Train B075X65R3X with 360-style point init/refine settings and export SPZ.\n"
	@printf "  make codex-dodecahedron-train-spz  Train the generated dodecahedron dataset with 360-style point init/refine settings and export SPZ.\n"
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

codex-image-fitting-train:
	conda run -n $(CONDA_ENV) python scripts/test/train_image_fitting_3dgs_mlx.py \
		$(IMAGE_FITTING_IMAGE_FLAGS) \
		--dataset-out "$(IMAGE_FITTING_DATASET_OUT)" \
		--out-dir "$(IMAGE_FITTING_TRAIN_OUT)" \
		--spz-out "$(IMAGE_FITTING_SPZ_OUT)" \
		--steps $(IMAGE_FITTING_STEPS) \
		--num-gaussians $(IMAGE_FITTING_N) \
		--width $(IMAGE_FITTING_WIDTH) \
		--height $(IMAGE_FITTING_HEIGHT) \
		--seed $(IMAGE_FITTING_SEED) \
		--camera-z $(IMAGE_FITTING_CAMERA_Z) \
		--init-xy-extent $(IMAGE_FITTING_INIT_XY_EXTENT) \
		--init-z-extent $(IMAGE_FITTING_INIT_Z_EXTENT) \
		--init-scale $(IMAGE_FITTING_INIT_SCALE) \
		--log-interval $(IMAGE_FITTING_LOG_INTERVAL) \
		--save-interval $(IMAGE_FITTING_SAVE_INTERVAL)

codex-scanner-points-train-spz2:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanner_points_multiview_3dgs_mlx2.py --data "$(SCANNER_DATASET)" --out-dir "$(SCANNER_POINTS_TRAIN2_OUT)" --out-spz "$(SCANNER_POINTS_TRAIN2_SPZ)" --out-model-npz "$(SCANNER_POINTS_TRAIN2_MODEL_NPZ)" --width $(SCANNER_POINTS_TRAIN2_WIDTH) --height $(SCANNER_POINTS_TRAIN2_HEIGHT) --max-frames $(SCANNER_POINTS_TRAIN2_FRAMES) --frame-step $(SCANNER_POINTS_TRAIN2_FRAME_STEP) --start-index $(SCANNER_POINTS_TRAIN2_START_INDEX) --max-points $(SCANNER_POINTS_TRAIN2_MAX_POINTS) --steps $(SCANNER_POINTS_TRAIN2_STEPS) --batch-size $(SCANNER_POINTS_TRAIN2_BATCH_SIZE) --log-interval $(SCANNER_POINTS_TRAIN2_LOG_INTERVAL) --mlx-cache-limit-gb $(SCANNER_POINTS_TRAIN2_MLX_CACHE_LIMIT_GB) --spz-scale-mode $(SCANNER_POINTS_TRAIN2_SPZ_SCALE_MODE) --spz-rotation-mode $(SCANNER_POINTS_TRAIN2_SPZ_ROTATION_MODE) --spz-quat-order $(SCANNER_POINTS_TRAIN2_SPZ_QUAT_ORDER) --spz-color-mode $(SCANNER_POINTS_TRAIN2_SPZ_COLOR_MODE) --refine-enabled

codex-360-points-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_360_points_multiview_3dgs_mlx.py --data "$(COLMAP_360_DATA)" --out-dir "$(COLMAP_360_OUT)" --out-spz "$(COLMAP_360_SPZ)" --out-model-npz "$(COLMAP_360_MODEL_NPZ)" --data-factor $(COLMAP_360_FACTOR) --test-every $(COLMAP_360_TEST_EVERY) --width $(COLMAP_360_WIDTH) --height $(COLMAP_360_HEIGHT) --max-frames $(COLMAP_360_MAX_FRAMES) --frame-step $(COLMAP_360_FRAME_STEP) --start-index $(COLMAP_360_START_INDEX) --eval-max-frames $(COLMAP_360_EVAL_FRAMES) --eval-frame-step $(COLMAP_360_EVAL_FRAME_STEP) --eval-start-index $(COLMAP_360_EVAL_START_INDEX) --max-points $(COLMAP_360_MAX_POINTS) --steps $(COLMAP_360_STEPS) --batch-size $(COLMAP_360_BATCH_SIZE) --log-interval $(COLMAP_360_LOG_INTERVAL) --step-image-interval $(COLMAP_360_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(COLMAP_360_MLX_CACHE_LIMIT_GB) --refine-enabled

codex-360-points-train-spz-refine: codex-360-points-train-spz

codex-sofa-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_sofa_points_multiview_3dgs_mlx.py --dataset b075x65r3x --data "$(SOFA_DATA)" --out-dir "$(SOFA_OUT)" --out-spz "$(SOFA_SPZ)" --out-model-npz "$(SOFA_MODEL_NPZ)" --width $(SOFA_WIDTH) --height $(SOFA_HEIGHT) --max-frames $(SOFA_MAX_FRAMES) --frame-step $(SOFA_FRAME_STEP) --start-index $(SOFA_START_INDEX) --max-points $(SOFA_MAX_POINTS) --steps $(SOFA_STEPS) --batch-size $(SOFA_BATCH_SIZE) --log-interval $(SOFA_LOG_INTERVAL) --step-image-interval $(SOFA_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(SOFA_MLX_CACHE_LIMIT_GB) --refine-enabled

codex-dodecahedron-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_dodecahedron_points_multiview_3dgs_mlx.py --dataset dodecahedron --dataset-out "$(DODECAHEDRON_DATASET_OUT)" --out-dir "$(DODECAHEDRON_OUT)" --out-spz "$(DODECAHEDRON_SPZ)" --out-model-npz "$(DODECAHEDRON_MODEL_NPZ)" --width $(DODECAHEDRON_WIDTH) --height $(DODECAHEDRON_HEIGHT) --num-cameras $(DODECAHEDRON_CAMERAS) --camera-radius $(DODECAHEDRON_CAMERA_RADIUS) --focal-scale $(DODECAHEDRON_FOCAL_SCALE) --max-frames $(DODECAHEDRON_MAX_FRAMES) --frame-step $(DODECAHEDRON_FRAME_STEP) --start-index $(DODECAHEDRON_START_INDEX) --max-points $(DODECAHEDRON_MAX_POINTS) --steps $(DODECAHEDRON_STEPS) --batch-size $(DODECAHEDRON_BATCH_SIZE) --log-interval $(DODECAHEDRON_LOG_INTERVAL) --step-image-interval $(DODECAHEDRON_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(DODECAHEDRON_MLX_CACHE_LIMIT_GB) --refine-enabled

codex-projection-guardrails:
	conda run -n $(CONDA_ENV) python scripts/test/training_viewspace_proxy_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/training_projection_viewspace_proxy_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/training_dense_3dgs_loop_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/projection_vjp_guardrails.py

clean:
	rm -rf $(XCODE_BUILD_DIR) build build-* dist *.egg-info python_package/*.egg-info
