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

SCANAPP_DEPTH_DATA ?= /Users/yangdunfu/Documents/iOSProject/ScanProject/20260618_154636
SCANAPP_DEPTH_OUT ?= outputs/scanapp_depth_train
SCANAPP_DEPTH_SPZ ?= $(SCANAPP_DEPTH_OUT)/trained_scanapp_depth.spz
SCANAPP_DEPTH_MODEL_NPZ ?= $(SCANAPP_DEPTH_OUT)/trained_model_params.npz
SCANAPP_DEPTH_WIDTH ?= 512
SCANAPP_DEPTH_HEIGHT ?= 512
SCANAPP_DEPTH_TARGET_POINTS ?= 262144
SCANAPP_DEPTH_MAX_FRAMES ?= 0
SCANAPP_DEPTH_FRAME_STEP ?= 1
SCANAPP_DEPTH_START_INDEX ?= 0
SCANAPP_DEPTH_EVAL_FRAMES ?= 0
SCANAPP_DEPTH_EVAL_FRAME_STEP ?= $(SCANAPP_DEPTH_FRAME_STEP)
SCANAPP_DEPTH_EVAL_START_INDEX ?= 0
SCANAPP_DEPTH_STEPS ?= $(COLMAP_360_STEPS)
SCANAPP_DEPTH_BATCH_SIZE ?= $(COLMAP_360_BATCH_SIZE)
SCANAPP_DEPTH_FRAME_SAMPLING ?= pingpong
SCANAPP_DEPTH_LOG_INTERVAL ?= $(COLMAP_360_LOG_INTERVAL)
SCANAPP_DEPTH_STEP_IMAGE_INTERVAL ?= 10
SCANAPP_DEPTH_MLX_CACHE_LIMIT_GB ?= $(COLMAP_360_MLX_CACHE_LIMIT_GB)
SCANAPP_DEPTH_GLOBAL_SCALE ?= 0.2
SCANAPP_DEPTH_SPZ_SCALE_MODE ?= direct
SCANAPP_DEPTH_SPZ_ROTATION_MODE ?= position_axis
SCANAPP_DEPTH_SPZ_QUAT_ORDER ?= xyzw
SCANAPP_DEPTH_SPZ_COLOR_MODE ?= sh
SCANAPP_DEPTH_MASKED_OUT ?= outputs/scanapp_depth_masked_train
SCANAPP_DEPTH_MASKED_SPZ ?= $(SCANAPP_DEPTH_MASKED_OUT)/trained_scanapp_depth_masked.spz
SCANAPP_DEPTH_MASKED_MODEL_NPZ ?= $(SCANAPP_DEPTH_MASKED_OUT)/trained_model_params.npz
SCANAPP_DEPTH_MASK_MIN ?= 0.05
SCANAPP_DEPTH_MASK_MAX ?= 5.0
SCANAPP_DEPTH_MASK_MIN_CONFIDENCE ?= 1
SCANAPP_DEPTH_MOBILE_PRIOR_OUT ?= outputs/scanapp_depth_mobile_prior_train
SCANAPP_DEPTH_MOBILE_PRIOR_SPZ ?= $(SCANAPP_DEPTH_MOBILE_PRIOR_OUT)/trained_scanapp_depth_mobile_prior.spz
SCANAPP_DEPTH_MOBILE_PRIOR_MODEL_NPZ ?= $(SCANAPP_DEPTH_MOBILE_PRIOR_OUT)/trained_model_params.npz
SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_FILTER ?= --keyframe-filter-enabled
SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_MIN_TRANSLATION ?= 0.05
SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_WINDOW ?= 8
SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_SHARPNESS_STRIDE ?= 8
SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_MIN_FRAMES ?= 16
SCANAPP_DEPTH_MIN_MOTION_QUALITY ?= 0
SCANAPP_DEPTH_SHARED_INTRINSICS ?= none
SCANAPP_DEPTH_COLMAP_POSE_DATA ?=
SCANAPP_DEPTH_COLMAP_POSE_INTRINSICS ?= --colmap-pose-intrinsics
SCANAPP_DEPTH_COLMAP_POSE_DEPTH_POINTS ?= --colmap-pose-depth-points
SCANAPP_DEPTH_COLMAP_POSE_FLAGS = $(if $(strip $(SCANAPP_DEPTH_COLMAP_POSE_DATA)),--colmap-pose-data "$(SCANAPP_DEPTH_COLMAP_POSE_DATA)" $(SCANAPP_DEPTH_COLMAP_POSE_INTRINSICS) $(SCANAPP_DEPTH_COLMAP_POSE_DEPTH_POINTS),)
SCANAPP_DEPTH_PER_FRAME_POINT_SAMPLES ?= 0
SCANAPP_DEPTH_MOBILE_PRIOR_INIT_MODE ?= disc
SCANAPP_DEPTH_MOBILE_PRIOR_NORMAL_KNN ?= 16
SCANAPP_DEPTH_MOBILE_PRIOR_TANGENT_KNN ?= 3
SCANAPP_DEPTH_MOBILE_PRIOR_NORMAL_SCALE_RATIO ?= 0.2
SCANAPP_DEPTH_MOBILE_PRIOR_TANGENT_SCALE_MULTIPLIER ?= 1.0
SCANAPP_DEPTH_MOBILE_PRIOR_REFINE_RESET_EVERY ?= 1782
SCANAPP_DEPTH_POSE_REFINE_OUT ?= outputs/scanapp_depth_pose_refine_train
SCANAPP_DEPTH_POSE_REFINE_SPZ ?= $(SCANAPP_DEPTH_POSE_REFINE_OUT)/trained_scanapp_depth_pose_refine.spz
SCANAPP_DEPTH_POSE_REFINE_MODEL_NPZ ?= $(SCANAPP_DEPTH_POSE_REFINE_OUT)/trained_model_params.npz
SCANAPP_DEPTH_POSE_REFINE_STOP_STEP ?= 3000
SCANAPP_DEPTH_POSE_ROT_LR ?= 0.0001
SCANAPP_DEPTH_POSE_TRANS_LR ?= 0.001
SCANAPP_DEPTH_POSE_ROT_REG ?= 0.01
SCANAPP_DEPTH_POSE_TRANS_REG ?= 0.01
SCANAPP_DEPTH_CONSISTENCY_OUT ?= outputs/scanapp_depth_consistency_train
SCANAPP_DEPTH_CONSISTENCY_SPZ ?= $(SCANAPP_DEPTH_CONSISTENCY_OUT)/trained_scanapp_depth_consistency.spz
SCANAPP_DEPTH_CONSISTENCY_MODEL_NPZ ?= $(SCANAPP_DEPTH_CONSISTENCY_OUT)/trained_model_params.npz
SCANAPP_DEPTH_CONSISTENCY_FILTER ?= --consistency-filter-enabled
SCANAPP_DEPTH_CONSISTENCY_NEIGHBOR_WINDOW ?= 2
SCANAPP_DEPTH_CONSISTENCY_MIN_VIEWS ?= 1
SCANAPP_DEPTH_CONSISTENCY_ABS_DEPTH_TOL ?= 0.08
SCANAPP_DEPTH_CONSISTENCY_REL_DEPTH_TOL ?= 0.03
SCANAPP_DEPTH_CONSISTENCY_KEEP_UNOBSERVED ?= --consistency-keep-unobserved
SCANAPP_DEPTH_CHUNKED_OUT ?= outputs/scanapp_depth_chunked_consistency_train
SCANAPP_DEPTH_CHUNK_SIZE ?= 8
SCANAPP_DEPTH_CHUNK_STRIDE ?= 4
SCANAPP_DEPTH_CHUNK_MAX_CHUNKS ?= 0
SCANAPP_DEPTH_CHUNK_DRY_RUN ?=
SCANAPP_DEPTH_CHUNK_KEEP_GOING ?=
SCANAPP_DEPTH_GSPLAT_DEFAULT_OUT ?= outputs/scanapp_depth_gsplat_default_medianK_960x720
SCANAPP_DEPTH_GSPLAT_DEFAULT_SPZ ?= $(SCANAPP_DEPTH_GSPLAT_DEFAULT_OUT)/trained_scanapp_depth_gsplat_default_medianK.spz
SCANAPP_DEPTH_GSPLAT_DEFAULT_MODEL_NPZ ?= $(SCANAPP_DEPTH_GSPLAT_DEFAULT_OUT)/trained_model_params.npz
SCANAPP_DEPTH_GSPLAT_DEFAULT_WIDTH ?= 960
SCANAPP_DEPTH_GSPLAT_DEFAULT_HEIGHT ?= 720
SCANAPP_DEPTH_GSPLAT_DEFAULT_STEPS ?= 2000
SCANAPP_DEPTH_GSPLAT_DEFAULT_TARGET_POINTS ?= 524288
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_OUT ?= outputs/scanapp_depth_normalized_schedule_train
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_SPZ ?= $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_OUT)/trained_scanapp_depth_normalized_schedule.spz
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_MODEL_NPZ ?= $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_OUT)/trained_model_params.npz
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_REFERENCE_WIDTH ?= 1920
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_REFERENCE_HEIGHT ?= 1440
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_FACTORS ?= 4,2
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_STAGE_STEPS ?=
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_STEPS ?= 2000
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_TARGET_POINTS ?= 262144
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_BLUR_MODE ?= mean
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_BLUR_KERNELS ?= 7,3
SCANAPP_DEPTH_NORMALIZED_SCHEDULE_SCALES_LR ?= 5e-4
SCANAPP_VIDEO_DATA ?= /Users/yangdunfu/Downloads/20260624_125437
SCANAPP_VIDEO_PREP_OUT ?= outputs/scanapp_video_20260624_125437_compat
SCANAPP_VIDEO_FFMPEG ?= ffmpeg
SCANAPP_VIDEO_IMAGE_EXTENSION ?= jpg
SCANAPP_VIDEO_JPEG_QUALITY ?= 2
SCANAPP_VIDEO_MAX_FRAMES ?= 0
SCANAPP_VIDEO_FRAME_STEP ?= 1
SCANAPP_VIDEO_START_INDEX ?= 0
SCANAPP_VIDEO_COPY_DEPTH ?= --no-copy-depth
SCANAPP_VIDEO_PREP_OVERWRITE ?=
COLMAP_POSE_REFINE_DATA ?= /Users/yangdunfu/Downloads/LidarSeries_20260623_100146_824
COLMAP_POSE_REFINE_OUT ?= outputs/colmap_pose_refine_splatking_20260623
COLMAP_POSE_REFINE_MATCHER ?= exhaustive
COLMAP_POSE_REFINE_USE_GPU ?= --no-use-gpu
COLMAP_POSE_REFINE_INTRINSICS ?= --no-refine-intrinsics
COLMAP_POSE_REFINE_COPY_IMAGES ?=
COLMAP_POSE_REFINE_OVERWRITE ?=
SCANAPP_COLMAP_SEED_DATA ?= /Users/yangdunfu/Downloads/20260623_151938
SCANAPP_COLMAP_SEED_OUT ?= outputs/scanapp_colmap_seed_20260623_151938
SCANAPP_COLMAP_SEED_SHARED_INTRINSICS ?= median
SCANAPP_COLMAP_SEED_MAX_FRAMES ?= 0
SCANAPP_COLMAP_SEED_FRAME_STEP ?= 1
SCANAPP_COLMAP_SEED_START_INDEX ?= 0
SCANAPP_COLMAP_SEED_COPY_IMAGES ?=
SCANAPP_COLMAP_SEED_OVERWRITE ?=
SCANAPP_BOUNDED_POSE_SEED_MODEL ?= outputs/scanapp_colmap_seed_20260623_151938
SCANAPP_BOUNDED_POSE_TEACHER_MODEL ?= outputs/scanapp_colmap_refine_20260623_151938/refined_colmap_text_model
SCANAPP_BOUNDED_POSE_OUT ?= outputs/scanapp_colmap_bounded_pose_20260623_151938
SCANAPP_BOUNDED_POSE_MAX_ROTATION_DEG ?= 3
SCANAPP_BOUNDED_POSE_MAX_TRANSLATION_M ?= 0.03
SCANAPP_BOUNDED_POSE_SMOOTH_WINDOW ?= 9
SCANAPP_BOUNDED_POSE_POINTS_SOURCE ?= teacher
SCANAPP_BOUNDED_POSE_COPY_IMAGES ?=
SCANAPP_BOUNDED_POSE_OVERWRITE ?=

COLMAP_360_ROOT ?= datasets/360_v2
COLMAP_360_SCENE ?= garden
COLMAP_360_DATA ?= $(COLMAP_360_ROOT)/$(COLMAP_360_SCENE)
COLMAP_360_FACTOR ?= $(if $(filter bonsai counter kitchen room,$(COLMAP_360_SCENE)),2,4)
COLMAP_360_TEST_EVERY ?= 8
COLMAP_360_TRAIN_SPLIT ?= train
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
COLMAP_360_MLX_CACHE_LIMIT_GB ?= 24

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

.PHONY: help env-check xcode-build pip-install pip-develop codex-xcode-test codex-random-png codex-training-smoke codex-dense-training-smoke codex-image-fitting-train codex-scanner-points-train-spz2 codex-scanapp-video-prepare codex-scanapp-depth-train-spz codex-scanapp-depth-masked-train-spz codex-scanapp-depth-mobile-prior-train-spz codex-scanapp-depth-pose-refine-train-spz codex-scanapp-depth-consistency-train-spz codex-scanapp-depth-chunked-consistency-train-spz codex-scanapp-depth-gsplat-default-train-spz codex-scanapp-depth-normalized-schedule-train-spz codex-scanapp-export-colmap-seed codex-scanapp-bounded-pose-correction codex-colmap-pose-refine codex-360-points-train-spz codex-360-points-train-spz-refine codex-sofa-train-spz codex-dodecahedron-train-spz codex-projection-guardrails clean

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
	@printf "  make codex-scanapp-video-prepare  Convert ScanApp RGB-video + JSONL captures to the per-frame depth trainer layout.\n"
	@printf "  make codex-scanapp-depth-train-spz  Train ScanApp iPhone depth frames with 360/gsplat-style settings and export SPZ.\n"
	@printf "  make codex-scanapp-depth-masked-train-spz  Train ScanApp depth frames with RGB loss masked to a depth range and export SPZ.\n"
	@printf "  make codex-scanapp-depth-mobile-prior-train-spz  Train ScanApp masked depth with PocketGS-style mobile priors and export SPZ.\n"
	@printf "  make codex-scanapp-depth-pose-refine-train-spz  Train ScanApp masked depth with learnable ARKit pose residuals and export SPZ.\n"
	@printf "  make codex-scanapp-depth-consistency-train-spz  Train ScanApp masked depth after cross-view depth consistency filtering and export SPZ.\n"
	@printf "  make codex-scanapp-depth-chunked-consistency-train-spz  Train overlapping local ScanApp consistency chunks and summarize them.\n"
	@printf "  make codex-scanapp-depth-gsplat-default-train-spz  Train ScanApp depth with 960x720 median K and gsplat default refine settings.\n"
	@printf "  make codex-scanapp-depth-normalized-schedule-train-spz  Train ScanApp depth with normalized world space and image-scale schedule.\n"
	@printf "  make codex-scanapp-export-colmap-seed  Export ScanApp RGB/K/pose metadata as a COLMAP text seed model.\n"
	@printf "  make codex-scanapp-bounded-pose-correction  Apply small, smoothed COLMAP pose corrections to a ScanApp seed model.\n"
	@printf "  make codex-colmap-pose-refine  Refine a COLMAP text seed model with feature matching, triangulation, and bundle adjustment.\n"
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

codex-scanapp-video-prepare:
	conda run -n $(CONDA_ENV) python scripts/test/prepare_scanapp_video_dataset.py --data "$(SCANAPP_VIDEO_DATA)" --out-dir "$(SCANAPP_VIDEO_PREP_OUT)" --ffmpeg-bin "$(SCANAPP_VIDEO_FFMPEG)" --image-extension $(SCANAPP_VIDEO_IMAGE_EXTENSION) --jpeg-quality $(SCANAPP_VIDEO_JPEG_QUALITY) --max-frames $(SCANAPP_VIDEO_MAX_FRAMES) --frame-step $(SCANAPP_VIDEO_FRAME_STEP) --start-index $(SCANAPP_VIDEO_START_INDEX) $(SCANAPP_VIDEO_COPY_DEPTH) $(SCANAPP_VIDEO_PREP_OVERWRITE)

codex-scanapp-depth-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanapp_depth_multiview_3dgs_mlx.py --data "$(SCANAPP_DEPTH_DATA)" --out-dir "$(SCANAPP_DEPTH_OUT)" --out-spz "$(SCANAPP_DEPTH_SPZ)" --out-model-npz "$(SCANAPP_DEPTH_MODEL_NPZ)" --width $(SCANAPP_DEPTH_WIDTH) --height $(SCANAPP_DEPTH_HEIGHT) --target-points $(SCANAPP_DEPTH_TARGET_POINTS) --max-frames $(SCANAPP_DEPTH_MAX_FRAMES) --frame-step $(SCANAPP_DEPTH_FRAME_STEP) --start-index $(SCANAPP_DEPTH_START_INDEX) --eval-max-frames $(SCANAPP_DEPTH_EVAL_FRAMES) --eval-frame-step $(SCANAPP_DEPTH_EVAL_FRAME_STEP) --eval-start-index $(SCANAPP_DEPTH_EVAL_START_INDEX) --steps $(SCANAPP_DEPTH_STEPS) --batch-size $(SCANAPP_DEPTH_BATCH_SIZE) --frame-sampling $(SCANAPP_DEPTH_FRAME_SAMPLING) --log-interval $(SCANAPP_DEPTH_LOG_INTERVAL) --step-image-interval $(SCANAPP_DEPTH_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(SCANAPP_DEPTH_MLX_CACHE_LIMIT_GB) --global-scale $(SCANAPP_DEPTH_GLOBAL_SCALE) --spz-scale-mode $(SCANAPP_DEPTH_SPZ_SCALE_MODE) --spz-rotation-mode $(SCANAPP_DEPTH_SPZ_ROTATION_MODE) --spz-quat-order $(SCANAPP_DEPTH_SPZ_QUAT_ORDER) --spz-color-mode $(SCANAPP_DEPTH_SPZ_COLOR_MODE) --refine-enabled

codex-scanapp-depth-masked-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanapp_depth_masked_multiview_3dgs_mlx.py --data "$(SCANAPP_DEPTH_DATA)" --out-dir "$(SCANAPP_DEPTH_MASKED_OUT)" --out-spz "$(SCANAPP_DEPTH_MASKED_SPZ)" --out-model-npz "$(SCANAPP_DEPTH_MASKED_MODEL_NPZ)" --width $(SCANAPP_DEPTH_WIDTH) --height $(SCANAPP_DEPTH_HEIGHT) --target-points $(SCANAPP_DEPTH_TARGET_POINTS) --max-frames $(SCANAPP_DEPTH_MAX_FRAMES) --frame-step $(SCANAPP_DEPTH_FRAME_STEP) --start-index $(SCANAPP_DEPTH_START_INDEX) --eval-max-frames $(SCANAPP_DEPTH_EVAL_FRAMES) --eval-frame-step $(SCANAPP_DEPTH_EVAL_FRAME_STEP) --eval-start-index $(SCANAPP_DEPTH_EVAL_START_INDEX) --steps $(SCANAPP_DEPTH_STEPS) --batch-size $(SCANAPP_DEPTH_BATCH_SIZE) --log-interval $(SCANAPP_DEPTH_LOG_INTERVAL) --step-image-interval $(SCANAPP_DEPTH_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(SCANAPP_DEPTH_MLX_CACHE_LIMIT_GB) --global-scale $(SCANAPP_DEPTH_GLOBAL_SCALE) --mask-min-depth $(SCANAPP_DEPTH_MASK_MIN) --mask-max-depth $(SCANAPP_DEPTH_MASK_MAX) --mask-min-confidence $(SCANAPP_DEPTH_MASK_MIN_CONFIDENCE) --spz-scale-mode $(SCANAPP_DEPTH_SPZ_SCALE_MODE) --spz-rotation-mode $(SCANAPP_DEPTH_SPZ_ROTATION_MODE) --spz-quat-order $(SCANAPP_DEPTH_SPZ_QUAT_ORDER) --spz-color-mode $(SCANAPP_DEPTH_SPZ_COLOR_MODE) --refine-enabled

codex-scanapp-depth-mobile-prior-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanapp_depth_mobile_prior_multiview_3dgs_mlx.py --data "$(SCANAPP_DEPTH_DATA)" --out-dir "$(SCANAPP_DEPTH_MOBILE_PRIOR_OUT)" --out-spz "$(SCANAPP_DEPTH_MOBILE_PRIOR_SPZ)" --out-model-npz "$(SCANAPP_DEPTH_MOBILE_PRIOR_MODEL_NPZ)" --width $(SCANAPP_DEPTH_WIDTH) --height $(SCANAPP_DEPTH_HEIGHT) --target-points $(SCANAPP_DEPTH_TARGET_POINTS) --max-frames $(SCANAPP_DEPTH_MAX_FRAMES) --frame-step $(SCANAPP_DEPTH_FRAME_STEP) --start-index $(SCANAPP_DEPTH_START_INDEX) --eval-max-frames $(SCANAPP_DEPTH_EVAL_FRAMES) --eval-frame-step $(SCANAPP_DEPTH_EVAL_FRAME_STEP) --eval-start-index $(SCANAPP_DEPTH_EVAL_START_INDEX) --steps $(SCANAPP_DEPTH_STEPS) --batch-size $(SCANAPP_DEPTH_BATCH_SIZE) --log-interval $(SCANAPP_DEPTH_LOG_INTERVAL) --step-image-interval $(SCANAPP_DEPTH_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(SCANAPP_DEPTH_MLX_CACHE_LIMIT_GB) --global-scale $(SCANAPP_DEPTH_GLOBAL_SCALE) --mask-min-depth $(SCANAPP_DEPTH_MASK_MIN) --mask-max-depth $(SCANAPP_DEPTH_MASK_MAX) --mask-min-confidence $(SCANAPP_DEPTH_MASK_MIN_CONFIDENCE) $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_FILTER) --keyframe-min-translation $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_MIN_TRANSLATION) --keyframe-window $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_WINDOW) --keyframe-sharpness-stride $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_SHARPNESS_STRIDE) --keyframe-min-frames $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_MIN_FRAMES) --prior-init-mode $(SCANAPP_DEPTH_MOBILE_PRIOR_INIT_MODE) --prior-normal-knn $(SCANAPP_DEPTH_MOBILE_PRIOR_NORMAL_KNN) --prior-tangent-knn $(SCANAPP_DEPTH_MOBILE_PRIOR_TANGENT_KNN) --prior-normal-scale-ratio $(SCANAPP_DEPTH_MOBILE_PRIOR_NORMAL_SCALE_RATIO) --prior-tangent-scale-multiplier $(SCANAPP_DEPTH_MOBILE_PRIOR_TANGENT_SCALE_MULTIPLIER) --spz-scale-mode $(SCANAPP_DEPTH_SPZ_SCALE_MODE) --spz-rotation-mode $(SCANAPP_DEPTH_SPZ_ROTATION_MODE) --spz-quat-order $(SCANAPP_DEPTH_SPZ_QUAT_ORDER) --spz-color-mode $(SCANAPP_DEPTH_SPZ_COLOR_MODE) --refine-reset-every $(SCANAPP_DEPTH_MOBILE_PRIOR_REFINE_RESET_EVERY) --refine-enabled

codex-scanapp-depth-pose-refine-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanapp_depth_pose_refine_multiview_3dgs_mlx.py --data "$(SCANAPP_DEPTH_DATA)" --out-dir "$(SCANAPP_DEPTH_POSE_REFINE_OUT)" --out-spz "$(SCANAPP_DEPTH_POSE_REFINE_SPZ)" --out-model-npz "$(SCANAPP_DEPTH_POSE_REFINE_MODEL_NPZ)" --width $(SCANAPP_DEPTH_WIDTH) --height $(SCANAPP_DEPTH_HEIGHT) --target-points $(SCANAPP_DEPTH_TARGET_POINTS) --max-frames $(SCANAPP_DEPTH_MAX_FRAMES) --frame-step $(SCANAPP_DEPTH_FRAME_STEP) --start-index $(SCANAPP_DEPTH_START_INDEX) --eval-max-frames $(SCANAPP_DEPTH_EVAL_FRAMES) --eval-frame-step $(SCANAPP_DEPTH_EVAL_FRAME_STEP) --eval-start-index $(SCANAPP_DEPTH_EVAL_START_INDEX) --steps $(SCANAPP_DEPTH_STEPS) --batch-size $(SCANAPP_DEPTH_BATCH_SIZE) --log-interval $(SCANAPP_DEPTH_LOG_INTERVAL) --step-image-interval $(SCANAPP_DEPTH_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(SCANAPP_DEPTH_MLX_CACHE_LIMIT_GB) --global-scale $(SCANAPP_DEPTH_GLOBAL_SCALE) --mask-min-depth $(SCANAPP_DEPTH_MASK_MIN) --mask-max-depth $(SCANAPP_DEPTH_MASK_MAX) --mask-min-confidence $(SCANAPP_DEPTH_MASK_MIN_CONFIDENCE) $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_FILTER) --keyframe-min-translation $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_MIN_TRANSLATION) --keyframe-window $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_WINDOW) --keyframe-sharpness-stride $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_SHARPNESS_STRIDE) --keyframe-min-frames $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_MIN_FRAMES) --prior-init-mode $(SCANAPP_DEPTH_MOBILE_PRIOR_INIT_MODE) --prior-normal-knn $(SCANAPP_DEPTH_MOBILE_PRIOR_NORMAL_KNN) --prior-tangent-knn $(SCANAPP_DEPTH_MOBILE_PRIOR_TANGENT_KNN) --prior-normal-scale-ratio $(SCANAPP_DEPTH_MOBILE_PRIOR_NORMAL_SCALE_RATIO) --prior-tangent-scale-multiplier $(SCANAPP_DEPTH_MOBILE_PRIOR_TANGENT_SCALE_MULTIPLIER) --pose-refine-enabled --pose-refine-stop-step $(SCANAPP_DEPTH_POSE_REFINE_STOP_STEP) --pose-rot-lr $(SCANAPP_DEPTH_POSE_ROT_LR) --pose-trans-lr $(SCANAPP_DEPTH_POSE_TRANS_LR) --pose-rot-reg $(SCANAPP_DEPTH_POSE_ROT_REG) --pose-trans-reg $(SCANAPP_DEPTH_POSE_TRANS_REG) --spz-scale-mode $(SCANAPP_DEPTH_SPZ_SCALE_MODE) --spz-rotation-mode $(SCANAPP_DEPTH_SPZ_ROTATION_MODE) --spz-quat-order $(SCANAPP_DEPTH_SPZ_QUAT_ORDER) --spz-color-mode $(SCANAPP_DEPTH_SPZ_COLOR_MODE) --refine-reset-every $(SCANAPP_DEPTH_MOBILE_PRIOR_REFINE_RESET_EVERY) --refine-enabled

codex-scanapp-depth-consistency-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanapp_depth_consistency_multiview_3dgs_mlx.py --data "$(SCANAPP_DEPTH_DATA)" --out-dir "$(SCANAPP_DEPTH_CONSISTENCY_OUT)" --out-spz "$(SCANAPP_DEPTH_CONSISTENCY_SPZ)" --out-model-npz "$(SCANAPP_DEPTH_CONSISTENCY_MODEL_NPZ)" --width $(SCANAPP_DEPTH_WIDTH) --height $(SCANAPP_DEPTH_HEIGHT) --target-points $(SCANAPP_DEPTH_TARGET_POINTS) --max-frames $(SCANAPP_DEPTH_MAX_FRAMES) --frame-step $(SCANAPP_DEPTH_FRAME_STEP) --start-index $(SCANAPP_DEPTH_START_INDEX) --eval-max-frames $(SCANAPP_DEPTH_EVAL_FRAMES) --eval-frame-step $(SCANAPP_DEPTH_EVAL_FRAME_STEP) --eval-start-index $(SCANAPP_DEPTH_EVAL_START_INDEX) --steps $(SCANAPP_DEPTH_STEPS) --batch-size $(SCANAPP_DEPTH_BATCH_SIZE) --log-interval $(SCANAPP_DEPTH_LOG_INTERVAL) --step-image-interval $(SCANAPP_DEPTH_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(SCANAPP_DEPTH_MLX_CACHE_LIMIT_GB) --global-scale $(SCANAPP_DEPTH_GLOBAL_SCALE) --mask-min-depth $(SCANAPP_DEPTH_MASK_MIN) --mask-max-depth $(SCANAPP_DEPTH_MASK_MAX) --mask-min-confidence $(SCANAPP_DEPTH_MASK_MIN_CONFIDENCE) $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_FILTER) --keyframe-min-translation $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_MIN_TRANSLATION) --keyframe-window $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_WINDOW) --keyframe-sharpness-stride $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_SHARPNESS_STRIDE) --keyframe-min-frames $(SCANAPP_DEPTH_MOBILE_PRIOR_KEYFRAME_MIN_FRAMES) --min-motion-quality $(SCANAPP_DEPTH_MIN_MOTION_QUALITY) --shared-intrinsics $(SCANAPP_DEPTH_SHARED_INTRINSICS) $(SCANAPP_DEPTH_COLMAP_POSE_FLAGS) --per-frame-point-samples $(SCANAPP_DEPTH_PER_FRAME_POINT_SAMPLES) --prior-init-mode $(SCANAPP_DEPTH_MOBILE_PRIOR_INIT_MODE) --prior-normal-knn $(SCANAPP_DEPTH_MOBILE_PRIOR_NORMAL_KNN) --prior-tangent-knn $(SCANAPP_DEPTH_MOBILE_PRIOR_TANGENT_KNN) --prior-normal-scale-ratio $(SCANAPP_DEPTH_MOBILE_PRIOR_NORMAL_SCALE_RATIO) --prior-tangent-scale-multiplier $(SCANAPP_DEPTH_MOBILE_PRIOR_TANGENT_SCALE_MULTIPLIER) $(SCANAPP_DEPTH_CONSISTENCY_FILTER) --consistency-neighbor-window $(SCANAPP_DEPTH_CONSISTENCY_NEIGHBOR_WINDOW) --consistency-min-views $(SCANAPP_DEPTH_CONSISTENCY_MIN_VIEWS) --consistency-abs-depth-tol $(SCANAPP_DEPTH_CONSISTENCY_ABS_DEPTH_TOL) --consistency-rel-depth-tol $(SCANAPP_DEPTH_CONSISTENCY_REL_DEPTH_TOL) $(SCANAPP_DEPTH_CONSISTENCY_KEEP_UNOBSERVED) --spz-scale-mode $(SCANAPP_DEPTH_SPZ_SCALE_MODE) --spz-rotation-mode $(SCANAPP_DEPTH_SPZ_ROTATION_MODE) --spz-quat-order $(SCANAPP_DEPTH_SPZ_QUAT_ORDER) --spz-color-mode $(SCANAPP_DEPTH_SPZ_COLOR_MODE) --refine-reset-every $(SCANAPP_DEPTH_MOBILE_PRIOR_REFINE_RESET_EVERY) --refine-enabled

codex-scanapp-depth-chunked-consistency-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanapp_depth_chunked_consistency_3dgs_mlx.py --data "$(SCANAPP_DEPTH_DATA)" --out-dir "$(SCANAPP_DEPTH_CHUNKED_OUT)" --width $(SCANAPP_DEPTH_WIDTH) --height $(SCANAPP_DEPTH_HEIGHT) --target-points $(SCANAPP_DEPTH_TARGET_POINTS) --max-frames $(SCANAPP_DEPTH_MAX_FRAMES) --frame-step $(SCANAPP_DEPTH_FRAME_STEP) --start-index $(SCANAPP_DEPTH_START_INDEX) --chunk-size $(SCANAPP_DEPTH_CHUNK_SIZE) --chunk-stride $(SCANAPP_DEPTH_CHUNK_STRIDE) --max-chunks $(SCANAPP_DEPTH_CHUNK_MAX_CHUNKS) --steps $(SCANAPP_DEPTH_STEPS) --batch-size $(SCANAPP_DEPTH_BATCH_SIZE) --log-interval $(SCANAPP_DEPTH_LOG_INTERVAL) --step-image-interval $(SCANAPP_DEPTH_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(SCANAPP_DEPTH_MLX_CACHE_LIMIT_GB) --global-scale $(SCANAPP_DEPTH_GLOBAL_SCALE) --mask-min-depth $(SCANAPP_DEPTH_MASK_MIN) --mask-max-depth $(SCANAPP_DEPTH_MASK_MAX) --mask-min-confidence $(SCANAPP_DEPTH_MASK_MIN_CONFIDENCE) --prior-init-mode $(SCANAPP_DEPTH_MOBILE_PRIOR_INIT_MODE) --prior-normal-knn $(SCANAPP_DEPTH_MOBILE_PRIOR_NORMAL_KNN) --prior-tangent-knn $(SCANAPP_DEPTH_MOBILE_PRIOR_TANGENT_KNN) --prior-normal-scale-ratio $(SCANAPP_DEPTH_MOBILE_PRIOR_NORMAL_SCALE_RATIO) --prior-tangent-scale-multiplier $(SCANAPP_DEPTH_MOBILE_PRIOR_TANGENT_SCALE_MULTIPLIER) $(SCANAPP_DEPTH_CONSISTENCY_FILTER) --consistency-neighbor-window $(SCANAPP_DEPTH_CONSISTENCY_NEIGHBOR_WINDOW) --consistency-min-views $(SCANAPP_DEPTH_CONSISTENCY_MIN_VIEWS) --consistency-abs-depth-tol $(SCANAPP_DEPTH_CONSISTENCY_ABS_DEPTH_TOL) --consistency-rel-depth-tol $(SCANAPP_DEPTH_CONSISTENCY_REL_DEPTH_TOL) $(SCANAPP_DEPTH_CONSISTENCY_KEEP_UNOBSERVED) --spz-scale-mode $(SCANAPP_DEPTH_SPZ_SCALE_MODE) --spz-rotation-mode $(SCANAPP_DEPTH_SPZ_ROTATION_MODE) --spz-quat-order $(SCANAPP_DEPTH_SPZ_QUAT_ORDER) --spz-color-mode $(SCANAPP_DEPTH_SPZ_COLOR_MODE) --refine-reset-every $(SCANAPP_DEPTH_MOBILE_PRIOR_REFINE_RESET_EVERY) --refine-enabled $(SCANAPP_DEPTH_CHUNK_DRY_RUN) $(SCANAPP_DEPTH_CHUNK_KEEP_GOING)

codex-scanapp-depth-gsplat-default-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanapp_depth_gsplat_default_medianK_3dgs_mlx.py --data "$(SCANAPP_DEPTH_DATA)" --out-dir "$(SCANAPP_DEPTH_GSPLAT_DEFAULT_OUT)" --out-spz "$(SCANAPP_DEPTH_GSPLAT_DEFAULT_SPZ)" --out-model-npz "$(SCANAPP_DEPTH_GSPLAT_DEFAULT_MODEL_NPZ)" --width $(SCANAPP_DEPTH_GSPLAT_DEFAULT_WIDTH) --height $(SCANAPP_DEPTH_GSPLAT_DEFAULT_HEIGHT) --target-points $(SCANAPP_DEPTH_GSPLAT_DEFAULT_TARGET_POINTS) --max-frames $(SCANAPP_DEPTH_MAX_FRAMES) --frame-step $(SCANAPP_DEPTH_FRAME_STEP) --start-index $(SCANAPP_DEPTH_START_INDEX) --eval-max-frames $(SCANAPP_DEPTH_EVAL_FRAMES) --eval-frame-step $(SCANAPP_DEPTH_EVAL_FRAME_STEP) --eval-start-index $(SCANAPP_DEPTH_EVAL_START_INDEX) --steps $(SCANAPP_DEPTH_GSPLAT_DEFAULT_STEPS) --batch-size $(SCANAPP_DEPTH_BATCH_SIZE) --log-interval $(SCANAPP_DEPTH_LOG_INTERVAL) --step-image-interval $(SCANAPP_DEPTH_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(SCANAPP_DEPTH_MLX_CACHE_LIMIT_GB) --global-scale $(SCANAPP_DEPTH_GLOBAL_SCALE) --mask-min-depth $(SCANAPP_DEPTH_MASK_MIN) --mask-max-depth $(SCANAPP_DEPTH_MASK_MAX) --mask-min-confidence $(SCANAPP_DEPTH_MASK_MIN_CONFIDENCE) --spz-scale-mode $(SCANAPP_DEPTH_SPZ_SCALE_MODE) --spz-rotation-mode $(SCANAPP_DEPTH_SPZ_ROTATION_MODE) --spz-quat-order $(SCANAPP_DEPTH_SPZ_QUAT_ORDER) --spz-color-mode $(SCANAPP_DEPTH_SPZ_COLOR_MODE) --refine-enabled

codex-scanapp-depth-normalized-schedule-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_scanapp_depth_normalized_schedule_3dgs_mlx.py --data "$(SCANAPP_DEPTH_DATA)" --out-dir "$(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_OUT)" --out-spz "$(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_SPZ)" --out-model-npz "$(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_MODEL_NPZ)" --reference-width $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_REFERENCE_WIDTH) --reference-height $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_REFERENCE_HEIGHT) --image-scale-factors $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_FACTORS) $(if $(strip $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_STAGE_STEPS)),--stage-steps $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_STAGE_STEPS),) --target-blur-mode $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_BLUR_MODE) --target-blur-kernels $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_BLUR_KERNELS) --scales-lr $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_SCALES_LR) --target-points $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_TARGET_POINTS) --max-frames $(SCANAPP_DEPTH_MAX_FRAMES) --frame-step $(SCANAPP_DEPTH_FRAME_STEP) --start-index $(SCANAPP_DEPTH_START_INDEX) --eval-max-frames $(SCANAPP_DEPTH_EVAL_FRAMES) --eval-frame-step $(SCANAPP_DEPTH_EVAL_FRAME_STEP) --eval-start-index $(SCANAPP_DEPTH_EVAL_START_INDEX) --steps $(SCANAPP_DEPTH_NORMALIZED_SCHEDULE_STEPS) --batch-size $(SCANAPP_DEPTH_BATCH_SIZE) --log-interval $(SCANAPP_DEPTH_LOG_INTERVAL) --step-image-interval $(SCANAPP_DEPTH_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(SCANAPP_DEPTH_MLX_CACHE_LIMIT_GB) --global-scale $(SCANAPP_DEPTH_GLOBAL_SCALE) --mask-min-depth $(SCANAPP_DEPTH_MASK_MIN) --mask-max-depth $(SCANAPP_DEPTH_MASK_MAX) --mask-min-confidence $(SCANAPP_DEPTH_MASK_MIN_CONFIDENCE) --spz-scale-mode $(SCANAPP_DEPTH_SPZ_SCALE_MODE) --spz-rotation-mode $(SCANAPP_DEPTH_SPZ_ROTATION_MODE) --spz-quat-order $(SCANAPP_DEPTH_SPZ_QUAT_ORDER) --spz-color-mode $(SCANAPP_DEPTH_SPZ_COLOR_MODE) --refine-enabled

codex-scanapp-export-colmap-seed:
	conda run -n $(CONDA_ENV) python scripts/test/export_scanapp_colmap_seed_model.py --data "$(SCANAPP_COLMAP_SEED_DATA)" --out-dir "$(SCANAPP_COLMAP_SEED_OUT)" --shared-intrinsics $(SCANAPP_COLMAP_SEED_SHARED_INTRINSICS) --max-frames $(SCANAPP_COLMAP_SEED_MAX_FRAMES) --frame-step $(SCANAPP_COLMAP_SEED_FRAME_STEP) --start-index $(SCANAPP_COLMAP_SEED_START_INDEX) $(SCANAPP_COLMAP_SEED_COPY_IMAGES) $(SCANAPP_COLMAP_SEED_OVERWRITE)

codex-scanapp-bounded-pose-correction:
	conda run -n $(CONDA_ENV) python scripts/test/apply_bounded_colmap_pose_correction.py --seed-model "$(SCANAPP_BOUNDED_POSE_SEED_MODEL)" --teacher-model "$(SCANAPP_BOUNDED_POSE_TEACHER_MODEL)" --out-dir "$(SCANAPP_BOUNDED_POSE_OUT)" --max-rotation-deg $(SCANAPP_BOUNDED_POSE_MAX_ROTATION_DEG) --max-translation-m $(SCANAPP_BOUNDED_POSE_MAX_TRANSLATION_M) --smooth-window $(SCANAPP_BOUNDED_POSE_SMOOTH_WINDOW) --points-source $(SCANAPP_BOUNDED_POSE_POINTS_SOURCE) $(SCANAPP_BOUNDED_POSE_COPY_IMAGES) $(SCANAPP_BOUNDED_POSE_OVERWRITE)

codex-colmap-pose-refine:
	conda run -n $(CONDA_ENV) python scripts/test/refine_colmap_seed_poses.py --data "$(COLMAP_POSE_REFINE_DATA)" --out-dir "$(COLMAP_POSE_REFINE_OUT)" --matcher $(COLMAP_POSE_REFINE_MATCHER) $(COLMAP_POSE_REFINE_USE_GPU) $(COLMAP_POSE_REFINE_INTRINSICS) $(COLMAP_POSE_REFINE_COPY_IMAGES) $(COLMAP_POSE_REFINE_OVERWRITE)

codex-360-points-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_360_points_multiview_3dgs_mlx.py --data "$(COLMAP_360_DATA)" --out-dir "$(COLMAP_360_OUT)" --out-spz "$(COLMAP_360_SPZ)" --out-model-npz "$(COLMAP_360_MODEL_NPZ)" --data-factor $(COLMAP_360_FACTOR) --test-every $(COLMAP_360_TEST_EVERY) --train-split $(COLMAP_360_TRAIN_SPLIT) --width $(COLMAP_360_WIDTH) --height $(COLMAP_360_HEIGHT) --max-frames $(COLMAP_360_MAX_FRAMES) --frame-step $(COLMAP_360_FRAME_STEP) --start-index $(COLMAP_360_START_INDEX) --eval-max-frames $(COLMAP_360_EVAL_FRAMES) --eval-frame-step $(COLMAP_360_EVAL_FRAME_STEP) --eval-start-index $(COLMAP_360_EVAL_START_INDEX) --max-points $(COLMAP_360_MAX_POINTS) --steps $(COLMAP_360_STEPS) --batch-size $(COLMAP_360_BATCH_SIZE) --log-interval $(COLMAP_360_LOG_INTERVAL) --step-image-interval $(COLMAP_360_STEP_IMAGE_INTERVAL) --mlx-cache-limit-gb $(COLMAP_360_MLX_CACHE_LIMIT_GB) --refine-enabled

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
