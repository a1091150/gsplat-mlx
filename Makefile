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
SCANNER_TRAIN_STEPS ?= 2000

FIXED_POINTS_DATASET ?= b075x65r3x
FIXED_POINTS_DATA ?= datasets/B075X65R3X
FIXED_POINTS_DATASET_OUT ?= outputs/fixed_points_dataset
FIXED_POINTS_TRAIN_OUT ?= outputs/fixed_points_train
FIXED_POINTS_WIDTH ?= 512
FIXED_POINTS_HEIGHT ?= 512
FIXED_POINTS_CAMERAS ?= 48
FIXED_POINTS_MAX_FRAMES ?= 0
FIXED_POINTS_FRAME_STEP ?= 0
FIXED_POINTS_START_INDEX ?= 0
FIXED_POINTS_GAUSSIANS ?= 2048
FIXED_POINTS_INIT_MODE ?= foreground
FIXED_POINTS_STEPS ?= 4000
FIXED_POINTS_GRID_INTERVAL ?= 200
FIXED_POINTS_GRID_TILES ?= 16
FIXED_POINTS_SEED ?= 84
FIXED_POINTS_SH_DEGREE_START ?= 0
FIXED_POINTS_SH_DEGREE_TARGET ?= 0
FIXED_POINTS_SH_DEGREE_SCHEDULE_INTERVAL ?= 1000

SPZ_VARIANTS_OUT ?= outputs/spz_variants
SPZ_VARIANTS_PREFIX ?= scanner_points
SPZ_VARIANTS_POSITION_MODES ?= scanner
SPZ_VARIANTS_SCALE_MODES ?= direct
SPZ_VARIANTS_ROTATION_MODES ?= position_axis
SPZ_VARIANTS_QUAT_ORDERS ?= xyzw
SPZ_VARIANTS_COLOR_MODES ?= sh

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
SCANNER_POINTS_TRAIN_MODEL_NPZ ?= outputs/scanner_points_multiview_train/trained_model_params.npz
SCANNER_POINTS_TRAIN_SPZ_SCALE_MODE ?= direct
SCANNER_POINTS_TRAIN_SPZ_ROTATION_MODE ?= position_axis
SCANNER_POINTS_TRAIN_SPZ_QUAT_ORDER ?= xyzw
SCANNER_POINTS_TRAIN_SPZ_COLOR_MODE ?= sh
SCANNER_POINTS_TRAIN_WIDTH ?= 512
SCANNER_POINTS_TRAIN_HEIGHT ?= 512
SCANNER_POINTS_TRAIN_MAX_POINTS ?= 75000
SCANNER_POINTS_TRAIN_FRAMES ?= 999
SCANNER_POINTS_TRAIN_FRAME_STEP ?= 1
SCANNER_POINTS_EVAL_FRAMES ?= 0
SCANNER_POINTS_EVAL_FRAME_STEP ?= $(SCANNER_POINTS_TRAIN_FRAME_STEP)
SCANNER_POINTS_EVAL_START_INDEX ?=
SCANNER_POINTS_TRAIN_STEPS ?= 2000
SCANNER_POINTS_TRAIN_BATCH_SIZE ?= 1
SCANNER_POINTS_TRAIN_FRAME_SAMPLING ?= shuffle
SCANNER_POINTS_TRAIN_FRAME_SHUFFLE_SEED ?= 7956
SCANNER_POINTS_TRAIN_POINT_SCALE ?= 0.01
SCANNER_POINTS_TRAIN_POINT_SCALE_MODE ?= scene_fraction
SCANNER_POINTS_TRAIN_POINT_SCALE_FRACTION ?= 0.005
SCANNER_POINTS_TRAIN_LOSS_MODE ?= l1_dssim
SCANNER_POINTS_TRAIN_SSIM_LAMBDA ?= 0.2
SCANNER_POINTS_TRAIN_SSIM_WINDOW_SIZE ?= 11
SCANNER_POINTS_TRAIN_LR_MEANS ?= 0.002
SCANNER_POINTS_TRAIN_LR_MEANS_FINAL ?= 0.0002
SCANNER_POINTS_TRAIN_LR_MEANS_DELAY_MULT ?= 1.0
SCANNER_POINTS_TRAIN_LR_MEANS_MAX_STEPS ?= $(SCANNER_POINTS_TRAIN_STEPS)
SCANNER_POINTS_TRAIN_LR_COLORS ?= 0.02
SCANNER_POINTS_TRAIN_LR_COLORS_FINAL ?= 0.005
SCANNER_POINTS_TRAIN_LR_COLORS_DELAY_MULT ?= 1.0
SCANNER_POINTS_TRAIN_LR_COLORS_MAX_STEPS ?= $(SCANNER_POINTS_TRAIN_STEPS)
SCANNER_POINTS_TRAIN_LR_SH_REST ?= 0.001
SCANNER_POINTS_TRAIN_LR_SH_REST_FINAL ?= 0.0001
SCANNER_POINTS_TRAIN_LR_SH_REST_DELAY_MULT ?= 1.0
SCANNER_POINTS_TRAIN_LR_SH_REST_MAX_STEPS ?= $(SCANNER_POINTS_TRAIN_STEPS)
SCANNER_POINTS_TRAIN_LR_OPACITY ?= 0.005
SCANNER_POINTS_TRAIN_LR_OPACITY_FINAL ?= 0.001
SCANNER_POINTS_TRAIN_LR_OPACITY_DELAY_MULT ?= 1.0
SCANNER_POINTS_TRAIN_LR_OPACITY_MAX_STEPS ?= $(SCANNER_POINTS_TRAIN_STEPS)
SCANNER_POINTS_TRAIN_LR_SCALES ?= 0.001
SCANNER_POINTS_TRAIN_LR_SCALES_FINAL ?= 0.0005
SCANNER_POINTS_TRAIN_LR_SCALES_DELAY_MULT ?= 1.0
SCANNER_POINTS_TRAIN_LR_SCALES_MAX_STEPS ?= $(SCANNER_POINTS_TRAIN_STEPS)
SCANNER_POINTS_TRAIN_LR_QUATS ?= 0.001
SCANNER_POINTS_TRAIN_LR_QUATS_FINAL ?= 0.0001
SCANNER_POINTS_TRAIN_LR_QUATS_DELAY_MULT ?= 1.0
SCANNER_POINTS_TRAIN_LR_QUATS_MAX_STEPS ?= $(SCANNER_POINTS_TRAIN_STEPS)
SCANNER_POINTS_TRAIN_COLOR_MODE ?= sh
SCANNER_POINTS_TRAIN_SH_DEGREE ?= 0
SCANNER_POINTS_TRAIN_MAX_SH_DEGREE ?= 3
SCANNER_POINTS_TRAIN_SH_DEGREE_START ?= 0
SCANNER_POINTS_TRAIN_SH_DEGREE_TARGET ?= 3
SCANNER_POINTS_TRAIN_SH_DEGREE_SCHEDULE_INTERVAL ?= 1000
SCANNER_POINTS_REFINE_ENABLED ?= 0
SCANNER_POINTS_REFINE_PRUNE_OPA ?= 0.005
SCANNER_POINTS_REFINE_GROW_GRAD2D ?= 0.0002
SCANNER_POINTS_REFINE_GROW_SCALE3D ?= 0.01
SCANNER_POINTS_REFINE_GROW_SCALE2D ?= 0.05
SCANNER_POINTS_REFINE_PRUNE_SCALE3D ?= 0.1
SCANNER_POINTS_REFINE_PRUNE_SCALE2D ?= 0.15
SCANNER_POINTS_REFINE_SCALE2D_STOP_ITER ?= 0
SCANNER_POINTS_REFINE_START_ITER ?= 500
SCANNER_POINTS_REFINE_STOP_ITER ?= 15000
SCANNER_POINTS_REFINE_RESET_EVERY ?= 3000
SCANNER_POINTS_REFINE_EVERY ?= 100
SCANNER_POINTS_REFINE_PAUSE_AFTER_RESET ?= 0
SCANNER_POINTS_REFINE_SCENE_SCALE ?= 1.0
SCANNER_POINTS_REFINE_SCENE_SCALE_MODE ?= points_extent
SCANNER_POINTS_REFINE_ABSGRAD ?= 0
SCANNER_POINTS_REFINE_REVISED_OPACITY ?= 0

COLMAP_360_ROOT ?= submodules/gsplat/examples/datasets/data/360_v2
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
COLMAP_360_MLX_CACHE_LIMIT_GB ?= 32

SOFA_DATA ?= datasets/B075X65R3X
SOFA_WIDTH ?= 512
SOFA_HEIGHT ?= 512
SOFA_MAX_FRAMES ?= 0
SOFA_FRAME_STEP ?= 1
SOFA_START_INDEX ?= 0
SOFA_MAX_POINTS ?= 50000
SOFA_STEPS ?= $(COLMAP_360_STEPS)
SOFA_BATCH_SIZE ?= $(COLMAP_360_BATCH_SIZE)
SOFA_OUT ?= outputs/sofa_train
SOFA_SPZ ?= $(SOFA_OUT)/trained_sofa.spz
SOFA_MODEL_NPZ ?= $(SOFA_OUT)/trained_model_params.npz
SOFA_LOG_INTERVAL ?= $(COLMAP_360_LOG_INTERVAL)
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
DODECAHEDRON_MLX_CACHE_LIMIT_GB ?= $(COLMAP_360_MLX_CACHE_LIMIT_GB)
CONDA_BASE := $(shell conda info --base 2>/dev/null)

IMAGE_FITTING_IMAGE_FLAGS =
ifneq ($(strip $(IMAGE_FITTING_IMG_PATH)),)
IMAGE_FITTING_IMAGE_FLAGS += --img-path "$(IMAGE_FITTING_IMG_PATH)"
endif

SCANNER_POINTS_SH_SCHEDULE_FLAGS = --sh-degree-schedule-interval $(SCANNER_POINTS_TRAIN_SH_DEGREE_SCHEDULE_INTERVAL)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_SH_DEGREE_START)),)
SCANNER_POINTS_SH_SCHEDULE_FLAGS += --sh-degree-start $(SCANNER_POINTS_TRAIN_SH_DEGREE_START)
endif
ifneq ($(strip $(SCANNER_POINTS_TRAIN_SH_DEGREE_TARGET)),)
SCANNER_POINTS_SH_SCHEDULE_FLAGS += --sh-degree-target $(SCANNER_POINTS_TRAIN_SH_DEGREE_TARGET)
endif

SCANNER_POINTS_MEANS_LR_FLAGS = --lr-means $(SCANNER_POINTS_TRAIN_LR_MEANS)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_MEANS_FINAL)),)
SCANNER_POINTS_MEANS_LR_FLAGS += --lr-means-final $(SCANNER_POINTS_TRAIN_LR_MEANS_FINAL)
endif
SCANNER_POINTS_MEANS_LR_FLAGS += --lr-means-delay-mult $(SCANNER_POINTS_TRAIN_LR_MEANS_DELAY_MULT)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_MEANS_MAX_STEPS)),)
SCANNER_POINTS_MEANS_LR_FLAGS += --lr-means-max-steps $(SCANNER_POINTS_TRAIN_LR_MEANS_MAX_STEPS)
endif

SCANNER_POINTS_OPTIMIZER_LR_FLAGS = $(SCANNER_POINTS_MEANS_LR_FLAGS)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-colors $(SCANNER_POINTS_TRAIN_LR_COLORS)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_COLORS_FINAL)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-colors-final $(SCANNER_POINTS_TRAIN_LR_COLORS_FINAL)
endif
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-colors-delay-mult $(SCANNER_POINTS_TRAIN_LR_COLORS_DELAY_MULT)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_COLORS_MAX_STEPS)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-colors-max-steps $(SCANNER_POINTS_TRAIN_LR_COLORS_MAX_STEPS)
endif
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_SH_REST)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-sh-rest $(SCANNER_POINTS_TRAIN_LR_SH_REST)
endif
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_SH_REST_FINAL)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-sh-rest-final $(SCANNER_POINTS_TRAIN_LR_SH_REST_FINAL)
endif
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-sh-rest-delay-mult $(SCANNER_POINTS_TRAIN_LR_SH_REST_DELAY_MULT)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_SH_REST_MAX_STEPS)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-sh-rest-max-steps $(SCANNER_POINTS_TRAIN_LR_SH_REST_MAX_STEPS)
endif
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-opacity $(SCANNER_POINTS_TRAIN_LR_OPACITY)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_OPACITY_FINAL)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-opacity-final $(SCANNER_POINTS_TRAIN_LR_OPACITY_FINAL)
endif
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-opacity-delay-mult $(SCANNER_POINTS_TRAIN_LR_OPACITY_DELAY_MULT)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_OPACITY_MAX_STEPS)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-opacity-max-steps $(SCANNER_POINTS_TRAIN_LR_OPACITY_MAX_STEPS)
endif
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-scales $(SCANNER_POINTS_TRAIN_LR_SCALES)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_SCALES_FINAL)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-scales-final $(SCANNER_POINTS_TRAIN_LR_SCALES_FINAL)
endif
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-scales-delay-mult $(SCANNER_POINTS_TRAIN_LR_SCALES_DELAY_MULT)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_SCALES_MAX_STEPS)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-scales-max-steps $(SCANNER_POINTS_TRAIN_LR_SCALES_MAX_STEPS)
endif
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-quats $(SCANNER_POINTS_TRAIN_LR_QUATS)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_QUATS_FINAL)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-quats-final $(SCANNER_POINTS_TRAIN_LR_QUATS_FINAL)
endif
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-quats-delay-mult $(SCANNER_POINTS_TRAIN_LR_QUATS_DELAY_MULT)
ifneq ($(strip $(SCANNER_POINTS_TRAIN_LR_QUATS_MAX_STEPS)),)
SCANNER_POINTS_OPTIMIZER_LR_FLAGS += --lr-quats-max-steps $(SCANNER_POINTS_TRAIN_LR_QUATS_MAX_STEPS)
endif

SCANNER_POINTS_LOSS_FLAGS = --loss-mode $(SCANNER_POINTS_TRAIN_LOSS_MODE)
SCANNER_POINTS_LOSS_FLAGS += --ssim-lambda $(SCANNER_POINTS_TRAIN_SSIM_LAMBDA)
SCANNER_POINTS_LOSS_FLAGS += --ssim-window-size $(SCANNER_POINTS_TRAIN_SSIM_WINDOW_SIZE)

SCANNER_POINTS_DATALOADER_FLAGS = --batch-size $(SCANNER_POINTS_TRAIN_BATCH_SIZE)
SCANNER_POINTS_DATALOADER_FLAGS += --frame-sampling $(SCANNER_POINTS_TRAIN_FRAME_SAMPLING)
SCANNER_POINTS_DATALOADER_FLAGS += --frame-shuffle-seed $(SCANNER_POINTS_TRAIN_FRAME_SHUFFLE_SEED)

SCANNER_POINTS_INITIALIZATION_FLAGS = --point-scale $(SCANNER_POINTS_TRAIN_POINT_SCALE)
SCANNER_POINTS_INITIALIZATION_FLAGS += --point-scale-mode $(SCANNER_POINTS_TRAIN_POINT_SCALE_MODE)
SCANNER_POINTS_INITIALIZATION_FLAGS += --point-scale-fraction $(SCANNER_POINTS_TRAIN_POINT_SCALE_FRACTION)
SCANNER_POINTS_INITIALIZATION_FLAGS += --refine-scene-scale-mode $(SCANNER_POINTS_REFINE_SCENE_SCALE_MODE)

SCANNER_POINTS_EVAL_FLAGS = --eval-max-frames $(SCANNER_POINTS_EVAL_FRAMES)
SCANNER_POINTS_EVAL_FLAGS += --eval-frame-step $(SCANNER_POINTS_EVAL_FRAME_STEP)
ifneq ($(strip $(SCANNER_POINTS_EVAL_START_INDEX)),)
SCANNER_POINTS_EVAL_FLAGS += --eval-start-index $(SCANNER_POINTS_EVAL_START_INDEX)
endif

SCANNER_POINTS_REFINE_FLAGS =
ifeq ($(SCANNER_POINTS_REFINE_ENABLED),1)
SCANNER_POINTS_REFINE_FLAGS += --refine-enabled
SCANNER_POINTS_REFINE_FLAGS += --refine-prune-opa $(SCANNER_POINTS_REFINE_PRUNE_OPA)
SCANNER_POINTS_REFINE_FLAGS += --refine-grow-grad2d $(SCANNER_POINTS_REFINE_GROW_GRAD2D)
SCANNER_POINTS_REFINE_FLAGS += --refine-grow-scale3d $(SCANNER_POINTS_REFINE_GROW_SCALE3D)
SCANNER_POINTS_REFINE_FLAGS += --refine-grow-scale2d $(SCANNER_POINTS_REFINE_GROW_SCALE2D)
SCANNER_POINTS_REFINE_FLAGS += --refine-prune-scale3d $(SCANNER_POINTS_REFINE_PRUNE_SCALE3D)
SCANNER_POINTS_REFINE_FLAGS += --refine-prune-scale2d $(SCANNER_POINTS_REFINE_PRUNE_SCALE2D)
SCANNER_POINTS_REFINE_FLAGS += --refine-scale2d-stop-iter $(SCANNER_POINTS_REFINE_SCALE2D_STOP_ITER)
SCANNER_POINTS_REFINE_FLAGS += --refine-start-iter $(SCANNER_POINTS_REFINE_START_ITER)
SCANNER_POINTS_REFINE_FLAGS += --refine-stop-iter $(SCANNER_POINTS_REFINE_STOP_ITER)
SCANNER_POINTS_REFINE_FLAGS += --refine-reset-every $(SCANNER_POINTS_REFINE_RESET_EVERY)
SCANNER_POINTS_REFINE_FLAGS += --refine-every $(SCANNER_POINTS_REFINE_EVERY)
SCANNER_POINTS_REFINE_FLAGS += --refine-pause-after-reset $(SCANNER_POINTS_REFINE_PAUSE_AFTER_RESET)
SCANNER_POINTS_REFINE_FLAGS += --refine-scene-scale $(SCANNER_POINTS_REFINE_SCENE_SCALE)
endif
ifeq ($(SCANNER_POINTS_REFINE_ABSGRAD),1)
SCANNER_POINTS_REFINE_FLAGS += --refine-absgrad
endif
ifeq ($(SCANNER_POINTS_REFINE_REVISED_OPACITY),1)
SCANNER_POINTS_REFINE_FLAGS += --refine-revised-opacity
endif

.PHONY: help env-check xcode-build pip-install pip-develop codex-xcode-test codex-random-png codex-training-smoke codex-dense-training-smoke codex-tiny-train codex-tiny-multiview-train codex-image-fitting-train codex-scanner-dataset-smoke codex-scanner-random-train codex-fixed-points-dataset codex-fixed-points-train codex-spz-variants codex-scanner-points-align codex-scanner-points-spz codex-scanner-points-train-spz codex-scanner-points-train-spz-refine codex-360-points-train-spz codex-360-points-train-spz-refine codex-sofa-train-spz codex-dodecahedron-train-spz codex-projection-guardrails clean

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
	@printf "  make codex-image-fitting-train  Train MLX 3DGS against one image or the synthetic image-fitting target.\n"
	@printf "  make codex-scanner-dataset-smoke  Render random Gaussians with scanner dataset cameras.\n"
	@printf "  make codex-scanner-random-train  Train random Gaussians against scanner dataset frames.\n"
	@printf "  make codex-fixed-points-dataset  Generate synthetic dodecahedron fixed-point dataset.\n"
	@printf "  make codex-fixed-points-train  Train fixed power-of-two Gaussians on the synthetic dodecahedron dataset.\n"
	@printf "  make codex-spz-variants  Export SPZ convention variants from a saved model parameter NPZ.\n"
	@printf "  make codex-scanner-points-align  Render points.ply with scanner dataset cameras.\n"
	@printf "  make codex-scanner-points-spz  Export scanner points.ply to SPZ.\n"
	@printf "  make codex-scanner-points-train-spz  Train points.ply Gaussians and export SPZ.\n"
	@printf "  make codex-scanner-points-train-spz-refine  Train points.ply Gaussians with refine/densify enabled and export SPZ.\n"
	@printf "  make codex-360-points-train-spz  Train a Mip-NeRF 360/COLMAP scene with gsplat default-style settings and export SPZ.\n"
	@printf "  make codex-360-points-train-spz-refine  Alias for the gsplat-default 360 refine/densify training target.\n"
	@printf "  make codex-sofa-train-spz  Train B075X65R3X with 360-style point init/refine settings and export SPZ.\n"
	@printf "  make codex-dodecahedron-train-spz  Train the generated dodecahedron dataset with 360-style point init/refine settings and export SPZ.\n"
	@printf "    Optional: SCANNER_POINTS_EVAL_FRAMES/FRAME_STEP/START_INDEX enable held-out eval compares.\n"
	@printf "    Optional: SCANNER_POINTS_TRAIN_SH_DEGREE_START/TARGET/SCHEDULE_INTERVAL configure progressive SH degree.\n"
	@printf "    Optional: SCANNER_POINTS_TRAIN_BATCH_SIZE/FRAME_SAMPLING/FRAME_SHUFFLE_SEED configure frame sampling.\n"
	@printf "    Optional: SCANNER_POINTS_TRAIN_POINT_SCALE_MODE/FRACTION and SCANNER_POINTS_REFINE_SCENE_SCALE_MODE configure scene-aware initialization.\n"
	@printf "    Optional: SCANNER_POINTS_TRAIN_LOSS_MODE/SSIM_LAMBDA/SSIM_WINDOW_SIZE configure image loss.\n"
	@printf "    Optional: SCANNER_POINTS_TRAIN_LR_* configure optimizer LR schedules.\n"
	@printf "    Optional: SCANNER_POINTS_REFINE_ENABLED=1 plus SCANNER_POINTS_REFINE_* thresholds enable gsplat-style refinement.\n"
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

codex-fixed-points-dataset:
	conda run -n $(CONDA_ENV) python scripts/test/train_fixed_points_3dgs_mlx.py \
		--dataset "$(FIXED_POINTS_DATASET)" \
		--data "$(FIXED_POINTS_DATA)" \
		--dataset-out "$(FIXED_POINTS_DATASET_OUT)" \
		--out-dir "$(FIXED_POINTS_TRAIN_OUT)" \
		--width $(FIXED_POINTS_WIDTH) \
		--height $(FIXED_POINTS_HEIGHT) \
		--num-cameras $(FIXED_POINTS_CAMERAS) \
		--max-frames $(FIXED_POINTS_MAX_FRAMES) \
		--frame-step $(FIXED_POINTS_FRAME_STEP) \
		--start-index $(FIXED_POINTS_START_INDEX) \
		--num-gaussians $(FIXED_POINTS_GAUSSIANS) \
		--init-mode $(FIXED_POINTS_INIT_MODE) \
		--steps $(FIXED_POINTS_STEPS) \
		--grid-interval $(FIXED_POINTS_GRID_INTERVAL) \
		--grid-tiles $(FIXED_POINTS_GRID_TILES) \
		--sh-degree-start $(FIXED_POINTS_SH_DEGREE_START) \
		--sh-degree-target $(FIXED_POINTS_SH_DEGREE_TARGET) \
		--sh-degree-schedule-interval $(FIXED_POINTS_SH_DEGREE_SCHEDULE_INTERVAL) \
		--seed $(FIXED_POINTS_SEED) \
		--dataset-only

codex-fixed-points-train:
	conda run -n $(CONDA_ENV) python scripts/test/train_fixed_points_3dgs_mlx.py \
		--dataset "$(FIXED_POINTS_DATASET)" \
		--data "$(FIXED_POINTS_DATA)" \
		--dataset-out "$(FIXED_POINTS_DATASET_OUT)" \
		--out-dir "$(FIXED_POINTS_TRAIN_OUT)" \
		--width $(FIXED_POINTS_WIDTH) \
		--height $(FIXED_POINTS_HEIGHT) \
		--num-cameras $(FIXED_POINTS_CAMERAS) \
		--max-frames $(FIXED_POINTS_MAX_FRAMES) \
		--frame-step $(FIXED_POINTS_FRAME_STEP) \
		--start-index $(FIXED_POINTS_START_INDEX) \
		--num-gaussians $(FIXED_POINTS_GAUSSIANS) \
		--init-mode $(FIXED_POINTS_INIT_MODE) \
		--steps $(FIXED_POINTS_STEPS) \
		--grid-interval $(FIXED_POINTS_GRID_INTERVAL) \
		--grid-tiles $(FIXED_POINTS_GRID_TILES) \
		--sh-degree-start $(FIXED_POINTS_SH_DEGREE_START) \
		--sh-degree-target $(FIXED_POINTS_SH_DEGREE_TARGET) \
		--sh-degree-schedule-interval $(FIXED_POINTS_SH_DEGREE_SCHEDULE_INTERVAL) \
		--seed $(FIXED_POINTS_SEED)

codex-spz-variants:
	conda run -n $(CONDA_ENV) python scripts/test/export_spz_variants_from_model_npz.py \
		--model-npz "$(SCANNER_POINTS_TRAIN_MODEL_NPZ)" \
		--out-dir "$(SPZ_VARIANTS_OUT)" \
		--prefix "$(SPZ_VARIANTS_PREFIX)" \
		--position-modes "$(SPZ_VARIANTS_POSITION_MODES)" \
		--scale-modes "$(SPZ_VARIANTS_SCALE_MODES)" \
		--rotation-modes "$(SPZ_VARIANTS_ROTATION_MODES)" \
		--quat-orders "$(SPZ_VARIANTS_QUAT_ORDERS)" \
		--color-modes "$(SPZ_VARIANTS_COLOR_MODES)"

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
		--out-model-npz "$(SCANNER_POINTS_TRAIN_MODEL_NPZ)" \
		--spz-scale-mode $(SCANNER_POINTS_TRAIN_SPZ_SCALE_MODE) \
		--spz-rotation-mode $(SCANNER_POINTS_TRAIN_SPZ_ROTATION_MODE) \
		--spz-quat-order $(SCANNER_POINTS_TRAIN_SPZ_QUAT_ORDER) \
		--spz-color-mode $(SCANNER_POINTS_TRAIN_SPZ_COLOR_MODE) \
		--width $(SCANNER_POINTS_TRAIN_WIDTH) \
		--height $(SCANNER_POINTS_TRAIN_HEIGHT) \
		--max-frames $(SCANNER_POINTS_TRAIN_FRAMES) \
		--frame-step $(SCANNER_POINTS_TRAIN_FRAME_STEP) \
		$(SCANNER_POINTS_EVAL_FLAGS) \
		--max-points $(SCANNER_POINTS_TRAIN_MAX_POINTS) \
		--steps $(SCANNER_POINTS_TRAIN_STEPS) \
		$(SCANNER_POINTS_DATALOADER_FLAGS) \
		$(SCANNER_POINTS_INITIALIZATION_FLAGS) \
		$(SCANNER_POINTS_LOSS_FLAGS) \
		$(SCANNER_POINTS_OPTIMIZER_LR_FLAGS) \
		--color-mode $(SCANNER_POINTS_TRAIN_COLOR_MODE) \
		--sh-degree $(SCANNER_POINTS_TRAIN_SH_DEGREE) \
		--max-sh-degree $(SCANNER_POINTS_TRAIN_MAX_SH_DEGREE) \
		$(SCANNER_POINTS_SH_SCHEDULE_FLAGS) \
		$(SCANNER_POINTS_REFINE_FLAGS)

codex-scanner-points-train-spz-refine: SCANNER_POINTS_TRAIN_OUT = outputs/scanner_points_multiview_train_refine
codex-scanner-points-train-spz-refine: SCANNER_POINTS_TRAIN_SPZ = outputs/scanner_points_multiview_train_refine/trained_scanner_points.spz
codex-scanner-points-train-spz-refine: SCANNER_POINTS_TRAIN_MODEL_NPZ = outputs/scanner_points_multiview_train_refine/trained_model_params.npz
codex-scanner-points-train-spz-refine: SCANNER_POINTS_REFINE_FLAGS = --refine-enabled --refine-prune-opa $(SCANNER_POINTS_REFINE_PRUNE_OPA) --refine-grow-grad2d $(SCANNER_POINTS_REFINE_GROW_GRAD2D) --refine-grow-scale3d $(SCANNER_POINTS_REFINE_GROW_SCALE3D) --refine-grow-scale2d $(SCANNER_POINTS_REFINE_GROW_SCALE2D) --refine-prune-scale3d $(SCANNER_POINTS_REFINE_PRUNE_SCALE3D) --refine-prune-scale2d $(SCANNER_POINTS_REFINE_PRUNE_SCALE2D) --refine-scale2d-stop-iter $(SCANNER_POINTS_REFINE_SCALE2D_STOP_ITER) --refine-start-iter $(SCANNER_POINTS_REFINE_START_ITER) --refine-stop-iter $(SCANNER_POINTS_REFINE_STOP_ITER) --refine-reset-every $(SCANNER_POINTS_REFINE_RESET_EVERY) --refine-every $(SCANNER_POINTS_REFINE_EVERY) --refine-pause-after-reset $(SCANNER_POINTS_REFINE_PAUSE_AFTER_RESET) --refine-scene-scale $(SCANNER_POINTS_REFINE_SCENE_SCALE) --refine-revised-opacity
codex-scanner-points-train-spz-refine: codex-scanner-points-train-spz

codex-360-points-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_360_points_multiview_3dgs_mlx.py --data "$(COLMAP_360_DATA)" --out-dir "$(COLMAP_360_OUT)" --out-spz "$(COLMAP_360_SPZ)" --out-model-npz "$(COLMAP_360_MODEL_NPZ)" --data-factor $(COLMAP_360_FACTOR) --test-every $(COLMAP_360_TEST_EVERY) --width $(COLMAP_360_WIDTH) --height $(COLMAP_360_HEIGHT) --max-frames $(COLMAP_360_MAX_FRAMES) --frame-step $(COLMAP_360_FRAME_STEP) --start-index $(COLMAP_360_START_INDEX) --eval-max-frames $(COLMAP_360_EVAL_FRAMES) --eval-frame-step $(COLMAP_360_EVAL_FRAME_STEP) --eval-start-index $(COLMAP_360_EVAL_START_INDEX) --max-points $(COLMAP_360_MAX_POINTS) --steps $(COLMAP_360_STEPS) --batch-size $(COLMAP_360_BATCH_SIZE) --log-interval $(COLMAP_360_LOG_INTERVAL) --mlx-cache-limit-gb $(COLMAP_360_MLX_CACHE_LIMIT_GB) --refine-enabled

codex-360-points-train-spz-refine: codex-360-points-train-spz

codex-sofa-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_sofa_points_multiview_3dgs_mlx.py --dataset b075x65r3x --data "$(SOFA_DATA)" --out-dir "$(SOFA_OUT)" --out-spz "$(SOFA_SPZ)" --out-model-npz "$(SOFA_MODEL_NPZ)" --width $(SOFA_WIDTH) --height $(SOFA_HEIGHT) --max-frames $(SOFA_MAX_FRAMES) --frame-step $(SOFA_FRAME_STEP) --start-index $(SOFA_START_INDEX) --max-points $(SOFA_MAX_POINTS) --steps $(SOFA_STEPS) --batch-size $(SOFA_BATCH_SIZE) --log-interval $(SOFA_LOG_INTERVAL) --mlx-cache-limit-gb $(SOFA_MLX_CACHE_LIMIT_GB) --refine-enabled

codex-dodecahedron-train-spz:
	conda run -n $(CONDA_ENV) python scripts/test/train_dodecahedron_points_multiview_3dgs_mlx.py --dataset dodecahedron --dataset-out "$(DODECAHEDRON_DATASET_OUT)" --out-dir "$(DODECAHEDRON_OUT)" --out-spz "$(DODECAHEDRON_SPZ)" --out-model-npz "$(DODECAHEDRON_MODEL_NPZ)" --width $(DODECAHEDRON_WIDTH) --height $(DODECAHEDRON_HEIGHT) --num-cameras $(DODECAHEDRON_CAMERAS) --camera-radius $(DODECAHEDRON_CAMERA_RADIUS) --focal-scale $(DODECAHEDRON_FOCAL_SCALE) --max-frames $(DODECAHEDRON_MAX_FRAMES) --frame-step $(DODECAHEDRON_FRAME_STEP) --start-index $(DODECAHEDRON_START_INDEX) --max-points $(DODECAHEDRON_MAX_POINTS) --steps $(DODECAHEDRON_STEPS) --batch-size $(DODECAHEDRON_BATCH_SIZE) --log-interval $(DODECAHEDRON_LOG_INTERVAL) --mlx-cache-limit-gb $(DODECAHEDRON_MLX_CACHE_LIMIT_GB) --refine-enabled

codex-projection-guardrails:
	conda run -n $(CONDA_ENV) python scripts/test/training_viewspace_proxy_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/training_projection_viewspace_proxy_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/training_dense_3dgs_loop_smoke.py
	conda run -n $(CONDA_ENV) python scripts/test/projection_vjp_guardrails.py

clean:
	rm -rf $(XCODE_BUILD_DIR) build build-* dist *.egg-info python_package/*.egg-info
