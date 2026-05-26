#pragma once

#include <vector>

#include <mlx/mlx.h>

namespace gsplat_core {
namespace mx = mlx::core;

struct RasterizeToPixels3DGSParams {
  int image_width;
  int image_height;
  int tile_size;
  bool use_backgrounds;
  bool use_masks;
  bool packed;
};

struct RasterizeToPixels3DGSInput {
  mx::array means2d;
  mx::array conics;
  mx::array colors;
  mx::array opacities;
  mx::array backgrounds;
  mx::array masks;
  mx::array tile_offsets;
  mx::array flatten_ids;
  mx::StreamOrDevice s;
  RasterizeToPixels3DGSParams params;
};

enum RasterizeToPixels3DGSOutputIndex {
  kRenderColors = 0,
  kRenderAlphas = 1,
  kLastIds = 2,
};

std::vector<mx::array> gsplat_rasterize_to_pixels_3dgs(
    const RasterizeToPixels3DGSInput& input);

}  // namespace gsplat_core
