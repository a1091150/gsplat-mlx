#pragma once

#include <cstdint>
#include <vector>

#include <mlx/mlx.h>

namespace gsplat_core {

struct IntersectTileParams {
  int I;
  int tile_size;
  int tile_width;
  int tile_height;
  bool sort;
  bool segmented;
  bool packed;
  bool use_conics;
  bool use_opacities;
};

struct IntersectTileInput {
  mlx::core::array means2d;
  mlx::core::array radii;
  mlx::core::array depths;
  mlx::core::array conics;
  mlx::core::array opacities;
  mlx::core::array image_ids;
  mlx::core::array gaussian_ids;
  mlx::core::StreamOrDevice s;
  IntersectTileParams params;
};

enum IntersectTileOutputIndex {
  kTilesPerGauss = 0,
  kIsectIds = 1,
  kFlattenIds = 2,
};

std::vector<mlx::core::array> gsplat_intersect_tile(
    const IntersectTileInput& input);

mlx::core::array gsplat_intersect_offset(
    const mlx::core::array& isect_ids,
    int I,
    int tile_width,
    int tile_height,
    mlx::core::StreamOrDevice s = mlx::core::Device::cpu);

}  // namespace gsplat_core
