#pragma once

#include <mlx/mlx.h>

namespace gsplat_core {
namespace mx = mlx::core;

struct SphericalHarmonicsInput {
  int degrees_to_use;
  mx::array dirs;
  mx::array coeffs;
  mx::array masks;
  mx::StreamOrDevice s;
  bool use_masks;
};

mx::array gsplat_spherical_harmonics_forward(
    const SphericalHarmonicsInput& input);

}  // namespace gsplat_core
