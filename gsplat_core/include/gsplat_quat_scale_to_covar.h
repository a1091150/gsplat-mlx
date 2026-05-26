#pragma once

#include <vector>

#include <mlx/mlx.h>

namespace gsplat_core {
namespace mx = mlx::core;

struct QuatScaleToCovarPreciInput {
  mx::array quats;
  mx::array scales;
  mx::StreamOrDevice s;
  bool compute_covar;
  bool compute_preci;
  bool triu;
};

enum QuatScaleToCovarPreciOutputIndex {
  kCovars = 0,
  kPrecis = 1,
};

std::vector<mx::array> gsplat_quat_scale_to_covar_preci_forward(
    const QuatScaleToCovarPreciInput& input);

}  // namespace gsplat_core
