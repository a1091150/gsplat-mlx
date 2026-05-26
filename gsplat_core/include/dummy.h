#pragma once

#include "mlx/array.h"

namespace gsplat_core {
namespace mx = mlx::core;

int dummy_add(int a, int b);
mx::array dummy_array_add(const mx::array& a, const mx::array& b);

}  // namespace gsplat_core
