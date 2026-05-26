#include "include/dummy.h"

#include "mlx/ops.h"

namespace gsplat_core {

int dummy_add(int a, int b) {
  return a + b;
}

mx::array dummy_array_add(const mx::array& a, const mx::array& b) {
  return mx::add(a, b);
}

}  // namespace gsplat_core
