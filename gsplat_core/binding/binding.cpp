#include <nanobind/nanobind.h>

#include "../include/dummy.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_gsplat_core, m) {
  m.def("dummy_add", &gsplat_core::dummy_add, "a"_a, "b"_a);
}
