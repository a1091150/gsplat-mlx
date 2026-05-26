#include <iostream>

#include "../include/dummy.h"

int main() {
  const int v = gsplat_core::dummy_add(20, 22);
  if (v != 42) {
    std::cerr << "dummy_add failed: expected 42, got " << v << "\n";
    return 1;
  }
  std::cout << "dummy_add ok: " << v << "\n";
  return 0;
}
