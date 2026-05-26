#include "include/helper.h"

#include <dlfcn.h>

#include <filesystem>
#include <stdexcept>

namespace gsplat_core {

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

}  // namespace gsplat_core
