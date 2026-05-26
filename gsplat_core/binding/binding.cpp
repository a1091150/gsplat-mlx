#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include <stdexcept>
#include <string>
#include <unordered_map>

#include <mlx/array.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

#include "../include/dummy.h"
#include "../include/gsplat_projection.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {

const mx::array& require_key(
    const std::unordered_map<std::string, mx::array>& inputs,
    const char* key) {
  auto it = inputs.find(key);
  if (it == inputs.end()) {
    throw std::runtime_error(
        std::string("projection_ewa_3dgs_fused_forward missing input: ") + key);
  }
  return it->second;
}

mx::array get_or_empty(
    const std::unordered_map<std::string, mx::array>& inputs,
    const char* key) {
  auto it = inputs.find(key);
  if (it == inputs.end()) {
    return mx::zeros({0}, mx::float32);
  }
  return it->second;
}

mx::array require_dict_array(nb::dict& inputs, const char* key) {
  nb::handle obj = inputs[key];
  auto* array = nb::inst_ptr<mx::array>(obj);
  if (array == nullptr) {
    throw std::runtime_error(
        std::string("dummy_array_add input is not an MLX array: ") + key);
  }
  return *array;
}

mx::array dummy_array_add_from_dict(nb::dict& inputs) {
  mx::array a = require_dict_array(inputs, "a");
  mx::array b = require_dict_array(inputs, "b");
  return gsplat_core::dummy_array_add(a, b);
}

nb::dict projection_ewa_3dgs_fused_forward(
    const std::unordered_map<std::string, mx::array>& inputs,
    int image_width,
    int image_height,
    float eps2d,
    float near_plane,
    float far_plane,
    float radius_clip,
    bool calc_compensations,
    int camera_model) {
  const auto& means = require_key(inputs, "means");
  const auto& viewmats = require_key(inputs, "viewmats");
  const auto& Ks = require_key(inputs, "Ks");
  mx::array covars = get_or_empty(inputs, "covars");
  mx::array quats = get_or_empty(inputs, "quats");
  mx::array scales = get_or_empty(inputs, "scales");
  mx::array opacities = get_or_empty(inputs, "opacities");

  const bool use_covars = covars.size() != 0;
  const bool use_opacities = opacities.size() != 0;
  gsplat_core::ProjectionEWA3DGSFusedInput input = {
      .means = means,
      .covars = covars,
      .quats = quats,
      .scales = scales,
      .opacities = opacities,
      .viewmats = viewmats,
      .Ks = Ks,
      .s = mx::Device::gpu,
      .params = {
          .image_width = image_width,
          .image_height = image_height,
          .eps2d = eps2d,
          .near_plane = near_plane,
          .far_plane = far_plane,
          .radius_clip = radius_clip,
          .calc_compensations = calc_compensations,
          .camera_model = camera_model,
          .use_covars = use_covars,
          .use_opacities = use_opacities,
      },
  };

  auto outputs = gsplat_core::gsplat_projection_ewa_3dgs_fused(input);
  nb::dict result;
  result["radii"] = outputs[gsplat_core::kRadii];
  result["means2d"] = outputs[gsplat_core::kMeans2D];
  result["depths"] = outputs[gsplat_core::kDepths];
  result["conics"] = outputs[gsplat_core::kConics];
  result["compensations"] = outputs[gsplat_core::kCompensations];
  return result;
}

}  // namespace

NB_MODULE(_gsplat_core, m) {
  nb::module_::import_("mlx.core");

  m.def("dummy_add", &gsplat_core::dummy_add, "a"_a, "b"_a);
  m.def("dummy_array_add", &dummy_array_add_from_dict, "inputs"_a);
  m.def(
      "projection_ewa_3dgs_fused_forward",
      &projection_ewa_3dgs_fused_forward,
      "inputs"_a,
      "image_width"_a,
      "image_height"_a,
      "eps2d"_a = 0.3f,
      "near_plane"_a = 0.01f,
      "far_plane"_a = 1.0e10f,
      "radius_clip"_a = 0.0f,
      "calc_compensations"_a = false,
      "camera_model"_a = 0);
}
