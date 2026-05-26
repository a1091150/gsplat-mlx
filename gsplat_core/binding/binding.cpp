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
#include "../include/gsplat_intersect.h"
#include "../include/gsplat_projection.h"
#include "../include/gsplat_rasterize.h"
#include "../include/gsplat_spherical_harmonics.h"

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
        std::string("gsplat_core binding missing input: ") + key);
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

nb::dict intersect_tile_forward(
    const std::unordered_map<std::string, mx::array>& inputs,
    int I,
    int tile_size,
    int tile_width,
    int tile_height,
    bool sort,
    bool segmented) {
  const auto& means2d = require_key(inputs, "means2d");
  const auto& radii = require_key(inputs, "radii");
  const auto& depths = require_key(inputs, "depths");
  mx::array conics = get_or_empty(inputs, "conics");
  mx::array opacities = get_or_empty(inputs, "opacities");
  mx::array image_ids = get_or_empty(inputs, "image_ids");
  mx::array gaussian_ids = get_or_empty(inputs, "gaussian_ids");

  gsplat_core::IntersectTileInput input = {
      .means2d = means2d,
      .radii = radii,
      .depths = depths,
      .conics = conics,
      .opacities = opacities,
      .image_ids = image_ids,
      .gaussian_ids = gaussian_ids,
      .s = mx::Device::cpu,
      .params = {
          .I = I,
          .tile_size = tile_size,
          .tile_width = tile_width,
          .tile_height = tile_height,
          .sort = sort,
          .segmented = segmented,
          .packed = image_ids.size() != 0 || gaussian_ids.size() != 0,
          .use_conics = conics.size() != 0,
          .use_opacities = opacities.size() != 0,
      },
  };

  auto outputs = gsplat_core::gsplat_intersect_tile(input);
  nb::dict result;
  result["tiles_per_gauss"] = outputs[gsplat_core::kTilesPerGauss];
  result["isect_ids"] = outputs[gsplat_core::kIsectIds];
  result["flatten_ids"] = outputs[gsplat_core::kFlattenIds];
  return result;
}

mx::array intersect_offset_forward(
    const mx::array& isect_ids,
    int I,
    int tile_width,
    int tile_height) {
  return gsplat_core::gsplat_intersect_offset(
      isect_ids, I, tile_width, tile_height, mx::Device::cpu);
}

nb::dict rasterize_to_pixels_3dgs_forward(
    const std::unordered_map<std::string, mx::array>& inputs,
    int image_width,
    int image_height,
    int tile_size) {
  const auto& means2d = require_key(inputs, "means2d");
  const auto& conics = require_key(inputs, "conics");
  const auto& colors = require_key(inputs, "colors");
  const auto& opacities = require_key(inputs, "opacities");
  const auto& tile_offsets = require_key(inputs, "tile_offsets");
  const auto& flatten_ids = require_key(inputs, "flatten_ids");
  mx::array backgrounds = get_or_empty(inputs, "backgrounds");
  mx::array masks = get_or_empty(inputs, "masks");

  gsplat_core::RasterizeToPixels3DGSInput input = {
      .means2d = means2d,
      .conics = conics,
      .colors = colors,
      .opacities = opacities,
      .backgrounds = backgrounds,
      .masks = masks,
      .tile_offsets = tile_offsets,
      .flatten_ids = flatten_ids,
      .s = mx::Device::cpu,
      .params = {
          .image_width = image_width,
          .image_height = image_height,
          .tile_size = tile_size,
          .use_backgrounds = backgrounds.size() != 0,
          .use_masks = masks.size() != 0,
          .packed = means2d.ndim() == 2,
      },
  };

  auto outputs = gsplat_core::gsplat_rasterize_to_pixels_3dgs(input);
  nb::dict result;
  result["render_colors"] = outputs[gsplat_core::kRenderColors];
  result["render_alphas"] = outputs[gsplat_core::kRenderAlphas];
  result["last_ids"] = outputs[gsplat_core::kLastIds];
  return result;
}

mx::array spherical_harmonics_forward(
    int degrees_to_use,
    const std::unordered_map<std::string, mx::array>& inputs) {
  const auto& dirs = require_key(inputs, "dirs");
  const auto& coeffs = require_key(inputs, "coeffs");
  mx::array masks = get_or_empty(inputs, "masks");

  gsplat_core::SphericalHarmonicsInput input = {
      .degrees_to_use = degrees_to_use,
      .dirs = dirs,
      .coeffs = coeffs,
      .masks = masks,
      .s = mx::Device::cpu,
      .use_masks = masks.size() != 0,
  };
  return gsplat_core::gsplat_spherical_harmonics_forward(input);
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
  m.def(
      "intersect_tile_forward",
      &intersect_tile_forward,
      "inputs"_a,
      "I"_a,
      "tile_size"_a,
      "tile_width"_a,
      "tile_height"_a,
      "sort"_a = true,
      "segmented"_a = false);
  m.def(
      "intersect_offset_forward",
      &intersect_offset_forward,
      "isect_ids"_a,
      "I"_a,
      "tile_width"_a,
      "tile_height"_a);
  m.def(
      "rasterize_to_pixels_3dgs_forward",
      &rasterize_to_pixels_3dgs_forward,
      "inputs"_a,
      "image_width"_a,
      "image_height"_a,
      "tile_size"_a);
  m.def(
      "spherical_harmonics_forward",
      &spherical_harmonics_forward,
      "degrees_to_use"_a,
      "inputs"_a);
}
