#include "include/gsplat_rasterize.h"

#include "include/helper.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/mlx.h"
#include "mlx/ops.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace gsplat_core {
namespace {

constexpr float kAlphaThreshold = 1.0f / 255.0f;
constexpr float kMaxAlpha = 0.99f;
constexpr float kTransmittanceThreshold = 1.0e-4f;

struct RasterizeToPixels3DGSKernelParams {
  uint32_t I;
  uint32_t N;
  uint32_t channels;
  uint32_t image_width;
  uint32_t image_height;
  uint32_t tile_size;
  uint32_t tile_width;
  uint32_t tile_height;
  uint32_t n_isects;
  uint32_t use_backgrounds;
};

void validate_rasterize_input(const RasterizeToPixels3DGSInput& input) {
  if (input.params.packed) {
    throw std::runtime_error(
        "rasterize_to_pixels_3dgs packed path is not implemented yet.");
  }
  if (input.params.use_masks) {
    throw std::runtime_error(
        "rasterize_to_pixels_3dgs masks path is not implemented yet.");
  }
  if (input.means2d.ndim() < 3 ||
      input.means2d.shape(static_cast<int>(input.means2d.ndim()) - 1) != 2) {
    throw std::runtime_error("means2d must have shape [..., N, 2].");
  }
  if (input.conics.ndim() != input.means2d.ndim() ||
      input.conics.shape(static_cast<int>(input.conics.ndim()) - 1) != 3) {
    throw std::runtime_error("conics must have shape [..., N, 3].");
  }
  if (input.colors.ndim() != input.means2d.ndim() ||
      input.colors.shape(static_cast<int>(input.colors.ndim()) - 2) !=
          input.means2d.shape(static_cast<int>(input.means2d.ndim()) - 2)) {
    throw std::runtime_error("colors must have shape [..., N, channels].");
  }
  if (input.opacities.ndim() + 1 != input.means2d.ndim()) {
    throw std::runtime_error("opacities must have shape [..., N].");
  }
  if (input.tile_offsets.ndim() < 3) {
    throw std::runtime_error(
        "tile_offsets must have shape [..., tile_height, tile_width].");
  }
  if (input.flatten_ids.ndim() != 1) {
    throw std::runtime_error("flatten_ids must have shape [n_isects].");
  }
}

mx::Shape image_prefix_shape(const mx::array& tile_offsets) {
  mx::Shape shape;
  const int ndim = static_cast<int>(tile_offsets.ndim());
  shape.reserve(std::max(0, ndim - 2));
  for (int i = 0; i < ndim - 2; ++i) {
    shape.push_back(tile_offsets.shape(i));
  }
  return shape;
}

mx::Shape render_colors_shape(const mx::array& tile_offsets,
                              int image_height,
                              int image_width,
                              int channels) {
  mx::Shape shape = image_prefix_shape(tile_offsets);
  shape.push_back(image_height);
  shape.push_back(image_width);
  shape.push_back(channels);
  return shape;
}

mx::Shape render_alphas_shape(const mx::array& tile_offsets,
                              int image_height,
                              int image_width) {
  mx::Shape shape = image_prefix_shape(tile_offsets);
  shape.push_back(image_height);
  shape.push_back(image_width);
  shape.push_back(1);
  return shape;
}

mx::Shape last_ids_shape(const mx::array& tile_offsets,
                         int image_height,
                         int image_width) {
  mx::Shape shape = image_prefix_shape(tile_offsets);
  shape.push_back(image_height);
  shape.push_back(image_width);
  return shape;
}

}  // namespace

std::vector<mx::array> gsplat_rasterize_to_pixels_3dgs(
    const RasterizeToPixels3DGSInput& input) {
  validate_rasterize_input(input);

  const int image_width = input.params.image_width;
  const int image_height = input.params.image_height;
  const int channels = input.colors.shape(static_cast<int>(input.colors.ndim()) - 1);

  const int means_ndim = static_cast<int>(input.means2d.ndim());
  const int tile_offsets_ndim = static_cast<int>(input.tile_offsets.ndim());
  const int N = input.means2d.shape(means_ndim - 2);
  const int tile_height = input.tile_offsets.shape(tile_offsets_ndim - 2);
  const int tile_width = input.tile_offsets.shape(tile_offsets_ndim - 1);
  const int I = static_cast<int>(
      input.tile_offsets.size() / (tile_height * tile_width));
  if (input.means2d.size() / 2 != static_cast<size_t>(I * N)) {
    throw std::runtime_error(
        "rasterize_to_pixels_3dgs dense path expects means2d size to equal I * N * 2.");
  }

  auto prim = std::make_shared<GSPlatRasterizeToPixels3DGS>(
      to_stream(input.s), input.params);
  std::vector<mx::Shape> output_shapes = {
      render_colors_shape(input.tile_offsets, image_height, image_width, channels),
      render_alphas_shape(input.tile_offsets, image_height, image_width),
      last_ids_shape(input.tile_offsets, image_height, image_width),
  };
  std::vector<mx::Dtype> output_types = {
      mx::float32,
      mx::float32,
      mx::int32,
  };
  std::vector<mx::array> inputs = {
      mx::contiguous(input.means2d),
      mx::contiguous(input.conics),
      mx::contiguous(input.colors),
      mx::contiguous(input.opacities),
      mx::contiguous(input.backgrounds),
      mx::contiguous(input.tile_offsets),
      mx::contiguous(input.flatten_ids),
  };
  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

void GSPlatRasterizeToPixels3DGS::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& means2d = inputs[0];
  const auto& conics = inputs[1];
  const auto& colors = inputs[2];
  const auto& opacities = inputs[3];
  const auto& backgrounds = inputs[4];
  const auto& tile_offsets = inputs[5];
  const auto& flatten_ids = inputs[6];
  mx::eval(means2d, conics, colors, opacities, backgrounds, tile_offsets,
           flatten_ids);

  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  const int means_ndim = static_cast<int>(means2d.ndim());
  const int tile_offsets_ndim = static_cast<int>(tile_offsets.ndim());
  const int N = means2d.shape(means_ndim - 2);
  const int channels = colors.shape(static_cast<int>(colors.ndim()) - 1);
  const int tile_height = tile_offsets.shape(tile_offsets_ndim - 2);
  const int tile_width = tile_offsets.shape(tile_offsets_ndim - 1);
  const int I = static_cast<int>(
      tile_offsets.size() / (tile_height * tile_width));
  const int n_isects = static_cast<int>(flatten_ids.size());

  const float* means_data = means2d.data<float>();
  const float* conics_data = conics.data<float>();
  const float* colors_data = colors.data<float>();
  const float* opacities_data = opacities.data<float>();
  const float* backgrounds_data =
      params_.use_backgrounds ? backgrounds.data<float>() : nullptr;
  const int32_t* tile_offsets_data = tile_offsets.data<int32_t>();
  const int32_t* flatten_ids_data = flatten_ids.data<int32_t>();
  float* render_colors = outputs[kRenderColors].data<float>();
  float* render_alphas = outputs[kRenderAlphas].data<float>();
  int32_t* last_ids = outputs[kLastIds].data<int32_t>();

  for (int image_id = 0; image_id < I; ++image_id) {
    for (int tile_y = 0; tile_y < tile_height; ++tile_y) {
      for (int tile_x = 0; tile_x < tile_width; ++tile_x) {
        const int tile_id = tile_y * tile_width + tile_x;
        const int offset_index = image_id * tile_height * tile_width + tile_id;
        const int range_start = tile_offsets_data[offset_index];
        const int range_end =
            (image_id == I - 1 && tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets_data[offset_index + 1];

        for (int local_y = 0; local_y < params_.tile_size; ++local_y) {
          const int y = tile_y * params_.tile_size + local_y;
          if (y >= params_.image_height) {
            continue;
          }
          for (int local_x = 0; local_x < params_.tile_size; ++local_x) {
            const int x = tile_x * params_.tile_size + local_x;
            if (x >= params_.image_width) {
              continue;
            }

            const int pix_id = image_id * params_.image_height *
                                   params_.image_width +
                               y * params_.image_width + x;
            const float px = static_cast<float>(x) + 0.5f;
            const float py = static_cast<float>(y) + 0.5f;
            float T = 1.0f;
            int32_t cur_idx = 0;

            for (int idx = range_start; idx < range_end; ++idx) {
              const int g = flatten_ids_data[idx];
              const float mean_x = means_data[g * 2];
              const float mean_y = means_data[g * 2 + 1];
              const float dx = mean_x - px;
              const float dy = mean_y - py;
              const float c0 = conics_data[g * 3];
              const float c1 = conics_data[g * 3 + 1];
              const float c2 = conics_data[g * 3 + 2];
              const float sigma =
                  0.5f * (c0 * dx * dx + c2 * dy * dy) + c1 * dx * dy;
              const float alpha =
                  std::min(kMaxAlpha, opacities_data[g] * std::exp(-sigma));
              if (sigma < 0.0f || alpha < kAlphaThreshold) {
                continue;
              }

              const float next_T = T * (1.0f - alpha);
              if (next_T <= kTransmittanceThreshold) {
                break;
              }

              const float visibility = alpha * T;
              for (int channel = 0; channel < channels; ++channel) {
                render_colors[static_cast<size_t>(pix_id * channels + channel)] +=
                    colors_data[g * channels + channel] * visibility;
              }
              cur_idx = idx;
              T = next_T;
            }

            render_alphas[static_cast<size_t>(pix_id)] = 1.0f - T;
            for (int channel = 0; channel < channels; ++channel) {
              const float background =
                  backgrounds_data == nullptr
                      ? 0.0f
                      : backgrounds_data[image_id * channels + channel];
              render_colors[static_cast<size_t>(pix_id * channels + channel)] +=
                  T * background;
            }
            last_ids[static_cast<size_t>(pix_id)] = cur_idx;
          }
        }
      }
    }
  }
}

#ifdef _METAL_
void GSPlatRasterizeToPixels3DGS::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  const auto& means2d = inputs[0];
  const auto& conics = inputs[1];
  const auto& colors = inputs[2];
  const auto& opacities = inputs[3];
  const auto& backgrounds = inputs[4];
  const auto& tile_offsets = inputs[5];
  const auto& flatten_ids = inputs[6];

  const uint32_t num_pixels =
      static_cast<uint32_t>(outputs[kLastIds].size());
  if (num_pixels == 0) {
    return;
  }

  const int means_ndim = static_cast<int>(means2d.ndim());
  const int tile_offsets_ndim = static_cast<int>(tile_offsets.ndim());
  const int tile_height = tile_offsets.shape(tile_offsets_ndim - 2);
  const int tile_width = tile_offsets.shape(tile_offsets_ndim - 1);
  const uint32_t n_tiles = static_cast<uint32_t>(tile_height * tile_width);
  RasterizeToPixels3DGSKernelParams kernel_params = {
      .I = static_cast<uint32_t>(tile_offsets.size() / n_tiles),
      .N = static_cast<uint32_t>(means2d.shape(means_ndim - 2)),
      .channels = static_cast<uint32_t>(
          colors.shape(static_cast<int>(colors.ndim()) - 1)),
      .image_width = static_cast<uint32_t>(params_.image_width),
      .image_height = static_cast<uint32_t>(params_.image_height),
      .tile_size = static_cast<uint32_t>(params_.tile_size),
      .tile_width = static_cast<uint32_t>(tile_width),
      .tile_height = static_cast<uint32_t>(tile_height),
      .n_isects = static_cast<uint32_t>(flatten_ids.size()),
      .use_backgrounds = static_cast<uint32_t>(params_.use_backgrounds),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("gsplat_core", current_binary_dir());
  auto kernel = d.get_kernel("gsplat_rasterize_to_pixels_3dgs_forward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(means2d, 1);
  compute_encoder.set_input_array(conics, 2);
  compute_encoder.set_input_array(colors, 3);
  compute_encoder.set_input_array(opacities, 4);
  compute_encoder.set_input_array(backgrounds, 5);
  compute_encoder.set_input_array(tile_offsets, 6);
  compute_encoder.set_input_array(flatten_ids, 7);
  compute_encoder.set_output_array(outputs[kRenderColors], 8);
  compute_encoder.set_output_array(outputs[kRenderAlphas], 9);
  compute_encoder.set_output_array(outputs[kLastIds], 10);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  const size_t tgp_size =
      std::min(static_cast<size_t>(num_pixels), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(num_pixels, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}
#else
void GSPlatRasterizeToPixels3DGS::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatRasterizeToPixels3DGS has no GPU implementation.");
}
#endif

std::vector<mx::array> GSPlatRasterizeToPixels3DGS::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "GSPlatRasterizeToPixels3DGS jvp is not implemented.");
}

std::vector<mx::array> GSPlatRasterizeToPixels3DGS::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatRasterizeToPixels3DGS vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GSPlatRasterizeToPixels3DGS::vmap(const std::vector<mx::array>&,
                                  const std::vector<int>&) {
  throw std::runtime_error(
      "GSPlatRasterizeToPixels3DGS vmap is not implemented.");
}

bool GSPlatRasterizeToPixels3DGS::is_equivalent(
    const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  const auto* other_ptr =
      dynamic_cast<const GSPlatRasterizeToPixels3DGS*>(&other);
  if (!other_ptr) {
    return false;
  }
  return params_.image_width == other_ptr->params_.image_width &&
         params_.image_height == other_ptr->params_.image_height &&
         params_.tile_size == other_ptr->params_.tile_size &&
         params_.use_backgrounds == other_ptr->params_.use_backgrounds &&
         params_.use_masks == other_ptr->params_.use_masks &&
         params_.packed == other_ptr->params_.packed;
}

}  // namespace gsplat_core
