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
constexpr float kMinOneMinusAlpha = 1.0e-6f;

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
  uint32_t use_masks;
};

mx::Shape render_colors_shape(const mx::array& tile_offsets,
                              int image_height,
                              int image_width,
                              int channels);

void validate_rasterize_input(const RasterizeToPixels3DGSInput& input) {
  if (input.params.packed) {
    throw std::runtime_error(
        "rasterize_to_pixels_3dgs packed path is not implemented yet.");
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
  if (input.params.use_masks) {
    if (input.masks.ndim() != input.tile_offsets.ndim()) {
      throw std::runtime_error(
          "masks must have the same shape as tile_offsets.");
    }
    for (int i = 0; i < static_cast<int>(input.masks.ndim()); ++i) {
      if (input.masks.shape(i) != input.tile_offsets.shape(i)) {
        throw std::runtime_error(
            "masks must have the same shape as tile_offsets.");
      }
    }
  }
}

void validate_rasterize_backward_input(
    const RasterizeToPixels3DGSBackwardInput& input) {
  RasterizeToPixels3DGSInput forward_input = {
      .means2d = input.means2d,
      .conics = input.conics,
      .colors = input.colors,
      .opacities = input.opacities,
      .backgrounds = input.backgrounds,
      .masks = input.masks,
      .tile_offsets = input.tile_offsets,
      .flatten_ids = input.flatten_ids,
      .s = input.s,
      .params = input.params,
  };
  validate_rasterize_input(forward_input);
  if (input.render_alphas.ndim() < 3 ||
      input.render_alphas.shape(static_cast<int>(input.render_alphas.ndim()) - 1) != 1) {
    throw std::runtime_error("render_alphas must have shape [..., H, W, 1].");
  }
  if (input.last_ids.ndim() + 1 != input.render_alphas.ndim()) {
    throw std::runtime_error("last_ids must have shape [..., H, W].");
  }
  if (input.v_render_colors.shape() !=
      render_colors_shape(input.tile_offsets,
                          input.params.image_height,
                          input.params.image_width,
                          input.colors.shape(static_cast<int>(input.colors.ndim()) - 1))) {
    throw std::runtime_error("v_render_colors shape mismatch.");
  }
  if (input.v_render_alphas.shape() != input.render_alphas.shape()) {
    throw std::runtime_error("v_render_alphas shape mismatch.");
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
      mx::contiguous(input.masks),
      mx::contiguous(input.tile_offsets),
      mx::contiguous(input.flatten_ids),
  };
  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

std::vector<mx::array> gsplat_rasterize_to_pixels_3dgs_backward(
    const RasterizeToPixels3DGSBackwardInput& input) {
  validate_rasterize_backward_input(input);

  auto prim = std::make_shared<GSPlatRasterizeToPixels3DGSBackward>(
      to_stream(input.s), input.params, input.absgrad);
  std::vector<mx::Shape> output_shapes = {
      input.absgrad ? input.means2d.shape() : mx::Shape{0},
      input.means2d.shape(),
      input.conics.shape(),
      input.colors.shape(),
      input.opacities.shape(),
      input.params.use_backgrounds ? input.backgrounds.shape() : mx::Shape{0},
  };
  std::vector<mx::Dtype> output_types = {
      mx::float32,
      mx::float32,
      mx::float32,
      mx::float32,
      mx::float32,
      mx::float32,
  };
  std::vector<mx::array> inputs = {
      mx::contiguous(input.means2d),
      mx::contiguous(input.conics),
      mx::contiguous(input.colors),
      mx::contiguous(input.opacities),
      mx::contiguous(input.backgrounds),
      mx::contiguous(input.masks),
      mx::contiguous(input.tile_offsets),
      mx::contiguous(input.flatten_ids),
      mx::contiguous(input.render_alphas),
      mx::contiguous(input.last_ids),
      mx::contiguous(input.v_render_colors),
      mx::contiguous(input.v_render_alphas),
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
  const auto& masks = inputs[5];
  const auto& tile_offsets = inputs[6];
  const auto& flatten_ids = inputs[7];
  mx::eval(means2d, conics, colors, opacities, backgrounds, masks,
           tile_offsets, flatten_ids);

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
  const bool* masks_data = params_.use_masks ? masks.data<bool>() : nullptr;
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
        const bool skip_tile =
            masks_data != nullptr && !masks_data[offset_index];

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

            for (int idx = range_start; !skip_tile && idx < range_end; ++idx) {
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

void GSPlatRasterizeToPixels3DGSBackward::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& means2d = inputs[0];
  const auto& conics = inputs[1];
  const auto& colors = inputs[2];
  const auto& opacities = inputs[3];
  const auto& backgrounds = inputs[4];
  const auto& masks = inputs[5];
  const auto& tile_offsets = inputs[6];
  const auto& flatten_ids = inputs[7];
  const auto& render_alphas = inputs[8];
  const auto& last_ids = inputs[9];
  const auto& v_render_colors = inputs[10];
  const auto& v_render_alphas = inputs[11];
  mx::eval(means2d, conics, colors, opacities, backgrounds, masks,
           tile_offsets, flatten_ids, render_alphas, last_ids,
           v_render_colors, v_render_alphas);

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
  const bool* masks_data = params_.use_masks ? masks.data<bool>() : nullptr;
  const int32_t* tile_offsets_data = tile_offsets.data<int32_t>();
  const int32_t* flatten_ids_data = flatten_ids.data<int32_t>();
  const float* render_alphas_data = render_alphas.data<float>();
  const int32_t* last_ids_data = last_ids.data<int32_t>();
  const float* v_render_colors_data = v_render_colors.data<float>();
  const float* v_render_alphas_data = v_render_alphas.data<float>();

  float* v_means_abs =
      absgrad_ ? outputs[kRasterVMeans2DAbs].data<float>() : nullptr;
  float* v_means = outputs[kRasterVMeans2D].data<float>();
  float* v_conics = outputs[kRasterVConics].data<float>();
  float* v_colors = outputs[kRasterVColors].data<float>();
  float* v_opacities = outputs[kRasterVOpacities].data<float>();
  float* v_backgrounds =
      params_.use_backgrounds ? outputs[kRasterVBackgrounds].data<float>() : nullptr;

  std::vector<int> valid_ids;
  std::vector<float> valid_alphas;
  std::vector<float> valid_vis;
  valid_ids.reserve(static_cast<size_t>(std::max(0, n_isects)));
  valid_alphas.reserve(static_cast<size_t>(std::max(0, n_isects)));
  valid_vis.reserve(static_cast<size_t>(std::max(0, n_isects)));

  for (int image_id = 0; image_id < I; ++image_id) {
    for (int tile_y = 0; tile_y < tile_height; ++tile_y) {
      for (int tile_x = 0; tile_x < tile_width; ++tile_x) {
        const int tile_id = tile_y * tile_width + tile_x;
        const int offset_index = image_id * tile_height * tile_width + tile_id;
        if (masks_data != nullptr && !masks_data[offset_index]) {
          continue;
        }
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
            const int32_t bin_final = last_ids_data[pix_id];
            valid_ids.clear();
            valid_alphas.clear();
            valid_vis.clear();

            const float px = static_cast<float>(x) + 0.5f;
            const float py = static_cast<float>(y) + 0.5f;
            for (int idx = range_start; idx < range_end && idx <= bin_final; ++idx) {
              const int g = flatten_ids_data[idx];
              const float dx = means_data[g * 2] - px;
              const float dy = means_data[g * 2 + 1] - py;
              const float c0 = conics_data[g * 3];
              const float c1 = conics_data[g * 3 + 1];
              const float c2 = conics_data[g * 3 + 2];
              const float sigma =
                  0.5f * (c0 * dx * dx + c2 * dy * dy) + c1 * dx * dy;
              const float vis = std::exp(-sigma);
              const float alpha = std::min(kMaxAlpha, opacities_data[g] * vis);
              if (sigma < 0.0f || alpha < kAlphaThreshold) {
                continue;
              }
              valid_ids.push_back(idx);
              valid_alphas.push_back(alpha);
              valid_vis.push_back(vis);
            }

            const float T_final = 1.0f - render_alphas_data[pix_id];
            float T = T_final;
            std::vector<float> buffer(static_cast<size_t>(channels), 0.0f);
            const float v_render_a = v_render_alphas_data[pix_id];
            const float* v_render_c =
                v_render_colors_data + static_cast<size_t>(pix_id) * channels;

            if (v_backgrounds != nullptr) {
              for (int channel = 0; channel < channels; ++channel) {
                v_backgrounds[image_id * channels + channel] +=
                    T_final * v_render_c[channel];
              }
            }

            for (int local = static_cast<int>(valid_ids.size()) - 1;
                 local >= 0;
                 --local) {
              const int g = flatten_ids_data[valid_ids[static_cast<size_t>(local)]];
              const float alpha = valid_alphas[static_cast<size_t>(local)];
              const float vis = valid_vis[static_cast<size_t>(local)];
              const float one_minus_alpha =
                  std::max(kMinOneMinusAlpha, 1.0f - alpha);
              const float ra = 1.0f / one_minus_alpha;
              T *= ra;
              const float fac = alpha * T;

              for (int channel = 0; channel < channels; ++channel) {
                v_colors[g * channels + channel] += fac * v_render_c[channel];
              }

              float v_alpha = 0.0f;
              for (int channel = 0; channel < channels; ++channel) {
                const float color = colors_data[g * channels + channel];
                v_alpha +=
                    (color * T - buffer[static_cast<size_t>(channel)] * ra) *
                    v_render_c[channel];
              }
              v_alpha += T_final * ra * v_render_a;
              if (backgrounds_data != nullptr) {
                float accum = 0.0f;
                for (int channel = 0; channel < channels; ++channel) {
                  accum += backgrounds_data[image_id * channels + channel] *
                           v_render_c[channel];
                }
                v_alpha += -T_final * ra * accum;
              }

              if (opacities_data[g] * vis <= kMaxAlpha) {
                const float dx = means_data[g * 2] - px;
                const float dy = means_data[g * 2 + 1] - py;
                const float c0 = conics_data[g * 3];
                const float c1 = conics_data[g * 3 + 1];
                const float c2 = conics_data[g * 3 + 2];
                const float v_sigma = -opacities_data[g] * vis * v_alpha;
                const float v_x =
                    v_sigma * (c0 * dx + c1 * dy);
                const float v_y =
                    v_sigma * (c1 * dx + c2 * dy);
                v_conics[g * 3] += 0.5f * v_sigma * dx * dx;
                v_conics[g * 3 + 1] += v_sigma * dx * dy;
                v_conics[g * 3 + 2] += 0.5f * v_sigma * dy * dy;
                v_means[g * 2] += v_x;
                v_means[g * 2 + 1] += v_y;
                if (v_means_abs != nullptr) {
                  v_means_abs[g * 2] += std::fabs(v_x);
                  v_means_abs[g * 2 + 1] += std::fabs(v_y);
                }
                v_opacities[g] += vis * v_alpha;
              }

              for (int channel = 0; channel < channels; ++channel) {
                buffer[static_cast<size_t>(channel)] +=
                    colors_data[g * channels + channel] * fac;
              }
            }
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
  const auto& masks = inputs[5];
  const auto& tile_offsets = inputs[6];
  const auto& flatten_ids = inputs[7];

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
      .use_masks = static_cast<uint32_t>(params_.use_masks),
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
  compute_encoder.set_input_array(masks, 6);
  compute_encoder.set_input_array(tile_offsets, 7);
  compute_encoder.set_input_array(flatten_ids, 8);
  compute_encoder.set_output_array(outputs[kRenderColors], 9);
  compute_encoder.set_output_array(outputs[kRenderAlphas], 10);
  compute_encoder.set_output_array(outputs[kLastIds], 11);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  const size_t tgp_size =
      std::min(static_cast<size_t>(num_pixels), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(num_pixels, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}

void GSPlatRasterizeToPixels3DGSBackward::eval_gpu(
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
  const auto& masks = inputs[5];
  const auto& tile_offsets = inputs[6];
  const auto& flatten_ids = inputs[7];
  const auto& render_alphas = inputs[8];
  const auto& last_ids = inputs[9];
  const auto& v_render_colors = inputs[10];
  const auto& v_render_alphas = inputs[11];

  const uint32_t num_pixels =
      static_cast<uint32_t>(render_alphas.size());
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
      .use_masks = static_cast<uint32_t>(params_.use_masks),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("gsplat_core", current_binary_dir());
  auto kernel = d.get_kernel("gsplat_rasterize_to_pixels_3dgs_backward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_bytes(static_cast<uint32_t>(absgrad_), 1);
  compute_encoder.set_input_array(means2d, 2);
  compute_encoder.set_input_array(conics, 3);
  compute_encoder.set_input_array(colors, 4);
  compute_encoder.set_input_array(opacities, 5);
  compute_encoder.set_input_array(backgrounds, 6);
  compute_encoder.set_input_array(masks, 7);
  compute_encoder.set_input_array(tile_offsets, 8);
  compute_encoder.set_input_array(flatten_ids, 9);
  compute_encoder.set_input_array(render_alphas, 10);
  compute_encoder.set_input_array(last_ids, 11);
  compute_encoder.set_input_array(v_render_colors, 12);
  compute_encoder.set_input_array(v_render_alphas, 13);
  compute_encoder.set_output_array(outputs[kRasterVMeans2DAbs], 14);
  compute_encoder.set_output_array(outputs[kRasterVMeans2D], 15);
  compute_encoder.set_output_array(outputs[kRasterVConics], 16);
  compute_encoder.set_output_array(outputs[kRasterVColors], 17);
  compute_encoder.set_output_array(outputs[kRasterVOpacities], 18);
  compute_encoder.set_output_array(outputs[kRasterVBackgrounds], 19);

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

void GSPlatRasterizeToPixels3DGSBackward::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatRasterizeToPixels3DGSBackward has no GPU implementation.");
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
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
  if (cotangents.size() < 2 || outputs.size() < 3) {
    throw std::runtime_error(
        "GSPlatRasterizeToPixels3DGS vjp expects render color and alpha cotangents.");
  }
  RasterizeToPixels3DGSBackwardInput input = {
      .means2d = primals[0],
      .conics = primals[1],
      .colors = primals[2],
      .opacities = primals[3],
      .backgrounds = primals[4],
      .masks = primals[5],
      .tile_offsets = primals[6],
      .flatten_ids = primals[7],
      .render_alphas = outputs[kRenderAlphas],
      .last_ids = outputs[kLastIds],
      .v_render_colors = cotangents[kRenderColors],
      .v_render_alphas = cotangents[kRenderAlphas],
      .s = stream(),
      .params = params_,
      .absgrad = false,
  };
  auto backward_outputs = gsplat_rasterize_to_pixels_3dgs_backward(input);
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  for (int argnum : argnums) {
    if (argnum == 0) {
      vjps.push_back(backward_outputs[kRasterVMeans2D]);
    } else if (argnum == 1) {
      vjps.push_back(backward_outputs[kRasterVConics]);
    } else if (argnum == 2) {
      vjps.push_back(backward_outputs[kRasterVColors]);
    } else if (argnum == 3) {
      vjps.push_back(backward_outputs[kRasterVOpacities]);
    } else if (argnum == 4 && params_.use_backgrounds) {
      vjps.push_back(backward_outputs[kRasterVBackgrounds]);
    } else {
      throw std::runtime_error(
          "GSPlatRasterizeToPixels3DGS vjp only supports means2d, conics, colors, opacities, and backgrounds.");
    }
  }
  return vjps;
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

std::vector<mx::array> GSPlatRasterizeToPixels3DGSBackward::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "GSPlatRasterizeToPixels3DGSBackward jvp is not implemented.");
}

std::vector<mx::array> GSPlatRasterizeToPixels3DGSBackward::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatRasterizeToPixels3DGSBackward vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GSPlatRasterizeToPixels3DGSBackward::vmap(const std::vector<mx::array>&,
                                          const std::vector<int>&) {
  throw std::runtime_error(
      "GSPlatRasterizeToPixels3DGSBackward vmap is not implemented.");
}

bool GSPlatRasterizeToPixels3DGSBackward::is_equivalent(
    const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  const auto* other_ptr =
      dynamic_cast<const GSPlatRasterizeToPixels3DGSBackward*>(&other);
  if (!other_ptr) {
    return false;
  }
  return params_.image_width == other_ptr->params_.image_width &&
         params_.image_height == other_ptr->params_.image_height &&
         params_.tile_size == other_ptr->params_.tile_size &&
         params_.use_backgrounds == other_ptr->params_.use_backgrounds &&
         params_.use_masks == other_ptr->params_.use_masks &&
         params_.packed == other_ptr->params_.packed &&
         absgrad_ == other_ptr->absgrad_;
}

}  // namespace gsplat_core
