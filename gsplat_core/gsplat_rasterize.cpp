#include "include/gsplat_rasterize.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/ops.h"

namespace gsplat_core {
namespace {

constexpr float kAlphaThreshold = 1.0f / 255.0f;
constexpr float kMaxAlpha = 0.99f;
constexpr float kTransmittanceThreshold = 1.0e-4f;

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

  mx::array means2d = mx::contiguous(input.means2d);
  mx::array conics = mx::contiguous(input.conics);
  mx::array colors = mx::contiguous(input.colors);
  mx::array opacities = mx::contiguous(input.opacities);
  mx::array backgrounds = mx::contiguous(input.backgrounds);
  mx::array tile_offsets = mx::contiguous(input.tile_offsets);
  mx::array flatten_ids = mx::contiguous(input.flatten_ids);
  mx::eval(means2d, conics, colors, opacities, backgrounds, tile_offsets,
           flatten_ids);

  const int means_ndim = static_cast<int>(means2d.ndim());
  const int tile_offsets_ndim = static_cast<int>(tile_offsets.ndim());
  const int N = means2d.shape(means_ndim - 2);
  const int channels = colors.shape(static_cast<int>(colors.ndim()) - 1);
  const int tile_height = tile_offsets.shape(tile_offsets_ndim - 2);
  const int tile_width = tile_offsets.shape(tile_offsets_ndim - 1);
  const int I = static_cast<int>(tile_offsets.size() / (tile_height * tile_width));
  const int n_isects = static_cast<int>(flatten_ids.size());
  const int image_width = input.params.image_width;
  const int image_height = input.params.image_height;
  const int tile_size = input.params.tile_size;

  if (means2d.size() / 2 != static_cast<size_t>(I * N)) {
    throw std::runtime_error(
        "rasterize_to_pixels_3dgs dense path expects means2d size to equal I * N * 2.");
  }

  std::vector<float> render_colors(
      static_cast<size_t>(I * image_height * image_width * channels), 0.0f);
  std::vector<float> render_alphas(
      static_cast<size_t>(I * image_height * image_width), 0.0f);
  std::vector<int32_t> last_ids(
      static_cast<size_t>(I * image_height * image_width), 0);

  const float* means_data = means2d.data<float>();
  const float* conics_data = conics.data<float>();
  const float* colors_data = colors.data<float>();
  const float* opacities_data = opacities.data<float>();
  const float* backgrounds_data =
      input.params.use_backgrounds ? backgrounds.data<float>() : nullptr;
  const int32_t* tile_offsets_data = tile_offsets.data<int32_t>();
  const int32_t* flatten_ids_data = flatten_ids.data<int32_t>();

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

        for (int local_y = 0; local_y < tile_size; ++local_y) {
          const int y = tile_y * tile_size + local_y;
          if (y >= image_height) {
            continue;
          }
          for (int local_x = 0; local_x < tile_size; ++local_x) {
            const int x = tile_x * tile_size + local_x;
            if (x >= image_width) {
              continue;
            }

            const int pix_id = image_id * image_height * image_width +
                               y * image_width + x;
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
                render_colors[static_cast<size_t>(
                    pix_id * channels + channel)] +=
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

  std::vector<mx::array> outputs;
  outputs.reserve(3);
  outputs.push_back(mx::array(
      render_colors.begin(),
      render_colors_shape(tile_offsets, image_height, image_width, channels),
      mx::float32));
  outputs.push_back(mx::array(
      render_alphas.begin(),
      render_alphas_shape(tile_offsets, image_height, image_width),
      mx::float32));
  outputs.push_back(mx::array(
      last_ids.begin(),
      last_ids_shape(tile_offsets, image_height, image_width),
      mx::int32));
  return outputs;
}

}  // namespace gsplat_core
