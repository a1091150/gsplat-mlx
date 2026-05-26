#include "include/gsplat_intersect.h"

#include "include/helper.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mx = mlx::core;

namespace gsplat_core {
namespace {

struct IntersectTileCountKernelParams {
  uint32_t numel;
  uint32_t tile_size;
  uint32_t tile_width;
  uint32_t tile_height;
};

struct IntersectOffsetKernelParams {
  uint32_t n_isects;
  uint32_t n_offsets;
  uint32_t n_tiles;
  uint32_t tile_n_bits;
};

struct TileRect {
  int min_x;
  int min_y;
  int max_x;
  int max_y;
};

int floor_log2_plus_one(int value) {
  if (value <= 0) {
    throw std::runtime_error("intersect expects positive image/tile counts.");
  }
  return static_cast<int>(std::floor(std::log2(static_cast<float>(value)))) + 1;
}

TileRect aabb_tile_rect(float mean_x,
                        float mean_y,
                        int radius_x,
                        int radius_y,
                        int tile_size,
                        int tile_width,
                        int tile_height) {
  if (radius_x <= 0 || radius_y <= 0) {
    return {0, 0, 0, 0};
  }

  const float tile_radius_x =
      static_cast<float>(radius_x) / static_cast<float>(tile_size);
  const float tile_radius_y =
      static_cast<float>(radius_y) / static_cast<float>(tile_size);
  const float tile_x = mean_x / static_cast<float>(tile_size);
  const float tile_y = mean_y / static_cast<float>(tile_size);

  TileRect rect = {};
  rect.min_x = std::min(
      std::max(0, static_cast<int>(std::floor(tile_x - tile_radius_x))),
      tile_width);
  rect.min_y = std::min(
      std::max(0, static_cast<int>(std::floor(tile_y - tile_radius_y))),
      tile_height);
  rect.max_x = std::min(
      std::max(0, static_cast<int>(std::ceil(tile_x + tile_radius_x))),
      tile_width);
  rect.max_y = std::min(
      std::max(0, static_cast<int>(std::ceil(tile_y + tile_radius_y))),
      tile_height);
  return rect;
}

int64_t encode_isect_id(int image_id,
                        int tile_id,
                        float depth,
                        int tile_n_bits) {
  int32_t depth_i32 = 0;
  std::memcpy(&depth_i32, &depth, sizeof(float));
  const int64_t iid_enc =
      static_cast<int64_t>(image_id) << (32 + tile_n_bits);
  const int64_t tile_enc = static_cast<int64_t>(tile_id) << 32;
  return iid_enc | tile_enc |
         static_cast<int64_t>(static_cast<uint32_t>(depth_i32));
}

void validate_intersect_tile_input(const IntersectTileInput& input) {
  if (input.params.packed) {
    throw std::runtime_error(
        "intersect_tile packed path is not implemented yet.");
  }
  if (input.params.segmented) {
    throw std::runtime_error(
        "intersect_tile segmented sort is not implemented yet.");
  }
  if (input.params.use_conics || input.params.use_opacities) {
    throw std::runtime_error(
        "intersect_tile AccuTile conics/opacities path is not implemented yet.");
  }
  if (input.means2d.ndim() < 3 ||
      input.means2d.shape(static_cast<int>(input.means2d.ndim()) - 1) != 2) {
    throw std::runtime_error("means2d must have shape [..., N, 2].");
  }
  if (input.radii.ndim() != input.means2d.ndim() ||
      input.radii.shape(static_cast<int>(input.radii.ndim()) - 1) != 2) {
    throw std::runtime_error("radii must have shape [..., N, 2].");
  }
  if (input.depths.ndim() + 1 != input.means2d.ndim()) {
    throw std::runtime_error("depths must have shape [..., N].");
  }
  if (input.means2d.dtype().val() != mx::float32.val() ||
      input.radii.dtype().val() != mx::int32.val() ||
      input.depths.dtype().val() != mx::float32.val()) {
    throw std::runtime_error(
        "intersect_tile dense path expects float32 means2d/depths and int32 radii.");
  }
}

mx::Shape scalar_shape_from_depths(const mx::array& depths) {
  mx::Shape shape;
  shape.reserve(depths.ndim());
  for (int i = 0; i < static_cast<int>(depths.ndim()); ++i) {
    shape.push_back(depths.shape(i));
  }
  return shape;
}

}  // namespace

mx::array gsplat_intersect_tile_count(const IntersectTileInput& input) {
  validate_intersect_tile_input(input);

  const int ndim = static_cast<int>(input.means2d.ndim());
  const int n = input.means2d.shape(ndim - 2);
  const int numel = static_cast<int>(input.depths.size());
  const int I = input.params.I;
  if (I <= 0 || numel % I != 0 || numel != I * n) {
    throw std::runtime_error(
        "intersect_tile_count dense path expects depths size to equal I * N.");
  }

  auto prim = std::make_shared<GSPlatIntersectTileCount>(
      to_stream(input.s), input.params);
  std::vector<mx::array> inputs = {
      mx::contiguous(input.means2d),
      mx::contiguous(input.radii),
      mx::contiguous(input.depths),
  };
  return mx::array(scalar_shape_from_depths(input.depths), mx::int32, prim, inputs);
}

std::vector<mx::array> gsplat_intersect_tile(const IntersectTileInput& input) {
  validate_intersect_tile_input(input);

  mx::array means2d = mx::contiguous(input.means2d);
  mx::array radii = mx::contiguous(input.radii);
  mx::array depths = mx::contiguous(input.depths);
  mx::eval(means2d, radii, depths);

  const int ndim = static_cast<int>(means2d.ndim());
  const int n = means2d.shape(ndim - 2);
  const int numel = static_cast<int>(depths.size());
  const int I = input.params.I;
  if (I <= 0 || numel % I != 0 || numel != I * n) {
    throw std::runtime_error(
        "intersect_tile dense path expects depths size to equal I * N.");
  }

  const int n_tiles = input.params.tile_width * input.params.tile_height;
  const int tile_n_bits = floor_log2_plus_one(n_tiles);
  const int image_n_bits = floor_log2_plus_one(I);
  if (image_n_bits + tile_n_bits > 32) {
    throw std::runtime_error(
        "intersect_tile image id and tile id require more than 32 bits.");
  }

  const float* means_data = means2d.data<float>();
  const int32_t* radii_data = radii.data<int32_t>();
  const float* depths_data = depths.data<float>();

  std::vector<int32_t> tiles_per_gauss(static_cast<size_t>(numel), 0);
  std::vector<int64_t> isect_ids;
  std::vector<int32_t> flatten_ids;

  for (int idx = 0; idx < numel; ++idx) {
    const int image_id = idx / n;
    const TileRect rect = aabb_tile_rect(
        means_data[idx * 2],
        means_data[idx * 2 + 1],
        radii_data[idx * 2],
        radii_data[idx * 2 + 1],
        input.params.tile_size,
        input.params.tile_width,
        input.params.tile_height);
    const int count = (rect.max_y - rect.min_y) * (rect.max_x - rect.min_x);
    tiles_per_gauss[static_cast<size_t>(idx)] = static_cast<int32_t>(count);

    for (int tile_y = rect.min_y; tile_y < rect.max_y; ++tile_y) {
      for (int tile_x = rect.min_x; tile_x < rect.max_x; ++tile_x) {
        const int tile_id = tile_y * input.params.tile_width + tile_x;
        isect_ids.push_back(encode_isect_id(
            image_id, tile_id, depths_data[idx], tile_n_bits));
        flatten_ids.push_back(static_cast<int32_t>(idx));
      }
    }
  }

  if (input.params.sort) {
    std::vector<size_t> order(isect_ids.size());
    for (size_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return isect_ids[a] < isect_ids[b];
    });

    std::vector<int64_t> sorted_ids(isect_ids.size());
    std::vector<int32_t> sorted_flatten_ids(flatten_ids.size());
    for (size_t i = 0; i < order.size(); ++i) {
      sorted_ids[i] = isect_ids[order[i]];
      sorted_flatten_ids[i] = flatten_ids[order[i]];
    }
    isect_ids = std::move(sorted_ids);
    flatten_ids = std::move(sorted_flatten_ids);
  }

  std::vector<mx::array> outputs;
  outputs.reserve(3);
  outputs.push_back(mx::array(
      tiles_per_gauss.begin(), scalar_shape_from_depths(depths), mx::int32));
  outputs.push_back(mx::array(
      isect_ids.begin(),
      mx::Shape{static_cast<int>(isect_ids.size())},
      mx::int64));
  outputs.push_back(mx::array(
      flatten_ids.begin(),
      mx::Shape{static_cast<int>(flatten_ids.size())},
      mx::int32));
  return outputs;
}

void GSPlatIntersectTileCount::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& means2d = inputs[0];
  const auto& radii = inputs[1];
  const auto& depths = inputs[2];
  mx::eval(means2d, radii, depths);

  auto& tiles_per_gauss = outputs[0];
  tiles_per_gauss.set_data(mx::allocator::malloc(tiles_per_gauss.nbytes()));
  std::memset(tiles_per_gauss.data<void>(), 0, tiles_per_gauss.nbytes());

  const int numel = static_cast<int>(depths.size());
  const float* means_data = means2d.data<float>();
  const int32_t* radii_data = radii.data<int32_t>();
  int32_t* out_data = tiles_per_gauss.data<int32_t>();

  for (int idx = 0; idx < numel; ++idx) {
    const TileRect rect = aabb_tile_rect(
        means_data[idx * 2],
        means_data[idx * 2 + 1],
        radii_data[idx * 2],
        radii_data[idx * 2 + 1],
        params_.tile_size,
        params_.tile_width,
        params_.tile_height);
    out_data[idx] =
        static_cast<int32_t>((rect.max_y - rect.min_y) *
                             (rect.max_x - rect.min_x));
  }
}

#ifdef _METAL_
void GSPlatIntersectTileCount::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& means2d = inputs[0];
  const auto& radii = inputs[1];
  const auto& depths = inputs[2];

  auto& tiles_per_gauss = outputs[0];
  tiles_per_gauss.set_data(mx::allocator::malloc(tiles_per_gauss.nbytes()));
  std::memset(tiles_per_gauss.data<void>(), 0, tiles_per_gauss.nbytes());

  const uint32_t numel = static_cast<uint32_t>(depths.size());
  if (numel == 0) {
    return;
  }

  IntersectTileCountKernelParams kernel_params = {
      .numel = numel,
      .tile_size = static_cast<uint32_t>(params_.tile_size),
      .tile_width = static_cast<uint32_t>(params_.tile_width),
      .tile_height = static_cast<uint32_t>(params_.tile_height),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("gsplat_core", current_binary_dir());
  auto kernel = d.get_kernel("gsplat_intersect_tile_count_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(means2d, 1);
  compute_encoder.set_input_array(radii, 2);
  compute_encoder.set_output_array(tiles_per_gauss, 3);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  const size_t tgp_size = std::min(static_cast<size_t>(numel), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(numel, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}
#else
void GSPlatIntersectTileCount::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatIntersectTileCount has no GPU implementation.");
}
#endif

std::vector<mx::array> GSPlatIntersectTileCount::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("GSPlatIntersectTileCount jvp is not implemented.");
}

std::vector<mx::array> GSPlatIntersectTileCount::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("GSPlatIntersectTileCount vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GSPlatIntersectTileCount::vmap(const std::vector<mx::array>&,
                               const std::vector<int>&) {
  throw std::runtime_error("GSPlatIntersectTileCount vmap is not implemented.");
}

bool GSPlatIntersectTileCount::is_equivalent(
    const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  const auto* other_ptr = dynamic_cast<const GSPlatIntersectTileCount*>(&other);
  if (!other_ptr) {
    return false;
  }
  return params_.I == other_ptr->params_.I &&
         params_.tile_size == other_ptr->params_.tile_size &&
         params_.tile_width == other_ptr->params_.tile_width &&
         params_.tile_height == other_ptr->params_.tile_height &&
         params_.sort == other_ptr->params_.sort &&
         params_.segmented == other_ptr->params_.segmented &&
         params_.packed == other_ptr->params_.packed &&
         params_.use_conics == other_ptr->params_.use_conics &&
         params_.use_opacities == other_ptr->params_.use_opacities;
}

mx::array gsplat_intersect_offset(const mx::array& isect_ids,
                                  int I,
                                  int tile_width,
                                  int tile_height,
                                  mx::StreamOrDevice s) {
  if (I <= 0 || tile_width <= 0 || tile_height <= 0) {
    throw std::runtime_error(
        "intersect_offset expects positive image and tile dimensions.");
  }
  if (isect_ids.dtype().val() != mx::int64.val()) {
    throw std::runtime_error("intersect_offset expects int64 isect_ids.");
  }

  auto prim = std::make_shared<GSPlatIntersectOffset>(
      to_stream(s), IntersectOffsetParams{I, tile_width, tile_height});
  std::vector<mx::array> inputs = {mx::contiguous(isect_ids)};
  return mx::array(mx::Shape{I, tile_height, tile_width}, mx::int32, prim, inputs);
}

void GSPlatIntersectOffset::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& ids = inputs[0];
  mx::eval(ids);

  auto& offsets = outputs[0];
  offsets.set_data(mx::allocator::malloc(offsets.nbytes()));
  std::memset(offsets.data<void>(), 0, offsets.nbytes());

  const int n_isects = static_cast<int>(ids.size());
  const int n_tiles = params_.tile_width * params_.tile_height;
  const int tile_n_bits = floor_log2_plus_one(n_tiles);
  const int n_offsets = params_.I * n_tiles;
  int32_t* offsets_data = offsets.data<int32_t>();
  const int64_t* id_data = ids.data<int64_t>();

  for (int out_idx = 0; out_idx < n_offsets; ++out_idx) {
    const int image_id = out_idx / n_tiles;
    const int tile_id = out_idx % n_tiles;
    const int64_t key =
        (static_cast<int64_t>(image_id) << tile_n_bits) | tile_id;

    int lo = 0;
    int hi = n_isects;
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2;
      const int64_t encoded = id_data[mid] >> 32;
      if (encoded < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    offsets_data[out_idx] = lo;
  }
}

#ifdef _METAL_
void GSPlatIntersectOffset::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& ids = inputs[0];
  auto& offsets = outputs[0];
  offsets.set_data(mx::allocator::malloc(offsets.nbytes()));
  std::memset(offsets.data<void>(), 0, offsets.nbytes());

  const int n_tiles = params_.tile_width * params_.tile_height;
  const uint32_t n_offsets = static_cast<uint32_t>(params_.I * n_tiles);
  if (n_offsets == 0) {
    return;
  }

  IntersectOffsetKernelParams kernel_params = {
      .n_isects = static_cast<uint32_t>(ids.size()),
      .n_offsets = n_offsets,
      .n_tiles = static_cast<uint32_t>(n_tiles),
      .tile_n_bits = static_cast<uint32_t>(floor_log2_plus_one(n_tiles)),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("gsplat_core", current_binary_dir());
  auto kernel = d.get_kernel("gsplat_intersect_offset_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(ids, 1);
  compute_encoder.set_output_array(offsets, 2);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  const size_t tgp_size = std::min(static_cast<size_t>(n_offsets), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(n_offsets, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}
#else
void GSPlatIntersectOffset::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("GSPlatIntersectOffset has no GPU implementation.");
}
#endif

std::vector<mx::array> GSPlatIntersectOffset::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("GSPlatIntersectOffset jvp is not implemented.");
}

std::vector<mx::array> GSPlatIntersectOffset::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("GSPlatIntersectOffset vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GSPlatIntersectOffset::vmap(const std::vector<mx::array>&,
                            const std::vector<int>&) {
  throw std::runtime_error("GSPlatIntersectOffset vmap is not implemented.");
}

bool GSPlatIntersectOffset::is_equivalent(const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  const auto* other_ptr = dynamic_cast<const GSPlatIntersectOffset*>(&other);
  if (!other_ptr) {
    return false;
  }
  return params_.I == other_ptr->params_.I &&
         params_.tile_width == other_ptr->params_.tile_width &&
         params_.tile_height == other_ptr->params_.tile_height;
}

}  // namespace gsplat_core
