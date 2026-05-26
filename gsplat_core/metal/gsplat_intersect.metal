#include <metal_stdlib>

using namespace metal;

struct IntersectTileCountKernelParams {
  uint numel;
  uint tile_size;
  uint tile_width;
  uint tile_height;
};

struct IntersectOffsetKernelParams {
  uint n_isects;
  uint n_offsets;
  uint n_tiles;
  uint tile_n_bits;
};

struct IntersectTileEncodeKernelParams {
  uint numel;
  uint n_per_image;
  uint tile_size;
  uint tile_width;
  uint tile_height;
  uint n_tiles;
  uint tile_n_bits;
};

inline int clamp_tile(int value, uint upper) {
  return min(max(0, value), int(upper));
}

kernel void gsplat_intersect_tile_count_kernel(
    constant IntersectTileCountKernelParams& params [[buffer(0)]],
    const device float* means2d [[buffer(1)]],
    const device int* radii [[buffer(2)]],
    device int* tiles_per_gauss [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= params.numel) {
    return;
  }

  int radius_x = radii[idx * 2];
  int radius_y = radii[idx * 2 + 1];
  if (radius_x <= 0 || radius_y <= 0) {
    tiles_per_gauss[idx] = 0;
    return;
  }

  float tile_size = float(params.tile_size);
  float tile_radius_x = float(radius_x) / tile_size;
  float tile_radius_y = float(radius_y) / tile_size;
  float tile_x = means2d[idx * 2] / tile_size;
  float tile_y = means2d[idx * 2 + 1] / tile_size;

  int min_x = clamp_tile(int(floor(tile_x - tile_radius_x)), params.tile_width);
  int min_y = clamp_tile(int(floor(tile_y - tile_radius_y)), params.tile_height);
  int max_x = clamp_tile(int(ceil(tile_x + tile_radius_x)), params.tile_width);
  int max_y = clamp_tile(int(ceil(tile_y + tile_radius_y)), params.tile_height);

  tiles_per_gauss[idx] = (max_y - min_y) * (max_x - min_x);
}

kernel void gsplat_intersect_tile_encode_kernel(
    constant IntersectTileEncodeKernelParams& params [[buffer(0)]],
    const device float* means2d [[buffer(1)]],
    const device int* radii [[buffer(2)]],
    const device float* depths [[buffer(3)]],
    const device int* tile_offsets [[buffer(4)]],
    device long* isect_ids [[buffer(5)]],
    device int* flatten_ids [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= params.numel) {
    return;
  }

  int radius_x = radii[idx * 2];
  int radius_y = radii[idx * 2 + 1];
  if (radius_x <= 0 || radius_y <= 0) {
    return;
  }

  float tile_size = float(params.tile_size);
  float tile_radius_x = float(radius_x) / tile_size;
  float tile_radius_y = float(radius_y) / tile_size;
  float tile_x = means2d[idx * 2] / tile_size;
  float tile_y = means2d[idx * 2 + 1] / tile_size;

  int min_x = clamp_tile(int(floor(tile_x - tile_radius_x)), params.tile_width);
  int min_y = clamp_tile(int(floor(tile_y - tile_radius_y)), params.tile_height);
  int max_x = clamp_tile(int(ceil(tile_x + tile_radius_x)), params.tile_width);
  int max_y = clamp_tile(int(ceil(tile_y + tile_radius_y)), params.tile_height);

  uint image_id = idx / params.n_per_image;
  uint depth_bits = as_type<uint>(depths[idx]);
  uint out_idx = uint(tile_offsets[idx]);
  for (int tile_yi = min_y; tile_yi < max_y; ++tile_yi) {
    for (int tile_xi = min_x; tile_xi < max_x; ++tile_xi) {
      uint tile_id = uint(tile_yi) * params.tile_width + uint(tile_xi);
      long encoded = (long(image_id) << (32 + params.tile_n_bits)) |
          (long(tile_id) << 32) |
          long(depth_bits);
      isect_ids[out_idx] = encoded;
      flatten_ids[out_idx] = int(idx);
      ++out_idx;
    }
  }
}

kernel void gsplat_intersect_offset_kernel(
    constant IntersectOffsetKernelParams& params [[buffer(0)]],
    const device long* isect_ids [[buffer(1)]],
    device int* offsets [[buffer(2)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= params.n_offsets) {
    return;
  }

  uint image_id = idx / params.n_tiles;
  uint tile_id = idx % params.n_tiles;
  long key = (long(image_id) << params.tile_n_bits) | long(tile_id);

  uint lo = 0;
  uint hi = params.n_isects;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    long encoded = isect_ids[mid] >> 32;
    if (encoded < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  offsets[idx] = int(lo);
}
