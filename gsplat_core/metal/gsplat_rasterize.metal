#include <metal_stdlib>

using namespace metal;

constant float kAlphaThreshold = 1.0f / 255.0f;
constant float kMaxAlpha = 0.99f;
constant float kTransmittanceThreshold = 1.0e-4f;
constant float kMinOneMinusAlpha = 1.0e-6f;
constant uint kMaxBackwardChannels = 64;

struct RasterizeToPixels3DGSKernelParams {
  uint I;
  uint N;
  uint channels;
  uint image_width;
  uint image_height;
  uint tile_size;
  uint tile_width;
  uint tile_height;
  uint n_isects;
  uint use_backgrounds;
  uint use_masks;
};

static inline void atomic_add_float(device float* ptr, float value) {
  device atomic_float* atomic_ptr =
      reinterpret_cast<device atomic_float*>(ptr);
  atomic_fetch_add_explicit(atomic_ptr, value, memory_order_relaxed);
}

kernel void gsplat_rasterize_to_pixels_3dgs_forward_kernel(
    constant RasterizeToPixels3DGSKernelParams& params [[buffer(0)]],
    const device float* means2d [[buffer(1)]],
    const device float* conics [[buffer(2)]],
    const device float* colors [[buffer(3)]],
    const device float* opacities [[buffer(4)]],
    const device float* backgrounds [[buffer(5)]],
    const device bool* masks [[buffer(6)]],
    const device int* tile_offsets [[buffer(7)]],
    const device int* flatten_ids [[buffer(8)]],
    device float* render_colors [[buffer(9)]],
    device float* render_alphas [[buffer(10)]],
    device int* last_ids [[buffer(11)]],
    uint pix_id [[thread_position_in_grid]]) {
  uint pixels_per_image = params.image_width * params.image_height;
  uint total_pixels = params.I * pixels_per_image;
  if (pix_id >= total_pixels) {
    return;
  }

  uint image_id = pix_id / pixels_per_image;
  uint pixel_in_image = pix_id % pixels_per_image;
  uint y = pixel_in_image / params.image_width;
  uint x = pixel_in_image % params.image_width;
  uint tile_x = x / params.tile_size;
  uint tile_y = y / params.tile_size;
  uint tile_id = tile_y * params.tile_width + tile_x;
  uint n_tiles = params.tile_width * params.tile_height;
  uint offset_index = image_id * n_tiles + tile_id;

  if (params.use_masks != 0 && !masks[offset_index]) {
    render_alphas[pix_id] = 0.0f;
    for (uint channel = 0; channel < params.channels; ++channel) {
      render_colors[pix_id * params.channels + channel] =
          params.use_backgrounds == 0
              ? 0.0f
              : backgrounds[image_id * params.channels + channel];
    }
    last_ids[pix_id] = 0;
    return;
  }

  int range_start = tile_offsets[offset_index];
  int range_end = (offset_index + 1 == params.I * n_tiles)
      ? int(params.n_isects)
      : tile_offsets[offset_index + 1];

  float px = float(x) + 0.5f;
  float py = float(y) + 0.5f;
  float T = 1.0f;
  int cur_idx = 0;

  for (int idx = range_start; idx < range_end; ++idx) {
    int g = flatten_ids[idx];
    float mean_x = means2d[g * 2];
    float mean_y = means2d[g * 2 + 1];
    float dx = mean_x - px;
    float dy = mean_y - py;
    float c0 = conics[g * 3];
    float c1 = conics[g * 3 + 1];
    float c2 = conics[g * 3 + 2];
    float sigma = 0.5f * (c0 * dx * dx + c2 * dy * dy) + c1 * dx * dy;
    float alpha = min(kMaxAlpha, opacities[g] * exp(-sigma));
    if (sigma < 0.0f || alpha < kAlphaThreshold) {
      continue;
    }

    float next_T = T * (1.0f - alpha);
    if (next_T <= kTransmittanceThreshold) {
      break;
    }

    float visibility = alpha * T;
    for (uint channel = 0; channel < params.channels; ++channel) {
      render_colors[pix_id * params.channels + channel] +=
          colors[g * params.channels + channel] * visibility;
    }
    cur_idx = idx;
    T = next_T;
  }

  render_alphas[pix_id] = 1.0f - T;
  for (uint channel = 0; channel < params.channels; ++channel) {
    float background = params.use_backgrounds == 0
        ? 0.0f
        : backgrounds[image_id * params.channels + channel];
    render_colors[pix_id * params.channels + channel] += T * background;
  }
  last_ids[pix_id] = cur_idx;
}

kernel void gsplat_rasterize_to_pixels_3dgs_backward_kernel(
    constant RasterizeToPixels3DGSKernelParams& params [[buffer(0)]],
    constant uint& absgrad [[buffer(1)]],
    const device float* means2d [[buffer(2)]],
    const device float* conics [[buffer(3)]],
    const device float* colors [[buffer(4)]],
    const device float* opacities [[buffer(5)]],
    const device float* backgrounds [[buffer(6)]],
    const device bool* masks [[buffer(7)]],
    const device int* tile_offsets [[buffer(8)]],
    const device int* flatten_ids [[buffer(9)]],
    const device float* render_alphas [[buffer(10)]],
    const device int* last_ids [[buffer(11)]],
    const device float* v_render_colors [[buffer(12)]],
    const device float* v_render_alphas [[buffer(13)]],
    device float* v_means2d_abs [[buffer(14)]],
    device float* v_means2d [[buffer(15)]],
    device float* v_conics [[buffer(16)]],
    device float* v_colors [[buffer(17)]],
    device float* v_opacities [[buffer(18)]],
    device float* v_backgrounds [[buffer(19)]],
    uint pix_id [[thread_position_in_grid]]) {
  uint pixels_per_image = params.image_width * params.image_height;
  uint total_pixels = params.I * pixels_per_image;
  if (pix_id >= total_pixels || params.channels > kMaxBackwardChannels) {
    return;
  }

  uint image_id = pix_id / pixels_per_image;
  uint pixel_in_image = pix_id % pixels_per_image;
  uint y = pixel_in_image / params.image_width;
  uint x = pixel_in_image % params.image_width;
  uint tile_x = x / params.tile_size;
  uint tile_y = y / params.tile_size;
  uint tile_id = tile_y * params.tile_width + tile_x;
  uint n_tiles = params.tile_width * params.tile_height;
  uint offset_index = image_id * n_tiles + tile_id;

  if (params.use_masks != 0 && !masks[offset_index]) {
    return;
  }

  int range_start = tile_offsets[offset_index];
  int range_end = (offset_index + 1 == params.I * n_tiles)
      ? int(params.n_isects)
      : tile_offsets[offset_index + 1];
  int bin_final = last_ids[pix_id];

  float px = float(x) + 0.5f;
  float py = float(y) + 0.5f;
  float T_final = 1.0f - render_alphas[pix_id];
  float T = T_final;
  float buffer[kMaxBackwardChannels];
  for (uint channel = 0; channel < params.channels; ++channel) {
    buffer[channel] = 0.0f;
  }
  float v_render_a = v_render_alphas[pix_id];
  const device float* v_render_c =
      v_render_colors + pix_id * params.channels;

  if (params.use_backgrounds != 0) {
    for (uint channel = 0; channel < params.channels; ++channel) {
      atomic_add_float(
          v_backgrounds + image_id * params.channels + channel,
          T_final * v_render_c[channel]);
    }
  }

  int last = min(range_end - 1, bin_final);
  for (int idx = last; idx >= range_start; --idx) {
    int g = flatten_ids[idx];
    float dx = means2d[g * 2] - px;
    float dy = means2d[g * 2 + 1] - py;
    float c0 = conics[g * 3];
    float c1 = conics[g * 3 + 1];
    float c2 = conics[g * 3 + 2];
    float sigma = 0.5f * (c0 * dx * dx + c2 * dy * dy) + c1 * dx * dy;
    float vis = exp(-sigma);
    float alpha = min(kMaxAlpha, opacities[g] * vis);
    if (sigma < 0.0f || alpha < kAlphaThreshold) {
      continue;
    }

    float one_minus_alpha = max(kMinOneMinusAlpha, 1.0f - alpha);
    float ra = 1.0f / one_minus_alpha;
    T *= ra;
    float fac = alpha * T;

    for (uint channel = 0; channel < params.channels; ++channel) {
      atomic_add_float(
          v_colors + g * params.channels + channel,
          fac * v_render_c[channel]);
    }

    float v_alpha = 0.0f;
    for (uint channel = 0; channel < params.channels; ++channel) {
      float color = colors[g * params.channels + channel];
      v_alpha += (color * T - buffer[channel] * ra) * v_render_c[channel];
    }
    v_alpha += T_final * ra * v_render_a;
    if (params.use_backgrounds != 0) {
      float accum = 0.0f;
      for (uint channel = 0; channel < params.channels; ++channel) {
        accum += backgrounds[image_id * params.channels + channel] *
                 v_render_c[channel];
      }
      v_alpha += -T_final * ra * accum;
    }

    if (opacities[g] * vis <= kMaxAlpha) {
      float v_sigma = -opacities[g] * vis * v_alpha;
      float v_x = v_sigma * (c0 * dx + c1 * dy);
      float v_y = v_sigma * (c1 * dx + c2 * dy);
      atomic_add_float(v_conics + g * 3, 0.5f * v_sigma * dx * dx);
      atomic_add_float(v_conics + g * 3 + 1, v_sigma * dx * dy);
      atomic_add_float(v_conics + g * 3 + 2, 0.5f * v_sigma * dy * dy);
      atomic_add_float(v_means2d + g * 2, v_x);
      atomic_add_float(v_means2d + g * 2 + 1, v_y);
      if (absgrad != 0) {
        atomic_add_float(v_means2d_abs + g * 2, abs(v_x));
        atomic_add_float(v_means2d_abs + g * 2 + 1, abs(v_y));
      }
      atomic_add_float(v_opacities + g, vis * v_alpha);
    }

    for (uint channel = 0; channel < params.channels; ++channel) {
      buffer[channel] += colors[g * params.channels + channel] * fac;
    }
  }
}
