#include <metal_stdlib>

using namespace metal;

constant float ALPHA_THRESHOLD = 1.0f / 255.0f;
constant float GAUSSIAN_EXTEND = 3.33f;
constant float MIN_COMPENSATION = 0.005f;

struct ProjectionKernelParams {
  uint B;
  uint C;
  uint N;
  uint image_width;
  uint image_height;
  float eps2d;
  float near_plane;
  float far_plane;
  float radius_clip;
  uint calc_compensations;
  uint camera_model;
  uint use_covars;
  uint use_opacities;
};

inline float3 read_float3(const device float* data, uint idx) {
  return float3(data[idx * 3], data[idx * 3 + 1], data[idx * 3 + 2]);
}

inline float4 read_float4(const device float* data, uint idx) {
  return float4(data[idx * 4], data[idx * 4 + 1], data[idx * 4 + 2], data[idx * 4 + 3]);
}

inline float3x3 quat_to_rotmat(float4 quat) {
  float w = quat.x;
  float x = quat.y;
  float y = quat.z;
  float z = quat.w;
  float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
  x *= inv_norm;
  y *= inv_norm;
  z *= inv_norm;
  w *= inv_norm;

  float x2 = x * x;
  float y2 = y * y;
  float z2 = z * z;
  float xy = x * y;
  float xz = x * z;
  float yz = y * z;
  float wx = w * x;
  float wy = w * y;
  float wz = w * z;

  return float3x3(
      float3(1.0f - 2.0f * (y2 + z2), 2.0f * (xy + wz), 2.0f * (xz - wy)),
      float3(2.0f * (xy - wz), 1.0f - 2.0f * (x2 + z2), 2.0f * (yz + wx)),
      float3(2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (x2 + y2)));
}

inline float3x3 covar_from_quat_scale(float4 quat, float3 scale) {
  float3x3 r = quat_to_rotmat(quat);
  float3x3 s = float3x3(
      float3(scale.x, 0.0f, 0.0f),
      float3(0.0f, scale.y, 0.0f),
      float3(0.0f, 0.0f, scale.z));
  float3x3 m = r * s;
  return m * transpose(m);
}

inline void persp_proj(float3 mean3d,
                       float3x3 cov3d,
                       float fx,
                       float fy,
                       float cx,
                       float cy,
                       uint width,
                       uint height,
                       thread float2x2& cov2d,
                       thread float2& mean2d) {
  float x = mean3d.x;
  float y = mean3d.y;
  float z = mean3d.z;

  float tan_fovx = 0.5f * float(width) / fx;
  float tan_fovy = 0.5f * float(height) / fy;
  float lim_x_pos = (float(width) - cx) / fx + 0.3f * tan_fovx;
  float lim_x_neg = cx / fx + 0.3f * tan_fovx;
  float lim_y_pos = (float(height) - cy) / fy + 0.3f * tan_fovy;
  float lim_y_neg = cy / fy + 0.3f * tan_fovy;

  float rz = 1.0f / z;
  float rz2 = rz * rz;
  float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
  float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

  float3x2 j = float3x2(
      float2(fx * rz, 0.0f),
      float2(0.0f, fy * rz),
      float2(-fx * tx * rz2, -fy * ty * rz2));
  cov2d = j * cov3d * transpose(j);
  mean2d = float2(fx * x * rz + cx, fy * y * rz + cy);
}

inline void ortho_proj(float3 mean3d,
                       float3x3 cov3d,
                       float fx,
                       float fy,
                       float cx,
                       float cy,
                       thread float2x2& cov2d,
                       thread float2& mean2d) {
  float3x2 j = float3x2(
      float2(fx, 0.0f),
      float2(0.0f, fy),
      float2(0.0f, 0.0f));
  cov2d = j * cov3d * transpose(j);
  mean2d = float2(fx * mean3d.x + cx, fy * mean3d.y + cy);
}

inline void fisheye_proj(float3 mean3d,
                         float3x3 cov3d,
                         float fx,
                         float fy,
                         float cx,
                         float cy,
                         thread float2x2& cov2d,
                         thread float2& mean2d) {
  float x = mean3d.x;
  float y = mean3d.y;
  float z = mean3d.z;
  float eps = 0.0000001f;
  float xy_len = length(float2(x, y)) + eps;
  float theta = atan2(xy_len, z + eps);
  mean2d = float2(x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy);

  float x2 = x * x + eps;
  float y2 = y * y;
  float xy = x * y;
  float x2y2 = x2 + y2;
  float x2y2z2_inv = 1.0f / (x2y2 + z * z);
  float b = atan2(xy_len, z) / xy_len / x2y2;
  float a = z * x2y2z2_inv / x2y2;

  float3x2 j = float3x2(
      float2(fx * (x2 * a + y2 * b), fy * xy * (a - b)),
      float2(fx * xy * (a - b), fy * (y2 * a + x2 * b)),
      float2(-fx * x * x2y2z2_inv, -fy * y * x2y2z2_inv));
  cov2d = j * cov3d * transpose(j);
}

inline float add_blur(float eps2d,
                      thread float2x2& covar,
                      thread float& compensation) {
  float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
  covar[0][0] += eps2d;
  covar[1][1] += eps2d;
  float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
  compensation = sqrt(max(MIN_COMPENSATION * MIN_COMPENSATION, det_orig / det_blur));
  return det_blur;
}

kernel void gsplat_projection_ewa_3dgs_fused_forward_kernel(
    constant ProjectionKernelParams& params [[buffer(0)]],
    const device float* means [[buffer(1)]],
    const device float* covars [[buffer(2)]],
    const device float* quats [[buffer(3)]],
    const device float* scales [[buffer(4)]],
    const device float* opacities [[buffer(5)]],
    const device float* viewmats [[buffer(6)]],
    const device float* Ks [[buffer(7)]],
    device int* radii [[buffer(8)]],
    device float* means2d [[buffer(9)]],
    device float* depths [[buffer(10)]],
    device float* conics [[buffer(11)]],
    device float* compensations [[buffer(12)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= params.B * params.C * params.N) {
    return;
  }

  uint bid = idx / (params.C * params.N);
  uint cid = (idx / params.N) % params.C;
  uint gid = idx % params.N;

  uint mean_off = bid * params.N * 3 + gid * 3;
  uint view_off = bid * params.C * 16 + cid * 16;
  uint k_off = bid * params.C * 9 + cid * 9;

  float3 mean_w = float3(means[mean_off], means[mean_off + 1], means[mean_off + 2]);
  float3x3 R = float3x3(
      float3(viewmats[view_off + 0], viewmats[view_off + 4], viewmats[view_off + 8]),
      float3(viewmats[view_off + 1], viewmats[view_off + 5], viewmats[view_off + 9]),
      float3(viewmats[view_off + 2], viewmats[view_off + 6], viewmats[view_off + 10]));
  float3 t = float3(viewmats[view_off + 3], viewmats[view_off + 7], viewmats[view_off + 11]);
  float3 mean_c = R * mean_w + t;

  if (mean_c.z < params.near_plane || mean_c.z > params.far_plane) {
    radii[idx * 2] = 0;
    radii[idx * 2 + 1] = 0;
    return;
  }

  float3x3 covar_w;
  if (params.use_covars != 0) {
    uint cov_off = bid * params.N * 6 + gid * 6;
    covar_w = float3x3(
        float3(covars[cov_off + 0], covars[cov_off + 1], covars[cov_off + 2]),
        float3(covars[cov_off + 1], covars[cov_off + 3], covars[cov_off + 4]),
        float3(covars[cov_off + 2], covars[cov_off + 4], covars[cov_off + 5]));
  } else {
    float4 quat = read_float4(quats, bid * params.N + gid);
    float3 scale = read_float3(scales, bid * params.N + gid);
    covar_w = covar_from_quat_scale(quat, scale);
  }
  float3x3 covar_c = R * covar_w * transpose(R);

  float fx = Ks[k_off + 0];
  float fy = Ks[k_off + 4];
  float cx = Ks[k_off + 2];
  float cy = Ks[k_off + 5];
  float2x2 covar2d;
  float2 mean2d_val;

  if (params.camera_model == 1) {
    ortho_proj(mean_c, covar_c, fx, fy, cx, cy, covar2d, mean2d_val);
  } else if (params.camera_model == 2) {
    fisheye_proj(mean_c, covar_c, fx, fy, cx, cy, covar2d, mean2d_val);
  } else {
    persp_proj(mean_c, covar_c, fx, fy, cx, cy, params.image_width, params.image_height,
               covar2d, mean2d_val);
  }

  float compensation = 0.0f;
  float det = add_blur(params.eps2d, covar2d, compensation);
  if (det <= 0.0f) {
    radii[idx * 2] = 0;
    radii[idx * 2 + 1] = 0;
    return;
  }

  float inv_det = 1.0f / det;
  float2x2 covar2d_inv = float2x2(
      float2(covar2d[1][1] * inv_det, -covar2d[0][1] * inv_det),
      float2(-covar2d[1][0] * inv_det, covar2d[0][0] * inv_det));

  float extend = GAUSSIAN_EXTEND;
  if (params.use_opacities != 0) {
    float opacity = opacities[bid * params.N + gid];
    if (params.calc_compensations != 0) {
      opacity *= compensation;
    }
    if (opacity < ALPHA_THRESHOLD) {
      radii[idx * 2] = 0;
      radii[idx * 2 + 1] = 0;
      return;
    }
    extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * log(opacity / ALPHA_THRESHOLD)));
  }

  float radius_x = ceil(extend * sqrt(covar2d[0][0]));
  float radius_y = ceil(extend * sqrt(covar2d[1][1]));
  if (radius_x <= params.radius_clip && radius_y <= params.radius_clip) {
    radii[idx * 2] = 0;
    radii[idx * 2 + 1] = 0;
    return;
  }
  if (mean2d_val.x + radius_x <= 0.0f ||
      mean2d_val.x - radius_x >= float(params.image_width) ||
      mean2d_val.y + radius_y <= 0.0f ||
      mean2d_val.y - radius_y >= float(params.image_height)) {
    radii[idx * 2] = 0;
    radii[idx * 2 + 1] = 0;
    return;
  }

  radii[idx * 2] = int(radius_x);
  radii[idx * 2 + 1] = int(radius_y);
  means2d[idx * 2] = mean2d_val.x;
  means2d[idx * 2 + 1] = mean2d_val.y;
  depths[idx] = mean_c.z;
  conics[idx * 3] = covar2d_inv[0][0];
  conics[idx * 3 + 1] = covar2d_inv[0][1];
  conics[idx * 3 + 2] = covar2d_inv[1][1];
  if (params.calc_compensations != 0) {
    compensations[idx] = compensation;
  }
}
