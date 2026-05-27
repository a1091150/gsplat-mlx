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

inline float4 quat_to_rotmat_vjp(float4 quat, float3x3 v_R) {
  float w = quat.x;
  float x = quat.y;
  float y = quat.z;
  float z = quat.w;
  float inv_norm = rsqrt(w * w + x * x + y * y + z * z);
  w *= inv_norm;
  x *= inv_norm;
  y *= inv_norm;
  z *= inv_norm;

  float g[9] = {
      v_R[0][0], v_R[1][0], v_R[2][0],
      v_R[0][1], v_R[1][1], v_R[2][1],
      v_R[0][2], v_R[1][2], v_R[2][2],
  };
  float4 v_quat_n = float4(
      2.0f * (x * (g[7] - g[5]) + y * (g[2] - g[6]) +
              z * (g[3] - g[1])),
      2.0f * (-2.0f * x * (g[4] + g[8]) + y * (g[1] + g[3]) +
              z * (g[2] + g[6]) + w * (g[7] - g[5])),
      2.0f * (x * (g[1] + g[3]) - 2.0f * y * (g[0] + g[8]) +
              z * (g[5] + g[7]) + w * (g[2] - g[6])),
      2.0f * (x * (g[2] + g[6]) + y * (g[5] + g[7]) -
              2.0f * z * (g[0] + g[4]) + w * (g[3] - g[1])));
  float4 quat_n = float4(w, x, y, z);
  float dot_vq_q = dot(v_quat_n, quat_n);
  return (v_quat_n - dot_vq_q * quat_n) * inv_norm;
}

inline void quat_scale_to_covar_vjp(float4 quat,
                                    float3 scale,
                                    float3x3 v_covar,
                                    thread float4& v_quat,
                                    thread float3& v_scale) {
  float3x3 R = quat_to_rotmat(quat);
  float3x3 M = float3x3(
      R[0] * scale.x,
      R[1] * scale.y,
      R[2] * scale.z);
  float3x3 v_covar_sym = v_covar + transpose(v_covar);
  float3x3 v_M = v_covar_sym * M;
  float3x3 v_R = float3x3(
      v_M[0] * scale.x,
      v_M[1] * scale.y,
      v_M[2] * scale.z);
  v_scale.x += dot(v_M[0], R[0]);
  v_scale.y += dot(v_M[1], R[1]);
  v_scale.z += dot(v_M[2], R[2]);
  v_quat += quat_to_rotmat_vjp(quat, v_R);
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

inline void add_blur_vjp(float eps2d,
                         float4 conic_blur,
                         float compensation,
                         float v_compensation,
                         thread float4& v_covar) {
  float det_conic_blur = conic_blur.x * conic_blur.w - conic_blur.y * conic_blur.z;
  float v_sqr_comp = v_compensation * 0.5f / (compensation + 1.0e-6f);
  float one_minus_sqr_comp = 1.0f - compensation * compensation;
  v_covar.x += v_sqr_comp * (one_minus_sqr_comp * conic_blur.x -
                             eps2d * det_conic_blur);
  v_covar.y += v_sqr_comp * (one_minus_sqr_comp * conic_blur.y);
  v_covar.z += v_sqr_comp * (one_minus_sqr_comp * conic_blur.z);
  v_covar.w += v_sqr_comp * (one_minus_sqr_comp * conic_blur.w -
                             eps2d * det_conic_blur);
}

inline void persp_proj_vjp(float3 mean3d,
                           float3x3 cov3d,
                           float fx,
                           float fy,
                           float cx,
                           float cy,
                           uint width,
                           uint height,
                           float4 v_cov2d,
                           float2 v_mean2d,
                           thread float3& v_mean3d,
                           thread float3x3& v_cov3d) {
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
  float rz3 = rz2 * rz;
  float x_rz = x * rz;
  float y_rz = y * rz;
  float tx = z * min(lim_x_pos, max(-lim_x_neg, x_rz));
  float ty = z * min(lim_y_pos, max(-lim_y_neg, y_rz));

  float J[6] = {
      fx * rz, 0.0f, -fx * tx * rz2,
      0.0f, fy * rz, -fy * ty * rz2,
  };
  float cov[9] = {
      cov3d[0][0], cov3d[1][0], cov3d[2][0],
      cov3d[0][1], cov3d[1][1], cov3d[2][1],
      cov3d[0][2], cov3d[1][2], cov3d[2][2],
  };
  float vcov2[4] = {v_cov2d.x, v_cov2d.y, v_cov2d.z, v_cov2d.w};
  float vcov3[9] = {};
  for (uint a = 0; a < 3; ++a) {
    for (uint b = 0; b < 3; ++b) {
      for (uint r = 0; r < 2; ++r) {
        for (uint c = 0; c < 2; ++c) {
          vcov3[a * 3 + b] += J[r * 3 + a] * vcov2[r * 2 + c] *
                              J[c * 3 + b];
        }
      }
    }
  }
  v_cov3d = float3x3(
      float3(vcov3[0], vcov3[3], vcov3[6]),
      float3(vcov3[1], vcov3[4], vcov3[7]),
      float3(vcov3[2], vcov3[5], vcov3[8]));

  v_mean3d.x += fx * rz * v_mean2d.x;
  v_mean3d.y += fy * rz * v_mean2d.y;
  v_mean3d.z += -(fx * x * v_mean2d.x + fy * y * v_mean2d.y) * rz2;

  float v_J[6] = {};
  for (uint r = 0; r < 2; ++r) {
    for (uint a = 0; a < 3; ++a) {
      float left = 0.0f;
      float right = 0.0f;
      for (uint c = 0; c < 2; ++c) {
        for (uint b = 0; b < 3; ++b) {
          left += vcov2[r * 2 + c] * J[c * 3 + b] * cov[a * 3 + b];
          right += vcov2[c * 2 + r] * J[c * 3 + b] * cov[b * 3 + a];
        }
      }
      v_J[r * 3 + a] = left + right;
    }
  }

  if (x_rz <= lim_x_pos && x_rz >= -lim_x_neg) {
    v_mean3d.x += -fx * rz2 * v_J[2];
  } else {
    v_mean3d.z += -fx * rz3 * tx * v_J[2];
  }
  if (y_rz <= lim_y_pos && y_rz >= -lim_y_neg) {
    v_mean3d.y += -fy * rz2 * v_J[5];
  } else {
    v_mean3d.z += -fy * rz3 * ty * v_J[5];
  }
  v_mean3d.z += -fx * rz2 * v_J[0] - fy * rz2 * v_J[4] +
                2.0f * fx * tx * rz3 * v_J[2] +
                2.0f * fy * ty * rz3 * v_J[5];
}

kernel void gsplat_projection_ewa_3dgs_fused_backward_kernel(
    constant ProjectionKernelParams& params [[buffer(0)]],
    const device float* means [[buffer(1)]],
    const device float* covars [[buffer(2)]],
    const device float* quats [[buffer(3)]],
    const device float* scales [[buffer(4)]],
    const device float* viewmats [[buffer(5)]],
    const device float* Ks [[buffer(6)]],
    const device int* radii [[buffer(7)]],
    const device float* conics [[buffer(8)]],
    const device float* compensations [[buffer(9)]],
    const device float* v_means2d [[buffer(10)]],
    const device float* v_depths [[buffer(11)]],
    const device float* v_conics [[buffer(12)]],
    const device float* v_compensations [[buffer(13)]],
    device float* v_means [[buffer(14)]],
    device float* v_covars [[buffer(15)]],
    device float* v_quats [[buffer(16)]],
    device float* v_scales [[buffer(17)]],
    uint gaussian_idx [[thread_position_in_grid]]) {
  if (gaussian_idx >= params.B * params.N) {
    return;
  }

  uint bid = gaussian_idx / params.N;
  uint gid = gaussian_idx % params.N;
  uint mean_off = bid * params.N * 3 + gid * 3;
  uint cov_off = bid * params.N * 6 + gid * 6;
  uint quat_off = bid * params.N * 4 + gid * 4;
  uint scale_off = bid * params.N * 3 + gid * 3;
  float3 mean_w = float3(means[mean_off], means[mean_off + 1], means[mean_off + 2]);
  float3 v_mean_w_total = float3(0.0f);
  float3x3 v_covar_w_total = float3x3(float3(0.0f), float3(0.0f), float3(0.0f));
  float4 quat = float4(0.0f);
  float3 scale = float3(0.0f);
  if (params.use_covars == 0) {
    quat = float4(quats[quat_off], quats[quat_off + 1],
                  quats[quat_off + 2], quats[quat_off + 3]);
    scale = float3(scales[scale_off], scales[scale_off + 1],
                   scales[scale_off + 2]);
  }

  for (uint cid = 0; cid < params.C; ++cid) {
    uint idx = (bid * params.C + cid) * params.N + gid;
    if (radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) {
      continue;
    }

    uint view_off = bid * params.C * 16 + cid * 16;
    uint k_off = bid * params.C * 9 + cid * 9;
    float3x3 R = float3x3(
        float3(viewmats[view_off + 0], viewmats[view_off + 4], viewmats[view_off + 8]),
        float3(viewmats[view_off + 1], viewmats[view_off + 5], viewmats[view_off + 9]),
        float3(viewmats[view_off + 2], viewmats[view_off + 6], viewmats[view_off + 10]));
    float3 t = float3(viewmats[view_off + 3], viewmats[view_off + 7], viewmats[view_off + 11]);
    float3 mean_c = R * mean_w + t;
    float3x3 covar_w;
    if (params.use_covars != 0) {
      covar_w = float3x3(
          float3(covars[cov_off + 0], covars[cov_off + 1], covars[cov_off + 2]),
          float3(covars[cov_off + 1], covars[cov_off + 3], covars[cov_off + 4]),
          float3(covars[cov_off + 2], covars[cov_off + 4], covars[cov_off + 5]));
    } else {
      covar_w = covar_from_quat_scale(quat, scale);
    }
    float3x3 covar_c = R * covar_w * transpose(R);

    float c0 = conics[idx * 3];
    float c1 = conics[idx * 3 + 1];
    float c2 = conics[idx * 3 + 2];
    float4 covar2d_inv = float4(c0, c1, c1, c2);
    float4 v_covar2d_inv =
        float4(v_conics[idx * 3], 0.5f * v_conics[idx * 3 + 1],
               0.5f * v_conics[idx * 3 + 1], v_conics[idx * 3 + 2]);
    float4 tmp = float4(
        covar2d_inv.x * v_covar2d_inv.x + covar2d_inv.y * v_covar2d_inv.z,
        covar2d_inv.x * v_covar2d_inv.y + covar2d_inv.y * v_covar2d_inv.w,
        covar2d_inv.z * v_covar2d_inv.x + covar2d_inv.w * v_covar2d_inv.z,
        covar2d_inv.z * v_covar2d_inv.y + covar2d_inv.w * v_covar2d_inv.w);
    float4 v_covar2d = -float4(
        tmp.x * covar2d_inv.x + tmp.y * covar2d_inv.z,
        tmp.x * covar2d_inv.y + tmp.y * covar2d_inv.w,
        tmp.z * covar2d_inv.x + tmp.w * covar2d_inv.z,
        tmp.z * covar2d_inv.y + tmp.w * covar2d_inv.w);
    if (params.calc_compensations != 0) {
      add_blur_vjp(params.eps2d, covar2d_inv, compensations[idx],
                   v_compensations[idx], v_covar2d);
    }

    float3 v_mean_c = float3(0.0f, 0.0f, v_depths[idx]);
    float3x3 v_covar_c;
    float2 v_mean2d_val = float2(v_means2d[idx * 2], v_means2d[idx * 2 + 1]);
    persp_proj_vjp(mean_c, covar_c, Ks[k_off + 0], Ks[k_off + 4],
                   Ks[k_off + 2], Ks[k_off + 5], params.image_width,
                   params.image_height, v_covar2d, v_mean2d_val, v_mean_c,
                   v_covar_c);

    v_mean_w_total += transpose(R) * v_mean_c;
    v_covar_w_total += transpose(R) * v_covar_c * R;
  }

  v_means[mean_off] = v_mean_w_total.x;
  v_means[mean_off + 1] = v_mean_w_total.y;
  v_means[mean_off + 2] = v_mean_w_total.z;
  if (params.use_covars != 0) {
    v_covars[cov_off] = v_covar_w_total[0][0];
    v_covars[cov_off + 1] = v_covar_w_total[1][0] + v_covar_w_total[0][1];
    v_covars[cov_off + 2] = v_covar_w_total[2][0] + v_covar_w_total[0][2];
    v_covars[cov_off + 3] = v_covar_w_total[1][1];
    v_covars[cov_off + 4] = v_covar_w_total[2][1] + v_covar_w_total[1][2];
    v_covars[cov_off + 5] = v_covar_w_total[2][2];
  } else {
    float4 v_quat = float4(0.0f);
    float3 v_scale = float3(0.0f);
    quat_scale_to_covar_vjp(quat, scale, v_covar_w_total, v_quat, v_scale);
    v_quats[quat_off] = v_quat.x;
    v_quats[quat_off + 1] = v_quat.y;
    v_quats[quat_off + 2] = v_quat.z;
    v_quats[quat_off + 3] = v_quat.w;
    v_scales[scale_off] = v_scale.x;
    v_scales[scale_off + 1] = v_scale.y;
    v_scales[scale_off + 2] = v_scale.z;
  }
}

kernel void gsplat_projection_ewa_3dgs_fused_backward_viewmats_kernel(
    constant ProjectionKernelParams& params [[buffer(0)]],
    const device float* means [[buffer(1)]],
    const device float* covars [[buffer(2)]],
    const device float* quats [[buffer(3)]],
    const device float* scales [[buffer(4)]],
    const device float* viewmats [[buffer(5)]],
    const device float* Ks [[buffer(6)]],
    const device int* radii [[buffer(7)]],
    const device float* conics [[buffer(8)]],
    const device float* compensations [[buffer(9)]],
    const device float* v_means2d [[buffer(10)]],
    const device float* v_depths [[buffer(11)]],
    const device float* v_conics [[buffer(12)]],
    const device float* v_compensations [[buffer(13)]],
    device float* v_viewmats [[buffer(14)]],
    uint camera_idx [[thread_position_in_grid]]) {
  if (camera_idx >= params.B * params.C) {
    return;
  }

  uint bid = camera_idx / params.C;
  uint cid = camera_idx % params.C;
  uint view_off = bid * params.C * 16 + cid * 16;
  uint k_off = bid * params.C * 9 + cid * 9;
  float3x3 R = float3x3(
      float3(viewmats[view_off + 0], viewmats[view_off + 4], viewmats[view_off + 8]),
      float3(viewmats[view_off + 1], viewmats[view_off + 5], viewmats[view_off + 9]),
      float3(viewmats[view_off + 2], viewmats[view_off + 6], viewmats[view_off + 10]));
  float3 t = float3(viewmats[view_off + 3], viewmats[view_off + 7], viewmats[view_off + 11]);
  float3x3 v_R_total = float3x3(
      float3(0.0f), float3(0.0f), float3(0.0f));
  float3 v_t_total = float3(0.0f);

  for (uint gid = 0; gid < params.N; ++gid) {
    uint idx = (bid * params.C + cid) * params.N + gid;
    if (radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) {
      continue;
    }

    uint mean_off = bid * params.N * 3 + gid * 3;
    uint cov_off = bid * params.N * 6 + gid * 6;
    uint quat_off = bid * params.N * 4 + gid * 4;
    uint scale_off = bid * params.N * 3 + gid * 3;
    float3 mean_w = float3(means[mean_off], means[mean_off + 1], means[mean_off + 2]);
    float3 mean_c = R * mean_w + t;
    float3x3 covar_w;
    if (params.use_covars != 0) {
      covar_w = float3x3(
          float3(covars[cov_off + 0], covars[cov_off + 1], covars[cov_off + 2]),
          float3(covars[cov_off + 1], covars[cov_off + 3], covars[cov_off + 4]),
          float3(covars[cov_off + 2], covars[cov_off + 4], covars[cov_off + 5]));
    } else {
      float4 quat = float4(quats[quat_off], quats[quat_off + 1],
                           quats[quat_off + 2], quats[quat_off + 3]);
      float3 scale = float3(scales[scale_off], scales[scale_off + 1],
                            scales[scale_off + 2]);
      covar_w = covar_from_quat_scale(quat, scale);
    }
    float3x3 covar_c = R * covar_w * transpose(R);

    float c0 = conics[idx * 3];
    float c1 = conics[idx * 3 + 1];
    float c2 = conics[idx * 3 + 2];
    float4 covar2d_inv = float4(c0, c1, c1, c2);
    float4 v_covar2d_inv =
        float4(v_conics[idx * 3], 0.5f * v_conics[idx * 3 + 1],
               0.5f * v_conics[idx * 3 + 1], v_conics[idx * 3 + 2]);
    float4 tmp = float4(
        covar2d_inv.x * v_covar2d_inv.x + covar2d_inv.y * v_covar2d_inv.z,
        covar2d_inv.x * v_covar2d_inv.y + covar2d_inv.y * v_covar2d_inv.w,
        covar2d_inv.z * v_covar2d_inv.x + covar2d_inv.w * v_covar2d_inv.z,
        covar2d_inv.z * v_covar2d_inv.y + covar2d_inv.w * v_covar2d_inv.w);
    float4 v_covar2d = -float4(
        tmp.x * covar2d_inv.x + tmp.y * covar2d_inv.z,
        tmp.x * covar2d_inv.y + tmp.y * covar2d_inv.w,
        tmp.z * covar2d_inv.x + tmp.w * covar2d_inv.z,
        tmp.z * covar2d_inv.y + tmp.w * covar2d_inv.w);
    if (params.calc_compensations != 0) {
      add_blur_vjp(params.eps2d, covar2d_inv, compensations[idx],
                   v_compensations[idx], v_covar2d);
    }

    float3 v_mean_c = float3(0.0f, 0.0f, v_depths[idx]);
    float3x3 v_covar_c;
    float2 v_mean2d_val = float2(v_means2d[idx * 2], v_means2d[idx * 2 + 1]);
    persp_proj_vjp(mean_c, covar_c, Ks[k_off + 0], Ks[k_off + 4],
                   Ks[k_off + 2], Ks[k_off + 5], params.image_width,
                   params.image_height, v_covar2d, v_mean2d_val, v_mean_c,
                   v_covar_c);

    v_t_total += v_mean_c;
    v_R_total[0] += mean_w.x * v_mean_c;
    v_R_total[1] += mean_w.y * v_mean_c;
    v_R_total[2] += mean_w.z * v_mean_c;
    v_R_total += v_covar_c * R * transpose(covar_w);
    v_R_total += transpose(v_covar_c) * R * covar_w;
  }

  for (uint row = 0; row < 3; ++row) {
    for (uint col = 0; col < 3; ++col) {
      v_viewmats[view_off + row * 4 + col] = v_R_total[col][row];
    }
    v_viewmats[view_off + row * 4 + 3] = v_t_total[row];
  }
}
