#include <metal_stdlib>

using namespace metal;

constant float C0 = 0.2820947917738781f;
constant float C1 = 0.48860251190292f;

struct SphericalHarmonicsKernelParams {
  uint n;
  uint k;
  uint degrees_to_use;
  uint use_masks;
};

inline float sh_channel_to_color(uint degree,
                                 uint channel,
                                 float3 dir,
                                 const device float* coeffs) {
  float result = C0 * coeffs[channel];
  if (degree < 1) {
    return result;
  }

  float norm = length(dir);
  if (norm == 0.0f) {
    return result;
  }
  float3 n_dir = dir / norm;
  float x = n_dir.x;
  float y = n_dir.y;
  float z = n_dir.z;

  result += C1 * (-y * coeffs[1 * 3 + channel] +
                  z * coeffs[2 * 3 + channel] -
                  x * coeffs[3 * 3 + channel]);
  if (degree < 2) {
    return result;
  }

  float z2 = z * z;
  float f_tmp0_b = -1.092548430592079f * z;
  float f_c1 = x * x - y * y;
  float f_s1 = 2.0f * x * y;
  float p_sh6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
  float p_sh7 = f_tmp0_b * x;
  float p_sh5 = f_tmp0_b * y;
  float p_sh8 = 0.5462742152960395f * f_c1;
  float p_sh4 = 0.5462742152960395f * f_s1;

  result += p_sh4 * coeffs[4 * 3 + channel] +
            p_sh5 * coeffs[5 * 3 + channel] +
            p_sh6 * coeffs[6 * 3 + channel] +
            p_sh7 * coeffs[7 * 3 + channel] +
            p_sh8 * coeffs[8 * 3 + channel];
  if (degree < 3) {
    return result;
  }

  float f_tmp0_c = -2.285228997322329f * z2 + 0.4570457994644658f;
  float f_tmp1_b = 1.445305721320277f * z;
  float f_c2 = x * f_c1 - y * f_s1;
  float f_s2 = x * f_s1 + y * f_c1;
  float p_sh12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
  float p_sh13 = f_tmp0_c * x;
  float p_sh11 = f_tmp0_c * y;
  float p_sh14 = f_tmp1_b * f_c1;
  float p_sh10 = f_tmp1_b * f_s1;
  float p_sh15 = -0.5900435899266435f * f_c2;
  float p_sh9 = -0.5900435899266435f * f_s2;

  result += p_sh9 * coeffs[9 * 3 + channel] +
            p_sh10 * coeffs[10 * 3 + channel] +
            p_sh11 * coeffs[11 * 3 + channel] +
            p_sh12 * coeffs[12 * 3 + channel] +
            p_sh13 * coeffs[13 * 3 + channel] +
            p_sh14 * coeffs[14 * 3 + channel] +
            p_sh15 * coeffs[15 * 3 + channel];
  if (degree < 4) {
    return result;
  }

  float f_tmp0_d = z * (-4.683325804901025f * z2 + 2.007139630671868f);
  float f_tmp1_c = 3.31161143515146f * z2 - 0.47308734787878f;
  float f_tmp2_b = -1.770130769779931f * z;
  float f_c3 = x * f_c2 - y * f_s2;
  float f_s3 = x * f_s2 + y * f_c2;
  float p_sh20 = 1.984313483298443f * z * p_sh12 -
                 1.006230589874905f * p_sh6;
  float p_sh21 = f_tmp0_d * x;
  float p_sh19 = f_tmp0_d * y;
  float p_sh22 = f_tmp1_c * f_c1;
  float p_sh18 = f_tmp1_c * f_s1;
  float p_sh23 = f_tmp2_b * f_c2;
  float p_sh17 = f_tmp2_b * f_s2;
  float p_sh24 = 0.6258357354491763f * f_c3;
  float p_sh16 = 0.6258357354491763f * f_s3;

  result += p_sh16 * coeffs[16 * 3 + channel] +
            p_sh17 * coeffs[17 * 3 + channel] +
            p_sh18 * coeffs[18 * 3 + channel] +
            p_sh19 * coeffs[19 * 3 + channel] +
            p_sh20 * coeffs[20 * 3 + channel] +
            p_sh21 * coeffs[21 * 3 + channel] +
            p_sh22 * coeffs[22 * 3 + channel] +
            p_sh23 * coeffs[23 * 3 + channel] +
            p_sh24 * coeffs[24 * 3 + channel];
  return result;
}

inline void sh_basis_values(uint degree, float3 dir, thread float* basis) {
  for (uint i = 0; i < 25; ++i) {
    basis[i] = 0.0f;
  }
  basis[0] = C0;
  if (degree < 1) {
    return;
  }

  float norm = length(dir);
  if (norm == 0.0f) {
    return;
  }
  float3 n_dir = dir / norm;
  float x = n_dir.x;
  float y = n_dir.y;
  float z = n_dir.z;

  basis[1] = -C1 * y;
  basis[2] = C1 * z;
  basis[3] = -C1 * x;
  if (degree < 2) {
    return;
  }

  float z2 = z * z;
  float f_tmp0_b = -1.092548430592079f * z;
  float f_c1 = x * x - y * y;
  float f_s1 = 2.0f * x * y;
  basis[4] = 0.5462742152960395f * f_s1;
  basis[5] = f_tmp0_b * y;
  basis[6] = 0.9461746957575601f * z2 - 0.3153915652525201f;
  basis[7] = f_tmp0_b * x;
  basis[8] = 0.5462742152960395f * f_c1;
  if (degree < 3) {
    return;
  }

  float f_tmp0_c = -2.285228997322329f * z2 + 0.4570457994644658f;
  float f_tmp1_b = 1.445305721320277f * z;
  float f_c2 = x * f_c1 - y * f_s1;
  float f_s2 = x * f_s1 + y * f_c1;
  basis[9] = -0.5900435899266435f * f_s2;
  basis[10] = f_tmp1_b * f_s1;
  basis[11] = f_tmp0_c * y;
  basis[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);
  basis[13] = f_tmp0_c * x;
  basis[14] = f_tmp1_b * f_c1;
  basis[15] = -0.5900435899266435f * f_c2;
  if (degree < 4) {
    return;
  }

  float f_tmp0_d = z * (-4.683325804901025f * z2 + 2.007139630671868f);
  float f_tmp1_c = 3.31161143515146f * z2 - 0.47308734787878f;
  float f_tmp2_b = -1.770130769779931f * z;
  float f_c3 = x * f_c2 - y * f_s2;
  float f_s3 = x * f_s2 + y * f_c2;
  basis[16] = 0.6258357354491763f * f_s3;
  basis[17] = f_tmp2_b * f_s2;
  basis[18] = f_tmp1_c * f_s1;
  basis[19] = f_tmp0_d * y;
  basis[20] =
      1.984313483298443f * z * basis[12] - 1.006230589874905f * basis[6];
  basis[21] = f_tmp0_d * x;
  basis[22] = f_tmp1_c * f_c1;
  basis[23] = f_tmp2_b * f_c2;
  basis[24] = 0.6258357354491763f * f_c3;
}

inline float3 sh_direction_vjp(uint degree,
                               float3 dir,
                               const device float* coeffs,
                               const device float* v_colors) {
  if (degree < 1) {
    return float3(0.0f);
  }

  float norm = length(dir);
  if (norm == 0.0f) {
    return float3(0.0f);
  }
  float inorm = 1.0f / norm;
  float3 n_dir = dir * inorm;
  float x = n_dir.x;
  float y = n_dir.y;
  float z = n_dir.z;

  float dx[25];
  float dy[25];
  float dz[25];
  for (uint i = 0; i < 25; ++i) {
    dx[i] = 0.0f;
    dy[i] = 0.0f;
    dz[i] = 0.0f;
  }

  dx[3] = -C1;
  dy[1] = -C1;
  dz[2] = C1;

  float z2 = z * z;
  float f_tmp0_b = -1.092548430592079f * z;
  float f_c1 = x * x - y * y;
  float f_s1 = 2.0f * x * y;
  float f_tmp0_b_z = -1.092548430592079f;
  float f_c1_x = 2.0f * x;
  float f_c1_y = -2.0f * y;
  float f_s1_x = 2.0f * y;
  float f_s1_y = 2.0f * x;
  float p_sh6_z = 2.0f * 0.9461746957575601f * z;
  if (degree >= 2) {
    dx[4] = 0.5462742152960395f * f_s1_x;
    dy[4] = 0.5462742152960395f * f_s1_y;
    dy[5] = f_tmp0_b;
    dz[5] = f_tmp0_b_z * y;
    dz[6] = p_sh6_z;
    dx[7] = f_tmp0_b;
    dz[7] = f_tmp0_b_z * x;
    dx[8] = 0.5462742152960395f * f_c1_x;
    dy[8] = 0.5462742152960395f * f_c1_y;
  }

  float f_tmp0_c = -2.285228997322329f * z2 + 0.4570457994644658f;
  float f_tmp1_b = 1.445305721320277f * z;
  float f_c2 = x * f_c1 - y * f_s1;
  float f_s2 = x * f_s1 + y * f_c1;
  float p_sh12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
  float f_tmp0_c_z = -2.285228997322329f * 2.0f * z;
  float f_tmp1_b_z = 1.445305721320277f;
  float f_c2_x = f_c1 + x * f_c1_x - y * f_s1_x;
  float f_c2_y = x * f_c1_y - f_s1 - y * f_s1_y;
  float f_s2_x = f_s1 + x * f_s1_x + y * f_c1_x;
  float f_s2_y = x * f_s1_y + f_c1 + y * f_c1_y;
  float p_sh12_z = 3.0f * 1.865881662950577f * z2 - 1.119528997770346f;
  if (degree >= 3) {
    dx[9] = -0.5900435899266435f * f_s2_x;
    dy[9] = -0.5900435899266435f * f_s2_y;
    dx[10] = f_tmp1_b * f_s1_x;
    dy[10] = f_tmp1_b * f_s1_y;
    dz[10] = f_tmp1_b_z * f_s1;
    dy[11] = f_tmp0_c;
    dz[11] = f_tmp0_c_z * y;
    dz[12] = p_sh12_z;
    dx[13] = f_tmp0_c;
    dz[13] = f_tmp0_c_z * x;
    dx[14] = f_tmp1_b * f_c1_x;
    dy[14] = f_tmp1_b * f_c1_y;
    dz[14] = f_tmp1_b_z * f_c1;
    dx[15] = -0.5900435899266435f * f_c2_x;
    dy[15] = -0.5900435899266435f * f_c2_y;
  }

  if (degree >= 4) {
    float f_tmp0_d = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float f_tmp1_c = 3.31161143515146f * z2 - 0.47308734787878f;
    float f_tmp2_b = -1.770130769779931f * z;
    float f_tmp0_d_z = 3.0f * -4.683325804901025f * z2 + 2.007139630671868f;
    float f_tmp1_c_z = 2.0f * 3.31161143515146f * z;
    float f_tmp2_b_z = -1.770130769779931f;
    float f_c3_x = f_c2 + x * f_c2_x - y * f_s2_x;
    float f_c3_y = x * f_c2_y - f_s2 - y * f_s2_y;
    float f_s3_x = f_s2 + y * f_c2_x + x * f_s2_x;
    float f_s3_y = x * f_s2_y + f_c2 + y * f_c2_y;
    dz[16] = 0.0f;
    dx[16] = 0.6258357354491763f * f_s3_x;
    dy[16] = 0.6258357354491763f * f_s3_y;
    dx[17] = f_tmp2_b * f_s2_x;
    dy[17] = f_tmp2_b * f_s2_y;
    dz[17] = f_tmp2_b_z * f_s2;
    dx[18] = f_tmp1_c * f_s1_x;
    dy[18] = f_tmp1_c * f_s1_y;
    dz[18] = f_tmp1_c_z * f_s1;
    dy[19] = f_tmp0_d;
    dz[19] = f_tmp0_d_z * y;
    dz[20] = 1.984313483298443f * (p_sh12 + z * p_sh12_z) -
             1.006230589874905f * p_sh6_z;
    dx[21] = f_tmp0_d;
    dz[21] = f_tmp0_d_z * x;
    dx[22] = f_tmp1_c * f_c1_x;
    dy[22] = f_tmp1_c * f_c1_y;
    dz[22] = f_tmp1_c_z * f_c1;
    dx[23] = f_tmp2_b * f_c2_x;
    dy[23] = f_tmp2_b * f_c2_y;
    dz[23] = f_tmp2_b_z * f_c2;
    dx[24] = 0.6258357354491763f * f_c3_x;
    dy[24] = 0.6258357354491763f * f_c3_y;
  }

  uint active_k = (degree + 1) * (degree + 1);
  float3 v_dir_n = float3(0.0f);
  for (uint b = 1; b < active_k; ++b) {
    float v_basis = coeffs[b * 3] * v_colors[0] +
                    coeffs[b * 3 + 1] * v_colors[1] +
                    coeffs[b * 3 + 2] * v_colors[2];
    v_dir_n += float3(dx[b], dy[b], dz[b]) * v_basis;
  }
  return (v_dir_n - dot(v_dir_n, n_dir) * n_dir) * inorm;
}

kernel void gsplat_spherical_harmonics_forward_kernel(
    constant SphericalHarmonicsKernelParams& params [[buffer(0)]],
    const device float* dirs [[buffer(1)]],
    const device float* coeffs [[buffer(2)]],
    const device bool* masks [[buffer(3)]],
    device float* colors [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= params.n) {
    return;
  }

  if (params.use_masks != 0 && !masks[idx]) {
    colors[idx * 3] = 0.0f;
    colors[idx * 3 + 1] = 0.0f;
    colors[idx * 3 + 2] = 0.0f;
    return;
  }

  float3 dir = float3(dirs[idx * 3], dirs[idx * 3 + 1], dirs[idx * 3 + 2]);
  const device float* elem_coeffs = coeffs + idx * params.k * 3;
  colors[idx * 3] =
      sh_channel_to_color(params.degrees_to_use, 0, dir, elem_coeffs);
  colors[idx * 3 + 1] =
      sh_channel_to_color(params.degrees_to_use, 1, dir, elem_coeffs);
  colors[idx * 3 + 2] =
      sh_channel_to_color(params.degrees_to_use, 2, dir, elem_coeffs);
}

kernel void gsplat_spherical_harmonics_backward_kernel(
    constant SphericalHarmonicsKernelParams& params [[buffer(0)]],
    const device float* dirs [[buffer(1)]],
    const device float* coeffs [[buffer(2)]],
    const device bool* masks [[buffer(3)]],
    const device float* v_colors [[buffer(4)]],
    device float* v_dirs [[buffer(5)]],
    device float* v_coeffs [[buffer(6)]],
    constant uint& compute_v_dirs [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= params.n) {
    return;
  }

  if (params.use_masks != 0 && !masks[idx]) {
    return;
  }

  float3 dir = float3(dirs[idx * 3], dirs[idx * 3 + 1], dirs[idx * 3 + 2]);
  const device float* elem_coeffs = coeffs + idx * params.k * 3;
  const device float* elem_v_colors = v_colors + idx * 3;
  device float* elem_v_coeffs = v_coeffs + idx * params.k * 3;
  uint active_k = (params.degrees_to_use + 1) * (params.degrees_to_use + 1);

  float basis[25];
  sh_basis_values(params.degrees_to_use, dir, basis);
  for (uint b = 0; b < active_k; ++b) {
    elem_v_coeffs[b * 3] = basis[b] * elem_v_colors[0];
    elem_v_coeffs[b * 3 + 1] = basis[b] * elem_v_colors[1];
    elem_v_coeffs[b * 3 + 2] = basis[b] * elem_v_colors[2];
  }

  if (compute_v_dirs == 0) {
    return;
  }

  float3 v_dir =
      sh_direction_vjp(params.degrees_to_use, dir, elem_coeffs, elem_v_colors);
  v_dirs[idx * 3] = v_dir.x;
  v_dirs[idx * 3 + 1] = v_dir.y;
  v_dirs[idx * 3 + 2] = v_dir.z;
}
