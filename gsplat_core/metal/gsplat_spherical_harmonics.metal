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
