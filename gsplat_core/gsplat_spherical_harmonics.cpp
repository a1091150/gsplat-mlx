#include "include/gsplat_spherical_harmonics.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/ops.h"

namespace gsplat_core {
namespace {

constexpr float kC0 = 0.2820947917738781f;
constexpr float kC1 = 0.48860251190292f;

mx::Shape dirs_shape(const mx::array& dirs) {
  mx::Shape shape;
  shape.reserve(dirs.ndim());
  for (int i = 0; i < static_cast<int>(dirs.ndim()); ++i) {
    shape.push_back(dirs.shape(i));
  }
  return shape;
}

void validate_input(const SphericalHarmonicsInput& input) {
  if (input.degrees_to_use < 0 || input.degrees_to_use > 4) {
    throw std::runtime_error("spherical_harmonics_forward supports degree 0..4.");
  }
  if (input.dirs.ndim() < 1 ||
      input.dirs.shape(static_cast<int>(input.dirs.ndim()) - 1) != 3) {
    throw std::runtime_error("dirs must have shape [..., 3].");
  }
  if (input.coeffs.ndim() < 2 ||
      input.coeffs.shape(static_cast<int>(input.coeffs.ndim()) - 1) != 3) {
    throw std::runtime_error("coeffs must have shape [..., K, 3].");
  }
  const int k = input.coeffs.shape(static_cast<int>(input.coeffs.ndim()) - 2);
  const int required_k = (input.degrees_to_use + 1) *
                         (input.degrees_to_use + 1);
  if (k < required_k) {
    throw std::runtime_error(
        "coeffs has fewer SH bases than degrees_to_use requires.");
  }
  if (input.coeffs.size() / (k * 3) != input.dirs.size() / 3) {
    throw std::runtime_error(
        "dirs and coeffs prefix dimensions must have the same size.");
  }
}

float sh_channel_to_color(int degree,
                          int channel,
                          float dir_x,
                          float dir_y,
                          float dir_z,
                          const float* coeffs) {
  float result = kC0 * coeffs[channel];
  if (degree < 1) {
    return result;
  }

  const float norm =
      std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
  if (norm == 0.0f) {
    return result;
  }
  const float x = dir_x / norm;
  const float y = dir_y / norm;
  const float z = dir_z / norm;

  result += kC1 * (-y * coeffs[1 * 3 + channel] +
                   z * coeffs[2 * 3 + channel] -
                   x * coeffs[3 * 3 + channel]);
  if (degree < 2) {
    return result;
  }

  const float z2 = z * z;
  const float f_tmp0_b = -1.092548430592079f * z;
  const float f_c1 = x * x - y * y;
  const float f_s1 = 2.0f * x * y;
  const float p_sh6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
  const float p_sh7 = f_tmp0_b * x;
  const float p_sh5 = f_tmp0_b * y;
  const float p_sh8 = 0.5462742152960395f * f_c1;
  const float p_sh4 = 0.5462742152960395f * f_s1;

  result += p_sh4 * coeffs[4 * 3 + channel] +
            p_sh5 * coeffs[5 * 3 + channel] +
            p_sh6 * coeffs[6 * 3 + channel] +
            p_sh7 * coeffs[7 * 3 + channel] +
            p_sh8 * coeffs[8 * 3 + channel];
  if (degree < 3) {
    return result;
  }

  const float f_tmp0_c = -2.285228997322329f * z2 + 0.4570457994644658f;
  const float f_tmp1_b = 1.445305721320277f * z;
  const float f_c2 = x * f_c1 - y * f_s1;
  const float f_s2 = x * f_s1 + y * f_c1;
  const float p_sh12 =
      z * (1.865881662950577f * z2 - 1.119528997770346f);
  const float p_sh13 = f_tmp0_c * x;
  const float p_sh11 = f_tmp0_c * y;
  const float p_sh14 = f_tmp1_b * f_c1;
  const float p_sh10 = f_tmp1_b * f_s1;
  const float p_sh15 = -0.5900435899266435f * f_c2;
  const float p_sh9 = -0.5900435899266435f * f_s2;

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

  const float f_tmp0_d =
      z * (-4.683325804901025f * z2 + 2.007139630671868f);
  const float f_tmp1_c = 3.31161143515146f * z2 - 0.47308734787878f;
  const float f_tmp2_b = -1.770130769779931f * z;
  const float f_c3 = x * f_c2 - y * f_s2;
  const float f_s3 = x * f_s2 + y * f_c2;
  const float p_sh20 =
      1.984313483298443f * z * p_sh12 - 1.006230589874905f * p_sh6;
  const float p_sh21 = f_tmp0_d * x;
  const float p_sh19 = f_tmp0_d * y;
  const float p_sh22 = f_tmp1_c * f_c1;
  const float p_sh18 = f_tmp1_c * f_s1;
  const float p_sh23 = f_tmp2_b * f_c2;
  const float p_sh17 = f_tmp2_b * f_s2;
  const float p_sh24 = 0.6258357354491763f * f_c3;
  const float p_sh16 = 0.6258357354491763f * f_s3;

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

}  // namespace

mx::array gsplat_spherical_harmonics_forward(
    const SphericalHarmonicsInput& input) {
  validate_input(input);

  mx::array dirs = mx::contiguous(input.dirs);
  mx::array coeffs = mx::contiguous(input.coeffs);
  mx::array masks = mx::contiguous(input.masks);
  mx::eval(dirs, coeffs, masks);

  const int k = coeffs.shape(static_cast<int>(coeffs.ndim()) - 2);
  const int n = static_cast<int>(dirs.size() / 3);
  const float* dirs_data = dirs.data<float>();
  const float* coeffs_data = coeffs.data<float>();
  const bool* masks_data =
      input.use_masks ? masks.data<bool>() : nullptr;

  std::vector<float> colors(static_cast<size_t>(n * 3), 0.0f);
  for (int elem = 0; elem < n; ++elem) {
    if (masks_data != nullptr && !masks_data[elem]) {
      continue;
    }
    const float* dir = dirs_data + elem * 3;
    const float* elem_coeffs = coeffs_data + elem * k * 3;
    for (int channel = 0; channel < 3; ++channel) {
      colors[static_cast<size_t>(elem * 3 + channel)] =
          sh_channel_to_color(input.degrees_to_use,
                              channel,
                              dir[0],
                              dir[1],
                              dir[2],
                              elem_coeffs);
    }
  }

  return mx::array(colors.begin(), dirs_shape(dirs), mx::float32);
}

}  // namespace gsplat_core
