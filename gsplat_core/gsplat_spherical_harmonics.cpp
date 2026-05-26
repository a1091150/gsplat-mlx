#include "include/gsplat_spherical_harmonics.h"

#include "include/helper.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace gsplat_core {
namespace {

constexpr float kC0 = 0.2820947917738781f;
constexpr float kC1 = 0.48860251190292f;

struct SphericalHarmonicsKernelParams {
  uint32_t n;
  uint32_t k;
  uint32_t degrees_to_use;
  uint32_t use_masks;
};

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
  if (input.dirs.dtype().val() != mx::float32.val() ||
      input.coeffs.dtype().val() != mx::float32.val()) {
    throw std::runtime_error(
        "spherical_harmonics_forward currently supports float32 dirs and coeffs.");
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
  if (input.use_masks && input.masks.size() != input.dirs.size() / 3) {
    throw std::runtime_error("masks must have shape matching dirs prefix dims.");
  }
}

void validate_backward_input(const SphericalHarmonicsBackwardInput& input) {
  if (input.degrees_to_use < 0 || input.degrees_to_use > 4) {
    throw std::runtime_error("spherical_harmonics_backward supports degree 0..4.");
  }
  if (input.dirs.dtype().val() != mx::float32.val() ||
      input.coeffs.dtype().val() != mx::float32.val() ||
      input.v_colors.dtype().val() != mx::float32.val()) {
    throw std::runtime_error(
        "spherical_harmonics_backward currently supports float32 arrays.");
  }
  if (input.dirs.ndim() < 1 ||
      input.dirs.shape(static_cast<int>(input.dirs.ndim()) - 1) != 3) {
    throw std::runtime_error("dirs must have shape [..., 3].");
  }
  if (input.v_colors.shape() != input.dirs.shape()) {
    throw std::runtime_error("v_colors must have the same shape as dirs.");
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
  if (input.use_masks && input.masks.size() != input.dirs.size() / 3) {
    throw std::runtime_error("masks must have shape matching dirs prefix dims.");
  }
}

int sh_basis_count(int degree) {
  return (degree + 1) * (degree + 1);
}

void sh_basis_values(int degree, float dir_x, float dir_y, float dir_z,
                     float* basis) {
  const int count = sh_basis_count(degree);
  for (int i = 0; i < count; ++i) {
    basis[i] = 0.0f;
  }
  basis[0] = kC0;
  if (degree < 1) {
    return;
  }

  const float norm =
      std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
  if (norm == 0.0f) {
    return;
  }
  const float x = dir_x / norm;
  const float y = dir_y / norm;
  const float z = dir_z / norm;

  basis[1] = -kC1 * y;
  basis[2] = kC1 * z;
  basis[3] = -kC1 * x;
  if (degree < 2) {
    return;
  }

  const float z2 = z * z;
  const float f_tmp0_b = -1.092548430592079f * z;
  const float f_c1 = x * x - y * y;
  const float f_s1 = 2.0f * x * y;
  basis[4] = 0.5462742152960395f * f_s1;
  basis[5] = f_tmp0_b * y;
  basis[6] = 0.9461746957575601f * z2 - 0.3153915652525201f;
  basis[7] = f_tmp0_b * x;
  basis[8] = 0.5462742152960395f * f_c1;
  if (degree < 3) {
    return;
  }

  const float f_tmp0_c = -2.285228997322329f * z2 + 0.4570457994644658f;
  const float f_tmp1_b = 1.445305721320277f * z;
  const float f_c2 = x * f_c1 - y * f_s1;
  const float f_s2 = x * f_s1 + y * f_c1;
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

  const float f_tmp0_d =
      z * (-4.683325804901025f * z2 + 2.007139630671868f);
  const float f_tmp1_c = 3.31161143515146f * z2 - 0.47308734787878f;
  const float f_tmp2_b = -1.770130769779931f * z;
  const float f_c3 = x * f_c2 - y * f_s2;
  const float f_s3 = x * f_s2 + y * f_c2;
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

  auto prim = std::make_shared<GSPlatSphericalHarmonics>(
      to_stream(input.s), input.degrees_to_use, input.use_masks);
  std::vector<mx::array> inputs = {
      mx::contiguous(input.dirs),
      mx::contiguous(input.coeffs),
      mx::contiguous(input.masks),
  };
  return mx::array(dirs_shape(input.dirs), mx::float32, prim, inputs);
}

std::vector<mx::array> gsplat_spherical_harmonics_backward(
    const SphericalHarmonicsBackwardInput& input) {
  validate_backward_input(input);

  auto prim = std::make_shared<GSPlatSphericalHarmonicsBackward>(
      to_stream(input.s),
      input.degrees_to_use,
      input.use_masks,
      input.compute_v_dirs);
  std::vector<mx::Shape> output_shapes = {
      input.compute_v_dirs ? dirs_shape(input.dirs) : mx::Shape{0},
      input.coeffs.shape(),
  };
  std::vector<mx::Dtype> output_types = {mx::float32, mx::float32};
  std::vector<mx::array> inputs = {
      mx::contiguous(input.dirs),
      mx::contiguous(input.coeffs),
      mx::contiguous(input.masks),
      mx::contiguous(input.v_colors),
  };
  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

void GSPlatSphericalHarmonics::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& dirs = inputs[0];
  const auto& coeffs = inputs[1];
  const auto& masks = inputs[2];
  mx::eval(dirs, coeffs, masks);

  auto& colors = outputs[0];
  colors.set_data(mx::allocator::malloc(colors.nbytes()));
  std::memset(colors.data<void>(), 0, colors.nbytes());

  const int k = coeffs.shape(static_cast<int>(coeffs.ndim()) - 2);
  const int n = static_cast<int>(dirs.size() / 3);
  const float* dirs_data = dirs.data<float>();
  const float* coeffs_data = coeffs.data<float>();
  const bool* masks_data = use_masks_ ? masks.data<bool>() : nullptr;
  float* colors_data = colors.data<float>();

  for (int elem = 0; elem < n; ++elem) {
    if (masks_data != nullptr && !masks_data[elem]) {
      continue;
    }
    const float* dir = dirs_data + elem * 3;
    const float* elem_coeffs = coeffs_data + elem * k * 3;
    for (int channel = 0; channel < 3; ++channel) {
      colors_data[elem * 3 + channel] =
          sh_channel_to_color(degrees_to_use_,
                              channel,
                              dir[0],
                              dir[1],
                              dir[2],
                              elem_coeffs);
    }
  }
}

void GSPlatSphericalHarmonicsBackward::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& dirs = inputs[0];
  const auto& coeffs = inputs[1];
  const auto& masks = inputs[2];
  const auto& v_colors = inputs[3];
  mx::eval(dirs, coeffs, masks, v_colors);

  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  const int k = coeffs.shape(static_cast<int>(coeffs.ndim()) - 2);
  const int active_k = sh_basis_count(degrees_to_use_);
  const int n = static_cast<int>(dirs.size() / 3);
  const float* dirs_data = dirs.data<float>();
  const float* coeffs_data = coeffs.data<float>();
  const bool* masks_data = use_masks_ ? masks.data<bool>() : nullptr;
  const float* v_colors_data = v_colors.data<float>();
  float* v_dirs_data =
      compute_v_dirs_ ? outputs[kSHVDirs].data<float>() : nullptr;
  float* v_coeffs_data = outputs[kSHVCoeffs].data<float>();
  constexpr float eps = 1.0e-3f;

  float basis[25];
  for (int elem = 0; elem < n; ++elem) {
    if (masks_data != nullptr && !masks_data[elem]) {
      continue;
    }

    const float* dir = dirs_data + elem * 3;
    const float* elem_coeffs = coeffs_data + elem * k * 3;
    const float* elem_v_colors = v_colors_data + elem * 3;
    float* elem_v_coeffs = v_coeffs_data + elem * k * 3;
    sh_basis_values(degrees_to_use_, dir[0], dir[1], dir[2], basis);
    for (int b = 0; b < active_k; ++b) {
      for (int channel = 0; channel < 3; ++channel) {
        elem_v_coeffs[b * 3 + channel] = basis[b] * elem_v_colors[channel];
      }
    }

    if (v_dirs_data == nullptr) {
      continue;
    }
    for (int axis = 0; axis < 3; ++axis) {
      float dir_plus[3] = {dir[0], dir[1], dir[2]};
      float dir_minus[3] = {dir[0], dir[1], dir[2]};
      dir_plus[axis] += eps;
      dir_minus[axis] -= eps;
      float grad = 0.0f;
      for (int channel = 0; channel < 3; ++channel) {
        const float plus =
            sh_channel_to_color(degrees_to_use_, channel,
                                dir_plus[0], dir_plus[1], dir_plus[2],
                                elem_coeffs);
        const float minus =
            sh_channel_to_color(degrees_to_use_, channel,
                                dir_minus[0], dir_minus[1], dir_minus[2],
                                elem_coeffs);
        grad += elem_v_colors[channel] * (plus - minus) / (2.0f * eps);
      }
      v_dirs_data[elem * 3 + axis] = grad;
    }
  }
}

#ifdef _METAL_
void GSPlatSphericalHarmonics::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& colors = outputs[0];
  colors.set_data(mx::allocator::malloc(colors.nbytes()));
  std::memset(colors.data<void>(), 0, colors.nbytes());

  const auto& dirs = inputs[0];
  const auto& coeffs = inputs[1];
  const auto& masks = inputs[2];
  const uint32_t n = static_cast<uint32_t>(dirs.size() / 3);
  if (n == 0) {
    return;
  }

  SphericalHarmonicsKernelParams kernel_params = {
      .n = n,
      .k = static_cast<uint32_t>(
          coeffs.shape(static_cast<int>(coeffs.ndim()) - 2)),
      .degrees_to_use = static_cast<uint32_t>(degrees_to_use_),
      .use_masks = static_cast<uint32_t>(use_masks_),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("gsplat_core", current_binary_dir());
  auto kernel = d.get_kernel("gsplat_spherical_harmonics_forward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(dirs, 1);
  compute_encoder.set_input_array(coeffs, 2);
  compute_encoder.set_input_array(masks, 3);
  compute_encoder.set_output_array(colors, 4);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  const size_t tgp_size = std::min(static_cast<size_t>(n), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(n, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}

void GSPlatSphericalHarmonicsBackward::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  const auto& dirs = inputs[0];
  const auto& coeffs = inputs[1];
  const auto& masks = inputs[2];
  const auto& v_colors = inputs[3];
  const uint32_t n = static_cast<uint32_t>(dirs.size() / 3);
  if (n == 0) {
    return;
  }

  SphericalHarmonicsKernelParams kernel_params = {
      .n = n,
      .k = static_cast<uint32_t>(
          coeffs.shape(static_cast<int>(coeffs.ndim()) - 2)),
      .degrees_to_use = static_cast<uint32_t>(degrees_to_use_),
      .use_masks = static_cast<uint32_t>(use_masks_),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("gsplat_core", current_binary_dir());
  auto kernel = d.get_kernel("gsplat_spherical_harmonics_backward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(dirs, 1);
  compute_encoder.set_input_array(coeffs, 2);
  compute_encoder.set_input_array(masks, 3);
  compute_encoder.set_input_array(v_colors, 4);
  compute_encoder.set_output_array(outputs[kSHVDirs], 5);
  compute_encoder.set_output_array(outputs[kSHVCoeffs], 6);
  compute_encoder.set_bytes(static_cast<uint32_t>(compute_v_dirs_), 7);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  const size_t tgp_size = std::min(static_cast<size_t>(n), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(n, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}
#else
void GSPlatSphericalHarmonics::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatSphericalHarmonics has no GPU implementation.");
}

void GSPlatSphericalHarmonicsBackward::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatSphericalHarmonicsBackward has no GPU implementation.");
}
#endif

std::vector<mx::array> GSPlatSphericalHarmonics::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("GSPlatSphericalHarmonics jvp is not implemented.");
}

std::vector<mx::array> GSPlatSphericalHarmonics::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("GSPlatSphericalHarmonics vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GSPlatSphericalHarmonics::vmap(const std::vector<mx::array>&,
                               const std::vector<int>&) {
  throw std::runtime_error("GSPlatSphericalHarmonics vmap is not implemented.");
}

bool GSPlatSphericalHarmonics::is_equivalent(
    const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  const auto* other_ptr =
      dynamic_cast<const GSPlatSphericalHarmonics*>(&other);
  if (!other_ptr) {
    return false;
  }
  return degrees_to_use_ == other_ptr->degrees_to_use_ &&
         use_masks_ == other_ptr->use_masks_;
}

std::vector<mx::array> GSPlatSphericalHarmonicsBackward::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "GSPlatSphericalHarmonicsBackward jvp is not implemented.");
}

std::vector<mx::array> GSPlatSphericalHarmonicsBackward::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatSphericalHarmonicsBackward vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GSPlatSphericalHarmonicsBackward::vmap(const std::vector<mx::array>&,
                                       const std::vector<int>&) {
  throw std::runtime_error(
      "GSPlatSphericalHarmonicsBackward vmap is not implemented.");
}

bool GSPlatSphericalHarmonicsBackward::is_equivalent(
    const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  const auto* other_ptr =
      dynamic_cast<const GSPlatSphericalHarmonicsBackward*>(&other);
  if (!other_ptr) {
    return false;
  }
  return degrees_to_use_ == other_ptr->degrees_to_use_ &&
         use_masks_ == other_ptr->use_masks_ &&
         compute_v_dirs_ == other_ptr->compute_v_dirs_;
}

}  // namespace gsplat_core
