#include "include/gsplat_quat_scale_to_covar.h"

#include "include/helper.h"

#include <algorithm>
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

struct Mat3 {
  float v[9];
};

struct QuatScaleToCovarPreciKernelParams {
  uint32_t n;
  uint32_t compute_covar;
  uint32_t compute_preci;
  uint32_t triu;
};

mx::Shape output_shape(const mx::array& quats, bool triu) {
  mx::Shape shape;
  const int ndim = static_cast<int>(quats.ndim());
  shape.reserve(static_cast<size_t>(ndim + (triu ? 0 : 1)));
  for (int i = 0; i < ndim - 1; ++i) {
    shape.push_back(quats.shape(i));
  }
  if (triu) {
    shape.push_back(6);
  } else {
    shape.push_back(3);
    shape.push_back(3);
  }
  return shape;
}

void validate_input(const QuatScaleToCovarPreciInput& input) {
  if (input.quats.dtype().val() != mx::float32.val() ||
      input.scales.dtype().val() != mx::float32.val()) {
    throw std::runtime_error(
        "quat_scale_to_covar_preci_forward currently supports float32 inputs.");
  }
  if (input.quats.ndim() < 1 ||
      input.quats.shape(static_cast<int>(input.quats.ndim()) - 1) != 4) {
    throw std::runtime_error("quats must have shape [..., 4].");
  }
  if (input.scales.ndim() < 1 ||
      input.scales.shape(static_cast<int>(input.scales.ndim()) - 1) != 3) {
    throw std::runtime_error("scales must have shape [..., 3].");
  }
  if (input.quats.size() / 4 != input.scales.size() / 3) {
    throw std::runtime_error(
        "quats and scales prefix dimensions must have the same size.");
  }
}

Mat3 quat_to_rotmat(const float* quat) {
  float w = quat[0];
  float x = quat[1];
  float y = quat[2];
  float z = quat[3];
  const float norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (norm == 0.0f) {
    throw std::runtime_error("quaternion norm must be non-zero.");
  }
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;

  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;
  const float wx = w * x;
  const float wy = w * y;
  const float wz = w * z;

  Mat3 r = {};
  r.v[0] = 1.0f - 2.0f * (y2 + z2);
  r.v[1] = 2.0f * (xy - wz);
  r.v[2] = 2.0f * (xz + wy);
  r.v[3] = 2.0f * (xy + wz);
  r.v[4] = 1.0f - 2.0f * (x2 + z2);
  r.v[5] = 2.0f * (yz - wx);
  r.v[6] = 2.0f * (xz - wy);
  r.v[7] = 2.0f * (yz + wx);
  r.v[8] = 1.0f - 2.0f * (x2 + y2);
  return r;
}

Mat3 covariance_from_rotation_scale(const Mat3& r,
                                    const float* scale,
                                    bool precision) {
  float s[3] = {scale[0], scale[1], scale[2]};
  if (precision) {
    if (s[0] == 0.0f || s[1] == 0.0f || s[2] == 0.0f) {
      throw std::runtime_error("precision path expects non-zero scales.");
    }
    s[0] = 1.0f / s[0];
    s[1] = 1.0f / s[1];
    s[2] = 1.0f / s[2];
  }

  Mat3 out = {};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      float value = 0.0f;
      for (int k = 0; k < 3; ++k) {
        value += r.v[row * 3 + k] * s[k] * s[k] * r.v[col * 3 + k];
      }
      out.v[row * 3 + col] = value;
    }
  }
  return out;
}

void write_matrix(const Mat3& matrix, bool triu, float* out) {
  if (triu) {
    out[0] = matrix.v[0];
    out[1] = matrix.v[1];
    out[2] = matrix.v[2];
    out[3] = matrix.v[4];
    out[4] = matrix.v[5];
    out[5] = matrix.v[8];
  } else {
    for (int i = 0; i < 9; ++i) {
      out[i] = matrix.v[i];
    }
  }
}

}  // namespace

std::vector<mx::array> gsplat_quat_scale_to_covar_preci_forward(
    const QuatScaleToCovarPreciInput& input) {
  validate_input(input);

  auto prim = std::make_shared<GSPlatQuatScaleToCovarPreci>(
      to_stream(input.s), input.compute_covar, input.compute_preci, input.triu);
  const mx::Shape out_shape = output_shape(input.quats, input.triu);
  std::vector<mx::Shape> output_shapes = {
      input.compute_covar ? out_shape : mx::Shape{0},
      input.compute_preci ? out_shape : mx::Shape{0},
  };
  std::vector<mx::Dtype> output_types = {mx::float32, mx::float32};
  std::vector<mx::array> inputs = {
      mx::contiguous(input.quats),
      mx::contiguous(input.scales),
  };
  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

void GSPlatQuatScaleToCovarPreci::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& quats = inputs[0];
  const auto& scales = inputs[1];
  mx::eval(quats, scales);

  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  const int n = static_cast<int>(quats.size() / 4);
  const int out_stride = triu_ ? 6 : 9;
  const float* quats_data = quats.data<float>();
  const float* scales_data = scales.data<float>();
  float* covars_data = compute_covar_ ? outputs[kCovars].data<float>() : nullptr;
  float* precis_data = compute_preci_ ? outputs[kPrecis].data<float>() : nullptr;

  for (int elem = 0; elem < n; ++elem) {
    const Mat3 r = quat_to_rotmat(quats_data + elem * 4);
    const float* scale = scales_data + elem * 3;
    if (compute_covar_) {
      const Mat3 covar = covariance_from_rotation_scale(r, scale, false);
      write_matrix(covar, triu_, covars_data + elem * out_stride);
    }
    if (compute_preci_) {
      const Mat3 preci = covariance_from_rotation_scale(r, scale, true);
      write_matrix(preci, triu_, precis_data + elem * out_stride);
    }
  }
}

#ifdef _METAL_
void GSPlatQuatScaleToCovarPreci::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  const auto& quats = inputs[0];
  const auto& scales = inputs[1];
  const uint32_t n = static_cast<uint32_t>(quats.size() / 4);
  if (n == 0) {
    return;
  }

  QuatScaleToCovarPreciKernelParams kernel_params = {
      .n = n,
      .compute_covar = static_cast<uint32_t>(compute_covar_),
      .compute_preci = static_cast<uint32_t>(compute_preci_),
      .triu = static_cast<uint32_t>(triu_),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("gsplat_core", current_binary_dir());
  auto kernel = d.get_kernel("gsplat_quat_scale_to_covar_preci_forward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(quats, 1);
  compute_encoder.set_input_array(scales, 2);
  compute_encoder.set_output_array(outputs[kCovars], 3);
  compute_encoder.set_output_array(outputs[kPrecis], 4);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  const size_t tgp_size = std::min(static_cast<size_t>(n), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(n, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}
#else
void GSPlatQuatScaleToCovarPreci::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatQuatScaleToCovarPreci has no GPU implementation.");
}
#endif

std::vector<mx::array> GSPlatQuatScaleToCovarPreci::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("GSPlatQuatScaleToCovarPreci jvp is not implemented.");
}

std::vector<mx::array> GSPlatQuatScaleToCovarPreci::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("GSPlatQuatScaleToCovarPreci vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GSPlatQuatScaleToCovarPreci::vmap(const std::vector<mx::array>&,
                                  const std::vector<int>&) {
  throw std::runtime_error("GSPlatQuatScaleToCovarPreci vmap is not implemented.");
}

bool GSPlatQuatScaleToCovarPreci::is_equivalent(
    const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  const auto* other_ptr =
      dynamic_cast<const GSPlatQuatScaleToCovarPreci*>(&other);
  if (!other_ptr) {
    return false;
  }
  return compute_covar_ == other_ptr->compute_covar_ &&
         compute_preci_ == other_ptr->compute_preci_ &&
         triu_ == other_ptr->triu_;
}

}  // namespace gsplat_core
