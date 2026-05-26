#include "include/gsplat_quat_scale_to_covar.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/ops.h"

namespace gsplat_core {
namespace {

struct Mat3 {
  float v[9];
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

  mx::array quats = mx::contiguous(input.quats);
  mx::array scales = mx::contiguous(input.scales);
  mx::eval(quats, scales);

  const int n = static_cast<int>(quats.size() / 4);
  const int out_stride = input.triu ? 6 : 9;
  const float* quats_data = quats.data<float>();
  const float* scales_data = scales.data<float>();

  std::vector<float> covars;
  std::vector<float> precis;
  if (input.compute_covar) {
    covars.resize(static_cast<size_t>(n * out_stride), 0.0f);
  }
  if (input.compute_preci) {
    precis.resize(static_cast<size_t>(n * out_stride), 0.0f);
  }

  for (int elem = 0; elem < n; ++elem) {
    const Mat3 r = quat_to_rotmat(quats_data + elem * 4);
    const float* scale = scales_data + elem * 3;
    if (input.compute_covar) {
      const Mat3 covar = covariance_from_rotation_scale(r, scale, false);
      write_matrix(
          covar, input.triu, covars.data() + elem * out_stride);
    }
    if (input.compute_preci) {
      const Mat3 preci = covariance_from_rotation_scale(r, scale, true);
      write_matrix(
          preci, input.triu, precis.data() + elem * out_stride);
    }
  }

  std::vector<mx::array> outputs;
  outputs.reserve(2);
  if (input.compute_covar) {
    outputs.push_back(
        mx::array(covars.begin(), output_shape(quats, input.triu), mx::float32));
  } else {
    outputs.push_back(mx::zeros({0}, mx::float32));
  }
  if (input.compute_preci) {
    outputs.push_back(
        mx::array(precis.begin(), output_shape(quats, input.triu), mx::float32));
  } else {
    outputs.push_back(mx::zeros({0}, mx::float32));
  }
  return outputs;
}

}  // namespace gsplat_core
