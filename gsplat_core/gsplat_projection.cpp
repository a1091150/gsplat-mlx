#include "include/gsplat_projection.h"

#include "include/helper.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace gsplat_core {

namespace {

struct ProjectionKernelParams {
  uint32_t B;
  uint32_t C;
  uint32_t N;
  uint32_t image_width;
  uint32_t image_height;
  float eps2d;
  float near_plane;
  float far_plane;
  float radius_clip;
  uint32_t calc_compensations;
  uint32_t camera_model;
  uint32_t use_covars;
  uint32_t use_opacities;
};

struct Mat3 {
  float v[9];
};

struct Mat2 {
  float v[4];
};

struct Vec3 {
  float v[3];
};

struct ProjectionEval {
  bool valid;
  float means2d[2];
  float depth;
  float conics[3];
  float cov2d[4];
  float compensation;
};

constexpr float kAlphaThreshold = 1.0f / 255.0f;
constexpr float kGaussianExtend = 3.33f;
constexpr float kMinCompensation = 0.005f;

mx::Shape projection_shape(const mx::array& means,
                           const mx::array& viewmats,
                           int tail_a,
                           int tail_b = -1) {
  if (means.ndim() < 2 || viewmats.ndim() < 3) {
    throw std::runtime_error(
        "projection_ewa_3dgs_fused_forward expects means [..., N, 3] "
        "and viewmats [..., C, 4, 4].");
  }
  const int means_ndim = static_cast<int>(means.ndim());
  const int viewmats_ndim = static_cast<int>(viewmats.ndim());
  const int n = means.shape(means_ndim - 2);
  const int c = viewmats.shape(viewmats_ndim - 3);
  mx::Shape shape;
  shape.reserve(static_cast<size_t>(means.ndim()) + 1);
  for (int i = 0; i < means_ndim - 2; ++i) {
    shape.push_back(means.shape(i));
  }
  shape.push_back(c);
  shape.push_back(n);
  shape.push_back(tail_a);
  if (tail_b > 0) {
    shape.push_back(tail_b);
  }
  return shape;
}

mx::Shape projection_scalar_shape(const mx::array& means,
                                  const mx::array& viewmats) {
  if (means.ndim() < 2 || viewmats.ndim() < 3) {
    throw std::runtime_error(
        "projection_ewa_3dgs_fused_forward expects means [..., N, 3] "
        "and viewmats [..., C, 4, 4].");
  }
  const int means_ndim = static_cast<int>(means.ndim());
  const int viewmats_ndim = static_cast<int>(viewmats.ndim());
  const int n = means.shape(means_ndim - 2);
  const int c = viewmats.shape(viewmats_ndim - 3);
  mx::Shape shape;
  shape.reserve(static_cast<size_t>(means.ndim()));
  for (int i = 0; i < means_ndim - 2; ++i) {
    shape.push_back(means.shape(i));
  }
  shape.push_back(c);
  shape.push_back(n);
  return shape;
}

void validate_inputs(const ProjectionEWA3DGSFusedInput& input) {
  const int means_ndim = static_cast<int>(input.means.ndim());
  const int viewmats_ndim = static_cast<int>(input.viewmats.ndim());
  const int ks_ndim = static_cast<int>(input.Ks.ndim());
  if (means_ndim < 2 || input.means.shape(means_ndim - 1) != 3) {
    throw std::runtime_error("means must have shape [..., N, 3].");
  }
  if (viewmats_ndim < 3 || input.viewmats.shape(viewmats_ndim - 1) != 4 ||
      input.viewmats.shape(viewmats_ndim - 2) != 4) {
    throw std::runtime_error("viewmats must have shape [..., C, 4, 4].");
  }
  if (ks_ndim < 3 || input.Ks.shape(ks_ndim - 1) != 3 ||
      input.Ks.shape(ks_ndim - 2) != 3) {
    throw std::runtime_error("Ks must have shape [..., C, 3, 3].");
  }
  if (!input.params.use_covars &&
      (input.quats.size() == 0 || input.scales.size() == 0)) {
    throw std::runtime_error(
        "projection_ewa_3dgs_fused_forward expects covars or quats+scales.");
  }
  if (input.viewspace_points.size() != 0 &&
      input.viewspace_points.shape() !=
          projection_shape(input.means, input.viewmats, 2)) {
    throw std::runtime_error(
        "viewspace_points must have shape [..., C, N, 2].");
  }
}

void validate_backward_inputs(const ProjectionEWA3DGSFusedBackwardInput& input) {
  ProjectionEWA3DGSFusedInput forward_input = {
      .means = input.means,
      .covars = input.covars,
      .quats = input.quats,
      .scales = input.scales,
      .opacities = mx::zeros({0}, mx::float32),
      .viewmats = input.viewmats,
      .Ks = input.Ks,
      .viewspace_points = mx::zeros({0}, mx::float32),
      .s = input.s,
      .params = input.params,
  };
  validate_inputs(forward_input);
  if (input.params.camera_model != 0) {
    throw std::runtime_error(
        "projection_ewa_3dgs_fused_backward currently supports pinhole only.");
  }
  if (input.radii.dtype().val() != mx::int32.val()) {
    throw std::runtime_error("radii must be int32.");
  }
  if (input.v_means2d.shape() !=
          projection_shape(input.means, input.viewmats, 2) ||
      input.v_conics.shape() !=
          projection_shape(input.means, input.viewmats, 3) ||
      input.v_depths.shape() !=
          projection_scalar_shape(input.means, input.viewmats)) {
    throw std::runtime_error("projection backward cotangent shape mismatch.");
  }
}

Mat3 read_covar(const float* covars) {
  return Mat3{{
      covars[0], covars[1], covars[2],
      covars[1], covars[3], covars[4],
      covars[2], covars[4], covars[5],
  }};
}

Mat3 quat_to_rotmat(const float* quat) {
  float w = quat[0];
  float x = quat[1];
  float y = quat[2];
  float z = quat[3];
  const float inv_norm = 1.0f / std::sqrt(w * w + x * x + y * y + z * z);
  w *= inv_norm;
  x *= inv_norm;
  y *= inv_norm;
  z *= inv_norm;
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;
  const float wx = w * x;
  const float wy = w * y;
  const float wz = w * z;
  return Mat3{{
      1.0f - 2.0f * (y2 + z2), 2.0f * (xy - wz), 2.0f * (xz + wy),
      2.0f * (xy + wz), 1.0f - 2.0f * (x2 + z2), 2.0f * (yz - wx),
      2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (x2 + y2),
  }};
}

Mat3 covar_from_quat_scale(const float* quat, const float* scale) {
  const Mat3 r = quat_to_rotmat(quat);
  Mat3 out = {};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      float value = 0.0f;
      for (int k = 0; k < 3; ++k) {
        value += r.v[row * 3 + k] * scale[k] * scale[k] *
                 r.v[col * 3 + k];
      }
      out.v[row * 3 + col] = value;
    }
  }
  return out;
}

Mat3 matmul(const Mat3& a, const Mat3& b) {
  Mat3 out = {};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      for (int k = 0; k < 3; ++k) {
        out.v[row * 3 + col] += a.v[row * 3 + k] * b.v[k * 3 + col];
      }
    }
  }
  return out;
}

Mat3 transpose(const Mat3& a) {
  return Mat3{{
      a.v[0], a.v[3], a.v[6],
      a.v[1], a.v[4], a.v[7],
      a.v[2], a.v[5], a.v[8],
  }};
}

Mat2 mat2_mul(const Mat2& a, const Mat2& b) {
  return Mat2{{
      a.v[0] * b.v[0] + a.v[1] * b.v[2],
      a.v[0] * b.v[1] + a.v[1] * b.v[3],
      a.v[2] * b.v[0] + a.v[3] * b.v[2],
      a.v[2] * b.v[1] + a.v[3] * b.v[3],
  }};
}

Mat2 mat2_transpose(const Mat2& a) {
  return Mat2{{a.v[0], a.v[2], a.v[1], a.v[3]}};
}

Mat2 mat2_scale(const Mat2& a, float scale) {
  return Mat2{{a.v[0] * scale, a.v[1] * scale, a.v[2] * scale, a.v[3] * scale}};
}

Vec3 mat3_vec_mul(const Mat3& a, const Vec3& x) {
  return Vec3{{
      a.v[0] * x.v[0] + a.v[1] * x.v[1] + a.v[2] * x.v[2],
      a.v[3] * x.v[0] + a.v[4] * x.v[1] + a.v[5] * x.v[2],
      a.v[6] * x.v[0] + a.v[7] * x.v[1] + a.v[8] * x.v[2],
  }};
}

Vec3 mat3_transpose_vec_mul(const Mat3& a, const Vec3& x) {
  return Vec3{{
      a.v[0] * x.v[0] + a.v[3] * x.v[1] + a.v[6] * x.v[2],
      a.v[1] * x.v[0] + a.v[4] * x.v[1] + a.v[7] * x.v[2],
      a.v[2] * x.v[0] + a.v[5] * x.v[1] + a.v[8] * x.v[2],
  }};
}

void add_blur_vjp(float eps2d,
                  const Mat2& conic_blur,
                  float compensation,
                  float v_compensation,
                  Mat2& v_covar) {
  const float det_conic_blur =
      conic_blur.v[0] * conic_blur.v[3] - conic_blur.v[1] * conic_blur.v[2];
  const float v_sqr_comp = v_compensation * 0.5f / (compensation + 1.0e-6f);
  const float one_minus_sqr_comp = 1.0f - compensation * compensation;
  v_covar.v[0] +=
      v_sqr_comp * (one_minus_sqr_comp * conic_blur.v[0] -
                    eps2d * det_conic_blur);
  v_covar.v[1] +=
      v_sqr_comp * (one_minus_sqr_comp * conic_blur.v[1]);
  v_covar.v[2] +=
      v_sqr_comp * (one_minus_sqr_comp * conic_blur.v[2]);
  v_covar.v[3] +=
      v_sqr_comp * (one_minus_sqr_comp * conic_blur.v[3] -
                    eps2d * det_conic_blur);
}

void persp_proj_vjp(const Vec3& mean3d,
                    const Mat3& cov3d,
                    float fx,
                    float fy,
                    float cx,
                    float cy,
                    int width,
                    int height,
                    const Mat2& v_cov2d,
                    const float* v_mean2d,
                    Vec3& v_mean3d,
                    Mat3& v_cov3d) {
  const float x = mean3d.v[0];
  const float y = mean3d.v[1];
  const float z = mean3d.v[2];
  const float tan_fovx = 0.5f * static_cast<float>(width) / fx;
  const float tan_fovy = 0.5f * static_cast<float>(height) / fy;
  const float lim_x_pos = (static_cast<float>(width) - cx) / fx + 0.3f * tan_fovx;
  const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
  const float lim_y_pos = (static_cast<float>(height) - cy) / fy + 0.3f * tan_fovy;
  const float lim_y_neg = cy / fy + 0.3f * tan_fovy;
  const float rz = 1.0f / z;
  const float rz2 = rz * rz;
  const float rz3 = rz2 * rz;
  const float x_rz = x * rz;
  const float y_rz = y * rz;
  const float tx = z * std::min(lim_x_pos, std::max(-lim_x_neg, x_rz));
  const float ty = z * std::min(lim_y_pos, std::max(-lim_y_neg, y_rz));

  const float J[6] = {
      fx * rz, 0.0f, -fx * tx * rz2,
      0.0f, fy * rz, -fy * ty * rz2,
  };

  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
          v_cov3d.v[a * 3 + b] +=
              J[r * 3 + a] * v_cov2d.v[r * 2 + c] * J[c * 3 + b];
        }
      }
    }
  }

  v_mean3d.v[0] += fx * rz * v_mean2d[0];
  v_mean3d.v[1] += fy * rz * v_mean2d[1];
  v_mean3d.v[2] +=
      -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2;

  float v_J[6] = {};
  for (int r = 0; r < 2; ++r) {
    for (int a = 0; a < 3; ++a) {
      float left = 0.0f;
      float right = 0.0f;
      for (int c = 0; c < 2; ++c) {
        for (int b = 0; b < 3; ++b) {
          left += v_cov2d.v[r * 2 + c] * J[c * 3 + b] * cov3d.v[a * 3 + b];
          right += v_cov2d.v[c * 2 + r] * J[c * 3 + b] * cov3d.v[b * 3 + a];
        }
      }
      v_J[r * 3 + a] = left + right;
    }
  }

  if (x_rz <= lim_x_pos && x_rz >= -lim_x_neg) {
    v_mean3d.v[0] += -fx * rz2 * v_J[2];
  } else {
    v_mean3d.v[2] += -fx * rz3 * tx * v_J[2];
  }
  if (y_rz <= lim_y_pos && y_rz >= -lim_y_neg) {
    v_mean3d.v[1] += -fy * rz2 * v_J[5];
  } else {
    v_mean3d.v[2] += -fy * rz3 * ty * v_J[5];
  }
  v_mean3d.v[2] += -fx * rz2 * v_J[0] - fy * rz2 * v_J[4] +
                   2.0f * fx * tx * rz3 * v_J[2] +
                   2.0f * fy * ty * rz3 * v_J[5];
}

void pos_w2c_vjp(const Mat3& R,
                 const Vec3& mean_w,
                 const Vec3& v_mean_c,
                 Mat3& v_R,
                 Vec3& v_t,
                 Vec3& v_mean_w) {
  for (int row = 0; row < 3; ++row) {
    v_t.v[row] += v_mean_c.v[row];
    for (int col = 0; col < 3; ++col) {
      v_R.v[row * 3 + col] += v_mean_c.v[row] * mean_w.v[col];
    }
  }
  Vec3 v_mean = mat3_transpose_vec_mul(R, v_mean_c);
  for (int i = 0; i < 3; ++i) {
    v_mean_w.v[i] += v_mean.v[i];
  }
}

void covar_w2c_vjp(const Mat3& R,
                   const Mat3& covar_w,
                   const Mat3& v_covar_c,
                   Mat3& v_R,
                   Mat3& v_covar_w) {
  Mat3 term1 = matmul(matmul(v_covar_c, R), transpose(covar_w));
  Mat3 term2 = matmul(matmul(transpose(v_covar_c), R), covar_w);
  Mat3 covar_grad = matmul(matmul(transpose(R), v_covar_c), R);
  for (int i = 0; i < 9; ++i) {
    v_R.v[i] += term1.v[i] + term2.v[i];
    v_covar_w.v[i] += covar_grad.v[i];
  }
}

void quat_to_rotmat_vjp(const float* quat, const Mat3& v_R, float* v_quat) {
  float w = quat[0];
  float x = quat[1];
  float y = quat[2];
  float z = quat[3];
  const float inv_norm = 1.0f / std::sqrt(w * w + x * x + y * y + z * z);
  w *= inv_norm;
  x *= inv_norm;
  y *= inv_norm;
  z *= inv_norm;

  const float* g = v_R.v;
  float v_quat_n[4] = {
      2.0f * (x * (g[7] - g[5]) + y * (g[2] - g[6]) +
              z * (g[3] - g[1])),
      2.0f * (-2.0f * x * (g[4] + g[8]) + y * (g[1] + g[3]) +
              z * (g[2] + g[6]) + w * (g[7] - g[5])),
      2.0f * (x * (g[1] + g[3]) - 2.0f * y * (g[0] + g[8]) +
              z * (g[5] + g[7]) + w * (g[2] - g[6])),
      2.0f * (x * (g[2] + g[6]) + y * (g[5] + g[7]) -
              2.0f * z * (g[0] + g[4]) + w * (g[3] - g[1])),
  };
  const float quat_n[4] = {w, x, y, z};
  float dot = 0.0f;
  for (int i = 0; i < 4; ++i) {
    dot += v_quat_n[i] * quat_n[i];
  }
  for (int i = 0; i < 4; ++i) {
    v_quat[i] += (v_quat_n[i] - dot * quat_n[i]) * inv_norm;
  }
}

void quat_scale_to_covar_vjp(const float* quat,
                             const float* scale,
                             const Mat3& v_covar,
                             float* v_quat,
                             float* v_scale) {
  Mat3 R = quat_to_rotmat(quat);
  Mat3 M = {};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      M.v[row * 3 + col] = R.v[row * 3 + col] * scale[col];
    }
  }
  Mat3 v_M = matmul(Mat3{{
                       v_covar.v[0] + v_covar.v[0],
                       v_covar.v[1] + v_covar.v[3],
                       v_covar.v[2] + v_covar.v[6],
                       v_covar.v[3] + v_covar.v[1],
                       v_covar.v[4] + v_covar.v[4],
                       v_covar.v[5] + v_covar.v[7],
                       v_covar.v[6] + v_covar.v[2],
                       v_covar.v[7] + v_covar.v[5],
                       v_covar.v[8] + v_covar.v[8],
                   }},
                   M);
  Mat3 v_R = {};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      v_R.v[row * 3 + col] = v_M.v[row * 3 + col] * scale[col];
      v_scale[col] += v_M.v[row * 3 + col] * R.v[row * 3 + col];
    }
  }
  quat_to_rotmat_vjp(quat, v_R, v_quat);
}

ProjectionEval projection_eval_one(const float* mean_w,
                                   const float* covar_or_null,
                                   const float* quat_or_null,
                                   const float* scale_or_null,
                                   const float* viewmat,
                                   const float* K,
                                   const ProjectionEWA3DGSFusedParams& params) {
  Mat3 R = Mat3{{
      viewmat[0], viewmat[1], viewmat[2],
      viewmat[4], viewmat[5], viewmat[6],
      viewmat[8], viewmat[9], viewmat[10],
  }};
  const float t[3] = {viewmat[3], viewmat[7], viewmat[11]};
  float mean_c[3] = {
      R.v[0] * mean_w[0] + R.v[1] * mean_w[1] + R.v[2] * mean_w[2] + t[0],
      R.v[3] * mean_w[0] + R.v[4] * mean_w[1] + R.v[5] * mean_w[2] + t[1],
      R.v[6] * mean_w[0] + R.v[7] * mean_w[1] + R.v[8] * mean_w[2] + t[2],
  };
  if (mean_c[2] < params.near_plane || mean_c[2] > params.far_plane) {
    return ProjectionEval{};
  }

  Mat3 covar_w = params.use_covars
                     ? read_covar(covar_or_null)
                     : covar_from_quat_scale(quat_or_null, scale_or_null);
  Mat3 covar_c = matmul(matmul(R, covar_w), transpose(R));

  const float fx = K[0];
  const float fy = K[4];
  const float cx = K[2];
  const float cy = K[5];
  const float x = mean_c[0];
  const float y = mean_c[1];
  const float z = mean_c[2];
  const float rz = 1.0f / z;
  const float rz2 = rz * rz;
  const float tan_fovx = 0.5f * static_cast<float>(params.image_width) / fx;
  const float tan_fovy = 0.5f * static_cast<float>(params.image_height) / fy;
  const float lim_x_pos =
      (static_cast<float>(params.image_width) - cx) / fx + 0.3f * tan_fovx;
  const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
  const float lim_y_pos =
      (static_cast<float>(params.image_height) - cy) / fy + 0.3f * tan_fovy;
  const float lim_y_neg = cy / fy + 0.3f * tan_fovy;
  const float tx = z * std::min(lim_x_pos, std::max(-lim_x_neg, x * rz));
  const float ty = z * std::min(lim_y_pos, std::max(-lim_y_neg, y * rz));
  const float J[6] = {fx * rz, 0.0f, -fx * tx * rz2,
                      0.0f, fy * rz, -fy * ty * rz2};
  float cov2d[4] = {};
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          cov2d[row * 2 + col] += J[row * 3 + a] *
                                  covar_c.v[a * 3 + b] *
                                  J[col * 3 + b];
        }
      }
    }
  }
  const float det_orig = cov2d[0] * cov2d[3] - cov2d[1] * cov2d[2];
  cov2d[0] += params.eps2d;
  cov2d[3] += params.eps2d;
  const float det = cov2d[0] * cov2d[3] - cov2d[1] * cov2d[2];
  if (det <= 0.0f) {
    return ProjectionEval{};
  }

  const float inv_det = 1.0f / det;
  ProjectionEval out = {};
  out.valid = true;
  out.means2d[0] = fx * x * rz + cx;
  out.means2d[1] = fy * y * rz + cy;
  out.depth = z;
  out.conics[0] = cov2d[3] * inv_det;
  out.conics[1] = -cov2d[1] * inv_det;
  out.conics[2] = cov2d[0] * inv_det;
  out.cov2d[0] = cov2d[0];
  out.cov2d[1] = cov2d[1];
  out.cov2d[2] = cov2d[2];
  out.cov2d[3] = cov2d[3];
  out.compensation =
      std::sqrt(std::max(kMinCompensation * kMinCompensation, det_orig / det));
  return out;
}

float projection_loss_one(const ProjectionEval& eval,
                          const float* v_means2d,
                          float v_depth,
                          const float* v_conics,
                          const float* v_compensation_or_null) {
  if (!eval.valid) {
    return 0.0f;
  }
  float loss = eval.means2d[0] * v_means2d[0] +
               eval.means2d[1] * v_means2d[1] +
               eval.depth * v_depth +
               eval.conics[0] * v_conics[0] +
               eval.conics[1] * v_conics[1] +
               eval.conics[2] * v_conics[2];
  if (v_compensation_or_null != nullptr) {
    loss += eval.compensation * (*v_compensation_or_null);
  }
  return loss;
}

}  // namespace

std::vector<mx::array> gsplat_projection_ewa_3dgs_fused(
    const ProjectionEWA3DGSFusedInput& input) {
  validate_inputs(input);

  auto prim = std::make_shared<GSPlatProjectionEWA3DGSFused>(
      to_stream(input.s), input.params);

  const auto scalar_shape = projection_scalar_shape(input.means, input.viewmats);
  std::vector<mx::Shape> output_shapes = {
      projection_shape(input.means, input.viewmats, 2),
      scalar_shape,
      projection_shape(input.means, input.viewmats, 3),
      input.params.calc_compensations ? scalar_shape : mx::Shape{0},
      projection_shape(input.means, input.viewmats, 2),
  };
  std::vector<mx::Dtype> output_types = {
      input.means.dtype(),
      input.means.dtype(),
      input.means.dtype(),
      input.means.dtype(),
      mx::int32,
  };

  std::vector<mx::array> inputs = {
      mx::contiguous(input.means),
      mx::contiguous(input.covars),
      mx::contiguous(input.quats),
      mx::contiguous(input.scales),
      mx::contiguous(input.opacities),
      mx::contiguous(input.viewmats),
      mx::contiguous(input.Ks),
      mx::contiguous(input.viewspace_points),
  };
  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

std::vector<mx::array> gsplat_projection_ewa_3dgs_fused_backward(
    const ProjectionEWA3DGSFusedBackwardInput& input) {
  validate_backward_inputs(input);

  auto prim = std::make_shared<GSPlatProjectionEWA3DGSFusedBackward>(
      to_stream(input.s), input.params, input.viewmats_requires_grad);

  std::vector<mx::Shape> output_shapes = {
      input.means.shape(),
      input.params.use_covars ? input.covars.shape() : mx::Shape{0},
      input.params.use_covars ? mx::Shape{0} : input.quats.shape(),
      input.params.use_covars ? mx::Shape{0} : input.scales.shape(),
      input.viewmats_requires_grad ? input.viewmats.shape() : mx::Shape{0},
  };
  std::vector<mx::Dtype> output_types = {
      mx::float32, mx::float32, mx::float32, mx::float32, mx::float32};
  std::vector<mx::array> inputs = {
      mx::contiguous(input.means),
      mx::contiguous(input.covars),
      mx::contiguous(input.quats),
      mx::contiguous(input.scales),
      mx::contiguous(input.viewmats),
      mx::contiguous(input.Ks),
      mx::contiguous(input.radii),
      mx::contiguous(input.conics),
      mx::contiguous(input.compensations),
      mx::contiguous(input.v_means2d),
      mx::contiguous(input.v_depths),
      mx::contiguous(input.v_conics),
      mx::contiguous(input.v_compensations),
  };
  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

#ifdef _METAL_
void GSPlatProjectionEWA3DGSFused::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  const auto& means = inputs[0];
  const auto& covars = inputs[1];
  const auto& quats = inputs[2];
  const auto& scales = inputs[3];
  const auto& opacities = inputs[4];
  const auto& viewmats = inputs[5];
  const auto& Ks = inputs[6];

  auto& radii = outputs[kRadii];
  auto& means2d = outputs[kMeans2D];
  auto& depths = outputs[kDepths];
  auto& conics = outputs[kConics];
  auto& compensations = outputs[kCompensations];

  const int means_ndim = static_cast<int>(means.ndim());
  const int viewmats_ndim = static_cast<int>(viewmats.ndim());
  const uint32_t n = static_cast<uint32_t>(means.shape(means_ndim - 2));
  const uint32_t c = static_cast<uint32_t>(viewmats.shape(viewmats_ndim - 3));
  const uint32_t b = static_cast<uint32_t>(means.size() / (n * 3));
  const uint32_t numel = b * c * n;

  if (numel == 0) {
    return;
  }

  ProjectionKernelParams kernel_params = {
      .B = b,
      .C = c,
      .N = n,
      .image_width = static_cast<uint32_t>(params_.image_width),
      .image_height = static_cast<uint32_t>(params_.image_height),
      .eps2d = params_.eps2d,
      .near_plane = params_.near_plane,
      .far_plane = params_.far_plane,
      .radius_clip = params_.radius_clip,
      .calc_compensations = static_cast<uint32_t>(params_.calc_compensations),
      .camera_model = static_cast<uint32_t>(params_.camera_model),
      .use_covars = static_cast<uint32_t>(params_.use_covars),
      .use_opacities = static_cast<uint32_t>(params_.use_opacities),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("gsplat_core", current_binary_dir());
  auto kernel = d.get_kernel("gsplat_projection_ewa_3dgs_fused_forward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(means, 1);
  compute_encoder.set_input_array(covars, 2);
  compute_encoder.set_input_array(quats, 3);
  compute_encoder.set_input_array(scales, 4);
  compute_encoder.set_input_array(opacities, 5);
  compute_encoder.set_input_array(viewmats, 6);
  compute_encoder.set_input_array(Ks, 7);
  compute_encoder.set_output_array(radii, 8);
  compute_encoder.set_output_array(means2d, 9);
  compute_encoder.set_output_array(depths, 10);
  compute_encoder.set_output_array(conics, 11);
  compute_encoder.set_output_array(compensations, 12);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  const size_t tgp_size = std::min(static_cast<size_t>(numel), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(numel, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}

void GSPlatProjectionEWA3DGSFusedBackward::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  if (!params_.use_covars) {
    throw std::runtime_error(
        "GSPlatProjectionEWA3DGSFusedBackward GPU path currently supports covars only.");
  }
  if (params_.camera_model != 0) {
    throw std::runtime_error(
        "GSPlatProjectionEWA3DGSFusedBackward GPU path currently supports pinhole only.");
  }
  if (viewmats_requires_grad_) {
    throw std::runtime_error(
        "GSPlatProjectionEWA3DGSFusedBackward GPU path does not support viewmats gradients yet.");
  }

  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  const auto& means = inputs[0];
  const auto& covars = inputs[1];
  const auto& viewmats = inputs[4];
  const auto& Ks = inputs[5];
  const auto& radii = inputs[6];
  const auto& conics = inputs[7];
  const auto& compensations = inputs[8];
  const auto& v_means2d = inputs[9];
  const auto& v_depths = inputs[10];
  const auto& v_conics = inputs[11];
  const auto& v_compensations = inputs[12];

  auto& v_means = outputs[kProjectionVMeans];
  auto& v_covars = outputs[kProjectionVCovars];

  const int means_ndim = static_cast<int>(means.ndim());
  const int viewmats_ndim = static_cast<int>(viewmats.ndim());
  const uint32_t n = static_cast<uint32_t>(means.shape(means_ndim - 2));
  const uint32_t c = static_cast<uint32_t>(viewmats.shape(viewmats_ndim - 3));
  const uint32_t b = static_cast<uint32_t>(means.size() / (n * 3));
  const uint32_t numel = b * n;
  if (numel == 0) {
    return;
  }

  ProjectionKernelParams kernel_params = {
      .B = b,
      .C = c,
      .N = n,
      .image_width = static_cast<uint32_t>(params_.image_width),
      .image_height = static_cast<uint32_t>(params_.image_height),
      .eps2d = params_.eps2d,
      .near_plane = params_.near_plane,
      .far_plane = params_.far_plane,
      .radius_clip = params_.radius_clip,
      .calc_compensations = static_cast<uint32_t>(params_.calc_compensations),
      .camera_model = static_cast<uint32_t>(params_.camera_model),
      .use_covars = static_cast<uint32_t>(params_.use_covars),
      .use_opacities = static_cast<uint32_t>(params_.use_opacities),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("gsplat_core", current_binary_dir());
  auto kernel =
      d.get_kernel("gsplat_projection_ewa_3dgs_fused_backward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(means, 1);
  compute_encoder.set_input_array(covars, 2);
  compute_encoder.set_input_array(viewmats, 3);
  compute_encoder.set_input_array(Ks, 4);
  compute_encoder.set_input_array(radii, 5);
  compute_encoder.set_input_array(conics, 6);
  compute_encoder.set_input_array(compensations, 7);
  compute_encoder.set_input_array(v_means2d, 8);
  compute_encoder.set_input_array(v_depths, 9);
  compute_encoder.set_input_array(v_conics, 10);
  compute_encoder.set_input_array(v_compensations, 11);
  compute_encoder.set_output_array(v_means, 12);
  compute_encoder.set_output_array(v_covars, 13);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  const size_t tgp_size = std::min(static_cast<size_t>(numel), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(numel, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}
#else
void GSPlatProjectionEWA3DGSFused::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatProjectionEWA3DGSFused has no GPU implementation.");
}

void GSPlatProjectionEWA3DGSFusedBackward::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatProjectionEWA3DGSFusedBackward has no GPU implementation.");
}
#endif

void GSPlatProjectionEWA3DGSFused::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& means = inputs[0];
  const auto& covars = inputs[1];
  const auto& quats = inputs[2];
  const auto& scales = inputs[3];
  const auto& opacities = inputs[4];
  const auto& viewmats = inputs[5];
  const auto& Ks = inputs[6];
  const auto& viewspace_points = inputs[7];
  mx::eval(means, covars, quats, scales, opacities, viewmats, Ks,
           viewspace_points);

  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  const int means_ndim = static_cast<int>(means.ndim());
  const int viewmats_ndim = static_cast<int>(viewmats.ndim());
  const int N = means.shape(means_ndim - 2);
  const int C = viewmats.shape(viewmats_ndim - 3);
  const int B = static_cast<int>(means.size() / (N * 3));
  const float* means_data = means.data<float>();
  const float* covars_data = params_.use_covars ? covars.data<float>() : nullptr;
  const float* quats_data = params_.use_covars ? nullptr : quats.data<float>();
  const float* scales_data = params_.use_covars ? nullptr : scales.data<float>();
  const float* opacities_data =
      params_.use_opacities ? opacities.data<float>() : nullptr;
  const float* viewmats_data = viewmats.data<float>();
  const float* Ks_data = Ks.data<float>();
  int32_t* radii_data = outputs[kRadii].data<int32_t>();
  float* means2d_data = outputs[kMeans2D].data<float>();
  float* depths_data = outputs[kDepths].data<float>();
  float* conics_data = outputs[kConics].data<float>();
  float* compensations_data =
      params_.calc_compensations
          ? outputs[kCompensations].data<float>()
          : nullptr;

  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      for (int n = 0; n < N; ++n) {
        const int idx = (b * C + c) * N + n;
        const int gaussian_off = b * N + n;
        ProjectionEval eval = projection_eval_one(
            means_data + b * N * 3 + n * 3,
            covars_data == nullptr ? nullptr : covars_data + gaussian_off * 6,
            quats_data == nullptr ? nullptr : quats_data + gaussian_off * 4,
            scales_data == nullptr ? nullptr : scales_data + gaussian_off * 3,
            viewmats_data + b * C * 16 + c * 16,
            Ks_data + b * C * 9 + c * 9,
            params_);
        if (!eval.valid) {
          continue;
        }
        float extend = kGaussianExtend;
        if (params_.use_opacities) {
          float opacity = opacities_data[gaussian_off];
          if (params_.calc_compensations) {
            opacity *= eval.compensation;
          }
          if (opacity < kAlphaThreshold) {
            continue;
          }
          extend = std::min(
              kGaussianExtend,
              std::sqrt(2.0f * std::log(opacity / kAlphaThreshold)));
        }
        const int radius_x =
            static_cast<int>(std::ceil(extend * std::sqrt(eval.cov2d[0])));
        const int radius_y =
            static_cast<int>(std::ceil(extend * std::sqrt(eval.cov2d[3])));
        if (radius_x <= params_.radius_clip || radius_y <= params_.radius_clip) {
          continue;
        }
        radii_data[idx * 2] = radius_x;
        radii_data[idx * 2 + 1] = radius_y;
        means2d_data[idx * 2] = eval.means2d[0];
        means2d_data[idx * 2 + 1] = eval.means2d[1];
        depths_data[idx] = eval.depth;
        conics_data[idx * 3] = eval.conics[0];
        conics_data[idx * 3 + 1] = eval.conics[1];
        conics_data[idx * 3 + 2] = eval.conics[2];
        if (compensations_data != nullptr) {
          compensations_data[idx] = eval.compensation;
        }
      }
    }
  }
}

void GSPlatProjectionEWA3DGSFusedBackward::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& means = inputs[0];
  const auto& covars = inputs[1];
  const auto& quats = inputs[2];
  const auto& scales = inputs[3];
  const auto& viewmats = inputs[4];
  const auto& Ks = inputs[5];
  const auto& radii = inputs[6];
  const auto& conics = inputs[7];
  const auto& compensations = inputs[8];
  const auto& v_means2d = inputs[9];
  const auto& v_depths = inputs[10];
  const auto& v_conics = inputs[11];
  const auto& v_compensations = inputs[12];
  mx::eval(means, covars, quats, scales, viewmats, Ks, radii, conics,
           compensations, v_means2d, v_depths, v_conics, v_compensations);

  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  const int means_ndim = static_cast<int>(means.ndim());
  const int viewmats_ndim = static_cast<int>(viewmats.ndim());
  const int N = means.shape(means_ndim - 2);
  const int C = viewmats.shape(viewmats_ndim - 3);
  const int B = static_cast<int>(means.size() / (N * 3));
  const float* means_data = means.data<float>();
  const float* covars_data = params_.use_covars ? covars.data<float>() : nullptr;
  const float* quats_data = params_.use_covars ? nullptr : quats.data<float>();
  const float* scales_data = params_.use_covars ? nullptr : scales.data<float>();
  const float* viewmats_data = viewmats.data<float>();
  const float* Ks_data = Ks.data<float>();
  const int32_t* radii_data = radii.data<int32_t>();
  const float* conics_data = conics.data<float>();
  const float* v_means2d_data = v_means2d.data<float>();
  const float* v_depths_data = v_depths.data<float>();
  const float* v_conics_data = v_conics.data<float>();
  const float* v_compensations_data =
      v_compensations.size() == 0 ? nullptr : v_compensations.data<float>();

  float* v_means_data = outputs[kProjectionVMeans].data<float>();
  float* v_covars_data =
      params_.use_covars ? outputs[kProjectionVCovars].data<float>() : nullptr;
  float* v_quats_data =
      params_.use_covars ? nullptr : outputs[kProjectionVQuats].data<float>();
  float* v_scales_data =
      params_.use_covars ? nullptr : outputs[kProjectionVScales].data<float>();
  float* v_viewmats_data = viewmats_requires_grad_
                               ? outputs[kProjectionVViewmats].data<float>()
                               : nullptr;
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      for (int n = 0; n < N; ++n) {
        const int idx = (b * C + c) * N + n;
        if (radii_data[idx * 2] <= 0 || radii_data[idx * 2 + 1] <= 0) {
          continue;
        }
        const int mean_off = b * N * 3 + n * 3;
        const int gaussian_off = b * N + n;
        const int view_off = b * C * 16 + c * 16;
        const int k_off = b * C * 9 + c * 9;
        const float* v_mean2d = v_means2d_data + idx * 2;
        const float* v_conic = v_conics_data + idx * 3;
        const float* view = viewmats_data + view_off;
        const float* K = Ks_data + k_off;
        Mat2 covar2d_inv = Mat2{{
            conics_data[idx * 3],
            conics_data[idx * 3 + 1],
            conics_data[idx * 3 + 1],
            conics_data[idx * 3 + 2],
        }};
        Mat2 v_covar2d_inv = Mat2{{
            v_conic[0],
            0.5f * v_conic[1],
            0.5f * v_conic[1],
            v_conic[2],
        }};
        Mat2 v_covar2d =
            mat2_scale(mat2_mul(mat2_mul(covar2d_inv, v_covar2d_inv),
                                covar2d_inv),
                       -1.0f);
        if (v_compensations_data != nullptr) {
          add_blur_vjp(params_.eps2d, covar2d_inv, compensations.data<float>()[idx],
                       v_compensations_data[idx], v_covar2d);
        }

        Mat3 R = Mat3{{
            view[0], view[1], view[2],
            view[4], view[5], view[6],
            view[8], view[9], view[10],
        }};
        Vec3 mean_w = Vec3{{
            means_data[mean_off],
            means_data[mean_off + 1],
            means_data[mean_off + 2],
        }};
        Vec3 t = Vec3{{view[3], view[7], view[11]}};
        Vec3 mean_c = mat3_vec_mul(R, mean_w);
        for (int i = 0; i < 3; ++i) {
          mean_c.v[i] += t.v[i];
        }

        Mat3 covar_w =
            params_.use_covars
                ? read_covar(covars_data + gaussian_off * 6)
                : covar_from_quat_scale(quats_data + gaussian_off * 4,
                                        scales_data + gaussian_off * 3);
        Mat3 covar_c = matmul(matmul(R, covar_w), transpose(R));

        Vec3 v_mean_c = Vec3{{0.0f, 0.0f, v_depths_data[idx]}};
        Mat3 v_covar_c = {};
        persp_proj_vjp(mean_c, covar_c, K[0], K[4], K[2], K[5],
                       params_.image_width, params_.image_height, v_covar2d,
                       v_mean2d, v_mean_c, v_covar_c);

        Vec3 v_mean_w = {};
        Vec3 v_t = {};
        Mat3 v_R = {};
        Mat3 v_covar_w = {};
        pos_w2c_vjp(R, mean_w, v_mean_c, v_R, v_t, v_mean_w);
        covar_w2c_vjp(R, covar_w, v_covar_c, v_R, v_covar_w);

        for (int axis = 0; axis < 3; ++axis) {
          v_means_data[mean_off + axis] += v_mean_w.v[axis];
        }
        if (params_.use_covars) {
          const int cov_off = gaussian_off * 6;
          v_covars_data[cov_off] += v_covar_w.v[0];
          v_covars_data[cov_off + 1] += v_covar_w.v[1] + v_covar_w.v[3];
          v_covars_data[cov_off + 2] += v_covar_w.v[2] + v_covar_w.v[6];
          v_covars_data[cov_off + 3] += v_covar_w.v[4];
          v_covars_data[cov_off + 4] += v_covar_w.v[5] + v_covar_w.v[7];
          v_covars_data[cov_off + 5] += v_covar_w.v[8];
        } else {
          const int quat_off = gaussian_off * 4;
          const int scale_off = gaussian_off * 3;
          quat_scale_to_covar_vjp(
              quats_data + quat_off, scales_data + scale_off, v_covar_w,
              v_quats_data + quat_off, v_scales_data + scale_off);
        }

        if (v_viewmats_data != nullptr) {
          for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
              v_viewmats_data[view_off + row * 4 + col] +=
                  v_R.v[row * 3 + col];
            }
            v_viewmats_data[view_off + row * 4 + 3] += v_t.v[row];
          }
        }
      }
    }
  }
}

std::vector<mx::array> GSPlatProjectionEWA3DGSFused::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("GSPlatProjectionEWA3DGSFused jvp is not implemented.");
}

std::vector<mx::array> GSPlatProjectionEWA3DGSFused::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
  if (cotangents.size() < (params_.calc_compensations ? 4 : 3)) {
    throw std::runtime_error(
        "GSPlatProjectionEWA3DGSFused vjp expects projection cotangents.");
  }
  if (outputs.size() < (params_.calc_compensations ? 4 : 3) ||
      outputs.size() <= kRadii) {
    throw std::runtime_error(
        "GSPlatProjectionEWA3DGSFused vjp expects projection forward outputs.");
  }
  if (params_.camera_model != 0) {
    throw std::runtime_error(
        "GSPlatProjectionEWA3DGSFused vjp currently supports pinhole only.");
  }

  const bool needs_viewmats =
      std::find(argnums.begin(), argnums.end(), 5) != argnums.end();
  ProjectionEWA3DGSFusedBackwardInput input = {
      .means = primals[0],
      .covars = primals[1],
      .quats = primals[2],
      .scales = primals[3],
      .viewmats = primals[5],
      .Ks = primals[6],
      .radii = outputs[kRadii],
      .conics = outputs[kConics],
      .compensations = params_.calc_compensations
                           ? outputs[kCompensations]
                           : mx::zeros({0}, mx::float32),
      .v_means2d = cotangents[kMeans2D],
      .v_depths = cotangents[kDepths],
      .v_conics = cotangents[kConics],
      .v_compensations = params_.calc_compensations
                             ? cotangents[kCompensations]
                             : mx::zeros({0}, mx::float32),
      .s = mx::Device::cpu,
      .params = params_,
      .viewmats_requires_grad = needs_viewmats,
  };
  auto backward_outputs = gsplat_projection_ewa_3dgs_fused_backward(input);
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  for (int argnum : argnums) {
    if (argnum == 0) {
      vjps.push_back(backward_outputs[kProjectionVMeans]);
    } else if (argnum == 1) {
      vjps.push_back(params_.use_covars
                         ? backward_outputs[kProjectionVCovars]
                         : mx::zeros_like(primals[1]));
    } else if (argnum == 2) {
      vjps.push_back(params_.use_covars
                         ? mx::zeros_like(primals[2])
                         : backward_outputs[kProjectionVQuats]);
    } else if (argnum == 3) {
      vjps.push_back(params_.use_covars
                         ? mx::zeros_like(primals[3])
                         : backward_outputs[kProjectionVScales]);
    } else if (argnum == 4 || argnum == 6) {
      vjps.push_back(mx::zeros_like(primals[argnum]));
    } else if (argnum == 5) {
      vjps.push_back(backward_outputs[kProjectionVViewmats]);
    } else if (argnum == 7) {
      vjps.push_back(cotangents[kMeans2D]);
    } else {
      throw std::runtime_error(
          "GSPlatProjectionEWA3DGSFused vjp received an unknown argnum.");
    }
  }
  return vjps;
}

std::pair<std::vector<mx::array>, std::vector<int>>
GSPlatProjectionEWA3DGSFused::vmap(const std::vector<mx::array>&,
                                   const std::vector<int>&) {
  throw std::runtime_error("GSPlatProjectionEWA3DGSFused vmap is not implemented.");
}

bool GSPlatProjectionEWA3DGSFused::is_equivalent(
    const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  const auto* other_ptr =
      dynamic_cast<const GSPlatProjectionEWA3DGSFused*>(&other);
  if (!other_ptr) {
    return false;
  }
  return params_.image_width == other_ptr->params_.image_width &&
         params_.image_height == other_ptr->params_.image_height &&
         params_.eps2d == other_ptr->params_.eps2d &&
         params_.near_plane == other_ptr->params_.near_plane &&
         params_.far_plane == other_ptr->params_.far_plane &&
         params_.radius_clip == other_ptr->params_.radius_clip &&
         params_.calc_compensations == other_ptr->params_.calc_compensations &&
         params_.camera_model == other_ptr->params_.camera_model &&
         params_.use_covars == other_ptr->params_.use_covars &&
         params_.use_opacities == other_ptr->params_.use_opacities;
}

std::vector<mx::array> GSPlatProjectionEWA3DGSFusedBackward::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "GSPlatProjectionEWA3DGSFusedBackward jvp is not implemented.");
}

std::vector<mx::array> GSPlatProjectionEWA3DGSFusedBackward::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatProjectionEWA3DGSFusedBackward vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GSPlatProjectionEWA3DGSFusedBackward::vmap(const std::vector<mx::array>&,
                                           const std::vector<int>&) {
  throw std::runtime_error(
      "GSPlatProjectionEWA3DGSFusedBackward vmap is not implemented.");
}

bool GSPlatProjectionEWA3DGSFusedBackward::is_equivalent(
    const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  const auto* other_ptr =
      dynamic_cast<const GSPlatProjectionEWA3DGSFusedBackward*>(&other);
  if (!other_ptr) {
    return false;
  }
  return params_.image_width == other_ptr->params_.image_width &&
         params_.image_height == other_ptr->params_.image_height &&
         params_.eps2d == other_ptr->params_.eps2d &&
         params_.near_plane == other_ptr->params_.near_plane &&
         params_.far_plane == other_ptr->params_.far_plane &&
         params_.radius_clip == other_ptr->params_.radius_clip &&
         params_.calc_compensations == other_ptr->params_.calc_compensations &&
         params_.camera_model == other_ptr->params_.camera_model &&
         params_.use_covars == other_ptr->params_.use_covars &&
         params_.use_opacities == other_ptr->params_.use_opacities &&
         viewmats_requires_grad_ == other_ptr->viewmats_requires_grad_;
}

}  // namespace gsplat_core
