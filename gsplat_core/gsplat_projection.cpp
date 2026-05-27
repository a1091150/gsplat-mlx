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

struct ProjectionEval {
  bool valid;
  float means2d[2];
  float depth;
  float conics[3];
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
      projection_shape(input.means, input.viewmats, 2),
      scalar_shape,
      projection_shape(input.means, input.viewmats, 3),
      input.params.calc_compensations ? scalar_shape : mx::Shape{0},
  };
  std::vector<mx::Dtype> output_types = {
      mx::int32,
      input.means.dtype(),
      input.means.dtype(),
      input.means.dtype(),
      input.means.dtype(),
  };

  std::vector<mx::array> inputs = {
      mx::contiguous(input.means),
      mx::contiguous(input.covars),
      mx::contiguous(input.quats),
      mx::contiguous(input.scales),
      mx::contiguous(input.opacities),
      mx::contiguous(input.viewmats),
      mx::contiguous(input.Ks),
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
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatProjectionEWA3DGSFusedBackward GPU finite-difference path is not implemented yet.");
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
    const std::vector<mx::array>&,
    std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
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
  const auto& compensations = inputs[8];
  const auto& v_means2d = inputs[9];
  const auto& v_depths = inputs[10];
  const auto& v_conics = inputs[11];
  const auto& v_compensations = inputs[12];
  mx::eval(means, covars, quats, scales, viewmats, Ks, radii,
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
  constexpr float eps = 1.0e-3f;

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
        const float* v_comp =
            v_compensations_data == nullptr ? nullptr : v_compensations_data + idx;

        auto loss = [&](const float* mean,
                        const float* covar,
                        const float* quat,
                        const float* scale,
                        const float* viewmat) {
          return projection_loss_one(
              projection_eval_one(mean, covar, quat, scale, viewmat,
                                  Ks_data + k_off, params_),
              v_mean2d, v_depths_data[idx], v_conic, v_comp);
        };

        for (int axis = 0; axis < 3; ++axis) {
          float plus_mean[3] = {means_data[mean_off],
                                means_data[mean_off + 1],
                                means_data[mean_off + 2]};
          float minus_mean[3] = {plus_mean[0], plus_mean[1], plus_mean[2]};
          plus_mean[axis] += eps;
          minus_mean[axis] -= eps;
          v_means_data[mean_off + axis] +=
              (loss(plus_mean,
                    covars_data == nullptr ? nullptr : covars_data + gaussian_off * 6,
                    quats_data == nullptr ? nullptr : quats_data + gaussian_off * 4,
                    scales_data == nullptr ? nullptr : scales_data + gaussian_off * 3,
                    viewmats_data + view_off) -
               loss(minus_mean,
                    covars_data == nullptr ? nullptr : covars_data + gaussian_off * 6,
                    quats_data == nullptr ? nullptr : quats_data + gaussian_off * 4,
                    scales_data == nullptr ? nullptr : scales_data + gaussian_off * 3,
                    viewmats_data + view_off)) /
              (2.0f * eps);
        }

        if (params_.use_covars) {
          const int cov_off = gaussian_off * 6;
          for (int axis = 0; axis < 6; ++axis) {
            float plus_covar[6];
            float minus_covar[6];
            for (int i = 0; i < 6; ++i) {
              plus_covar[i] = covars_data[cov_off + i];
              minus_covar[i] = covars_data[cov_off + i];
            }
            plus_covar[axis] += eps;
            minus_covar[axis] -= eps;
            v_covars_data[cov_off + axis] +=
                (loss(means_data + mean_off, plus_covar, nullptr, nullptr,
                      viewmats_data + view_off) -
                 loss(means_data + mean_off, minus_covar, nullptr, nullptr,
                      viewmats_data + view_off)) /
                (2.0f * eps);
          }
        } else {
          const int quat_off = gaussian_off * 4;
          const int scale_off = gaussian_off * 3;
          for (int axis = 0; axis < 4; ++axis) {
            float plus_quat[4];
            float minus_quat[4];
            for (int i = 0; i < 4; ++i) {
              plus_quat[i] = quats_data[quat_off + i];
              minus_quat[i] = quats_data[quat_off + i];
            }
            plus_quat[axis] += eps;
            minus_quat[axis] -= eps;
            v_quats_data[quat_off + axis] +=
                (loss(means_data + mean_off, nullptr, plus_quat,
                      scales_data + scale_off, viewmats_data + view_off) -
                 loss(means_data + mean_off, nullptr, minus_quat,
                      scales_data + scale_off, viewmats_data + view_off)) /
                (2.0f * eps);
          }
          for (int axis = 0; axis < 3; ++axis) {
            float plus_scale[3];
            float minus_scale[3];
            for (int i = 0; i < 3; ++i) {
              plus_scale[i] = scales_data[scale_off + i];
              minus_scale[i] = scales_data[scale_off + i];
            }
            plus_scale[axis] += eps;
            minus_scale[axis] -= eps;
            v_scales_data[scale_off + axis] +=
                (loss(means_data + mean_off, nullptr, quats_data + quat_off,
                      plus_scale, viewmats_data + view_off) -
                 loss(means_data + mean_off, nullptr, quats_data + quat_off,
                      minus_scale, viewmats_data + view_off)) /
                (2.0f * eps);
          }
        }

        if (v_viewmats_data != nullptr) {
          for (int axis = 0; axis < 12; ++axis) {
            float plus_view[16];
            float minus_view[16];
            for (int i = 0; i < 16; ++i) {
              plus_view[i] = viewmats_data[view_off + i];
              minus_view[i] = viewmats_data[view_off + i];
            }
            plus_view[axis] += eps;
            minus_view[axis] -= eps;
            v_viewmats_data[view_off + axis] +=
                (loss(means_data + mean_off,
                      covars_data == nullptr ? nullptr : covars_data + gaussian_off * 6,
                      quats_data == nullptr ? nullptr : quats_data + gaussian_off * 4,
                      scales_data == nullptr ? nullptr : scales_data + gaussian_off * 3,
                      plus_view) -
                 loss(means_data + mean_off,
                      covars_data == nullptr ? nullptr : covars_data + gaussian_off * 6,
                      quats_data == nullptr ? nullptr : quats_data + gaussian_off * 4,
                      scales_data == nullptr ? nullptr : scales_data + gaussian_off * 3,
                      minus_view)) /
                (2.0f * eps);
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
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("GSPlatProjectionEWA3DGSFused vjp is not implemented.");
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
