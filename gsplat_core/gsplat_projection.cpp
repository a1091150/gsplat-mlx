#include "include/gsplat_projection.h"

#include "include/helper.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

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
#else
void GSPlatProjectionEWA3DGSFused::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GSPlatProjectionEWA3DGSFused has no GPU implementation.");
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

}  // namespace gsplat_core
