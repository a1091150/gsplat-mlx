#pragma once

#include <vector>

#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace gsplat_core {
namespace mx = mlx::core;

struct ProjectionEWA3DGSFusedParams {
  int image_width;
  int image_height;
  float eps2d;
  float near_plane;
  float far_plane;
  float radius_clip;
  bool calc_compensations;
  int camera_model;
  bool use_covars;
  bool use_opacities;
};

struct ProjectionEWA3DGSFusedInput {
  mx::array means;
  mx::array covars;
  mx::array quats;
  mx::array scales;
  mx::array opacities;
  mx::array viewmats;
  mx::array Ks;
  mx::StreamOrDevice s;
  ProjectionEWA3DGSFusedParams params;
};

struct ProjectionEWA3DGSFusedBackwardInput {
  mx::array means;
  mx::array covars;
  mx::array quats;
  mx::array scales;
  mx::array viewmats;
  mx::array Ks;
  mx::array radii;
  mx::array conics;
  mx::array compensations;
  mx::array v_means2d;
  mx::array v_depths;
  mx::array v_conics;
  mx::array v_compensations;
  mx::StreamOrDevice s;
  ProjectionEWA3DGSFusedParams params;
  bool viewmats_requires_grad;
};

enum ProjectionEWA3DGSFusedOutputIndex {
  kRadii = 0,
  kMeans2D = 1,
  kDepths = 2,
  kConics = 3,
  kCompensations = 4,
};

enum ProjectionEWA3DGSFusedBackwardOutputIndex {
  kProjectionVMeans = 0,
  kProjectionVCovars = 1,
  kProjectionVQuats = 2,
  kProjectionVScales = 3,
  kProjectionVViewmats = 4,
};

std::vector<mx::array> gsplat_projection_ewa_3dgs_fused(
    const ProjectionEWA3DGSFusedInput& input);

std::vector<mx::array> gsplat_projection_ewa_3dgs_fused_backward(
    const ProjectionEWA3DGSFusedBackwardInput& input);

class GSPlatProjectionEWA3DGSFused : public mx::Primitive {
 public:
  GSPlatProjectionEWA3DGSFused(mx::Stream stream,
                               ProjectionEWA3DGSFusedParams params)
      : mx::Primitive(stream), params_(params) {}

  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array>& primals,
                             const std::vector<mx::array>& tangents,
                             const std::vector<int>& argnums) override;

  std::vector<mx::array> vjp(const std::vector<mx::array>& primals,
                             const std::vector<mx::array>& cotangents,
                             const std::vector<int>& argnums,
                             const std::vector<mx::array>& outputs) override;

  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "GSPlatProjectionEWA3DGSFused";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  ProjectionEWA3DGSFusedParams params_;
};

class GSPlatProjectionEWA3DGSFusedBackward : public mx::Primitive {
 public:
  GSPlatProjectionEWA3DGSFusedBackward(mx::Stream stream,
                                       ProjectionEWA3DGSFusedParams params,
                                       bool viewmats_requires_grad)
      : mx::Primitive(stream),
        params_(params),
        viewmats_requires_grad_(viewmats_requires_grad) {}

  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array>& primals,
                             const std::vector<mx::array>& tangents,
                             const std::vector<int>& argnums) override;

  std::vector<mx::array> vjp(const std::vector<mx::array>& primals,
                             const std::vector<mx::array>& cotangents,
                             const std::vector<int>& argnums,
                             const std::vector<mx::array>& outputs) override;

  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "GSPlatProjectionEWA3DGSFusedBackward";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  ProjectionEWA3DGSFusedParams params_;
  bool viewmats_requires_grad_;
};

}  // namespace gsplat_core
