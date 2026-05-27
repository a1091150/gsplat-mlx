#pragma once

#include <vector>

#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace gsplat_core {
namespace mx = mlx::core;

struct QuatScaleToCovarPreciInput {
  mx::array quats;
  mx::array scales;
  mx::StreamOrDevice s;
  bool compute_covar;
  bool compute_preci;
  bool triu;
};

struct QuatScaleToCovarPreciBackwardInput {
  mx::array quats;
  mx::array scales;
  mx::array v_covars;
  mx::array v_precis;
  mx::StreamOrDevice s;
  bool triu;
  bool use_v_covars;
  bool use_v_precis;
};

enum QuatScaleToCovarPreciOutputIndex {
  kCovars = 0,
  kPrecis = 1,
};

enum QuatScaleToCovarPreciBackwardOutputIndex {
  kVQuats = 0,
  kVScales = 1,
};

std::vector<mx::array> gsplat_quat_scale_to_covar_preci_forward(
    const QuatScaleToCovarPreciInput& input);

std::vector<mx::array> gsplat_quat_scale_to_covar_preci_backward(
    const QuatScaleToCovarPreciBackwardInput& input);

class GSPlatQuatScaleToCovarPreci : public mx::Primitive {
 public:
  GSPlatQuatScaleToCovarPreci(mx::Stream stream,
                              bool compute_covar,
                              bool compute_preci,
                              bool triu)
      : mx::Primitive(stream),
        compute_covar_(compute_covar),
        compute_preci_(compute_preci),
        triu_(triu) {}

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
    return "GSPlatQuatScaleToCovarPreci";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  bool compute_covar_;
  bool compute_preci_;
  bool triu_;
};

class GSPlatQuatScaleToCovarPreciBackward : public mx::Primitive {
 public:
  GSPlatQuatScaleToCovarPreciBackward(mx::Stream stream,
                                      bool triu,
                                      bool use_v_covars,
                                      bool use_v_precis)
      : mx::Primitive(stream),
        triu_(triu),
        use_v_covars_(use_v_covars),
        use_v_precis_(use_v_precis) {}

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
    return "GSPlatQuatScaleToCovarPreciBackward";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  bool triu_;
  bool use_v_covars_;
  bool use_v_precis_;
};

}  // namespace gsplat_core
