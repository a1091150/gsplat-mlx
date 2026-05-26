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

enum QuatScaleToCovarPreciOutputIndex {
  kCovars = 0,
  kPrecis = 1,
};

std::vector<mx::array> gsplat_quat_scale_to_covar_preci_forward(
    const QuatScaleToCovarPreciInput& input);

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

}  // namespace gsplat_core
