#pragma once

#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace gsplat_core {
namespace mx = mlx::core;

struct SphericalHarmonicsInput {
  int degrees_to_use;
  mx::array dirs;
  mx::array coeffs;
  mx::array masks;
  mx::StreamOrDevice s;
  bool use_masks;
};

struct SphericalHarmonicsBackwardInput {
  int degrees_to_use;
  mx::array dirs;
  mx::array coeffs;
  mx::array masks;
  mx::array v_colors;
  mx::StreamOrDevice s;
  bool use_masks;
  bool compute_v_dirs;
};

enum SphericalHarmonicsBackwardOutputIndex {
  kSHVDirs = 0,
  kSHVCoeffs = 1,
};

mx::array gsplat_spherical_harmonics_forward(
    const SphericalHarmonicsInput& input);

std::vector<mx::array> gsplat_spherical_harmonics_backward(
    const SphericalHarmonicsBackwardInput& input);

class GSPlatSphericalHarmonics : public mx::Primitive {
 public:
  GSPlatSphericalHarmonics(mx::Stream stream,
                           int degrees_to_use,
                           bool use_masks)
      : mx::Primitive(stream),
        degrees_to_use_(degrees_to_use),
        use_masks_(use_masks) {}

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
    return "GSPlatSphericalHarmonics";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int degrees_to_use_;
  bool use_masks_;
};

class GSPlatSphericalHarmonicsBackward : public mx::Primitive {
 public:
  GSPlatSphericalHarmonicsBackward(mx::Stream stream,
                                   int degrees_to_use,
                                   bool use_masks,
                                   bool compute_v_dirs)
      : mx::Primitive(stream),
        degrees_to_use_(degrees_to_use),
        use_masks_(use_masks),
        compute_v_dirs_(compute_v_dirs) {}

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
    return "GSPlatSphericalHarmonicsBackward";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int degrees_to_use_;
  bool use_masks_;
  bool compute_v_dirs_;
};

}  // namespace gsplat_core
