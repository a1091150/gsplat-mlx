#pragma once

#include <vector>

#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace gsplat_core {
namespace mx = mlx::core;

struct RasterizeToPixels3DGSParams {
  int image_width;
  int image_height;
  int tile_size;
  bool use_backgrounds;
  bool use_masks;
  bool packed;
};

struct RasterizeToPixels3DGSInput {
  mx::array means2d;
  mx::array conics;
  mx::array colors;
  mx::array opacities;
  mx::array backgrounds;
  mx::array masks;
  mx::array tile_offsets;
  mx::array flatten_ids;
  mx::StreamOrDevice s;
  RasterizeToPixels3DGSParams params;
};

enum RasterizeToPixels3DGSOutputIndex {
  kRenderColors = 0,
  kRenderAlphas = 1,
  kLastIds = 2,
};

std::vector<mx::array> gsplat_rasterize_to_pixels_3dgs(
    const RasterizeToPixels3DGSInput& input);

class GSPlatRasterizeToPixels3DGS : public mx::Primitive {
 public:
  GSPlatRasterizeToPixels3DGS(mx::Stream stream,
                              RasterizeToPixels3DGSParams params)
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
    return "GSPlatRasterizeToPixels3DGS";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  RasterizeToPixels3DGSParams params_;
};

}  // namespace gsplat_core
