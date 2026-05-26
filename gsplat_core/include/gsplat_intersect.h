#pragma once

#include <cstdint>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace gsplat_core {

struct IntersectTileParams {
  int I;
  int tile_size;
  int tile_width;
  int tile_height;
  bool sort;
  bool segmented;
  bool packed;
  bool use_conics;
  bool use_opacities;
};

struct IntersectTileInput {
  mlx::core::array means2d;
  mlx::core::array radii;
  mlx::core::array depths;
  mlx::core::array conics;
  mlx::core::array opacities;
  mlx::core::array image_ids;
  mlx::core::array gaussian_ids;
  mlx::core::StreamOrDevice s;
  IntersectTileParams params;
};

enum IntersectTileOutputIndex {
  kTilesPerGauss = 0,
  kIsectIds = 1,
  kFlattenIds = 2,
};

std::vector<mlx::core::array> gsplat_intersect_tile(
    const IntersectTileInput& input);

std::vector<mlx::core::array> gsplat_intersect_tile_gpu_staged(
    const IntersectTileInput& input);

mlx::core::array gsplat_intersect_tile_count(const IntersectTileInput& input);

mlx::core::array gsplat_intersect_tile_offsets(
    const mlx::core::array& tiles_per_gauss,
    mlx::core::StreamOrDevice s = mlx::core::Device::cpu);

std::vector<mlx::core::array> gsplat_intersect_tile_encode(
    const IntersectTileInput& input,
    const mlx::core::array& tile_offsets,
    int total_isects);

std::vector<mlx::core::array> gsplat_intersect_tile_sort(
    const mlx::core::array& isect_ids,
    const mlx::core::array& flatten_ids,
    mlx::core::StreamOrDevice s = mlx::core::Device::cpu);

mlx::core::array gsplat_intersect_offset(
    const mlx::core::array& isect_ids,
    int I,
    int tile_width,
    int tile_height,
    mlx::core::StreamOrDevice s = mlx::core::Device::cpu);

struct IntersectOffsetParams {
  int I;
  int tile_width;
  int tile_height;
};

struct IntersectTileEncodeParams {
  int I;
  int tile_size;
  int tile_width;
  int tile_height;
  int total_isects;
};

class GSPlatIntersectTileCount : public mlx::core::Primitive {
 public:
  GSPlatIntersectTileCount(mlx::core::Stream stream, IntersectTileParams params)
      : mlx::core::Primitive(stream), params_(params) {}

  void eval_cpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;
  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  std::vector<mlx::core::array> jvp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<mlx::core::array> vjp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mlx::core::array>& outputs) override;

  std::pair<std::vector<mlx::core::array>, std::vector<int>> vmap(
      const std::vector<mlx::core::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "GSPlatIntersectTileCount";
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override;

 private:
  IntersectTileParams params_;
};

class GSPlatIntersectTileEncode : public mlx::core::Primitive {
 public:
  GSPlatIntersectTileEncode(mlx::core::Stream stream,
                            IntersectTileEncodeParams params)
      : mlx::core::Primitive(stream), params_(params) {}

  void eval_cpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;
  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  std::vector<mlx::core::array> jvp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<mlx::core::array> vjp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mlx::core::array>& outputs) override;

  std::pair<std::vector<mlx::core::array>, std::vector<int>> vmap(
      const std::vector<mlx::core::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "GSPlatIntersectTileEncode";
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override;

 private:
  IntersectTileEncodeParams params_;
};

class GSPlatIntersectOffset : public mlx::core::Primitive {
 public:
  GSPlatIntersectOffset(mlx::core::Stream stream, IntersectOffsetParams params)
      : mlx::core::Primitive(stream), params_(params) {}

  void eval_cpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;
  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  std::vector<mlx::core::array> jvp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<mlx::core::array> vjp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mlx::core::array>& outputs) override;

  std::pair<std::vector<mlx::core::array>, std::vector<int>> vmap(
      const std::vector<mlx::core::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "GSPlatIntersectOffset";
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override;

 private:
  IntersectOffsetParams params_;
};

}  // namespace gsplat_core
