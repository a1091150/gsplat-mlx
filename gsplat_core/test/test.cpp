#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/dummy.h"
#include "../include/gsplat_intersect.h"
#include "../include/gsplat_projection.h"
#include "../include/gsplat_quat_scale_to_covar.h"
#include "../include/gsplat_rasterize.h"
#include "../include/gsplat_spherical_harmonics.h"

namespace mx = mlx::core;

namespace {

struct ProjectionExpected {
  int radii[2];
  float means2d[2];
  float depth;
  float conics[3];
  float compensation;
};

void expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void expect_shape(const mx::array& array,
                  const std::vector<int>& expected,
                  const std::string& name) {
  expect(array.ndim() == expected.size(), name + " ndim mismatch");
  for (size_t i = 0; i < expected.size(); ++i) {
    expect(array.shape(static_cast<int>(i)) == expected[i],
           name + " shape mismatch at dim " + std::to_string(i));
  }
}

void expect_dtype(const mx::array& array,
                  mx::Dtype dtype,
                  const std::string& name) {
  expect(array.dtype().val() == dtype.val(), name + " dtype mismatch");
}

void expect_close(float actual,
                  float expected,
                  float tolerance,
                  const std::string& name) {
  if (std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(
        name + " mismatch: expected " + std::to_string(expected) +
        ", got " + std::to_string(actual));
  }
}

ProjectionExpected reference_pinhole_projection(float x,
                                                float y,
                                                float z,
                                                float variance,
                                                float opacity,
                                                int image_width,
                                                int image_height,
                                                float fx,
                                                float fy,
                                                float cx,
                                                float cy,
                                                float eps2d,
                                                bool calc_compensations) {
  const float tan_fovx = 0.5f * static_cast<float>(image_width) / fx;
  const float tan_fovy = 0.5f * static_cast<float>(image_height) / fy;
  const float lim_x_pos =
      (static_cast<float>(image_width) - cx) / fx + 0.3f * tan_fovx;
  const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
  const float lim_y_pos =
      (static_cast<float>(image_height) - cy) / fy + 0.3f * tan_fovy;
  const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

  const float rz = 1.0f / z;
  const float rz2 = rz * rz;
  const float tx = z * std::min(lim_x_pos, std::max(-lim_x_neg, x * rz));
  const float ty = z * std::min(lim_y_pos, std::max(-lim_y_neg, y * rz));

  const float j00 = fx * rz;
  const float j02 = -fx * tx * rz2;
  const float j11 = fy * rz;
  const float j12 = -fy * ty * rz2;

  float cov00 = variance * (j00 * j00 + j02 * j02);
  float cov01 = variance * (j02 * j12);
  float cov11 = variance * (j11 * j11 + j12 * j12);
  const float det_orig = cov00 * cov11 - cov01 * cov01;

  cov00 += eps2d;
  cov11 += eps2d;
  const float det = cov00 * cov11 - cov01 * cov01;
  const float min_compensation = 0.005f;
  const float compensation = std::sqrt(std::max(
      min_compensation * min_compensation, det_orig / det));

  float extend = 3.33f;
  float opacity_for_bounds = opacity;
  if (calc_compensations) {
    opacity_for_bounds *= compensation;
  }
  const float alpha_threshold = 1.0f / 255.0f;
  if (opacity_for_bounds >= alpha_threshold) {
    extend = std::min(
        extend,
        std::sqrt(2.0f * std::log(opacity_for_bounds / alpha_threshold)));
  }

  const float radius_x = std::ceil(extend * std::sqrt(cov00));
  const float radius_y = std::ceil(extend * std::sqrt(cov11));

  ProjectionExpected expected = {};
  expected.radii[0] = static_cast<int>(radius_x);
  expected.radii[1] = static_cast<int>(radius_y);
  expected.means2d[0] = fx * x * rz + cx;
  expected.means2d[1] = fy * y * rz + cy;
  expected.depth = z;
  expected.conics[0] = cov11 / det;
  expected.conics[1] = -cov01 / det;
  expected.conics[2] = cov00 / det;
  expected.compensation = compensation;
  return expected;
}

void test_dummy_add() {
  const int value = gsplat_core::dummy_add(20, 22);
  expect(value == 42, "dummy_add failed");
  std::cout << "dummy_add ok\n";
}

void test_dummy_array_add() {
  mx::array a({1.0f, 2.0f, 3.0f}, mx::float32);
  mx::array b({4.0f, 5.0f, 6.0f}, mx::float32);
  mx::array out = gsplat_core::dummy_array_add(a, b);
  out.eval();

  expect_shape(out, {3}, "dummy_array_add output");
  expect_dtype(out, mx::float32, "dummy_array_add output");

  const float* data = out.data<float>();
  expect(data[0] == 5.0f, "dummy_array_add output[0] mismatch");
  expect(data[1] == 7.0f, "dummy_array_add output[1] mismatch");
  expect(data[2] == 9.0f, "dummy_array_add output[2] mismatch");
  std::cout << "dummy_array_add ok\n";
}

void test_projection_ewa_3dgs_fused_shapes() {
  mx::array means(
      {0.0f, 0.0f, 1.0f, 0.25f, -0.25f, 2.0f},
      {1, 2, 3},
      mx::float32);
  mx::array quats(
      {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
      {1, 2, 4},
      mx::float32);
  mx::array scales(
      {0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f},
      {1, 2, 3},
      mx::float32);
  mx::array opacities({0.8f, 0.6f}, {1, 2}, mx::float32);
  mx::array viewmats(
      {1.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 1.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 1.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 1.0f},
      {1, 1, 4, 4},
      mx::float32);
  mx::array Ks(
      {50.0f, 0.0f, 32.0f,
       0.0f, 50.0f, 32.0f,
       0.0f, 0.0f, 1.0f},
      {1, 1, 3, 3},
      mx::float32);

  gsplat_core::ProjectionEWA3DGSFusedInput input = {
      .means = means,
      .covars = mx::zeros({0}, mx::float32, mx::Device::cpu),
      .quats = quats,
      .scales = scales,
      .opacities = opacities,
      .viewmats = viewmats,
      .Ks = Ks,
      .s = mx::Device::cpu,
      .params = {
          .image_width = 64,
          .image_height = 64,
          .eps2d = 0.3f,
          .near_plane = 0.01f,
          .far_plane = 100.0f,
          .radius_clip = 0.0f,
          .calc_compensations = true,
          .camera_model = 0,
          .use_covars = false,
          .use_opacities = true,
      },
  };

  std::vector<mx::array> outputs =
      gsplat_core::gsplat_projection_ewa_3dgs_fused(input);
  expect(outputs.size() == 5, "projection output count mismatch");

  expect_shape(outputs[gsplat_core::kRadii], {1, 1, 2, 2}, "radii");
  expect_shape(outputs[gsplat_core::kMeans2D], {1, 1, 2, 2}, "means2d");
  expect_shape(outputs[gsplat_core::kDepths], {1, 1, 2}, "depths");
  expect_shape(outputs[gsplat_core::kConics], {1, 1, 2, 3}, "conics");
  expect_shape(outputs[gsplat_core::kCompensations], {1, 1, 2}, "compensations");

  expect_dtype(outputs[gsplat_core::kRadii], mx::int32, "radii");
  expect_dtype(outputs[gsplat_core::kMeans2D], mx::float32, "means2d");
  expect_dtype(outputs[gsplat_core::kDepths], mx::float32, "depths");
  expect_dtype(outputs[gsplat_core::kConics], mx::float32, "conics");
  expect_dtype(outputs[gsplat_core::kCompensations], mx::float32, "compensations");

  mx::eval(outputs);
  expect(outputs[gsplat_core::kRadii].data<int32_t>()[0] == 0,
         "CPU projection fallback should zero radii");
  std::cout << "projection_ewa_3dgs_fused CPU shape smoke ok\n";
}

void expect_projection_values(const std::vector<mx::array>& outputs,
                              const std::vector<ProjectionExpected>& expected,
                              const std::string& name) {
  mx::eval(outputs);

  const int32_t* radii = outputs[gsplat_core::kRadii].data<int32_t>();
  const float* means2d = outputs[gsplat_core::kMeans2D].data<float>();
  const float* depths = outputs[gsplat_core::kDepths].data<float>();
  const float* conics = outputs[gsplat_core::kConics].data<float>();
  const float* compensations =
      outputs[gsplat_core::kCompensations].data<float>();

  for (size_t i = 0; i < expected.size(); ++i) {
    expect(radii[i * 2] == expected[i].radii[0],
           name + " radii x mismatch at gaussian " + std::to_string(i));
    expect(radii[i * 2 + 1] == expected[i].radii[1],
           name + " radii y mismatch at gaussian " + std::to_string(i));
    expect_close(means2d[i * 2], expected[i].means2d[0], 1.0e-4f,
                 name + " means2d x");
    expect_close(means2d[i * 2 + 1], expected[i].means2d[1], 1.0e-4f,
                 name + " means2d y");
    expect_close(depths[i], expected[i].depth, 1.0e-5f, name + " depth");
    expect_close(conics[i * 3], expected[i].conics[0], 1.0e-4f,
                 name + " conic xx");
    expect_close(conics[i * 3 + 1], expected[i].conics[1], 1.0e-4f,
                 name + " conic xy");
    expect_close(conics[i * 3 + 2], expected[i].conics[2], 1.0e-4f,
                 name + " conic yy");
    expect_close(compensations[i], expected[i].compensation, 1.0e-4f,
                 name + " compensation");
  }
}

void test_projection_ewa_3dgs_fused_gpu_numeric() {
  constexpr int image_width = 64;
  constexpr int image_height = 64;
  constexpr float fx = 50.0f;
  constexpr float fy = 50.0f;
  constexpr float cx = 32.0f;
  constexpr float cy = 32.0f;
  constexpr float eps2d = 0.3f;

  mx::array means(
      {0.0f, 0.0f, 1.0f, 0.25f, -0.25f, 2.0f},
      {1, 2, 3},
      mx::float32);
  mx::array quats(
      {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
      {1, 2, 4},
      mx::float32);
  mx::array scales(
      {0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f},
      {1, 2, 3},
      mx::float32);
  mx::array opacities({0.8f, 0.6f}, {1, 2}, mx::float32);
  mx::array viewmats(
      {1.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 1.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 1.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 1.0f},
      {1, 1, 4, 4},
      mx::float32);
  mx::array Ks(
      {fx, 0.0f, cx,
       0.0f, fy, cy,
       0.0f, 0.0f, 1.0f},
      {1, 1, 3, 3},
      mx::float32);

  const std::vector<ProjectionExpected> expected = {
      reference_pinhole_projection(0.0f, 0.0f, 1.0f, 0.01f, 0.8f,
                                   image_width, image_height, fx, fy, cx, cy,
                                   eps2d, true),
      reference_pinhole_projection(0.25f, -0.25f, 2.0f, 0.04f, 0.6f,
                                   image_width, image_height, fx, fy, cx, cy,
                                   eps2d, true),
  };

  gsplat_core::ProjectionEWA3DGSFusedInput quat_scale_input = {
      .means = means,
      .covars = mx::zeros({0}, mx::float32, mx::Device::gpu),
      .quats = quats,
      .scales = scales,
      .opacities = opacities,
      .viewmats = viewmats,
      .Ks = Ks,
      .s = mx::Device::gpu,
      .params = {
          .image_width = image_width,
          .image_height = image_height,
          .eps2d = eps2d,
          .near_plane = 0.01f,
          .far_plane = 100.0f,
          .radius_clip = 0.0f,
          .calc_compensations = true,
          .camera_model = 0,
          .use_covars = false,
          .use_opacities = true,
      },
  };
  expect_projection_values(
      gsplat_core::gsplat_projection_ewa_3dgs_fused(quat_scale_input),
      expected,
      "projection quats/scales");

  mx::array covars(
      {0.01f, 0.0f, 0.0f, 0.01f, 0.0f, 0.01f,
       0.04f, 0.0f, 0.0f, 0.04f, 0.0f, 0.04f},
      {1, 2, 6},
      mx::float32);
  gsplat_core::ProjectionEWA3DGSFusedInput covars_input = quat_scale_input;
  covars_input.covars = covars;
  covars_input.quats = mx::zeros({0}, mx::float32, mx::Device::gpu);
  covars_input.scales = mx::zeros({0}, mx::float32, mx::Device::gpu);
  covars_input.params.use_covars = true;
  expect_projection_values(
      gsplat_core::gsplat_projection_ewa_3dgs_fused(covars_input),
      expected,
      "projection covars");

  std::cout << "projection_ewa_3dgs_fused GPU numeric smoke ok\n";
}

void test_intersect_tile_and_offset_dense_aabb() {
  mx::array means2d(
      {20.0f, 20.0f, 50.0f, 50.0f, 8.0f, 8.0f},
      {1, 3, 2},
      mx::float32);
  mx::array radii(
      {10, 10, 5, 5, 0, 0},
      {1, 3, 2},
      mx::int32);
  mx::array depths(
      {1.0f, 0.5f, 2.0f},
      {1, 3},
      mx::float32);

  gsplat_core::IntersectTileInput input = {
      .means2d = means2d,
      .radii = radii,
      .depths = depths,
      .conics = mx::zeros({0}, mx::float32),
      .opacities = mx::zeros({0}, mx::float32),
      .image_ids = mx::zeros({0}, mx::int64),
      .gaussian_ids = mx::zeros({0}, mx::int64),
      .s = mx::Device::cpu,
      .params = {
          .I = 1,
          .tile_size = 16,
          .tile_width = 4,
          .tile_height = 4,
          .sort = true,
          .segmented = false,
          .packed = false,
          .use_conics = false,
          .use_opacities = false,
      },
  };

  std::vector<mx::array> outputs = gsplat_core::gsplat_intersect_tile(input);
  expect(outputs.size() == 3, "intersect_tile output count mismatch");
  expect_shape(outputs[gsplat_core::kTilesPerGauss], {1, 3}, "tiles_per_gauss");
  expect_shape(outputs[gsplat_core::kIsectIds], {8}, "isect_ids");
  expect_shape(outputs[gsplat_core::kFlattenIds], {8}, "flatten_ids");
  expect_dtype(outputs[gsplat_core::kTilesPerGauss], mx::int32, "tiles_per_gauss");
  expect_dtype(outputs[gsplat_core::kIsectIds], mx::int64, "isect_ids");
  expect_dtype(outputs[gsplat_core::kFlattenIds], mx::int32, "flatten_ids");

  mx::eval(outputs);
  const int32_t* tiles_per_gauss =
      outputs[gsplat_core::kTilesPerGauss].data<int32_t>();
  const int64_t* isect_ids = outputs[gsplat_core::kIsectIds].data<int64_t>();
  const int32_t* flatten_ids =
      outputs[gsplat_core::kFlattenIds].data<int32_t>();

  expect(tiles_per_gauss[0] == 4, "tiles_per_gauss[0] mismatch");
  expect(tiles_per_gauss[1] == 4, "tiles_per_gauss[1] mismatch");
  expect(tiles_per_gauss[2] == 0, "tiles_per_gauss[2] mismatch");

  const int expected_tiles[8] = {0, 1, 4, 5, 10, 11, 14, 15};
  const int expected_flatten[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  for (int i = 0; i < 8; ++i) {
    expect(static_cast<int>((isect_ids[i] >> 32) & 0x1f) == expected_tiles[i],
           "intersect tile id mismatch at " + std::to_string(i));
    expect(flatten_ids[i] == expected_flatten[i],
           "intersect flatten id mismatch at " + std::to_string(i));
  }

  mx::array offsets = gsplat_core::gsplat_intersect_offset(
      outputs[gsplat_core::kIsectIds], 1, 4, 4, mx::Device::cpu);
  expect_shape(offsets, {1, 4, 4}, "intersect offsets");
  expect_dtype(offsets, mx::int32, "intersect offsets");
  offsets.eval();

  const int expected_offsets[16] = {
      0, 1, 2, 2,
      2, 3, 4, 4,
      4, 4, 4, 5,
      6, 6, 6, 7,
  };
  const int32_t* offset_data = offsets.data<int32_t>();
  for (int i = 0; i < 16; ++i) {
    expect(offset_data[i] == expected_offsets[i],
           "intersect offset mismatch at " + std::to_string(i));
  }

  std::cout << "intersect_tile/intersect_offset dense AABB smoke ok\n";
}

void test_intersect_tile_count_gpu_dense_aabb() {
  mx::array means2d(
      {20.0f, 20.0f, 50.0f, 50.0f, 8.0f, 8.0f},
      {1, 3, 2},
      mx::float32);
  mx::array radii(
      {10, 10, 5, 5, 0, 0},
      {1, 3, 2},
      mx::int32);
  mx::array depths(
      {1.0f, 0.5f, 2.0f},
      {1, 3},
      mx::float32);

  gsplat_core::IntersectTileInput input = {
      .means2d = means2d,
      .radii = radii,
      .depths = depths,
      .conics = mx::zeros({0}, mx::float32),
      .opacities = mx::zeros({0}, mx::float32),
      .image_ids = mx::zeros({0}, mx::int64),
      .gaussian_ids = mx::zeros({0}, mx::int64),
      .s = mx::Device::gpu,
      .params = {
          .I = 1,
          .tile_size = 16,
          .tile_width = 4,
          .tile_height = 4,
          .sort = true,
          .segmented = false,
          .packed = false,
          .use_conics = false,
          .use_opacities = false,
      },
  };

  mx::array counts = gsplat_core::gsplat_intersect_tile_count(input);
  expect_shape(counts, {1, 3}, "intersect tile count gpu");
  expect_dtype(counts, mx::int32, "intersect tile count gpu");
  counts.eval();

  const int32_t* data = counts.data<int32_t>();
  expect(data[0] == 4, "intersect tile count gpu[0] mismatch");
  expect(data[1] == 4, "intersect tile count gpu[1] mismatch");
  expect(data[2] == 0, "intersect tile count gpu[2] mismatch");

  std::cout << "intersect_tile_count GPU dense AABB smoke ok\n";
}

void test_rasterize_to_pixels_3dgs_dense_reference() {
  mx::array means2d({1.0f, 1.0f}, {1, 1, 2}, mx::float32);
  mx::array conics({1.0f, 0.0f, 1.0f}, {1, 1, 3}, mx::float32);
  mx::array colors({1.0f, 0.0f, 0.0f}, {1, 1, 3}, mx::float32);
  mx::array opacities({0.5f}, {1, 1}, mx::float32);
  mx::array backgrounds({0.1f, 0.2f, 0.3f}, {1, 3}, mx::float32);
  mx::array tile_offsets({0}, {1, 1, 1}, mx::int32);
  mx::array flatten_ids({0}, {1}, mx::int32);

  gsplat_core::RasterizeToPixels3DGSInput input = {
      .means2d = means2d,
      .conics = conics,
      .colors = colors,
      .opacities = opacities,
      .backgrounds = backgrounds,
      .masks = mx::zeros({0}, mx::bool_, mx::Device::cpu),
      .tile_offsets = tile_offsets,
      .flatten_ids = flatten_ids,
      .s = mx::Device::cpu,
      .params = {
          .image_width = 2,
          .image_height = 2,
          .tile_size = 2,
          .use_backgrounds = true,
          .use_masks = false,
          .packed = false,
      },
  };

  std::vector<mx::array> outputs =
      gsplat_core::gsplat_rasterize_to_pixels_3dgs(input);
  expect(outputs.size() == 3, "rasterize output count mismatch");
  expect_shape(outputs[gsplat_core::kRenderColors], {1, 2, 2, 3},
               "render_colors");
  expect_shape(outputs[gsplat_core::kRenderAlphas], {1, 2, 2, 1},
               "render_alphas");
  expect_shape(outputs[gsplat_core::kLastIds], {1, 2, 2}, "last_ids");
  expect_dtype(outputs[gsplat_core::kRenderColors], mx::float32,
               "render_colors");
  expect_dtype(outputs[gsplat_core::kRenderAlphas], mx::float32,
               "render_alphas");
  expect_dtype(outputs[gsplat_core::kLastIds], mx::int32, "last_ids");

  mx::eval(outputs);
  const float expected_alpha = 0.5f * std::exp(-0.25f);
  const float expected_T = 1.0f - expected_alpha;
  const float expected_rgb[3] = {
      expected_alpha + expected_T * 0.1f,
      expected_T * 0.2f,
      expected_T * 0.3f,
  };
  const float* render_colors = outputs[gsplat_core::kRenderColors].data<float>();
  const float* render_alphas = outputs[gsplat_core::kRenderAlphas].data<float>();
  const int32_t* last_ids = outputs[gsplat_core::kLastIds].data<int32_t>();
  for (int pixel = 0; pixel < 4; ++pixel) {
    expect_close(render_alphas[pixel], expected_alpha, 1.0e-5f,
                 "rasterize alpha");
    expect(last_ids[pixel] == 0, "rasterize last id mismatch");
    for (int channel = 0; channel < 3; ++channel) {
      expect_close(render_colors[pixel * 3 + channel],
                   expected_rgb[channel],
                   1.0e-5f,
                   "rasterize color");
    }
  }

  std::cout << "rasterize_to_pixels_3dgs dense reference smoke ok\n";
}

void test_spherical_harmonics_forward_reference() {
  mx::array dirs(
      {0.0f, 0.0f, 1.0f,
       1.0f, 0.0f, 0.0f},
      {2, 3},
      mx::float32);
  mx::array coeffs(
      {1.0f, 2.0f, 3.0f,
       0.1f, 0.2f, 0.3f,
       0.4f, 0.5f, 0.6f,
       0.7f, 0.8f, 0.9f,
       4.0f, 5.0f, 6.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f},
      {2, 4, 3},
      mx::float32);

  gsplat_core::SphericalHarmonicsInput degree0_input = {
      .degrees_to_use = 0,
      .dirs = dirs,
      .coeffs = coeffs,
      .masks = mx::zeros({0}, mx::bool_, mx::Device::cpu),
      .s = mx::Device::cpu,
      .use_masks = false,
  };
  mx::array degree0 =
      gsplat_core::gsplat_spherical_harmonics_forward(degree0_input);
  expect_shape(degree0, {2, 3}, "spherical_harmonics degree0");
  expect_dtype(degree0, mx::float32, "spherical_harmonics degree0");
  degree0.eval();

  constexpr float c0 = 0.2820947917738781f;
  const float* degree0_data = degree0.data<float>();
  expect_close(degree0_data[0], c0 * 1.0f, 1.0e-6f, "SH degree0 red");
  expect_close(degree0_data[1], c0 * 2.0f, 1.0e-6f, "SH degree0 green");
  expect_close(degree0_data[2], c0 * 3.0f, 1.0e-6f, "SH degree0 blue");
  expect_close(degree0_data[3], c0 * 4.0f, 1.0e-6f, "SH degree0 red elem1");

  gsplat_core::SphericalHarmonicsInput degree1_input = degree0_input;
  degree1_input.degrees_to_use = 1;
  mx::array degree1 =
      gsplat_core::gsplat_spherical_harmonics_forward(degree1_input);
  degree1.eval();

  constexpr float c1 = 0.48860251190292f;
  const float* degree1_data = degree1.data<float>();
  expect_close(degree1_data[0], c0 * 1.0f + c1 * 0.4f, 1.0e-6f,
               "SH degree1 red z");
  expect_close(degree1_data[1], c0 * 2.0f + c1 * 0.5f, 1.0e-6f,
               "SH degree1 green z");
  expect_close(degree1_data[2], c0 * 3.0f + c1 * 0.6f, 1.0e-6f,
               "SH degree1 blue z");
  expect_close(degree1_data[3], c0 * 4.0f, 1.0e-6f,
               "SH degree1 red x");
  expect_close(degree1_data[4], c0 * 5.0f, 1.0e-6f,
               "SH degree1 green x");
  expect_close(degree1_data[5], c0 * 6.0f, 1.0e-6f,
               "SH degree1 blue x");

  mx::array masks({true, false}, {2}, mx::bool_);
  gsplat_core::SphericalHarmonicsInput masked_input = degree0_input;
  masked_input.masks = masks;
  masked_input.use_masks = true;
  mx::array masked =
      gsplat_core::gsplat_spherical_harmonics_forward(masked_input);
  masked.eval();
  const float* masked_data = masked.data<float>();
  expect_close(masked_data[0], c0 * 1.0f, 1.0e-6f, "SH masked elem0");
  expect_close(masked_data[3], 0.0f, 1.0e-6f, "SH masked elem1");

  std::cout << "spherical_harmonics_forward reference smoke ok\n";
}

void test_spherical_harmonics_forward_gpu_reference() {
  mx::array dirs(
      {0.0f, 0.0f, 1.0f,
       1.0f, 0.0f, 0.0f},
      {2, 3},
      mx::float32);
  mx::array coeffs(
      {1.0f, 2.0f, 3.0f,
       0.1f, 0.2f, 0.3f,
       0.4f, 0.5f, 0.6f,
       0.7f, 0.8f, 0.9f,
       4.0f, 5.0f, 6.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f},
      {2, 4, 3},
      mx::float32);
  mx::array masks({true, false}, {2}, mx::bool_);

  gsplat_core::SphericalHarmonicsInput input = {
      .degrees_to_use = 1,
      .dirs = dirs,
      .coeffs = coeffs,
      .masks = masks,
      .s = mx::Device::gpu,
      .use_masks = true,
  };
  mx::array colors =
      gsplat_core::gsplat_spherical_harmonics_forward(input);
  expect_shape(colors, {2, 3}, "spherical_harmonics gpu colors");
  expect_dtype(colors, mx::float32, "spherical_harmonics gpu colors");
  colors.eval();

  constexpr float c0 = 0.2820947917738781f;
  constexpr float c1 = 0.48860251190292f;
  const float* data = colors.data<float>();
  expect_close(data[0], c0 * 1.0f + c1 * 0.4f, 1.0e-6f,
               "SH GPU degree1 masked red");
  expect_close(data[1], c0 * 2.0f + c1 * 0.5f, 1.0e-6f,
               "SH GPU degree1 masked green");
  expect_close(data[2], c0 * 3.0f + c1 * 0.6f, 1.0e-6f,
               "SH GPU degree1 masked blue");
  expect_close(data[3], 0.0f, 1.0e-6f, "SH GPU masked red");
  expect_close(data[4], 0.0f, 1.0e-6f, "SH GPU masked green");
  expect_close(data[5], 0.0f, 1.0e-6f, "SH GPU masked blue");

  std::cout << "spherical_harmonics_forward GPU reference smoke ok\n";
}

void test_quat_scale_to_covar_preci_reference() {
  mx::array quats(
      {1.0f, 0.0f, 0.0f, 0.0f,
       2.0f, 0.0f, 0.0f, 0.0f},
      {2, 4},
      mx::float32);
  mx::array scales(
      {2.0f, 3.0f, 4.0f,
       0.5f, 2.0f, 4.0f},
      {2, 3},
      mx::float32);

  gsplat_core::QuatScaleToCovarPreciInput triu_input = {
      .quats = quats,
      .scales = scales,
      .s = mx::Device::cpu,
      .compute_covar = true,
      .compute_preci = true,
      .triu = true,
  };
  std::vector<mx::array> triu_outputs =
      gsplat_core::gsplat_quat_scale_to_covar_preci_forward(triu_input);
  expect(triu_outputs.size() == 2, "quat_scale output count mismatch");
  expect_shape(triu_outputs[gsplat_core::kCovars], {2, 6}, "quat covars triu");
  expect_shape(triu_outputs[gsplat_core::kPrecis], {2, 6}, "quat precis triu");
  expect_dtype(triu_outputs[gsplat_core::kCovars], mx::float32,
               "quat covars triu");
  expect_dtype(triu_outputs[gsplat_core::kPrecis], mx::float32,
               "quat precis triu");
  mx::eval(triu_outputs);

  const float* covars = triu_outputs[gsplat_core::kCovars].data<float>();
  const float* precis = triu_outputs[gsplat_core::kPrecis].data<float>();
  const float expected_covars[12] = {
      4.0f, 0.0f, 0.0f, 9.0f, 0.0f, 16.0f,
      0.25f, 0.0f, 0.0f, 4.0f, 0.0f, 16.0f,
  };
  const float expected_precis[12] = {
      0.25f, 0.0f, 0.0f, 1.0f / 9.0f, 0.0f, 1.0f / 16.0f,
      4.0f, 0.0f, 0.0f, 0.25f, 0.0f, 1.0f / 16.0f,
  };
  for (int i = 0; i < 12; ++i) {
    expect_close(covars[i], expected_covars[i], 1.0e-6f,
                 "quat covars triu");
    expect_close(precis[i], expected_precis[i], 1.0e-6f,
                 "quat precis triu");
  }

  gsplat_core::QuatScaleToCovarPreciInput full_input = triu_input;
  full_input.triu = false;
  full_input.compute_preci = false;
  std::vector<mx::array> full_outputs =
      gsplat_core::gsplat_quat_scale_to_covar_preci_forward(full_input);
  expect_shape(full_outputs[gsplat_core::kCovars], {2, 3, 3},
               "quat covars full");
  expect_shape(full_outputs[gsplat_core::kPrecis], {0}, "quat precis empty");
  mx::eval(full_outputs);

  const float* full_covars = full_outputs[gsplat_core::kCovars].data<float>();
  const float expected_full0[9] = {
      4.0f, 0.0f, 0.0f,
      0.0f, 9.0f, 0.0f,
      0.0f, 0.0f, 16.0f,
  };
  for (int i = 0; i < 9; ++i) {
    expect_close(full_covars[i], expected_full0[i], 1.0e-6f,
                 "quat covars full");
  }

  std::cout << "quat_scale_to_covar_preci reference smoke ok\n";
}

void test_quat_scale_to_covar_preci_gpu_reference() {
  mx::array quats(
      {1.0f, 0.0f, 0.0f, 0.0f,
       2.0f, 0.0f, 0.0f, 0.0f},
      {2, 4},
      mx::float32);
  mx::array scales(
      {2.0f, 3.0f, 4.0f,
       0.5f, 2.0f, 4.0f},
      {2, 3},
      mx::float32);

  gsplat_core::QuatScaleToCovarPreciInput triu_input = {
      .quats = quats,
      .scales = scales,
      .s = mx::Device::gpu,
      .compute_covar = true,
      .compute_preci = true,
      .triu = true,
  };
  std::vector<mx::array> triu_outputs =
      gsplat_core::gsplat_quat_scale_to_covar_preci_forward(triu_input);
  expect_shape(triu_outputs[gsplat_core::kCovars], {2, 6},
               "quat gpu covars triu");
  expect_shape(triu_outputs[gsplat_core::kPrecis], {2, 6},
               "quat gpu precis triu");
  mx::eval(triu_outputs);

  const float* covars = triu_outputs[gsplat_core::kCovars].data<float>();
  const float* precis = triu_outputs[gsplat_core::kPrecis].data<float>();
  const float expected_covars[12] = {
      4.0f, 0.0f, 0.0f, 9.0f, 0.0f, 16.0f,
      0.25f, 0.0f, 0.0f, 4.0f, 0.0f, 16.0f,
  };
  const float expected_precis[12] = {
      0.25f, 0.0f, 0.0f, 1.0f / 9.0f, 0.0f, 1.0f / 16.0f,
      4.0f, 0.0f, 0.0f, 0.25f, 0.0f, 1.0f / 16.0f,
  };
  for (int i = 0; i < 12; ++i) {
    expect_close(covars[i], expected_covars[i], 1.0e-6f,
                 "quat gpu covars triu");
    expect_close(precis[i], expected_precis[i], 1.0e-6f,
                 "quat gpu precis triu");
  }

  gsplat_core::QuatScaleToCovarPreciInput full_input = triu_input;
  full_input.triu = false;
  full_input.compute_preci = false;
  std::vector<mx::array> full_outputs =
      gsplat_core::gsplat_quat_scale_to_covar_preci_forward(full_input);
  expect_shape(full_outputs[gsplat_core::kCovars], {2, 3, 3},
               "quat gpu covars full");
  expect_shape(full_outputs[gsplat_core::kPrecis], {0},
               "quat gpu precis empty");
  mx::eval(full_outputs);

  const float* full_covars = full_outputs[gsplat_core::kCovars].data<float>();
  const float expected_full0[9] = {
      4.0f, 0.0f, 0.0f,
      0.0f, 9.0f, 0.0f,
      0.0f, 0.0f, 16.0f,
  };
  for (int i = 0; i < 9; ++i) {
    expect_close(full_covars[i], expected_full0[i], 1.0e-6f,
                 "quat gpu covars full");
  }

  std::cout << "quat_scale_to_covar_preci GPU reference smoke ok\n";
}

void test_3dgs_forward_chain_smoke() {
  constexpr int image_width = 16;
  constexpr int image_height = 16;
  constexpr int tile_size = 8;
  constexpr int tile_width = 2;
  constexpr int tile_height = 2;
  constexpr float c0 = 0.2820947917738781f;

  mx::array means({0.0f, 0.0f, 2.0f}, {1, 1, 3}, mx::float32);
  mx::array quats({1.0f, 0.0f, 0.0f, 0.0f}, {1, 1, 4}, mx::float32);
  mx::array scales({0.1f, 0.1f, 0.1f}, {1, 1, 3}, mx::float32);
  mx::array projection_opacities({0.8f}, {1, 1}, mx::float32);
  mx::array raster_opacities({0.8f}, {1, 1, 1}, mx::float32);
  mx::array viewmats(
      {1.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 1.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 1.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 1.0f},
      {1, 1, 4, 4},
      mx::float32);
  mx::array Ks(
      {20.0f, 0.0f, 8.0f,
       0.0f, 20.0f, 8.0f,
       0.0f, 0.0f, 1.0f},
      {1, 1, 3, 3},
      mx::float32);

  gsplat_core::ProjectionEWA3DGSFusedInput projection_input = {
      .means = means,
      .covars = mx::zeros({0}, mx::float32, mx::Device::gpu),
      .quats = quats,
      .scales = scales,
      .opacities = projection_opacities,
      .viewmats = viewmats,
      .Ks = Ks,
      .s = mx::Device::gpu,
      .params = {
          .image_width = image_width,
          .image_height = image_height,
          .eps2d = 0.3f,
          .near_plane = 0.01f,
          .far_plane = 100.0f,
          .radius_clip = 0.0f,
          .calc_compensations = false,
          .camera_model = 0,
          .use_covars = false,
          .use_opacities = true,
      },
  };
  std::vector<mx::array> projection =
      gsplat_core::gsplat_projection_ewa_3dgs_fused(projection_input);
  expect_shape(projection[gsplat_core::kRadii], {1, 1, 1, 2},
               "chain projection radii");
  expect_shape(projection[gsplat_core::kMeans2D], {1, 1, 1, 2},
               "chain projection means2d");
  expect_shape(projection[gsplat_core::kDepths], {1, 1, 1},
               "chain projection depths");
  expect_shape(projection[gsplat_core::kConics], {1, 1, 1, 3},
               "chain projection conics");

  mx::array dirs({0.0f, 0.0f, 1.0f}, {1, 1, 1, 3}, mx::float32);
  mx::array coeffs({1.0f / c0, 0.0f, 0.0f}, {1, 1, 1, 1, 3}, mx::float32);
  gsplat_core::SphericalHarmonicsInput sh_input = {
      .degrees_to_use = 0,
      .dirs = dirs,
      .coeffs = coeffs,
      .masks = mx::zeros({0}, mx::bool_, mx::Device::cpu),
      .s = mx::Device::cpu,
      .use_masks = false,
  };
  mx::array colors = gsplat_core::gsplat_spherical_harmonics_forward(sh_input);
  expect_shape(colors, {1, 1, 1, 3}, "chain SH colors");

  gsplat_core::IntersectTileInput intersect_input = {
      .means2d = projection[gsplat_core::kMeans2D],
      .radii = projection[gsplat_core::kRadii],
      .depths = projection[gsplat_core::kDepths],
      .conics = mx::zeros({0}, mx::float32),
      .opacities = mx::zeros({0}, mx::float32),
      .image_ids = mx::zeros({0}, mx::int64),
      .gaussian_ids = mx::zeros({0}, mx::int64),
      .s = mx::Device::cpu,
      .params = {
          .I = 1,
          .tile_size = tile_size,
          .tile_width = tile_width,
          .tile_height = tile_height,
          .sort = true,
          .segmented = false,
          .packed = false,
          .use_conics = false,
          .use_opacities = false,
      },
  };
  std::vector<mx::array> intersections =
      gsplat_core::gsplat_intersect_tile(intersect_input);
  expect(intersections[gsplat_core::kIsectIds].size() > 0,
         "chain intersect should produce tile hits");

  mx::array tile_offsets = gsplat_core::gsplat_intersect_offset(
      intersections[gsplat_core::kIsectIds],
      1,
      tile_width,
      tile_height,
      mx::Device::cpu);
  expect_shape(tile_offsets, {1, tile_height, tile_width},
               "chain tile offsets");

  mx::array backgrounds({0.0f, 0.0f, 0.0f}, {1, 3}, mx::float32);
  gsplat_core::RasterizeToPixels3DGSInput raster_input = {
      .means2d = projection[gsplat_core::kMeans2D],
      .conics = projection[gsplat_core::kConics],
      .colors = colors,
      .opacities = raster_opacities,
      .backgrounds = backgrounds,
      .masks = mx::zeros({0}, mx::bool_, mx::Device::cpu),
      .tile_offsets = tile_offsets,
      .flatten_ids = intersections[gsplat_core::kFlattenIds],
      .s = mx::Device::cpu,
      .params = {
          .image_width = image_width,
          .image_height = image_height,
          .tile_size = tile_size,
          .use_backgrounds = true,
          .use_masks = false,
          .packed = false,
      },
  };
  std::vector<mx::array> render =
      gsplat_core::gsplat_rasterize_to_pixels_3dgs(raster_input);
  expect_shape(render[gsplat_core::kRenderColors], {1, 16, 16, 3},
               "chain render colors");
  expect_shape(render[gsplat_core::kRenderAlphas], {1, 16, 16, 1},
               "chain render alphas");
  expect_shape(render[gsplat_core::kLastIds], {1, 16, 16},
               "chain last ids");

  mx::eval(render);
  const float* render_colors = render[gsplat_core::kRenderColors].data<float>();
  const float* render_alphas = render[gsplat_core::kRenderAlphas].data<float>();
  float alpha_sum = 0.0f;
  float red_sum = 0.0f;
  float green_sum = 0.0f;
  float blue_sum = 0.0f;
  for (int pixel = 0; pixel < image_width * image_height; ++pixel) {
    alpha_sum += render_alphas[pixel];
    red_sum += render_colors[pixel * 3];
    green_sum += render_colors[pixel * 3 + 1];
    blue_sum += render_colors[pixel * 3 + 2];
  }

  expect(alpha_sum > 0.01f, "chain render alpha should be nonzero");
  expect(red_sum > 0.01f, "chain render red should be nonzero");
  expect(green_sum < 1.0e-5f, "chain render green should stay zero");
  expect(blue_sum < 1.0e-5f, "chain render blue should stay zero");

  std::cout << "3dgs forward chain smoke ok\n";
}

}  // namespace

int main() {
  try {
    mx::set_default_device(mx::Device::cpu);
    test_dummy_add();
    test_dummy_array_add();
    test_projection_ewa_3dgs_fused_shapes();
    test_projection_ewa_3dgs_fused_gpu_numeric();
    test_intersect_tile_and_offset_dense_aabb();
    test_intersect_tile_count_gpu_dense_aabb();
    test_rasterize_to_pixels_3dgs_dense_reference();
    test_spherical_harmonics_forward_reference();
    test_spherical_harmonics_forward_gpu_reference();
    test_quat_scale_to_covar_preci_reference();
    test_quat_scale_to_covar_preci_gpu_reference();
    test_3dgs_forward_chain_smoke();
    std::cout << "gsplat_core C++ smoke tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "gsplat_core C++ smoke tests failed: " << e.what() << "\n";
    return 1;
  }
}
