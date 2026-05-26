#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/dummy.h"
#include "../include/gsplat_projection.h"

namespace mx = mlx::core;

namespace {

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

}  // namespace

int main() {
  try {
    mx::set_default_device(mx::Device::cpu);
    test_dummy_add();
    test_dummy_array_add();
    test_projection_ewa_3dgs_fused_shapes();
    std::cout << "gsplat_core C++ smoke tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "gsplat_core C++ smoke tests failed: " << e.what() << "\n";
    return 1;
  }
}
