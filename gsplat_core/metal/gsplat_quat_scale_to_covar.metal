#include <metal_stdlib>

using namespace metal;

struct QuatScaleToCovarPreciKernelParams {
  uint n;
  uint compute_covar;
  uint compute_preci;
  uint triu;
};

struct Mat3 {
  float v[9];
};

inline Mat3 quat_to_rotmat(float4 quat) {
  float w = quat.x;
  float x = quat.y;
  float y = quat.z;
  float z = quat.w;
  float inv_norm = rsqrt(w * w + x * x + y * y + z * z);
  w *= inv_norm;
  x *= inv_norm;
  y *= inv_norm;
  z *= inv_norm;

  float x2 = x * x;
  float y2 = y * y;
  float z2 = z * z;
  float xy = x * y;
  float xz = x * z;
  float yz = y * z;
  float wx = w * x;
  float wy = w * y;
  float wz = w * z;

  Mat3 r;
  r.v[0] = 1.0f - 2.0f * (y2 + z2);
  r.v[1] = 2.0f * (xy - wz);
  r.v[2] = 2.0f * (xz + wy);
  r.v[3] = 2.0f * (xy + wz);
  r.v[4] = 1.0f - 2.0f * (x2 + z2);
  r.v[5] = 2.0f * (yz - wx);
  r.v[6] = 2.0f * (xz - wy);
  r.v[7] = 2.0f * (yz + wx);
  r.v[8] = 1.0f - 2.0f * (x2 + y2);
  return r;
}

inline Mat3 matrix_from_rotation_scale(Mat3 r, float3 scale, bool precision) {
  float3 s = scale;
  if (precision) {
    s = 1.0f / s;
  }

  float ss[3] = {s.x * s.x, s.y * s.y, s.z * s.z};
  Mat3 out;
  for (uint row = 0; row < 3; ++row) {
    for (uint col = 0; col < 3; ++col) {
      float value = 0.0f;
      for (uint k = 0; k < 3; ++k) {
        value += r.v[row * 3 + k] * ss[k] * r.v[col * 3 + k];
      }
      out.v[row * 3 + col] = value;
    }
  }
  return out;
}

inline void write_matrix(Mat3 matrix, bool triu, device float* out, uint idx) {
  if (triu) {
    uint base = idx * 6;
    out[base] = matrix.v[0];
    out[base + 1] = matrix.v[1];
    out[base + 2] = matrix.v[2];
    out[base + 3] = matrix.v[4];
    out[base + 4] = matrix.v[5];
    out[base + 5] = matrix.v[8];
  } else {
    uint base = idx * 9;
    for (uint i = 0; i < 9; ++i) {
      out[base + i] = matrix.v[i];
    }
  }
}

inline float matrix_cotangent_dot(Mat3 matrix,
                                  bool triu,
                                  const device float* cot,
                                  uint idx) {
  if (triu) {
    uint base = idx * 6;
    return matrix.v[0] * cot[base] +
           matrix.v[1] * cot[base + 1] +
           matrix.v[2] * cot[base + 2] +
           matrix.v[4] * cot[base + 3] +
           matrix.v[5] * cot[base + 4] +
           matrix.v[8] * cot[base + 5];
  }
  uint base = idx * 9;
  float result = 0.0f;
  for (uint i = 0; i < 9; ++i) {
    result += matrix.v[i] * cot[base + i];
  }
  return result;
}

inline float quat_scale_loss(float4 quat,
                             float3 scale,
                             bool triu,
                             const device float* v_covars,
                             const device float* v_precis,
                             bool use_v_covars,
                             bool use_v_precis,
                             uint idx) {
  Mat3 r = quat_to_rotmat(quat);
  float result = 0.0f;
  if (use_v_covars) {
    result += matrix_cotangent_dot(
        matrix_from_rotation_scale(r, scale, false), triu, v_covars, idx);
  }
  if (use_v_precis) {
    result += matrix_cotangent_dot(
        matrix_from_rotation_scale(r, scale, true), triu, v_precis, idx);
  }
  return result;
}

kernel void gsplat_quat_scale_to_covar_preci_forward_kernel(
    constant QuatScaleToCovarPreciKernelParams& params [[buffer(0)]],
    const device float* quats [[buffer(1)]],
    const device float* scales [[buffer(2)]],
    device float* covars [[buffer(3)]],
    device float* precis [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= params.n) {
    return;
  }

  uint quat_base = idx * 4;
  uint scale_base = idx * 3;
  float4 quat = float4(
      quats[quat_base],
      quats[quat_base + 1],
      quats[quat_base + 2],
      quats[quat_base + 3]);
  float3 scale = float3(
      scales[scale_base],
      scales[scale_base + 1],
      scales[scale_base + 2]);
  Mat3 r = quat_to_rotmat(quat);

  if (params.compute_covar != 0) {
    Mat3 covar = matrix_from_rotation_scale(r, scale, false);
    write_matrix(covar, params.triu != 0, covars, idx);
  }
  if (params.compute_preci != 0) {
    Mat3 preci = matrix_from_rotation_scale(r, scale, true);
    write_matrix(preci, params.triu != 0, precis, idx);
  }
}

kernel void gsplat_quat_scale_to_covar_preci_backward_kernel(
    constant QuatScaleToCovarPreciKernelParams& params [[buffer(0)]],
    const device float* quats [[buffer(1)]],
    const device float* scales [[buffer(2)]],
    const device float* v_covars [[buffer(3)]],
    const device float* v_precis [[buffer(4)]],
    device float* v_quats [[buffer(5)]],
    device float* v_scales [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= params.n) {
    return;
  }

  uint quat_base = idx * 4;
  uint scale_base = idx * 3;
  float4 quat = float4(
      quats[quat_base],
      quats[quat_base + 1],
      quats[quat_base + 2],
      quats[quat_base + 3]);
  float3 scale = float3(
      scales[scale_base],
      scales[scale_base + 1],
      scales[scale_base + 2]);
  bool triu = params.triu != 0;
  bool use_v_covars = params.compute_covar != 0;
  bool use_v_precis = params.compute_preci != 0;
  constexpr float eps = 1.0e-3f;

  for (uint axis = 0; axis < 4; ++axis) {
    float4 plus_quat = quat;
    float4 minus_quat = quat;
    plus_quat[axis] += eps;
    minus_quat[axis] -= eps;
    v_quats[quat_base + axis] =
        (quat_scale_loss(plus_quat, scale, triu, v_covars, v_precis,
                         use_v_covars, use_v_precis, idx) -
         quat_scale_loss(minus_quat, scale, triu, v_covars, v_precis,
                         use_v_covars, use_v_precis, idx)) /
        (2.0f * eps);
  }

  for (uint axis = 0; axis < 3; ++axis) {
    float3 plus_scale = scale;
    float3 minus_scale = scale;
    plus_scale[axis] += eps;
    minus_scale[axis] -= eps;
    v_scales[scale_base + axis] =
        (quat_scale_loss(quat, plus_scale, triu, v_covars, v_precis,
                         use_v_covars, use_v_precis, idx) -
         quat_scale_loss(quat, minus_scale, triu, v_covars, v_precis,
                         use_v_covars, use_v_precis, idx)) /
        (2.0f * eps);
  }
}
