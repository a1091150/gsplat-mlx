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

inline Mat3 cotangent_matrix(bool triu, const device float* cot, uint idx) {
  Mat3 out;
  for (uint i = 0; i < 9; ++i) {
    out.v[i] = 0.0f;
  }
  if (triu) {
    uint base = idx * 6;
    out.v[0] = cot[base];
    out.v[1] = 0.5f * cot[base + 1];
    out.v[2] = 0.5f * cot[base + 2];
    out.v[3] = 0.5f * cot[base + 1];
    out.v[4] = cot[base + 3];
    out.v[5] = 0.5f * cot[base + 4];
    out.v[6] = 0.5f * cot[base + 2];
    out.v[7] = 0.5f * cot[base + 4];
    out.v[8] = cot[base + 5];
  } else {
    uint base = idx * 9;
    for (uint i = 0; i < 9; ++i) {
      out.v[i] = cot[base + i];
    }
  }
  return out;
}

inline void quat_to_rotmat_vjp(float4 quat, Mat3 v_r, thread float4& v_quat) {
  float w = quat.x;
  float x = quat.y;
  float y = quat.z;
  float z = quat.w;
  float inv_norm = rsqrt(w * w + x * x + y * y + z * z);
  w *= inv_norm;
  x *= inv_norm;
  y *= inv_norm;
  z *= inv_norm;

  float4 v_quat_n = float4(0.0f);
  v_quat_n.y += 2.0f * y * v_r.v[1] + 2.0f * z * v_r.v[2] +
                2.0f * y * v_r.v[3] - 4.0f * x * v_r.v[4] -
                2.0f * w * v_r.v[5] + 2.0f * z * v_r.v[6] +
                2.0f * w * v_r.v[7] - 4.0f * x * v_r.v[8];
  v_quat_n.z += -4.0f * y * v_r.v[0] + 2.0f * x * v_r.v[1] +
                2.0f * w * v_r.v[2] + 2.0f * x * v_r.v[3] +
                2.0f * z * v_r.v[5] - 2.0f * w * v_r.v[6] +
                2.0f * z * v_r.v[7] - 4.0f * y * v_r.v[8];
  v_quat_n.w += -4.0f * z * v_r.v[0] - 2.0f * w * v_r.v[1] +
                2.0f * x * v_r.v[2] + 2.0f * w * v_r.v[3] -
                4.0f * z * v_r.v[4] + 2.0f * y * v_r.v[5] +
                2.0f * x * v_r.v[6] + 2.0f * y * v_r.v[7];
  v_quat_n.x += -2.0f * z * v_r.v[1] + 2.0f * y * v_r.v[2] +
                2.0f * z * v_r.v[3] - 2.0f * x * v_r.v[5] -
                2.0f * y * v_r.v[6] + 2.0f * x * v_r.v[7];

  float4 quat_n = float4(w, x, y, z);
  v_quat += (v_quat_n - dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

inline void quat_scale_vjp(float4 quat,
                           float3 scale,
                           Mat3 r,
                           Mat3 v_matrix,
                           bool precision,
                           thread float4& v_quat,
                           thread float3& v_scale) {
  float3 s = scale;
  if (precision) {
    s = 1.0f / s;
  }

  Mat3 m;
  Mat3 v_m;
  Mat3 v_r;
  for (uint i = 0; i < 9; ++i) {
    m.v[i] = 0.0f;
    v_m.v[i] = 0.0f;
    v_r.v[i] = 0.0f;
  }

  for (uint row = 0; row < 3; ++row) {
    for (uint col = 0; col < 3; ++col) {
      m.v[row * 3 + col] = r.v[row * 3 + col] * s[col];
    }
  }
  for (uint row = 0; row < 3; ++row) {
    for (uint col = 0; col < 3; ++col) {
      float value = 0.0f;
      for (uint k = 0; k < 3; ++k) {
        value += (v_matrix.v[row * 3 + k] + v_matrix.v[k * 3 + row]) *
                 m.v[k * 3 + col];
      }
      v_m.v[row * 3 + col] = value;
      v_r.v[row * 3 + col] = value * s[col];
    }
  }
  quat_to_rotmat_vjp(quat, v_r, v_quat);

  for (uint col = 0; col < 3; ++col) {
    float grad = 0.0f;
    for (uint row = 0; row < 3; ++row) {
      grad += r.v[row * 3 + col] * v_m.v[row * 3 + col];
    }
    if (precision) {
      v_scale[col] += -s[col] * s[col] * grad;
    } else {
      v_scale[col] += grad;
    }
  }
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

  Mat3 r = quat_to_rotmat(quat);
  float4 v_quat = float4(0.0f);
  float3 v_scale = float3(0.0f);
  if (use_v_covars) {
    quat_scale_vjp(
        quat, scale, r, cotangent_matrix(triu, v_covars, idx), false,
        v_quat, v_scale);
  }
  if (use_v_precis) {
    quat_scale_vjp(
        quat, scale, r, cotangent_matrix(triu, v_precis, idx), true,
        v_quat, v_scale);
  }

  v_quats[quat_base] = v_quat.x;
  v_quats[quat_base + 1] = v_quat.y;
  v_quats[quat_base + 2] = v_quat.z;
  v_quats[quat_base + 3] = v_quat.w;
  v_scales[scale_base] = v_scale.x;
  v_scales[scale_base + 1] = v_scale.y;
  v_scales[scale_base + 2] = v_scale.z;
}
