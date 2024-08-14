// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/scalar-rr2-p5-div.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


void xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vmagic_bias = 0x1.8000FEp23f;
  const float vminus_log2e = -0x1.715476p0f;
  const float vln2_hi = 0x1.62E400p-1f;
  const float vln2_lo = 0x1.7F7D1Cp-20f;
  const float vc5 = -0x1.0F9F9Cp-7f;
  const float vc4 = 0x1.573A1Ap-5f;
  const float vc3 = -0x1.555A80p-3f;
  const float vc2 = 0x1.FFFDC6p-2f;
  const float vc1 = -0x1.FFFFF6p-1f;
  const float vone = 1.0f;
  const float vdenorm_cutoff = 0x1.5D589Ep+6f;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    input += 2;

    const float vz0 = fabsf(vx0);
    const float vz1 = fabsf(vx1);

    float vn0 = vz0 * vminus_log2e + vmagic_bias;
    float vn1 = vz1 * vminus_log2e + vmagic_bias;

    const float vs0 = uint32_as_float(float_as_uint32(vn0) << 23);
    const float vs1 = uint32_as_float(float_as_uint32(vn1) << 23);

    vn0 -= vmagic_bias;
    vn1 -= vmagic_bias;

    float vt0 = vn0 * vln2_hi + vz0;
    float vt1 = vn1 * vln2_hi + vz1;

    vt0 = vn0 * vln2_lo + vt0;
    vt1 = vn1 * vln2_lo + vt1;

    float vp0 = vt0 * vc5 + vc4;
    float vp1 = vt1 * vc5 + vc4;

    vp0 = vt0 * vp0 + vc3;
    vp1 = vt1 * vp1 + vc3;

    vp0 = vt0 * vp0 + vc2;
    vp1 = vt1 * vp1 + vc2;

    vp0 = vt0 * vp0 + vc1;
    vp1 = vt1 * vp1 + vc1;

    vt0 *= vs0;
    vt1 *= vs1;

    const float ve0 = vt0 * vp0 + vs0;
    const float ve1 = vt1 * vp1 + vs1;

    const float vd0 = ve0 + vone;
    const float vd1 = ve1 + vone;

    float vf0 = ve0 / vd0;
    float vf1 = ve1 / vd1;

    if XNN_UNPREDICTABLE(vz0 > vdenorm_cutoff) {
      vf0 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz1 > vdenorm_cutoff) {
      vf1 = 0.0f;
    }

    if XNN_UNPREDICTABLE(vx0 > 0.0f) {
      vf0 = vone - vf0;
    }
    if XNN_UNPREDICTABLE(vx1 > 0.0f) {
      vf1 = vone - vf1;
    }

    output[0] = vf0;
    output[1] = vf1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float vx = *input;

    const float vz = fabsf(vx);

    float vn = vz * vminus_log2e + vmagic_bias;
    const float vs = uint32_as_float(float_as_uint32(vn) << 23);
    vn -= vmagic_bias;

    float vt = vn * vln2_hi + vz;
    vt = vn * vln2_lo + vt;

    float vp = vt * vc5 + vc4;
    vp = vt * vp + vc3;
    vp = vt * vp + vc2;
    vp = vt * vp + vc1;

    vt *= vs;
    const float ve = vt * vp + vs;
    const float vd = ve + vone;

    float vf = ve / vd;
    if XNN_UNPREDICTABLE(vz > vdenorm_cutoff) {
      vf = 0.0f;
    }
    if XNN_UNPREDICTABLE(vx > 0.0f) {
      vf = vone - vf;
    }

    *output = vf;
  }
}
