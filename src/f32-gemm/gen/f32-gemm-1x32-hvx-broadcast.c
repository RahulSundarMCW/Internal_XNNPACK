// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/hvx-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/simd/f32-hvx.h"

#include "xnnpack/gemm.h"

void xnn_f32_gemm_ukernel_1x32__hvx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    HVX_Vector vacc0x0 = xnn_load_f32(w + 0);
    w += 32;

    size_t k = kc;
    do {
      XNN_SIMD_CONST_F32(va0, *(uint32_t *)a0);
      a0 += 1;

      const HVX_Vector vb0 = *((const HVX_Vector *)(w));
      w += 32;

      vacc0x0 = xnn_fmadd_qf32(va0, vb0, vacc0x0);

      k -= sizeof(float);
    } while (k != 0);

    if XNN_LIKELY(nc >= 32) {
      *((HVX_UVector *)c0) = vacc0x0;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 32;
    } else {
      xnn_store_tail_f32(c0, vacc0x0, nc);
      nc = 0;
    }
  } while (nc != 0);
}
