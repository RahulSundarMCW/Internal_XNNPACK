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

void xnn_f32_gemm_minmax_ukernel_4x64__hvx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  XNN_SIMD_CONST_F32(vmin, params->scalar.min);
  XNN_SIMD_CONST_F32(vmax, params->scalar.max);
  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    HVX_Vector vacc0x0 = xnn_load_f32(w + 0);
    HVX_Vector vacc0x1 = xnn_load_f32(w + 32);
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc1x1 = vacc0x1;
    HVX_Vector vacc2x0 = vacc0x0;
    HVX_Vector vacc2x1 = vacc0x1;
    HVX_Vector vacc3x0 = vacc0x0;
    HVX_Vector vacc3x1 = vacc0x1;
    w += 64;

    size_t k = kc;
    do {
      XNN_SIMD_CONST_F32(va0, *(uint32_t *)a0);
      a0 += 1;
      XNN_SIMD_CONST_F32(va1, *(uint32_t *)a1);
      a1 += 1;
      XNN_SIMD_CONST_F32(va2, *(uint32_t *)a2);
      a2 += 1;
      XNN_SIMD_CONST_F32(va3, *(uint32_t *)a3);
      a3 += 1;

      const HVX_Vector vb0 = *((const HVX_Vector *)(w));
      const HVX_Vector vb1 = *((const HVX_Vector *)(w + 32));
      w += 64;

      vacc0x0 = xnn_fmadd_qf32(va0, vb0, vacc0x0);
      vacc1x0 = xnn_fmadd_qf32(va1, vb0, vacc1x0);
      vacc2x0 = xnn_fmadd_qf32(va2, vb0, vacc2x0);
      vacc3x0 = xnn_fmadd_qf32(va3, vb0, vacc3x0);
      vacc0x1 = xnn_fmadd_qf32(va0, vb1, vacc0x1);
      vacc1x1 = xnn_fmadd_qf32(va1, vb1, vacc1x1);
      vacc2x1 = xnn_fmadd_qf32(va2, vb1, vacc2x1);
      vacc3x1 = xnn_fmadd_qf32(va3, vb1, vacc3x1);

      k -= sizeof(float);
    } while (k != 0);

    // clamp results with min & max
    vacc0x0 = Q6_Vw_vmax_VwVw(vmin, vacc0x0);
    vacc1x0 = Q6_Vw_vmax_VwVw(vmin, vacc1x0);
    vacc2x0 = Q6_Vw_vmax_VwVw(vmin, vacc2x0);
    vacc3x0 = Q6_Vw_vmax_VwVw(vmin, vacc3x0);
    vacc0x1 = Q6_Vw_vmax_VwVw(vmin, vacc0x1);
    vacc1x1 = Q6_Vw_vmax_VwVw(vmin, vacc1x1);
    vacc2x1 = Q6_Vw_vmax_VwVw(vmin, vacc2x1);
    vacc3x1 = Q6_Vw_vmax_VwVw(vmin, vacc3x1);

    vacc0x0 = Q6_Vw_vmin_VwVw(vmax, vacc0x0);
    vacc1x0 = Q6_Vw_vmin_VwVw(vmax, vacc1x0);
    vacc2x0 = Q6_Vw_vmin_VwVw(vmax, vacc2x0);
    vacc3x0 = Q6_Vw_vmin_VwVw(vmax, vacc3x0);
    vacc0x1 = Q6_Vw_vmin_VwVw(vmax, vacc0x1);
    vacc1x1 = Q6_Vw_vmin_VwVw(vmax, vacc1x1);
    vacc2x1 = Q6_Vw_vmin_VwVw(vmax, vacc2x1);
    vacc3x1 = Q6_Vw_vmin_VwVw(vmax, vacc3x1);
    if XNN_LIKELY(nc >= 64) {
      *((HVX_UVector *)c0) = vacc0x0;
      *((HVX_UVector *)(c0 + 32)) = vacc0x1;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      *((HVX_UVector *)c1) = vacc1x0;
      *((HVX_UVector *)(c1 + 32)) = vacc1x1;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      *((HVX_UVector *)c2) = vacc2x0;
      *((HVX_UVector *)(c2 + 32)) = vacc2x1;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      *((HVX_UVector *)c3) = vacc3x0;
      *((HVX_UVector *)(c3 + 32)) = vacc3x1;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 64;
    } else {
      if (nc & 32) {
        *((HVX_UVector *)c0) = vacc0x0;
        *((HVX_UVector *)c1) = vacc1x0;
        *((HVX_UVector *)c2) = vacc2x0;
        *((HVX_UVector *)c3) = vacc3x0;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;
        vacc3x0 = vacc3x1;

        c0 += 32;
        c1 += 32;
        c2 += 32;
        c3 += 32;
        nc ^= 32;
      }
      xnn_store_tail_f32(c0, vacc0x0, nc);
      xnn_store_tail_f32(c1, vacc1x0, nc);
      xnn_store_tail_f32(c2, vacc2x0, nc);
      xnn_store_tail_f32(c3, vacc3x0, nc);
      nc = 0;
    }
  } while (nc != 0);
}
