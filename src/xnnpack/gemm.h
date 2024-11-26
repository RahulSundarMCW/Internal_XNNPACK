// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif


#define XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr_, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride) \
  void fn_name(                                            \
      size_t mr,                                           \
      size_t nc,                                           \
      size_t kc,                                           \
      const datatype* a,                                   \
      size_t a_stride,                                     \
      const datatype* w,                                   \
      datatype* c,                                         \
      size_t cm_stride,                                    \
      size_t cn_stride,                                    \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "bf16-gemm/bf16-gemm-minmax.h"
#include "f16-f32acc-gemm/f16-f32acc-gemm-minmax.h"
#include "f16-gemm/f16-gemm-minmax.h"
#include "f32-gemm/f32-gemm-minmax.h"
#include "f32-gemm/f32-gemm-relu.h"
#include "f32-gemm/f32-gemm-goi-minmax.h"
#undef XNN_GEMM

#define XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr_, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride) \
  XNN_INTERNAL void fn_name(                                 \
      size_t mr,                                             \
      size_t nc,                                             \
      size_t kc,                                             \
      const datatype* a,                                     \
      size_t a_stride,                                       \
      const datatype* w,                                     \
      datatype* c,                                           \
      size_t cm_stride,                                      \
      size_t cn_stride,                                      \
      const datatype* acc,                                   \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f32-gemminc/f32-gemminc-minmax.h"
#undef XNN_GEMM

#define XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr_, nr_, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride) \
  XNN_INTERNAL void fn_name(                                   \
      size_t mr,                                               \
      size_t nr,                                               \
      size_t k,                                                \
      const datatype* a,                                       \
      size_t a_stride,                                         \
      const void* w,                                           \
      datatype* c,                                             \
      size_t cm_stride,                                        \
      size_t cn_stride,                                        \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f32-qc4w-gemm/f32-qc4w-gemm-minmax.h"
#include "qu8-gemm/qu8-gemm-minmax-rndnu.h"
#include "qu8-gemm/qu8-gemm-minmax-fp32.h"
#undef XNN_GEMM

#define XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr_, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride) \
  XNN_INTERNAL void fn_name(                                 \
      size_t mr,                                             \
      size_t nc,                                             \
      size_t kc,                                             \
      const datatype* a,                                     \
      size_t a_stride,                                       \
      const void* w,                                         \
      datatype* c,                                           \
      size_t cm_stride,                                      \
      size_t cn_stride,                                      \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f32-qc8w-gemm/f32-qc8w-gemm-minmax.h"
#include "f32-qc8w-gemm/f32-qc8w-gemm-relu.h"
#include "qs8-qc8w-gemm/qs8-qc8w-gemm-minmax-fp32.h"
#undef XNN_GEMM

#define XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr_, nr_, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride) \
  XNN_INTERNAL void fn_name(                                                                           \
      size_t mr,                                                                                       \
      size_t nr,                                                                                       \
      size_t k,                                                                                        \
      const int8_t* a,                                                                                 \
      size_t a_stride,                                                                                 \
      const void* w,                                                                                   \
      datatype* c,                                                                                     \
      size_t cm_stride,                                                                                \
      size_t cn_stride,                                                                                \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],                                      \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "qd8-f16-qc4w-gemm/qd8-f16-qc4w-gemm-minmax.h"
#include "qd8-f16-qb4w-gemm/qd8-f16-qb4w-gemm-minmax.h"
#include "qd8-f16-qc8w-gemm/qd8-f16-qc8w-gemm-minmax.h"
#include "qd8-f32-qb4w-gemm/qd8-f32-qb4w-gemm-minmax.h"
#include "qd8-f32-qc4w-gemm/qd8-f32-qc4w-gemm-minmax.h"
#include "qd8-f32-qc8w-gemm/qd8-f32-qc8w-gemm-minmax.h"
#undef XNN_GEMM

#define XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride) \
  XNN_INTERNAL void fn_name(                                       \
      size_t m,                                                    \
      size_t n,                                                    \
      size_t k,                                                    \
      const void* lhs_packed,                                      \
      const void* rhs_packed,                                      \
      datatype* dst,                                               \
      size_t dst_stride_row,                                       \
      size_t dst_stride_col,                                       \
      params minmax_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "qp8-f32-qb4w-gemm/qp8-f32-qb4w-gemm-minmax.h"
#include "qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-minmax.h"
#undef XNN_GEMM

#ifdef __cplusplus
}  // extern "C"
#endif
