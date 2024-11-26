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


#define XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr_, nr_, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride) \
  XNN_INTERNAL void fn_name(                        \
      size_t mr,                                    \
      size_t nr,                                    \
      size_t kc,                                    \
      size_t ks,                                    \
      const datatype** a,                           \
      const datatype* w,                            \
      datatype* c,                                  \
      size_t cm_stride,                             \
      size_t cn_stride,                             \
      size_t a_offset,                              \
      const datatype* zero,                         \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f16-igemm/f16-igemm-minmax.h"
#include "f16-f32acc-igemm/f16-f32acc-igemm-minmax.h"
#include "f32-igemm/f32-igemm-minmax.h"
#include "f32-igemm/f32-igemm-relu.h"
#undef XNN_GEMM

#define XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr_, nr_, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride) \
  XNN_INTERNAL void fn_name(                                    \
      size_t mr,                                                \
      size_t nr,                                                \
      size_t kc,                                                \
      size_t ks,                                                \
      const datatype** a,                                       \
      const void* w,                                            \
      datatype* c,                                              \
      size_t cm_stride,                                         \
      size_t cn_stride,                                         \
      size_t a_offset,                                          \
      const datatype* zero,                                     \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "qs8-qc8w-igemm/qs8-qc8w-igemm-minmax-fp32.h"
#include "qu8-igemm/qu8-igemm-minmax-fp32.h"
#include "qu8-igemm/qu8-igemm-minmax-rndnu.h"
#undef XNN_GEMM

#define XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr_, nr_, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride) \
  XNN_INTERNAL void fn_name(                                                      \
      size_t mr,                                                                  \
      size_t nr,                                                                  \
      size_t kc,                                                                  \
      size_t ks,                                                                  \
      const int8_t** a,                                                           \
      const void* w,                                                              \
      datatype*,                                                                  \
      size_t cm_stride,                                                           \
      size_t cn_stride,                                                           \
      size_t a_offset,                                                            \
      const int8_t* zero_sentinel,                                                \
      const int8_t* zero_data,                                                    \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],                 \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "qd8-f16-qc8w-igemm/qd8-f16-qc8w-igemm-minmax.h"
#include "qd8-f32-qc8w-igemm/qd8-f32-qc8w-igemm-minmax.h"
#undef XNN_GEMM

#ifdef __cplusplus
}  // extern "C"
#endif
