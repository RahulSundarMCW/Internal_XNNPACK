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
  XNN_INTERNAL void fn_name(                       \
      size_t mr,                                   \
      size_t nc,                                   \
      size_t kc,                                   \
      const datatype* a,                           \
      const datatype* w,                           \
      datatype* c,                                 \
      size_t cm_stride,                            \
      size_t cn_stride,                            \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f32-ppmm/f32-ppmm-minmax.h"
#undef XNN_GEMM

#ifdef __cplusplus
}  // extern "C"
#endif
