// Copyright 2023 Google LLC
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

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, datatype, output_type, params_type) \
  XNN_INTERNAL void fn_name(                       \
      size_t batch,                                \
      const datatype* input,                       \
      output_type* output,                         \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f16-f32acc-rsum/f16-f32acc-rsum.h"
#include "f16-rminmax/f16-rmax.h"
#include "f16-rminmax/f16-rmin.h"
#include "f16-rminmax/f16-rminmax.h"
#include "f16-rsum/f16-rsum.h"
#include "f32-rminmax/f32-rmax.h"
#include "f32-rminmax/f32-rmin.h"
#include "f32-rminmax/f32-rminmax.h"
#include "f32-rsum/f32-rsum.h"
#include "qs8-rsum/qs8-rsum.h"
#include "qu8-rsum/qu8-rsum.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, datatype, output_type, params_type) \
  XNN_INTERNAL void fn_name(                      \
      size_t batch,                               \
      const datatype* input,                      \
      output_type* output,                        \
      const void* params);
#include "u8-rmax/u8-rmax.h"
#undef XNN_UKERNEL

#define DECLARE_F32_RDSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t rows,                                  \
      size_t channels,                              \
      const float* input,                           \
      size_t input_stride,                          \
      const float* zero,                            \
      float* output,                                \
      const struct xnn_f32_scale_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx_c16)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx_c32)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx_c64)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx512f_c128)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__neon_c16)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__neon_c32)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__neon_c64)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__scalar_c4)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__sse_c16)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__sse_c32)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__sse_c64)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64)

#define DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                               \
      size_t rows,                                         \
      size_t channels,                                     \
      const xnn_float16* input,                   \
      size_t input_stride,                                 \
      const xnn_float16* zero,                    \
      float* output,                                       \
      const struct xnn_f16_f32acc_scale_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64)

#define DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t rows,                                  \
      size_t channels,                              \
      const int8_t* input,                          \
      size_t input_stride,                          \
      const int8_t* zero,                           \
      int32_t* output,                              \
      const struct xnn_qs8_rsum_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__avx2_c32)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__avx2_c64)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__avx512skx_c64)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__avx512skx_c128)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__neon_c16)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__neon_c32)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__neon_c64)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__scalar_c4)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__sse41_c16)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__sse41_c32)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__sse41_c64)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__wasmsimd_c16)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__wasmsimd_c32)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__wasmsimd_c64)

#define DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t rows,                                  \
      size_t channels,                              \
      const uint8_t* input,                          \
      size_t input_stride,                          \
      const uint8_t* zero,                           \
      uint32_t* output,                              \
      const struct xnn_qs8_rsum_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(xnn_qu8_rdsum_ukernel_7p7x__neon_u16)
DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(xnn_qu8_rdsum_ukernel_7p7x__neon_u32)
DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(xnn_qu8_rdsum_ukernel_7p7x__neon_u64)
DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4)
DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16)
DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32)
DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64)
DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(xnn_qu8_rdsum_ukernel_7p7x__wasmsimd_c16)
DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(xnn_qu8_rdsum_ukernel_7p7x__wasmsimd_c32)
DECLARE_QU8_RDSUM_UKERNEL_FUNCTION(xnn_qu8_rdsum_ukernel_7p7x__wasmsimd_c64)

#ifdef __cplusplus
}  // extern "C"
#endif
