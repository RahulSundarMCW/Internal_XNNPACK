// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F16_IBILINEAR_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                            \
      size_t output_pixels,                             \
      size_t channels,                                  \
      const xnn_float16** input,               \
      size_t input_offset,                              \
      const xnn_float16* weights,              \
      xnn_float16* output,                     \
      size_t output_increment);

DECLARE_F16_IBILINEAR_UKERNEL_FUNCTION(xnn_f16_ibilinear_ukernel__fma3_c8)
DECLARE_F16_IBILINEAR_UKERNEL_FUNCTION(xnn_f16_ibilinear_ukernel__fma3_c16)

DECLARE_F16_IBILINEAR_UKERNEL_FUNCTION(xnn_f16_ibilinear_ukernel__neonfp16arith_c8)
DECLARE_F16_IBILINEAR_UKERNEL_FUNCTION(xnn_f16_ibilinear_ukernel__neonfp16arith_c16)


#define DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                            \
      size_t output_pixels,                             \
      size_t channels,                                  \
      const float** input,                              \
      size_t input_offset,                              \
      const float* weights,                             \
      float* output,                                    \
      size_t output_increment);

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__scalar_c1)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__scalar_c2)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__scalar_c4)

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__neon_c4)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__neon_c8)

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__neonfma_c4)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__neonfma_c8)

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__sse_c4)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__sse_c8)

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__wasmsimd_c4)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__wasmsimd_c8)

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_c4)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_c8)


#define DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t output_pixels,                            \
      size_t channels,                                 \
      const int8_t** input,                            \
      size_t input_offset,                             \
      const int16_t* weights,                          \
      int8_t* output,                                  \
      size_t output_increment);

DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__scalar_c1)
DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__scalar_c2)
DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__scalar_c4)

DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__neon_c8)
DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__neon_c16)

DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__sse2_c8)
DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__sse2_c16)

DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__sse41_c8)
DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__sse41_c16)

DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8)
DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c16)

DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c8)
DECLARE_S8_IBILINEAR_UKERNEL_FUNCTION(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c16)

#define DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t output_pixels,                            \
      size_t channels,                                 \
      const uint8_t** input,                           \
      size_t input_offset,                             \
      const int16_t* weights,                          \
      uint8_t* output,                                 \
      size_t output_increment);


DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__scalar_c1)
DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__scalar_c2)
DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__scalar_c4)

DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__neon_c8)
DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__neon_c16)

DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__sse2_c8)
DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__sse2_c16)

DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__sse41_c8)
DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__sse41_c16)

DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__wasmsimd_dot16x2_c8)
DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__wasmsimd_dot16x2_c16)

DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__wasmsimd_mul32_c8)
DECLARE_U8_IBILINEAR_UKERNEL_FUNCTION(xnn_u8_ibilinear_ukernel__wasmsimd_mul32_c16)

#define XNN_UKERNEL(arch_flags, fn_name, pixel_tile, channel_tile, datatype) \
  XNN_INTERNAL void fn_name(                           \
      size_t output_pixels,                            \
      size_t channels,                                 \
      const datatype** input,                          \
      size_t input_offset,                             \
      const datatype* weights,                         \
      datatype* output,                                \
      size_t output_increment);
#include "f16-ibilinear-chw/f16-ibilinear-chw.h"
#include "f32-ibilinear-chw/f32-ibilinear-chw.h"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
