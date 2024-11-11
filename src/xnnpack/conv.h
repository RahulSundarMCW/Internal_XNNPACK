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


#define DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t input_height,                             \
      size_t input_width,                              \
      size_t output_y_start,                           \
      size_t output_y_end,                             \
      const float* input,                              \
      const float* zero,                               \
      const float* weights,                            \
      float* output,                                   \
      size_t input_padding_top,                        \
      size_t output_channels,                          \
      size_t output_height_stride,                     \
      size_t output_width_stride,                      \
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__neon_2x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__neon_2x2)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__scalar_1x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__aarch64_neonfma_2x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__aarch64_neonfma_2x2)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__neon_2x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__neon_2x2)

DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__aarch64_neonfma_2x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__aarch64_neonfma_2x2)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__neon_2x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__neon_2x2)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__scalar_1x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x8__aarch64_neonfma_2x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x8__aarch64_neonfma_2x2)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x8__neon_2x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x8__neon_2x2)


#define XNN_UKERNEL(                                                                                                   \
  arch_flags, fn_name, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile,    \
  input_widths, datatype, params_type)                                                                                 \
  XNN_INTERNAL void fn_name(                                                                                           \
    size_t input_height, size_t input_width, size_t output_y_start, size_t output_y_end, const datatype* input,        \
    const datatype* zero, const datatype* weights, datatype* output, size_t input_padding_top, size_t output_channels, \
    size_t output_height_stride, size_t output_channel_stride,                                                         \
    const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f32-conv-hwc2chw/f32-conv-hwc2chw.h"
#include "f16-conv-hwc2chw/f16-conv-hwc2chw.h"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
