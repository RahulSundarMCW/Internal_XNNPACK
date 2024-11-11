// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile, input_widths, datatype, params_type, init_params)  \
  XNN_UKERNEL(arch_flags, ukernel, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile, input_widths, datatype, params_type)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile, input_widths, datatype, params_type)   \
  XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, kernel_size, subsampling, padding_right, padding_left, input_channels, output_channels_tile, input_widths, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, 3, 2, 1, 1, 3, 4, 1, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2, 3, 2, 1, 1, 3, 4, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2, 3, 2, 1, 1, 3, 4, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, 3, 2, 1, 1, 3, 4, 1, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, 3, 2, 1, 1, 3, 4, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, 3, 2, 1, 1, 3, 4, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
