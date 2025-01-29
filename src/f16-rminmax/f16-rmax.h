// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, datatype, output_type, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, batch_tile, datatype, output_type, params_type)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, datatype, output_type, params_type) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, datatype, output_type, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_rmax_ukernel__neonfp16arith_u8, 8, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_rmax_ukernel__neonfp16arith_u16_acc2, 16, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_rmax_ukernel__neonfp16arith_u24_acc3, 24, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_rmax_ukernel__neonfp16arith_u32_acc2, 32, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4, 32, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512fp16, xnn_f16_rmax_ukernel__avx512fp16_u32, 32, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512fp16, xnn_f16_rmax_ukernel__avx512fp16_u64_acc2, 64, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512fp16, xnn_f16_rmax_ukernel__avx512fp16_u96_acc3, 96, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512fp16, xnn_f16_rmax_ukernel__avx512fp16_u128_acc2, 128, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512fp16, xnn_f16_rmax_ukernel__avx512fp16_u128_acc4, 128, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f16_rmax_ukernel__avx512skx_u16, 16, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f16_rmax_ukernel__avx512skx_u32_acc2, 32, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f16_rmax_ukernel__avx512skx_u48_acc3, 48, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f16_rmax_ukernel__avx512skx_u64_acc2, 64, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_f16_rmax_ukernel__avx512skx_u64_acc4, 64, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_f16c, xnn_f16_rmax_ukernel__f16c_u32, 32, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

XNN_UKERNEL_WITH_PARAMS(0, xnn_f16_rmax_ukernel__scalar_u1, 1, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f16_rmax_ukernel__scalar_u2_acc2, 2, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f16_rmax_ukernel__scalar_u3_acc3, 3, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f16_rmax_ukernel__scalar_u4_acc2, 4, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f16_rmax_ukernel__scalar_u4_acc4, 4, xnn_float16, xnn_float16, struct xnn_f16_default_params, NULL)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
