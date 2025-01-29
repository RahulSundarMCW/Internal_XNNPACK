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
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_rsum_ukernel__scalar_u1, 1, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_rsum_ukernel__scalar_u2, 2, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_rsum_ukernel__scalar_u4, 4, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qu8_rsum_ukernel__neon_u16, 16, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qu8_rsum_ukernel__neon_u32_acc2, 32, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qu8_rsum_ukernel__neon_u64_acc2, 64, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qu8_rsum_ukernel__neon_u64_acc4, 64, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_rsum_ukernel__sse2_u16, 16, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_rsum_ukernel__sse2_u32_acc2, 32, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_rsum_ukernel__sse2_u64_acc2, 64, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_rsum_ukernel__sse2_u64_acc4, 64, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qu8_rsum_ukernel__avx2_u32, 32, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qu8_rsum_ukernel__avx2_u64_acc2, 64, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qu8_rsum_ukernel__avx2_u128_acc2, 128, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qu8_rsum_ukernel__avx2_u128_acc4, 128, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_rsum_ukernel__wasmsimd_u8, 8, int8_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_rsum_ukernel__wasmsimd_u16_acc2, 16, int8_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_rsum_ukernel__wasmsimd_u32_acc2, 32, int8_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qs8_rsum_ukernel__wasmsimd_u32_acc4, 32, int8_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
