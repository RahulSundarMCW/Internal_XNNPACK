// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
// Arguments are:
// XNN_GEMM_MINMAX(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, is_igemm, datatype, params_type, init_fn, pack_fn)

#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64, 4, false, 1, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_4x8__asm_aarch64_neonfp16arith_ld64, 4, false, 4, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_6x8__asm_aarch64_neonfp16arith_ld64, 4, false, 6, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_8x8__asm_aarch64_neonfp16arith_ld64, 4, false, 8, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32, 2, false, 1, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64, 4, false, 1, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32, 2, false, 4, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64, 4, false, 4, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55, 4, false, 6, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0, 4, false, 6, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75, 4, false, 6, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32, 2, false, 6, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM64, xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64, 4, false, 6, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(XNN_ARCH_ARM, xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64, 4, false, 1, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM, xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64, 4, false, 4, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM, xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64, 4, false, 6, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM, xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64, 4, false, 8, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM, xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64, 4, false, 1, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM, xnn_f16_gemm_minmax_ukernel_4x16__neonfp16arith_ld64, 4, false, 4, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM, xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64, 4, false, 6, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
XNN_GEMM(XNN_ARCH_ARM, xnn_f16_gemm_minmax_ukernel_8x16__neonfp16arith_ld64, 4, false, 8, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_1x32__avx512fp16_broadcast, 1, false, 1, 32, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_4x32__avx512fp16_broadcast, 1, false, 4, 32, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_5x32__avx512fp16_broadcast, 1, false, 5, 32, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_6x32__avx512fp16_broadcast, 1, false, 6, 32, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_7x32__avx512fp16_broadcast, 1, false, 7, 32, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_8x32__avx512fp16_broadcast, 1, false, 8, 32, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_1x64__avx512fp16_broadcast, 1, false, 1, 64, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_4x64__avx512fp16_broadcast, 1, false, 4, 64, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_5x64__avx512fp16_broadcast, 1, false, 5, 64, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_6x64__avx512fp16_broadcast, 1, false, 6, 64, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_7x64__avx512fp16_broadcast, 1, false, 7, 64, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4)
XNN_GEMM(xnn_arch_x86_avx512fp16, xnn_f16_gemm_minmax_ukernel_8x64__avx512fp16_broadcast, 1, false, 8, 64, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4)
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_gemm_minmax_ukernel_1x8__avx2_broadcast, 1, false, 1, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_gemm_minmax_ukernel_4x8__avx2_broadcast, 1, false, 4, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_gemm_minmax_ukernel_5x8__avx2_broadcast, 1, false, 5, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_gemm_minmax_ukernel_6x8__avx2_broadcast, 1, false, 6, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_gemm_minmax_ukernel_7x8__avx2_broadcast, 1, false, 7, 8, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast, 1, false, 1, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_gemm_minmax_ukernel_3x16__avx2_broadcast, 1, false, 3, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast, 1, false, 4, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_gemm_minmax_ukernel_5x16__avx2_broadcast, 1, false, 5, 16, 1, 1, false, xnn_f16_default_params, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
