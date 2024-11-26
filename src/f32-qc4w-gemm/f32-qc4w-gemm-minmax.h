// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, 2, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, 2, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, 2, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, 4, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, 4, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, 2, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, 2, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, 2, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, 2, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, 4, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neon_dup_ld64, 2, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neon_lane_ld64, 2, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, 2, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, 2, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, 4, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neon_dup_ld64, 2, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neon_lane_ld64, 2, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, 2, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, 2, true, 5, 8, 1, 1, 5, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc4w_gemm_minmax_ukernel_5x8__neon_lane_ld64, 2, true, 5, 8, 1, 1, 5, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, 2, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, 4, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc4w_gemm_minmax_ukernel_6x8__neon_dup_ld64, 2, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc4w_gemm_minmax_ukernel_6x8__neon_lane_ld64, 2, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc4w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, 2, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc4w_gemm_minmax_ukernel_1x8__sse41_dup, 4, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc4w_gemm_minmax_ukernel_3x8__sse41_dup, 4, true, 3, 8, 1, 1, 3, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc4w_gemm_minmax_ukernel_4x8__sse41_dup, 4, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc4w_gemm_minmax_ukernel_5x8__sse41_dup, 4, true, 5, 8, 1, 1, 5, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc4w_gemm_minmax_ukernel_6x8__sse41_dup, 4, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc4w_gemm_minmax_ukernel_1x16__avx_broadcast, 2, true, 1, 16, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc4w_gemm_minmax_ukernel_2x16__avx_broadcast, 2, true, 2, 16, 1, 1, 2, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc4w_gemm_minmax_ukernel_3x16__avx_broadcast, 2, true, 3, 16, 1, 1, 3, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc4w_gemm_minmax_ukernel_4x16__avx_broadcast, 2, true, 4, 16, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc4w_gemm_minmax_ukernel_5x16__avx_broadcast, 2, true, 5, 16, 1, 1, 5, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc4w_gemm_minmax_ukernel_6x16__avx_broadcast, 2, true, 6, 16, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc4w_gemm_minmax_ukernel_7x16__avx_broadcast, 2, true, 7, 16, 1, 1, 7, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc4w_gemm_minmax_ukernel_8x16__avx_broadcast, 2, true, 8, 16, 1, 1, 8, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc4w_gemm_minmax_ukernel_1x16__fma3_broadcast, 2, true, 1, 16, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc4w_gemm_minmax_ukernel_2x16__fma3_broadcast, 2, true, 2, 16, 1, 1, 2, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc4w_gemm_minmax_ukernel_3x16__fma3_broadcast, 2, true, 3, 16, 1, 1, 3, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc4w_gemm_minmax_ukernel_4x16__fma3_broadcast, 2, true, 4, 16, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc4w_gemm_minmax_ukernel_5x16__fma3_broadcast, 2, true, 5, 16, 1, 1, 5, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc4w_gemm_minmax_ukernel_6x16__fma3_broadcast, 2, true, 6, 16, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc4w_gemm_minmax_ukernel_7x16__fma3_broadcast, 2, true, 7, 16, 1, 1, 7, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc4w_gemm_minmax_ukernel_8x16__fma3_broadcast, 2, true, 8, 16, 1, 1, 8, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc4w_gemm_minmax_ukernel_1x16__avx2_broadcast, 2, true, 1, 16, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc4w_gemm_minmax_ukernel_2x16__avx2_broadcast, 2, true, 2, 16, 1, 1, 2, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc4w_gemm_minmax_ukernel_3x16__avx2_broadcast, 2, true, 3, 16, 1, 1, 3, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc4w_gemm_minmax_ukernel_4x16__avx2_broadcast, 2, true, 4, 16, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc4w_gemm_minmax_ukernel_5x16__avx2_broadcast, 2, true, 5, 16, 1, 1, 5, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc4w_gemm_minmax_ukernel_6x16__avx2_broadcast, 2, true, 6, 16, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc4w_gemm_minmax_ukernel_7x16__avx2_broadcast, 2, true, 7, 16, 1, 1, 7, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc4w_gemm_minmax_ukernel_8x16__avx2_broadcast, 2, true, 8, 16, 1, 1, 8, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc4w_gemm_minmax_ukernel_1x32__avx512skx_broadcast, 2, true, 1, 32, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc4w_gemm_minmax_ukernel_2x32__avx512skx_broadcast, 2, true, 2, 32, 1, 1, 2, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc4w_gemm_minmax_ukernel_3x32__avx512skx_broadcast, 2, true, 3, 32, 1, 1, 3, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc4w_gemm_minmax_ukernel_4x32__avx512skx_broadcast, 2, true, 4, 32, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc4w_gemm_minmax_ukernel_5x32__avx512skx_broadcast, 2, true, 5, 32, 1, 1, 5, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc4w_gemm_minmax_ukernel_6x32__avx512skx_broadcast, 2, true, 6, 32, 1, 1, 6, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc4w_gemm_minmax_ukernel_7x32__avx512skx_broadcast, 2, true, 7, 32, 1, 1, 7, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc4w_gemm_minmax_ukernel_8x32__avx512skx_broadcast, 2, true, 8, 32, 1, 1, 8, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, 2, true, 1, 4, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, 2, true, 2, 4, 1, 1, 2, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc4w_gemm_minmax_ukernel_4x2__wasm, 2, true, 4, 2, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, 2, true, 4, 4, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_GEMM(0, xnn_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, 2, true, 1, 4, 1, 1, 1, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, 2, true, 2, 4, 1, 1, 2, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc4w_gemm_minmax_ukernel_4x2__scalar, 2, true, 4, 2, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, 2, true, 4, 4, 1, 1, 4, false, float, struct xnn_f32_qc4w_minmax_params, xnn_init_f32_qc4w_minmax_scalar_params, xnn_pack_f32_qc4w_gemm_goi_w, NULL)
