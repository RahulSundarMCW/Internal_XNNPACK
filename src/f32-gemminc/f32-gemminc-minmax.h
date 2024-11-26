// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53, 8, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75, 8, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm, 8, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53, 4, true, 1, 12, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, 8, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, 8, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, 2, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53, 4, true, 4, 12, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm, 8, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73, 8, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm, 8, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, 2, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_1x8__neon_dup_ld64, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_1x8__neon_lane_ld64, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_1x8__neonfma_dup_ld64, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_1x8s4__neon, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_1x8s4__neonfma, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, 2, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_4x8__neon_dup_ld64, 2, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_4x8__neon_dup_ld128, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_4x8__neon_lane_ld64, 2, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_4x8__neon_lane_ld128, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8__neonfma_dup_ld64, 2, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8__neonfma_dup_ld128, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_4x8s4__neon, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_4x8s4__neonfma, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, 2, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_5x8__neon_lane_ld64, 2, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, 2, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_6x8__neon_dup_ld64, 2, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_6x8__neon_dup_ld128, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_6x8__neon_lane_ld64, 2, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_6x8__neon_lane_ld128, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__neonfma_dup_ld64, 2, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8__neonfma_dup_ld128, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_6x8s4__neon, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_6x8s4__neonfma, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_gemminc_minmax_ukernel_8x8s4__neon, 4, true, 8, 8, 1, 4, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemminc_minmax_ukernel_8x8s4__neonfma, 4, true, 8, 8, 1, 4, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8__sse_dup, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8__sse_load1, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8s4__sse, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8__sse_dup, 4, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8__sse_load1, 1, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8s4__sse, 4, true, 3, 8, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8__sse_dup, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8__sse_load1, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8s4__sse, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8__sse_dup, 4, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8__sse_load1, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8s4__sse, 4, true, 5, 8, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_gemminc_minmax_ukernel_1x8__avx_broadcast, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_gemminc_minmax_ukernel_1x16__avx_broadcast, 1, true, 1, 16, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_gemminc_minmax_ukernel_3x16__avx_broadcast, 1, true, 3, 16, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_gemminc_minmax_ukernel_4x8__avx_broadcast, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_gemminc_minmax_ukernel_4x16__avx_broadcast, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_gemminc_minmax_ukernel_5x8__avx_broadcast, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_gemminc_minmax_ukernel_5x16__avx_broadcast, 1, true, 5, 16, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_gemminc_minmax_ukernel_6x8__avx_broadcast, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_gemminc_minmax_ukernel_7x8__avx_broadcast, 1, true, 7, 8, 1, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_1x8__fma3_broadcast, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_1x16__fma3_broadcast, 1, true, 1, 16, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_1x16s4__fma3_broadcast, 4, true, 1, 16, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_3x16__fma3_broadcast, 1, true, 3, 16, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_3x16s4__fma3_broadcast, 4, true, 3, 16, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_4x8__fma3_broadcast, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_4x16__fma3_broadcast, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_4x16s4__fma3_broadcast, 4, true, 4, 16, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_5x8__fma3_broadcast, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_5x16__fma3_broadcast, 1, true, 5, 16, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_5x16s4__fma3_broadcast, 4, true, 5, 16, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_6x8__fma3_broadcast, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_7x8__fma3_broadcast, 1, true, 7, 8, 1, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_gemminc_minmax_ukernel_8x8__fma3_broadcast, 1, true, 8, 8, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_GEMM(xnn_arch_x86_avx512f, xnn_f32_gemminc_minmax_ukernel_1x16__avx512f_broadcast, 1, true, 1, 16, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512f, xnn_f32_gemminc_minmax_ukernel_4x16__avx512f_broadcast, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512f, xnn_f32_gemminc_minmax_ukernel_5x16__avx512f_broadcast, 1, true, 5, 16, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512f, xnn_f32_gemminc_minmax_ukernel_6x16__avx512f_broadcast, 1, true, 6, 16, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512f, xnn_f32_gemminc_minmax_ukernel_7x16__avx512f_broadcast, 1, true, 7, 16, 1, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512f, xnn_f32_gemminc_minmax_ukernel_8x16__avx512f_broadcast, 1, true, 8, 16, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8__wasmsimd_arm_loadsplat, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8__wasmsimd_arm_splat, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8__wasmsimd_x86_loadsplat, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8__wasmsimd_x86_splat, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8s4__wasmsimd_arm, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8s4__wasmsimd_x86, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8__wasmsimd_arm_loadsplat, 1, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8__wasmsimd_arm_splat, 4, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8__wasmsimd_x86_loadsplat, 1, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8__wasmsimd_x86_splat, 4, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8s4__wasmsimd_arm, 4, true, 3, 8, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8s4__wasmsimd_x86, 4, true, 3, 8, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8__wasmsimd_arm_loadsplat, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8__wasmsimd_arm_splat, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8__wasmsimd_x86_loadsplat, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8__wasmsimd_x86_splat, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8s4__wasmsimd_arm, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8s4__wasmsimd_x86, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8__wasmsimd_arm_loadsplat, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8__wasmsimd_arm_splat, 4, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8__wasmsimd_x86_loadsplat, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8__wasmsimd_x86_splat, 4, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8s4__wasmsimd_arm, 4, true, 5, 8, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8s4__wasmsimd_x86, 4, true, 5, 8, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8__wasmsimd_arm_loadsplat, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8__wasmsimd_arm_splat, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8__wasmsimd_x86_loadsplat, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8__wasmsimd_x86_splat, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8s4__wasmsimd_arm, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8s4__wasmsimd_x86, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8__wasmrelaxedsimd_splat, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8s4__wasmrelaxedsimd, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat, 4, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat, 1, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8__wasmrelaxedsimd_splat, 4, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8s4__wasmrelaxedsimd, 4, true, 3, 8, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma, 4, true, 3, 8, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8__wasmrelaxedsimd_splat, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8s4__wasmrelaxedsimd, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat, 4, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8__wasmrelaxedsimd_splat, 4, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8s4__wasmrelaxedsimd, 4, true, 5, 8, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma, 4, true, 5, 8, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8__wasmrelaxedsimd_splat, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8s4__wasmrelaxedsimd, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x4__wasm, 1, true, 1, 4, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_2x4__wasm, 1, true, 2, 4, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x4__wasm, 1, true, 4, 4, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_1x4__scalar, 1, true, 1, 4, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_2x4__scalar, 1, true, 2, 4, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemminc_minmax_ukernel_4x4__scalar, 1, true, 4, 4, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemminc_goi_w, NULL)
