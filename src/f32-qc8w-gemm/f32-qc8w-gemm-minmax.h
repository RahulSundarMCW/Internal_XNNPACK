// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64, 2, true, 4, 1, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128, 4, true, 4, 1, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, 2, true, 4, 2, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128, 4, true, 4, 2, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, 2, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, 2, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128, 4, true, 1, 16, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, 2, true, 4, 2, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64, 2, true, 4, 2, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, 2, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64, 2, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64, 2, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, 2, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128, 4, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64, 2, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64, 2, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, 2, true, 6, 2, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64, 2, true, 6, 2, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, 2, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64, 2, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64, 2, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, 2, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_dup, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_load1, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__sse41, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_3x8__sse41_dup, 4, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_3x8__sse41_load1, 1, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__sse41, 4, true, 3, 8, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__sse41, 4, true, 4, 2, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_dup, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_load1, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__sse41, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__sse41_dup, 4, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__sse41_load1, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__sse41, 4, true, 5, 8, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__sse41_dup, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__sse41_load1, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__sse41, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx_broadcast, 1, true, 1, 16, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx_broadcast, 1, true, 2, 16, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx_broadcast, 1, true, 3, 16, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx_broadcast, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx_broadcast, 1, true, 5, 16, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx_broadcast, 1, true, 6, 16, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx_broadcast, 1, true, 7, 16, 1, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx_broadcast, 1, true, 8, 16, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc8w_gemm_minmax_ukernel_1x16__fma3_broadcast, 1, true, 1, 16, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc8w_gemm_minmax_ukernel_2x16__fma3_broadcast, 1, true, 2, 16, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc8w_gemm_minmax_ukernel_3x16__fma3_broadcast, 1, true, 3, 16, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc8w_gemm_minmax_ukernel_4x16__fma3_broadcast, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc8w_gemm_minmax_ukernel_5x16__fma3_broadcast, 1, true, 5, 16, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc8w_gemm_minmax_ukernel_6x16__fma3_broadcast, 1, true, 6, 16, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc8w_gemm_minmax_ukernel_7x16__fma3_broadcast, 1, true, 7, 16, 1, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_fma3, xnn_f32_qc8w_gemm_minmax_ukernel_8x16__fma3_broadcast, 1, true, 8, 16, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__avx2_broadcast, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_1x16s4__avx2_broadcast, 4, true, 1, 16, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_2x16s4__avx2_broadcast, 4, true, 2, 16, 1, 4, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_3x16s4__avx2_broadcast, 4, true, 3, 16, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__avx2_broadcast, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_4x16s4__avx2_broadcast, 4, true, 4, 16, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__avx2_broadcast, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_5x16s4__avx2_broadcast, 4, true, 5, 16, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__avx2_broadcast, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_6x16s4__avx2_broadcast, 4, true, 6, 16, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_7x8__avx2_broadcast, 1, true, 7, 8, 1, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_8x8__avx2_broadcast, 1, true, 8, 8, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx2_broadcast, 1, true, 1, 16, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx2_broadcast, 1, true, 2, 16, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx2_broadcast, 1, true, 3, 16, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx2_broadcast, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx2_broadcast, 1, true, 5, 16, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx2_broadcast, 1, true, 6, 16, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx2_broadcast, 1, true, 7, 16, 1, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx2_broadcast, 1, true, 8, 16, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx512skx_broadcast, 1, true, 1, 16, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_1x32__avx512skx_broadcast, 1, true, 1, 32, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx512skx_broadcast, 1, true, 2, 16, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_2x32__avx512skx_broadcast, 1, true, 2, 32, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx512skx_broadcast, 1, true, 3, 16, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_3x32__avx512skx_broadcast, 1, true, 3, 32, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx512skx_broadcast, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_4x32__avx512skx_broadcast, 1, true, 4, 32, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx512skx_broadcast, 1, true, 5, 16, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_5x32__avx512skx_broadcast, 1, true, 5, 32, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx512skx_broadcast, 1, true, 6, 16, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_6x32__avx512skx_broadcast, 1, true, 6, 32, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx512skx_broadcast, 1, true, 7, 16, 1, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_7x32__avx512skx_broadcast, 1, true, 7, 32, 1, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx512skx_broadcast, 1, true, 8, 16, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_f32_qc8w_gemm_minmax_ukernel_8x32__avx512skx_broadcast, 1, true, 8, 32, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmsimd_arm, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmsimd_x86, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat, 1, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat, 4, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat, 1, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat, 4, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmsimd_arm, 4, true, 3, 8, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmsimd_x86, 4, true, 3, 8, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmsimd_arm, 4, true, 4, 2, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmsimd_x86, 4, true, 4, 2, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmsimd_arm, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmsimd_x86, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat, 4, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat, 4, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmsimd_arm, 4, true, 5, 8, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmsimd_x86, 4, true, 5, 8, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmsimd_arm, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmsimd_x86, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat, 1, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma, 4, true, 1, 8, 1, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat, 4, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat, 1, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat, 4, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd, 4, true, 3, 8, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma, 4, true, 3, 8, 1, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd, 4, true, 4, 2, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma, 4, true, 4, 2, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma, 4, true, 4, 8, 1, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat, 4, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat, 1, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat, 4, true, 5, 8, 1, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd, 4, true, 5, 8, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma, 4, true, 5, 8, 1, 4, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat, 1, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat, 4, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma, 4, true, 6, 8, 1, 4, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, 1, true, 1, 4, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, 1, true, 2, 4, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm, 1, true, 4, 2, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, 1, true, 4, 4, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, 1, true, 1, 4, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, 1, true, 2, 4, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar, 1, true, 4, 2, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, 1, true, 4, 4, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_qs8w_gemm_goi_w, NULL)
