// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, 8, true, 1, 4, 8, 1, 1, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, 8, true, 2, 4, 8, 1, 2, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, 8, true, 3, 4, 8, 1, 3, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, 8, true, 4, 4, 8, 1, 4, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, 8, true, 5, 4, 8, 1, 5, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, 8, true, 1, 4, 8, 1, 1, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, 8, true, 2, 4, 8, 1, 2, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, 8, true, 3, 4, 8, 1, 3, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, 8, true, 4, 4, 8, 1, 4, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, 8, true, 5, 4, 8, 1, 5, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, 8, true, 1, 8, 2, 1, 1, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, 8, true, 4, 8, 2, 1, 4, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, 8, true, 5, 8, 2, 1, 5, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, 8, true, 6, 8, 2, 1, 6, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, 8, true, 1, 4, 8, 1, 1, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, 8, true, 2, 4, 8, 1, 2, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, 8, true, 3, 4, 8, 1, 3, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, 8, true, 4, 4, 8, 1, 4, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, 8, true, 5, 4, 8, 1, 5, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, 8, true, 1, 4, 8, 1, 1, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, 8, true, 2, 4, 8, 1, 2, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, 8, true, 3, 4, 8, 1, 3, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, 8, true, 4, 4, 8, 1, 4, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_bf16, xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, 8, true, 5, 4, 8, 1, 5, false, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w, NULL)
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

