// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ARCH_ARM
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7, 8, true, 1, 8, 1, 1, 1, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm, 8, true, 1, 8, 1, 1, 1, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, 8, true, 4, 8, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm, 8, true, 4, 8, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53, 8, true, 4, 8, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm, 8, true, 4, 8, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64, 8, true, 4, 8, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm, 8, true, 4, 8, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53, 8, true, 4, 16, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm, 8, true, 4, 16, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75, 8, true, 4, 16, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm, 8, true, 4, 16, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64, 8, true, 4, 16, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm, 8, true, 4, 16, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, 8, true, 1, 8, 1, 1, 1, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, 8, true, 1, 16, 1, 1, 1, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu16_ukernel_1x16__neon_mlal_lane, 8, true, 1, 16, 1, 1, 1, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu16_scalar_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane, 8, true, 2, 8, 1, 1, 2, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, 8, true, 2, 16, 1, 1, 2, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane, 8, true, 3, 8, 1, 1, 3, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane, 8, true, 3, 16, 1, 1, 3, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, 8, true, 4, 8, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, 8, true, 4, 16, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane, 8, true, 6, 8, 1, 1, 6, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, 8, true, 6, 16, 1, 1, 6, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_pack_qu8_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

XNN_GEMM(0, xnn_qu8_gemm_minmax_rndnu_ukernel_1x2__scalar, 1, true, 1, 2, 1, 1, 1, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_scalar_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_qu8_gemm_minmax_rndnu_ukernel_1x4__scalar, 1, true, 1, 4, 1, 1, 1, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_scalar_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_qu8_gemm_minmax_rndnu_ukernel_2x2__scalar, 1, true, 2, 2, 1, 1, 2, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_scalar_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_qu8_gemm_minmax_rndnu_ukernel_2x4__scalar, 1, true, 2, 4, 1, 1, 2, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_scalar_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_qu8_gemm_minmax_rndnu_ukernel_3x2__scalar, 1, true, 3, 2, 1, 1, 3, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_scalar_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_qu8_gemm_minmax_rndnu_ukernel_3x4__scalar, 1, true, 3, 4, 1, 1, 3, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_scalar_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_qu8_gemm_minmax_rndnu_ukernel_4x2__scalar, 1, true, 4, 2, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_scalar_params, xnn_pack_qu8_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_qu8_gemm_minmax_rndnu_ukernel_4x4__scalar, 1, true, 4, 4, 1, 1, 4, false, uint8_t, union xnn_qu8_conv_minmax_params, xnn_init_qu8_conv_minmax_rndnu_scalar_params, xnn_pack_qu8_gemm_goi_w, NULL)
