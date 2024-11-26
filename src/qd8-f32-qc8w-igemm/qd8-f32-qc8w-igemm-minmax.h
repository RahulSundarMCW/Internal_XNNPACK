// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__avx512amx, 64, true, 1, 16, 4, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c4__avx512amx, 64, true, 7, 16, 4, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x16c4__avx512amx, 64, true, 16, 16, 4, 1, 16, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x16c4__avx512amx_prfm, 64, true, 16, 16, 4, 1, 16, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x32c4__avx512amx, 64, true, 1, 32, 4, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x32c4__avx512amx, 64, true, 7, 32, 4, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x32c4__avx512amx, 64, true, 16, 32, 4, 1, 16, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x32c4__avx512amx_prfm, 64, true, 16, 32, 4, 1, 16, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x64c4__avx512amx, 64, true, 1, 64, 4, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x64c4__avx512amx, 64, true, 7, 64, 4, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x64c4__avx512amx, 64, true, 16, 64, 4, 1, 16, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512amx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x64c4__avx512amx_prfm, 64, true, 16, 64, 4, 1, 16, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, 2, true, 1, 2, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4__scalar, 2, true, 1, 4, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8__scalar, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, 2, true, 2, 2, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4__scalar, 2, true, 2, 4, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8__scalar, 2, true, 2, 8, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4__scalar, 2, true, 4, 4, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8__neon_mlal_lane, 8, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8__neon_mlal_lane_prfm, 8, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal, 16, true, 1, 8, 2, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c4__neondot, 4, true, 1, 8, 4, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__aarch64_neondot_ld128, 8, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__neondot_ld64, 8, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, 16, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16__neon_mlal_lane, 8, true, 1, 16, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16__neon_mlal_lane_prfm, 8, true, 1, 16, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot, 4, true, 1, 16, 4, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128, 8, true, 1, 16, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neondot_ld64, 8, true, 1, 16, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm, 16, true, 1, 16, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x32c4__neondot, 4, true, 1, 32, 4, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8__neon_mlal_lane, 8, true, 2, 8, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8__neon_mlal_lane_prfm, 8, true, 2, 8, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal, 16, true, 2, 8, 2, 4, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c4__neondot, 4, true, 2, 8, 4, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__neoni8mm, 16, true, 2, 8, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x16__neon_mlal_lane, 8, true, 2, 16, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x16__neon_mlal_lane_prfm, 8, true, 2, 16, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x16c4__neondot, 4, true, 2, 16, 4, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, 16, true, 2, 16, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x32c4__neondot, 4, true, 2, 32, 4, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8__neon_mlal_lane, 8, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8__neon_mlal_lane_prfm, 8, true, 3, 8, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__neoni8mm, 16, true, 3, 8, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x16__neon_mlal_lane, 8, true, 3, 16, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x16__neon_mlal_lane_prfm, 8, true, 3, 16, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x16c8__neoni8mm, 16, true, 3, 16, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8__neon_mlal_lane, 8, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8__neon_mlal_lane_prfm, 8, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55, 8, true, 4, 8, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__neondot, 4, true, 4, 8, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__neoni8mm, 16, true, 4, 8, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16__neon_mlal_lane, 8, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16__neon_mlal_lane_prfm, 8, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, 16, true, 4, 16, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128, 16, true, 4, 16, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__neondot, 4, true, 4, 16, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, 16, true, 4, 16, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x32c4__neondot, 4, true, 4, 32, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8__neon_mlal_lane, 8, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8__neon_mlal_lane_prfm, 8, true, 6, 8, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8c4__neondot, 4, true, 6, 8, 4, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8c8__neoni8mm, 16, true, 6, 8, 8, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x16__neon_mlal_lane, 8, true, 6, 16, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x16__neon_mlal_lane_prfm, 8, true, 6, 16, 1, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x16c4__neondot, 4, true, 6, 16, 4, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x16c8__neoni8mm, 16, true, 6, 16, 8, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x32c4__neondot, 4, true, 6, 32, 4, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c4__neondot, 4, true, 8, 8, 4, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__neoni8mm, 16, true, 8, 8, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c4__neondot, 4, true, 8, 16, 4, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, 16, true, 8, 16, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x32c4__neondot, 4, true, 8, 32, 4, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512skx, 8, true, 1, 16, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c8__avx512skx, 8, true, 5, 16, 8, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c8__avx512skx, 8, true, 7, 16, 8, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__avx512skx, 8, true, 8, 16, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512skx_prfm, 8, true, 1, 16, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c8__avx512skx_prfm, 8, true, 5, 16, 8, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c8__avx512skx_prfm, 8, true, 7, 16, 8, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__avx512skx_prfm, 8, true, 8, 16, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__avx512vnni, 8, true, 1, 16, 4, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__avx512vnni, 8, true, 4, 16, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c4__avx512vnni, 8, true, 5, 16, 4, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c4__avx512vnni, 8, true, 7, 16, 4, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c4__avx512vnni, 8, true, 8, 16, 4, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x16c4__avx512vnni, 8, true, 9, 16, 4, 1, 9, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x16c4__avx512vnni, 8, true, 10, 16, 4, 1, 10, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x16c4__avx512vnni, 8, true, 12, 16, 4, 1, 12, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x16c4__avx512vnni, 8, true, 14, 16, 4, 1, 14, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__avx512vnni_prfm, 8, true, 1, 16, 4, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__avx512vnni_prfm, 8, true, 4, 16, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c4__avx512vnni_prfm, 8, true, 5, 16, 4, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c4__avx512vnni_prfm, 8, true, 7, 16, 4, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c4__avx512vnni_prfm, 8, true, 8, 16, 4, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x16c4__avx512vnni_prfm, 8, true, 9, 16, 4, 1, 9, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x16c4__avx512vnni_prfm, 8, true, 10, 16, 4, 1, 10, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x16c4__avx512vnni_prfm, 8, true, 12, 16, 4, 1, 12, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x16c4__avx512vnni_prfm, 8, true, 14, 16, 4, 1, 14, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512vnni, 16, true, 1, 16, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c8__avx512vnni, 16, true, 5, 16, 8, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c8__avx512vnni, 16, true, 7, 16, 8, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__avx512vnni, 16, true, 8, 16, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x16c8__avx512vnni, 16, true, 9, 16, 8, 1, 9, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x16c8__avx512vnni, 16, true, 10, 16, 8, 1, 10, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x16c8__avx512vnni, 16, true, 12, 16, 8, 1, 12, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x16c8__avx512vnni, 16, true, 14, 16, 8, 1, 14, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512vnni_prfm, 16, true, 1, 16, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c8__avx512vnni_prfm, 16, true, 5, 16, 8, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c8__avx512vnni_prfm, 16, true, 7, 16, 8, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__avx512vnni_prfm, 16, true, 8, 16, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x16c8__avx512vnni_prfm, 16, true, 9, 16, 8, 1, 9, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x16c8__avx512vnni_prfm, 16, true, 10, 16, 8, 1, 10, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x16c8__avx512vnni_prfm, 16, true, 12, 16, 8, 1, 12, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx512vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x16c8__avx512vnni_prfm, 16, true, 14, 16, 8, 1, 14, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx256vnni, 16, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avx256vnni, 16, true, 5, 8, 8, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avx256vnni, 16, true, 7, 8, 8, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avx256vnni, 16, true, 8, 8, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x8c8__avx256vnni, 16, true, 9, 8, 8, 1, 9, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x8c8__avx256vnni, 16, true, 10, 8, 8, 1, 10, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x8c8__avx256vnni, 16, true, 12, 8, 8, 1, 12, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x8c8__avx256vnni, 16, true, 14, 8, 8, 1, 14, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx256vnni_prfm, 16, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avx256vnni_prfm, 16, true, 5, 8, 8, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avx256vnni_prfm, 16, true, 7, 8, 8, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avx256vnni_prfm, 16, true, 8, 8, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x8c8__avx256vnni_prfm, 16, true, 9, 8, 8, 1, 9, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x8c8__avx256vnni_prfm, 16, true, 10, 8, 8, 1, 10, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x8c8__avx256vnni_prfm, 16, true, 12, 8, 8, 1, 12, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256vnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x8c8__avx256vnni_prfm, 16, true, 14, 8, 8, 1, 14, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avxvnni, 16, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, 16, true, 2, 8, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__avxvnni, 16, true, 3, 8, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni, 16, true, 4, 8, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, 16, true, 5, 8, 8, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8c8__avxvnni, 16, true, 6, 8, 8, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avxvnni, 16, true, 7, 8, 8, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avxvnni, 16, true, 8, 8, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avxvnni_prfm, 16, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni_prfm, 16, true, 2, 8, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__avxvnni_prfm, 16, true, 3, 8, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, 16, true, 4, 8, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, 16, true, 5, 8, 8, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8c8__avxvnni_prfm, 16, true, 6, 8, 8, 1, 6, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avxvnni_prfm, 16, true, 7, 8, 8, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avxvnni, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avxvnni_prfm, 16, true, 8, 8, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__wasm, 2, true, 1, 2, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4__wasm, 2, true, 1, 4, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8__wasm, 2, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__wasm, 2, true, 2, 2, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4__wasm, 2, true, 2, 4, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8__wasm, 2, true, 2, 8, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4__wasm, 2, true, 4, 4, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__wasmusdot, 8, true, 1, 16, 4, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__wasmusdot_u2, 8, true, 1, 16, 4, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__wasmusdot, 8, true, 4, 16, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__wasmusdot_u2, 8, true, 4, 16, 4, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmusdot, 8, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmusdot_u2, 8, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__wasmusdot, 8, true, 2, 8, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__wasmusdot_u2, 8, true, 2, 8, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__wasmusdot, 8, true, 3, 8, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__wasmusdot_u2, 8, true, 3, 8, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__wasmusdot, 8, true, 4, 8, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_usdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__wasmusdot_u2, 8, true, 4, 8, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmsdot, 8, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmsdot_u2, 8, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__wasmsdot, 8, true, 2, 8, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__wasmsdot_u2, 8, true, 2, 8, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__wasmsdot, 8, true, 3, 8, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__wasmsdot_u2, 8, true, 3, 8, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__wasmsdot, 8, true, 4, 8, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__wasmsdot_u2, 8, true, 4, 8, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c16__wasmsdot, 16, true, 1, 4, 16, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c16__wasmsdot, 16, true, 2, 4, 16, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c16__wasmsdot, 16, true, 3, 4, 16, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_wasm_sdot, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c16__wasmsdot, 16, true, 4, 4, 16, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld64, 8, true, 1, 4, 2, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld128, 8, true, 1, 4, 2, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64, 8, true, 1, 4, 2, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, 8, true, 1, 4, 2, 4, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64, 8, true, 1, 4, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld128, 8, true, 1, 4, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld64, 8, true, 2, 4, 2, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld128, 8, true, 2, 4, 2, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, 8, true, 2, 4, 2, 4, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128, 8, true, 2, 4, 2, 4, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64, 8, true, 2, 4, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld128, 8, true, 2, 4, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld64, 8, true, 3, 4, 2, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128, 8, true, 3, 4, 2, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, 8, true, 3, 4, 2, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128, 8, true, 3, 4, 2, 4, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64, 8, true, 3, 4, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld128, 8, true, 3, 4, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld64, 8, true, 4, 4, 2, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld128, 8, true, 4, 4, 2, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64, 8, true, 4, 4, 2, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128, 8, true, 4, 4, 2, 4, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64, 8, true, 4, 4, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld128, 8, true, 4, 4, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_GEMM(xnn_arch_x86_avx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__avx_ld64, 8, true, 1, 4, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__avx_ld128, 8, true, 1, 4, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse2_ld64, 8, true, 1, 4, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse2_ld128, 8, true, 1, 4, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse41_ld64, 8, true, 1, 4, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse41_ld128, 8, true, 1, 4, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__avx_ld64, 8, true, 2, 4, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__avx_ld128, 8, true, 2, 4, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__sse2_ld64, 8, true, 2, 4, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__sse2_ld128, 8, true, 2, 4, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__sse41_ld64, 8, true, 2, 4, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__sse41_ld128, 8, true, 2, 4, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__avx_ld64, 8, true, 3, 4, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__avx_ld128, 8, true, 3, 4, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse2_ld64, 8, true, 3, 4, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse2_ld128, 8, true, 3, 4, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse41_ld64, 8, true, 3, 4, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse41_ld128, 8, true, 3, 4, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__avx_ld64, 8, true, 4, 4, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__avx_ld128, 8, true, 4, 4, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse2_ld64, 8, true, 4, 4, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(0, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse2_ld128, 8, true, 4, 4, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse41_ld64, 8, true, 4, 4, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_sse4_1, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse41_ld128, 8, true, 4, 4, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx2, 8, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__avx2, 8, true, 2, 8, 8, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__avx2, 8, true, 3, 8, 8, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__avx2, 8, true, 4, 8, 8, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_GEMM(xnn_arch_x86_avx256skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx256skx, 8, true, 1, 8, 8, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avx256skx, 8, true, 5, 8, 8, 1, 5, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avx256skx, 8, true, 7, 8, 8, 1, 7, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx256skx, xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avx256skx, 8, true, 8, 8, 8, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w, NULL)
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

