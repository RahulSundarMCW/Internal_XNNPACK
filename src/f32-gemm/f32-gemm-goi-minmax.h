// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, NULL, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, 4, true, 1, 8, 1, 1, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, NULL, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, NULL, NULL)
#endif  // XNN_ARCH_ARM64

