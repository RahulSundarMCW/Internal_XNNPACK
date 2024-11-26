// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, 4, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, 1, true, 8, 8, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, 1, true, 8, 8, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, 1, true, 8, 8, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, 1, true, 8, 8, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_ppmm_minmax_ukernel_4x8__neon, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_ppmm_minmax_ukernel_4x16__neon, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, 1, true, 4, 16, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, 1, true, 8, 8, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon_fma, xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, 1, true, 8, 8, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_ppmm_minmax_ukernel_8x8__neon, 1, true, 8, 8, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_arm_neon, xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, 1, true, 8, 8, 1, 1, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_GEMM(0, xnn_f32_ppmm_minmax_ukernel_4x8__sse, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, 1, true, 4, 8, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_GEMM(0, xnn_f32_ppmm_minmax_ukernel_2x4__scalar, 1, true, 2, 4, 1, 1, 2, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_ppmm_minmax_ukernel_3x3__scalar, 1, true, 3, 3, 1, 1, 3, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_ppmm_minmax_ukernel_4x2__scalar, 1, true, 4, 2, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_ppmm_minmax_ukernel_4x4__scalar, 1, true, 4, 4, 1, 1, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w, NULL)
