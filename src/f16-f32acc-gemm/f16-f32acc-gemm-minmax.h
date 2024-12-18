// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
// Arguments are:
// XNN_GEMM_MINMAX(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, is_igemm, datatype, params_type, init_fn, pack_fn)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_gemm_minmax_ukernel_1x8__avx2_broadcast, 1, false, 1, 8, 1, 1, false, xnn_f16_default_params, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_gemm_minmax_ukernel_1x16__avx2_broadcast, 1, false, 1, 16, 1, 1, false, xnn_f16_default_params, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_gemm_minmax_ukernel_3x16__avx2_broadcast, 1, false, 3, 16, 1, 1, false, xnn_f16_default_params, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_gemm_minmax_ukernel_4x8__avx2_broadcast, 1, false, 4, 8, 1, 1, false, xnn_f16_default_params, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_gemm_minmax_ukernel_4x16__avx2_broadcast, 1, false, 4, 16, 1, 1, false, xnn_f16_default_params, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_gemm_minmax_ukernel_5x8__avx2_broadcast, 1, false, 5, 8, 1, 1, false, xnn_f16_default_params, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_gemm_minmax_ukernel_5x16__avx2_broadcast, 1, false, 5, 16, 1, 1, false, xnn_f16_default_params, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_gemm_minmax_ukernel_6x8__avx2_broadcast, 1, false, 6, 8, 1, 1, false, xnn_f16_default_params, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_gemm_minmax_ukernel_7x8__avx2_broadcast, 1, false, 7, 8, 1, 1, false, xnn_f16_default_params, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
