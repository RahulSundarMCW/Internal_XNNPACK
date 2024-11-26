// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_igemm_minmax_ukernel_1x8__avx2_broadcast, 1, true, 1, 8, 1, 1, 1, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_igemm_minmax_ukernel_1x16__avx2_broadcast, 1, true, 1, 16, 1, 1, 1, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_igemm_minmax_ukernel_3x16__avx2_broadcast, 1, true, 3, 16, 1, 1, 3, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_igemm_minmax_ukernel_4x8__avx2_broadcast, 1, true, 4, 8, 1, 1, 4, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_igemm_minmax_ukernel_4x16__avx2_broadcast, 1, true, 4, 16, 1, 1, 4, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_igemm_minmax_ukernel_5x8__avx2_broadcast, 1, true, 5, 8, 1, 1, 5, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_igemm_minmax_ukernel_5x16__avx2_broadcast, 1, true, 5, 16, 1, 1, 5, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_igemm_minmax_ukernel_6x8__avx2_broadcast, 1, true, 6, 8, 1, 1, 6, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_conv_goki_w, NULL)
XNN_GEMM(xnn_arch_x86_avx2, xnn_f16_f32acc_igemm_minmax_ukernel_7x8__avx2_broadcast, 1, true, 7, 8, 1, 1, 7, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params, xnn_pack_f16_conv_goki_w, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

