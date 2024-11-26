// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x4c16s2__aarch64_neondot, 2, true, 1, 4, 16, 2, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_kai_qs4_weights_and_biases, xnn_packed_stride_kai_qs4_weights_and_biases)
XNN_GEMM(xnn_arch_arm_neon_dot, xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot, 2, true, 1, 8, 16, 2, 1, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_kai_qs4_weights_and_biases, xnn_packed_stride_kai_qs4_weights_and_biases)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qp8_f32_qc4w_gemm_minmax_ukernel_4x4c16s2__neoni8mm, 2, true, 4, 4, 16, 2, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_kai_qs4_weights_and_biases, xnn_packed_stride_kai_qs4_weights_and_biases)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qp8_f32_qc4w_gemm_minmax_ukernel_4x8c16s2__neoni8mm, 2, true, 4, 8, 16, 2, 4, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_kai_qs4_weights_and_biases, xnn_packed_stride_kai_qs4_weights_and_biases)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x4c16s2__neoni8mm_mstep2, 2, true, 8, 4, 16, 2, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_kai_qs4_weights_and_biases, xnn_packed_stride_kai_qs4_weights_and_biases)
XNN_GEMM(xnn_arch_arm_neon_i8mm, xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x8c16s2__neoni8mm_mstep2, 2, true, 8, 8, 16, 2, 8, false, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params, xnn_pack_kai_qs4_weights_and_biases, xnn_packed_stride_kai_qs4_weights_and_biases)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM64)

