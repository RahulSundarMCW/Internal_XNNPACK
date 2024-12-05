// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
// Arguments are:
// XNN_GEMM_MINMAX(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, is_igemm, datatype, params_type, init_fn, pack_fn)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_GEMM(XNN_ARCH_ARM, xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, 8, false, 1, 4, 8, 1, false, xnn_bf16_default_params, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w)
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
// XNN_GEMM(XNN_ARCH_ARM, )
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64