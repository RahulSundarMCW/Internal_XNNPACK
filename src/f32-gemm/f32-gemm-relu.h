// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
// Arguments are:
// XNN_GEMM_MINMAX(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, is_igemm, datatype, params_type, init_fn, pack_fn)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat, 1, false, 1, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat, 4, false, 1, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd, 4, false, 1, 8, 1, 4, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat, 4, false, 4, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat, 4, false, 5, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_loadsplat, 1, false, 6, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_6x8s4__wasmsimd, 4, false, 6, 8, 1, 4, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_loadsplat, 1, 3, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_splat, 4, false, 3, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_3x8s4__wasmsimd, 4, false, 3, 8, 1, 4, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_4x2c4__wasmsimd, 4, false, 4, 2, 4, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_loadsplat, 1, false, 4, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_4x8s4__wasmsimd, 4, false, 4, 8, 1, 4, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_loadsplat, 1, false, 5, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_5x8s4__wasmsimd, 4, false, 5, 8, 1, 4, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMSIMD, xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_splat, 4, false, 6, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat, 4, false, 1, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_splat, 4, false, 3, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_3x8s4__wasmrelaxedsimd_fma, 4, false, 3, 8, 1, 4, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat, 4, false, 4, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_4x8s4__wasmrelaxedsimd_fma, 4, false, 4, 8, 1, 4, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_splat, 4, false, 5, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat, 4, false, 6, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, 1, false, 1, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma, 4, false, 1, 8, 1, 4, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat, 1, false, 3, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_4x2c4__wasmrelaxedsimd_fma, 4, false, 4, 2, 4, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat, 1, false, 4, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat, 1, false, 5, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_5x8s4__wasmrelaxedsimd_fma, 4, false, 5, 8, 1, 4, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat, 1, false, 6, 8, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_gemm_relu_ukernel_6x8s4__wasmrelaxedsimd_fma, 4, false, 6, 8, 1, 4, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(XNN_ARCH_WASM, xnn_f32_gemm_relu_ukernel_1x4__wasm, 1, false, 1, 4, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASM, xnn_f32_gemm_relu_ukernel_2x4__wasm, 1, false, 2, 4, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASM, xnn_f32_gemm_relu_ukernel_4x2__wasm, 1, false, 4, 2, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_WASM, xnn_f32_gemm_relu_ukernel_4x4__wasm, 1, false, 4, 4, , 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
XNN_GEMM(XNN_ARCH_RISCV, xnn_f32_gemm_relu_ukernel_1x4v__rvv, 1, false, 1, 4, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(XNN_ARCH_RISCV, xnn_f32_gemm_relu_ukernel_7x4v__rvv, 1, false, 7, 4, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

// SCALAR
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_1x4__scalar, 1, false, 1, 4, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x2__scalar, 1, false, 4, 2, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x4__scalar, 1, false, 4, 4, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_2x4__scalar, 1, false, 2, 4, 1, 1, false, float, struct xnn_f32_minmax_params , NULL, xnn_pack_f32_gemm_goi_w)
