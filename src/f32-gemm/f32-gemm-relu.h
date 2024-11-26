// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat, 1, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat, 4, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd, 4, true, 1, 8, 1, 4, 1, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_loadsplat, 1, true, 3, 8, 1, 1, 3, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_splat, 4, true, 3, 8, 1, 1, 3, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_3x8s4__wasmsimd, 4, true, 3, 8, 1, 4, 3, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x2c4__wasmsimd, 4, true, 4, 2, 4, 1, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_loadsplat, 1, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat, 4, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x8s4__wasmsimd, 4, true, 4, 8, 1, 4, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_loadsplat, 1, true, 5, 8, 1, 1, 5, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat, 4, true, 5, 8, 1, 1, 5, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_5x8s4__wasmsimd, 4, true, 5, 8, 1, 4, 5, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_loadsplat, 1, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_splat, 4, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_6x8s4__wasmsimd, 4, true, 6, 8, 1, 4, 6, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat, 4, true, 1, 8, 1, 1, 1, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma, 4, true, 1, 8, 1, 4, 1, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 3, 8, 1, 1, 3, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_splat, 4, true, 3, 8, 1, 1, 3, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_3x8s4__wasmrelaxedsimd_fma, 4, true, 3, 8, 1, 4, 3, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x2c4__wasmrelaxedsimd_fma, 4, true, 4, 2, 4, 1, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat, 4, true, 4, 8, 1, 1, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x8s4__wasmrelaxedsimd_fma, 4, true, 4, 8, 1, 4, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 5, 8, 1, 1, 5, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_splat, 4, true, 5, 8, 1, 1, 5, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_5x8s4__wasmrelaxedsimd_fma, 4, true, 5, 8, 1, 4, 5, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat, 1, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat, 4, true, 6, 8, 1, 1, 6, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_6x8s4__wasmrelaxedsimd_fma, 4, true, 6, 8, 1, 4, 6, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_1x4__wasm, 1, true, 1, 4, 1, 1, 1, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_2x4__wasm, 1, true, 2, 4, 1, 1, 2, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x2__wasm, 1, true, 4, 2, 1, 1, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x4__wasm, 1, true, 4, 4, 1, 1, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_1x4__scalar, 1, true, 1, 4, 1, 1, 1, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_2x4__scalar, 1, true, 2, 4, 1, 1, 2, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x2__scalar, 1, true, 4, 2, 1, 1, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(0, xnn_f32_gemm_relu_ukernel_4x4__scalar, 1, true, 4, 4, 1, 1, 4, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)

#if XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)
XNN_GEMM(xnn_arch_riscv_vector, xnn_f32_gemm_relu_ukernel_1x4v__rvv, 1, true, 1, 4, 1, 1, 1, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
XNN_GEMM(xnn_arch_riscv_vector, xnn_f32_gemm_relu_ukernel_7x4v__rvv, 1, true, 7, 4, 1, 1, 7, false, float, struct xnn_f32_relu_params, NULL, xnn_pack_f32_gemm_goi_w, NULL)
#endif  // XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)

