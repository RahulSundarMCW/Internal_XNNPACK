// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params)  \
  XNN_UKERNEL(arch_flags, ukernel, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype)   \
  XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1, 3, 3, 1, 1, 1, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1_acc2, 3, 3, 1, 1, 1, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1_acc3, 3, 3, 1, 1, 1, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1_acc4, 3, 3, 1, 1, 1, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1, 3, 3, 1, 1, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1_acc2, 3, 3, 1, 1, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_3x1, 3, 3, 1, 1, 3, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_4x1, 3, 3, 1, 1, 4, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_5x1, 3, 3, 1, 1, 5, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_6x1, 3, 3, 1, 1, 6, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1, 3, 3, 2, 1, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc2, 3, 3, 2, 1, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc3, 3, 3, 2, 1, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc4, 3, 3, 2, 1, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_2x1, 3, 3, 2, 1, 4, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_2x1_acc2, 3, 3, 2, 1, 4, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_3x1, 3, 3, 2, 1, 6, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_4x1, 3, 3, 2, 1, 8, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1, 5, 5, 1, 2, 1, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc2, 5, 5, 1, 2, 1, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc3, 5, 5, 1, 2, 1, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc4, 5, 5, 1, 2, 1, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc5, 5, 5, 1, 2, 1, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1, 5, 5, 1, 2, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1_acc2, 5, 5, 1, 2, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1_acc3, 5, 5, 1, 2, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_3x1, 5, 5, 1, 2, 3, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_3x1_acc2, 5, 5, 1, 2, 3, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1, 5, 5, 2, 2, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc2, 5, 5, 2, 2, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc3, 5, 5, 2, 2, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc4, 5, 5, 2, 2, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc5, 5, 5, 2, 2, 2, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1, 5, 5, 2, 2, 4, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1_acc2, 5, 5, 2, 2, 4, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1_acc3, 5, 5, 2, 2, 4, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_3x1, 5, 5, 2, 2, 6, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_3x1_acc2, 5, 5, 2, 2, 6, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif