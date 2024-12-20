// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This file contains internal functions that are not part of the public API.

#ifndef THIRD_PARTY_XNNPACK_SRC_XNNPACK_INTERNAL_H_
#define THIRD_PARTY_XNNPACK_SRC_XNNPACK_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"
#include "pthreadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

/// If set, try to pack the quantized values for use by a GEMM.
#define XNN_FLAG_MAYBE_PACK_FOR_GEMM 0x00000080
#define XNN_FLAG_MAYBE_PACK_FOR_QB4W_GEMM 0x00000100

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qc4w(
    size_t input_channels,              //
    size_t output_channels,             //
    size_t input_stride,                //
    size_t output_stride,               //
    uint8_t kernel_zero_point,          //
    const float* kernel_scale,          //
    const void* kernel,                 //
    const float* bias,                  //
    float output_min,                   //
    float output_max,                   //
    uint32_t flags,                     //
    xnn_code_cache_t code_cache,        //
    xnn_weights_cache_t weights_cache,  //
    xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qc4w(
    xnn_operator_t fully_connected_op,  //
    const int8_t* input,                //
    float* output);

enum xnn_status xnn_reshape_fully_connected_nc_qp8_f32_qc4w(
    xnn_operator_t fully_connected_op,  //
    size_t batch_size,                  //
    pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_f32_qp8(uint32_t flags,  //
                                              xnn_operator_t* convert_op_out);

enum xnn_status xnn_reshape_convert_nc_f32_qp8(xnn_operator_t convert_op,  //
                                               size_t batch_size,          //
                                               size_t channels,            //
                                               size_t input_stride,        //
                                               pthreadpool_t threadpool);

enum xnn_status xnn_setup_convert_nc_f32_qp8(xnn_operator_t convert_op,  //
                                             const float* input,         //
                                             int8_t* output);

enum xnn_status xnn_create_pack_lh_x32(uint32_t flags,
                                       xnn_operator_t* pack_lh_op_out);

enum xnn_status xnn_reshape_pack_lh_x32(xnn_operator_t pack_lh_op,
                                        size_t batch_size, size_t channels,
                                        size_t* output_size_bytes,
                                        pthreadpool_t threadpool);

enum xnn_status xnn_setup_pack_lh_x32(xnn_operator_t pack_lh_op,
                                      const void* input, void* output);

enum xnn_status xnn_define_pack_lh(xnn_subgraph_t subgraph, uint32_t input_id,
                                   uint32_t output_id, uint32_t flags);

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qb4w(
    size_t input_channels,              //
    size_t output_channels,             //
    size_t input_stride,                //
    size_t output_stride,               //
    size_t block_size,                  //
    uint8_t kernel_zero_point,          //
    const uint16_t* kernel_scale,       //
    const void* kernel,                 //
    const float* bias,                  //
    float output_min,                   //
    float output_max,                   //
    uint32_t flags,                     //
    xnn_code_cache_t code_cache,        //
    xnn_weights_cache_t weights_cache,  //
    xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qb4w(
    xnn_operator_t fully_connected_op,  //
    const int8_t* input,                //
    float* output);

enum xnn_status xnn_reshape_fully_connected_nc_qp8_f32_qb4w(
    xnn_operator_t fully_connected_op,  //
    size_t batch_size,                  //
    pthreadpool_t threadpool);

enum xnn_status xnn_create_fully_connected_nc_pf32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_code_cache_t code_cache, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_create_convolution2d_nhwc_pf32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_code_cache_t code_cache, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_INTERNAL_H_
