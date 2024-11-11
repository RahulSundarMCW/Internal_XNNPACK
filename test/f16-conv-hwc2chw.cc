// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f16-conv-hwc2chw
//   Generator: tools/generate-conv-hwc2chw-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/conv.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "conv-hwc2chw-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, datatype, params_type, init_params) XNN_TEST_CONV_HWC2CHW_INPUT_WIDTH_EQ(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);\
XNN_TEST_CONV_HWC2CHW_INPUT_WIDTH_DIV(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                           \
XNN_TEST_CONV_HWC2CHW_INPUT_WIDTH_LT(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                            \
XNN_TEST_CONV_HWC2CHW_INPUT_WIDTH_GT(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                            \
XNN_TEST_CONV_HWC2CHW_OUTPUT_CHANNELS_LT(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                        \
XNN_TEST_CONV_HWC2CHW_OUTPUT_CHANNELS_GT(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                        \
XNN_TEST_CONV_HWC2CHW_OUTPUT_CHANNELS_DIV(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                       \
XNN_TEST_CONV_HWC2CHW_INPUT_HEIGHT_LT(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                           \
XNN_TEST_CONV_HWC2CHW_INPUT_HEIGHT_GT(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                           \
XNN_TEST_CONV_HWC2CHW_PADDING_TOP(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                               \
XNN_TEST_CONV_HWC2CHW_PADDING_BOTTOM(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                            \
XNN_TEST_CONV_HWC2CHW_OUTPUT_Y_START(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                            \
XNN_TEST_CONV_HWC2CHW_OUTPUT_Y_END(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                              \
XNN_TEST_CONV_HWC2CHW_QMIN(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);                                                                                                                                                                                                      \
XNN_TEST_CONV_HWC2CHW_QMAX(ukernel,arch_flags, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, init_params);
#include "f16-conv-hwc2chw/f16-conv-hwc2chw.h"
#undef XNN_UKERNEL_WITH_PARAMS
