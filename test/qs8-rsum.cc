// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: qs8-rsum
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "rsum-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, datatype, output_type, params_type, init_params)  \
XNN_TEST_REDUCE_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype, output_type, ukernel, init_params);            \
XNN_TEST_REDUCE_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype, output_type, ukernel, init_params);           \
XNN_TEST_REDUCE_BATCH_LT(ukernel, arch_flags, batch_tile, datatype, output_type, ukernel, init_params);            \
XNN_TEST_REDUCE_BATCH_GT(ukernel, arch_flags, batch_tile, datatype, output_type, ukernel, init_params);            \
XNN_TEST_REDUCE_SCALE(ukernel, arch_flags, batch_tile, datatype, output_type, ukernel, init_params);               \
XNN_TEST_REDUCE_OVERFLOW_ACCUMULATOR(ukernel, arch_flags, batch_tile, datatype, output_type, ukernel, init_params);
#include "qs8-rsum/qs8-rsum.h"
#undef XNN_UKERNEL_WITH_PARAMS
