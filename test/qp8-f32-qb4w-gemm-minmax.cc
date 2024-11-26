// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: qp8-f32-qb4w-gemm-minmax
//   Generator: tools/generate-gemm-test.py

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/igemm.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/packw.h"
#include "xnnpack/ppmm.h"
#include "xnnpack/requantization.h"
#include "gemm-microkernel-tester.h"
#include "next_prime.h"

namespace {

std::vector<GemmTestParams> CreateTests(
    size_t k_block, bool is_pipelined,
    size_t mr, size_t nr, size_t kr, size_t sr,
    size_t mr_packed,
    bool is_igemm,
    std::function<void(GemmMicrokernelTester& tester)> test_func) {
  
  size_t adj_k_block = is_pipelined ? k_block * 2 : k_block;
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  size_t nr_scale = get_batch_scale<float>();
  if (nr_scale != 0) {
    nr = nr * nr_scale;
  }
  std::string nrs = std::to_string(nr);

  const GemmMicrokernelTester tester = GemmMicrokernelTester()
      .mr(mr).nr(nr).kr(kr).sr(sr).mr_packed(mr_packed);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .b_zero_point(8)
          .bl(32)
      , test_func));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .a_stride(xnnpack::NextPrime(k_block + 1))
            .b_zero_point(8)
            .bl(32)
        , test_func));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block).iterations(1)
          .b_zero_point(8)
          .bl(32)
      , test_func)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block).iterations(1)
          .b_zero_point(8)
          .bl(32)
      , test_func)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block).iterations(1)
          .b_zero_point(8)
          .bl(32)
      , test_func)
      .loop_n(1, nr));
  if (is_pipelined) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kb2s,
        tester.clone()
          .m(mr).n(nr).k(k_block * 2)
          .b_zero_point(8)
          .bl(32)
      , test_func));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_eq_" + kb2s + "_strided_a",
          tester.clone()
              .m(mr).n(nr).k(k_block * 2)
              .a_stride(xnnpack::NextPrime(k_block * 2 + 1))
              .b_zero_point(8)
            .bl(32)
          , test_func));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kb2s + "_subtile",
        tester.clone()
            .k(k_block * 2).iterations(1)
            .b_zero_point(8)
            .bl(32)
        , test_func)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "bl",
      tester.clone()
          .m(mr).n(nr).k(k_block * 12)
          .b_zero_point(8)
      , test_func)
      .loop_k(k_block, k_block * 12, k_block, LoopStepType::Linear)
      .loop_bl(32, k_block * 32, 32));

  return gemm_tests;
}

}  // namespace


template <typename T>
auto GetValue(T* ptr) -> std::optional<T> {
    if (ptr) return *ptr;
    return std::nullopt;
}

template <typename T>
auto GetValue(T val) -> std::optional<T> {
    return val;
}

template <typename... Args>
void CallTestWithValidArgs(GemmMicrokernelTester& tester, Args... args) {
    auto validArgsTuple = std::tuple_cat(
        ([](auto&& arg) {
            auto opt_val = GetValue(arg);
            if (opt_val) return std::make_tuple(*opt_val);
            return std::tuple<>();
        }(args))...
    );
    std::apply([&tester](auto&&... unpackedArgs) {
        tester.Test(unpackedArgs...);
    }, validArgsTuple);
}


#define XNN_GEMM(arch_flags, ukernel, k_block, is_pipelined,                                     \
    mr, nr, kr, sr, mr_packed, dummy, datatype, params_type, init_params, pack_fn, packed_stride)\
INSTANTIATE_TEST_SUITE_P(                                                                        \
    ukernel, GemmTest,                                                                           \
    testing::ValuesIn(CreateTests(                                                               \
        k_block, is_pipelined, mr, nr, kr, sr,                                                   \
        mr_packed,                                                                               \
        /*is_igemm=*/false,                                                                      \
        [](GemmMicrokernelTester& tester) {                                                      \
          TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                  \
          CallTestWithValidArgs(tester, (ukernel, init_params, pack_fn, packed_stride));         \
        })),                                                                                     \
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {                                \
      return info.param.test_name;                                                               \
    });
#include "qp8-f32-qb4w-gemm/qp8-f32-qb4w-gemm-minmax.h"
#undef XNN_UKERNEL_WITH_PARAMS
