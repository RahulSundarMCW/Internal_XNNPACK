// // Copyright (c) Facebook, Inc. and its affiliates.
// // All rights reserved.
// //
// // Copyright 2019 Google LLC
// //
// // This source code is licensed under the BSD-style license found in the
// // LICENSE file in the root directory of this source tree.
// //
// // Auto-generated file. Do not edit!
// //   Microkernel: f32-gemminc-minmax
// //   Generator: tools/generate-gemm-test.py

// #include <cstddef>
// #include <functional>
// #include <string>
// #include <vector>

// #include <gtest/gtest.h>
// #include "xnnpack/allocator.h"
// #include "xnnpack/common.h"
// #include "xnnpack/gemm.h"
// #include "xnnpack/igemm.h"
// #include "xnnpack/isa-checks.h"
// #include "xnnpack/microparams-init.h"
// #include "xnnpack/pack.h"
// #include "xnnpack/packw.h"
// #include "xnnpack/ppmm.h"
// #include "xnnpack/requantization.h"
// #include "gemm-microkernel-tester.h"
// #include "next_prime.h"

// namespace {

// std::vector<GemmTestParams> CreateTests(
//     size_t k_block, bool is_pipelined,
//     size_t mr, size_t nr, size_t kr, size_t sr,
//     bool is_igemm,
//     std::function<void(GemmMicrokernelTester& tester)> test_func) {
  
//   size_t adj_k_block = is_pipelined ? k_block * 2 : k_block;
//   std::string kbs = std::to_string(k_block);
//   std::string kb2s = std::to_string(k_block * 2);
//   std::string akbs = std::to_string(adj_k_block);
//   std::string nrs = std::to_string(nr);

//   const GemmMicrokernelTester tester = GemmMicrokernelTester()
//       .mr(mr).nr(nr).kr(kr).sr(sr);

//   std::vector<GemmTestParams> gemm_tests;
//   gemm_tests.reserve(42);

//   gemm_tests.push_back(GemmTestParams(
//       "k_eq_" + kbs,
//       tester.clone()
//           .m(mr).n(nr).k(k_block)
//       , test_func));
//   if (!is_igemm) {
//     gemm_tests.push_back(GemmTestParams(
//         "k_eq_" + kbs + "_strided_a",
//         tester.clone()
//             .m(mr).n(nr).k(k_block)
//             .a_stride(xnnpack::NextPrime(k_block + 1))
//         , test_func));
//   }
//   gemm_tests.push_back(GemmTestParams(
//       "k_eq_" + kbs + "_subtile",
//       tester.clone()
//           .k(k_block).iterations(1)
//       , test_func)
//       .loop_n(1, nr)
//       .loop_m(1, mr));
//   gemm_tests.push_back(GemmTestParams(
//       "k_eq_" + kbs + "_subtile_m",
//       tester.clone()
//           .n(nr).k(k_block).iterations(1)
//       , test_func)
//       .loop_m(1, mr));
//   gemm_tests.push_back(GemmTestParams(
//       "k_eq_" + kbs + "_subtile_n",
//       tester.clone()
//           .m(mr).k(k_block).iterations(1)
//       , test_func)
//       .loop_n(1, nr));
//   if (is_pipelined) {
//     gemm_tests.push_back(GemmTestParams(
//         "k_eq_" + kb2s,
//         tester.clone()
//           .m(mr).n(nr).k(k_block * 2)
//       , test_func));
//     if (!is_igemm) {
//       gemm_tests.push_back(GemmTestParams(
//           "k_eq_" + kb2s + "_strided_a",
//           tester.clone()
//               .m(mr).n(nr).k(k_block * 2)
//               .a_stride(xnnpack::NextPrime(k_block * 2 + 1))
//           , test_func));
//     }
//     gemm_tests.push_back(GemmTestParams(
//         "k_eq_" + kb2s + "_subtile",
//         tester.clone()
//             .k(k_block * 2).iterations(1)
//         , test_func)
//         .loop_n(1, nr)
//         .loop_m(1, mr));
//   }
//   if (k_block > 1) {
//     gemm_tests.push_back(GemmTestParams(
//         "k_lt_" + akbs,
//         tester.clone()
//             .m(mr).n(nr)
//         , test_func)
//         .loop_k(1, adj_k_block - 1));
//     if (!is_igemm) {
//       gemm_tests.push_back(GemmTestParams(
//           "k_lt_" + akbs + "_strided_a",
//           tester.clone()
//               .m(mr).n(nr)
//               .a_stride(xnnpack::NextPrime(adj_k_block + 1))
//           , test_func)
//           .loop_k(1, adj_k_block - 1));
//     }
//     gemm_tests.push_back(GemmTestParams(
//         "k_lt_" + akbs + "_subtile",
//         tester.clone()
//             .iterations(1)
//         , test_func)
//         .loop_k(1, adj_k_block - 1)
//         .loop_n(1, nr)
//         .loop_m(1, mr));
//   }
//   gemm_tests.push_back(GemmTestParams(
//       "k_gt_" + akbs,
//       tester.clone()
//           .m(mr).n(nr)
//       , test_func)
//       .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
//   if (is_igemm) {
//     gemm_tests.push_back(GemmTestParams(
//         "k_gt_" + akbs + "_strided_a",
//         tester.clone()
//             .m(mr).n(nr)
//             .a_stride(xnnpack::NextPrime(adj_k_block * 2 + 1))
//       , test_func)
//       .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
//   }
//   gemm_tests.push_back(GemmTestParams(
//       "k_gt_" + akbs + "_subtile",
//       tester.clone()
//           .iterations(1)
//       , test_func)
//       .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
//       .loop_n(1, nr)
//       .loop_m(1, mr));
//   if (k_block > 1) {
//     gemm_tests.push_back(GemmTestParams(
//         "k_div_" + kbs,
//         tester.clone()
//             .m(mr).n(nr)
//         , test_func)
//         .loop_k(adj_k_block + k_block, k_block * 5, k_block));
//     if (is_igemm) {
//       gemm_tests.push_back(GemmTestParams(
//           "k_div_" + kbs + "_strided_a",
//           tester.clone()
//               .m(mr).n(nr)
//               .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
//           , test_func)
//           .loop_k(adj_k_block + k_block, k_block * 3, k_block));
//     }
//     gemm_tests.push_back(GemmTestParams(
//         "k_div_" + kbs + "_subtile",
//         tester.clone()
//             .iterations(1)
//         , test_func)
//         .loop_k(adj_k_block + k_block, k_block * 5, k_block)
//         .loop_n(1, nr)
//         .loop_m(1, mr));
//   }
//   gemm_tests.push_back(GemmTestParams(
//       "n_gt_" + nrs,
//       tester.clone()
//           .m(mr)
//       , test_func)
//       .loop_n(nr + 1, nr * 2 - 1)
//       .loop_k(1, k_block * 3, k_block + 1));
//   if (!is_igemm) {
//     gemm_tests.push_back(GemmTestParams(
//         "n_gt_" + nrs + "_strided_a",
//         tester.clone()
//             .m(mr)
//             .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
//         , test_func)
//         .loop_n(nr + 1, nr * 2 - 1)
//         .loop_k(1, k_block * 3, k_block));
//   }
//   gemm_tests.push_back(GemmTestParams(
//       "n_gt_" + nrs + "_subtile",
//       tester.clone()
//           .iterations(1)
//       , test_func)
//       .loop_n(nr + 1, nr * 2 - 1)
//       .loop_k(1, k_block * 3, k_block + 1)
//       .loop_m(1, mr));
//   gemm_tests.push_back(GemmTestParams(
//       "n_div_" + nrs,
//       tester.clone()
//           .m(mr)
//       , test_func)
//       .loop_n(nr * 2, nr * 3, nr)
//       .loop_k(1, k_block * 3, k_block + 1));
//   if (!is_igemm) {
//     gemm_tests.push_back(GemmTestParams(
//         "n_div_" + nrs + "_strided_a",
//         tester.clone()
//             .m(mr)
//             .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
//         , test_func)
//         .loop_n(nr * 2, nr * 3, nr)
//         .loop_k(1, k_block * 3, k_block));
//   }
//   gemm_tests.push_back(GemmTestParams(
//       "n_div_" + nrs + "_subtile",
//       tester.clone()
//           .iterations(1)
//       , test_func)
//       .loop_n(nr * 2, nr * 3, nr)
//       .loop_k(1, k_block * 3, k_block + 1)
//       .loop_m(1, mr));
//   if (is_igemm) {
//     gemm_tests.push_back(GemmTestParams(
//         "small_kernel",
//         tester.clone()
//             .m(mr).n(nr).ks(3)
//         , test_func)
//         .loop_k(1, k_block * 3, k_block + 1));
//     gemm_tests.push_back(GemmTestParams(
//         "small_kernel_subtile",
//         tester.clone()
//             .ks(3).iterations(1)
//         , test_func)
//         .loop_k(1, k_block * 3, k_block + 1)
//         .loop_n(1, nr)
//         .loop_m(1, mr));
//     gemm_tests.push_back(GemmTestParams(
//         "n_gt_" + nrs + "_small_kernel",
//         tester.clone()
//             .m(mr).ks(3)
//         , test_func)
//         .loop_n(nr + 1, nr * 2 - 1)
//         .loop_k(1, k_block * 3, k_block + 1));
//     gemm_tests.push_back(GemmTestParams(
//         "n_div_" + nrs + "_small_kernel",
//         tester.clone()
//             .m(mr).ks(3)
//         , test_func)
//         .loop_n(nr * 2, nr * 3, nr)
//         .loop_k(1, k_block * 3, k_block + 1));
//   }
//   gemm_tests.push_back(GemmTestParams(
//       "strided_cm_subtile",
//       tester.clone()
//           .mr(mr).nr(nr).kr(kr).sr(sr)
//           .cm_stride(xnnpack::NextPrime(nr + 1))
//           .iterations(1)
//       , test_func)
//       .loop_k(1, k_block * 3, k_block + 1)
//       .loop_n(1, nr)
//       .loop_m(1, mr));
//   if (is_igemm) {
//     gemm_tests.push_back(GemmTestParams(
//         "a_offset",
//         tester.clone()
//             .m(mr).n(nr).ks(3)
//             .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
//         , test_func)
//         .loop_k(1, k_block * 3, k_block + 1));
//     gemm_tests.push_back(GemmTestParams(
//         "zero",
//         tester.clone()
//             .m(mr).n(nr).ks(3)
//             .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
//         , test_func)
//         .loop_k(1, k_block * 3, k_block + 1)
//         .loop_zi(0, mr - 1));
//   }
//   gemm_tests.push_back(GemmTestParams(
//       "qmin",
//       tester.clone()
//           .m(mr).n(nr).k(k_block).qmin(128)
//       , test_func));
//   gemm_tests.push_back(GemmTestParams(
//       "qmax",
//       tester.clone()
//           .m(mr).n(nr).k(k_block).qmax(128)
//       , test_func));
//   gemm_tests.push_back(GemmTestParams(
//       "strided_cm",
//       tester.clone()
//           .m(mr).n(nr).k(k_block)
//           .cm_stride(xnnpack::NextPrime(nr + 1))
//       , test_func));

//   return gemm_tests;
// }

// }  // namespace

// #define XNN_GEMM(arch_flags, ukernel, k_block, is_pipelined,                         \
//     mr, nr, kr, sr, mr_packed, is_igemm, datatype, params_type, init_params, pack_fn)\
// INSTANTIATE_TEST_SUITE_P(                                                            \
//     ukernel, GemmTest,                                                               \
//     testing::ValuesIn(CreateTests(                                                   \
//         k_block, is_pipelined, mr, nr, kr, sr,                                       \
//         /*is_igemm=*/false,                                                          \
//         [](GemmMicrokernelTester& tester) {                                          \
//           TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                      \
//           tester.Test(ukernel, init_params, pack_fn);                                \
//         })),                                                                         \
//     [](const testing::TestParamInfo<GemmTest::ParamType>& info) {                    \
//       return info.param.test_name;                                                   \
//     });
// #include "f32-gemminc/f32-gemminc-minmax.h"
// #undef XNN_UKERNEL_WITH_PARAMS
