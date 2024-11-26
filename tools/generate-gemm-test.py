#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import collections
import os
import re
import sys
import zlib
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon

parser = argparse.ArgumentParser(description="XNNPACK generator")
parser.add_argument("-k", "--ukernel", required=True,
                    help="microkernel")
parser.add_argument(
    "-o",
    "--output-test",
    action="append",
    metavar="FILE",
    required=True,
    help="Test output (C++ source) file(s)")
parser.set_defaults(defines=list())


GEMM_CREATE_TESTS_CODE = """\
std::vector<GemmTestParams> CreateTests(
    size_t k_block, bool is_pipelined,
    size_t mr, size_t nr, size_t kr, size_t sr,
    $if DATATYPE in ('qp8'):
      size_t mr_packed,
    bool is_igemm,
    bool unsigned_inputs,
    std::function<void(GemmMicrokernelTester& tester)> test_func) {
  
  size_t adj_k_block = is_pipelined ? k_block * 2 : k_block;
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  size_t nr_scale = get_batch_scale<datatype>();
  if (nr_scale != 0) {
    nr = nr * nr_scale;
  }
  std::string nrs = std::to_string(nr);

  $if DATATYPE in ('qp8',):
    const GemmMicrokernelTester tester = GemmMicrokernelTester()
        .mr(mr).nr(nr).kr(kr).sr(sr).mr_packed(mr_packed).unsigned_inputs(unsigned_inputs);
  $else:
    const GemmMicrokernelTester tester = GemmMicrokernelTester()
        .mr(mr).nr(nr).kr(kr).sr(sr).unsigned_inputs(unsigned_inputs);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      tester.clone()
          .m(mr).n(nr).k(k_block)
          $if KERNELTYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if KERNELTYPE in ['qb4w']:
            .bl(32)
      , test_func));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .a_stride(xnnpack::NextPrime(k_block + 1))
            $if KERNELTYPE in ['qb4w', 'qc4w']:
              .b_zero_point(8)
            $if KERNELTYPE in ['qb4w']:
              .bl(32)
        , test_func));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block).iterations(1)
          $if KERNELTYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if KERNELTYPE in ['qb4w']:
            .bl(32)
      , test_func)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block).iterations(1)
          $if KERNELTYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if KERNELTYPE in ['qb4w']:
            .bl(32)
      , test_func)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block).iterations(1)
          $if KERNELTYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if KERNELTYPE in ['qb4w']:
            .bl(32)
      , test_func)
      .loop_n(1, nr));
  if (is_pipelined) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kb2s,
        tester.clone()
          .m(mr).n(nr).k(k_block * 2)
          $if KERNELTYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if KERNELTYPE in ['qb4w']:
            .bl(32)
      , test_func));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_eq_" + kb2s + "_strided_a",
          tester.clone()
              .m(mr).n(nr).k(k_block * 2)
              .a_stride(xnnpack::NextPrime(k_block * 2 + 1))
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
            $if KERNELTYPE in ['qb4w']:
              .bl(32)
          , test_func));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kb2s + "_subtile",
        tester.clone()
            .k(k_block * 2).iterations(1)
            $if KERNELTYPE in ['qb4w', 'qc4w']:
              .b_zero_point(8)
            $if KERNELTYPE in ['qb4w']:
              .bl(32)
        , test_func)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  $if KERNELTYPE not in ['qb4w']:
      if (k_block > 1) {
        gemm_tests.push_back(GemmTestParams(
            "k_lt_" + akbs,
            tester.clone()
                .m(mr).n(nr)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_k(1, adj_k_block - 1));
        if (!is_igemm) {
          gemm_tests.push_back(GemmTestParams(
              "k_lt_" + akbs + "_strided_a",
              tester.clone()
                  .m(mr).n(nr)
                  .a_stride(xnnpack::NextPrime(adj_k_block + 1))
                  $if KERNELTYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                  $if KERNELTYPE in ['qb4w']:
                    .bl(32)
              , test_func)
              .loop_k(1, adj_k_block - 1));
        }
        gemm_tests.push_back(GemmTestParams(
            "k_lt_" + akbs + "_subtile",
            tester.clone()
                .iterations(1)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_k(1, adj_k_block - 1)
            .loop_n(1, nr)
            .loop_m(1, mr));
      }
      gemm_tests.push_back(GemmTestParams(
          "k_gt_" + akbs,
          tester.clone()
              .m(mr).n(nr)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func)
          .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
      if (is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "k_gt_" + akbs + "_strided_a",
            tester.clone()
                .m(mr).n(nr)
                .a_stride(xnnpack::NextPrime(adj_k_block * 2 + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func)
          .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
      }
      gemm_tests.push_back(GemmTestParams(
          "k_gt_" + akbs + "_subtile",
          tester.clone()
              .iterations(1)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func)
          .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
          .loop_n(1, nr)
          .loop_m(1, mr));
      if (k_block > 1) {
        gemm_tests.push_back(GemmTestParams(
            "k_div_" + kbs,
            tester.clone()
                .m(mr).n(nr)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_k(adj_k_block + k_block, k_block * 5, k_block));
        if (is_igemm) {
          gemm_tests.push_back(GemmTestParams(
              "k_div_" + kbs + "_strided_a",
              tester.clone()
                  .m(mr).n(nr)
                  .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                  $if KERNELTYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                  $if KERNELTYPE in ['qb4w']:
                    .bl(32)
              , test_func)
              .loop_k(adj_k_block + k_block, k_block * 3, k_block));
        }
        gemm_tests.push_back(GemmTestParams(
            "k_div_" + kbs + "_subtile",
            tester.clone()
                .iterations(1)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_k(adj_k_block + k_block, k_block * 5, k_block)
            .loop_n(1, nr)
            .loop_m(1, mr));
      }
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs,
          tester.clone()
              .m(mr)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func)
          .loop_n({nr + 1}, {nr * 2 - 1}{", 4" if nr_scale != 0 else ""})
          .loop_k(1, k_block * 3, k_block + 1));
      if (!is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "n_gt_" + nrs + "_strided_a",
            tester.clone()
                .m(mr)
                .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_n({nr + 1}, {nr * 2 - 1}{", 4" if nr_scale != 0 else ""})
            .loop_k(1, k_block * 3, k_block));
      }
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs + "_subtile",
          tester.clone()
              .iterations(1)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func)
          .loop_n({nr + 1}, {nr * 2 - 1}{", 4" if nr_scale != 0 else ""})
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_m(1, mr));
      gemm_tests.push_back(GemmTestParams(
          "n_div_" + nrs,
          tester.clone()
              .m(mr)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func)
          .loop_n(nr * 2, nr * 3, nr)
          .loop_k(1, k_block * 3, k_block + 1));
      if (!is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "n_div_" + nrs + "_strided_a",
            tester.clone()
                .m(mr)
                .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_n(nr * 2, nr * 3, nr)
            .loop_k(1, k_block * 3, k_block));
      }
      gemm_tests.push_back(GemmTestParams(
          "n_div_" + nrs + "_subtile",
          tester.clone()
              .iterations(1)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func)
          .loop_n(nr * 2, nr * 3, nr)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_m(1, mr));
      if (is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "small_kernel",
            tester.clone()
                .m(mr).n(nr).ks(3)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "small_kernel_subtile",
            tester.clone()
                .ks(3).iterations(1)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_k(1, k_block * 3, k_block + 1)
            .loop_n(1, nr)
            .loop_m(1, mr));
        gemm_tests.push_back(GemmTestParams(
            "n_gt_" + nrs + "_small_kernel",
            tester.clone()
                .m(mr).ks(3)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_n({nr + 1}, {nr * 2 - 1}{", 4" if nr_scale != 0 else ""})
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "n_div_" + nrs + "_small_kernel",
            tester.clone()
                .m(mr).ks(3)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_n(nr * 2, nr * 3, nr)
            .loop_k(1, k_block * 3, k_block + 1));
      }
      gemm_tests.push_back(GemmTestParams(
          "strided_cm_subtile",
          tester.clone()
              .mr(mr).nr(nr).kr(kr).sr(sr)
              .cm_stride(xnnpack::NextPrime(nr + 1))
              .iterations(1)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_n(1, nr)
          .loop_m(1, mr));
      if (is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "a_offset",
            tester.clone()
                .m(mr).n(nr).ks(3)
                .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "zero",
            tester.clone()
                .m(mr).n(nr).ks(3)
                .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func)
            .loop_k(1, k_block * 3, k_block + 1)
            .loop_zi(0, mr - 1));
      }
      $if ACTIVATION == "MINMAX":
        gemm_tests.push_back(GemmTestParams(
            "qmin",
            tester.clone()
                .m(mr).n(nr).k(k_block).qmin(128)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func));
        gemm_tests.push_back(GemmTestParams(
            "qmax",
            tester.clone()
                .m(mr).n(nr).k(k_block).qmax(128)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func));
      gemm_tests.push_back(GemmTestParams(
          "strided_cm",
          tester.clone()
              .m(mr).n(nr).k(k_block)
              .cm_stride(xnnpack::NextPrime(nr + 1))
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func));
      $if DATATYPE == "qu8":
        gemm_tests.push_back(GemmTestParams(
            "no_a_zero_point",
            tester.clone()
                .m(mr).n(nr).a_zero_point(0)
            , test_func)
            .loop_k(1, k_block * 3, k_block + 1));
      $if DATATYPE == "qu8":
        gemm_tests.push_back(GemmTestParams(
            "no_b_zero_point",
            tester.clone()
                .m(mr).n(nr).b_zero_point(0)
            , test_func)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "b_zero_point",
            tester.clone()
                .m(mr).n(nr).k(k_block)
            , test_func)
            .loop_bzp(0, 255));
        gemm_tests.push_back(GemmTestParams(
            "no_zero_point",
            tester.clone()
                .m(mr).n(nr)
                .a_zero_point(0)
                .b_zero_point(0)
            , test_func)
            .loop_k(1, k_block * 3, k_block + 1));
  $if KERNELTYPE in ['qb4w']:
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
"""

# GEMM_TEST_CODE = """\
TEST_TEMPLATE = """\
#define XNN_GEMM(arch_flags, ukernel, k_block, is_pipelined,
    mr, nr, kr, sr, mr_packed, unsigned_inputs, datatype, params_type, init_params, pack_fn, packed_stride)
INSTANTIATE_TEST_SUITE_P(
    ukernel, GemmTest,
    testing::ValuesIn(CreateTests(
        k_block, is_pipelined, mr, nr, kr, sr,
        $if DATATYPE in ('qp8',):
          mr_packed,
        /*is_igemm=*/${"true" if UKERNEL_TYPE.startswith("IGEMM") else "false"},
        unsigned_inputs,
        [](GemmMicrokernelTester& tester) {
          TEST_REQUIRES_ARCH_FLAGS(arch_flags);
          tester.Test(${", ".join(TEST_ARGS)});
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });

$if TEST_NAME.startswith('GENERATE') and DATATYPE in ['f32', 'f16']:
  TEST(ukernel, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= ${MR}; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= ${KBLOCK * 2}; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            $if NR > 1:
              .nr(${NR})
            $if KR > 1:
              .kr(${KR})
            $if SR > 1:
              .sr(${SR})
            .m(m)
            $if NR > 1:
              .n(${NR})
            .k(k)
            .iterations(1)
            $if KERNELTYPE in ['qb4w', 'qc4w']:
              .b_zero_point(8)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }
  }

$if TEST_NAME.startswith('GENERATE') and DATATYPE in ['f32', 'f16'] and PROTOTYPE is not None:
  #if XNN_ENABLE_ASSEMBLY
    TEST(ukernel;, matches_assembly) {
      GemmMicrokernelTester()
        $if MR > 1:
          .mr(${MR})
        $if NR > 1:
          .nr(${NR})
        $if KR > 1:
          .kr(${KR})
        $if SR > 1:
          .sr(${SR})
        $if MR > 1:
          .m(${MR})
        $if NR > 1:
          .n(${NR})
        .k(${KBLOCK})
        .Test(
            ${", ".join(TEST_ARGS)},
            &${PROTOTYPE});
    }
  #endif // XNN_ENABLE_ASSEMBLY

"""


def main(args):
  options = parser.parse_args(args)
  num_output_files = len(options.output_test)
  ukernel = options.ukernel

  test_header = """\
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {ukernel}
//   Generator: {generator}

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
""".format(ukernel=ukernel, generator=sys.argv[0])
  
  test_cases = ""
  
  test_outputs = collections.defaultdict(str)

  parts = ukernel.split("-")
  datatype = parts[0]
  if len(ukernel.split("-")) > 3:
    datatype, ukernel_type, activation, _ = ukernel.split("-", 3)
  elif len(ukernel.split("-")) > 2:
    datatype, ukernel_type, activation = ukernel.split("-", 2)
  else:
    datatype, ukernel_type = ukernel.split("-", 1)

  kerneltype = datatype

  if datatype in ["f16", "f32"] and ukernel_type in ["qc8w", "qc4w"]:
    if len(ukernel.split("-")) > 3 :
      datatype, kerneltype, ukernel_type, activation = ukernel.split("-", 4)
    else:
      datatype, kerneltype, ukernel_type = ukernel.split("-", 3)
    datatype = f"{datatype}-{kerneltype}" 
 
  if (
    datatype in ("qd8", "qp8")
    and ukernel_type in ["f16", "f32"]
    and activation in ["qc8w", "qc4w", "qb4w", "f32acc"]
  ):
    datatype, _, kerneltype, ukernel_type, activation = ukernel.split("-", 5)

  if (ukernel_type in ["qc8w", "qc4w", "qb4w", "f32acc"]):
    # print(f"ukernel_type = {ukernel_type}")
    if len(parts) > 5 :
        datatype, _, kerneltype, ukernel_type, activation, _ = parts[:6]
    elif len(parts) > 4 :
        datatype, kerneltype, ukernel_type, activation, _ = parts[:5]
    elif len(parts) > 3 :
        datatype, kerneltype, ukernel_type, activation = parts[:4]

  folder_parts = []
  for part in parts:
    folder_parts.append(part)
    if part in ["gemm", "igemm", "ppmm", "gemminc"]:
        break
  folder = "-".join(folder_parts)

  if "minmax" in parts:
    activation = "minmax"
  elif "relu" in parts:
    activation = "relu"
  else:
    activation = "linear"

  if "fp32" in parts:
    requantization = "fp32"
  elif "rndnu" in parts:
    requantization = "rndnu"
  else:
    requantization = None
  
  create_tests_args = {
    "DATATYPE": datatype,
    "ACTIVATION": activation.upper(),
    "KERNELTYPE": kerneltype,
  }
  create_tests = xngen.preprocess(GEMM_CREATE_TESTS_CODE, create_tests_args)
  create_tests = (
    "namespace {\n\n"
    + "\n".join([create_tests])
    + "\n}  // namespace\n"
  )
  tests = test_header + "\n" + create_tests + "\n" + test_cases
  
  test_args = ["ukernel", "init_params", "pack_fn", "packed_stride"]
  if requantization:
    requantization_datatype = {"qc8": "qs8"}.get(datatype, datatype)
    test_args.append(
      "xnn_%s_requantize_%s" % (requantization_datatype, requantization)
    )

  tests += xnncommon.make_multiline_macro(xngen.preprocess(
      TEST_TEMPLATE,
      {
        "TEST_ARGS": test_args,
        "DATATYPE": datatype,
        "UKERNEL_TYPE": ukernel_type.upper(),
        "TEST_NAME": ukernel.upper().replace("UKERNEL_", ""),
      },
  ))
  
  tests += f'#include "{xnncommon.xnnpack_src()}{folder}/{ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"

  output_index = zlib.crc32(bytes(ukernel, "utf-8")) % num_output_files
  test_outputs[options.output_test[output_index]] += "" 

  for output_name, content in test_outputs.items():
    print(f"Debug: options.output_test = {options.output_test}")
    # print(f"Debug: Final ukernel_type = {ukernel_type}")
    xnncommon.overwrite_if_changed(output_name, tests + content)

if __name__ == "__main__":
  main(sys.argv[1:])
