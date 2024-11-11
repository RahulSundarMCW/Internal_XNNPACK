#!/usr/bin/env python
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(
    description="Test generator for CONV HWC2CHW micro-kernels"
)
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=["ConvHWC2CHWMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-k", "--ukernel", metavar="FILE", required=True,
                    help="Microkernel type")
parser.add_argument(
    "-o",
    "--output",
    metavar="FILE",
    required=True,
    help="Output (C++ source) file",
)
parser.set_defaults(defines=list())


TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, kernel_height, subsampling, padding_right, padding_left, input_channels, channel_tile, width_tile, datatype, params_type, init_params) \
XNN_TEST_CONV_HWC2CHW_INPUT_WIDTH_EQ(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_INPUT_WIDTH_DIV(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_INPUT_WIDTH_LT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_INPUT_WIDTH_GT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_OUTPUT_CHANNELS_LT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_OUTPUT_CHANNELS_GT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_OUTPUT_CHANNELS_DIV(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_INPUT_HEIGHT_LT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_INPUT_HEIGHT_GT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_PADDING_TOP(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_PADDING_BOTTOM(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_OUTPUT_Y_START(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_OUTPUT_Y_END(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_QMIN(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_CONV_HWC2CHW_QMAX(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
"""


def split_ukernel_name(name):
  match = re.fullmatch(
      r"xnn_(f16|f32)_conv_hwc2chw_ukernel_(\d+)x(\d+)s(\d+)(p1)c(\d+)x(\d+)__(.+)_(\d+)x(\d+)?",
      name,
  )
  assert match is not None
  kernel_height, kernel_width = int(match.group(2)), int(match.group(3))
  assert kernel_height == kernel_width
  subsampling = int(match.group(4))
  padding_right = 1
  padding_left = 1
  input_channels = int(match.group(6))
  channel_tile = int(match.group(7))
  height_tile = int(match.group(9))
  width_tile = int(match.group(10))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(8))
  return (
      kernel_height,
      kernel_width,
      subsampling,
      padding_left,
      padding_right,
      input_channels,
      channel_tile,
      height_tile,
      width_tile,
      arch,
      isa,
  )


def main(args):
  options = parser.parse_args(args)
  tester = options.tester
  tester_header = {
  "ConvHWC2CHWMicrokernelTester": "conv-hwc2chw-microkernel-tester.h",
  }[tester]
  ukernel = options.ukernel

  tests = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/conv.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "conv-hwc2chw-microkernel-tester.h"
""".format(specification=options.ukernel, generator=sys.argv[0])
  ukernel_parts = options.ukernel.split("-")
  datatype = ukernel_parts[0]
  op = ukernel_parts[1]
  test_args = ["kernel_height"]
  test_args.append("subsampling")
  test_args.append("padding_right")
  test_args.append("padding_left")
  test_args.append("input_channels")
  test_args.append("channel_tile")
  test_args.append("width_tile")
  test_args.append("init_params")
  tests += xnncommon.make_multiline_macro(xngen.preprocess(
    TEST_TEMPLATE,
    {
        "TEST_ARGS": test_args,
        "TESTER": tester,
        "DATATYPE": datatype,
    },
  ))
  folder = datatype + "-" + ("conv-hwc2chw" if datatype.startswith("f") else op)
  tests += f'#include "{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"

  xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
