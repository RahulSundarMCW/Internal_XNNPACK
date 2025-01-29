#!/usr/bin/env python
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import math
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xngen
import xnncommon


parser = argparse.ArgumentParser(
  description='Reduce microkernel test generator')
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=["ReduceMicrokernelTester", "RSumMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-k", "--ukernel", metavar="FILE", required=True,
                    help="Microkernel type")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())

OP_TYPES = {
    "rmax": "Max",
    "rmin": "Min",
    "rminmax": "MinMax",
}

REDUCE_TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, datatype, output_type, params_type, init_params)
XNN_TEST_REDUCE_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype, output_type, ${", ".join(TEST_ARGS)});
XNN_TEST_REDUCE_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype, output_type, ${", ".join(TEST_ARGS)});
XNN_TEST_REDUCE_BATCH_LT(ukernel, arch_flags, batch_tile, datatype, output_type, ${", ".join(TEST_ARGS)});
XNN_TEST_REDUCE_BATCH_GT(ukernel, arch_flags, batch_tile, datatype, output_type, ${", ".join(TEST_ARGS)});
$if TESTER in ["RSumMicrokernelTester"]:
  XNN_TEST_REDUCE_SCALE(ukernel, arch_flags, batch_tile, datatype, output_type, ${", ".join(TEST_ARGS)});
  XNN_TEST_REDUCE_OVERFLOW_ACCUMULATOR(ukernel, arch_flags, batch_tile, datatype, output_type, ${", ".join(TEST_ARGS)});
"""


def main(args):
  options = parser.parse_args(args)
  
  tester = options.tester
  tester_header = {
      "ReduceMicrokernelTester": "reduce-microkernel-tester.h",
      "RSumMicrokernelTester": "rsum-microkernel-tester.h",
  }[tester]
  ukernel = options.ukernel

  tests = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {microkernel}
//   Generator: {generator}


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "{tester_header}"
""".format(microkernel=options.ukernel, generator=sys.argv[0],
           tester_header=tester_header)

  ukernel_parts = options.ukernel.split("-")
  datatype = ukernel_parts[0]
  op = ukernel_parts[1]

  test_args = ["ukernel"]
  if tester in ["ReduceMicrokernelTester"]:
     op_type = OP_TYPES[op]
     test_args.append("%s::OpType::%s" % (tester, op_type))
  if datatype != "u8":
    test_args.append("init_params")

  tests += xnncommon.make_multiline_macro(xngen.preprocess(
    REDUCE_TEST_TEMPLATE,
    {
        "TEST_ARGS": test_args,
        "TESTER": tester,
        "DATATYPE": datatype,
    },
  ))
  
  parts = ukernel.split("-")
  folder_parts = []
  for part in parts:
    folder_parts.append(part)
    if part in ["rmax", "rmin", "rsum", "rminmax"]:
        break
  folder = "-".join(folder_parts)

  tests += f'#include "{xnncommon.xnnpack_src()}{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"
  tests = tests.replace("f16-rmin/f16-rmin.h", "f16-rminmax/f16-rmin.h")
  tests = tests.replace("f16-rmax/f16-rmax.h", "f16-rminmax/f16-rmax.h")
  tests = tests.replace("f32-rmin/f32-rmin.h", "f32-rminmax/f32-rmin.h")
  tests = tests.replace("f32-rmax/f32-rmax.h", "f32-rminmax/f32-rmax.h")

  xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
