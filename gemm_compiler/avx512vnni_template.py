#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from gemm_compiler import avx512f_template


class Avx512Vnni(avx512f_template.Avx512F):
  """All SIMD features for avx512vnni."""

  def isa(self):
    return 'avx512vnni'

  def a_registers(self, idx):
    return 'zmm2'

  def scale_registers(self):
    return ['zmm10', 'zmm11', 'zmm2', 'zmm3']

  def w_registers(self):
    return ['zmm6', 'zmm7', 'zmm8', 'zmm9']

  def acc_registers(self):
    return [
        'mm5',
        'mm12',
        'mm14',
        'mm15',
        'mm16',
        'mm17',
        'mm18',
        'mm19',
        'mm20',
        'mm21',
        'mm22',
        'mm23',
        'mm24',
        'mm25',
        'mm26',
        'mm27',
        'mm28',
        'mm29',
        'mm30',
        'mm4',
        'mm8',
        'mm9',
    ]

  def function_name(self):
    return (
        f'xnn_qd8_f32_qc8w_gemm_minmax_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c4__asm_amd64_{self.isa()}'
    )

  def zp_scale(self, pos):
    regs = ['10', '11']
    return regs[pos]

  # kc = round_up_po2(kc, channels)
  def adjust_kc(self):
    channels = 4
    ret = """
      add {kc_reg}, {channels}
      and {kc_reg}, {neg_channels}\n""".format(
        kc_reg=self.kc_register(), channels=channels - 1, neg_channels=-channels
    )
    return ret

  def quantization_params(self):
    return """
    mov {quantization_params_reg}, [rsp + 88]
    """.format(quantization_params_reg=self.quantization_params_register())

  def quantization_params_register(self):
    return self.k_register()

  def input_asm(self):
    in_asm = {
        'loop': [
            'vpbroadcastd {AM}, [{AM_ptr} + {a_offset}]\n',
        ],
        'compute': ['vpdpbusd  z{ACC}, {A}, {W}\n'],
    }
    return in_asm

  def weights_asm(self):
    w_asm = {
        'loop': [
            'vmovaps  {W}, [{W_ptr} + {offset}]\n',
        ],
        'after': 'add {W}, {w_step}\n',
    }
    return w_asm

  def compute_asm(self):
    c_asm = {
        'loop': ['vpdpbusd  z{ACC}, {A}, {W}\n'],
    }
    return c_asm

  # Quantization parameters are pushed to the stack at this offset.
  def quantization_params_offset(self):
    return 8

  def dequantize(self):
    accumulators = self.acc_registers()
    ret = ''
    ret += '\n# Convert from int32 to float.\n'
    for nr in range(0, self.n * self.m):
      ret += 'vcvtdq2ps z{ACC}, z{ACC}\n'.format(ACC=accumulators[nr])
    ret += '# Load quantization_params pointer from stack\n'
    ret += 'mov {quantization_params_reg}, [rsp + {offset}]\n'.format(
        quantization_params_reg=self.quantization_params_register(),
        offset=self.stack_size() + self.quantization_params_offset(),
    )
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        ret += (
            'vmulps z{ACC}, z{ACC}, DWORD PTR [{quantization_params_reg} +'
            ' {offset}]{{1to16}}\n'.format(
                ACC=accumulators[nr * self.m + mr],
                offset=4 + mr * 8,
                quantization_params_reg=self.quantization_params_register(),
            )
        )
    output_scale = 'vmovaps {W_SCALE}, [{W} + {offset}]\n'
    # output scales
    for nr in range(0, self.n):
      ret += output_scale.format(
          W=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
          W_SCALE=self.scale_registers()[nr],
      )
    ret += self.increment_ptr(
        ptr=self.w_ptr_register(), step=self.register_bytes() * self.n
    )
    # biases
    for nr in range(0, self.n):
      ret += output_scale.format(
          W=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
          W_SCALE=self.w_registers()[nr],
      )
    ret += self.increment_ptr(
        ptr=self.w_ptr_register(), step=self.register_bytes() * self.n
    )
    # Intel gets points here for its fma instructions which can accumulate into
    # any of the registers. For once, Intel has saner instructions than Arm.
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        ret += 'vfmadd213ps z{ACC}, {SCALE}, {BIAS}\n'.format(
            ACC=accumulators[nr * self.m + mr],
            SCALE=self.scale_registers()[nr],
            BIAS=self.w_registers()[nr],
        )

    return ret

  def outer_loop_prepare(self):
    # outside the outer loop
    zp_scale_load_push = (
        """mov {tmp_reg}, [{quantization_params_reg} + {zp_offset}]
      vpbroadcastd {tmp_s_reg}, {tmp_reg}
      vmovaps zmmword ptr [rsp + {offset}], {tmp_s_reg}\n"""
    )
    ret = '\n# Load quantization_params pointer from stack\n'
    ret += 'mov {quantization_params_reg}, [rsp + {offset}]\n'.format(
        quantization_params_reg=self.quantization_params_register(),
        offset=self.stack_size() + self.quantization_params_offset(),
    )
    for mr in range(0, self.m, 1):
      ret += zp_scale_load_push.format(
          tmp_reg=self.register_map_dword(self.tmp_gp_registers()[0]),
          quantization_params_reg=self.quantization_params_register(),
          tmp_s_reg=self.w_registers()[0],
          offset=self.stupid_offset() + mr * 64,
          zp_offset=mr * 8,
      )
    return ret

  def init_accumulators(self):
    ret = '# Initialize accumulators with k_sum * input zero point.\n'
    accumulators = self.acc_registers()

    ksum_x16 = 'vmovaps  {KSUM}, [{W} + {offset}]\n'
    vksum = 'vpmulld z{ACC}, {KSUM}, ZMMWORD PTR [rsp + {offset}]\n'

    for nr in range(0, self.n):
      ret += ksum_x16.format(
          W=self.w_ptr_register(),
          KSUM=self.w_registers()[nr],
          offset=self.register_bytes() * nr,
      )
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        ret += vksum.format(
            ACC=accumulators[nr * self.m + mr],
            KSUM=self.w_registers()[nr],
            pos=int((mr % 2) * 2),
            offset=self.stupid_offset() + mr * 64,
        )

    ret += self.increment_ptr(
        ptr=self.w_ptr_register(), step=self.register_bytes() * self.n
    )
    return ret

  def stupid_offset(self):
    size = self.m * 16 + self.c_ptr_stack_offset()
    return math.ceil(size / 64) * 64

  def stack_size(self):
    size = self.stupid_offset() + self.m * 64
    # round up to multiple of 64.
    return math.ceil(size / 64) * 64
