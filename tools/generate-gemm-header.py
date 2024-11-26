import argparse
import codecs
import codecs
from collections import defaultdict
import io
import math
import os
import os
import re
import sys
import platform
from playsound import playsound

import xnncommon
from xnncommon import _ARCH_TO_MACRO_MAP, _ISA_TO_MACRO_MAP
import yaml

try:
 _DATATYPE_TO_CTYPE_MAP = {
     "s8": "int8_t",
     "u8": "uint8_t",
     "qs8": "int8_t",
     "qu8": "uint8_t",
     "s16": "int16_t",
     "u16": "uint16_t",
     "s32": "int32_t",
     "u32": "uint32_t",
     "s64": "int64_t",
     "u64": "uint64_t",
     "bf16": "xnn_bfloat16",
     "f16": "xnn_float16",
     "f32": "float",
 }
 
 params_map = {
     "minmax": "minmax",
     "rndd": "rnd",
     "rndne": "rnd",
     "rndu": "rnd",
     "rndz": "rnd",
     "elu": "elu",
     "lrelu": "lrelu",
     "relu": "relu",
 }
 
 yamls = {
   "bf16-gemm-minmax": "bf16-gemm",  #DONE
   "f16-f32acc-gemm-minmax": "f16-f32acc-gemm",  #DONE
   "f16-f32acc-igemm-minmax": "f16-f32acc-igemm",
   "f16-gemm-minmax": "f16-gemm",  #Done
   "f16-igemm-minmax": "f16-igemm",  #Done
   "f32-gemm": "f32-gemm",
   "f32-gemminc-minmax": "f32-gemminc",
   "f32-gemm-goi-minmax": "f32-gemm",  #Done
   "f32-gemm-minmax": "f32-gemm",  #Done
   "f32-gemm-relu": "f32-gemm",  #Done
   "f32-qc8w-gemm": "f32-qc8w-gemm",
   "f32-igemm": "f32-igemm",
   "f32-igemm-minmax": "f32-igemm",  #Done
   "f32-igemm-relu": "f32-igemm",  #Done
   "f32-qc4w-gemm-minmax": "f32-qc4w-gemm",  #Done
   "f32-qc8w-gemm-minmax": "f32-qc8w-gemm",  #Done
   "f32-qc8w-gemm-relu": "f32-qc8w-gemm",  #Done
   "f32-ppmm-minmax": "f32-ppmm",  #Done
   "qd8-f16-qc4w-gemm-minmax": "qd8-f16-qc4w-gemm",  #Done Q
   "qd8-f16-qb4w-gemm-minmax": "qd8-f16-qb4w-gemm",  #Done Q
   "qd8-f16-qc8w-gemm-minmax": "qd8-f16-qc8w-gemm",  #Done Q
   "qd8-f16-qc8w-igemm-minmax": "qd8-f16-qc8w-igemm",
   "qd8-f32-qb4w-gemm-minmax": "qd8-f32-qb4w-gemm",  #Done Q
   "qd8-f32-qc4w-gemm-minmax": "qd8-f32-qc4w-gemm",  #Done Q
   "qd8-f32-qc8w-gemm-minmax": "qd8-f32-qc8w-gemm",  #Done
   "qd8-f32-qc8w-igemm-minmax": "qd8-f32-qc8w-igemm",  #Done Q
   "qu8-gemm-minmax-rndnu": "qu8-gemm",  #Done C
   "qs8-qc8w-gemm-minmax-fp32": "qs8-qc8w-gemm",
   "qs8-qc8w-igemm-minmax-fp32": "qs8-qc8w-igemm",
   "qu8-gemm-minmax-fp32": "qu8-gemm",  #Done C
   "qu8-igemm-minmax-fp32": "qu8-igemm",  #Done C
   "qu8-igemm-minmax-rndnu": "qu8-igemm",  #Done C
   "qp8-f32-qb4w-gemm-minmax": "qp8-f32-qb4w-gemm",  #Done M
   "qp8-f32-qc4w-gemm-minmax": "qp8-f32-qc4w-gemm",  #Done M
 }
 
 HEADER = """// Copyright 2023 Google LLC
 //
 // This source code is licensed under the BSD-style license found in the
 // LICENSE file in the root directory of this source tree.
 // Arguments are:
 // XNN_GEMM(arch_flags, fn_name, k_block, is_pipelined, mr, nr, kr, sr, mr_packed, datatype, params_type, init_fn, pack_fn, pack_stride)
 
 """
 
 def split_ukernel_name(name):
     match = re.fullmatch(
         r"xnn_((?:[a-z0-9]+_?)+)_(gemm|igemm|ppmm|goi)(_(minmax|relu|none)(_(fp32|rndnu|rndnu16|none))?)?_ukernel__(.+)",
         name,
     )
 
     if match:
         data_type = match.group(1)  # Extract data type like 'qp8_f32_qb4w'
         op = match.group(2)         # Extract operation type
         activation = match.group(3) or ""  # Extract activation type (e.g., minmax, relu, etc.)
         target_name = match.group(6)  # Extract target name
 
         common_name = name.split("__")[0]
         common_parts = common_name.split("_")
         param_spec = common_parts[-1]
 
         if "s" in param_spec:
             param_spec, sr = param_spec.split("s", 1)
             sr = int(sr)
         else:
             sr = 1
         if "c" in param_spec:
             param_spec, kr = param_spec.split("c", 1)
             kr = int(kr)
         else:
             kr = 1
         if "v" in param_spec:
             vector_tile = True
             param_spec, _ = param_spec.split("v", 1)
         else:
             vector_tile = False
 
         mr, nr = map(int, param_spec.split("x"))
 
         arch, isa, assembly = xnncommon.parse_target_name(target_name)
 
         mr_packed = re.search(r"mstep([0-9]+)", target_name)
         if mr_packed:
             mr_packed = mr // int(mr_packed.group(1))
         else:
             mr_packed = mr
 
         requantization = common_parts[-3] if len(common_parts) > 2 else None
         if requantization not in ["fp32", "rndnu", "rndnu16", "none"]:
             requantization = None
 
         print(f"Name: {name}, ISA: {isa}")
         return data_type, mr, nr, kr, sr, mr_packed, vector_tile, requantization, op, activation, arch, isa, assembly
 
     # Fallback logic for unknown names
     data_type = None
     for key in _DATATYPE_TO_CTYPE_MAP.keys():
         if key in name:
             data_type = key
             break
 
     if not data_type:
         data_type = "unknown"
 
     common_name, target_name = name.split("__", 1)
     param_spec = common_name.split("_")[-1]
 
     if "s" in param_spec:
         param_spec, sr = param_spec.split("s", 1)
         sr = int(sr)
     else:
         sr = 1
     if "c" in param_spec:
         param_spec, kr = param_spec.split("c", 1)
         kr = int(kr)
     else:
         kr = 1
     if "v" in param_spec:
         vector_tile = True
         param_spec, _ = param_spec.split("v", 1)
     else:
         vector_tile = False
 
     mr, nr = map(int, param_spec.split("x"))
 
     arch, isa, assembly = xnncommon.parse_target_name(target_name)
 
     print(f"Name: {name}, ISA: {isa}")
     return data_type, mr, nr, kr, sr, mr, vector_tile, "unknown", "", arch, isa, assembly
 
 isas = {
     "v6": "xnn_arch_arm_v6",
     "armsimd32": "xnn_arch_arm_v6",
     "vfpv2": "xnn_arch_arm_vfpv2",
     "vfpv3": "xnn_arch_arm_vfpv3",
     "neon": "xnn_arch_arm_neon",
     "neonfp16": "xnn_arch_arm_neon_fp16",
     "neonfma": "xnn_arch_arm_neon_fma",
     "neonv8": "xnn_arch_arm_neon_v8",
     "fp16arith": "xnn_arch_arm_fp16_arith",
     "neonfp16arith": "xnn_arch_arm_neon_fp16_arith",
     "neondotfp16arith":"xnn_arch_arm_neon_dot_fp16_arith",
     "neonbf16": "xnn_arch_arm_neon_bf16",
     "neondot": "xnn_arch_arm_neon_dot",
     "neon_i8mm": "xnn_arch_arm_neon_i8mm",
     "neoni8mm": "xnn_arch_arm_neon_i8mm",
     "sse": "0",
     "sse2": "0",
     "ssse3": "xnn_arch_x86_ssse3",
     "sse41": "xnn_arch_x86_sse4_1",
     "avx": "xnn_arch_x86_avx",
     "f16c": "xnn_arch_x86_f16c",
     "fma3": "xnn_arch_x86_fma3",
     "avx2": "xnn_arch_x86_avx2",
     "avx512f": "xnn_arch_x86_avx512f",
     "avx512vbmi": "xnn_arch_x86_avx512vbmi",
     "avx512skx": "xnn_arch_x86_avx512skx",
     "avx512vnni": "xnn_arch_x86_avx512vnni",
     "avx512vnnigfni": "xnn_arch_x86_avx512vnnigfni",
     "avx512amx": "xnn_arch_x86_avx512amx",
     "avx512fp16": "xnn_arch_x86_avx512fp16",
     "avxvnni": "xnn_arch_x86_avxvnni",
     "avxvnniint8": "xnn_arch_x86_avxvnniint8",
     "avx256skx": "xnn_arch_x86_avx256skx",
     "avx256vnni": "xnn_arch_x86_avx256vnni",
     "avx256vnnigfni": "xnn_arch_x86_avx256vnnigfni",
     "rvv": "xnn_arch_riscv_vector",
     "rvvfp16arith": "xnn_arch_riscv_vector_fp16_arith",
     "vlenb": "xnn_arch_riscv_vlenb",
     # xnn_arch_vsx = 1 << 0,
     # xnn_arch_vsx3 = 1 << 1,
     # xnn_arch_mma = 1 << 2,
     "is_x86": "xnn_arch_wasm_is_x86",
     "wasmblendvps": "xnn_arch_wasm_blendvps",
     "pshufb": "xnn_arch_wasm_pshufb",
     "sdot": "xnn_arch_wasm_sdot",
     "usdot": "xnn_arch_wasm_usdot",
     "fma": "xnn_arch_wasm_fma",
     "wasmpshufb": "xnn_arch_wasm_pshufb",
     "wasmsdot": "xnn_arch_wasm_sdot",
     "wasmusdot": "xnn_arch_wasm_usdot",
     "wasmfma": "xnn_arch_wasm_fma",
     "hvx": "xnn_arch_hvx",
     "wasm": "0",
     "wasmsimd": "0",
     "wasm32": "0",
     "wasmrelaxedsimd": "0",
     None: "0",
 }
 
 yamls_inverted = defaultdict(list)
 
 for i in yamls.items():
  yamls_inverted[i[1]].append(i[0])
 
 files = []
 hdrs = []
 for i in yamls_inverted.items():
  for j in i[1]:
   src_path = "/home/mcw/Documents/Google_Project/Internal_XNNPACK/src/" + i[0] + "/" + j + ".h"
   dst = src_path
 
   hdrs.append(src_path)
   files.append(j)
 
   output = HEADER
   in_define = ""
 
   src = "/home/mcw/Documents/Google_Project/Internal_XNNPACK/test/" + j + ".yaml"
 
   with codecs.open(src, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
     raise ValueError("expected a list of micro-kernels in the spec")
 
    for ukernel_spec in spec_yaml:
     name = ukernel_spec["name"]
     init_fn = ukernel_spec.get("init", "NULL")
     pack_fn = ukernel_spec.get("pack", "NULL")
     packed_stride = ukernel_spec.get("packed-stride", "NULL")
     kblock = ukernel_spec.get("k-block")
     pipelined = ukernel_spec.get("pipelined", "false")
     #  unsigned_inputs = ukernel_spec.get("unsigned-inputs", "false")
 
     data_type, mr, nr, kr, sr, mr_packed, is_pipelined, op, activation, arch, isa, assembly = split_ukernel_name(name)
     ctype = _DATATYPE_TO_CTYPE_MAP[data_type]
 
     guard = _ISA_TO_MACRO_MAP.get(isa, "")
     isa = isas[isa]
     arch = [_ARCH_TO_MACRO_MAP[i] for i in arch]
 
     if arch:
      if guard != "":
       define = "#if " + guard + " && (" + " || ".join(arch) + ")\n"
      else:
       define = "#if " + " || ".join(arch) + "\n"
     else:
      if guard != "":
       define = "#if " + guard + "\n"
      else:
       define = ""
 
     if in_define != define:
      if in_define != "":
       output += "#endif  // " + in_define[4:]
      output += "\n"
      output += define
      in_define = define
     
     FILE_TO_PARAMS_TYPE = {
        "bf16-gemm-minmax": "struct xnn_bf16_minmax_params", 
        "f16-gemm-minmax": "union xnn_f16_minmax_params", 
        "f16-f32acc-gemm-minmax": "union xnn_f16_minmax_params", 
        "f16-f32acc-igemm-minmax": "union xnn_f16_minmax_params",
        "f16-igemm-minmax": "union xnn_f16_minmax_params",   
        "f32-gemm": "struct xnn_f32_default_params",
        "f32-gemm-goi-minmax": "union xnn_f32_minmax_params",   
        "f32-gemm-minmax": "union xnn_f32_minmax_params",
        "f32-gemminc-minmax": "union xnn_f32_minmax_params",   
        "f32-gemm-relu": "struct xnn_f32_relu_params",   
        "f32-qc8w-gemm": "struct xnn_f32_default_params",
        "f32-igemm": "struct xnn_f32_default_params",
        "f32-igemm-minmax": "union xnn_f32_minmax_params",  
        "f32-igemm-relu": "struct xnn_f32_relu_params",   
        "f32-qc4w-gemm-minmax": "struct xnn_f32_qc4w_minmax_params",   
        "f32-qc8w-gemm-minmax": "union xnn_f32_minmax_params",   
        "f32-qc8w-gemm-relu": "struct xnn_f32_relu_params",   
        "f32-ppmm-minmax": "union xnn_f32_minmax_params",   
        "qd8-f16-qc4w-gemm-minmax": "struct xnn_f16_qc4w_minmax_params", #QuantP   
        "qd8-f16-qb4w-gemm-minmax": "struct xnn_f16_qb4w_minmax_params", #QuantP   
        "qd8-f16-qc8w-gemm-minmax": "union xnn_f16_minmax_params", #QuantP   
        "qd8-f16-qc8w-igemm-minmax": "union xnn_f16_minmax_params", #QuantP
        "qd8-f32-qb4w-gemm-minmax": "struct xnn_f32_qb4w_minmax_params", #QuantP   
        "qd8-f32-qc4w-gemm-minmax": "struct xnn_f32_qc4w_minmax_params", #QuantP   
        "qd8-f32-qc8w-gemm-minmax": "union xnn_f32_minmax_params", #QuantP
        "qd8-f32-qc8w-igemm-minmax": "union xnn_f32_minmax_params", #QuantP   
        "qu8-gemm-minmax-rndnu": "union xnn_qu8_conv_minmax_params",   
        "qs8-qc8w-gemm-minmax-fp32": "union xnn_qs8_qc8w_conv_minmax_params",
        "qs8-qc8w-igemm-minmax-fp32": "union xnn_qs8_qc8w_conv_minmax_params",
        "qu8-gemm-minmax-fp32": "union xnn_qu8_conv_minmax_params",   
        "qu8-igemm-minmax-fp32": "union xnn_qu8_conv_minmax_params",   
        "qu8-igemm-minmax-rndnu": "union xnn_qu8_conv_minmax_params",   
        "qp8-f32-qb4w-gemm-minmax": "struct xnn_f32_qb4w_minmax_params", #MinmaxP   
        "qp8-f32-qc4w-gemm-minmax": "union xnn_f32_minmax_params", #MinmaxP
     }
     
     params_type = FILE_TO_PARAMS_TYPE[j] if j in FILE_TO_PARAMS_TYPE else None
     print(f"params_type: {params_type}")
    
     output += "XNN_GEMM(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n" % (
      isa,
      name,
      kblock,
      "true" if pipelined else "false",
      mr,
      nr,
      kr,
      sr,
      mr_packed,
      ctype,
      params_type,
      init_fn,
      pack_fn,
      packed_stride
     )
 
    if in_define != "":
     output += "#endif  // " + in_define[4:] + "\n"
 
   with codecs.open(dst, "w", encoding="utf-8") as output_file:
    output_file.write(output)
 
 
 print("MICROKERNEL_DEPS = [")
 print(",\n".join(['    "' + i + '"' for i in hdrs]))
 print("]")
 print(" ".join(files))

except Exception as e:
    print("An error occurred:", e)

finally:
    print("COMPLETED")
    # playsound("/home/mcw/Downloads/sound.mp3")
