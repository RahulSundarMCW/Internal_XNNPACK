#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qu8-rdsum/scalar.c.in -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-scalar.c &

################################## ARM NEON ###################################
tools/xngen src/qu8-rdsum/neon.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-neon-u16.c &
tools/xngen src/qu8-rdsum/neon.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-neon-u32.c &
tools/xngen src/qu8-rdsum/neon.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-neon-u64.c &

################################### x86 SSSE3 #################################
tools/xngen src/qu8-rdsum/ssse3.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-ssse3-c16.c &
tools/xngen src/qu8-rdsum/ssse3.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-ssse3-c32.c &
tools/xngen src/qu8-rdsum/ssse3.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-ssse3-c64.c &

################################### Wasm SIMD #################################
tools/xngen src/qs8-rdsum/wasmsimd.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -D DATATYPE=QU8 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-wasmsimd-c16.c &
tools/xngen src/qs8-rdsum/wasmsimd.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -D DATATYPE=QU8 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-wasmsimd-c32.c &
tools/xngen src/qs8-rdsum/wasmsimd.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -D DATATYPE=QU8 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-wasmsimd-c64.c &

################################### RISC-V Vector #############################
tools/xngen src/qu8-rdsum/rvv.c.in -D LMUL=1 -D ACCUMULATORS=7 -D DATATYPE=QU8 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-rvv-u1v.c &
tools/xngen src/qu8-rdsum/rvv.c.in -D LMUL=2 -D ACCUMULATORS=7 -D DATATYPE=QU8 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-rvv-u2v.c &

wait
