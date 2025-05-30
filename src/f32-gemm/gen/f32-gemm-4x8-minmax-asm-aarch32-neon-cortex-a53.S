// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/4x8-aarch32-neon-cortex-a53.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

.syntax unified


// void xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r5 -> sp + 0
//     const float* a,                       r3
//     size_t a_stride,          sp + 100 -> (r7)
//     const float* w,           sp + 104 -> r9
//     float* c,                 sp + 108 -> r11
//     size_t cm_stride,         sp + 112 -> (r6)
//     size_t cn_stride,         sp + 116 -> (r0)
//     const xnn_f32_minmax_params* params)  sp + 120 -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r3  d0 d4
// A1  r12  d1 d5
// A2  r10  d2 d6
// A3   r7  d3 d7
// B    r9  d8,  d9, d10, d11
// B       d12, d13, d14, d15
// C0  r11 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15
// clamp  (r5) d4 d5 d6 d7
// temp r0, r2 for Cortex-A53 loads
// unused r14 (lr)

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53
        .arm
#ifndef __APPLE__
        .arch       armv7-a
        .fpu        neon
#endif
        # Push 100 bytes
        # r2 will be reloaded in outer loop
        VPUSH       {d8-d15}                                // 64
        PUSH        {r2, r4, r5, r6, r7, r8, r9, r10, r11}  // +36 = 100

        LDR         r7, [sp, 100]           // a_stride
        LDR         r11, [sp, 108]          // c
        LDR         r6, [sp, 112]           // cm_stride
        LDR         r9, [sp, 104]           // w

        # Clamp A and C pointers
        CMP         r0, 2                   // if mr >= 2
        ADD         r12, r3, r7             //   a1 = a0 + a_stride
        ADD         r4, r11, r6             //   c1 = c0 + cm_stride
        MOVLO       r12, r3                 // a1
        MOVLO       r4, r11                 // c1
                                        // if mr > 2
        ADD         r10, r12, r7            //   a2 = a1 + a_stride
        ADD         r8, r4, r6              //   c2 = c1 + cm_stride
        MOVLS       r10, r12                // a2
        MOVLS       r8, r4                  // c2

        CMP         r0, 4                   // if mr >=4
        ADD         r7, r10, r7             //   a3 = a2 + a_stride
        ADD         r6, r8, r6              //   c3 = c2 + cm_stride
        MOVLO       r7, r10                 // a3
        MOVLO       r6, r8                  // c3

        .p2align    3
0:
        # Load initial bias from w into accumulators
        VLDM        r9!, {d16-d19}          // Bias

        SUBS        r5, r2, 16              // kc - 16
        VMOV        q10, q8
        VMOV        q11, q9
        VMOV        q12, q8
        VMOV        q13, q9
        VMOV        q14, q8
        VMOV        q15, q9
        BLO         4f                      // less than 4 channels?

        # Prologue
        VLD1.32     {d0},  [r3]!            // A0
        VLD1.32     {d1}, [r12]!            // A1
        VLD1.32     {d2}, [r10]!            // A2
        VLD1.32     {d3},  [r7]!            // A3
        SUBS        r5, r5, 16
        VLDM        r9, {d8-d11}            // B0
        LDR         r0, [r9, 56]            // B1 low   VMOV is in BLOCK 0
        LDR         r2, [r9, 60]            // B1 high
        VLDR        d13, [r9, 40]           // B1

        BLO         2f                      // less than 4 channels?  skip main loop

        # Main loop - 4 floats of A (16 bytes)
        # 32 FMA + 8 LD64 A + 8 LDR B
        .p2align    3
1:
        # First group of 16 FMA, Second group loads
        # BLOCK 0
        VLD1.32     {d4}, [r3]!             // A0
        VMOV        d15, r0, r2             // b1 VMOV b from second group
        VMLA.F32    q8, q4, d0[0]
        LDR         r0, [r12]               // A1 low
        VMLA.F32    q10, q4, d1[0]
        LDR         r2, [r12, 4]            // A1 high
        VMLA.F32    q12, q4, d2[0]

        # BLOCK 1
        VLDR        d12, [r9, 32]           // B1
        VMOV        d5, r0, r2              // a1 VMOV
        VMLA.F32    q14, q4, d3[0]
        LDR         r0, [r9, 72]            // B0 low
        VMLA.F32    q9, q5, d0[0]
        LDR         r2, [r9, 76]            // B0 high
        VMLA.F32    q11, q5, d1[0]

        # BLOCK 2
        VLD1.32     {d6}, [r10]!            // A2
        VMOV        d9, r0, r2              // b0 VMOV
        VMLA.F32    q13, q5, d2[0]
        LDR         r0, [r7]                // A3 low
        VMLA.F32    q15, q5, d3[0]
        LDR         r2, [r7, 4]             // A3 high
        VMLA.F32    q8, q6, d0[1]

        # BLOCK 3
        VLDR        d14, [r9, 48]           // B1
        VMOV        d7, r0, r2              // a3 VMOV
        VMLA.F32    q10, q6, d1[1]
        LDR         r0, [r9, 88]            // B0 low
        VMLA.F32    q12, q6, d2[1]
        LDR         r2, [r9, 92]            // B0 high
        VMLA.F32    q14, q6, d3[1]

        # BLOCK 4
        VLDR        d8, [r9, 64]            // B0
        VMOV        d11, r0, r2             // B0 VMOV
        VMLA.F32    q9, q7, d0[1]
        LDR         r0, [r9, 104]           // B1 low   VMOV is in BLOCK 0
        VMLA.F32    q11, q7, d1[1]
        LDR         r2, [r9, 108]           // B1 high
        VMLA.F32    q13, q7, d2[1]

        # BLOCK 5
        VLDR        d10, [r9, 80]           // B0
        VMOV        d13, r0, r2             // b1 VMOV b from second group
        VMLA.F32    q15, q7, d3[1]
        LDR         r0, [r9, 120]           // B1 low   VMOV is in BLOCK 0
        NOP
        LDR         r2, [r9, 124]           // B1 high
        NOP

        # Second group of 16 FMA, First group of loads
        # BLOCK 0
        VLD1.32     {d0}, [r3]!             // A0
        VMOV        d15, r0, r2             // b1 VMOV b from second group
        VMLA.F32    q8, q4, d4[0]
        LDR         r0, [r12, 8]            // A1 low
        VMLA.F32    q10, q4, d5[0]
        LDR         r2, [r12, 12]           // A1 high
        VMLA.F32    q12, q4, d6[0]
        # NOP

        # BLOCK 1
        VLDR        d12, [r9, 96]           // B1
        VMOV        d1, r0, r2              // a1 VMOV
        VMLA.F32    q14, q4, d7[0]
        LDR         r0, [r9, 136]           // B0 low
        VMLA.F32    q9, q5, d4[0]
        LDR         r2, [r9, 140]           // B0 high
        VMLA.F32    q11, q5, d5[0]
        # NOP

        # BLOCK 2
        VLD1.32     {d2}, [r10]!            // A2
        VMOV        d9, r0, r2              // b0 VMOV
        VMLA.F32    q13, q5, d6[0]
        LDR         r0, [r7, 8]             // A3 low
        VMLA.F32    q15, q5, d7[0]
        LDR         r2, [r7, 12]            // A3 high
        VMLA.F32    q8, q6, d4[1]
        # NOP

        # BLOCK 3
        VLDR        d14, [r9, 112]          // B1
        VMOV        d3, r0, r2              // a3 VMOV
        VMLA.F32    q10, q6, d5[1]
        LDR         r0, [r9, 152]           // B0 low
        VMLA.F32    q12, q6, d6[1]
        LDR         r2, [r9, 156]           // B0 high
        VMLA.F32    q14, q6, d7[1]
        ADD         r12, r12, 16            // A1++

        # BLOCK 4
        VLDR        d8, [r9, 128]           // B0
        VMOV        d11, r0, r2             // B0 VMOV
        VMLA.F32    q9, q7, d4[1]
        LDR         r0, [r9, 168]           // B1 low
        VMLA.F32    q11, q7, d5[1]
        LDR         r2, [r9, 172]           // B1 high
        VMLA.F32    q13, q7, d6[1]
        ADD         r7, r7, 16              // A3++

        # BLOCK 5
        VLDR        d10, [r9, 144]          // B0
        VMOV        d13, r0, r2             // b1 VMOV b
        VMLA.F32    q15, q7, d7[1]
        LDR         r0, [r9, 184]           // B1 low   VMOV is in BLOCK 0
        SUBS        r5, r5, 16
        LDR         r2, [r9, 188]           // B1 high
        ADD         r9, r9, 128             // B++
        BHS         1b

        # Epilogue - 4 floats of A (16 bytes)
2:
        # First group of 16 FMA, Second group loads
        # BLOCK 0
        VLD1.32     {d4}, [r3]!             // A0
        VMOV        d15, r0, r2             // b1 VMOV b from second group
        VMLA.F32    q8, q4, d0[0]
        LDR         r0, [r12]               // A1 low
        VMLA.F32    q10, q4, d1[0]
        LDR         r2, [r12, 4]            // A1 high
        VMLA.F32    q12, q4, d2[0]
        # NOP

        # BLOCK 1
        VLDR        d12, [r9, 32]           // B1
        VMOV        d5, r0, r2              // a1 VMOV
        VMLA.F32    q14, q4, d3[0]
        LDR         r0, [r9, 72]            // B0 low
        VMLA.F32    q9, q5, d0[0]
        LDR         r2, [r9, 76]            // B0 high
        VMLA.F32    q11, q5, d1[0]
        # NOP

        # BLOCK 2
        VLD1.32     {d6}, [r10]!            // A2
        VMOV        d9, r0, r2              // b0 VMOV
        VMLA.F32    q13, q5, d2[0]
        LDR         r0, [r7]                // A3 low
        VMLA.F32    q15, q5, d3[0]
        LDR         r2, [r7, 4]             // A3 high
        VMLA.F32    q8, q6, d0[1]
        # NOP

        # BLOCK 3
        VLDR        d14, [r9, 48]           // B1
        VMOV        d7, r0, r2              // a3 VMOV
        VMLA.F32    q10, q6, d1[1]
        LDR         r0, [r9, 88]            // B0 low
        VMLA.F32    q12, q6, d2[1]
        LDR         r2, [r9, 92]            // B0 high
        VMLA.F32    q14, q6, d3[1]
        # NOP

        # BLOCK 4
        VLDR        d8, [r9, 64]            // B0
        VMOV        d11, r0, r2             // B0 VMOV
        VMLA.F32    q9, q7, d0[1]
        LDR         r0, [r9, 104]           // B1 low
        VMLA.F32    q11, q7, d1[1]
        LDR         r2, [r9, 108]           // B1 high
        VMLA.F32    q13, q7, d2[1]
        # NOP

        # BLOCK 5
        VLDR        d10, [r9, 80]           // B0
        VMOV        d13, r0, r2             // b1 VMOV b
        VMLA.F32    q15, q7, d3[1]
        LDR         r0, [r9, 120]           // B1 low   VMOV is in BLOCK 0
        NOP
        LDR         r2, [r9, 124]           // B1 high
        NOP
        NOP

        # Second group of 16 FMA, First group of loads
        # BLOCK 0
        VLDR        d12, [r9, 96]           // B1
        VMOV        d15, r0, r2             // b1 VMOV b from second group
        VMLA.F32    q8, q4, d4[0]
        VMLA.F32    q10, q4, d5[0]
        VMLA.F32    q12, q4, d6[0]

        # BLOCK 1
        VLDR        d14, [r9, 112]          // B1
        VMLA.F32    q14, q4, d7[0]
        VMLA.F32    q9, q5, d4[0]
        VMLA.F32    q11, q5, d5[0]
        ADD         r12, r12, 8             // A1++

        # BLOCK 2
        ADD         r7, r7, 8               // A3++ VLDR B1 lands here
        ADD         r9, r9, 128             // B++
        VMLA.F32    q13, q5, d6[0]
        VMLA.F32    q15, q5, d7[0]
        VMLA.F32    q8, q6, d4[1]

        # BLOCK 3
        VMLA.F32    q10, q6, d5[1]
        VMLA.F32    q12, q6, d6[1]
        VMLA.F32    q14, q6, d7[1]
        TST         r5, 15

        # BLOCK 4
        VMLA.F32    q9, q7, d4[1]
        VMLA.F32    q11, q7, d5[1]
        VMLA.F32    q13, q7, d6[1]

        # BLOCK 5
        VMLA.F32    q15, q7, d7[1]

        # Is there a remainder?- 1 to 3 floats of A (4, 8 or 12 bytes)
        BNE         4f

        .p2align    3
3:
        # Load params pointer
        LDR         r0, [sp, 116]           // cn_stride
        LDR         r5, [sp, 120]           // params
        LDR         r2, [sp]                // kc
        SUBS        r1, r1, 8

        # Load min/max values
        VLD1.32     {d4[],d5[]}, [r5]!
        VLD1.32     {d6[],d7[]}, [r5]

        # Clamp
        VMAX.F32    q8,  q8, q2
        VMAX.F32    q9,  q9, q2
        VMAX.F32    q10, q10, q2
        VMAX.F32    q11, q11, q2
        VMAX.F32    q12, q12, q2
        VMAX.F32    q13, q13, q2
        VMAX.F32    q14, q14, q2
        VMAX.F32    q15, q15, q2
        VMIN.F32    q8,  q8, q3
        VMIN.F32    q9,  q9, q3
        VMIN.F32    q10, q10, q3
        VMIN.F32    q11, q11, q3
        VMIN.F32    q12, q12, q3
        VMIN.F32    q13, q13, q3
        VMIN.F32    q14, q14, q3
        VMIN.F32    q15, q15, q3

        # Store full 4 x 8
        BLO         6f
        VST1.32     {d16-d19}, [r11], r0
        SUB         r7, r7, r2
        VST1.32     {d20-d23}, [r4], r0
        SUB         r10, r10, r2
        VST1.32     {d24-d27}, [r8], r0
        SUB         r12, r12, r2
        VST1.32     {d28-d31}, [r6], r0
        SUB         r3, r3, r2
        BHI         0b

        ADD         sp, sp, 4
        POP         {r4, r5, r6, r7, r8, r9, r10, r11}
        VPOP        {d8-d15}
        BX          lr

        .p2align    3
4:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TST         r5, 8
        BEQ         5f

        # Remainder - 2 floats of A (8 bytes)
        VLD1.32     {d0}, [r3]!             // A0
        VLDM        r9!, {d8-d11}           // B0
        VLD1.32     {d1}, [r12]!            // A1
        VLD1.32     {d2}, [r10]!            // A2
        VLD1.32     {d3}, [ r7]!            // A3

        VMLA.F32    q8, q4, d0[0]
        VMLA.F32    q9, q5, d0[0]
        VMLA.F32    q10, q4, d1[0]
        VMLA.F32    q11, q5, d1[0]
        VLDM        r9!, {d12-d15}          // B1
        VMLA.F32    q12, q4, d2[0]
        VMLA.F32    q13, q5, d2[0]
        VMLA.F32    q14, q4, d3[0]
        VMLA.F32    q15, q5, d3[0]
        VMLA.F32    q8, q6, d0[1]
        VMLA.F32    q9, q7, d0[1]
        VMLA.F32    q10, q6, d1[1]
        VMLA.F32    q11, q7, d1[1]
        VMLA.F32    q12, q6, d2[1]
        VMLA.F32    q13, q7, d2[1]
        VMLA.F32    q14, q6, d3[1]
        VMLA.F32    q15, q7, d3[1]

        # Is there a remainder?- 1 float of A (4 bytes)
        TST         r5, 4
        BEQ         3b

5:
        # Remainder- 1 float of A (4 bytes)
        VLDM        r3!,  {s0}              // A0
        VLDM        r9!, {d8-d11}           // B0
        VLDM        r12!, {s2}              // A1
        VLDM        r10!, {s4}              // A2
        VLDM        r7!, {s6}               // A3
        VMLA.F32    q8, q4, d0[0]
        VMLA.F32    q9, q5, d0[0]
        VMLA.F32    q10, q4, d1[0]
        VMLA.F32    q11, q5, d1[0]
        VMLA.F32    q12, q4, d2[0]
        VMLA.F32    q13, q5, d2[0]
        VMLA.F32    q14, q4, d3[0]
        VMLA.F32    q15, q5, d3[0]
        B           3b

        # Store odd width
6:
        TST         r1, 4
        BEQ         7f
        VST1.32     {d16-d17}, [r11]!
        VST1.32     {d20-d21},  [r4]!
        VMOV        q8,  q9
        VMOV        q10, q11
        VST1.32     {d24-d25},  [r8]!
        VST1.32     {d28-d29},  [r6]!
        VMOV        q12, q13
        VMOV        q14, q15

7:
        TST         r1, 2
        BEQ         8f
        VST1.32     {d16}, [r11]!
        VST1.32     {d20},  [r4]!
        VMOV        d16, d17
        VMOV        d20, d21
        VST1.32     {d24},  [r8]!
        VST1.32     {d28},  [r6]!
        VMOV        d24, d25
        VMOV        d28, d29

8:
        TST         r1, 1
        BEQ         9f
        VST1.32     {d16[0]}, [r11]
        VST1.32     {d20[0]},  [r4]
        VST1.32     {d24[0]},  [r8]
        VST1.32     {d28[0]},  [r6]

9:
        ADD         sp, sp, 4
        POP         {r4, r5, r6, r7, r8, r9, r10, r11}
        VPOP        {d8-d15}
        BX          lr

END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
