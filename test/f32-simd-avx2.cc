// Auto-generated file. Do not edit!
//   Template: test/f32-simd.cc.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/common.h"

#if XNN_ARCH_X86 || XNN_ARCH_X86_64

#include "xnnpack/isa-checks.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "xnnpack/simd/f32-avx2.h"

#include "replicable_random_device.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace xnnpack {

class F32SimdAVX2Test : public ::testing::Test {
 protected:
  void SetUp() override {
    TEST_REQUIRES_X86_AVX2;
    inputs_.resize(3 * xnn_simd_size_f32);
    output_.resize(xnn_simd_size_f32);
    std::uniform_real_distribution<float> f32dist(-10.0f, 10.0f);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return f32dist(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<float> inputs_;
  std::vector<float> output_;
};

TEST_F(F32SimdAVX2Test, SetZero) {
  xnn_storeu_f32(output_.data(), xnn_zero_f32());
  EXPECT_THAT(output_, testing::Each(testing::Eq(0.0f)));
}

TEST_F(F32SimdAVX2Test, Add) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_add_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] + inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdAVX2Test, Mul) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_mul_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] * inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdAVX2Test, Fmadd) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t c =
      xnn_loadu_f32(inputs_.data() + 2 * xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_fmadd_f32(a, b, c);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
#if XNN_SIMD_HAS_NATIVE_FMA
    // If an arch claims to support FMA, it better also round things correctly.
    ASSERT_EQ(output_[k],
              static_cast<float>(
                  static_cast<double>(inputs_[k]) *
                      static_cast<double>(inputs_[k + xnn_simd_size_f32]) +
                  static_cast<double>(inputs_[k + 2 * xnn_simd_size_f32])));
#else
    ASSERT_EQ(output_[k],
              inputs_[k] * inputs_[k + xnn_simd_size_f32] +
                  inputs_[k + 2 * xnn_simd_size_f32]);
#endif  // XNN_SIMD_HAS_NATIVE_FMA
  }
}

TEST_F(F32SimdAVX2Test, Fmsub) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t c =
      xnn_loadu_f32(inputs_.data() + 2 * xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_fmsub_f32(a, b, c);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
#if XNN_SIMD_HAS_NATIVE_FMA
    // If an arch claims to support FMA, it better also round things correctly.
    ASSERT_EQ(output_[k],
              static_cast<float>(
                  static_cast<double>(inputs_[k]) *
                      static_cast<double>(inputs_[k + xnn_simd_size_f32]) -
                  static_cast<double>(inputs_[k + 2 * xnn_simd_size_f32])));
#else
    ASSERT_EQ(output_[k],
              inputs_[k] * inputs_[k + xnn_simd_size_f32] -
                  inputs_[k + 2 * xnn_simd_size_f32]);
#endif  // XNN_SIMD_HAS_NATIVE_FMA
  }
}

TEST_F(F32SimdAVX2Test, Fnmadd) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t c =
      xnn_loadu_f32(inputs_.data() + 2 * xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_fnmadd_f32(a, b, c);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
#if XNN_SIMD_HAS_NATIVE_FMA
    // If an arch claims to support FMA, it better also round things correctly.
    ASSERT_EQ(output_[k],
              static_cast<float>(
                  static_cast<double>(-inputs_[k]) *
                      static_cast<double>(inputs_[k + xnn_simd_size_f32]) +
                  static_cast<double>(inputs_[k + 2 * xnn_simd_size_f32])));
#else
    ASSERT_EQ(output_[k],
              -inputs_[k] * inputs_[k + xnn_simd_size_f32] +
                  inputs_[k + 2 * xnn_simd_size_f32]);
#endif  // XNN_SIMD_HAS_NATIVE_FMA
  }
}

TEST_F(F32SimdAVX2Test, Sub) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_sub_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] - inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdAVX2Test, Div) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_div_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_NEAR(output_[k], inputs_[k] / inputs_[k + xnn_simd_size_f32],
    2 * std::numeric_limits<float>::epsilon() * std::abs(output_[k]));
  }
}

TEST_F(F32SimdAVX2Test, Max) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_max_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::max(inputs_[k], inputs_[k + xnn_simd_size_f32]));
  }
}

TEST_F(F32SimdAVX2Test, Min) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_min_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::min(inputs_[k], inputs_[k + xnn_simd_size_f32]));
  }
}

TEST_F(F32SimdAVX2Test, Abs) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t res = xnn_abs_f32(a);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::abs(inputs_[k]));
  }
}

TEST_F(F32SimdAVX2Test, Neg) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t res = xnn_neg_f32(a);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], -inputs_[k]);
  }
}

TEST_F(F32SimdAVX2Test, And) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_and_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(*(uint32_t *)&output_[k],
              *(uint32_t *)&inputs_[k] &
                  *(uint32_t *)&inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdAVX2Test, Or) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_or_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(*(uint32_t *)&output_[k],
              *(uint32_t *)&inputs_[k] |
                  *(uint32_t *)&inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdAVX2Test, Xor) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_xor_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(*(uint32_t *)&output_[k],
              *(uint32_t *)&inputs_[k] ^
                  *(uint32_t *)&inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdAVX2Test, StoreTail) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_f32;
      num_elements++) {
    std::fill(output_.begin(), output_.end(), 0.0f);
    xnn_store_tail_f32(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]);
    }
    for (size_t k = num_elements; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(output_[k], 0.0f);
    }
  }
}

}  // namespace xnnpack

#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64