// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/requantization.h"
#include "next_prime.h"
#include "replicable_random_device.h"

class AvgPoolMicrokernelTester {
 public:
  AvgPoolMicrokernelTester& output_pixels(size_t output_pixels) {
    assert(output_pixels != 0);
    this->output_pixels_ = output_pixels;
    return *this;
  }

  size_t output_pixels() const {
    return this->output_pixels_;
  }

  AvgPoolMicrokernelTester& step(size_t step) {
    assert(step != 0);
    this->step_ = step;
    return *this;
  }

  size_t step() const {
    return this->step_;
  }

  AvgPoolMicrokernelTester& input_offset(size_t input_offset) {
    assert(input_offset != 0);
    this->input_offset_ = input_offset;
    return *this;
  }

  size_t input_offset() const {
    return this->input_offset_;
  }

  AvgPoolMicrokernelTester& zero_index_mod2(size_t zero_index_mod2) {
    this->zero_index_mod2_ = zero_index_mod2;
    return *this;
  }

  size_t zero_index_mod2() const {
    return this->zero_index_mod2_;
  }

  AvgPoolMicrokernelTester& pooling_elements(size_t pooling_elements) {
    assert(pooling_elements != 0);
    this->pooling_elements_ = pooling_elements;
    return *this;
  }

  size_t pooling_elements() const {
    return this->pooling_elements_;
  }

  AvgPoolMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  AvgPoolMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  AvgPoolMicrokernelTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  float input_scale() const {
    return this->input_scale_;
  }

  AvgPoolMicrokernelTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  AvgPoolMicrokernelTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  float output_scale() const {
    return this->output_scale_;
  }

  AvgPoolMicrokernelTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  AvgPoolMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  AvgPoolMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  AvgPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_avgpool_minmax_unipass_ukernel_fn avgpool_minmax, xnn_init_f16_scaleminmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    xnnpack::Buffer<const xnn_float16*> indirect_input((output_pixels() - 1) * step() + pooling_elements());
    xnnpack::Buffer<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
      input_offset() + indirect_input.size() * channels());
    xnnpack::Buffer<xnn_float16> zero(channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16), 0);
    xnnpack::Buffer<xnn_float16> output((output_pixels() - 1) * output_stride() + channels());
    xnnpack::Buffer<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(xnn_float16), input.end(), std::nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const xnn_float16* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += row[c + input_offset()];
            }
          }
          output_ref[x * channels() + c] = acc / float(pooling_elements());
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min_as_float = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      float output_max_as_float = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;
      const xnn_float16 output_min_as_half = static_cast<xnn_float16>(output_min_as_float);
      const xnn_float16 output_max_as_half = static_cast<xnn_float16>(output_max_as_float);
      output_min_as_float = output_min_as_half;
      output_max_as_float = output_max_as_half;

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max_as_float), output_min_as_float);
      }

      // Prepare parameters.
      xnn_f16_scaleminmax_params params;
      init_params(&params, static_cast<xnn_float16>(1.0f / float(pooling_elements())), output_min_as_half, output_max_as_half);

      // Call optimized micro-kernel.
      avgpool_minmax(output_pixels(), pooling_elements(), channels(),
        reinterpret_cast<const xnn_float16**>(indirect_input.data()), input_offset() * sizeof(xnn_float16), zero.data(),
        output.data(),
        step() * sizeof(void*),
        (output_stride()) * sizeof(xnn_float16),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(output[x * output_stride() + c], output_min_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_NEAR(
              output[x * output_stride() + c],
              output_ref[x * channels() + c],
              std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) * 3.0e-3f))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_avgpool_minmax_unipass_ukernel_fn avgpool_minmax, xnn_init_f32_scaleminmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    xnnpack::Buffer<const float*> indirect_input((output_pixels() - 1) * step() + pooling_elements());
    xnnpack::Buffer<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      input_offset() + indirect_input.size() * channels());
    xnnpack::Buffer<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float), 0.0f);
    xnnpack::Buffer<float> output((output_pixels() - 1) * output_stride() + channels());
    xnnpack::Buffer<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(float), input.end(), std::nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const float* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += row[c + input_offset()];
            }
          }
          output_ref[x * channels() + c] = acc / float(pooling_elements());
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float output_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Prepare parameters.
      xnn_f32_scaleminmax_params params;
      init_params(&params, 1.0f / float(pooling_elements()), output_min, output_max);

      // Call optimized micro-kernel.
      avgpool_minmax(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float), zero.data(),
        output.data(),
        step() * sizeof(void*),
        (output_stride()) * sizeof(float),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_NEAR(
              output[x * output_stride() + c],
              output_ref[x * channels() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-6f)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f16_pavgpool_minmax_unipass_ukernel_fn pavgpool_minmax, xnn_init_f16_scaleminmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;
    std::uniform_real_distribution<float> m32dist(0.1f, 0.5f);

    xnnpack::Buffer<const xnn_float16*> indirect_input((output_pixels() - 1) * step() + pooling_elements());
    xnnpack::Buffer<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
      input_offset() + indirect_input.size() * channels());
    xnnpack::Buffer<xnn_float16> zero(channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16), 0);
    xnnpack::Buffer<xnn_float16> multiplier(output_pixels());
    xnnpack::Buffer<xnn_float16> output((output_pixels() - 1) * output_stride() + channels());
    xnnpack::Buffer<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(xnn_float16), input.end(), std::nanf(""));
      std::generate(multiplier.begin(), multiplier.end(), [&]() { return m32dist(rng); });

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const xnn_float16* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += row[c + input_offset()];
            }
          }
          output_ref[x * channels() + c] = acc * multiplier[x];
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min_as_float = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      float output_max_as_float = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;
      const xnn_float16 output_min_as_half = static_cast<xnn_float16>(output_min_as_float);
      const xnn_float16 output_max_as_half = static_cast<xnn_float16>(output_max_as_float);
      output_min_as_float = output_min_as_half;
      output_max_as_float = output_max_as_half;

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max_as_float), output_min_as_float);
      }

      // Prepare parameters.
      xnn_f16_scaleminmax_params params;
      init_params(&params, 0.0f, output_min_as_half, output_max_as_half);

      // Call optimized micro-kernel.
      pavgpool_minmax(output_pixels(), pooling_elements(), channels(),
        reinterpret_cast<const xnn_float16**>(indirect_input.data()), input_offset() * sizeof(xnn_float16), zero.data(),
        multiplier.data(), output.data(),
        step() * sizeof(void*),
        (output_stride()) * sizeof(xnn_float16),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(output[x * output_stride() + c], output_min_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_NEAR(
              output[x * output_stride() + c],
              output_ref[x * channels() + c],
              std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) * 3.0e-3f))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_pavgpool_minmax_unipass_ukernel_fn pavgpool_minmax, xnn_init_f32_scaleminmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;
    std::uniform_real_distribution<float> m32dist(0.1f, 0.5f);

    xnnpack::Buffer<const float*> indirect_input((output_pixels() - 1) * step() + pooling_elements());
    xnnpack::Buffer<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      input_offset() + indirect_input.size() * channels());
    xnnpack::Buffer<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float), 0.0f);
    xnnpack::Buffer<float> multiplier(output_pixels());
    xnnpack::Buffer<float> output((output_pixels() - 1) * output_stride() + channels());
    xnnpack::Buffer<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(float), input.end(), std::nanf(""));
      std::generate(multiplier.begin(), multiplier.end(), [&]() { return m32dist(rng); });

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const float* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += row[c + input_offset()];
            }
          }
          output_ref[x * channels() + c] = acc * multiplier[x];
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float output_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Prepare parameters.
      xnn_f32_scaleminmax_params params;
      init_params(&params, 0.0f, output_min, output_max);

      // Call optimized micro-kernel.
      pavgpool_minmax(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float), zero.data(),
        multiplier.data(), output.data(),
        step() * sizeof(void*),
        (output_stride()) * sizeof(float),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_NEAR(
              output[x * output_stride() + c],
              output_ref[x * channels() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-6f)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  struct Kernel {
    explicit Kernel(xnn_f16_avgpool_minmax_unipass_ukernel_fn fn, xnn_init_f16_scaleminmax_params_fn init) {
      dispatch = [fn, init](const AvgPoolMicrokernelTester& tester) { tester.Test(fn, init); };
    }
    explicit Kernel(xnn_f16_pavgpool_minmax_unipass_ukernel_fn fn, xnn_init_f16_scaleminmax_params_fn init) {
      dispatch = [fn, init](const AvgPoolMicrokernelTester& tester) { tester.Test(fn, init); };
    }
    explicit Kernel(xnn_f32_avgpool_minmax_unipass_ukernel_fn fn, xnn_init_f32_scaleminmax_params_fn init) {
      dispatch = [fn, init](const AvgPoolMicrokernelTester& tester) { tester.Test(fn, init); };
    }
    explicit Kernel(xnn_f32_pavgpool_minmax_unipass_ukernel_fn fn, xnn_init_f32_scaleminmax_params_fn init) {
      dispatch = [fn, init](const AvgPoolMicrokernelTester& tester) { tester.Test(fn, init); };
    }
    std::function<void(const AvgPoolMicrokernelTester&)> dispatch;
  };

  void Test(const Kernel& kernel) const {
    kernel.dispatch(*this);
  }

 private:
  size_t output_pixels_{1};
  size_t pooling_elements_{1};
  size_t channels_{1};
  size_t input_offset_{0};
  size_t zero_index_mod2_{SIZE_MAX};
  size_t step_{1};
  size_t output_stride_{0};
  float input_scale_{1.25f};
  float output_scale_{0.75f};
  uint8_t input_zero_point_{121};
  uint8_t output_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{3};
};
