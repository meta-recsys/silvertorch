// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

namespace st::ops::faster_repeat_interleave {
using at::Tensor;

Tensor faster_repeat_interleave_cuda(
    const Tensor& input,
    const Tensor& repeats,
    const std::optional<Tensor>& repeats_cumsum);

Tensor faster_repeat_interleave_with_cumsum_cuda(
    const Tensor& input,
    const Tensor& repeats_cumsum,
    int64_t desired_length,
    int64_t padding_value);

Tensor consecutive_int32_repeat_interleave_cuda(
    const Tensor& repeats,
    int64_t start,
    const std::optional<Tensor>& repeats_cumsum);

Tensor consecutive_int64_repeat_interleave_cuda(
    const Tensor& repeats,
    int64_t start,
    const std::optional<Tensor>& repeats_cumsum);

Tensor consecutive_int32_repeat_interleave_with_cumsum_cuda(
    const Tensor& repeats_cumsum,
    int64_t start);

Tensor consecutive_int64_repeat_interleave_with_cumsum_cuda(
    const Tensor& repeats_cumsum,
    int64_t start);

void consecutive_int32_repeat_interleave_with_cumsum_cuda_raw(
    const int32_t repeats_size,
    const int32_t* repeats_cumsum,
    const int32_t start,
    const int32_t output_size,
    int32_t* output,
    const c10::cuda::CUDAStream& stream = at::cuda::getCurrentCUDAStream());

void consecutive_int64_repeat_interleave_with_cumsum_cuda_raw(
    const int64_t repeats_size,
    const int64_t* repeats_cumsum,
    const int64_t start,
    const int64_t output_size,
    int64_t* output,
    const c10::cuda::CUDAStream& stream = at::cuda::getCurrentCUDAStream());

Tensor arange_interleave_with_cumsum_cuda(
    const Tensor& counts_cumsum,
    const Tensor& starts,
    int64_t step);

} // namespace st::ops::faster_repeat_interleave
