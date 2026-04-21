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

#pragma once
#include <ATen/ATen.h>

namespace st::ops::faster_repeat_interleave {

template <typename repeat_t, bool cuda_t>
repeat_t const* assure_repeats_cumsum(
    at::Tensor const& repeats,
    std::optional<at::Tensor> const& repeat_cumsum,
    at::Tensor& repeat_cumsum_placeholder,
    int64_t& result_size) {
  TORCH_CHECK(
      repeats.dim() == 1,
      "faster_repeat_interleave only accept 1D vector as repeat");
  if constexpr (cuda_t) {
    TORCH_CHECK(repeats.is_cuda());
  }
  TORCH_CHECK(
      repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
      "repeats has to be Long or Int tensor");
  TORCH_CHECK(repeats.is_contiguous());
  repeat_t const* repeat_cumsum_data_ptr = nullptr;
  if (repeat_cumsum) {
    TORCH_CHECK(
        repeat_cumsum->dim() == 1,
        "faster_repeat_interleave only accept 1D repeat_cumsum");
    TORCH_CHECK(
        repeats.scalar_type() == repeat_cumsum->scalar_type(),
        "faster_repeat_interleave expects that repeats and repeat_cumsum have same dtype");
    if constexpr (cuda_t) {
      TORCH_CHECK(repeat_cumsum->is_cuda());
    }
    TORCH_CHECK(repeat_cumsum->is_contiguous());
    repeat_cumsum_data_ptr = repeat_cumsum->data_ptr<repeat_t>();
    result_size = (*repeat_cumsum)[-1].template item<int64_t>();
  } else {
    repeat_cumsum_placeholder = repeats.cumsum(0, repeats.scalar_type());
    repeat_cumsum_data_ptr = repeat_cumsum_placeholder.data_ptr<repeat_t>();
    result_size = repeat_cumsum_placeholder[-1].template item<int64_t>();
  }
  return repeat_cumsum_data_ptr;
}

} // namespace st::ops::faster_repeat_interleave
