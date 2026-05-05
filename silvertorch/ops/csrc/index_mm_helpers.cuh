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

#include <torch/torch.h> // @manual

#ifndef __HIP_PLATFORM_AMD__
#include <cuda.h> // @manual
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#else
#include <hip_bf16.h> // @manual
#include <hip_fp16.h> // @manual
#include <hip_runtime.h> // @manual
#endif

namespace st::ops::simple_index_mm {

template <typename INPUT_T, typename RETURN_T>
struct should_handle_scaling_in_post_process
    : std::integral_constant<
          bool,
          std::is_same_v<INPUT_T, int32_t> &&
              std::is_same_v<RETURN_T, c10::Half>> {};

template <typename INPUT_T, typename RETURN_T>
constexpr bool should_handle_scaling_in_post_process_v =
    should_handle_scaling_in_post_process<INPUT_T, RETURN_T>::value;

static_assert(
    should_handle_scaling_in_post_process_v<int32_t, c10::Half> == true,
    "should_handle_scaling_in_post_process_v<int32_t, c10::Half> must be true");

template <typename INPUT_T, typename RETURN_T, typename DIVISOR_T>
__inline__ __device__ RETURN_T post_process(INPUT_T result, DIVISOR_T divisor) {
  if constexpr (should_handle_scaling_in_post_process_v<INPUT_T, RETURN_T>) {
    if constexpr (std::is_same_v<DIVISOR_T, c10::Half>) {
      return c10::Half(result) / divisor;
    }
    if constexpr (std::is_same_v<DIVISOR_T, int32_t>) {
      return c10::Half(__float2half(
          static_cast<float>(result) / static_cast<float>(divisor)));
    }
    static_assert(
        std::is_same_v<DIVISOR_T, c10::Half> ||
            std::is_same_v<DIVISOR_T, int32_t>,
        "Unsupported DIVISOR_T type for post_process.");
  }

  return result;
}

} // namespace st::ops::simple_index_mm
