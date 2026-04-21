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
#include <cstdint>
#include <vector>

namespace at {
class Tensor;
} // namespace at

// Drop-in replacement for glog's CHECK_NOTNULL using torch's TORCH_CHECK,
// so the OSS build does not require folly/glog. Returns the pointer after
// asserting it is non-null.
#ifndef CHECK_NOTNULL
#define CHECK_NOTNULL(x)                                  \
  ([&]() -> decltype(x) {                                 \
    auto* _ptr = (x);                                     \
    TORCH_CHECK(_ptr != nullptr, #x " returned nullptr"); \
    return _ptr;                                          \
  }())
#endif

namespace st::ops::bloom_index {

[[maybe_unused]] constexpr uint64_t MAX_K = 64;
[[maybe_unused]] constexpr uint64_t MAX_B = 65536;
[[maybe_unused]] constexpr int64_t C_BITS_IN_UINT64 = 64;
[[maybe_unused]] constexpr int64_t C_BLOOM_V2_COL_BUNDLE_SIZE = 32;
[[maybe_unused]] constexpr int64_t C_BITS_IN_BLOOM_V2_COL_BUNDLE =
    C_BITS_IN_UINT64 * C_BLOOM_V2_COL_BUNDLE_SIZE;
[[maybe_unused]] constexpr int64_t MAX_K_V2 = 10;
[[maybe_unused]] constexpr int64_t MAX_B_V2 = INT_MAX;

enum Operator : uint8_t {
  AND = 1,
  OR = 2,
  NOT = 3,
  TERM = 4,
  // EMPTY query means full match.
  EMPTY = 5,
};

template <bool is_bloom_index_v2>
struct QueryPlanOneBitsPType;

template <>
struct QueryPlanOneBitsPType<false> {
  using type = int16_t;
};

template <>
struct QueryPlanOneBitsPType<true> {
  using type = uint64_t;
};

template <bool is_bloom_index_v2>
struct QueryPlan {
 public:
  using oneBitsPositionType =
      typename QueryPlanOneBitsPType<is_bloom_index_v2>::type;
  QueryPlan() {
    offsets.push_back(0);
  }

  // operator types
  std::vector<Operator> operators;
  // offsets of the parameters for each operator.
  std::vector<int32_t> offsets;
  // parameters for the operators.
  //  1. operator != TERM: the index of operator output from operators array.
  //  2. operator == term: term signature index from term_signature array.
  std::vector<int32_t> parameters;
  // '1' bit positions. each term will contain k positions.
  // for bloom index v1, oneBitsPosition is a vector of int16_t which contains
  // k positions for the feature.
  // for bloom index v2, oneBitsPosition is a vector of int64_t which contains
  // k * k_multiplier hashes for the feature.
  std::vector<oneBitsPositionType> oneBitsPosition;
};

template <bool is_bloom_index_v2>
int32_t get_max_stack_size(
    const std::vector<QueryPlan<is_bloom_index_v2>>& query_plans);

template <bool is_bloom_index_v2>
uint32_t encode_query_plans(
    const std::vector<QueryPlan<is_bloom_index_v2>>& query_plans,
    std::vector<char>& query_plan_data_vec /* output */,
    std::vector<size_t>& query_plan_offsets_vec /* output */);

template <bool is_bloom_index_v2>
void decode_query_plan(
    const at::Tensor& query_plan_data_cuda,
    const at::Tensor& query_plan_offsets,
    size_t offset,
    at::Tensor& operators /* output */,
    at::Tensor& offsets /* output */,
    at::Tensor& parameters /* output */,
    at::Tensor& oneBitsPosition /* output */);

size_t align_size(size_t size);
} // namespace st::ops::bloom_index
