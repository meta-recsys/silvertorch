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

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cstring>

#include "bloom_index_util.h"

namespace st::ops::bloom_index {

using at::Tensor;

int32_t get_max_stack_size_impl(
    const std::vector<Operator>& operators,
    const std::vector<int32_t>& offsets) {
  int32_t max_stack_size = 0;
  int32_t stack_size = 0;
  for (size_t j = 0; j < operators.size(); ++j) {
    switch (operators[j]) {
      case Operator::TERM:
      case Operator::EMPTY:
        stack_size++;
        break;

      case Operator::AND:
      case Operator::OR:
      case Operator::NOT:
        // NOLINTNEXTLINE: Array accesses with `[]` will segfault
        int32_t children = offsets[j + 1] - offsets[j];
        stack_size -= children;
        stack_size += 1;
        break;
    }

    if (stack_size > max_stack_size) {
      max_stack_size = stack_size;
    }

    TORCH_CHECK(
        stack_size >= 0,
        "stack size can't never go negative, it means some coding problem in decode_query.");
  }
  return max_stack_size;
}

template <bool is_bloom_index_v2>
int32_t get_max_stack_size(
    const std::vector<QueryPlan<is_bloom_index_v2>>& query_plans) {
  int32_t max_stack_size = 0;
  for (const auto& query_plan : query_plans) {
    int32_t s =
        get_max_stack_size_impl(query_plan.operators, query_plan.offsets);
    if (s > max_stack_size) {
      max_stack_size = s;
    }
  }
  return max_stack_size;
}

size_t align_size(size_t size) {
  static constexpr size_t c_MEMORY_ALIGN = 8;
  if (size % c_MEMORY_ALIGN != 0) {
    size = (size / c_MEMORY_ALIGN + 1) * c_MEMORY_ALIGN;
  }
  return size;
}

template <typename T>
void _add_size_and_data(
    const std::vector<T>& vec,
    std::vector<char>& query_plan_data_vec /* output */,
    std::vector<size_t>& query_plan_data_offsets_vec /* output */) {
  using ValueType = std::conditional_t<std::is_same_v<T, Operator>, uint8_t, T>;
  auto offset = query_plan_data_vec.size();
  query_plan_data_vec.resize(
      offset + align_size(vec.size() * sizeof(ValueType)));
  // @lint-ignore CLANGSECURITY facebook-security-vulnerable-memcpy
  if (!vec.empty()) {
    std::memcpy(
        query_plan_data_vec.data() + offset,
        vec.data(),
        vec.size() * sizeof(ValueType));
  }
  query_plan_data_offsets_vec.push_back(offset);
  query_plan_data_offsets_vec.push_back(
      offset + vec.size() * sizeof(ValueType));
}

template <bool is_bloom_index_v2>
uint32_t encode_query_plans(
    const std::vector<QueryPlan<is_bloom_index_v2>>& query_plans,
    std::vector<char>& query_plan_data_vec /* output */,
    std::vector<size_t>& query_plan_data_offsets_vec /* output */) {
  TORCH_CHECK(query_plan_data_vec.empty());
  TORCH_CHECK(query_plan_data_offsets_vec.empty());
  static_assert(
      sizeof(QueryPlan<is_bloom_index_v2>) == 96,
      "Need to update C_NUM_OFFSETS_PER_PLAN, QueryPlanCuda and below code if you add fields to QueryPlan");
  uint32_t max_stack_size = get_max_stack_size<is_bloom_index_v2>(query_plans);
  for (const auto& plan : query_plans) {
    _add_size_and_data(
        plan.operators, query_plan_data_vec, query_plan_data_offsets_vec);
    _add_size_and_data(
        plan.offsets, query_plan_data_vec, query_plan_data_offsets_vec);
    _add_size_and_data(
        plan.parameters, query_plan_data_vec, query_plan_data_offsets_vec);
    _add_size_and_data(
        plan.oneBitsPosition, query_plan_data_vec, query_plan_data_offsets_vec);
  }
  return max_stack_size;
}

template <bool is_bloom_index_v2>
void decode_query_plan(
    const Tensor& query_plan_data_cuda,
    const Tensor& query_plan_offsets,
    size_t offset,
    Tensor& operators /* output */,
    Tensor& offsets /* output */,
    Tensor& parameters /* output */,
    Tensor& oneBitsPosition /* output */) {
  static_assert(
      sizeof(QueryPlan<is_bloom_index_v2>) == 96,
      "Need to update C_NUM_OFFSETS_PER_PLAN, QueryPlanCuda and below code if you add fields to QueryPlan");

  auto query_plan_offsets_acc = query_plan_offsets.accessor<int64_t, 1>();
  // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  auto operators_start = query_plan_offsets_acc[offset];
  auto operators_end = query_plan_offsets_acc[offset + 1];
  auto offsets_start = query_plan_offsets_acc[offset + 2];
  auto offsets_end = query_plan_offsets_acc[offset + 3];
  auto parameters_start = query_plan_offsets_acc[offset + 4];
  auto parameters_end = query_plan_offsets_acc[offset + 5];
  auto oneBitsPosition_start = query_plan_offsets_acc[offset + 6];
  auto oneBitsPosition_end = query_plan_offsets_acc[offset + 7];
  // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)

  operators = at::from_blob(
      query_plan_data_cuda.data_ptr<int8_t>() + operators_start,
      {static_cast<int64_t>(operators_end - operators_start)}, // size
      c10::TensorOptions().dtype(c10::CppTypeToScalarType<uint8_t>::value));

  offsets = at::from_blob(
      query_plan_data_cuda.data_ptr<int8_t>() + offsets_start,
      {static_cast<int64_t>(
          (offsets_end - offsets_start) * sizeof(int8_t) /
          sizeof(int32_t))}, // size
      c10::TensorOptions().dtype(c10::CppTypeToScalarType<int32_t>::value));

  parameters = at::from_blob(
      query_plan_data_cuda.data_ptr<int8_t>() + parameters_start,
      {static_cast<int64_t>(
          (parameters_end - parameters_start) * sizeof(int8_t) /
          sizeof(int32_t))}, // size
      c10::TensorOptions().dtype(c10::CppTypeToScalarType<int32_t>::value));

  oneBitsPosition = at::from_blob(
      query_plan_data_cuda.data_ptr<int8_t>() + oneBitsPosition_start,
      {static_cast<int64_t>(
          (oneBitsPosition_end - oneBitsPosition_start) * sizeof(int8_t) /
          sizeof(typename QueryPlanOneBitsPType<
                 is_bloom_index_v2>::type))}, // size
      c10::TensorOptions().dtype(
          c10::CppTypeToScalarType<
              typename QueryPlanOneBitsPType<is_bloom_index_v2>::type>::value));
}

// Template instantiations
template int32_t get_max_stack_size<false>(
    const std::vector<QueryPlan<false>>&);

template int32_t get_max_stack_size<true>(const std::vector<QueryPlan<true>>&);

template uint32_t encode_query_plans<false>(
    const std::vector<QueryPlan<false>>&,
    std::vector<char>&,
    std::vector<size_t>&);

template uint32_t encode_query_plans<true>(
    const std::vector<QueryPlan<true>>&,
    std::vector<char>&,
    std::vector<size_t>&);

template void decode_query_plan<false>(
    const at::Tensor&,
    const at::Tensor&,
    size_t,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&);

template void decode_query_plan<true>(
    const at::Tensor&,
    const at::Tensor&,
    size_t,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&);

} // namespace st::ops::bloom_index

// OSS-only: provide a Python module entry point so `import silvertorch._C`
// works after `pip install`. Internal BUCK builds load the .so via
// torch.ops.load_library which uses dlopen directly (no PyInit needed) and
// do not define TORCH_EXTENSION_NAME, so this block is excluded internally.
// All ops register themselves via TORCH_LIBRARY_FRAGMENT static initializers
// elsewhere in the extension; this module body is intentionally empty.
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h> // @manual  OSS-only entry point; cpp_extension provides this
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  (void)m;
}
#endif
