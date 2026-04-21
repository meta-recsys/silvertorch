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

#include <torch/library.h>
#include <torch/torch.h>

#include "bloom_index_search.h"
#include "bloom_index_util.h"

namespace st::ops::bloom_search {

using at::Tensor;
using namespace st::ops::bloom_index;
namespace {

inline uint64_t signature_match(
    const int64_t* bloom_index,
    const int64_t column_count,
    const int64_t* bundle_b_offsets, // only used for bloom_index_v2
    int64_t column,
    int64_t k,
    int64_t hash_k, // only used for bloom_index_v2
    const typename QueryPlanOneBitsPType<false>::type*
        onebits_positions_v1, // only used for bloom_index_v1, int16_t.
    const typename QueryPlanOneBitsPType<true>::type*
        onebits_positions_v2 // only used for bloom_index_v2, int64_t.
) {
  if (onebits_positions_v2 != nullptr) {
    uint64_t result = ~(static_cast<uint64_t>(0));
    int64_t bundle = column / C_BLOOM_V2_COL_BUNDLE_SIZE;
    int64_t bundle_b_start = bundle_b_offsets[bundle];
    int64_t b = bundle_b_offsets[bundle + 1] - bundle_b_start;
    int64_t column_in_bundle = column % C_BLOOM_V2_COL_BUNDLE_SIZE;
    std::array<int16_t, MAX_K_V2> used_b_p = {0};
    int32_t p = 0;
    int32_t resolved = 0;
    while (p < hash_k && resolved < k) {
      int16_t b_p = static_cast<int16_t>(onebits_positions_v2[p++] % b);
      for (int32_t i = 0; i < resolved; ++i) {
        if (used_b_p[i] == b_p) {
          goto duplicate_continue;
        }
      }
      used_b_p[resolved++] = b_p;
      result &= bloom_index
          [(bundle_b_start + b_p) * C_BLOOM_V2_COL_BUNDLE_SIZE +
           column_in_bundle];
    duplicate_continue:
      continue;
    }
    return result;
  } else {
    uint64_t result = ~(static_cast<uint64_t>(0));
    for (int64_t i = 0; i < k; ++i) {
      result &= bloom_index[onebits_positions_v1[i] * column_count + column];
    }
    return result;
  }
}

uint64_t run_query_plan(
    const int64_t* bloom_index,
    const int64_t column_count,
    const int64_t* bundle_b_offsets, // only used for bloom_index_v2
    const int64_t column,
    const int64_t k,
    const int64_t hash_k, // only used for bloom_index_v2
    const Tensor& bloom_query_plans_data,
    const Tensor& bloom_query_plans_offsets,
    size_t query_plan_index,
    bool is_bloom_index_v2) {
  const int8_t* query_plan_data_ptr = bloom_query_plans_data.data_ptr<int8_t>();
  const int64_t* query_plan_data_offsets_ptr =
      bloom_query_plans_offsets.data_ptr<int64_t>();
  size_t start_offset = query_plan_index * 8;
  int64_t operators_start = query_plan_data_offsets_ptr[start_offset];
  int64_t operators_end = query_plan_data_offsets_ptr[start_offset + 1];
  int64_t offsets_start = query_plan_data_offsets_ptr[start_offset + 2];
  int64_t onebits_pos_start = query_plan_data_offsets_ptr[start_offset + 6];
  std::stack<uint64_t> process_stack;
  int32_t one_bits_position_offset = 0;
  const uint8_t* ops_ptr =
      (reinterpret_cast<const uint8_t*>(query_plan_data_ptr + operators_start));
  const int32_t* offsets_ptr =
      (reinterpret_cast<const int32_t*>(query_plan_data_ptr + offsets_start));

  const typename QueryPlanOneBitsPType<false>::type* onebits_positions_v1 =
      nullptr;
  const typename QueryPlanOneBitsPType<true>::type* onebits_positions_v2 =
      nullptr;
  if (is_bloom_index_v2) {
    onebits_positions_v2 =
        (reinterpret_cast<const typename QueryPlanOneBitsPType<true>::type*>(
            query_plan_data_ptr + onebits_pos_start));
  } else {
    onebits_positions_v1 =
        (reinterpret_cast<const typename QueryPlanOneBitsPType<false>::type*>(
            query_plan_data_ptr + onebits_pos_start));
  }
  for (size_t i = 0; i < operators_end - operators_start; ++i) {
    Operator op = static_cast<Operator>(ops_ptr[i]);
    switch (op) {
      case Operator::TERM: {
        process_stack.push(signature_match(
            bloom_index,
            column_count,
            bundle_b_offsets,
            column,
            k,
            hash_k,
            is_bloom_index_v2
                ? nullptr
                : (onebits_positions_v1 + (one_bits_position_offset * k)),
            is_bloom_index_v2
                ? (onebits_positions_v2 + (one_bits_position_offset * hash_k))
                : nullptr));
        ++one_bits_position_offset;
        break;
      }
      case Operator::AND: {
        uint64_t result = ~(static_cast<uint64_t>(0));
        int32_t children = offsets_ptr[i + 1] - offsets_ptr[i];
        for (int32_t j = 0; j < children; ++j) {
          result &= process_stack.top();
          process_stack.pop();
        }
        process_stack.push(result);
        break;
      }
      case Operator::OR: {
        uint64_t result = 0;
        int32_t children = offsets_ptr[i + 1] - offsets_ptr[i];
        for (int32_t j = 0; j < children; ++j) {
          result |= process_stack.top();
          process_stack.pop();
        }
        process_stack.push(result);
        break;
      }
      case Operator::NOT: {
        TORCH_CHECK(
            offsets_ptr[i + 1] - offsets_ptr[i] == 1,
            "NOT query should have 1 parameter, but got ",
            offsets_ptr[i + 1] - offsets_ptr[i]);
        uint64_t result = process_stack.top();
        process_stack.pop();
        process_stack.push(~result);
        break;
      }
      case Operator::EMPTY: {
        process_stack.push(UINT64_MAX);
        break;
      }
      default: {
        TORCH_CHECK(false, "unknown operator: ", op);
      }
    }
  }
  return process_stack.top();
}

Tensor bloom_index_search_common_cpu(
    const Tensor& bloom_index,
    const Tensor* bloom_bundle_b_offsets, // only used for bloom_index_v2
    const Tensor& bloom_query_plans_data,
    const Tensor& bloom_query_plans_offsets,
    int64_t k,
    int64_t hash_k, // only used for bloom_index_v2
    bool return_bool_mask,
    bool is_bloom_index_v2) {
  TORCH_CHECK(bloom_index.is_contiguous());
  TORCH_CHECK(bloom_index.scalar_type() == at::ScalarType::Long);
  int64_t column_count = -1;
  const int64_t* bundle_b_offsets = nullptr;
  static constexpr uint64_t C_MASK = 1ULL << 63;
  if (is_bloom_index_v2) {
    TORCH_CHECK(bloom_index.dim() == 1);
    TORCH_CHECK(bloom_bundle_b_offsets != nullptr);
    TORCH_CHECK(bloom_bundle_b_offsets->is_contiguous());
    bundle_b_offsets = bloom_bundle_b_offsets->data_ptr<int64_t>();
    column_count =
        (bloom_bundle_b_offsets->numel() - 1) * C_BLOOM_V2_COL_BUNDLE_SIZE;
  } else {
    TORCH_CHECK(bloom_index.dim() == 2);
    column_count = bloom_index.size(1);
  }
  int64_t query_plan_count = (bloom_query_plans_offsets.numel()) / 8;
  const int64_t* bloom_index_ptr = bloom_index.data_ptr<int64_t>();

  Tensor document_mask;
  uint64_t* document_mask_bit_ptr = nullptr;
  bool* document_mask_bool_ptr = nullptr;
  if (return_bool_mask) {
    document_mask = at::empty(
        {query_plan_count,
         static_cast<int64_t>(column_count * C_BITS_IN_UINT64)},
        c10::TensorOptions().dtype(at::kBool));
    document_mask_bool_ptr = document_mask.data_ptr<bool>();
  } else {
    document_mask = at::empty(
        {query_plan_count, column_count},
        c10::TensorOptions().dtype(at::kLong));
    document_mask_bit_ptr =
        reinterpret_cast<uint64_t*>(document_mask.data_ptr<int64_t>());
  }
  for (int64_t q = 0; q < query_plan_count; ++q) {
    for (int64_t c = 0; c < column_count; ++c) {
      // NOLINTNEXTLINE: suppress error of using nullable bundle_b_offsets.
      uint64_t column_result = run_query_plan(
          bloom_index_ptr,
          column_count,
          bundle_b_offsets, // only used for bloom_index_v2
          c,
          k,
          hash_k, // only used for bloom_index_v2
          bloom_query_plans_data,
          bloom_query_plans_offsets,
          q,
          is_bloom_index_v2);
      if (return_bool_mask) {
        for (int64_t i = 0; i < C_BITS_IN_UINT64; ++i) {
          document_mask_bool_ptr[q * column_count * 64 + c * 64 + i] =
              ((column_result & C_MASK) != 0);
          column_result <<= 1;
        }
      } else {
        document_mask_bit_ptr[q * column_count + c] = column_result;
      }
    }
  }
  return document_mask;
}

} // namespace

Tensor bloom_index_search_batch_cpu(
    const Tensor& bloom_index,
    const Tensor& bloom_bundle_b_offsets,
    const Tensor& bloom_query_plans_data,
    const Tensor& bloom_query_plans_offsets,
    int64_t k,
    int64_t hash_k,
    bool return_bool_mask) {
  return bloom_index_search_common_cpu(
      bloom_index,
      &bloom_bundle_b_offsets,
      bloom_query_plans_data,
      bloom_query_plans_offsets,
      k,
      hash_k,
      return_bool_mask,
      /*is_bloom_index_v2*/ true);
}

static std::tuple<Tensor, Tensor, Tensor> generate_column_info_for_clusters_cpu(
    const Tensor& selected_cluster_offsets,
    const Tensor& selected_cluster_lengths) {
  TORCH_CHECK(selected_cluster_offsets.is_cpu());
  TORCH_CHECK(selected_cluster_lengths.is_cpu());
  TORCH_CHECK(
      selected_cluster_offsets.sizes() == selected_cluster_lengths.sizes());
  TORCH_CHECK(
      selected_cluster_offsets.dim() == 1 ||
          selected_cluster_offsets.dim() == 2,
      "selected_cluster_offsets must be 1D or 2D, got ",
      selected_cluster_offsets.dim(),
      "D");

  const auto n = selected_cluster_lengths.numel();
  auto column_counts =
      at::empty({n}, selected_cluster_lengths.options().dtype(at::kInt));
  auto start_column_ids =
      at::empty({n}, selected_cluster_lengths.options().dtype(at::kInt));
  auto first_item_offsets_in_column =
      at::empty({n}, selected_cluster_lengths.options().dtype(at::kChar));

  const auto* cluster_offsets_ptr =
      selected_cluster_offsets.data_ptr<int64_t>();
  const auto* cluster_lengths_ptr =
      selected_cluster_lengths.data_ptr<int64_t>();
  auto* column_counts_ptr = column_counts.data_ptr<int32_t>();
  auto* start_column_ids_ptr = start_column_ids.data_ptr<int32_t>();
  auto* first_item_offsets_ptr =
      first_item_offsets_in_column.data_ptr<int8_t>();

  for (int64_t i = 0; i < n; ++i) {
    int64_t local_cluster_len = cluster_lengths_ptr[i];
    int64_t local_cluster_start = cluster_offsets_ptr[i];
    int32_t start_col_id =
        static_cast<int32_t>(local_cluster_start / C_BITS_IN_UINT64);
    int8_t offset_in_col =
        static_cast<int8_t>(local_cluster_start % C_BITS_IN_UINT64);
    start_column_ids_ptr[i] = start_col_id;
    first_item_offsets_ptr[i] = offset_in_col;
    column_counts_ptr[i] = local_cluster_len > 0
        ? static_cast<int32_t>(
              (local_cluster_len + offset_in_col + C_BITS_IN_UINT64 - 1) /
              C_BITS_IN_UINT64)
        : 0;
  }

  return std::make_tuple(
      std::move(column_counts),
      std::move(start_column_ids),
      std::move(first_item_offsets_in_column));
}

TORCH_LIBRARY_FRAGMENT(st, m) {
  m.def(
      "generate_column_info_for_clusters("
      "Tensor selected_cluster_offsets, "
      "Tensor selected_cluster_lengths"
      ") -> (Tensor, Tensor, Tensor)");

  m.def(
      "bloom_index_search_batch("
      "Tensor bloom_index, "
      "Tensor bloom_bundle_b_offsets, "
      "Tensor bloom_query_plans_data, "
      "Tensor bloom_query_plans_offsets, "
      "int k, "
      "int hash_k, "
      "bool return_bool_mask=True)"
      "-> Tensor");
}

TORCH_LIBRARY_IMPL(st, CPU, m) {
  m.impl(
      "bloom_index_search_batch",
      torch::dispatch(
          c10::DispatchKey::CPU, TORCH_FN(bloom_index_search_batch_cpu)));
}

TORCH_LIBRARY_IMPL(st, CPU, m) {
  m.impl(
      "generate_column_info_for_clusters",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(generate_column_info_for_clusters_cpu)));
}
} // namespace st::ops::bloom_search
