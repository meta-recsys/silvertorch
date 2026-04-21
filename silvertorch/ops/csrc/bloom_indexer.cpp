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

#include <ATen/ceil_div.h> // @manual=//caffe2:ATen-cpu
#include <torch/library.h>

#include "bloom_index_util.cuh"

namespace st::ops::bloom_indexer {

using namespace st::ops::bloom_index;

using at::Tensor;

std::tuple<Tensor, Tensor> bloom_index_build_cpu(
    const Tensor& feature_ids,
    const Tensor& feature_offsets,
    const Tensor& feature_values,
    double b_multiplier,
    int64_t k,
    bool /*fast_build*/) {
  TORCH_CHECK(feature_ids.scalar_type() == at::kInt);
  TORCH_CHECK(feature_offsets.scalar_type() == at::kLong);
  TORCH_CHECK(feature_values.scalar_type() == at::kLong);
  TORCH_CHECK(b_multiplier > 1.0, "b_multiplier must be greater than 1.0");
  int32_t* feature_ids_ptr = feature_ids.data_ptr<int32_t>();
  int64_t* feature_offsets_ptr = feature_offsets.data_ptr<int64_t>();
  int64_t* feature_values_ptr = feature_values.data_ptr<int64_t>();

  int64_t feature_count = feature_ids.numel();
  int64_t document_count = (feature_offsets.numel() - 1) / feature_count;
  int64_t column_bundle_count =
      at::ceil_div(document_count, C_BITS_IN_BLOOM_V2_COL_BUNDLE);

  Tensor bundled_column_b_offsets =
      at::empty({column_bundle_count + 1}, at::kLong);
  int64_t* bundled_column_b_offsets_ptr =
      bundled_column_b_offsets.data_ptr<int64_t>();
  bundled_column_b_offsets_ptr[0] = 0;
  for (int i = 0; i < column_bundle_count; ++i) {
    int64_t max_feature_count_in_column = 0;
    for (int j = 0; j < C_BITS_IN_BLOOM_V2_COL_BUNDLE; ++j) {
      int64_t doc_id = i * C_BITS_IN_BLOOM_V2_COL_BUNDLE + j;
      if (doc_id < document_count) {
        int64_t doc_feature_count = 0;
        for (int f = 0; f < feature_count; ++f) {
          int64_t feature_idx = doc_id * feature_count + f;
          doc_feature_count += feature_offsets_ptr[feature_idx + 1] -
              feature_offsets_ptr[feature_idx];
        }
        max_feature_count_in_column =
            std::max(max_feature_count_in_column, doc_feature_count);
      }
    }
    bundled_column_b_offsets_ptr[i + 1] = bundled_column_b_offsets_ptr[i] +
        static_cast<int64_t>(static_cast<double>(
                                 max_feature_count_in_column * k) *
                             b_multiplier);
  }

  std::array<typename BIndexType<true>::type, MAX_K> used_b_p = {0};
  Tensor bloom_index = at::zeros(
      {bundled_column_b_offsets_ptr[column_bundle_count] *
       C_BLOOM_V2_COL_BUNDLE_SIZE},
      at::kLong);
  int64_t* bloom_index_ptr = bloom_index.data_ptr<int64_t>();
  for (int doc_id = 0; doc_id < document_count; ++doc_id) {
    int64_t column_bundle_id = doc_id / C_BITS_IN_BLOOM_V2_COL_BUNDLE;
    int64_t doc_offset_in_bundle = doc_id % C_BITS_IN_BLOOM_V2_COL_BUNDLE;
    int64_t column_offset_in_bundle = doc_offset_in_bundle / C_BITS_IN_UINT64;
    int64_t doc_offset_in_column = doc_offset_in_bundle % C_BITS_IN_UINT64;
    TORCH_CHECK(doc_offset_in_column == doc_id % C_BITS_IN_UINT64);
    int64_t* bloom_index_bundle_ptr = bloom_index_ptr +
        bundled_column_b_offsets_ptr[column_bundle_id] *
            C_BLOOM_V2_COL_BUNDLE_SIZE;
    int64_t column_b = bundled_column_b_offsets_ptr[column_bundle_id + 1] -
        bundled_column_b_offsets_ptr[column_bundle_id];
    for (int i = 0; i < feature_count; ++i) {
      int64_t value_start = feature_offsets_ptr[doc_id * feature_count + i];
      int64_t value_end = feature_offsets_ptr[doc_id * feature_count + i + 1];
      int64_t feature_id = static_cast<int64_t>(feature_ids_ptr[i]);
      while (value_start < value_end) {
        for (int j = 0; j < k; ++j) {
          used_b_p[j] = 0;
        }
        assign_one_bits_position<typename BIndexType<true>::type>(
            feature_id,
            feature_values_ptr[value_start++],
            column_b,
            k,
            used_b_p.data());
        for (int j = 0; j < k; ++j) {
          bloom_index_bundle_ptr
              [used_b_p[j] * C_BLOOM_V2_COL_BUNDLE_SIZE +
               column_offset_in_bundle] |= (1LL << (63 - doc_offset_in_column));
        }
      }
    }
  }
  return std::make_tuple(
      std::move(bloom_index), std::move(bundled_column_b_offsets));
}

TORCH_LIBRARY_FRAGMENT(st, m) {
  m.def(
      "bloom_index_build("
      "Tensor feature_ids, "
      "Tensor feature_offsets, "
      "Tensor feature_values, "
      "float b_multiplier, "
      "int k, "
      "bool fast_build=False"
      ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(st, CPU, m) {
  m.impl(
      "bloom_index_build",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(bloom_index_build_cpu)));
}

} // namespace st::ops::bloom_indexer
