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

#include <vector_types.h> // For int4 definition.
#include <tuple>

#include <ATen/ATen.h>

namespace st::ops::fused_kmean_ann {

std::tuple<at::Tensor, at::Tensor, at::Tensor> round_cluster_to_warp(
    const at::Tensor& selected_cluster_lengths);

union WarpPayload {
  struct {
    int32_t doc_start_index;
    int32_t write_index;
    int32_t cluster_ids_row;
    uint32_t filtering_mask;
  };
  int4 payload;
};

union RemainingPayload {
  struct {
    int32_t doc_index;
    int32_t write_index;
    int32_t cluster_ids_row;
    bool filtering_mask;
  };
  int4 payload;
};

std::tuple<at::Tensor, int64_t> generate_warp_payload(
    const at::Tensor& cluster_warp_size,
    const at::Tensor& cluster_offsets,
    const at::Tensor& selected_cluster_ids,
    const at::Tensor& cluster_warp_rounded_length_cumsum,
    const uint64_t* filtering_bitmask_ptr,
    int64_t filtering_bitmask_column,
    const uint64_t* filtering_bitmask_index_ptr,
    int64_t max_tensor_size_per_row);

std::tuple<at::Tensor, int64_t> generate_remaining_payload(
    const at::Tensor& selected_cluster_lengths,
    const at::Tensor& cluster_offsets,
    const at::Tensor& selected_cluster_ids,
    const at::Tensor& cluster_warp_rounded_length_cumsum,
    const at::Tensor& cluster_remaining_length_cumsum,
    const uint64_t* filtering_bitmask_ptr,
    int64_t filtering_bitmask_column,
    const uint64_t* filtering_bitmask_index_ptr,
    int64_t max_tensor_size_per_row);

std::tuple<at::Tensor, at::Tensor> fused_kmean_ann_cuda(
    const at::Tensor& cluster_offsets,
    const at::Tensor& cluster_ids,
    const at::Tensor& cluster_length,
    const at::Tensor& embeddings,
    const at::Tensor& queries,
    int64_t max_tensor_size_per_row,
    const std::optional<at::Tensor>& filtering_bit_mask,
    int64_t invalid_index_value,
    int64_t divisor_for_int8,
    const std::optional<at::Tensor>& filtering_bit_index,
    const std::optional<at::Tensor>& per_embedding_scale);

} // namespace st::ops::fused_kmean_ann
