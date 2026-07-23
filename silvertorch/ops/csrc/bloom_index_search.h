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

namespace st::ops::bloom_search {

at::Tensor bloom_index_search_batch_cpu(
    const at::Tensor& bloom_index,
    const at::Tensor& bloom_bundle_b_offsets,
    const at::Tensor& bloom_query_plans_data,
    const at::Tensor& bloom_query_plans_offsets,
    int64_t k,
    int64_t hash_k,
    bool return_bool_mask);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
bloom_index_search_batch_return_partial_response_cpu(
    const at::Tensor& bloom_index,
    const at::Tensor& bloom_bundle_b_offsets,
    const at::Tensor& bloom_query_plans_data,
    const at::Tensor& bloom_query_plans_offsets,
    const at::Tensor& selected_cluster_offsets,
    const at::Tensor& selected_cluster_lengths,
    int64_t k,
    int64_t hash_k,
    const std::optional<at::Tensor>& query_plan_index);

// Runs bloom_index_search_batch_return_partial_response over a list of bloom
// indices (chunks), returning a per-chunk list of results. The query plans are
// shared across all chunks.
std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
bloom_index_search_batch_return_partial_response_multiple_cpu(
    const std::vector<at::Tensor>& list_bloom_index,
    const std::vector<at::Tensor>& list_bloom_bundle_b_offsets,
    const at::Tensor& bloom_query_plans_data,
    const at::Tensor& bloom_query_plans_offsets,
    const std::vector<at::Tensor>& list_selected_cluster_offsets,
    const std::vector<at::Tensor>& list_selected_cluster_lengths,
    int64_t k,
    int64_t hash_k,
    const std::optional<at::Tensor>& query_plan_index,
    bool fuse);

} // namespace st::ops::bloom_search
