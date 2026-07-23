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

#include <ATen/Dispatch.h> // @manual
#include <torch/library.h> // @manual
#include <torch/torch.h> // @manual

#include <algorithm>
#include <utility>

namespace st::ops::fresh_index_post_processing {

using at::PackedTensorAccessor64;
using at::Tensor;

namespace {

template <typename DATA_T>
void select_store(
    const DATA_T* source_data_ptr,
    DATA_T* result_data_ptr,
    size_t data_dim,
    size_t row,
    size_t col_in_dst,
    size_t col_in_src,
    size_t n_items_in_one_row_dst) {
  const auto src_idx = data_dim * col_in_src;
  const auto dst_idx = data_dim * (row * n_items_in_one_row_dst + col_in_dst);
  for (size_t i = 0; i < data_dim; i++) {
    result_data_ptr[dst_idx + i] = source_data_ptr[src_idx + i];
  }
}

Tensor gather_data_from_main_and_fresh_index_cpu_impl(
    const Tensor& top_indices_indices,
    const Tensor& main_indices,
    const Tensor& fresh_indices,
    const Tensor& main_data,
    const Tensor& fresh_data,
    int64_t fresh_vs_full_div_index,
    int64_t /* kernel_version */) {
  const auto data_scalar_type = main_data.scalar_type();

  const auto result_num_rows = top_indices_indices.size(0);
  const auto result_num_cols = top_indices_indices.size(1);
  TORCH_CHECK(
      main_data.dim() > 0 && main_data.dim() <= 2,
      "The main data dim must be in [1, 2], main_data.dim()=",
      main_data.dim());
  TORCH_CHECK(
      fresh_data.dim() > 0 && fresh_data.dim() <= 2,
      "The fresh data dim must be in [1, 2], fresh_data.dim()=",
      fresh_data.dim());
  TORCH_CHECK(
      main_data.dim() == fresh_data.dim(),
      "The two data dims must be the same, main_data.dim()=",
      main_data.dim(),
      ", fresh_data.dim()=",
      fresh_data.dim());

  const auto data_dim = main_data.dim() == 2 ? main_data.size(1) : 1;

  auto result_data = at::full(
      {result_num_rows * result_num_cols * data_dim},
      0,
      main_data.options().dtype(data_scalar_type));
  if (main_data.dim() == 2) {
    result_data =
        result_data.reshape({result_num_rows * result_num_cols, data_dim});
  } else {
    result_data = result_data.reshape({result_num_rows * result_num_cols});
  }

  AT_DISPATCH_INDEX_TYPES(
      main_indices.scalar_type(),
      "gather_data_from_main_and_fresh_index_cpu_impl_index",
      [&]() {
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            main_data.scalar_type(),
            "gather_data_from_main_and_fresh_index_cpu_impl",
            [&]() {
              const PackedTensorAccessor64<int64_t, 2>
                  top_indices_indices_accessor =
                      top_indices_indices.packed_accessor64<int64_t, 2>();
              const PackedTensorAccessor64<index_t, 2> main_indices_accessor =
                  main_indices.packed_accessor64<index_t, 2>();
              const PackedTensorAccessor64<index_t, 2> fresh_indices_accessor =
                  fresh_indices.packed_accessor64<index_t, 2>();
              for (int global_row = 0; global_row < top_indices_indices.size(0);
                   global_row++) {
                for (int global_col = 0;
                     global_col < top_indices_indices.size(1);
                     global_col++) {
                  const auto col_in_dst = global_col;
                  const auto top_indices_index =
                      top_indices_indices_accessor[global_row][col_in_dst];
                  if (top_indices_index < fresh_vs_full_div_index) {
                    auto col_in_src =
                        fresh_indices_accessor[global_row][top_indices_index];
                    if (col_in_src < 0 || col_in_src >= fresh_data.size(0)) {
                      // Prevent out of bound access in case of padded offsets.
                      col_in_src = 0;
                    }
                    select_store<scalar_t>(
                        fresh_data.data_ptr<scalar_t>(),
                        result_data.mutable_data_ptr<scalar_t>(),
                        data_dim,
                        global_row,
                        col_in_dst,
                        col_in_src,
                        top_indices_indices.size(1));
                  } else {
                    auto col_in_src =
                        main_indices_accessor[global_row]
                                             [top_indices_index -
                                              fresh_vs_full_div_index];
                    if (col_in_src < 0 || col_in_src >= main_data.size(0)) {
                      // Prevent out of bound access in case of padded offsets.
                      col_in_src = 0;
                    }
                    select_store<scalar_t>(
                        main_data.data_ptr<scalar_t>(),
                        result_data.mutable_data_ptr<scalar_t>(),
                        data_dim,
                        global_row,
                        col_in_dst,
                        col_in_src,
                        top_indices_indices.size(1));
                  }
                }
              }
            });
      });

  return result_data;
}

c10::Dict<std::string, Tensor> gather_data_from_main_and_fresh_index_cpu(
    const Tensor& top_indices_indices,
    const Tensor& main_indices,
    const Tensor& fresh_indices,
    const std::vector<std::string>& keys,
    const std::vector<Tensor>& list_main_data,
    const std::vector<Tensor>& list_fresh_data,
    int64_t fresh_vs_full_div_index,
    int64_t kernel_version) {
  TORCH_CHECK(
      keys.size() == list_main_data.size(),
      "keys and list_main_data should have the same size. ",
      "keys.size()=",
      keys.size(),
      ", list_main_data.size()=",
      list_main_data.size());
  TORCH_CHECK(
      keys.size() == list_fresh_data.size(),
      "keys and list_fresh_data should have the same size. ",
      "keys.size()=",
      keys.size(),
      ", list_fresh_data.size()=",
      list_fresh_data.size());
  for (size_t i = 0; i < keys.size(); i++) {
    const auto& key = keys[i];
    const auto& main_data = list_main_data[i];
    const auto& fresh_data = list_fresh_data[i];
    TORCH_CHECK(
        main_data.dim() == fresh_data.dim(),
        "main_data and fresh_data should have the same dim. ",
        "key=",
        key,
        ", main_data.dim()=",
        main_data.dim(),
        ", fresh_data.dim()=",
        fresh_data.dim());
    if (main_data.dim() > 1) {
      TORCH_CHECK(
          main_data.size(-1) == fresh_data.size(-1),
          "main_data and fresh_data should have the same number of element in the last dim. "
          "key=",
          key,
          ", main_data.size(-1)=",
          main_data.size(-1),
          ", fresh_data.size(-1)=",
          fresh_data.size(-1));
    }
    TORCH_CHECK(
        main_data.scalar_type() == fresh_data.scalar_type(),
        "main_data and fresh_data should have the same scalar_type. ",
        "key=",
        key,
        ", main_data.scalar_type()=",
        main_data.scalar_type(),
        ", fresh_data.scalar_type()=",
        fresh_data.scalar_type());
  }

  auto res = c10::Dict<std::string, Tensor>();
  for (size_t i = 0; i < keys.size(); i++) {
    res.insert(
        keys[i],
        gather_data_from_main_and_fresh_index_cpu_impl(
            top_indices_indices,
            main_indices,
            fresh_indices,
            list_main_data[i],
            list_fresh_data[i],
            fresh_vs_full_div_index,
            kernel_version));
  }

  return res;
}

// Fused: concatenate main+fresh scores (FRESH-first), take the global top-k
// without deduping, then gather the requested aux data from main and fresh in
// one op. This owns the fresh-first layout internally, so callers never
// construct the merged score tensor or pass a fresh/main pivot.
std::tuple<Tensor, Tensor, Tensor, c10::Dict<std::string, Tensor>>
take_top_k_and_gather_from_main_and_fresh_cpu(
    const Tensor& main_scores,
    const Tensor& fresh_scores,
    const Tensor& main_indices,
    const Tensor& fresh_indices,
    int64_t k,
    const std::vector<std::string>& keys,
    const std::vector<Tensor>& list_main_data,
    const std::vector<Tensor>& list_fresh_data,
    int64_t kernel_version) {
  const auto fresh_vs_full_div_index = fresh_scores.size(1);
  auto merged_scores = at::cat({fresh_scores, main_scores}, 1 /* dim */);
  const auto k_clamped = std::min<int64_t>(k, merged_scores.size(1));

  auto [top_scores, top_indices_indices] = at::topk(
      merged_scores,
      k_clamped,
      1 /* dim */,
      true /* largest */,
      true /* sorted */);

  auto is_from_fresh = at::lt(top_indices_indices, fresh_vs_full_div_index);

  auto gathered = gather_data_from_main_and_fresh_index_cpu(
      top_indices_indices,
      main_indices,
      fresh_indices,
      keys,
      list_main_data,
      list_fresh_data,
      fresh_vs_full_div_index,
      kernel_version);

  return std::make_tuple(
      std::move(top_scores),
      std::move(top_indices_indices),
      std::move(is_from_fresh),
      std::move(gathered));
}

} // namespace

TORCH_LIBRARY_FRAGMENT(st, m) {
  m.def(
      "take_top_k_and_gather_from_main_and_fresh("
      "Tensor main_scores, "
      "Tensor fresh_scores, "
      "Tensor main_indices, "
      "Tensor fresh_indices, "
      "int k, "
      "str[] keys, "
      "Tensor[] list_main_data, "
      "Tensor[] list_fresh_data, "
      "int kernel_version=2 "
      ") -> (Tensor, Tensor, Tensor, Dict(str, Tensor))");
}

TORCH_LIBRARY_IMPL(st, CPU, m) {
  m.impl(
      "take_top_k_and_gather_from_main_and_fresh",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(take_top_k_and_gather_from_main_and_fresh_cpu)));
}

} // namespace st::ops::fresh_index_post_processing
