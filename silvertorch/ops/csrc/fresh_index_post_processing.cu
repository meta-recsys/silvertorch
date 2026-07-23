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
#include <ATen/Dispatch.h> // @manual
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <cuda.h> // @manual
#include <torch/torch.h> // @manual

#include <algorithm>
#include <tuple>
#include <utility>

namespace st::ops::fresh_index_post_processing {

using at::PackedTensorAccessor64;
using at::Tensor;

namespace {

template <typename DATA_T>
__inline__ __device__ void select_store(
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

template <typename DATA_T, typename INDEX_T>
__global__ void gather_data_from_main_and_fresh_kernel_v2(
    const PackedTensorAccessor64<int64_t, 2> top_indices_indices,
    const PackedTensorAccessor64<INDEX_T, 2> main_indices,
    const PackedTensorAccessor64<INDEX_T, 2> fresh_indices,
    const DATA_T* main_source_data_ptr,
    int64_t main_source_data_total_items,
    const DATA_T* fresh_source_data_ptr,
    int64_t fresh_source_data_total_items,
    DATA_T* result_data_ptr,
    int64_t fresh_vs_full_div_column_index,
    int64_t total_indices,
    size_t data_dim,
    TORCH_DSA_KERNEL_ARGS) {
  for (int64_t global_index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       global_index < total_indices;
       global_index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const auto dim_0 = global_index / data_dim / top_indices_indices.size(1);
    const auto dim_1 = global_index / data_dim % top_indices_indices.size(1);
    const auto dim_2 = global_index % data_dim;

    const auto top_indices_index = top_indices_indices[dim_0][dim_1];
    if (top_indices_index < fresh_vs_full_div_column_index) {
      const auto local_index = top_indices_index;
      auto dim0_in_src = fresh_indices[dim_0][local_index];
      if (dim0_in_src < 0 || dim0_in_src >= fresh_source_data_total_items) {
        // Prevent out of bound access in case of padded offsets.
        dim0_in_src = 0;
      }
      result_data_ptr
          [dim_0 * data_dim * top_indices_indices.size(1) + dim_1 * data_dim +
           dim_2] = fresh_source_data_ptr[dim0_in_src * data_dim + dim_2];

    } else {
      const auto local_index =
          top_indices_index - fresh_vs_full_div_column_index;
      auto dim0_in_src = main_indices[dim_0][local_index];
      if (dim0_in_src < 0 || dim0_in_src >= main_source_data_total_items) {
        // Prevent out of bound access in case of padded offsets.
        dim0_in_src = 0;
      }
      result_data_ptr
          [dim_0 * data_dim * top_indices_indices.size(1) + dim_1 * data_dim +
           dim_2] = main_source_data_ptr[dim0_in_src * data_dim + dim_2];
    }
  }
}

template <typename DATA_T, typename INDEX_T>
__global__ void gather_data_from_main_and_fresh_kernel(
    const PackedTensorAccessor64<int64_t, 2> top_indices_indices,
    const PackedTensorAccessor64<INDEX_T, 2> main_indices,
    const PackedTensorAccessor64<INDEX_T, 2> fresh_indices,
    const DATA_T* main_source_data_ptr,
    int64_t main_source_data_total_items,
    const DATA_T* fresh_source_data_ptr,
    int64_t fresh_source_data_total_items,
    DATA_T* result_data_ptr,
    int64_t fresh_vs_full_div_column_index,
    int64_t total_indices,
    size_t data_dim,
    TORCH_DSA_KERNEL_ARGS) {
  for (int64_t global_index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       global_index < total_indices;
       global_index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const auto row = global_index / top_indices_indices.size(1);
    const auto col_in_dst = global_index % top_indices_indices.size(1);
    const auto top_indices_index = top_indices_indices[row][col_in_dst];
    if (top_indices_index < fresh_vs_full_div_column_index) {
      const auto local_index = top_indices_index;
      auto col_in_src = fresh_indices[row][local_index];
      if (col_in_src < 0 || col_in_src >= fresh_source_data_total_items) {
        // Prevent out of bound access in case of padded offsets.
        col_in_src = 0;
      }
      select_store<DATA_T>(
          fresh_source_data_ptr,
          result_data_ptr,
          data_dim,
          row,
          col_in_dst,
          col_in_src,
          top_indices_indices.size(1));

    } else {
      const auto local_index =
          top_indices_index - fresh_vs_full_div_column_index;
      auto col_in_src = main_indices[row][local_index];
      if (col_in_src < 0 || col_in_src >= main_source_data_total_items) {
        // Prevent out of bound access in case of padded offsets.
        col_in_src = 0;
      }
      select_store<DATA_T>(
          main_source_data_ptr,
          result_data_ptr,
          data_dim,
          row,
          col_in_dst,
          col_in_src,
          top_indices_indices.size(1));
    }
  }
}

Tensor gather_data_from_main_and_fresh_index_cuda_impl(
    const Tensor& top_indices_indices,
    const Tensor& main_indices,
    const Tensor& fresh_indices,
    const Tensor& main_data,
    const Tensor& fresh_data,
    int64_t fresh_vs_full_div_index,
    int64_t kernel_version) {
  const auto data_scalar_type = main_data.scalar_type();

  const auto result_num_rows = top_indices_indices.size(0);
  const auto result_num_cols = top_indices_indices.size(1);
  auto total_indices = result_num_rows * result_num_cols;

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
  if (kernel_version == 2) {
    total_indices = result_num_rows * result_num_cols * data_dim;
  }

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

  constexpr int64_t kThreads = 256L;
  const auto kBlocks =
      128L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const auto block_size = std::min<int64_t>(kThreads, total_indices);
  const auto grid_size =
      std::min<int64_t>(kBlocks, (total_indices + block_size - 1) / block_size);

  AT_DISPATCH_INDEX_TYPES(
      main_indices.scalar_type(),
      "gather_data_from_main_and_fresh_index_cuda_impl_index",
      [&]() {
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            main_data.scalar_type(),
            "gather_data_from_main_and_fresh_index_cuda_impl",
            [&]() {
              if (kernel_version == 2) {
                TORCH_DSA_KERNEL_LAUNCH(
                    gather_data_from_main_and_fresh_kernel_v2,
                    grid_size,
                    block_size,
                    0,
                    at::cuda::getCurrentCUDAStream(),
                    top_indices_indices.packed_accessor64<int64_t, 2>(),
                    main_indices.packed_accessor64<index_t, 2>(),
                    fresh_indices.packed_accessor64<index_t, 2>(),
                    main_data.data_ptr<scalar_t>(),
                    main_data.size(0),
                    fresh_data.data_ptr<scalar_t>(),
                    fresh_data.size(0),
                    result_data.mutable_data_ptr<scalar_t>(),
                    fresh_vs_full_div_index,
                    total_indices,
                    data_dim);
              } else {
                TORCH_DSA_KERNEL_LAUNCH(
                    gather_data_from_main_and_fresh_kernel,
                    grid_size,
                    block_size,
                    0,
                    at::cuda::getCurrentCUDAStream(),
                    top_indices_indices.packed_accessor64<int64_t, 2>(),
                    main_indices.packed_accessor64<index_t, 2>(),
                    fresh_indices.packed_accessor64<index_t, 2>(),
                    main_data.data_ptr<scalar_t>(),
                    main_data.size(0),
                    fresh_data.data_ptr<scalar_t>(),
                    fresh_data.size(0),
                    result_data.mutable_data_ptr<scalar_t>(),
                    fresh_vs_full_div_index,
                    total_indices,
                    data_dim);
              }
            });
      });

  return result_data;
}

c10::Dict<std::string, Tensor> gather_data_from_main_and_fresh_index_cuda(
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

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(top_indices_indices.get_device());

  auto res = c10::Dict<std::string, Tensor>();
  for (size_t i = 0; i < keys.size(); i++) {
    res.insert(
        keys[i],
        gather_data_from_main_and_fresh_index_cuda_impl(
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

std::tuple<Tensor, Tensor, Tensor, c10::Dict<std::string, Tensor>>
take_top_k_and_gather_from_main_and_fresh_cuda(
    const Tensor& main_scores,
    const Tensor& fresh_scores,
    const Tensor& main_indices,
    const Tensor& fresh_indices,
    int64_t k,
    const std::vector<std::string>& keys,
    const std::vector<Tensor>& list_main_data,
    const std::vector<Tensor>& list_fresh_data,
    int64_t kernel_version) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(main_scores.get_device());

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

  auto gathered = gather_data_from_main_and_fresh_index_cuda(
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

TORCH_LIBRARY_IMPL(st, CUDA, m) {
  m.impl(
      "take_top_k_and_gather_from_main_and_fresh",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(take_top_k_and_gather_from_main_and_fresh_cuda)));
}

} // namespace st::ops::fresh_index_post_processing
