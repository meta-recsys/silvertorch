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
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <cuda.h> // @manual
#include <torch/torch.h> // @manual
#include <ATen/cuda/DeviceUtils.cuh>
#include <algorithm>
#include <tuple>
#include <utility>

#include "fused_kmean_ann.cuh"
#include "simple_index_mm.cuh"

#define BLOOM_INDEX_CUDA
#include "bloom_index_util.cuh"

#define MAYBE_TORCH_DSA_KERNEL_LAUNCH(kernel, blocks, ...)  \
  {                                                         \
    if (blocks > 0) {                                       \
      TORCH_DSA_KERNEL_LAUNCH(kernel, blocks, __VA_ARGS__); \
    }                                                       \
  }

namespace st::ops::fused_kmean_ann {

using at::PackedTensorAccessor32;
using at::PackedTensorAccessor64;
using at::Tensor;
using namespace st::ops::simple_index_mm;
using namespace st::ops::bloom_index;

namespace {

constexpr int64_t kWarpThreadCount = 32;
constexpr int64_t kBlockSize = 256L;

// Multi-chunk warp payload kernel: processes ALL warps from ALL indices in one
// launch. Each warp resolves its chunk via `warp_chunk_indices`, converts the
// global cluster index to local, and accesses per-chunk data via GPU pointer
// lists.
__global__ void generate_warp_payload_multi_chunk_kernel(
    const int64_t* const* __restrict__ cluster_offsets_list,
    const int64_t* const* __restrict__ cluster_ids_list,
    const int32_t* const* __restrict__ cluster_warp_rounded_length_cumsum_list,
    const int32_t* __restrict__ global_assigned,
    const int32_t* __restrict__ global_cws_cumsum,
    const int32_t* __restrict__ warp_chunk_indices,
    const int32_t* __restrict__ per_chunk_cluster_cumsum,
    WarpPayload* __restrict__ warp_payloads,
    const int32_t* const* __restrict__ partial_mask_column_counts_cumsum_list,
    const int8_t* const* __restrict__ partial_mask_first_item_offset_in_column_list,
    const uint64_t* const* __restrict__ partial_mask_column_results_list,
    const uint64_t* __restrict__ filtering_bitmask_index_ptr,
    const int32_t* __restrict__ per_chunk_clusters_per_row,
    const int32_t* __restrict__ per_chunk_write_base,
    int32_t combined_max,
    int64_t total_needed_warps,
    TORCH_DSA_KERNEL_ARGS) {
  for (int64_t warp_index = blockIdx.x * blockDim.x + threadIdx.x;
       warp_index < total_needed_warps;
       warp_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    WarpPayload warp_payload;
    int32_t chunk_index = __ldg(warp_chunk_indices + warp_index);
    int32_t global_cluster_index = global_assigned[warp_index];
    int32_t chunk_cluster_offset = (chunk_index == 0)
        ? 0
        : __ldg(per_chunk_cluster_cumsum + chunk_index - 1);
    int32_t local_cluster_index = global_cluster_index - chunk_cluster_offset;
    int32_t clusters_per_row = __ldg(per_chunk_clusters_per_row + chunk_index);
    int32_t row = local_cluster_index / clusters_per_row;
    int32_t col = local_cluster_index % clusters_per_row;

    int64_t cluster_id =
        cluster_ids_list[chunk_index][row * clusters_per_row + col];
    int32_t cluster_start =
        static_cast<int32_t>(cluster_offsets_list[chunk_index][cluster_id]);
    int32_t doc_offset = static_cast<int32_t>(
        (warp_index -
         ((global_cluster_index == 0)
              ? 0
              : global_cws_cumsum[global_cluster_index - 1])) *
        kWarpThreadCount);
    warp_payload.doc_start_index = cluster_start + doc_offset;

    int32_t write_base = __ldg(per_chunk_write_base + chunk_index);
    int32_t warp_rounded_length_prev = (col == 0)
        ? 0
        : cluster_warp_rounded_length_cumsum_list[chunk_index]
                                                 [row * clusters_per_row + col -
                                                  1];
    warp_payload.write_index =
        row * combined_max + write_base + warp_rounded_length_prev + doc_offset;
    warp_payload.cluster_ids_row = row;

    int32_t mask_row = filtering_bitmask_index_ptr == nullptr
        ? row
        : static_cast<int32_t>(*(filtering_bitmask_index_ptr + row));
    int64_t cluster_index_in_mask =
        static_cast<int64_t>(mask_row) * clusters_per_row + col;
    int64_t cluster_start_mask_column = (cluster_index_in_mask == 0)
        ? 0
        : partial_mask_column_counts_cumsum_list[chunk_index]
                                                [cluster_index_in_mask - 1];
    warp_payload.filtering_mask = get_next_32_bit_mask(
        partial_mask_column_results_list[chunk_index] +
            cluster_start_mask_column,
        doc_offset +
            static_cast<int32_t>(partial_mask_first_item_offset_in_column_list
                                     [chunk_index][cluster_index_in_mask]));

    reinterpret_cast<int4*>(warp_payloads)[warp_index] = warp_payload.payload;
  }
}

// Multi-chunk remaining payload kernel: iterates over ALL clusters from ALL
// indices. Each thread handles one cluster's remaining docs.
__global__ void generate_remaining_payload_multi_chunk_kernel(
    const int64_t* const* __restrict__ cluster_offsets_list,
    const int64_t* const* __restrict__ cluster_ids_list,
    const int64_t* const* __restrict__ cluster_length_list,
    const int32_t* const* __restrict__ cluster_warp_rounded_length_cumsum_list,
    const int32_t* const* __restrict__ cluster_remaining_length_cumsum_list,
    const int32_t* __restrict__ cluster_chunk_indices,
    const int32_t* __restrict__ per_chunk_cluster_cumsum,
    const int32_t* __restrict__ per_chunk_remaining_output_offset,
    RemainingPayload* __restrict__ remaining_payloads,
    const int32_t* const* __restrict__ partial_mask_column_counts_cumsum_list,
    const int8_t* const* __restrict__ partial_mask_first_item_offset_in_column_list,
    const uint64_t* const* __restrict__ partial_mask_column_results_list,
    const uint64_t* __restrict__ filtering_bitmask_index_ptr,
    const int32_t* __restrict__ per_chunk_clusters_per_row,
    const int32_t* __restrict__ per_chunk_write_base,
    int32_t combined_max,
    int64_t total_clusters,
    TORCH_DSA_KERNEL_ARGS) {
  for (int64_t cluster_flat_index = blockIdx.x * blockDim.x + threadIdx.x;
       cluster_flat_index < total_clusters;
       cluster_flat_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int32_t chunk_index = __ldg(cluster_chunk_indices + cluster_flat_index);
    int32_t chunk_cluster_offset = (chunk_index == 0)
        ? 0
        : __ldg(per_chunk_cluster_cumsum + chunk_index - 1);
    int32_t local_index =
        static_cast<int32_t>(cluster_flat_index) - chunk_cluster_offset;
    int32_t clusters_per_row = __ldg(per_chunk_clusters_per_row + chunk_index);
    int32_t row = local_index / clusters_per_row;
    int32_t col = local_index % clusters_per_row;

    int32_t cur_cluster_length =
        static_cast<int32_t>(cluster_length_list[chunk_index][local_index]);
    int32_t warp_count = cur_cluster_length / kWarpThreadCount;
    int32_t already_handled = warp_count * kWarpThreadCount;
    int32_t remaining_docs = cur_cluster_length - already_handled;

    int64_t cluster_id = cluster_ids_list[chunk_index][local_index];
    int32_t doc_start =
        static_cast<int32_t>(cluster_offsets_list[chunk_index][cluster_id]) +
        already_handled;

    const int32_t* local_remaining_cumsum =
        cluster_remaining_length_cumsum_list[chunk_index];
    int32_t local_output_offset =
        (local_index == 0) ? 0 : local_remaining_cumsum[local_index - 1];
    int32_t global_output_offset =
        __ldg(per_chunk_remaining_output_offset + chunk_index) +
        local_output_offset;

    int32_t write_base = __ldg(per_chunk_write_base + chunk_index);
    int32_t write_index_offset = row * combined_max + write_base;
    write_index_offset +=
        cluster_warp_rounded_length_cumsum_list[chunk_index]
                                               [row * clusters_per_row +
                                                clusters_per_row - 1];
    int32_t row_start_remaining =
        (row == 0) ? 0 : local_remaining_cumsum[row * clusters_per_row - 1];
    int32_t remaining_in_row = local_output_offset - row_start_remaining;
    write_index_offset += remaining_in_row;

    int32_t mask_row = filtering_bitmask_index_ptr == nullptr
        ? row
        : static_cast<int32_t>(*(filtering_bitmask_index_ptr + row));
    int64_t cluster_index_in_mask =
        static_cast<int64_t>(mask_row) * clusters_per_row + col;
    int64_t cluster_start_mask_column = (cluster_index_in_mask == 0)
        ? 0
        : partial_mask_column_counts_cumsum_list[chunk_index]
                                                [cluster_index_in_mask - 1];
    int32_t mask_offset = already_handled +
        static_cast<int32_t>(partial_mask_first_item_offset_in_column_list
                                 [chunk_index][cluster_index_in_mask]);

    for (int32_t i = 0; i < remaining_docs; ++i) {
      RemainingPayload remaining_payload;
      remaining_payload.doc_index = doc_start + i;
      remaining_payload.cluster_ids_row = row;
      remaining_payload.write_index = write_index_offset + i;
      remaining_payload.filtering_mask = get_bit_64_bit_mask(
          partial_mask_column_results_list[chunk_index] +
              cluster_start_mask_column,
          mask_offset + i);
      reinterpret_cast<int4*>(remaining_payloads)[global_output_offset + i] =
          remaining_payload.payload;
    }
  }
}

template <
    typename EMBEDDING_T,
    typename RETURN_T,
    int DIM,
    bool FILTER,
    typename DIVISOR_T>
__inline__ __device__ void store(
    const EMBEDDING_T* embeddings,
    const EMBEDDING_T* queries,
    int64_t embeddings_index,
    int64_t row,
    DIVISOR_T divisor_for_int8,
    RETURN_T* results,
    int32_t* indices,
    bool filtering_mask) {
  if constexpr (FILTER) {
    if (!filtering_mask) {
      return;
    }
  }

  *results = get_score<
      EMBEDDING_T,
      RETURN_T,
      DIM,
      1,
      false /*EMB_ON_LOCAL_MEM=*/,
      DIVISOR_T>(
      embeddings + embeddings_index * DIM,
      queries + row * DIM,
      divisor_for_int8);
  *indices = static_cast<int32_t>(embeddings_index);
}

template <
    typename EMBEDDING_T,
    typename RETURN_T,
    int DIM,
    bool FILTER,
    typename DIVISOR_T>
__inline__ __device__ void store_with_direct_query(
    const EMBEDDING_T* embeddings,
    const EMBEDDING_T* query_ptr,
    int64_t embeddings_index,
    DIVISOR_T divisor_for_int8,
    RETURN_T* results,
    int32_t* indices,
    bool filtering_mask) {
  if constexpr (FILTER) {
    if (!filtering_mask) {
      return;
    }
  }

  *results = get_score<
      EMBEDDING_T,
      RETURN_T,
      DIM,
      1,
      false /*EMB_ON_LOCAL_MEM=*/,
      DIVISOR_T,
      true /*QUERY_ON_LOCAL_MEM=*/>(
      embeddings + embeddings_index * DIM, query_ptr, divisor_for_int8);
  *indices = static_cast<int32_t>(embeddings_index);
}

__global__ void generate_cluster_warp_size(
    const int64_t* selected_cluster_lengths,
    int64_t cluster_length_size,
    int32_t* cluster_warp_size,
    int32_t* cluster_warp_rounded_length,
    int32_t* cluster_remaining_length,
    TORCH_DSA_KERNEL_ARGS) {
  for (int64_t process_index = blockIdx.x * blockDim.x + threadIdx.x;
       process_index < cluster_length_size;
       process_index += blockDim.x * gridDim.x) {
    int32_t cur_cluster_length =
        static_cast<int32_t>(selected_cluster_lengths[process_index]);
    int32_t needed_warp_count = cur_cluster_length / kWarpThreadCount;
    cluster_warp_size[process_index] = needed_warp_count;
    cluster_warp_rounded_length[process_index] =
        needed_warp_count * kWarpThreadCount;
    cluster_remaining_length[process_index] =
        cur_cluster_length % kWarpThreadCount;
  }
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> round_cluster_to_warp(
    const Tensor& selected_cluster_lengths) {
  auto cluster_warp_size = at::empty(
      {selected_cluster_lengths.numel()},
      selected_cluster_lengths.options().dtype(at::kInt),
      at::MemoryFormat::Contiguous);

  auto cluster_warp_rounded_length = at::empty(
      {selected_cluster_lengths.numel()},
      selected_cluster_lengths.options().dtype(at::kInt),
      at::MemoryFormat::Contiguous);

  auto cluster_remaining_length = at::empty(
      {selected_cluster_lengths.numel()},
      selected_cluster_lengths.options().dtype(at::kInt),
      at::MemoryFormat::Contiguous);

  auto grid_size = std::min(
      (selected_cluster_lengths.numel() + kBlockSize - 1) / kBlockSize,
      128L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);

  at::cuda::CUDAGuard device_guard(selected_cluster_lengths.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_DSA_KERNEL_LAUNCH(
      generate_cluster_warp_size,
      grid_size,
      kBlockSize,
      0,
      stream,
      selected_cluster_lengths.data_ptr<int64_t>(),
      selected_cluster_lengths.numel(),
      cluster_warp_size.mutable_data_ptr<int32_t>(),
      cluster_warp_rounded_length.mutable_data_ptr<int32_t>(),
      cluster_remaining_length.mutable_data_ptr<int32_t>());

  return std::make_tuple(
      std::move(cluster_warp_size),
      cluster_warp_rounded_length.reshape(
          {selected_cluster_lengths.size(0), -1}),
      std::move(cluster_remaining_length));
}

namespace {

__global__ void generate_remaining_payload_kernel(
    const PackedTensorAccessor64<int64_t, 1> cluster_offsets,
    const PackedTensorAccessor64<int64_t, 2> selected_cluster_ids,
    const PackedTensorAccessor64<int64_t, 2> selected_cluster_lengths,
    const PackedTensorAccessor32<int32_t, 2> cluster_warp_rounded_length_cumsum,
    const int32_t* cluster_remaining_length_cumsum,
    const uint64_t* filtering_bitmask_ptr,
    RemainingPayload* remaining_payloads,
    int64_t filtering_bitmask_column,
    const uint64_t* filtering_bitmask_index_ptr,
    int64_t max_tensor_size_per_row,
    TORCH_DSA_KERNEL_ARGS) {
  int64_t cluster_length_column_count = selected_cluster_lengths.size(1);
  int64_t total_selected_clusters =
      selected_cluster_lengths.size(0) * cluster_length_column_count;

  for (int64_t process_index = blockIdx.x * blockDim.x + threadIdx.x;
       process_index < total_selected_clusters;
       process_index += blockDim.x * gridDim.x) {
    int32_t row =
        static_cast<int32_t>(process_index / cluster_length_column_count);
    int32_t column =
        static_cast<int32_t>(process_index % cluster_length_column_count);
    int32_t cur_cluster_length =
        static_cast<int32_t>(selected_cluster_lengths[row][column]);
    int32_t needed_warp_count = cur_cluster_length / kWarpThreadCount;
    int32_t already_handled_docs = needed_warp_count * kWarpThreadCount;

    int64_t cluster_id = selected_cluster_ids[row][column];
    int32_t cluster_id_start_index =
        static_cast<int32_t>(cluster_offsets[cluster_id]);
    int32_t doc_idex_start = cluster_id_start_index + already_handled_docs;
    int32_t remaining_docs = cur_cluster_length - already_handled_docs;

    int32_t remaining_payloads_offset = (process_index == 0)
        ? 0
        : cluster_remaining_length_cumsum[process_index - 1];

    int32_t last_row_all_remaining =
        ((row == 0) ? 0
                    : cluster_remaining_length_cumsum
                          [row * cluster_length_column_count - 1]);
    int32_t remaining_payloads_offset_in_row =
        (remaining_payloads_offset - last_row_all_remaining);
    int32_t write_index_offset =
        row * static_cast<int32_t>(max_tensor_size_per_row);
    write_index_offset +=
        cluster_warp_rounded_length_cumsum[row]
                                          [cluster_length_column_count - 1];
    write_index_offset += remaining_payloads_offset_in_row;

    static constexpr int32_t kBitMaskSizePerUInt64 = 64;
    int32_t next_filtering_bitmask_index =
        (doc_idex_start + kBitMaskSizePerUInt64 - 1) / kBitMaskSizePerUInt64 *
        kBitMaskSizePerUInt64;
    uint64_t filtering_bitmask = 0;
    if (filtering_bitmask_ptr != nullptr) {
      int32_t final_row = filtering_bitmask_index_ptr == nullptr
          ? row
          : static_cast<int32_t>(*(filtering_bitmask_index_ptr + row));
      filtering_bitmask =
          *(filtering_bitmask_ptr + final_row * filtering_bitmask_column +
            doc_idex_start / kBitMaskSizePerUInt64);
    }

    for (int32_t i = 0; i < remaining_docs; ++i) {
      RemainingPayload remaining_payload;
      remaining_payload.doc_index = doc_idex_start + i;
      remaining_payload.cluster_ids_row = row;
      remaining_payload.write_index = write_index_offset + i;

      if (filtering_bitmask_column) {
        if (remaining_payload.doc_index == next_filtering_bitmask_index) {
          int32_t final_row = filtering_bitmask_index_ptr == nullptr
              ? row
              : static_cast<int32_t>(*(filtering_bitmask_index_ptr + row));
          filtering_bitmask =
              *(filtering_bitmask_ptr + final_row * filtering_bitmask_column +
                remaining_payload.doc_index / kBitMaskSizePerUInt64);
        }

        remaining_payload.filtering_mask =
            get_bit_64_bit_mask(filtering_bitmask, remaining_payload.doc_index);
      }

      reinterpret_cast<int4*>(
          remaining_payloads)[remaining_payloads_offset + i] =
          remaining_payload.payload;
    }
  }
}

} // namespace

std::tuple<Tensor, int64_t> generate_remaining_payload(
    const Tensor& selected_cluster_lengths,
    const Tensor& cluster_offsets,
    const Tensor& selected_cluster_ids,
    const Tensor& cluster_warp_rounded_length_cumsum,
    const Tensor& cluster_remaining_length_cumsum,
    const uint64_t* filtering_bitmask_ptr,
    int64_t filtering_bitmask_column,
    const uint64_t* filtering_bitmask_index_ptr,
    int64_t max_tensor_size_per_row) {
  TORCH_CHECK(cluster_warp_rounded_length_cumsum.dim() == 2);
  TORCH_CHECK(cluster_remaining_length_cumsum.dim() == 1);

  int64_t remaining_docs = cluster_remaining_length_cumsum[-1].item<int32_t>();
  auto remaining_payloads = at::empty(
      {remaining_docs * static_cast<int64_t>(sizeof(RemainingPayload))},
      selected_cluster_lengths.options().dtype(at::kChar),
      at::MemoryFormat::Contiguous);

  auto grid_size = std::min(
      (selected_cluster_lengths.numel() + kBlockSize - 1) / kBlockSize,
      128L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);

  TORCH_DSA_KERNEL_LAUNCH(
      generate_remaining_payload_kernel,
      grid_size,
      kBlockSize,
      0,
      at::cuda::getCurrentCUDAStream(),
      cluster_offsets.packed_accessor64<int64_t, 1>(),
      selected_cluster_ids.packed_accessor64<int64_t, 2>(),
      selected_cluster_lengths.packed_accessor64<int64_t, 2>(),
      cluster_warp_rounded_length_cumsum.packed_accessor32<int32_t, 2>(),
      cluster_remaining_length_cumsum.data_ptr<int32_t>(),
      filtering_bitmask_ptr,
      reinterpret_cast<RemainingPayload*>(
          remaining_payloads.mutable_data_ptr<int8_t>()),
      filtering_bitmask_column,
      filtering_bitmask_index_ptr,
      max_tensor_size_per_row);

  return std::make_tuple(std::move(remaining_payloads), remaining_docs);
}

namespace {

__global__ void generate_warp_payload_kernel(
    const PackedTensorAccessor64<int64_t, 1> cluster_offsets,
    const PackedTensorAccessor64<int64_t, 2> selected_cluster_ids,
    const PackedTensorAccessor32<int32_t, 2> cluster_warp_rounded_length_cumsum,
    const int32_t* per_warp_index_in_selected_cluster_ids,
    const int32_t* cluster_warp_size_cumsum,
    const uint64_t* filtering_bitmask_ptr,
    WarpPayload* warp_payloads,
    int64_t filtering_bitmask_column,
    const uint64_t* filtering_bitmask_index_ptr,
    int64_t total_needed_warps,
    int64_t max_tensor_size_per_row,
    TORCH_DSA_KERNEL_ARGS) {
  int64_t clusters_per_row = selected_cluster_ids.size(1);

  for (int64_t process_warp = blockIdx.x * blockDim.x + threadIdx.x;
       process_warp < total_needed_warps;
       process_warp += blockDim.x * gridDim.x) {
    WarpPayload warp_payload;
    int32_t cluster_id_index =
        per_warp_index_in_selected_cluster_ids[process_warp];

    int32_t row = static_cast<int32_t>(cluster_id_index / clusters_per_row);
    int64_t cluster_ids_column = cluster_id_index % clusters_per_row;
    int64_t cluster_id = selected_cluster_ids[row][cluster_ids_column];

    int32_t cluster_start_doc_index =
        static_cast<int32_t>(cluster_offsets[cluster_id]);
    int32_t doc_offset_in_cur_cluster = static_cast<int32_t>(
        (process_warp -
         ((cluster_id_index == 0)
              ? 0
              : cluster_warp_size_cumsum[cluster_id_index - 1])) *
        kWarpThreadCount);
    warp_payload.doc_start_index =
        cluster_start_doc_index + doc_offset_in_cur_cluster;

    warp_payload.write_index = static_cast<int32_t>(
        row * max_tensor_size_per_row +
        ((cluster_ids_column == 0)
             ? 0
             : cluster_warp_rounded_length_cumsum[row]
                                                 [cluster_ids_column - 1]) +
        doc_offset_in_cur_cluster);

    warp_payload.cluster_ids_row = row;

    if (filtering_bitmask_ptr != nullptr) {
      int32_t final_row = filtering_bitmask_index_ptr == nullptr
          ? row
          : static_cast<int32_t>(*(filtering_bitmask_index_ptr + row));
      warp_payload.filtering_mask = get_next_32_bit_mask(
          filtering_bitmask_ptr + final_row * filtering_bitmask_column,
          warp_payload.doc_start_index);
    }

    reinterpret_cast<int4*>(warp_payloads)[process_warp] = warp_payload.payload;
  }
}

} // namespace

std::tuple<Tensor, int64_t> generate_warp_payload(
    const Tensor& cluster_warp_size,
    const Tensor& cluster_offsets,
    const Tensor& selected_cluster_ids,
    const Tensor& cluster_warp_rounded_length_cumsum,
    const uint64_t* filtering_bitmask_ptr,
    int64_t filtering_bitmask_column,
    const uint64_t* filtering_bitmask_index_ptr,
    int64_t max_tensor_size_per_row) {
  TORCH_CHECK(cluster_warp_size.dim() == 1);
  TORCH_CHECK(cluster_warp_rounded_length_cumsum.dim() == 2);
  auto per_warp_index_in_selected_cluster_ids =
      torch::arange(
          cluster_warp_size.numel(),
          cluster_warp_size.options().dtype(at::kInt))
          .repeat_interleave(cluster_warp_size);

  auto cluster_warp_size_cumsum = cluster_warp_size.cumsum(
      /*dim=*/0, /*dtype=*/at::kInt);

  auto warp_payloads = at::empty(
      {per_warp_index_in_selected_cluster_ids.numel() *
       static_cast<int64_t>(sizeof(WarpPayload))},
      per_warp_index_in_selected_cluster_ids.options().dtype(at::kChar),
      at::MemoryFormat::Contiguous);

  int64_t total_needed_warps = per_warp_index_in_selected_cluster_ids.numel();
  auto grid_size = std::min(
      (total_needed_warps + kBlockSize - 1) / kBlockSize,
      128L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);

  MAYBE_TORCH_DSA_KERNEL_LAUNCH(
      generate_warp_payload_kernel,
      grid_size,
      kBlockSize,
      0,
      at::cuda::getCurrentCUDAStream(),
      cluster_offsets.packed_accessor64<int64_t, 1>(),
      selected_cluster_ids.packed_accessor64<int64_t, 2>(),
      cluster_warp_rounded_length_cumsum.packed_accessor32<int32_t, 2>(),
      per_warp_index_in_selected_cluster_ids.data_ptr<int32_t>(),
      cluster_warp_size_cumsum.data_ptr<int32_t>(),
      filtering_bitmask_ptr,
      reinterpret_cast<WarpPayload*>(warp_payloads.mutable_data_ptr<int8_t>()),
      filtering_bitmask_column,
      filtering_bitmask_index_ptr,
      total_needed_warps,
      max_tensor_size_per_row);

  return std::make_tuple(std::move(warp_payloads), total_needed_warps);
}

namespace {

__global__ void generate_warp_payload_with_partial_masks_kernel(
    const PackedTensorAccessor64<int64_t, 1> cluster_offsets,
    const PackedTensorAccessor64<int64_t, 2> cluster_ids,
    const PackedTensorAccessor32<int32_t, 2> cluster_warp_rounded_length_cumsum,
    const int32_t* assigned_cluster_id_index,
    const int32_t* cluster_warp_size_cumsum,
    WarpPayload* warp_payloads,
    const int32_t* partial_mask_column_counts_cumsum,
    const int8_t* partial_mask_first_item_offset_in_column,
    const uint64_t* partial_mask_column_results,
    const uint64_t* filtering_bitmask_index_ptr,
    int64_t total_needed_warps,
    int64_t max_tensor_size_per_row,
    TORCH_DSA_KERNEL_ARGS) {
  int64_t clusters_per_row = cluster_ids.size(1);

  for (int64_t process_warp = blockIdx.x * blockDim.x + threadIdx.x;
       process_warp < total_needed_warps;
       process_warp += blockDim.x * gridDim.x) {
    WarpPayload warp_payload;
    int32_t cluster_id_index = assigned_cluster_id_index[process_warp];

    int32_t row = static_cast<int32_t>(cluster_id_index / clusters_per_row);
    int64_t cluster_ids_column = cluster_id_index % clusters_per_row;
    int64_t cluster_id = cluster_ids[row][cluster_ids_column];

    int32_t cluster_start_doc_index =
        static_cast<int32_t>(cluster_offsets[cluster_id]);
    int32_t doc_offset_in_cur_cluster = static_cast<int32_t>(
        (process_warp -
         ((cluster_id_index == 0)
              ? 0
              : cluster_warp_size_cumsum[cluster_id_index - 1])) *
        kWarpThreadCount);
    warp_payload.doc_start_index =
        cluster_start_doc_index + doc_offset_in_cur_cluster;

    warp_payload.write_index = static_cast<int32_t>(
        row * max_tensor_size_per_row +
        ((cluster_ids_column == 0)
             ? 0
             : cluster_warp_rounded_length_cumsum[row]
                                                 [cluster_ids_column - 1]) +
        doc_offset_in_cur_cluster);

    warp_payload.cluster_ids_row = row;

    int32_t mask_row = filtering_bitmask_index_ptr == nullptr
        ? row
        : static_cast<int32_t>(*(filtering_bitmask_index_ptr + row));
    int64_t cluster_index_in_mask =
        mask_row * clusters_per_row + cluster_ids_column;
    int64_t cluster_start_mask_column_index = cluster_index_in_mask == 0
        ? 0
        : partial_mask_column_counts_cumsum[cluster_index_in_mask - 1];
    warp_payload.filtering_mask = get_next_32_bit_mask(
        partial_mask_column_results + cluster_start_mask_column_index,
        doc_offset_in_cur_cluster +
            static_cast<int32_t>(partial_mask_first_item_offset_in_column
                                     [cluster_index_in_mask]));
    reinterpret_cast<int4*>(warp_payloads)[process_warp] = warp_payload.payload;
  }
}

} // namespace

std::tuple<Tensor, int64_t> generate_warp_payload_with_partial_masks(
    const Tensor& cluster_warp_size,
    const Tensor& cluster_offsets,
    const Tensor& cluster_ids,
    const Tensor& cluster_warp_rounded_length_cumsum,
    const Tensor& cluster_warp_size_cumsum,
    int32_t total_warps,
    const int32_t* partial_mask_column_counts_cumsum,
    const int8_t* partial_mask_first_item_offset_in_column,
    const uint64_t* partial_mask_column_results,
    const uint64_t* filtering_bitmask_index_ptr,
    int64_t max_tensor_size_per_row) {
  TORCH_CHECK(cluster_warp_size.dim() == 1);
  TORCH_CHECK(cluster_warp_rounded_length_cumsum.dim() == 2);

  auto input = torch::arange(
      cluster_warp_size.numel(), cluster_warp_size.options().dtype(at::kInt));
  auto assigned_cluster_id_index = input.repeat_interleave(cluster_warp_size);

  auto warp_payloads = at::empty(
      {assigned_cluster_id_index.numel() *
       static_cast<int64_t>(sizeof(WarpPayload))},
      assigned_cluster_id_index.options().dtype(at::kChar),
      at::MemoryFormat::Contiguous);

  int64_t total_needed_warps = assigned_cluster_id_index.numel();
  auto grid_size = std::min(
      (total_needed_warps + kBlockSize - 1) / kBlockSize,
      128L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);

  MAYBE_TORCH_DSA_KERNEL_LAUNCH(
      generate_warp_payload_with_partial_masks_kernel,
      grid_size,
      kBlockSize,
      0,
      at::cuda::getCurrentCUDAStream(),
      cluster_offsets.packed_accessor64<int64_t, 1>(),
      cluster_ids.packed_accessor64<int64_t, 2>(),
      cluster_warp_rounded_length_cumsum.packed_accessor32<int32_t, 2>(),
      assigned_cluster_id_index.data_ptr<int32_t>(),
      cluster_warp_size_cumsum.data_ptr<int32_t>(),
      reinterpret_cast<WarpPayload*>(warp_payloads.mutable_data_ptr<int8_t>()),
      partial_mask_column_counts_cumsum,
      partial_mask_first_item_offset_in_column,
      partial_mask_column_results,
      filtering_bitmask_index_ptr,
      total_needed_warps,
      max_tensor_size_per_row);

  return std::make_tuple(std::move(warp_payloads), total_needed_warps);
}

namespace {

__global__ void generate_remaining_payload_with_partial_masks_kernel(
    const PackedTensorAccessor64<int64_t, 1> cluster_offsets,
    const PackedTensorAccessor64<int64_t, 2> cluster_ids,
    const PackedTensorAccessor64<int64_t, 2> cluster_length,
    const PackedTensorAccessor32<int32_t, 2> cluster_warp_rounded_length_cumsum,
    const int32_t* cluster_remaining_length_cumsum,
    RemainingPayload* remaining_payloads,
    const int32_t* partial_mask_column_counts_cumsum,
    const int8_t* partial_mask_first_item_offset_in_column,
    const uint64_t* partial_mask_column_results,
    const uint64_t* filtering_bitmask_index_ptr,
    int64_t max_tensor_size_per_row,
    TORCH_DSA_KERNEL_ARGS) {
  int64_t cluster_length_column_count = cluster_length.size(1);
  int64_t cluster_length_size =
      cluster_length.size(0) * cluster_length_column_count;

  for (int64_t process_index = blockIdx.x * blockDim.x + threadIdx.x;
       process_index < cluster_length_size;
       process_index += blockDim.x * gridDim.x) {
    int32_t row =
        static_cast<int32_t>(process_index / cluster_length_column_count);
    int32_t column =
        static_cast<int32_t>(process_index % cluster_length_column_count);
    int32_t cur_cluster_length =
        static_cast<int32_t>(cluster_length[row][column]);
    int32_t needed_warp_count = cur_cluster_length / kWarpThreadCount;
    int32_t already_handled_docs = needed_warp_count * kWarpThreadCount;

    int64_t cluster_id = cluster_ids[row][column];
    int32_t cluster_id_start_index =
        static_cast<int32_t>(cluster_offsets[cluster_id]);
    int32_t doc_idex_start = cluster_id_start_index + already_handled_docs;
    int32_t remaining_docs = cur_cluster_length - already_handled_docs;

    int32_t remaining_payloads_offset = (process_index == 0)
        ? 0
        : cluster_remaining_length_cumsum[process_index - 1];

    int32_t last_row_all_remaining =
        ((row == 0) ? 0
                    : cluster_remaining_length_cumsum
                          [row * cluster_length_column_count - 1]);
    int32_t remaining_payloads_offset_in_row =
        (remaining_payloads_offset - last_row_all_remaining);
    int32_t write_index_offset =
        row * static_cast<int32_t>(max_tensor_size_per_row);
    write_index_offset +=
        cluster_warp_rounded_length_cumsum[row]
                                          [cluster_length_column_count - 1];
    write_index_offset += remaining_payloads_offset_in_row;

    int32_t mask_row = filtering_bitmask_index_ptr == nullptr
        ? row
        : static_cast<int32_t>(*(filtering_bitmask_index_ptr + row));
    int64_t cluster_index_in_mask =
        mask_row * cluster_length_column_count + column;
    int64_t cluster_start_mask_column_index = cluster_index_in_mask == 0
        ? 0
        : partial_mask_column_counts_cumsum[cluster_index_in_mask - 1];
    int32_t mask_column_offset =
        already_handled_docs +
        static_cast<int32_t>(
            partial_mask_first_item_offset_in_column[cluster_index_in_mask]);
    for (int32_t i = 0; i < remaining_docs; ++i) {
      RemainingPayload remaining_payload;
      remaining_payload.doc_index = doc_idex_start + i;
      remaining_payload.cluster_ids_row = row;
      remaining_payload.write_index = write_index_offset + i;
      remaining_payload.filtering_mask = get_bit_64_bit_mask(
          partial_mask_column_results + cluster_start_mask_column_index,
          mask_column_offset + i);
      reinterpret_cast<int4*>(
          remaining_payloads)[remaining_payloads_offset + i] =
          remaining_payload.payload;
    }
  }
}

} // namespace

std::tuple<Tensor, int64_t> generate_remaining_payload_with_partial_masks(
    const Tensor& cluster_length,
    const Tensor& cluster_offsets,
    const Tensor& cluster_ids,
    const Tensor& cluster_warp_rounded_length_cumsum,
    const Tensor& cluster_remaining_length_cumsum,
    int32_t remaining_docs,
    const int32_t* partial_mask_column_counts_cumsum,
    const int8_t* partial_mask_first_item_offset_in_column,
    const uint64_t* partial_mask_column_results,
    const uint64_t* filtering_bitmask_index_ptr,
    int64_t max_tensor_size_per_row) {
  TORCH_CHECK(cluster_warp_rounded_length_cumsum.dim() == 2);
  TORCH_CHECK(cluster_remaining_length_cumsum.dim() == 1);

  auto remaining_payloads = at::empty(
      {remaining_docs * static_cast<int64_t>(sizeof(RemainingPayload))},
      cluster_length.options().dtype(at::kChar),
      at::MemoryFormat::Contiguous);

  auto grid_size = std::min(
      (cluster_length.numel() + kBlockSize - 1) / kBlockSize,
      128L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);

  TORCH_DSA_KERNEL_LAUNCH(
      generate_remaining_payload_with_partial_masks_kernel,
      grid_size,
      kBlockSize,
      0,
      at::cuda::getCurrentCUDAStream(),
      cluster_offsets.packed_accessor64<int64_t, 1>(),
      cluster_ids.packed_accessor64<int64_t, 2>(),
      cluster_length.packed_accessor64<int64_t, 2>(),
      cluster_warp_rounded_length_cumsum.packed_accessor32<int32_t, 2>(),
      cluster_remaining_length_cumsum.data_ptr<int32_t>(),
      reinterpret_cast<RemainingPayload*>(
          remaining_payloads.mutable_data_ptr<int8_t>()),
      partial_mask_column_counts_cumsum,
      partial_mask_first_item_offset_in_column,
      partial_mask_column_results,
      filtering_bitmask_index_ptr,
      max_tensor_size_per_row);

  return std::make_tuple(std::move(remaining_payloads), remaining_docs);
}

namespace {

template <bool FILTER>
__inline__ __device__ bool get_filtering_mask(
    uint32_t /* filtering_mask */,
    int32_t /* pos */) {
  return true;
}

template <>
__inline__ __device__ bool get_filtering_mask<true>(
    uint32_t filtering_mask,
    int32_t pos) {
  return get_bit_32_bit_mask(filtering_mask, pos);
}

template <int DIM, bool FILTER>
__global__ void __launch_bounds__(kBlockSize, 2) process_cluster_v4_pipelined(
    const int8_t* __restrict__ embeddings,
    const int8_t* __restrict__ queries,
    const WarpPayload* __restrict__ warp_payloads,
    c10::Half* results,
    int32_t* indices,
    int32_t total_needed_warps,
    int32_t divisor_for_int8,
    TORCH_DSA_KERNEL_ARGS) {
  constexpr int kWarpsInBlock = kBlockSize / kWarpThreadCount;
  __shared__ int4 shared_queries[2][kWarpsInBlock][DIM / 16];

  int buffer_idx = 0;
  uint32_t warp_index = blockIdx.x * blockDim.y + threadIdx.y;

  if (warp_index < total_needed_warps) {
    WarpPayload warp_payload;
    warp_payload.payload = __ldg(
        reinterpret_cast<const int4* __restrict__>(warp_payloads) + warp_index);
    const int4* query_ptr = reinterpret_cast<const int4*>(
        queries + warp_payload.cluster_ids_row * DIM);
#pragma unroll
    for (uint32_t d = threadIdx.x; d < DIM / 16; d += kWarpThreadCount) {
      shared_queries[buffer_idx][threadIdx.y][d] = __ldg(query_ptr + d);
    }
  }

  for (; warp_index < total_needed_warps;
       warp_index += blockDim.y * gridDim.x) {
    WarpPayload warp_payload;
    warp_payload.payload = __ldg(
        reinterpret_cast<const int4* __restrict__>(warp_payloads) + warp_index);

    WARP_SYNC();

    uint32_t next_warp_index = warp_index + blockDim.y * gridDim.x;
    int next_buffer_idx = 1 - buffer_idx;
    if (next_warp_index < total_needed_warps) {
      WarpPayload next_payload;
      next_payload.payload = __ldg(
          reinterpret_cast<const int4* __restrict__>(warp_payloads) +
          next_warp_index);
      const int4* next_query_ptr = reinterpret_cast<const int4*>(
          queries + next_payload.cluster_ids_row * DIM);
#pragma unroll
      for (uint32_t d = threadIdx.x; d < DIM / 16; d += kWarpThreadCount) {
        shared_queries[next_buffer_idx][threadIdx.y][d] =
            __ldg(next_query_ptr + d);
      }
    }

    int64_t doc_index = warp_payload.doc_start_index + threadIdx.x;
    int64_t write_index = warp_payload.write_index + threadIdx.x;

    bool filtering_mask = true;
    if constexpr (FILTER) {
      filtering_mask =
          get_filtering_mask<FILTER>(warp_payload.filtering_mask, threadIdx.x);
      if (!filtering_mask) {
        buffer_idx = next_buffer_idx;
        continue;
      }
    }

    const int4* emb_ptr =
        reinterpret_cast<const int4*>(embeddings + doc_index * DIM);

    int32_t score = 0;
#pragma unroll
    for (int i = 0; i < DIM / 16; ++i) {
      int4 emb16 = __ldg(emb_ptr + i);
      int4 query16 = shared_queries[buffer_idx][threadIdx.y][i];
      score = cuda_mm_types<int8_t>::multi_add(emb16.x, query16.x, score);
      score = cuda_mm_types<int8_t>::multi_add(emb16.y, query16.y, score);
      score = cuda_mm_types<int8_t>::multi_add(emb16.z, query16.z, score);
      score = cuda_mm_types<int8_t>::multi_add(emb16.w, query16.w, score);
    }

    results[write_index] = c10::Half(
        static_cast<float>(score) / static_cast<float>(divisor_for_int8));
    indices[write_index] = static_cast<int32_t>(doc_index);

    buffer_idx = next_buffer_idx;
  }
}

template <
    typename EMBEDDING_T,
    typename RETURN_T,
    int DIM,
    bool FILTER,
    typename DIVISOR_INPUT,
    bool QUERY_ON_SHARED_MEM = false>
__global__ void process_cluster(
    const EMBEDDING_T* embeddings,
    const EMBEDDING_T* queries,
    const WarpPayload* __restrict__ warp_payloads,
    RETURN_T* results,
    int32_t* indices,
    int32_t total_needed_warps,
    DIVISOR_INPUT divisor_for_int8,
    TORCH_DSA_KERNEL_ARGS) {
  constexpr int kWarpsInBlock = kBlockSize / kWarpThreadCount;
  __shared__ EMBEDDING_T shared_queries[kWarpsInBlock][DIM];

  for (uint32_t warp_index = blockIdx.x * blockDim.y + threadIdx.y;
       warp_index < total_needed_warps;
       warp_index += blockDim.y * gridDim.x) {
    WarpPayload warp_payload;
    warp_payload.payload = __ldg(
        reinterpret_cast<const int4* __restrict__>(warp_payloads) + warp_index);

    int64_t doc_index = warp_payload.doc_start_index + threadIdx.x;
    int64_t write_index = warp_payload.write_index + threadIdx.x;

    decltype(resolve_divisor(divisor_for_int8, doc_index)) divisor_value =
        resolve_divisor(divisor_for_int8, doc_index);

    if constexpr (QUERY_ON_SHARED_MEM) {
      const EMBEDDING_T* query = queries + warp_payload.cluster_ids_row * DIM;
#pragma unroll
      for (uint32_t d = threadIdx.x; d < DIM; d += kWarpThreadCount) {
        shared_queries[threadIdx.y][d] = *(query + d);
      }
#ifndef USE_ROCM
      __syncwarp();
#endif

      store_with_direct_query<
          EMBEDDING_T,
          RETURN_T,
          DIM,
          FILTER,
          decltype(divisor_value)>(
          embeddings,
          shared_queries[threadIdx.y],
          doc_index,
          divisor_value,
          results + write_index,
          indices + write_index,
          get_filtering_mask<FILTER>(warp_payload.filtering_mask, threadIdx.x));
    } else {
      store<EMBEDDING_T, RETURN_T, DIM, FILTER, decltype(divisor_value)>(
          embeddings,
          queries,
          doc_index,
          static_cast<int64_t>(warp_payload.cluster_ids_row),
          divisor_value,
          results + write_index,
          indices + write_index,
          get_filtering_mask<FILTER>(warp_payload.filtering_mask, threadIdx.x));
    }
  }
}

template <
    typename EMBEDDING_T,
    typename RETURN_T,
    int DIM,
    bool FILTER,
    typename DIVISOR_INPUT>
__global__ void process_cluster_remaining(
    const EMBEDDING_T* embeddings,
    const EMBEDDING_T* queries,
    const RemainingPayload* __restrict__ remaining_payloads,
    RETURN_T* results,
    int32_t* indices,
    int32_t remaining_docs,
    DIVISOR_INPUT divisor_for_int8,
    TORCH_DSA_KERNEL_ARGS) {
  for (uint32_t process_index = blockIdx.x * blockDim.x + threadIdx.x;
       process_index < remaining_docs;
       process_index += blockDim.x * gridDim.x) {
    RemainingPayload remaining_payload;
    remaining_payload.payload = __ldg(
        reinterpret_cast<const int4* __restrict__>(remaining_payloads) +
        process_index);

    int32_t doc_index = remaining_payload.doc_index;
    int32_t write_index = remaining_payload.write_index;

    decltype(resolve_divisor(divisor_for_int8, doc_index)) divisor_value =
        resolve_divisor(divisor_for_int8, doc_index);

    store<EMBEDDING_T, RETURN_T, DIM, FILTER, decltype(divisor_value)>(
        embeddings,
        queries,
        doc_index,
        remaining_payload.cluster_ids_row,
        divisor_value,
        results + write_index,
        indices + write_index,
        remaining_payload.filtering_mask);
  }
}

template <
    typename EMBEDDING_T,
    typename RETURN_T,
    int DIM,
    bool FILTER,
    bool HAS_PER_EMB_SCALE>
__global__ void process_cluster_multi_emb(
    const EMBEDDING_T* const* __restrict__ embedding_list,
    const EMBEDDING_T* queries,
    const WarpPayload* __restrict__ warp_payloads,
    const int32_t* __restrict__ warp_chunk_indices,
    RETURN_T* results,
    int32_t* indices,
    int32_t total_needed_warps,
    int32_t divisor_for_int8_scalar,
    const c10::Half* const* __restrict__ per_emb_scale_list,
    TORCH_DSA_KERNEL_ARGS) {
  for (uint32_t warp_index = blockIdx.x * blockDim.y + threadIdx.y;
       warp_index < total_needed_warps;
       warp_index += blockDim.y * gridDim.x) {
    WarpPayload warp_payload;
    warp_payload.payload = __ldg(
        reinterpret_cast<const int4* __restrict__>(warp_payloads) + warp_index);

    int32_t chunk_idx = __ldg(warp_chunk_indices + warp_index);
    const EMBEDDING_T* embeddings = embedding_list[chunk_idx];

    int64_t doc_index = warp_payload.doc_start_index + threadIdx.x;
    int64_t write_index = warp_payload.write_index + threadIdx.x;

    if constexpr (HAS_PER_EMB_SCALE) {
      auto divisor_value = *(per_emb_scale_list[chunk_idx] + doc_index);
      store<EMBEDDING_T, RETURN_T, DIM, FILTER, decltype(divisor_value)>(
          embeddings,
          queries,
          doc_index,
          static_cast<int64_t>(warp_payload.cluster_ids_row),
          divisor_value,
          results + write_index,
          indices + write_index,
          get_filtering_mask<FILTER>(warp_payload.filtering_mask, threadIdx.x));
    } else {
      store<EMBEDDING_T, RETURN_T, DIM, FILTER, int32_t>(
          embeddings,
          queries,
          doc_index,
          static_cast<int64_t>(warp_payload.cluster_ids_row),
          divisor_for_int8_scalar,
          results + write_index,
          indices + write_index,
          get_filtering_mask<FILTER>(warp_payload.filtering_mask, threadIdx.x));
    }
  }
}

template <
    typename EMBEDDING_T,
    typename RETURN_T,
    int DIM,
    bool FILTER,
    bool HAS_PER_EMB_SCALE>
__global__ void process_cluster_remaining_multi_emb(
    const EMBEDDING_T* const* __restrict__ embedding_list,
    const EMBEDDING_T* queries,
    const RemainingPayload* __restrict__ remaining_payloads,
    const int32_t* __restrict__ remaining_chunk_indices,
    RETURN_T* results,
    int32_t* indices,
    int32_t remaining_docs,
    int32_t divisor_for_int8_scalar,
    const c10::Half* const* __restrict__ per_emb_scale_list,
    TORCH_DSA_KERNEL_ARGS) {
  for (uint32_t process_index = blockIdx.x * blockDim.x + threadIdx.x;
       process_index < remaining_docs;
       process_index += blockDim.x * gridDim.x) {
    RemainingPayload remaining_payload;
    remaining_payload.payload = __ldg(
        reinterpret_cast<const int4* __restrict__>(remaining_payloads) +
        process_index);

    int32_t chunk_idx = __ldg(remaining_chunk_indices + process_index);
    const EMBEDDING_T* embeddings = embedding_list[chunk_idx];

    int32_t doc_index = remaining_payload.doc_index;
    int32_t write_index = remaining_payload.write_index;

    if constexpr (HAS_PER_EMB_SCALE) {
      auto divisor_value = *(per_emb_scale_list[chunk_idx] + doc_index);
      store<EMBEDDING_T, RETURN_T, DIM, FILTER, decltype(divisor_value)>(
          embeddings,
          queries,
          doc_index,
          remaining_payload.cluster_ids_row,
          divisor_value,
          results + write_index,
          indices + write_index,
          remaining_payload.filtering_mask);
    } else {
      store<EMBEDDING_T, RETURN_T, DIM, FILTER, int32_t>(
          embeddings,
          queries,
          doc_index,
          remaining_payload.cluster_ids_row,
          divisor_for_int8_scalar,
          results + write_index,
          indices + write_index,
          remaining_payload.filtering_mask);
    }
  }
}

} // namespace

template <typename EMBEDDING_TYPE, typename RETURN_TYPE, int DIM, bool FILTER>
inline bool try_launch_pipelined_kernel(
    bool use_pipelined_kernel,
    const Tensor& embeddings,
    const Tensor& queries,
    const Tensor& warp_payloads,
    Tensor& results,
    Tensor& indices,
    int32_t total_needed_warps,
    int32_t divisor_for_int8,
    const Tensor& remaining_payloads,
    int32_t remaining_docs,
    int64_t grid_size,
    int64_t remaining_grid_size) {
  if constexpr (
      std::is_same_v<EMBEDDING_TYPE, int8_t> &&
      std::is_same_v<RETURN_TYPE, c10::Half>) {
    if (use_pipelined_kernel) {
      constexpr int64_t kWarpsInBlock = kBlockSize / kWarpThreadCount;
      MAYBE_TORCH_DSA_KERNEL_LAUNCH(
          (process_cluster_v4_pipelined<DIM, FILTER>),
          grid_size,
          dim3(kWarpThreadCount, kWarpsInBlock),
          0,
          at::cuda::getCurrentCUDAStream(),
          embeddings.data_ptr<EMBEDDING_TYPE>(),
          queries.data_ptr<EMBEDDING_TYPE>(),
          reinterpret_cast<WarpPayload*>(warp_payloads.data_ptr<int8_t>()),
          results.mutable_data_ptr<RETURN_TYPE>(),
          indices.mutable_data_ptr<int32_t>(),
          total_needed_warps,
          divisor_for_int8);

      MAYBE_TORCH_DSA_KERNEL_LAUNCH(
          (process_cluster_remaining<
              EMBEDDING_TYPE,
              RETURN_TYPE,
              DIM,
              FILTER,
              int32_t>),
          remaining_grid_size,
          kBlockSize,
          0,
          at::cuda::getCurrentCUDAStream(),
          embeddings.data_ptr<EMBEDDING_TYPE>(),
          queries.data_ptr<EMBEDDING_TYPE>(),
          reinterpret_cast<RemainingPayload*>(
              remaining_payloads.data_ptr<int8_t>()),
          results.mutable_data_ptr<RETURN_TYPE>(),
          indices.mutable_data_ptr<int32_t>(),
          remaining_docs,
          divisor_for_int8);
      return true;
    }
  }
  return false;
}

std::tuple<Tensor, Tensor> fused_kmean_ann_run_payloads(
    const Tensor& selected_cluster_ids,
    const Tensor& embeddings,
    const Tensor& queries,
    int64_t max_tensor_size_per_row,
    int64_t invalid_index_value,
    int64_t divisor_for_int8,
    bool has_filtering_bitmask,
    const Tensor& warp_payloads,
    int64_t total_needed_warps,
    const Tensor& remaining_payloads,
    int64_t remaining_docs,
    const std::optional<Tensor>& per_embedding_scale,
    bool query_on_shared_mem = false) {
#if defined(USE_ROCM)
  query_on_shared_mem = true;
#endif
  int64_t total_result_size =
      selected_cluster_ids.size(0) * max_tensor_size_per_row;
  constexpr int64_t kWarpsInBlock = kBlockSize / kWarpThreadCount;
  auto grid_size = std::min(
      (total_needed_warps * kWarpThreadCount + kBlockSize - 1) / kBlockSize,
      192L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
  auto remaining_grid_size = std::min(
      (remaining_docs + kBlockSize - 1) / kBlockSize,
      192L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
  Tensor results;
  Tensor indices = at::full(
      {total_result_size},
      static_cast<int32_t>(invalid_index_value),
      c10::TensorOptions().dtype(at::kInt).device(
          selected_cluster_ids.device()));

#define INVOKE_KERNEL_WITH_TYPE_DIM_FILTER(                                   \
    EMBEDDING_TYPE, RETURN_TYPE, DIM, FILTER)                                 \
  {                                                                           \
    if (per_embedding_scale.has_value()) {                                    \
      if (query_on_shared_mem) {                                              \
        MAYBE_TORCH_DSA_KERNEL_LAUNCH(                                        \
            (process_cluster<                                                 \
                EMBEDDING_TYPE,                                               \
                RETURN_TYPE,                                                  \
                DIM,                                                          \
                FILTER,                                                       \
                c10::Half*,                                                   \
                true>),                                                       \
            grid_size,                                                        \
            dim3(kWarpThreadCount, kWarpsInBlock),                            \
            0,                                                                \
            at::cuda::getCurrentCUDAStream(),                                 \
            embeddings.data_ptr<EMBEDDING_TYPE>(),                            \
            queries.data_ptr<EMBEDDING_TYPE>(),                               \
            reinterpret_cast<WarpPayload*>(warp_payloads.data_ptr<int8_t>()), \
            results.mutable_data_ptr<RETURN_TYPE>(),                          \
            indices.mutable_data_ptr<int32_t>(),                              \
            static_cast<int32_t>(total_needed_warps),                         \
            per_embedding_scale.value().mutable_data_ptr<at::Half>());        \
      } else {                                                                \
        MAYBE_TORCH_DSA_KERNEL_LAUNCH(                                        \
            (process_cluster<                                                 \
                EMBEDDING_TYPE,                                               \
                RETURN_TYPE,                                                  \
                DIM,                                                          \
                FILTER,                                                       \
                c10::Half*,                                                   \
                false>),                                                      \
            grid_size,                                                        \
            dim3(kWarpThreadCount, kWarpsInBlock),                            \
            0,                                                                \
            at::cuda::getCurrentCUDAStream(),                                 \
            embeddings.data_ptr<EMBEDDING_TYPE>(),                            \
            queries.data_ptr<EMBEDDING_TYPE>(),                               \
            reinterpret_cast<WarpPayload*>(warp_payloads.data_ptr<int8_t>()), \
            results.mutable_data_ptr<RETURN_TYPE>(),                          \
            indices.mutable_data_ptr<int32_t>(),                              \
            static_cast<int32_t>(total_needed_warps),                         \
            per_embedding_scale.value().mutable_data_ptr<at::Half>());        \
      }                                                                       \
                                                                              \
      MAYBE_TORCH_DSA_KERNEL_LAUNCH(                                          \
          (process_cluster_remaining<                                         \
              EMBEDDING_TYPE,                                                 \
              RETURN_TYPE,                                                    \
              DIM,                                                            \
              FILTER,                                                         \
              c10::Half*>),                                                   \
          remaining_grid_size,                                                \
          kBlockSize,                                                         \
          0,                                                                  \
          at::cuda::getCurrentCUDAStream(),                                   \
          embeddings.data_ptr<EMBEDDING_TYPE>(),                              \
          queries.data_ptr<EMBEDDING_TYPE>(),                                 \
          reinterpret_cast<RemainingPayload*>(                                \
              remaining_payloads.data_ptr<int8_t>()),                         \
          results.mutable_data_ptr<RETURN_TYPE>(),                            \
          indices.mutable_data_ptr<int32_t>(),                                \
          static_cast<int32_t>(remaining_docs),                               \
          per_embedding_scale.value().mutable_data_ptr<at::Half>());          \
    } else {                                                                  \
      if (query_on_shared_mem) {                                              \
        MAYBE_TORCH_DSA_KERNEL_LAUNCH(                                        \
            (process_cluster<                                                 \
                EMBEDDING_TYPE,                                               \
                RETURN_TYPE,                                                  \
                DIM,                                                          \
                FILTER,                                                       \
                int32_t,                                                      \
                true>),                                                       \
            grid_size,                                                        \
            dim3(kWarpThreadCount, kWarpsInBlock),                            \
            0,                                                                \
            at::cuda::getCurrentCUDAStream(),                                 \
            embeddings.data_ptr<EMBEDDING_TYPE>(),                            \
            queries.data_ptr<EMBEDDING_TYPE>(),                               \
            reinterpret_cast<WarpPayload*>(warp_payloads.data_ptr<int8_t>()), \
            results.mutable_data_ptr<RETURN_TYPE>(),                          \
            indices.mutable_data_ptr<int32_t>(),                              \
            static_cast<int32_t>(total_needed_warps),                         \
            static_cast<int32_t>(divisor_for_int8));                          \
      } else {                                                                \
        MAYBE_TORCH_DSA_KERNEL_LAUNCH(                                        \
            (process_cluster<                                                 \
                EMBEDDING_TYPE,                                               \
                RETURN_TYPE,                                                  \
                DIM,                                                          \
                FILTER,                                                       \
                int32_t,                                                      \
                false>),                                                      \
            grid_size,                                                        \
            dim3(kWarpThreadCount, kWarpsInBlock),                            \
            0,                                                                \
            at::cuda::getCurrentCUDAStream(),                                 \
            embeddings.data_ptr<EMBEDDING_TYPE>(),                            \
            queries.data_ptr<EMBEDDING_TYPE>(),                               \
            reinterpret_cast<WarpPayload*>(warp_payloads.data_ptr<int8_t>()), \
            results.mutable_data_ptr<RETURN_TYPE>(),                          \
            indices.mutable_data_ptr<int32_t>(),                              \
            static_cast<int32_t>(total_needed_warps),                         \
            static_cast<int32_t>(divisor_for_int8));                          \
      }                                                                       \
                                                                              \
      MAYBE_TORCH_DSA_KERNEL_LAUNCH(                                          \
          (process_cluster_remaining<                                         \
              EMBEDDING_TYPE,                                                 \
              RETURN_TYPE,                                                    \
              DIM,                                                            \
              FILTER,                                                         \
              int32_t>),                                                      \
          remaining_grid_size,                                                \
          kBlockSize,                                                         \
          0,                                                                  \
          at::cuda::getCurrentCUDAStream(),                                   \
          embeddings.data_ptr<EMBEDDING_TYPE>(),                              \
          queries.data_ptr<EMBEDDING_TYPE>(),                                 \
          reinterpret_cast<RemainingPayload*>(                                \
              remaining_payloads.data_ptr<int8_t>()),                         \
          results.mutable_data_ptr<RETURN_TYPE>(),                            \
          indices.mutable_data_ptr<int32_t>(),                                \
          static_cast<int32_t>(remaining_docs),                               \
          static_cast<int32_t>(divisor_for_int8));                            \
    }                                                                         \
  }

#define INVOKE_KERNEL_WITH_TYPE_DIM(EMBEDDING_TYPE, RETURN_TYPE, DIM) \
  {                                                                   \
    if (has_filtering_bitmask) {                                      \
      INVOKE_KERNEL_WITH_TYPE_DIM_FILTER(                             \
          EMBEDDING_TYPE, RETURN_TYPE, DIM, true);                    \
    } else {                                                          \
      INVOKE_KERNEL_WITH_TYPE_DIM_FILTER(                             \
          EMBEDDING_TYPE, RETURN_TYPE, DIM, false);                   \
    }                                                                 \
  }

#define INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                        \
    EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, DIM)          \
  {                                                                \
    results = at::full(                                            \
        {total_result_size},                                       \
        cuda_mm_types<RETURN_TYPE>::C_MIN_NEG,                     \
        c10::TensorOptions()                                       \
            .dtype(RETURN_SCALAR_TYPE)                             \
            .device(selected_cluster_ids.device()));               \
    INVOKE_KERNEL_WITH_TYPE_DIM(EMBEDDING_TYPE, RETURN_TYPE, DIM); \
  }

#define INVOKE_KERNEL_WITH_TYPE(                                    \
    EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE)                \
  {                                                                 \
    switch (embeddings.size(1)) {                                   \
      case 16:                                                      \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 16);   \
        break;                                                      \
      case 32:                                                      \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 32);   \
        break;                                                      \
      case 64:                                                      \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 64);   \
        break;                                                      \
      case 96:                                                      \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 96);   \
        break;                                                      \
      case 128:                                                     \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 128);  \
        break;                                                      \
      case 192:                                                     \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 192);  \
        break;                                                      \
      case 256:                                                     \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 256);  \
        break;                                                      \
      case 384:                                                     \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 384);  \
        break;                                                      \
      case 512:                                                     \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 512);  \
        break;                                                      \
      case 768:                                                     \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 768);  \
        break;                                                      \
      case 1024:                                                    \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 1024); \
        break;                                                      \
      case 1280:                                                    \
        INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM(                         \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 1280); \
        break;                                                      \
      default:                                                      \
        TORCH_CHECK(false, "unsupported DIM ", embeddings.size(1)); \
    }                                                               \
  }

  if (embeddings.scalar_type() == at::kFloat) {
    INVOKE_KERNEL_WITH_TYPE(float, at::kFloat, float);
  } else if (embeddings.scalar_type() == at::kHalf) {
    INVOKE_KERNEL_WITH_TYPE(c10::Half, at::kHalf, c10::Half);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
  } else if (embeddings.scalar_type() == at::kBFloat16) {
    INVOKE_KERNEL_WITH_TYPE(c10::BFloat16, at::kBFloat16, c10::BFloat16);
#endif
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 610 || !defined(__CUDA_ARCH__))
  } else if (embeddings.scalar_type() == at::kChar) {
    const bool should_apply_divisor = per_embedding_scale.has_value() ||
        divisor_for_int8 != c_INVALID_DIVISOR_FOR_INT8;
    if (should_apply_divisor) {
      INVOKE_KERNEL_WITH_TYPE(int8_t, at::kHalf, c10::Half);
    } else {
      INVOKE_KERNEL_WITH_TYPE(int8_t, at::kInt, int32_t);
    }
#endif
  } else {
    TORCH_CHECK(false, "unsupported scalar type ", embeddings.scalar_type());
  }

#undef INVOKE_KERNEL_WITH_TYPE
#undef INVOKE_KERNEL_WITH_SCALAR_TYPE_DIM
#undef INVOKE_KERNEL_WITH_TYPE_DIM
#undef INVOKE_KERNEL_WITH_TYPE_DIM_FILTER

  return std::make_tuple(
      results.reshape({selected_cluster_ids.size(0), -1}),
      indices.reshape({selected_cluster_ids.size(0), -1}));
}

std::tuple<Tensor, Tensor> fused_kmean_ann_run_payloads_multi_emb(
    int64_t batch_size,
    const at::Device& device,
    const std::vector<Tensor>& list_embeddings,
    const Tensor& queries,
    int64_t max_tensor_size_per_row,
    int64_t invalid_index_value,
    int64_t divisor_for_int8,
    const Tensor& warp_payloads,
    int64_t total_needed_warps,
    const Tensor& remaining_payloads,
    int64_t remaining_docs,
    const Tensor& warp_chunk_indices,
    const Tensor& remaining_chunk_indices,
    const std::optional<std::vector<Tensor>>& list_per_embedding_scale) {
  TORCH_CHECK(!list_embeddings.empty());
  int64_t embedding_dim = list_embeddings[0].size(1);
  auto embedding_scalar_type = list_embeddings[0].scalar_type();

  int64_t total_result_size = batch_size * max_tensor_size_per_row;
  constexpr int64_t kWarpsInBlock = kBlockSize / kWarpThreadCount;
  auto grid_size = std::min(
      (total_needed_warps * kWarpThreadCount + kBlockSize - 1) / kBlockSize,
      192L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
  auto remaining_grid_size = std::min(
      (remaining_docs + kBlockSize - 1) / kBlockSize,
      192L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
  Tensor results;
  Tensor indices = at::full(
      {total_result_size},
      static_cast<int32_t>(invalid_index_value),
      c10::TensorOptions().dtype(at::kInt).device(device));

  const bool has_per_emb_scale = list_per_embedding_scale.has_value();

  std::vector<const void*> emb_ptrs;
  emb_ptrs.reserve(list_embeddings.size());
  for (const auto& emb : list_embeddings) {
    emb_ptrs.push_back(emb.data_ptr());
  }
  size_t emb_ptr_bytes = sizeof(const void*) * emb_ptrs.size();
  Tensor emb_ptr_tensor = at::empty(
      {static_cast<int64_t>(emb_ptr_bytes)},
      c10::TensorOptions().dtype(at::kChar).device(device),
      at::MemoryFormat::Contiguous);
  AT_CUDA_CHECK(cudaMemcpyAsync(
      emb_ptr_tensor.data_ptr<int8_t>(),
      emb_ptrs.data(),
      emb_ptr_bytes,
      cudaMemcpyHostToDevice,
      at::cuda::getCurrentCUDAStream()));

  std::vector<const void*> scale_ptrs;
  Tensor scale_ptr_tensor;
  const c10::Half* const* per_emb_scale_list_ptr = nullptr;
  if (has_per_emb_scale) {
    scale_ptrs.reserve(list_per_embedding_scale.value().size());
    for (const auto& scale : list_per_embedding_scale.value()) {
      scale_ptrs.push_back(scale.data_ptr());
    }
    size_t scale_ptr_bytes = sizeof(const void*) * scale_ptrs.size();
    scale_ptr_tensor = at::empty(
        {static_cast<int64_t>(scale_ptr_bytes)},
        c10::TensorOptions().dtype(at::kChar).device(device),
        at::MemoryFormat::Contiguous);
    AT_CUDA_CHECK(cudaMemcpyAsync(
        scale_ptr_tensor.data_ptr<int8_t>(),
        scale_ptrs.data(),
        scale_ptr_bytes,
        cudaMemcpyHostToDevice,
        at::cuda::getCurrentCUDAStream()));
    per_emb_scale_list_ptr = reinterpret_cast<const c10::Half* const*>(
        scale_ptr_tensor.data_ptr<int8_t>());
  }

#define INVOKE_MULTI_EMB_KERNEL_WITH_TYPE_DIM_FILTER(                       \
    EMBEDDING_TYPE, RETURN_TYPE, DIM, FILTER)                               \
  {                                                                         \
    if (has_per_emb_scale) {                                                \
      MAYBE_TORCH_DSA_KERNEL_LAUNCH(                                        \
          (process_cluster_multi_emb<                                       \
              EMBEDDING_TYPE,                                               \
              RETURN_TYPE,                                                  \
              DIM,                                                          \
              FILTER,                                                       \
              true>),                                                       \
          grid_size,                                                        \
          dim3(kWarpThreadCount, kWarpsInBlock),                            \
          0,                                                                \
          at::cuda::getCurrentCUDAStream(),                                 \
          reinterpret_cast<const EMBEDDING_TYPE* const*>(                   \
              emb_ptr_tensor.data_ptr<int8_t>()),                           \
          queries.data_ptr<EMBEDDING_TYPE>(),                               \
          reinterpret_cast<WarpPayload*>(warp_payloads.data_ptr<int8_t>()), \
          warp_chunk_indices.data_ptr<int32_t>(),                           \
          results.mutable_data_ptr<RETURN_TYPE>(),                          \
          indices.mutable_data_ptr<int32_t>(),                              \
          static_cast<int32_t>(total_needed_warps),                         \
          static_cast<int32_t>(divisor_for_int8),                           \
          per_emb_scale_list_ptr);                                          \
                                                                            \
      MAYBE_TORCH_DSA_KERNEL_LAUNCH(                                        \
          (process_cluster_remaining_multi_emb<                             \
              EMBEDDING_TYPE,                                               \
              RETURN_TYPE,                                                  \
              DIM,                                                          \
              FILTER,                                                       \
              true>),                                                       \
          remaining_grid_size,                                              \
          kBlockSize,                                                       \
          0,                                                                \
          at::cuda::getCurrentCUDAStream(),                                 \
          reinterpret_cast<const EMBEDDING_TYPE* const*>(                   \
              emb_ptr_tensor.data_ptr<int8_t>()),                           \
          queries.data_ptr<EMBEDDING_TYPE>(),                               \
          reinterpret_cast<RemainingPayload*>(                              \
              remaining_payloads.data_ptr<int8_t>()),                       \
          remaining_chunk_indices.data_ptr<int32_t>(),                      \
          results.mutable_data_ptr<RETURN_TYPE>(),                          \
          indices.mutable_data_ptr<int32_t>(),                              \
          static_cast<int32_t>(remaining_docs),                             \
          static_cast<int32_t>(divisor_for_int8),                           \
          per_emb_scale_list_ptr);                                          \
    } else {                                                                \
      MAYBE_TORCH_DSA_KERNEL_LAUNCH(                                        \
          (process_cluster_multi_emb<                                       \
              EMBEDDING_TYPE,                                               \
              RETURN_TYPE,                                                  \
              DIM,                                                          \
              FILTER,                                                       \
              false>),                                                      \
          grid_size,                                                        \
          dim3(kWarpThreadCount, kWarpsInBlock),                            \
          0,                                                                \
          at::cuda::getCurrentCUDAStream(),                                 \
          reinterpret_cast<const EMBEDDING_TYPE* const*>(                   \
              emb_ptr_tensor.data_ptr<int8_t>()),                           \
          queries.data_ptr<EMBEDDING_TYPE>(),                               \
          reinterpret_cast<WarpPayload*>(warp_payloads.data_ptr<int8_t>()), \
          warp_chunk_indices.data_ptr<int32_t>(),                           \
          results.mutable_data_ptr<RETURN_TYPE>(),                          \
          indices.mutable_data_ptr<int32_t>(),                              \
          static_cast<int32_t>(total_needed_warps),                         \
          static_cast<int32_t>(divisor_for_int8),                           \
          per_emb_scale_list_ptr);                                          \
                                                                            \
      MAYBE_TORCH_DSA_KERNEL_LAUNCH(                                        \
          (process_cluster_remaining_multi_emb<                             \
              EMBEDDING_TYPE,                                               \
              RETURN_TYPE,                                                  \
              DIM,                                                          \
              FILTER,                                                       \
              false>),                                                      \
          remaining_grid_size,                                              \
          kBlockSize,                                                       \
          0,                                                                \
          at::cuda::getCurrentCUDAStream(),                                 \
          reinterpret_cast<const EMBEDDING_TYPE* const*>(                   \
              emb_ptr_tensor.data_ptr<int8_t>()),                           \
          queries.data_ptr<EMBEDDING_TYPE>(),                               \
          reinterpret_cast<RemainingPayload*>(                              \
              remaining_payloads.data_ptr<int8_t>()),                       \
          remaining_chunk_indices.data_ptr<int32_t>(),                      \
          results.mutable_data_ptr<RETURN_TYPE>(),                          \
          indices.mutable_data_ptr<int32_t>(),                              \
          static_cast<int32_t>(remaining_docs),                             \
          static_cast<int32_t>(divisor_for_int8),                           \
          per_emb_scale_list_ptr);                                          \
    }                                                                       \
  }

#define INVOKE_MULTI_EMB_KERNEL_WITH_TYPE_DIM(    \
    EMBEDDING_TYPE, RETURN_TYPE, DIM)             \
  {                                               \
    INVOKE_MULTI_EMB_KERNEL_WITH_TYPE_DIM_FILTER( \
        EMBEDDING_TYPE, RETURN_TYPE, DIM, true);  \
  }

#define INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(                        \
    EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, DIM)                    \
  {                                                                          \
    results = at::full(                                                      \
        {total_result_size},                                                 \
        cuda_mm_types<RETURN_TYPE>::C_MIN_NEG,                               \
        c10::TensorOptions().dtype(RETURN_SCALAR_TYPE).device(device));      \
    INVOKE_MULTI_EMB_KERNEL_WITH_TYPE_DIM(EMBEDDING_TYPE, RETURN_TYPE, DIM); \
  }

#define INVOKE_MULTI_EMB_KERNEL_WITH_TYPE(                          \
    EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE)                \
  {                                                                 \
    switch (embedding_dim) {                                        \
      case 16:                                                      \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 16);   \
        break;                                                      \
      case 32:                                                      \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 32);   \
        break;                                                      \
      case 64:                                                      \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 64);   \
        break;                                                      \
      case 96:                                                      \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 96);   \
        break;                                                      \
      case 128:                                                     \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 128);  \
        break;                                                      \
      case 192:                                                     \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 192);  \
        break;                                                      \
      case 256:                                                     \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 256);  \
        break;                                                      \
      case 384:                                                     \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 384);  \
        break;                                                      \
      case 512:                                                     \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 512);  \
        break;                                                      \
      case 768:                                                     \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 768);  \
        break;                                                      \
      case 1024:                                                    \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 1024); \
        break;                                                      \
      case 1280:                                                    \
        INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM(               \
            EMBEDDING_TYPE, RETURN_SCALAR_TYPE, RETURN_TYPE, 1280); \
        break;                                                      \
      default:                                                      \
        TORCH_CHECK(false, "unsupported DIM ", embedding_dim);      \
    }                                                               \
  }

  if (embedding_scalar_type == at::kFloat) {
    INVOKE_MULTI_EMB_KERNEL_WITH_TYPE(float, at::kFloat, float);
  } else if (embedding_scalar_type == at::kHalf) {
    INVOKE_MULTI_EMB_KERNEL_WITH_TYPE(c10::Half, at::kHalf, c10::Half);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
  } else if (embedding_scalar_type == at::kBFloat16) {
    INVOKE_MULTI_EMB_KERNEL_WITH_TYPE(
        c10::BFloat16, at::kBFloat16, c10::BFloat16);
#endif
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 610 || !defined(__CUDA_ARCH__))
  } else if (embedding_scalar_type == at::kChar) {
    const bool should_apply_divisor =
        has_per_emb_scale || divisor_for_int8 != c_INVALID_DIVISOR_FOR_INT8;
    if (should_apply_divisor) {
      INVOKE_MULTI_EMB_KERNEL_WITH_TYPE(int8_t, at::kHalf, c10::Half);
    } else {
      INVOKE_MULTI_EMB_KERNEL_WITH_TYPE(int8_t, at::kInt, int32_t);
    }
#endif
  } else {
    TORCH_CHECK(false, "unsupported scalar type ", embedding_scalar_type);
  }

#undef INVOKE_MULTI_EMB_KERNEL_WITH_TYPE
#undef INVOKE_MULTI_EMB_KERNEL_WITH_SCALAR_TYPE_DIM
#undef INVOKE_MULTI_EMB_KERNEL_WITH_TYPE_DIM
#undef INVOKE_MULTI_EMB_KERNEL_WITH_TYPE_DIM_FILTER

  return std::make_tuple(
      results.reshape({batch_size, -1}), indices.reshape({batch_size, -1}));
}

std::tuple<Tensor, Tensor> fused_kmean_ann_cuda(
    const Tensor& cluster_offsets,
    const Tensor& cluster_ids,
    const Tensor& cluster_length,
    const Tensor& embeddings,
    const Tensor& queries,
    int64_t max_tensor_size_per_row,
    const std::optional<Tensor>& filtering_bit_mask,
    int64_t invalid_index_value,
    int64_t divisor_for_int8,
    const std::optional<Tensor>& filtering_bit_index,
    const std::optional<Tensor>& per_embedding_scale) {
  TORCH_CHECK(cluster_offsets.is_cuda());
  TORCH_CHECK(cluster_offsets.is_contiguous());
  TORCH_CHECK(cluster_offsets.dim() == 1);

  TORCH_CHECK(cluster_ids.is_cuda());
  TORCH_CHECK(cluster_ids.is_contiguous());
  TORCH_CHECK(cluster_ids.dim() == 2);

  TORCH_CHECK(cluster_length.is_cuda());
  TORCH_CHECK(cluster_length.is_contiguous());
  TORCH_CHECK(cluster_length.dim() == 2);

  TORCH_CHECK(embeddings.is_cuda());
  TORCH_CHECK(embeddings.is_contiguous());
  TORCH_CHECK(embeddings.dim() == 2);

  TORCH_CHECK(queries.is_cuda());
  TORCH_CHECK(queries.is_contiguous());
  TORCH_CHECK(queries.dim() == 2);

  TORCH_CHECK(cluster_ids.size(0) == cluster_length.size(0));
  TORCH_CHECK(cluster_ids.size(0) == queries.size(0));
  TORCH_CHECK(cluster_ids.size(1) == cluster_length.size(1));
  TORCH_CHECK(embeddings.size(1) == queries.size(1));

  const uint64_t* filtering_bitmask_ptr = nullptr;
  int64_t filtering_bitmask_column = 0;
  const uint64_t* filtering_bitmask_index_ptr = nullptr;
  if (filtering_bit_mask) {
    TORCH_CHECK(filtering_bit_mask->is_cuda());
    if (filtering_bit_index) {
      TORCH_CHECK(filtering_bit_index->size(0) == cluster_ids.size(0));
      TORCH_CHECK(filtering_bit_index->is_cuda());
      TORCH_CHECK(filtering_bit_index->dim() == 1);
      TORCH_CHECK(filtering_bit_index->size(0) >= filtering_bit_mask->size(0));
      filtering_bitmask_index_ptr = reinterpret_cast<const uint64_t*>(
          filtering_bit_index->data_ptr<int64_t>());
    } else {
      TORCH_CHECK(filtering_bit_mask->size(0) == cluster_ids.size(0));
    }
    TORCH_CHECK(filtering_bit_mask->scalar_type() == at::kLong);
    TORCH_CHECK(filtering_bit_mask->is_contiguous());
    TORCH_CHECK(filtering_bit_mask->dim() == 2);
    filtering_bitmask_ptr = reinterpret_cast<const uint64_t*>(
        filtering_bit_mask->data_ptr<int64_t>());
    filtering_bitmask_column = filtering_bit_mask->size(1);
  }

  if (per_embedding_scale.has_value()) {
    TORCH_CHECK(
        per_embedding_scale->is_cuda(), "per_embedding_scale is not on GPU");
    TORCH_CHECK(
        per_embedding_scale->is_contiguous(),
        "per_embedding_scale is not contiguous");
    TORCH_CHECK(
        per_embedding_scale->dim() == 1, "per_embedding_scale is not 1D");
    TORCH_CHECK(
        per_embedding_scale->size(0) == embeddings.size(0),
        "per_embedding_scale size does not match embedding size");
    TORCH_CHECK(
        per_embedding_scale->scalar_type() == at::kHalf,
        "per_embedding_scale is not half");
  }

  max_tensor_size_per_row = (max_tensor_size_per_row + kWarpThreadCount - 1) /
      kWarpThreadCount * kWarpThreadCount;

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cluster_ids.get_device());

  auto
      [cluster_warp_size,
       cluster_warp_rounded_length,
       cluster_remaining_length] = round_cluster_to_warp(cluster_length);

  TORCH_CHECK(cluster_warp_rounded_length.dim() == 2);
  TORCH_CHECK(cluster_remaining_length.dim() == 1);
  auto cluster_warp_rounded_length_cumsum =
      cluster_warp_rounded_length.cumsum(/*dim=*/1, /*dtype=*/at::kInt);
  auto cluster_remaining_length_cumsum =
      cluster_remaining_length.cumsum(/*dim=*/0, /*dtype=*/at::kInt);

  auto [warp_payloads, total_needed_warps] = generate_warp_payload(
      cluster_warp_size,
      cluster_offsets,
      cluster_ids,
      cluster_warp_rounded_length_cumsum,
      filtering_bitmask_ptr,
      filtering_bitmask_column,
      filtering_bitmask_index_ptr,
      max_tensor_size_per_row);

  auto [remaining_payloads, remaining_docs] = generate_remaining_payload(
      cluster_length,
      cluster_offsets,
      cluster_ids,
      cluster_warp_rounded_length_cumsum,
      cluster_remaining_length_cumsum,
      filtering_bitmask_ptr,
      filtering_bitmask_column,
      filtering_bitmask_index_ptr,
      max_tensor_size_per_row);

  return fused_kmean_ann_run_payloads(
      cluster_ids,
      embeddings,
      queries,
      max_tensor_size_per_row,
      invalid_index_value,
      divisor_for_int8,
      filtering_bitmask_ptr != nullptr,
      warp_payloads,
      total_needed_warps,
      remaining_payloads,
      remaining_docs,
      per_embedding_scale,
      /*query_on_shared_mem=*/false);
}

std::tuple<Tensor, Tensor> fused_kmean_ann_cuda_with_partial_masks(
    const Tensor& cluster_offsets,
    const Tensor& cluster_ids,
    const Tensor& cluster_length,
    const Tensor& embeddings,
    const Tensor& queries,
    int64_t max_tensor_size_per_row,
    const Tensor& partial_mask_column_counts_cumsum,
    const Tensor& partial_mask_first_item_offset_in_column,
    const Tensor& partial_mask_column_results,
    int64_t invalid_index_value,
    int64_t divisor_for_int8,
    const std::optional<Tensor>& filtering_bit_index,
    const std::optional<Tensor>& per_embedding_scale,
    const std::optional<Tensor>& cluster_warp_size_opt,
    const std::optional<Tensor>& cluster_warp_rounded_length_cumsum_opt,
    const std::optional<Tensor>& cluster_remaining_length_cumsum_opt,
    const std::optional<Tensor>& cluster_warp_size_cumsum_opt,
    int64_t total_cluster_rounded_warps,
    int64_t total_cluster_remaining_warps) {
  TORCH_CHECK(cluster_offsets.is_cuda());
  TORCH_CHECK(cluster_offsets.is_contiguous());
  TORCH_CHECK(cluster_offsets.dim() == 1);

  TORCH_CHECK(cluster_ids.is_cuda());
  TORCH_CHECK(cluster_ids.is_contiguous());
  TORCH_CHECK(cluster_ids.dim() == 2);

  TORCH_CHECK(cluster_length.is_cuda());
  TORCH_CHECK(cluster_length.is_contiguous());
  TORCH_CHECK(cluster_length.dim() == 2);

  TORCH_CHECK(embeddings.is_cuda());
  TORCH_CHECK(embeddings.is_contiguous());
  TORCH_CHECK(embeddings.dim() == 2);

  TORCH_CHECK(queries.is_cuda());
  TORCH_CHECK(queries.is_contiguous());
  TORCH_CHECK(queries.dim() == 2);

  TORCH_CHECK(cluster_ids.size(0) == cluster_length.size(0));
  TORCH_CHECK(cluster_ids.size(0) == queries.size(0));
  TORCH_CHECK(cluster_ids.size(1) == cluster_length.size(1));
  TORCH_CHECK(embeddings.size(1) == queries.size(1));

  TORCH_CHECK(partial_mask_column_counts_cumsum.is_cuda());
  TORCH_CHECK(partial_mask_first_item_offset_in_column.is_cuda());
  TORCH_CHECK(partial_mask_column_results.is_cuda());
  TORCH_CHECK(partial_mask_column_counts_cumsum.dim() == 1);
  TORCH_CHECK(partial_mask_first_item_offset_in_column.dim() == 1);
  TORCH_CHECK(partial_mask_column_results.dim() == 1);
  TORCH_CHECK(partial_mask_column_results.scalar_type() == at::kLong);
  TORCH_CHECK(partial_mask_column_results.is_contiguous());
  TORCH_CHECK(
      partial_mask_column_counts_cumsum.size(0) ==
      partial_mask_first_item_offset_in_column.size(0));

  const int32_t* partial_mask_column_counts_cumsum_ptr =
      partial_mask_column_counts_cumsum.data_ptr<int32_t>();
  const int8_t* partial_mask_first_item_offset_in_column_ptr =
      partial_mask_first_item_offset_in_column.data_ptr<int8_t>();
  const uint64_t* partial_mask_column_results_ptr =
      reinterpret_cast<const uint64_t*>(
          partial_mask_column_results.data_ptr<int64_t>());

  const uint64_t* filtering_bitmask_index_ptr = nullptr;
  if (filtering_bit_index) {
    TORCH_CHECK(filtering_bit_index->is_cuda());
    TORCH_CHECK(filtering_bit_index->size(0) == cluster_ids.size(0));
    TORCH_CHECK(filtering_bit_index->dim() == 1);
    filtering_bitmask_index_ptr = reinterpret_cast<const uint64_t*>(
        filtering_bit_index->data_ptr<int64_t>());
  } else {
    TORCH_CHECK(
        partial_mask_first_item_offset_in_column.size(0) ==
        cluster_ids.numel());
  }

  if (per_embedding_scale.has_value()) {
    TORCH_CHECK(per_embedding_scale->is_cuda());
    TORCH_CHECK(per_embedding_scale->is_contiguous());
    TORCH_CHECK(per_embedding_scale->dim() == 1);
    TORCH_CHECK(per_embedding_scale->size(0) == embeddings.size(0));
    TORCH_CHECK(per_embedding_scale->scalar_type() == at::kHalf);
  }

  max_tensor_size_per_row = (max_tensor_size_per_row + kWarpThreadCount - 1) /
      kWarpThreadCount * kWarpThreadCount;

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cluster_ids.get_device());

  Tensor cluster_warp_size_tensor;
  Tensor cluster_warp_rounded_length_cumsum_tensor;
  Tensor cluster_remaining_length_cumsum_tensor;
  Tensor cluster_warp_size_cumsum_tensor;

  if (cluster_warp_size_opt.has_value() &&
      cluster_warp_rounded_length_cumsum_opt.has_value() &&
      cluster_remaining_length_cumsum_opt.has_value() &&
      cluster_warp_size_cumsum_opt.has_value()) {
    cluster_warp_size_tensor = cluster_warp_size_opt.value();
    cluster_warp_rounded_length_cumsum_tensor =
        cluster_warp_rounded_length_cumsum_opt.value();
    cluster_remaining_length_cumsum_tensor =
        cluster_remaining_length_cumsum_opt.value();
    cluster_warp_size_cumsum_tensor = cluster_warp_size_cumsum_opt.value();
  } else {
    auto
        [cluster_warp_size_computed,
         cluster_warp_rounded_length,
         cluster_remaining_length] = round_cluster_to_warp(cluster_length);

    TORCH_CHECK(cluster_warp_rounded_length.dim() == 2);
    TORCH_CHECK(cluster_remaining_length.dim() == 1);
    cluster_warp_size_tensor = cluster_warp_size_computed;
    cluster_warp_rounded_length_cumsum_tensor =
        cluster_warp_rounded_length.cumsum(/*dim=*/1, /*dtype=*/at::kInt);
    cluster_remaining_length_cumsum_tensor =
        cluster_remaining_length.cumsum(/*dim=*/0, /*dtype=*/at::kInt);
    cluster_warp_size_cumsum_tensor =
        cluster_warp_size_tensor.cumsum(/*dim=*/0, /*dtype=*/at::kInt);
  }

  int32_t total_warps_computed = (total_cluster_rounded_warps > 0)
      ? total_cluster_rounded_warps
      : cluster_warp_size_cumsum_tensor[-1].item<int32_t>();

  auto [warp_payloads, total_needed_warps] =
      generate_warp_payload_with_partial_masks(
          cluster_warp_size_tensor,
          cluster_offsets,
          cluster_ids,
          cluster_warp_rounded_length_cumsum_tensor,
          cluster_warp_size_cumsum_tensor,
          total_warps_computed,
          partial_mask_column_counts_cumsum_ptr,
          partial_mask_first_item_offset_in_column_ptr,
          partial_mask_column_results_ptr,
          filtering_bitmask_index_ptr,
          max_tensor_size_per_row);

  int32_t total_remaining_docs_computed = (total_cluster_remaining_warps > 0)
      ? total_cluster_remaining_warps
      : cluster_remaining_length_cumsum_tensor[-1].item<int32_t>();
  auto [remaining_payloads, remaining_docs] =
      generate_remaining_payload_with_partial_masks(
          cluster_length,
          cluster_offsets,
          cluster_ids,
          cluster_warp_rounded_length_cumsum_tensor,
          cluster_remaining_length_cumsum_tensor,
          total_remaining_docs_computed,
          partial_mask_column_counts_cumsum_ptr,
          partial_mask_first_item_offset_in_column_ptr,
          partial_mask_column_results_ptr,
          filtering_bitmask_index_ptr,
          max_tensor_size_per_row);

  return fused_kmean_ann_run_payloads(
      cluster_ids,
      embeddings,
      queries,
      max_tensor_size_per_row,
      invalid_index_value,
      divisor_for_int8,
      true,
      warp_payloads,
      total_needed_warps,
      remaining_payloads,
      remaining_docs,
      per_embedding_scale);
}

std::tuple<std::vector<Tensor>, std::vector<Tensor>>
fused_kmean_ann_with_partial_masks_multiple_cuda(
    const Tensor& queries,
    const std::vector<Tensor>& list_cluster_offsets,
    const std::vector<Tensor>& list_selected_cluster_ids,
    const std::vector<Tensor>& list_selected_cluster_lengths,
    const std::vector<Tensor>& list_embeddings,
    const std::vector<int64_t>& list_max_tensor_size_per_row,
    const std::vector<Tensor>& list_partial_mask_column_counts_cumsum,
    const std::vector<Tensor>& list_partial_mask_first_item_offset_in_column,
    const std::vector<Tensor>& list_partial_mask_column_results,
    int64_t invalid_index_value,
    int64_t divisor_for_int8,
    const std::optional<Tensor>& filtering_bit_index,
    const std::optional<std::vector<Tensor>>& list_per_embedding_scale,
    bool fuse) {
  const auto n = list_cluster_offsets.size();
  TORCH_CHECK(
      list_selected_cluster_ids.size() == n,
      "list_selected_cluster_ids must have the same size as list_cluster_offsets");
  TORCH_CHECK(
      list_selected_cluster_lengths.size() == n,
      "list_selected_cluster_lengths must have the same size as list_cluster_offsets");
  TORCH_CHECK(
      list_embeddings.size() == n,
      "list_embeddings must have the same size as list_cluster_offsets");
  TORCH_CHECK(
      list_max_tensor_size_per_row.size() == n,
      "list_max_tensor_size_per_row must have the same size as list_cluster_offsets");
  TORCH_CHECK(
      list_partial_mask_column_counts_cumsum.size() == n,
      "list_partial_mask_column_counts_cumsum must have the same size as list_cluster_offsets");
  TORCH_CHECK(
      list_partial_mask_first_item_offset_in_column.size() == n,
      "list_partial_mask_first_item_offset_in_column must have the same size as list_cluster_offsets");
  TORCH_CHECK(
      list_partial_mask_column_results.size() == n,
      "list_partial_mask_column_results must have the same size as list_cluster_offsets");
  const bool has_per_embedding_scale = list_per_embedding_scale.has_value();
  if (has_per_embedding_scale) {
    TORCH_CHECK(
        list_per_embedding_scale->size() == n,
        "list_per_embedding_scale must have the same size as list_cluster_offsets");
  }

  if (!fuse) {
    auto list_of_scores = std::vector<Tensor>();
    auto list_of_indices = std::vector<Tensor>();
    list_of_scores.reserve(n);
    list_of_indices.reserve(n);

    for (size_t i = 0; i < n; ++i) {
      const std::optional<Tensor> per_embedding_scale = has_per_embedding_scale
          ? std::optional<Tensor>(list_per_embedding_scale.value()[i])
          : std::nullopt;
      auto [scores, indices] = fused_kmean_ann_cuda_with_partial_masks(
          list_cluster_offsets[i],
          list_selected_cluster_ids[i],
          list_selected_cluster_lengths[i],
          list_embeddings[i],
          queries,
          list_max_tensor_size_per_row[i],
          list_partial_mask_column_counts_cumsum[i],
          list_partial_mask_first_item_offset_in_column[i],
          list_partial_mask_column_results[i],
          invalid_index_value,
          divisor_for_int8,
          filtering_bit_index,
          per_embedding_scale,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          0,
          0);
      list_of_scores.push_back(std::move(scores));
      list_of_indices.push_back(std::move(indices));
    }

    return std::make_tuple(
        std::move(list_of_scores), std::move(list_of_indices));
  }

  TORCH_CHECK(n > 0, "Expected at least one index set");

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(list_selected_cluster_ids[0].get_device());
  const at::Device device = list_selected_cluster_ids[0].device();

  const int64_t batch_size = queries.size(0);

  std::vector<int64_t> per_index_rounded_max_sizes;
  per_index_rounded_max_sizes.reserve(n);
  int64_t combined_max_size = 0;
  for (size_t i = 0; i < n; ++i) {
    int64_t rounded_max =
        (list_max_tensor_size_per_row[i] + kWarpThreadCount - 1) /
        kWarpThreadCount * kWarpThreadCount;
    per_index_rounded_max_sizes.push_back(rounded_max);
    combined_max_size += rounded_max;
  }

  const uint64_t* filtering_bitmask_index_ptr = nullptr;
  if (filtering_bit_index) {
    TORCH_CHECK(filtering_bit_index->is_cuda());
    filtering_bitmask_index_ptr = reinterpret_cast<const uint64_t*>(
        filtering_bit_index->data_ptr<int64_t>());
  }

  struct PerIndexWarpInfo {
    Tensor cluster_warp_size;
    Tensor cluster_warp_rounded_length_cumsum;
    Tensor cluster_remaining_length_cumsum;
    Tensor cluster_warp_size_cumsum;
    const int32_t* partial_mask_column_counts_cumsum_ptr;
    const int8_t* partial_mask_first_item_offset_in_column_ptr;
    const uint64_t* partial_mask_column_results_ptr;
  };
  std::vector<PerIndexWarpInfo> per_index_info;
  per_index_info.reserve(n);

  std::vector<Tensor> warp_total_slices;
  std::vector<Tensor> remaining_total_slices;
  warp_total_slices.reserve(n);
  remaining_total_slices.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    TORCH_CHECK(list_cluster_offsets[i].is_cuda());
    TORCH_CHECK(list_cluster_offsets[i].is_contiguous());
    TORCH_CHECK(list_selected_cluster_ids[i].is_cuda());
    TORCH_CHECK(list_selected_cluster_ids[i].is_contiguous());
    TORCH_CHECK(list_selected_cluster_ids[i].dim() == 2);
    TORCH_CHECK(list_selected_cluster_lengths[i].is_cuda());
    TORCH_CHECK(list_selected_cluster_lengths[i].is_contiguous());
    TORCH_CHECK(list_selected_cluster_lengths[i].dim() == 2);
    TORCH_CHECK(list_embeddings[i].is_cuda());
    TORCH_CHECK(list_embeddings[i].is_contiguous());
    TORCH_CHECK(list_embeddings[i].dim() == 2);
    TORCH_CHECK(
        list_selected_cluster_ids[i].size(0) == batch_size,
        "All index sets must have the same batch size");
    TORCH_CHECK(list_partial_mask_column_counts_cumsum[i].is_cuda());
    TORCH_CHECK(list_partial_mask_first_item_offset_in_column[i].is_cuda());
    TORCH_CHECK(list_partial_mask_column_results[i].is_cuda());

    const int32_t* partial_mask_counts_ptr =
        list_partial_mask_column_counts_cumsum[i].data_ptr<int32_t>();
    const int8_t* partial_mask_offset_ptr =
        list_partial_mask_first_item_offset_in_column[i].data_ptr<int8_t>();
    const uint64_t* partial_mask_results_ptr =
        reinterpret_cast<const uint64_t*>(
            list_partial_mask_column_results[i].data_ptr<int64_t>());

    auto
        [cluster_warp_size,
         cluster_warp_rounded_length,
         cluster_remaining_length] =
            round_cluster_to_warp(list_selected_cluster_lengths[i]);

    auto cluster_warp_rounded_length_cumsum =
        cluster_warp_rounded_length.cumsum(/*dim=*/1, /*dtype=*/at::kInt);
    auto cluster_remaining_length_cumsum =
        cluster_remaining_length.cumsum(/*dim=*/0, /*dtype=*/at::kInt);
    auto cluster_warp_size_cumsum =
        cluster_warp_size.cumsum(/*dim=*/0, /*dtype=*/at::kInt);

    per_index_info.push_back(
        {cluster_warp_size,
         cluster_warp_rounded_length_cumsum,
         cluster_remaining_length_cumsum,
         cluster_warp_size_cumsum,
         partial_mask_counts_ptr,
         partial_mask_offset_ptr,
         partial_mask_results_ptr});

    warp_total_slices.push_back(cluster_warp_size_cumsum.narrow(
        0, cluster_warp_size_cumsum.size(0) - 1, 1));
    remaining_total_slices.push_back(cluster_remaining_length_cumsum.narrow(
        0, cluster_remaining_length_cumsum.size(0) - 1, 1));
  }

  Tensor all_totals_gpu =
      at::cat({at::cat(warp_total_slices), at::cat(remaining_total_slices)});
  Tensor all_totals_cpu = all_totals_gpu.cpu();
  auto* totals_ptr = all_totals_cpu.data_ptr<int32_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  auto int_opts = c10::TensorOptions().dtype(at::kInt).device(device);

  std::vector<int64_t> per_index_warp_counts;
  std::vector<int64_t> per_index_remaining_counts;
  std::vector<int32_t> per_index_clusters_per_row_vec;
  std::vector<int32_t> per_index_write_base_vec;
  std::vector<int64_t> per_index_cluster_counts;
  std::vector<int32_t> per_index_remaining_output_offset_vec;
  per_index_warp_counts.reserve(n);
  per_index_remaining_counts.reserve(n);
  per_index_clusters_per_row_vec.reserve(n);
  per_index_write_base_vec.reserve(n);
  per_index_cluster_counts.reserve(n);
  per_index_remaining_output_offset_vec.reserve(n);

  int64_t total_warps = 0;
  int64_t total_remaining = 0;
  int64_t total_clusters = 0;
  int32_t write_base_offset = 0;
  int32_t remaining_out_offset = 0;

  for (size_t i = 0; i < n; ++i) {
    int32_t warp_count = totals_ptr[i];
    int32_t remaining_count = totals_ptr[n + i];
    per_index_warp_counts.push_back(warp_count);
    per_index_remaining_counts.push_back(remaining_count);
    per_index_clusters_per_row_vec.push_back(
        static_cast<int32_t>(list_selected_cluster_ids[i].size(1)));
    per_index_write_base_vec.push_back(write_base_offset);
    per_index_cluster_counts.push_back(
        per_index_info[i].cluster_warp_size.numel());
    per_index_remaining_output_offset_vec.push_back(remaining_out_offset);
    total_warps += warp_count;
    total_remaining += remaining_count;
    total_clusters += per_index_info[i].cluster_warp_size.numel();
    write_base_offset += static_cast<int32_t>(per_index_rounded_max_sizes[i]);
    remaining_out_offset += remaining_count;
  }

  std::vector<Tensor> cluster_warp_size_parts;
  cluster_warp_size_parts.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    cluster_warp_size_parts.push_back(per_index_info[i].cluster_warp_size);
  }
  Tensor global_cluster_warp_sizes = at::cat(cluster_warp_size_parts);
  Tensor global_cws_cumsum =
      global_cluster_warp_sizes.cumsum(/*dim=*/0, /*dtype=*/at::kInt);

  Tensor global_assigned = torch::arange(total_clusters, int_opts)
                               .repeat_interleave(global_cluster_warp_sizes);

  Tensor warp_counts_tensor = at::tensor(per_index_warp_counts, int_opts);
  Tensor warp_chunk_indices = torch::arange(static_cast<int64_t>(n), int_opts)
                                  .repeat_interleave(warp_counts_tensor);

  Tensor cluster_counts_tensor = at::tensor(per_index_cluster_counts, int_opts);
  Tensor cluster_counts_cumsum = cluster_counts_tensor.cumsum(0, at::kInt);
  Tensor cluster_chunk_indices =
      torch::arange(static_cast<int64_t>(n), int_opts)
          .repeat_interleave(cluster_counts_tensor);

  auto upload_ptrs = [&](const auto& ptrs) {
    size_t bytes = sizeof(ptrs[0]) * ptrs.size();
    Tensor t = at::empty(
        {static_cast<int64_t>(bytes)},
        c10::TensorOptions().dtype(at::kChar).device(device));
    AT_CUDA_CHECK(cudaMemcpyAsync(
        t.data_ptr<int8_t>(),
        ptrs.data(),
        bytes,
        cudaMemcpyHostToDevice,
        stream));
    return t;
  };

  std::vector<const int64_t*> cluster_offsets_ptrs, cluster_ids_ptrs,
      cluster_lengths_ptrs;
  std::vector<const int32_t*> warp_rounded_length_cumsum_ptrs,
      remaining_length_cumsum_ptrs, mask_column_counts_cumsum_ptrs;
  std::vector<const int8_t*> mask_first_item_offset_ptrs;
  std::vector<const uint64_t*> mask_column_results_ptrs;
  for (size_t i = 0; i < n; ++i) {
    cluster_offsets_ptrs.push_back(list_cluster_offsets[i].data_ptr<int64_t>());
    cluster_ids_ptrs.push_back(
        list_selected_cluster_ids[i].data_ptr<int64_t>());
    cluster_lengths_ptrs.push_back(
        list_selected_cluster_lengths[i].data_ptr<int64_t>());
    warp_rounded_length_cumsum_ptrs.push_back(
        per_index_info[i]
            .cluster_warp_rounded_length_cumsum.data_ptr<int32_t>());
    remaining_length_cumsum_ptrs.push_back(
        per_index_info[i].cluster_remaining_length_cumsum.data_ptr<int32_t>());
    mask_column_counts_cumsum_ptrs.push_back(
        per_index_info[i].partial_mask_column_counts_cumsum_ptr);
    mask_first_item_offset_ptrs.push_back(
        per_index_info[i].partial_mask_first_item_offset_in_column_ptr);
    mask_column_results_ptrs.push_back(
        per_index_info[i].partial_mask_column_results_ptr);
  }
  Tensor cluster_offsets_gpu = upload_ptrs(cluster_offsets_ptrs);
  Tensor cluster_ids_gpu = upload_ptrs(cluster_ids_ptrs);
  Tensor cluster_lengths_gpu = upload_ptrs(cluster_lengths_ptrs);
  Tensor warp_rounded_length_cumsum_gpu =
      upload_ptrs(warp_rounded_length_cumsum_ptrs);
  Tensor remaining_length_cumsum_gpu =
      upload_ptrs(remaining_length_cumsum_ptrs);
  Tensor mask_column_counts_gpu = upload_ptrs(mask_column_counts_cumsum_ptrs);
  Tensor mask_first_item_offset_gpu = upload_ptrs(mask_first_item_offset_ptrs);
  Tensor mask_column_results_gpu = upload_ptrs(mask_column_results_ptrs);

  Tensor per_chunk_clusters_per_row =
      at::tensor(per_index_clusters_per_row_vec, int_opts);
  Tensor per_chunk_write_base = at::tensor(per_index_write_base_vec, int_opts);
  Tensor per_chunk_remaining_output_offset =
      at::tensor(per_index_remaining_output_offset_vec, int_opts);

  Tensor combined_warp_payloads = at::empty(
      {total_warps * static_cast<int64_t>(sizeof(WarpPayload))},
      c10::TensorOptions().dtype(at::kChar).device(device));
  Tensor combined_remaining_payloads = at::empty(
      {total_remaining * static_cast<int64_t>(sizeof(RemainingPayload))},
      c10::TensorOptions().dtype(at::kChar).device(device));

  {
    auto grid = std::min(
        (total_warps + kBlockSize - 1) / kBlockSize,
        192L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
    MAYBE_TORCH_DSA_KERNEL_LAUNCH(
        generate_warp_payload_multi_chunk_kernel,
        grid,
        kBlockSize,
        0,
        stream,
        reinterpret_cast<const int64_t* const*>(
            cluster_offsets_gpu.data_ptr<int8_t>()),
        reinterpret_cast<const int64_t* const*>(
            cluster_ids_gpu.data_ptr<int8_t>()),
        reinterpret_cast<const int32_t* const*>(
            warp_rounded_length_cumsum_gpu.data_ptr<int8_t>()),
        global_assigned.data_ptr<int32_t>(),
        global_cws_cumsum.data_ptr<int32_t>(),
        warp_chunk_indices.data_ptr<int32_t>(),
        cluster_counts_cumsum.data_ptr<int32_t>(),
        reinterpret_cast<WarpPayload*>(
            combined_warp_payloads.mutable_data_ptr<int8_t>()),
        reinterpret_cast<const int32_t* const*>(
            mask_column_counts_gpu.data_ptr<int8_t>()),
        reinterpret_cast<const int8_t* const*>(
            mask_first_item_offset_gpu.data_ptr<int8_t>()),
        reinterpret_cast<const uint64_t* const*>(
            mask_column_results_gpu.data_ptr<int8_t>()),
        filtering_bitmask_index_ptr,
        per_chunk_clusters_per_row.data_ptr<int32_t>(),
        per_chunk_write_base.data_ptr<int32_t>(),
        static_cast<int32_t>(combined_max_size),
        total_warps);
  }

  {
    auto grid = std::min(
        (total_clusters + kBlockSize - 1) / kBlockSize,
        192L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
    MAYBE_TORCH_DSA_KERNEL_LAUNCH(
        generate_remaining_payload_multi_chunk_kernel,
        grid,
        kBlockSize,
        0,
        stream,
        reinterpret_cast<const int64_t* const*>(
            cluster_offsets_gpu.data_ptr<int8_t>()),
        reinterpret_cast<const int64_t* const*>(
            cluster_ids_gpu.data_ptr<int8_t>()),
        reinterpret_cast<const int64_t* const*>(
            cluster_lengths_gpu.data_ptr<int8_t>()),
        reinterpret_cast<const int32_t* const*>(
            warp_rounded_length_cumsum_gpu.data_ptr<int8_t>()),
        reinterpret_cast<const int32_t* const*>(
            remaining_length_cumsum_gpu.data_ptr<int8_t>()),
        cluster_chunk_indices.data_ptr<int32_t>(),
        cluster_counts_cumsum.data_ptr<int32_t>(),
        per_chunk_remaining_output_offset.data_ptr<int32_t>(),
        reinterpret_cast<RemainingPayload*>(
            combined_remaining_payloads.mutable_data_ptr<int8_t>()),
        reinterpret_cast<const int32_t* const*>(
            mask_column_counts_gpu.data_ptr<int8_t>()),
        reinterpret_cast<const int8_t* const*>(
            mask_first_item_offset_gpu.data_ptr<int8_t>()),
        reinterpret_cast<const uint64_t* const*>(
            mask_column_results_gpu.data_ptr<int8_t>()),
        filtering_bitmask_index_ptr,
        per_chunk_clusters_per_row.data_ptr<int32_t>(),
        per_chunk_write_base.data_ptr<int32_t>(),
        static_cast<int32_t>(combined_max_size),
        total_clusters);
  }

  Tensor remaining_counts = at::tensor(per_index_remaining_counts, int_opts);
  Tensor remaining_chunk_indices =
      torch::arange(static_cast<int64_t>(n), int_opts)
          .repeat_interleave(remaining_counts);

  auto [combined_scores, combined_indices] =
      fused_kmean_ann_run_payloads_multi_emb(
          batch_size,
          device,
          list_embeddings,
          queries,
          combined_max_size,
          invalid_index_value,
          divisor_for_int8,
          combined_warp_payloads,
          total_warps,
          combined_remaining_payloads,
          total_remaining,
          warp_chunk_indices,
          remaining_chunk_indices,
          list_per_embedding_scale);

  auto list_of_scores = combined_scores.split_with_sizes(
      at::IntArrayRef(per_index_rounded_max_sizes), /*dim=*/1);
  auto list_of_indices = combined_indices.split_with_sizes(
      at::IntArrayRef(per_index_rounded_max_sizes), /*dim=*/1);

  std::vector<Tensor> scores_vec(list_of_scores.begin(), list_of_scores.end());
  std::vector<Tensor> indices_vec(
      list_of_indices.begin(), list_of_indices.end());

  return std::make_tuple(std::move(scores_vec), std::move(indices_vec));
}

TORCH_LIBRARY_IMPL(st, CUDA, m) {
  m.impl(
      "fused_kmean_ann",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(fused_kmean_ann_cuda)));
}

TORCH_LIBRARY_IMPL(st, CUDA, m) {
  m.impl(
      "fused_kmean_ann_with_partial_masks",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(fused_kmean_ann_cuda_with_partial_masks)));
}

TORCH_LIBRARY_IMPL(st, CUDA, m) {
  m.impl(
      "fused_kmean_ann_with_partial_masks_multiple",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(fused_kmean_ann_with_partial_masks_multiple_cuda)));
}

} // namespace st::ops::fused_kmean_ann

#undef MAYBE_TORCH_DSA_KERNEL_LAUNCH
