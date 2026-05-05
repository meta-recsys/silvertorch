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

TORCH_LIBRARY_IMPL(st, CUDA, m) {
  m.impl(
      "fused_kmean_ann",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(fused_kmean_ann_cuda)));
}

} // namespace st::ops::fused_kmean_ann

#undef MAYBE_TORCH_DSA_KERNEL_LAUNCH
