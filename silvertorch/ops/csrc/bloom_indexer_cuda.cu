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
#include <ATen/ceil_div.h> // @manual=//caffe2:ATen-cpu
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
#include <cub/cub.cuh>

#define BLOOM_INDEX_CUDA
#include "bloom_index_util.cuh"
#include "faster_repeat_interleave.cuh"

// OSS / internal / ROCm portability shim for CUB.
//
// Three target environments:
//   1. Internal fbcode CUDA — passes -DCUB_WRAPPED_NAMESPACE=at_cuda_detail to
//      nvcc, which causes <cub/cub.cuh> to wrap all of CUB into
//      ::at_cuda_detail::cub (avoids ODR conflicts with libtorch's CUB).
//   2. OSS pip / cpp_extension CUDA — does NOT define CUB_WRAPPED_NAMESPACE,
//      CUB lives in its native ::cub namespace.
//   3. AMD ROCm — fbcode hipify rewrites <cub/cub.cuh> to <hipcub/hipcub.hpp>
//      and the namespace lives at ::hipcub. USE_ROCM is defined.
//
// (Background: PyTorch's <ATen/cuda/cub.cuh> wrapper used to handle this for
// us, but its shim is incompatible with CCCL >= 2.5 / CUDA 12.5+ when used
// from a standalone extension, so we include <cub/cub.cuh> directly and
// redefine the helper macros below.)
#if defined(USE_ROCM)
#define ST_CUB_NS ::hipcub
#elif defined(CUB_WRAPPED_NAMESPACE)
#define ST_CUB_NS ::CUB_WRAPPED_NAMESPACE::cub
#else
#define ST_CUB_NS ::cub
#endif

#ifndef CUB_WRAPPER
#define CUB_WRAPPER(func, ...)                                      \
  do {                                                              \
    size_t temp_storage_bytes = 0;                                  \
    func(nullptr, temp_storage_bytes, __VA_ARGS__);                 \
    auto temp_storage = at::empty(                                  \
        {static_cast<int64_t>(temp_storage_bytes)},                 \
        at::TensorOptions().dtype(at::kByte).device(at::kCUDA));    \
    func(temp_storage.data_ptr(), temp_storage_bytes, __VA_ARGS__); \
    AT_CUDA_CHECK(cudaGetLastError());                              \
  } while (0)
#endif
#ifndef NO_ROCM
#define NO_ROCM(x) x
#endif
#ifndef ROCM_HIPCUB
#define ROCM_HIPCUB(x) x
#endif

namespace st {
namespace ops {
namespace bloom_indexer {

using at::PackedTensorAccessor32;
using at::PackedTensorAccessor64;
using at::Tensor;
using namespace st::ops::bloom_index;
using namespace st::ops::faster_repeat_interleave;

namespace {
static constexpr int64_t BLOOM_BUILD_BLOCK_THREADS = 256;
#if USE_ROCM
#define BLOOM_BUILD_V2_WARP_PER_BLOCK \
  (BLOOM_BUILD_BLOCK_THREADS / C10_WARP_SIZE)
#define BLOOM_NO_SIGNATURE_BUILD_DOC_PER_THREAD \
  (C_BITS_IN_UINT64 / C10_WARP_SIZE)
#define BLOOM_SAVE_MEM_BUILD_WARP_COUNT_PER_BLOCK \
  (BLOOM_BUILD_BLOCK_THREADS / C10_WARP_SIZE)
#else
static constexpr int64_t BLOOM_BUILD_V2_WARP_PER_BLOCK =
    BLOOM_BUILD_BLOCK_THREADS / C10_WARP_SIZE;
static constexpr int64_t BLOOM_NO_SIGNATURE_BUILD_DOC_PER_THREAD =
    C_BITS_IN_UINT64 / C10_WARP_SIZE;
static constexpr int64_t BLOOM_SAVE_MEM_BUILD_WARP_COUNT_PER_BLOCK =
    BLOOM_BUILD_BLOCK_THREADS / C10_WARP_SIZE;
#endif

static inline void bloom_index_feature_check(
    const Tensor& feature_ids,
    const Tensor& feature_offsets,
    const Tensor& feature_values,
    double b_multiplier = 2.0) {
  TORCH_CHECK(feature_ids.is_contiguous(), "Feature ids are not contiguous");
  TORCH_CHECK(
      feature_offsets.is_contiguous(), "Feature offsets are not contiguous");
  TORCH_CHECK(
      feature_values.is_contiguous(), "Feature values are not contiguous");
  TORCH_CHECK(
      (feature_offsets.numel() - 1) % feature_ids.numel() == 0,
      "Feature offsets size ",
      feature_offsets.numel(),
      " must be a multiple of the feature ids size ",
      feature_ids.numel());
  TORCH_CHECK(feature_ids.is_cuda());
  TORCH_CHECK(feature_offsets.is_cuda());
  TORCH_CHECK(feature_values.is_cuda());
  TORCH_CHECK(feature_ids.scalar_type() == at::kInt);
  TORCH_CHECK(feature_offsets.scalar_type() == at::kLong);
  TORCH_CHECK(feature_values.scalar_type() == at::kLong);
  TORCH_CHECK(b_multiplier > 1.0, "b_multiplier must be greater than 1.0");
}

class ColumnIdxToBundleKey {
 public:
  ColumnIdxToBundleKey() = default;
  __device__ __forceinline__ uint32_t operator()(uint32_t column_idx) const {
    return column_idx / C_BLOOM_V2_COL_BUNDLE_SIZE;
  }
};

template <bool is_bloom_index_v2>
__global__ void generate_document_signature(
    const PackedTensorAccessor32<int32_t, 1> index_feature_ids,
    const PackedTensorAccessor64<int64_t, 1> index_feature_offsets,
    const PackedTensorAccessor64<int64_t, 1> index_feature_values,
    const int64_t* column_bundle_b_offsets,
    const int64_t* column_bundle_signature_offsets,
    PackedTensorAccessor64<int8_t, 1> document_signature,
    int64_t b,
    int64_t k,
    int64_t document_count,
    TORCH_DSA_KERNEL_ARGS) {
  int64_t feature_count = index_feature_ids.size(0);
  // Stride loop:
  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for (int64_t process_index = blockIdx.x * blockDim.x + threadIdx.x;
       process_index < document_count;
       process_index += blockDim.x * gridDim.x) {
    int64_t signature_offset;
    if constexpr (is_bloom_index_v2) {
      int64_t bundle_id = process_index / C_BITS_IN_BLOOM_V2_COL_BUNDLE;
      int64_t doc_offset_in_bundle =
          process_index % C_BITS_IN_BLOOM_V2_COL_BUNDLE;
      int64_t column_bundle_signature_offset =
          __ldg(column_bundle_signature_offsets + bundle_id);
      int64_t signature_size =
          __ldg(column_bundle_signature_offsets + bundle_id + 1) -
          column_bundle_signature_offset;
      signature_offset =
          column_bundle_signature_offset * C_BITS_IN_BLOOM_V2_COL_BUNDLE +
          doc_offset_in_bundle * signature_size;
      b = __ldg(column_bundle_b_offsets + bundle_id + 1) -
          __ldg(column_bundle_b_offsets + bundle_id);
    } else {
      signature_offset = process_index * round_bits_to_bytes(b);
    }
    int64_t offsets_start = process_index * feature_count;
    int8_t* document_signature_ptr =
        document_signature.data() + signature_offset;
    for (int64_t i = 0; i < feature_count; ++i) {
      int64_t value_start = index_feature_offsets[offsets_start + i];
      int64_t value_end = index_feature_offsets[offsets_start + i + 1];
      int64_t feature_id = static_cast<int64_t>(index_feature_ids[i]);

      for (; value_start < value_end; ++value_start) {
        assign_document_signature<typename BIndexType<is_bloom_index_v2>::type>(
            feature_id,
            index_feature_values[value_start],
            b,
            k,
            document_signature_ptr);
      }
    }
  }
}

template <bool is_bloom_index_v2>
__global__ void generate_bloom_index_kernel(
    const int8_t* document_signature,
    int64_t column_b_count, // v1: b * column_count, v2: b_count
    // start bloom index v2 only parameters.
    const int64_t* column_bundle_b_offsets,
    const int64_t* column_bundle_signature_offsets,
    const int64_t*
        assigned_bundle_ids, // assigned bundle ids for each column_b bundle.
    // end bloom index v2 only parameters.
    // start bloom index v1 only parameters.
    int64_t
        bloom_index_column_count, // v1: row length for 2d v2 bloom index.
                                  // NOTE: column_count contains reserved slots.
    int64_t b, // v1: b, v2: -1
    // end bloom index v1 only parameters.
    int64_t* bloom_index, // v1: [b, column_count], v2: [b_count]
    TORCH_DSA_KERNEL_ARGS) {
  for (int64_t b_idx = blockIdx.x * blockDim.x + threadIdx.x;
       b_idx < column_b_count;
       b_idx += blockDim.x * gridDim.x) {
    int64_t current_b_p;
    const int8_t* column_signature_start_ptr;
    int64_t column_signature_size;
    int64_t output_index;
    if constexpr (is_bloom_index_v2) {
      int64_t bundle_column_b_id = b_idx / C_BLOOM_V2_COL_BUNDLE_SIZE;
      int64_t bundle_id = __ldg(assigned_bundle_ids + bundle_column_b_id);
      column_signature_size =
          __ldg(column_bundle_signature_offsets + bundle_id + 1) -
          __ldg(column_bundle_signature_offsets + bundle_id);
      int64_t doc_offset_in_bundle =
          (b_idx * C_BITS_IN_UINT64) % C_BITS_IN_BLOOM_V2_COL_BUNDLE;
      column_signature_start_ptr = document_signature +
          __ldg(column_bundle_signature_offsets + bundle_id) *
              C_BITS_IN_BLOOM_V2_COL_BUNDLE +
          doc_offset_in_bundle * column_signature_size;
      current_b_p =
          bundle_column_b_id - __ldg(column_bundle_b_offsets + bundle_id);
      output_index = b_idx;
    } else {
      current_b_p = b_idx % b;
      int64_t column_id = b_idx / b;
      column_signature_size = round_bits_to_bytes(b);
      column_signature_start_ptr = document_signature +
          column_signature_size * C_BITS_IN_UINT64 * column_id;
      output_index = current_b_p * bloom_index_column_count + column_id;
    }
    uint64_t result = 0;
#pragma unroll
    for (int64_t i = 0; i < C_BITS_IN_UINT64; ++i) {
      result <<= 1;
      result += static_cast<uint64_t>(get_bit(
          column_signature_start_ptr + i * column_signature_size, current_b_p));
    }

    bloom_index[output_index] = result;
  }
}

// This kernel is used to generate b_sizes and signature_sizes for bloom index
// v2. b_sizes and signature_sizes are used to generate b_offsets and
// signature_offsets for bloom index v2.
__global__ void generate_column_b_and_signature_sizes_kernel(
    const int64_t* feature_offsets,
    int64_t feature_count,
    int64_t document_count,
    int64_t column_count,
    int64_t k,
    double b_multiplier,
    int64_t* b_sizes,
    int64_t* signature_sizes) {
  uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id = thread_id / C10_WARP_SIZE;
  uint32_t tidx = thread_id % C10_WARP_SIZE;
  namespace cub = ST_CUB_NS;
  using WarpReduce = cub::WarpReduce<int32_t>;
  __shared__ typename WarpReduce::TempStorage
      temp_storage[BLOOM_BUILD_V2_WARP_PER_BLOCK];
  // stride loop to handle all columns.
  for (uint32_t i = warp_id; i < column_count;
       i += gridDim.x * BLOOM_BUILD_V2_WARP_PER_BLOCK) {
    int32_t max_feature_count = 0; // max feature count for column.
#pragma unroll
    for (uint32_t j = 0; j < C_BITS_IN_UINT64 / C10_WARP_SIZE; ++j) {
      int32_t document_id = i * C_BITS_IN_UINT64 + j * C10_WARP_SIZE + tidx;
      if (document_id < document_count) {
        int32_t doc_feature_count = 0;
        for (uint32_t k = 0; k < feature_count; ++k) {
          int64_t feature_idx = document_id * feature_count + k;
          doc_feature_count += static_cast<int32_t>(
              feature_offsets[feature_idx + 1] - feature_offsets[feature_idx]);
        }
        max_feature_count = std::max(max_feature_count, doc_feature_count);
      }
    }

    int32_t column_max_feature_count =
        WarpReduce(temp_storage[warp_id % BLOOM_BUILD_V2_WARP_PER_BLOCK])
            .Reduce(max_feature_count, cub::Max());
    if (tidx == 0) {
      int64_t b_size =
          static_cast<int64_t>(column_max_feature_count * k * b_multiplier);
      CUDA_KERNEL_ASSERT(b_size <= MAX_B_V2 && "b_size exceeds MAX_B_V2");
      b_sizes[i + 1] = b_size;
      signature_sizes[i + 1] = round_bits_to_bytes(b_size);
      if (i == 0) {
        b_sizes[0] = 0;
        signature_sizes[0] = 0;
      }
    }
  }
}

// This function launches generate_column_b_and_signature_sizes_kernel,
// and then generate b_offsets and signature_offsets for bloom index v2.
// since the size of b and signature are same for a bundle of columns, we
// just keep the b_offsets and signature_offsets per bundle.
std::tuple<Tensor, Tensor> generate_column_b_and_signature_offsets(
    const Tensor& feature_offsets,
    int64_t feature_count,
    int64_t document_count,
    int64_t column_count,
    int64_t k,
    double b_multiplier) {
  int64_t block_size = BLOOM_BUILD_BLOCK_THREADS;
  int64_t grid_size = std::min(
      at::ceil_div(column_count, BLOOM_BUILD_V2_WARP_PER_BLOCK),
      128L *
          CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
              ->multiProcessorCount);
  Tensor b_sizes = at::empty(
      {column_count + 1},
      c10::TensorOptions().dtype(at::kLong).device(feature_offsets.device()));
  Tensor signature_sizes = at::empty(
      {column_count + 1},
      c10::TensorOptions().dtype(at::kLong).device(feature_offsets.device()));
  generate_column_b_and_signature_sizes_kernel<<<
      grid_size,
      block_size,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      feature_offsets.data_ptr<int64_t>(),
      feature_count,
      document_count,
      column_count,
      k,
      b_multiplier,
      b_sizes.data_ptr<int64_t>(),
      signature_sizes.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if constexpr (C_BLOOM_V2_COL_BUNDLE_SIZE == 1) {
    at::cumsum_out(b_sizes, b_sizes, 0);
    at::cumsum_out(signature_sizes, signature_sizes, 0);
    return std::make_tuple(std::move(b_sizes), std::move(signature_sizes));
  } else {
    using counting_iter_t =
        ST_CUB_NS::CountingInputIterator<uint32_t, uint32_t>;
    using bundle_key_iter_t = ST_CUB_NS::
        TransformInputIterator<uint32_t, ColumnIdxToBundleKey, counting_iter_t>;
    bundle_key_iter_t bundle_key_iter(
        counting_iter_t(0), ColumnIdxToBundleKey());
    int64_t bundle_count =
        at::ceil_div(column_count, C_BLOOM_V2_COL_BUNDLE_SIZE);
    Tensor bundle_b_sizes = at::zeros({bundle_count + 1}, b_sizes.options());
    Tensor bundle_signature_sizes =
        at::zeros({bundle_count + 1}, signature_sizes.options());
    Tensor bundle_unique_ids =
        at::empty({bundle_count}, b_sizes.options().dtype(at::kUInt32));
    Tensor num_of_runs_out =
        at::empty({1}, b_sizes.options().dtype(at::kUInt32));
    CUB_WRAPPER(
        ST_CUB_NS::DeviceReduce::ReduceByKey,
        bundle_key_iter,
        bundle_unique_ids.data_ptr<uint32_t>(),
        b_sizes.data_ptr<int64_t>() + 1,
        bundle_b_sizes.data_ptr<int64_t>() + 1,
        num_of_runs_out.data_ptr<uint32_t>(),
        ST_CUB_NS::Max(),
        column_count,
        at::cuda::getCurrentCUDAStream());
    CUB_WRAPPER(
        ST_CUB_NS::DeviceReduce::ReduceByKey,
        bundle_key_iter,
        bundle_unique_ids.data_ptr<uint32_t>(),
        signature_sizes.data_ptr<int64_t>() + 1,
        bundle_signature_sizes.data_ptr<int64_t>() + 1,
        num_of_runs_out.data_ptr<uint32_t>(),
        ST_CUB_NS::Max(),
        column_count,
        at::cuda::getCurrentCUDAStream());
    at::cumsum_out(bundle_b_sizes, bundle_b_sizes, 0);
    at::cumsum_out(bundle_signature_sizes, bundle_signature_sizes, 0);
    return std::make_tuple(
        std::move(bundle_b_sizes), std::move(bundle_signature_sizes));
  }
}

template <bool is_bloom_index_v2>
Tensor generate_document_signature_launch(
    const Tensor& feature_ids,
    const Tensor& feature_offsets,
    const Tensor& feature_values,
    // start bloom index v2 only parameters.
    const int64_t* column_bundle_b_offsets,
    const int64_t* column_bundle_signature_offsets,
    // end bloom index v2 only parameters.
    // start bloom index v1 only parameters.
    int64_t b,
    // end bloom index v1 only parameters.
    // signature count:
    // v1: compute_column_count * C_BITS_IN_UINT64 * round_bits_to_bytes(b)
    // v2: column_signature_offsets[-1] * C_BITS_IN_UINT64
    int64_t signature_count,
    int64_t k,
    int64_t document_count) {
  Tensor document_signature = at::zeros(
      {signature_count},
      c10::TensorOptions().dtype(at::kChar).device(feature_values.device()));
  int64_t block_size = BLOOM_BUILD_BLOCK_THREADS;
  int64_t grid_size = std::min(
      at::ceil_div(document_count, block_size),
      128L *
          CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
              ->multiProcessorCount);
  TORCH_DSA_KERNEL_LAUNCH(
      generate_document_signature<is_bloom_index_v2>,
      grid_size,
      block_size,
      0,
      at::cuda::getCurrentCUDAStream(),
      feature_ids.packed_accessor32<int32_t, 1>(),
      feature_offsets.packed_accessor64<int64_t, 1>(),
      feature_values.packed_accessor64<int64_t, 1>(),
      column_bundle_b_offsets,
      column_bundle_signature_offsets,
      document_signature.packed_accessor64<int8_t, 1>(),
      b, // b is not used in bloom index v2.
      k,
      document_count);
  return document_signature;
}

template <bool is_bloom_index_v2>
Tensor generate_bloom_index_launch(
    const Tensor& document_signature,
    // b_count: total b count for computation.
    //   v1: b * column_count
    //   v2: column_bundle_b_offsets[-1] * C_BLOOM_V2_COL_BUNDLE_SIZE
    int64_t column_b_count,
    // start bloom index v2 only parameters.
    const int64_t* column_bundle_b_offsets,
    int64_t compute_column_count,
    const int64_t* column_bundle_signature_offsets,
    // end bloom index v2 only parameters.
    // start bloom index v1 only parameters.
    // bloom_index_column_count: row length for 2d v2 bloom index.
    // NOTE: bloom_index_column_count contains reserved slots for v1.
    int64_t bloom_index_column_count,
    int64_t b // v1: b, v2: -1
    // end bloom index v1 only parameters.
) {
  int64_t block_size = BLOOM_BUILD_BLOCK_THREADS;
  int64_t grid_size = std::min(
      at::ceil_div(column_b_count, block_size),
      128L *
          CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
              ->multiProcessorCount);
  Tensor bloom_index;
  Tensor assigned_bundle_ids;
  int64_t* assigned_bundle_ids_ptr = nullptr;
  if constexpr (is_bloom_index_v2) {
    bloom_index = at::zeros(
        {column_b_count},
        c10::TensorOptions().dtype(at::kLong).device(
            document_signature.device()));
    int64_t bundle_b_count =
        at::ceil_div(column_b_count, C_BLOOM_V2_COL_BUNDLE_SIZE);
    assigned_bundle_ids = at::empty(
        {bundle_b_count},
        c10::TensorOptions().dtype(at::kLong).device(
            document_signature.device()));
    assigned_bundle_ids_ptr = assigned_bundle_ids.data_ptr<int64_t>();
    consecutive_int64_repeat_interleave_with_cumsum_cuda_raw(
        // bundle count
        at::ceil_div(compute_column_count, C_BLOOM_V2_COL_BUNDLE_SIZE),
        column_bundle_b_offsets + 1,
        0,
        bundle_b_count,
        assigned_bundle_ids_ptr);
  } else {
    bloom_index = at::zeros(
        {b, bloom_index_column_count},
        c10::TensorOptions().dtype(at::kLong).device(
            document_signature.device()));
  }
  TORCH_DSA_KERNEL_LAUNCH(
      generate_bloom_index_kernel<is_bloom_index_v2>,
      grid_size,
      block_size,
      0,
      at::cuda::getCurrentCUDAStream(),
      document_signature.data_ptr<int8_t>(),
      column_b_count,
      column_bundle_b_offsets, // v2 only parameter.
      column_bundle_signature_offsets, // v2 only parameter.
      assigned_bundle_ids_ptr, // v2 only parameter.
      bloom_index_column_count, // v1 only parameter.
      b, // v1 only parameter.
      bloom_index.data_ptr<int64_t>());
  return bloom_index;
}

template <bool is_bloom_index_v2>
__device__ __forceinline__ void update_bloom_column_per_k(
    int64_t* bloom_index,
    // bloom_index_column_count is for v1 only.
    // it is not exactly column_count, since reserved slots
    // might be included in v1.
    int64_t bloom_index_column_count,
    // column_bundle_b_offsets is for v2 only.
    // it is designed to get the offset of bloom index for a column.
    const int64_t* column_bundle_b_offsets,
    uint32_t bundle_id, // v2 only.
    // b_positions are stored in shared memory as designed in
    // generate_bloom_index_without_signature_kernel.
    const uint32_t* __shared__ b_positions,
    uint32_t local_tid_in_warp,
    // since 1 warp is used to compute 1 column (64 docs), so it is possible
    // that 1 thread in the warp is responsible for multiple docs.
    // if so, each thread will do multiple rounds of computation, and each
    // round will update 1 doc, for example, if warp size is 32,
    // in each round, we will only update 32 docs as a batch, so we need
    // a offset to indicate the batch, and we can know where to set the bits
    // to the int64_t index.
    uint32_t batch_offset,
    uint32_t column_id) {
  uint32_t pos = b_positions[local_tid_in_warp];
  if (pos == static_cast<uint32_t>(-1)) {
    return;
  }
  uint64_t b_column_output = 0;
  for (uint32_t i = 0; i < C10_WARP_SIZE; ++i) {
    if (pos == b_positions[i]) {
      if (i < local_tid_in_warp) {
        return;
      }
      b_column_output |= (1ULL << (C_BITS_IN_UINT64 - 1 - i));
    }
  }
  b_column_output >>= batch_offset;
  int64_t column_index_in_bloom_index;
  if constexpr (is_bloom_index_v2) {
    column_index_in_bloom_index =
        column_bundle_b_offsets[bundle_id] * C_BLOOM_V2_COL_BUNDLE_SIZE +
        pos * C_BLOOM_V2_COL_BUNDLE_SIZE +
        (column_id % C_BLOOM_V2_COL_BUNDLE_SIZE);
  } else {
    column_index_in_bloom_index = pos * bloom_index_column_count + column_id;
  }
  bloom_index[column_index_in_bloom_index] |= b_column_output;
}

template <bool is_bloom_index_v2>
__global__ void generate_bloom_index_without_signature_kernel(
    const PackedTensorAccessor32<int32_t, 1> index_feature_ids,
    const PackedTensorAccessor64<int64_t, 1> index_feature_offsets,
    const PackedTensorAccessor64<int64_t, 1> index_feature_values,
    const int64_t* column_bundle_b_offsets, // v2 only.
    const int32_t*
        aligned_feature_indices_ptr, // optional: if provided, use directly
                                     // instead of computing positions
    int64_t b,
    int64_t k,
    int64_t document_count,
    int64_t column_count, // (doucment_count + 64 - 1) / 64
    int64_t
        bloom_index_column_count, // v1 only: row length for 2d v2 bloom index.
                                  // NOTE: it may count reserved slots.
    int64_t* bloom_index, // v1: [b, column_count], v2: [b_count]
    TORCH_DSA_KERNEL_ARGS) {
  namespace cub = ST_CUB_NS;
  using WarpReduce = cub::WarpReduce<uint32_t>;
  __shared__ typename WarpReduce::TempStorage
      temp_storage[BLOOM_SAVE_MEM_BUILD_WARP_COUNT_PER_BLOCK];
  __shared__ uint32_t shared_pos[BLOOM_BUILD_BLOCK_THREADS];
  typename BIndexType<is_bloom_index_v2>::type used_b_p[MAX_K];
  int64_t feature_count = index_feature_ids.size(0);
  uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t warp_id = thread_id / C10_WARP_SIZE;
  uint32_t local_warp_id = warp_id % BLOOM_SAVE_MEM_BUILD_WARP_COUNT_PER_BLOCK;
  uint32_t tidx = thread_id % C10_WARP_SIZE;
  for (uint32_t i = warp_id; i < column_count;
       i += gridDim.x * BLOOM_SAVE_MEM_BUILD_WARP_COUNT_PER_BLOCK) {
    // get right b, v1: static b, v2: dynamic b.
    int64_t resolved_b = b;
    uint32_t bundle_id = 0;
    if constexpr (is_bloom_index_v2) {
      bundle_id = static_cast<uint32_t>(i / C_BLOOM_V2_COL_BUNDLE_SIZE);
      resolved_b = __ldg(column_bundle_b_offsets + bundle_id + 1) -
          __ldg(column_bundle_b_offsets + bundle_id);
    }

    uint32_t warp_start_doc_id = i * C_BITS_IN_UINT64;
    for (uint32_t feature_idx = 0; feature_idx < feature_count; ++feature_idx) {
      // compute max cardinality of current column for each feature.
      int64_t feature_id = static_cast<int64_t>(index_feature_ids[feature_idx]);
      for (uint32_t j = 0; j < BLOOM_NO_SIGNATURE_BUILD_DOC_PER_THREAD; ++j) {
        uint32_t cardinality = 0;
        uint32_t doc_id = warp_start_doc_id + j * C10_WARP_SIZE + tidx;
        int64_t offset_start = static_cast<int64_t>(doc_id) * feature_count;
        int64_t value_start = 0;
        int64_t value_end = 0;
        if (doc_id < document_count) {
          value_start = index_feature_offsets[offset_start + feature_idx];
          value_end = index_feature_offsets[offset_start + feature_idx + 1];
          cardinality = static_cast<uint32_t>(value_end - value_start);
        }
        uint32_t max_cardinality = WarpReduce(temp_storage[local_warp_id])
                                       .Reduce(cardinality, cub::Max());
        WARP_SYNC();

        // for each cardinality/feature_value, update bloom index for current
        // column.
        for (uint32_t c = 0; c < max_cardinality; ++c) {
          bool has_value = false;
          if (value_start + c < value_end) {
            if (aligned_feature_indices_ptr != nullptr) {
              // Use pre-computed aligned indices directly (k must be 1)
              used_b_p[0] =
                  static_cast<typename BIndexType<is_bloom_index_v2>::type>(
                      aligned_feature_indices_ptr[value_start + c]);
            } else {
              int64_t feature_value = index_feature_values[value_start + c];
              assign_one_bits_position<
                  typename BIndexType<is_bloom_index_v2>::type>(
                  feature_id, feature_value, resolved_b, k, used_b_p);
            }
            has_value = true;
          }
          for (int64_t k_idx = 0; k_idx < k; ++k_idx) {
            shared_pos[threadIdx.x] =
                has_value ? used_b_p[k_idx] : static_cast<uint32_t>(-1);
            WARP_SYNC();
            update_bloom_column_per_k<is_bloom_index_v2>(
                bloom_index,
                bloom_index_column_count,
                column_bundle_b_offsets,
                bundle_id,
                &shared_pos[local_warp_id * C10_WARP_SIZE],
                tidx,
                j * C10_WARP_SIZE,
                i);
            WARP_SYNC();
          }
        }
      }
    }
  }
}

template <bool is_bloom_index_v2>
Tensor generate_bloom_index_without_signature_launch(
    const Tensor& feature_ids,
    const Tensor& feature_offsets,
    const Tensor& feature_values,
    const int64_t* column_bundle_b_offsets, // v2 only.
    const int32_t*
        aligned_feature_indices_ptr, // optional: if provided, use directly
    int64_t b,
    int64_t k,
    // column_b_count: total b count for computation.
    // v1: b * column_count
    // v2: column_bundle_b_offsets[-1] * C_BLOOM_V2_COL_BUNDLE_SIZE
    int64_t column_b_count,
    // bloom_index_column_count is for v1 only.
    // it is not exactly column_count, since reserved slots
    // might be included in v1.
    int64_t bloom_index_column_count,
    int64_t compute_column_count,
    int64_t document_count) {
  Tensor bloom_index;
  if constexpr (is_bloom_index_v2) {
    // v2: use column_b_count to allocate index.
    bloom_index = at::zeros(
        {column_b_count},
        c10::TensorOptions().dtype(at::kLong).device(feature_values.device()));
  } else {
    bloom_index = at::zeros(
        {b, bloom_index_column_count},
        c10::TensorOptions().dtype(at::kLong).device(feature_values.device()));
  }

  int64_t block_size = BLOOM_BUILD_BLOCK_THREADS;
  int64_t grid_size = std::min(
      at::ceil_div(
          compute_column_count, BLOOM_SAVE_MEM_BUILD_WARP_COUNT_PER_BLOCK),
      128L *
          CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
              ->multiProcessorCount);
  TORCH_DSA_KERNEL_LAUNCH(
      generate_bloom_index_without_signature_kernel<is_bloom_index_v2>,
      grid_size,
      block_size,
      0,
      at::cuda::getCurrentCUDAStream(),
      feature_ids.packed_accessor32<int32_t, 1>(),
      feature_offsets.packed_accessor64<int64_t, 1>(),
      feature_values.packed_accessor64<int64_t, 1>(),
      column_bundle_b_offsets,
      aligned_feature_indices_ptr,
      b,
      k,
      document_count,
      compute_column_count,
      bloom_index_column_count,
      bloom_index.data_ptr<int64_t>());
  return bloom_index;
}

__global__ void generate_bloom_index_transpose_kernel(
    const PackedTensorAccessor32<int32_t, 1> index_feature_ids,
    const PackedTensorAccessor64<int64_t, 1> index_feature_offsets,
    const PackedTensorAccessor64<int64_t, 1> index_feature_values,
    int64_t* output,
    int64_t b,
    int64_t k,
    int64_t b_words,
    int64_t document_count,
    TORCH_DSA_KERNEL_ARGS) {
  int64_t feature_count = index_feature_ids.size(0);
  for (int64_t doc_id = blockIdx.x * blockDim.x + threadIdx.x;
       doc_id < document_count;
       doc_id += blockDim.x * gridDim.x) {
    int64_t offsets_start = doc_id * feature_count;
    int64_t* doc_output = output + doc_id * b_words;
    for (int64_t i = 0; i < feature_count; ++i) {
      int64_t value_start = index_feature_offsets[offsets_start + i];
      int64_t value_end = index_feature_offsets[offsets_start + i + 1];
      int64_t feature_id = static_cast<int64_t>(index_feature_ids[i]);
      for (; value_start < value_end; ++value_start) {
        typename BIndexType<false>::type used_b_p[MAX_K];
        assign_one_bits_position<typename BIndexType<false>::type>(
            feature_id, index_feature_values[value_start], b, k, used_b_p);
        for (int64_t j = 0; j < k; ++j) {
          int64_t b_p = static_cast<int64_t>(used_b_p[j]);
          doc_output[b_p / C_BITS_IN_UINT64] |=
              (1LL << (C_BITS_IN_UINT64 - 1 - b_p % C_BITS_IN_UINT64));
        }
      }
    }
  }
}

} // namespace

template <bool is_bloom_index_v2>
std::tuple<Tensor, Tensor> bloom_index_build_common(
    const Tensor& feature_ids,
    const Tensor& feature_offsets,
    const Tensor& feature_values,
    // start bloom index v2 only parameters.
    double b_multiplier,
    // end bloom index v2 only parameters.
    // start bloom index v1 only parameters.
    int64_t b,
    int64_t reserved_num_docs,
    // end bloom index v1 only parameters.
    int64_t k,
    bool fast_build,
    const Tensor* aligned_feature_indices) {
  bloom_index_feature_check(feature_ids, feature_offsets, feature_values);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(feature_ids.device().index());
  // a holder for column_bundle_b_offsets_ptr and column_signature_offsets_ptr
  // for v2.
  Tensor column_bundle_b_offsets;
  Tensor column_bundle_signature_offsets;
  std::tuple<Tensor, Tensor> column_offsets_for_b_and_signature;
  int64_t* column_bundle_b_offsets_ptr = nullptr;
  int64_t* column_bundle_signature_offsets_ptr = nullptr;
  int64_t document_count = (feature_offsets.numel() - 1) / feature_ids.numel();
  int64_t compute_column_count = at::ceil_div(document_count, C_BITS_IN_UINT64);
  int64_t bloom_index_column_count = -1; // for v1 only, due to reserved slots.
  int64_t signature_count;
  int64_t b_count; // total b count for computation.
  if constexpr (is_bloom_index_v2) {
    TORCH_CHECK(b_multiplier > 1.0, "b_multiplier must be greater than 1.0");
    TORCH_CHECK(reserved_num_docs <= 0, "reserved docs is not supported in v2");
    int64_t feature_count = feature_ids.size(0);
    auto [bundle_b_offsets, bundle_signature_offsets] =
        generate_column_b_and_signature_offsets(
            feature_offsets,
            feature_count,
            document_count,
            compute_column_count,
            k,
            b_multiplier);
    column_bundle_b_offsets = std::move(bundle_b_offsets);
    column_bundle_signature_offsets = std::move(bundle_signature_offsets);
    column_bundle_b_offsets_ptr = column_bundle_b_offsets.data_ptr<int64_t>();
    column_bundle_signature_offsets_ptr =
        column_bundle_signature_offsets.data_ptr<int64_t>();
    signature_count = column_bundle_signature_offsets[-1].item<int64_t>() *
        C_BITS_IN_BLOOM_V2_COL_BUNDLE;
    b_count = column_bundle_b_offsets[-1].item<int64_t>() *
        C_BLOOM_V2_COL_BUNDLE_SIZE;
  } else {
    TORCH_CHECK(b < MAX_B, "b should be less than ", MAX_B);
    TORCH_CHECK(k < MAX_K, "k should be less than ", MAX_K);
    int64_t total_docs =
        document_count + ((reserved_num_docs > 0) ? reserved_num_docs : 0);
    bloom_index_column_count = at::ceil_div(total_docs, C_BITS_IN_UINT64);
    signature_count =
        compute_column_count * C_BITS_IN_UINT64 * round_bits_to_bytes(b);
    b_count = b * compute_column_count;
  }

  const int32_t* aligned_feature_indices_ptr = nullptr;
  if (aligned_feature_indices) {
    TORCH_CHECK(
        aligned_feature_indices->scalar_type() == at::kInt,
        "aligned_feature_indices must be int32");
    TORCH_CHECK(
        aligned_feature_indices->numel() == feature_values.numel(),
        "aligned_feature_indices must have the same length as feature_values");
    TORCH_CHECK(k == 1, "k must be 1 when aligned_feature_indices is provided");
    TORCH_CHECK(
        aligned_feature_indices->is_cuda(),
        "aligned_feature_indices must be on CUDA");
    TORCH_CHECK(
        aligned_feature_indices->is_contiguous(),
        "aligned_feature_indices must be contiguous");
    int64_t max_index = aligned_feature_indices->max().item<int64_t>();
    TORCH_CHECK(
        b == max_index + 2,
        "b must be equal to max(aligned_feature_indices) + 2");
    TORCH_CHECK(
        !is_bloom_index_v2, "aligned_feature_indices is not supported in v2");
    TORCH_CHECK(
        fast_build,
        "fast_build must be true when aligned_feature_indices is provided");
    aligned_feature_indices_ptr = aligned_feature_indices->data_ptr<int32_t>();
  }

  if (fast_build) {
    Tensor bloom_index =
        generate_bloom_index_without_signature_launch<is_bloom_index_v2>(
            feature_ids,
            feature_offsets,
            feature_values,
            column_bundle_b_offsets_ptr, // v2 only parameter.
            aligned_feature_indices_ptr, // v1 only parameter.
            b, // v1 only parameter.
            k,
            b_count,
            bloom_index_column_count,
            compute_column_count,
            document_count);
    return std::make_tuple(
        std::move(bloom_index), std::move(column_bundle_b_offsets));
  } else {
    Tensor document_signature =
        generate_document_signature_launch<is_bloom_index_v2>(
            feature_ids,
            feature_offsets,
            feature_values,
            column_bundle_b_offsets_ptr, // v2 only parameter.
            column_bundle_signature_offsets_ptr, // v2 only parameter.
            b, // v1 only parameter.
            signature_count,
            k,
            document_count);
    Tensor bloom_index = generate_bloom_index_launch<is_bloom_index_v2>(
        document_signature,
        b_count,
        column_bundle_b_offsets_ptr, // v2 only parameter.
        compute_column_count, // v2 only parameter.
        column_bundle_signature_offsets_ptr, // v2 only parameter.
        bloom_index_column_count, // v1 only parameter.
        b // v1 only parameter.
    );
    return std::make_tuple(
        std::move(bloom_index), std::move(column_bundle_b_offsets));
  }
}

std::tuple<Tensor, Tensor> bloom_index_build_cuda(
    const Tensor& feature_ids,
    const Tensor& feature_offsets,
    const Tensor& feature_values,
    double b_multiplier,
    int64_t k,
    bool fast_build) {
  return bloom_index_build_common<true>(
      feature_ids,
      feature_offsets,
      feature_values,
      b_multiplier,
      -1,
      -1,
      k,
      fast_build,
      nullptr);
}

TORCH_LIBRARY_IMPL(st, CUDA, m) {
  m.impl(
      "bloom_index_build",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(bloom_index_build_cuda)));
}

} // namespace bloom_indexer
} // namespace ops
} // namespace st

#undef BLOOM_INDEX_CUDA
