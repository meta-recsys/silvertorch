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
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

#define BLOOM_INDEX_CUDA
#include "bloom_index_search.cuh"
#include "bloom_index_util.cuh"
#include "bloom_index_util.h"

namespace st {
namespace ops {
namespace bloom_search {

using at::PackedTensorAccessor32;
using at::PackedTensorAccessor64;
using at::Tensor;
using namespace st::ops::bloom_index;

namespace {

static constexpr int32_t C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL = 16;
static constexpr int32_t C_NUM_OFFSETS_PER_PLAN = 8;

template <bool is_bloom_index_v2>
struct QueryPlanCuda {
 public:
  QueryPlanCuda(
      PackedTensorAccessor32<uint8_t, 1> operators,
      PackedTensorAccessor32<int32_t, 1> offsets,
      PackedTensorAccessor32<int32_t, 1> parameters,
      PackedTensorAccessor32<
          typename QueryPlanOneBitsPType<is_bloom_index_v2>::type,
          1> oneBitsPosition)
      : operators(operators),
        offsets(offsets),
        parameters(parameters),
        oneBitsPosition(oneBitsPosition) {}

  PackedTensorAccessor32<uint8_t, 1> operators;
  PackedTensorAccessor32<int32_t, 1> offsets;
  PackedTensorAccessor32<int32_t, 1> parameters;
  PackedTensorAccessor32<
      typename QueryPlanOneBitsPType<is_bloom_index_v2>::type,
      1>
      oneBitsPosition;
};

// This class is created to allocate cuda memory once and copy once to improve
// performance. This class is using visitor pattern. usage:
//   1. call add_size_and_data for all the potential GPU memory allocations.
//   2. call visit to create GPU tensor for all tensors.
//   3. add_size_without_data must be called after all add_size_and_data.
template <bool is_bloom_index_v2>
class TensorCreationVisitor {
 public:
  TensorCreationVisitor(const at::Device& device)
      : m_offset(0), m_size_to_copy(0), m_device(device) {}

  template <typename T>
  void add_size_and_data(const std::vector<T>& vec) {
    TORCH_CHECK(m_cudaVec.numel() == 0);
    using ValueType =
        std::conditional_t<std::is_same_v<T, Operator>, uint8_t, T>;
    auto offset = m_hostVec.size();
    m_hostVec.resize(offset + align_size(vec.size() * sizeof(ValueType)));
    // @lint-ignore CLANGSECURITY facebook-security-vulnerable-memcpy
    std::memcpy(
        m_hostVec.data() + offset, vec.data(), vec.size() * sizeof(ValueType));
    m_size_to_copy = m_hostVec.size();
  }

  void add_size_and_data(const QueryPlan<is_bloom_index_v2>& query_plan) {
    TORCH_CHECK(m_cudaVec.numel() == 0);
    // Sequence here need to be exact same as visit.
    add_size_and_data(query_plan.operators);
    add_size_and_data(query_plan.offsets);
    add_size_and_data(query_plan.parameters);
    add_size_and_data(query_plan.oneBitsPosition);
  }

  void add_size_without_data(size_t size) {
    m_hostVec.resize(m_hostVec.size() + align_size(size));
  }

  template <typename T>
  Tensor visit(const std::vector<T>& vec) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(m_device.index());

    if (m_cudaVec.numel() == 0) {
      TORCH_CHECK(!m_hostVec.empty());
      m_cudaVec = at::empty(
          {static_cast<int64_t>(m_hostVec.size())},
          c10::TensorOptions().dtype(at::kChar).device(m_device),
          at::MemoryFormat::Contiguous);

      AT_CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<void*>(m_cudaVec.mutable_data_ptr<int8_t>()),
          reinterpret_cast<void*>(m_hostVec.data()),
          m_size_to_copy,
          cudaMemcpyHostToDevice,
          at::cuda::getCurrentCUDAStream()));
    }

    return vector_to_tensor<T>(vec);
  }

  // Convert query plan to a variant that uses cuda-compatible data
  // structures.
  QueryPlanCuda<is_bloom_index_v2> visit(
      const QueryPlan<is_bloom_index_v2>& query_plan) {
    // Sequence here need to be exact same as add_size_and_data.
    auto operators = visit(query_plan.operators);
    auto offsets = visit(query_plan.offsets);
    auto parameters = visit(query_plan.parameters);
    auto oneBitsPosition = visit(query_plan.oneBitsPosition);
    return QueryPlanCuda<is_bloom_index_v2>(
        operators.template packed_accessor32<uint8_t, 1>(),
        offsets.template packed_accessor32<int32_t, 1>(),
        parameters.template packed_accessor32<int32_t, 1>(),
        oneBitsPosition.template packed_accessor32<
            typename QueryPlanOneBitsPType<is_bloom_index_v2>::type,
            1>());
  }

  const QueryPlanCuda<is_bloom_index_v2>* visit_with_data(
      const std::vector<QueryPlanCuda<is_bloom_index_v2>>&
          query_plan_cuda_host) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(m_device.index());

    TORCH_CHECK(
        m_offset == m_size_to_copy,
        "add_size_without_data must call only after all add_size_and_data, same as visit_with_data.");

    size_t size_of_obj =
        query_plan_cuda_host.size() * sizeof(QueryPlanCuda<is_bloom_index_v2>);
    size_t size_to_add = align_size(size_of_obj);
    TORCH_CHECK(m_offset + size_to_add <= m_hostVec.size());

    QueryPlanCuda<is_bloom_index_v2>* query_plan_cuda_ptr =
        reinterpret_cast<QueryPlanCuda<is_bloom_index_v2>*>(
            m_cudaVec.data_ptr<int8_t>() + m_offset);
    AT_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(query_plan_cuda_ptr),
        reinterpret_cast<const void*>(query_plan_cuda_host.data()),
        size_of_obj,
        cudaMemcpyHostToDevice,
        at::cuda::getCurrentCUDAStream()));

    m_offset += size_to_add;
    return query_plan_cuda_ptr;
  }

 private:
  // Populate a tensor with data from a vector.
  template <typename T>
  Tensor vector_to_tensor(const std::vector<T>& vec) {
    using ValueType =
        std::conditional_t<std::is_same_v<T, Operator>, uint8_t, T>;

    size_t size_to_add = align_size(vec.size() * sizeof(ValueType));
    TORCH_CHECK(m_offset + size_to_add <= m_hostVec.size());

    Tensor ret = torch::from_blob(
        m_cudaVec.data_ptr<int8_t>() + m_offset,
        {static_cast<int64_t>(vec.size())}, // size
        c10::TensorOptions().dtype(c10::CppTypeToScalarType<ValueType>::value));

    m_offset += size_to_add;
    return ret;
  }

  size_t m_offset;
  size_t m_size_to_copy;
  std::vector<char> m_hostVec;
  Tensor m_cudaVec;
  at::Device m_device;
};

// Prepare query plans from the query plans data.
template <bool is_bloom_index_v2>
__inline__ __device__ uint64_t signature_match(
    const int64_t* __restrict__ bloom_index,
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t column_b,
    const int64_t bundle_b_start,
    int64_t column,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    typename QueryPlanOneBitsPType<is_bloom_index_v2>::type oneBitsPosition[]);

template <>
__inline__ __device__ uint64_t signature_match<false /*is_bloom_v2*/>(
    const int64_t* __restrict__ bloom_index,
    const int64_t total_column_count,
    const int64_t column_b,
    const int64_t bundle_b_start,
    int64_t column,
    int64_t k,
    int64_t /*p_k*/,
    typename QueryPlanOneBitsPType<false /*is_bloom_v2*/>::type
        oneBitsPosition[]) {
  uint64_t result = ~(static_cast<uint64_t>(0));
  for (int64_t i = 0; i < k; ++i) {
    result &= bloom_index
        [static_cast<int64_t>(oneBitsPosition[i]) * total_column_count +
         column];
  }
  return result;
}

template <>
__inline__ __device__ uint64_t signature_match<true>(
    const int64_t* __restrict__ bloom_index,
    const int64_t /*total_column_count*/,
    const int64_t b,
    const int64_t bundle_b_start,
    int64_t column,
    int64_t k,
    int64_t p_k,
    typename QueryPlanOneBitsPType<true>::type oneBitsPosition[]) {
  uint64_t result = ~(static_cast<uint64_t>(0));
  // TODO(zhenwang): get rid of duplicate check here.
  // we use MAX_K_V2 here because I observed that if we use large number
  // here (e.g. MAX_K), the process_op will sometimes get illegal memory
  // access error, meanwhile, the faster_process_op is working fine.
  // My guess is that it may hit stack size limit.
  std::array<typename BIndexType<true>::type, MAX_K_V2> used_b_p = {0};
  int32_t p = 0;
  int32_t resolved = 0;
  int64_t column_in_bundle = column % C_BLOOM_V2_COL_BUNDLE_SIZE;
  while (p < p_k && resolved < k) {
    auto b_p =
        static_cast<typename BIndexType<true>::type>(oneBitsPosition[p++] % b);
    bool duplicate = false;
    for (int32_t i = 0; i < resolved; ++i) {
      if (used_b_p[i] == b_p) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      used_b_p[resolved++] = b_p;
      result &= bloom_index
          [(bundle_b_start + b_p) * C_BLOOM_V2_COL_BUNDLE_SIZE +
           column_in_bundle];
    }
  }
  return result;
}

template <bool is_bloom_index_v2>
__inline__ __device__ int64_t
get_one_bits_position_step(int64_t k, int64_t p_k);

template <>
__inline__ __device__ int64_t
get_one_bits_position_step<false /*is_bloom_v2*/>(int64_t k, int64_t p_k) {
  return k;
}

template <>
__inline__ __device__ int64_t
get_one_bits_position_step<true>(int64_t k, int64_t p_k) {
  return p_k;
}

template <bool is_bloom_index_v2>
__device__ uint64_t process_op(
    const int64_t* __restrict__ bloom_index, // if v1, its 2d, if v2, its 1d.
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t column_b,
    const int64_t bundle_b_start,
    int64_t column,
    int32_t op_index,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    const QueryPlanCuda<is_bloom_index_v2>& query_plan) {
  int32_t param_start = query_plan.offsets[op_index];
  int32_t param_end = query_plan.offsets[op_index + 1];
  if (query_plan.operators[op_index] == Operator::TERM) {
    return signature_match<is_bloom_index_v2>(
        bloom_index,
        total_column_count,
        column_b,
        bundle_b_start,
        column,
        k,
        p_k,
        query_plan.oneBitsPosition.data() +
            (query_plan.parameters[param_start] *
             get_one_bits_position_step<is_bloom_index_v2>(k, p_k)));
  }

  if (query_plan.operators[op_index] == Operator::AND) {
    uint64_t result = ~(static_cast<uint64_t>(0));
    for (; param_start < param_end; ++param_start) {
      result &= process_op<is_bloom_index_v2>(
          bloom_index,
          total_column_count,
          column_b,
          bundle_b_start,
          column,
          query_plan.parameters[param_start],
          k,
          p_k,
          query_plan);
    }
    return result;
  }

  if (query_plan.operators[op_index] == Operator::OR) {
    uint64_t result = 0;
    for (; param_start < param_end; ++param_start) {
      result |= process_op<is_bloom_index_v2>(
          bloom_index,
          total_column_count,
          column_b,
          bundle_b_start,
          column,
          query_plan.parameters[param_start],
          k,
          p_k,
          query_plan);
    }
    return result;
  }

  if (query_plan.operators[op_index] == Operator::NOT) {
    return ~process_op<is_bloom_index_v2>(
        bloom_index,
        total_column_count,
        column_b,
        bundle_b_start,
        column,
        query_plan.parameters[param_start],
        k,
        p_k,
        query_plan);
  }

  if (query_plan.operators[op_index] == Operator::EMPTY) {
    return UINT64_MAX;
  }

  // Shouldn't go here.
  return 0;
}

template <int32_t STACK_SIZE>
class Stack {
 public:
  __device__ Stack() : m_cur(0) {}

  __inline__ __device__ void push(uint64_t value) {
    m_value[m_cur++] = value;
  }

  __inline__ __device__ uint64_t pop() {
    // m_cur can't never be less than 0, since we run the check at
    // get_max_stack_size method.
    return m_value[--m_cur];
  }

 private:
  int32_t m_cur;
  uint64_t m_value[STACK_SIZE];
};

// Use fix size stack as a fast route.
template <int32_t STACK_SIZE, bool is_bloom_index_v2>
__inline__ __device__ uint64_t fast_process_op(
    const int64_t* __restrict__ bloom_index, // if v1, its 2d, if v2, its 1d.
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t column_b,
    const int64_t bundle_b_start,
    int64_t column,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    const QueryPlanCuda<is_bloom_index_v2>& query_plan) {
  Stack<STACK_SIZE> stack;
  int32_t one_bits_position_cur = 0;
  for (size_t j = 0; j < query_plan.operators.size(0); ++j) {
    switch (query_plan.operators[j]) {
      case Operator::TERM: {
        // push to stack.
        stack.push(
            signature_match<is_bloom_index_v2>(
                bloom_index,
                total_column_count,
                column_b,
                bundle_b_start,
                column,
                k,
                p_k,
                query_plan.oneBitsPosition.data() +
                    (one_bits_position_cur++ *
                     get_one_bits_position_step<is_bloom_index_v2>(k, p_k))));
        break;
      }

      case Operator::AND: {
        uint64_t result = ~(static_cast<uint64_t>(0));
        int32_t children = query_plan.offsets[j + 1] - query_plan.offsets[j];
        for (int32_t i = 0; i < children; ++i) {
          result &= stack.pop();
        }
        stack.push(result);
        break;
      }

      case Operator::OR: {
        uint64_t result = 0;
        int32_t children = query_plan.offsets[j + 1] - query_plan.offsets[j];
        for (int32_t i = 0; i < children; ++i) {
          result |= stack.pop();
        }
        stack.push(result);
        break;
      }

      case Operator::NOT: {
        stack.push(~stack.pop());
        break;
      }

      case Operator::EMPTY: {
        stack.push(UINT64_MAX);
        break;
      }
    }
  }

  return stack.pop();
}

template <int32_t STACK_SIZE, bool is_bloom_index_v2>
__device__ uint64_t run_query_plan(
    const int64_t* __restrict__ bloom_index, // if v1, its 2d, if v2, its 1d.
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t* __restrict__ bundle_b_offsets, // useful for bloom index v2
                                                  // only.
    int64_t column,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    const QueryPlanCuda<is_bloom_index_v2>& query_plan) {
  int64_t b = -1;
  int64_t bundle_b_start = -1;
  if constexpr (is_bloom_index_v2) {
    int64_t bundle = column / C_BLOOM_V2_COL_BUNDLE_SIZE;
    bundle_b_start = __ldg(bundle_b_offsets + bundle);
    b = __ldg(bundle_b_offsets + bundle + 1) - bundle_b_start;
  }
  return fast_process_op<STACK_SIZE, is_bloom_index_v2>(
      bloom_index,
      total_column_count,
      b,
      bundle_b_start,
      column,
      k,
      p_k,
      query_plan);
}

// fall back to recursive if requires stack size is too large.
// We use 0 to indicate this.
template <>
__device__ uint64_t run_query_plan<0, true>(
    const int64_t* __restrict__ bloom_index, // if v1, its 2d, if v2, its 1d.
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t* __restrict__ bundle_b_offsets, // useful for bloom index v2
                                                  // only.
    int64_t column,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    const QueryPlanCuda<true>& query_plan) {
  // Start from last one, and then recursive from backward.
  int32_t root_op_index = query_plan.operators.size(0) - 1;
  int64_t bundle = column / C_BLOOM_V2_COL_BUNDLE_SIZE;
  int64_t bundle_b_start = __ldg(bundle_b_offsets + bundle);
  int64_t b = __ldg(bundle_b_offsets + bundle + 1) - bundle_b_start;
  return process_op<true>(
      bloom_index,
      total_column_count,
      b,
      bundle_b_start,
      column,
      root_op_index,
      k,
      p_k,
      query_plan);
}
template <>
__device__ uint64_t run_query_plan<0, false /*is_bloom_v2*/>(
    const int64_t* bloom_index, // if v1, its 2d, if v2, its 1d.
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t* bundle_b_offsets, // useful for bloom index v2 only.
    int64_t column,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    const QueryPlanCuda<false /*is_bloom_v2*/>& query_plan) {
  // Start from last one, and then recursive from backward.
  int32_t root_op_index = query_plan.operators.size(0) - 1;
  return process_op<false /*is_bloom_v2*/>(
      bloom_index,
      total_column_count,
      -1,
      -1,
      column,
      root_op_index,
      k,
      p_k,
      query_plan);
}

union MaskBlock {
  struct {
    float4 data1;
    float4 data2;
    float4 data3;
    float4 data4;
  };
  bool bits[C_BITS_IN_UINT64];
};

template <typename RETURN_T>
__device__ __inline__ void
store(RETURN_T* dest, uint64_t result, int64_t column) {
  reinterpret_cast<uint64_t*>(dest)[column] = result;
}

template <>
__device__ __inline__ void
store<bool>(bool* dest, uint64_t result, int64_t column) {
  //     lower doc id put at higher bits in unit64_t, for example:
  //     |< high                                       low >|
  //      Doc0, Doc1, Doc2 ...... Doc 60, Doc61, Doc62, Doc63
  MaskBlock mask;
#pragma unroll
  for (int i = 0; i < C_BITS_IN_UINT64; i++) {
    static constexpr uint64_t C_MASK = 1ULL << 63;
    mask.bits[i] = ((result & C_MASK) != 0);
    result <<= 1;
  }

  reinterpret_cast<MaskBlock*>(dest)[column] = mask;
}

template <typename RETURN_T, int32_t STACK_SIZE, bool is_bloom_index_v2>
__global__ void process_documents(
    const int64_t* bloom_index, // if v1, its 2d, if v2, its 1d.
    const bool* column_mask_ptr,
    // total_column_count is
    //   bloom_index.size(1) for v1
    //   (bundle_b_offsets.numel()-1)*C_BLOOM_V2_COL_BUNDLE_SIZE for v2.
    const int64_t total_column_count,
    const int64_t* bundle_b_offsets, // useful for bloom index v2 only.
    PackedTensorAccessor64<RETURN_T, 2> document_mask,
    const QueryPlanCuda<is_bloom_index_v2>* query_plan,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    TORCH_DSA_KERNEL_ARGS) {
  int64_t row_count = document_mask.size(0);
  int64_t total_document_mask_block = row_count * total_column_count;

  // Stride loop:
  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for (int64_t document_index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       document_index < total_document_mask_block;
       document_index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    int64_t row = document_index / total_column_count;
    int64_t column = document_index % total_column_count;

    if (column_mask_ptr == nullptr || column_mask_ptr[column]) {
      uint64_t result = run_query_plan<STACK_SIZE, is_bloom_index_v2>(
          bloom_index,
          total_column_count,
          bundle_b_offsets,
          column,
          k,
          p_k,
          query_plan[row]);

      store<RETURN_T>(
          document_mask.data() + row * document_mask.size(1), result, column);
    }
  }
}

__global__ void generate_column_info_for_clusters_kernel(
    const int64_t* __restrict__ selected_cluster_offsets_ptr,
    const int64_t* __restrict__ selected_cluster_lengths_ptr,
    int64_t total_cluster_count,
    int32_t* column_counts,
    int32_t* start_column_ids,
    int8_t* first_item_offsets_in_column,
    TORCH_DSA_KERNEL_ARGS) {
  // Stride loop:
  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for (int64_t cluster_index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cluster_index < total_cluster_count;
       cluster_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t local_cluster_len = selected_cluster_lengths_ptr[cluster_index];
    int64_t local_cluster_start = selected_cluster_offsets_ptr[cluster_index];
    int32_t start_col_id =
        static_cast<int32_t>(local_cluster_start / C_BITS_IN_UINT64);
    int8_t offset_in_col =
        static_cast<int8_t>(local_cluster_start % C_BITS_IN_UINT64);
    start_column_ids[cluster_index] = start_col_id;
    first_item_offsets_in_column[cluster_index] = offset_in_col;
    column_counts[cluster_index] = local_cluster_len > 0
        ? static_cast<int32_t>(
              (local_cluster_len + offset_in_col + C_BITS_IN_UINT64 - 1) /
              C_BITS_IN_UINT64)
        : 0;
  }
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> generate_column_info_for_clusters(
    const Tensor& selected_cluster_offsets,
    const Tensor& selected_cluster_lengths) {
  // this funciton will be just one step of a kernel, so I didn't add
  // device guard in this function, the guard will be the caller
  // function's responsibility.
  constexpr int64_t kThreads = 256L;
  int64_t kBlocks = 128L *
      CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
          ->multiProcessorCount;
  int64_t total_cluster_count = selected_cluster_lengths.numel();
  int64_t block_size = total_cluster_count <= 128 ? 128L : kThreads;
  int64_t grid_size =
      std::min(kBlocks, (total_cluster_count + block_size - 1) / block_size);
  auto device = selected_cluster_lengths.device();
  Tensor column_counts = at::empty(
      {total_cluster_count},
      c10::TensorOptions().dtype(at::kInt).device(device));
  Tensor start_column_ids = at::empty(
      {total_cluster_count},
      c10::TensorOptions().dtype(at::kInt).device(device));
  Tensor first_item_offsets_in_column = at::empty(
      {total_cluster_count},
      c10::TensorOptions().dtype(at::kChar).device(device));
  at::cuda::CUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_DSA_KERNEL_LAUNCH(
      generate_column_info_for_clusters_kernel,
      grid_size,
      block_size,
      0,
      stream,
      selected_cluster_offsets.data_ptr<int64_t>(),
      selected_cluster_lengths.data_ptr<int64_t>(),
      total_cluster_count,
      column_counts.data_ptr<int32_t>(),
      start_column_ids.data_ptr<int32_t>(),
      first_item_offsets_in_column.data_ptr<int8_t>());
  return std::make_tuple(
      std::move(column_counts),
      std::move(start_column_ids),
      std::move(first_item_offsets_in_column));
}

template <bool is_bloom_index_v2>
Tensor bloom_index_search_common(
    const Tensor& bloom_index,
    const Tensor*
        bloom_bundle_b_offsets, // only used when is_bloom_index_v2 is true
    const QueryPlanCuda<is_bloom_index_v2>* query_plan_cuda,
    size_t query_plans_size,
    int32_t query_plans_max_stack_size,
    int64_t k,
    int64_t onebit_position_k, // only used when is_bloom_index_v2 is true
    bool return_bool_mask) {
  int64_t* bundle_b_offsets_ptr = nullptr;
  int64_t total_column_count;
  if constexpr (is_bloom_index_v2) {
    TORCH_CHECK(bloom_bundle_b_offsets != nullptr);
    bundle_b_offsets_ptr = bloom_bundle_b_offsets->data_ptr<int64_t>();
    total_column_count =
        (bloom_bundle_b_offsets->numel() - 1) * C_BLOOM_V2_COL_BUNDLE_SIZE;
  } else {
    total_column_count = bloom_index.size(1);
  }
  constexpr int64_t kThreads = 256L;
  auto block_size = kThreads;
  auto grid_size = std::min(
      (total_column_count * static_cast<int64_t>(query_plans_size) +
       block_size - 1) /
          block_size,
      128L *
          CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
              ->multiProcessorCount);

  int64_t document_mask_alloc_size;
  at::ScalarType document_mask_alloc_type;
  if (return_bool_mask) {
    document_mask_alloc_size =
        static_cast<int64_t>(total_column_count * C_BITS_IN_UINT64);
    document_mask_alloc_type = at::kBool;
  } else {
    document_mask_alloc_size = static_cast<int64_t>(total_column_count);
    document_mask_alloc_type = at::kLong;
  }

  Tensor document_mask;
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(bloom_index.device().index());
  document_mask = at::empty(
      {static_cast<int64_t>(query_plans_size), document_mask_alloc_size},
      c10::TensorOptions()
          .dtype(document_mask_alloc_type)
          .device(bloom_index.device()));

#define LAUNCH_KERNEL_WITH_RETURN_TYPE_STACK_SIZE(RETURN_TYPE, STACK_SIZE) \
  {                                                                        \
    TORCH_DSA_KERNEL_LAUNCH(                                               \
        (process_documents<RETURN_TYPE, STACK_SIZE, is_bloom_index_v2>),   \
        grid_size,                                                         \
        block_size,                                                        \
        0,                                                                 \
        at::cuda::getCurrentCUDAStream(),                                  \
        bloom_index.data_ptr<int64_t>(),                                   \
        nullptr,                                                           \
        total_column_count,                                                \
        bundle_b_offsets_ptr,                                              \
        document_mask.packed_accessor64<RETURN_TYPE, 2>(),                 \
        query_plan_cuda,                                                   \
        k,                                                                 \
        onebit_position_k);                                                \
  }

#define LAUNCH_KERNEL_WITH_RETURN_TYPE(RETURN_TYPE)                          \
  {                                                                          \
    if (query_plans_max_stack_size <= C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL) { \
      LAUNCH_KERNEL_WITH_RETURN_TYPE_STACK_SIZE(                             \
          RETURN_TYPE, C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL);                 \
    } else {                                                                 \
      LAUNCH_KERNEL_WITH_RETURN_TYPE_STACK_SIZE(RETURN_TYPE, 0);             \
    }                                                                        \
  }

  if (return_bool_mask) {
    LAUNCH_KERNEL_WITH_RETURN_TYPE(bool);
  } else {
    LAUNCH_KERNEL_WITH_RETURN_TYPE(int64_t);
  }

#undef LAUNCH_KERNEL_WITH_RETURN_TYPE_STACK_SIZE
#undef LAUNCH_KERNEL_WITH_RETURN_TYPE

  return document_mask;
}

Tensor bloom_index_search_batch_cuda(
    const Tensor& bloom_index,
    const Tensor& bloom_bundle_b_offsets,
    const Tensor& bloom_query_plans_data,
    const Tensor& bloom_query_plans_offsets,
    int64_t k,
    int64_t onebit_hash_k,
    bool return_bool_mask) {
  TORCH_CHECK(bloom_index.is_cuda());
  TORCH_CHECK(bloom_index.dim() == 1);
  TORCH_CHECK(bloom_index.is_contiguous());

  // Move plans to CPU for decoding (decode_query_plan uses from_blob +
  // data_ptr)
  auto plans_data_cpu = bloom_query_plans_data.cpu();
  auto plans_offsets_cpu = bloom_query_plans_offsets.cpu();

  int64_t query_plans_size = plans_offsets_cpu.numel() / 8;
  TensorCreationVisitor<true> tensor_creation_visitor(bloom_index.device());

  std::vector<QueryPlan<true>> query_plans;
  for (int64_t i = 0; i < query_plans_size; ++i) {
    Tensor operators_t, offsets_t, parameters_t, oneBitsPosition_t;
    decode_query_plan<true>(
        plans_data_cpu,
        plans_offsets_cpu,
        i * 8,
        operators_t,
        offsets_t,
        parameters_t,
        oneBitsPosition_t);
    QueryPlan<true> plan;
    plan.operators.clear();
    plan.offsets.clear(); // remove default {0} from constructor
    auto* ops_ptr = operators_t.data_ptr<uint8_t>();
    for (int64_t j = 0; j < operators_t.numel(); ++j) {
      plan.operators.push_back(static_cast<Operator>(ops_ptr[j]));
    }
    auto* offsets_ptr = offsets_t.data_ptr<int32_t>();
    for (int64_t j = 0; j < offsets_t.numel(); ++j) {
      plan.offsets.push_back(offsets_ptr[j]);
    }
    auto* params_ptr = parameters_t.data_ptr<int32_t>();
    for (int64_t j = 0; j < parameters_t.numel(); ++j) {
      plan.parameters.push_back(params_ptr[j]);
    }
    auto* onebits_ptr =
        reinterpret_cast<const uint64_t*>(oneBitsPosition_t.data_ptr());
    for (int64_t j = 0; j < oneBitsPosition_t.numel(); ++j) {
      plan.oneBitsPosition.push_back(onebits_ptr[j]);
    }
    query_plans.push_back(std::move(plan));
  }

  int32_t query_plans_max_stack_size = get_max_stack_size<true>(query_plans);
  for (int64_t i = 0; i < query_plans_size; ++i) {
    tensor_creation_visitor.add_size_and_data(query_plans[i]);
  }
  tensor_creation_visitor.add_size_without_data(
      query_plans_size * sizeof(QueryPlanCuda<true>));

  std::vector<QueryPlanCuda<true>> query_plan_cudas_host;
  for (int64_t i = 0; i < query_plans_size; ++i) {
    query_plan_cudas_host.push_back(
        tensor_creation_visitor.visit(query_plans[i]));
  }
  const QueryPlanCuda<true>* query_plan_cuda =
      tensor_creation_visitor.visit_with_data(query_plan_cudas_host);

  return bloom_index_search_common<true>(
      bloom_index,
      &bloom_bundle_b_offsets,
      query_plan_cuda,
      query_plans_size,
      query_plans_max_stack_size,
      k,
      onebit_hash_k,
      return_bool_mask);
}

static std::tuple<Tensor, Tensor, Tensor>
generate_column_info_for_clusters_cuda(
    const Tensor& selected_cluster_offsets,
    const Tensor& selected_cluster_lengths) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(selected_cluster_offsets.get_device());
  auto [column_counts, start_column_ids, first_item_offsets_in_column] =
      generate_column_info_for_clusters(
          selected_cluster_offsets, selected_cluster_lengths);
  return std::make_tuple(
      column_counts, start_column_ids, first_item_offsets_in_column);
}

// Register operators
TORCH_LIBRARY_IMPL(st, CUDA, m) {
  m.impl(
      "generate_column_info_for_clusters",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(generate_column_info_for_clusters_cuda)));
}

TORCH_LIBRARY_IMPL(st, CUDA, m) {
  m.impl(
      "bloom_index_search_batch",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(bloom_index_search_batch_cuda)));
}
} // namespace bloom_search
} // namespace ops
} // namespace st
