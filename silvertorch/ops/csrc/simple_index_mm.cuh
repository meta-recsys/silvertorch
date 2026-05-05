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

#include <torch/torch.h> // @manual
#include "index_mm_helpers.cuh" // @manual

#ifndef __HIP_PLATFORM_AMD__
#include <cuda.h> // @manual
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#else
#include <hip_bf16.h> // @manual
#include <hip_fp16.h> // @manual
#include <hip_runtime.h> // @manual
#endif

namespace st::ops {

namespace simple_index_mm {

[[maybe_unused]] constexpr int32_t c_INVALID_DIVISOR_FOR_INT8 = -1;

template <typename T>
struct cuda_mm_types;

template <>
struct cuda_mm_types<float> {
  using VECTOR_LOAD_TYPE = float4;
  static constexpr float C_MIN_NEG = -FLT_MAX;

  static __inline__ __device__ float init() {
    return 0.0f;
  }

  static __inline__ __device__ float
  multi_add(float embedding, float query, float result) {
    return fmaf(embedding, query, result);
  }

  static __inline__ __device__ float final_result(float result) {
    return result;
  }
};

template <typename DIVISOR_T>
inline __device__ auto resolve_divisor(DIVISOR_T divisor, int64_t index) {
  if constexpr (std::is_same_v<DIVISOR_T, int32_t>) {
    return divisor;
  } else if constexpr (std::is_same_v<DIVISOR_T, c10::Half*>) {
    return *(divisor + index);
  } else {
    printf("Should never reach here, %s\n", __PRETTY_FUNCTION__);
    return static_cast<int32_t>(0);
  }
}

#ifdef __HIP_PLATFORM_AMD__
static __device__ __forceinline__ int __dp4a(const int a, const int b, int c) {
  return __builtin_amdgcn_sdot4(a, b, c, false);
}
#endif

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 610 || !defined(__CUDA_ARCH__))
template <>
struct cuda_mm_types<int8_t> {
  using VECTOR_LOAD_TYPE = int4;
  static __inline__ __device__ int32_t init() {
    return 0;
  }

  static __inline__ __device__ int32_t
  multi_add(int32_t embedding, int32_t query, int32_t result) {
    return __dp4a(embedding, query, result);
  }

  static __inline__ __device__ int32_t final_result(int32_t result) {
    return result;
  }
};
#endif

template <>
struct cuda_mm_types<c10::Half> {
  using VECTOR_LOAD_TYPE = float4;
  static constexpr c10::Half C_MIN_NEG =
      c10::Half(0xFBFF, c10::Half::from_bits());

  static __inline__ __device__ __half2 init() {
    return __float2half2_rn(0.0f);
  }

  static __inline__ __device__ __half2
  multi_add(float embedding, float query, __half2 result) {
    return __hfma2(
        reinterpret_cast<__half2&>(embedding),
        reinterpret_cast<__half2&>(query),
        result);
  }

  static __inline__ __device__ c10::Half final_result(__half2 result) {
    return c10::Half(__hadd(result.x, result.y));
  }
};

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
struct cuda_mm_types<c10::BFloat16> {
  using VECTOR_LOAD_TYPE = float4;
  static constexpr c10::BFloat16 C_MIN_NEG =
      c10::BFloat16(0xFF7F, c10::BFloat16::from_bits());

  static __inline__ __device__ __nv_bfloat162 init() {
#ifndef __HIP_PLATFORM_AMD__
    return __float2bfloat162_rn(0.0f);
#else
    return __bfloat162bfloat162(__float2bfloat16(0.0f));
#endif
  }

  static __inline__ __device__ __nv_bfloat162
  multi_add(float embedding, float query, __nv_bfloat162 result) {
    return __hfma2(
        reinterpret_cast<__nv_bfloat162&>(embedding),
        reinterpret_cast<__nv_bfloat162&>(query),
        result);
  }

  static __inline__ __device__ c10::BFloat16 final_result(
      __nv_bfloat162 result) {
    return c10::BFloat16(__hadd(result.x, result.y));
  }
};
#endif

template <>
struct cuda_mm_types<int32_t> {
  static constexpr int32_t C_MIN_NEG = INT_MIN;
};

template <>
struct cuda_mm_types<int16_t> {
  static constexpr int16_t C_MIN_NEG = SHRT_MIN;
};

template <
    typename T,
    int DIM,
    int THREADS_PER_LINE,
    bool EMB_ON_LOCAL_MEM,
    bool QUERY_ON_LOCAL_MEM = false>
__inline__ __device__ decltype(auto) dotproduct_block(
    const T* __restrict__ embeddings,
    const T* __restrict__ query) {
  static_assert(
      THREADS_PER_LINE == 1 || THREADS_PER_LINE == 2 || THREADS_PER_LINE == 4 ||
      THREADS_PER_LINE == 8);
  using VECTOR_LOAD_TYPE = typename cuda_mm_types<T>::VECTOR_LOAD_TYPE;
  static constexpr uint64_t VECTOR_LOAD_SIZE = 16 / sizeof(T);
  static_assert(DIM % VECTOR_LOAD_SIZE == 0);

  auto index = threadIdx.x % THREADS_PER_LINE;

  const VECTOR_LOAD_TYPE* __restrict__ embeddings4 =
      reinterpret_cast<const VECTOR_LOAD_TYPE* __restrict__>(embeddings);
  const VECTOR_LOAD_TYPE* __restrict__ query4 =
      reinterpret_cast<const VECTOR_LOAD_TYPE* __restrict__>(query);

  auto result = cuda_mm_types<T>::init();
#pragma unroll
  for (int64_t i = 0; i < DIM / VECTOR_LOAD_SIZE / THREADS_PER_LINE; ++i) {
    VECTOR_LOAD_TYPE embeddings_data;
    if constexpr (EMB_ON_LOCAL_MEM) {
      embeddings_data = embeddings4[i * THREADS_PER_LINE + index];
    } else {
      embeddings_data = __ldg(embeddings4 + i * THREADS_PER_LINE + index);
    }
    VECTOR_LOAD_TYPE query_data;
    if constexpr (QUERY_ON_LOCAL_MEM) {
      query_data = query4[i * THREADS_PER_LINE + index];
    } else {
      query_data = __ldg(query4 + i * THREADS_PER_LINE + index);
    }

#define FLOAT_COMPUTE(PARAM)            \
  result = cuda_mm_types<T>::multi_add( \
      embeddings_data.PARAM, query_data.PARAM, result);

    FLOAT_COMPUTE(x);
    FLOAT_COMPUTE(y);
    FLOAT_COMPUTE(z);
    FLOAT_COMPUTE(w);

#undef FLOAT_COMPUTE
  }

#ifndef __HIP_PLATFORM_AMD__
  if constexpr (THREADS_PER_LINE == 8) {
    result += __shfl_down_sync(0xFFFFFFFF, result, 4);
    result += __shfl_down_sync(0xFFFFFFFF, result, 2);
    result += __shfl_down_sync(0xFFFFFFFF, result, 1);
  }

  if constexpr (THREADS_PER_LINE == 4) {
    result += __shfl_down_sync(0xFFFFFFFF, result, 2);
    result += __shfl_down_sync(0xFFFFFFFF, result, 1);
  }

  if constexpr (THREADS_PER_LINE == 2) {
    result += __shfl_down_sync(0xFFFFFFFF, result, 1);
  }
#endif

  return result;
}

template <
    typename T,
    int DIM,
    int THREADS_PER_LINE,
    bool EMB_ON_LOCAL_MEM,
    bool QUERY_ON_LOCAL_MEM = false>
__inline__ __device__ decltype(auto) dotproduct(
    const T* embeddings,
    const T* query,
    std::enable_if_t<(DIM * sizeof(T) <= 256)>* = nullptr) {
  auto result = dotproduct_block<
      T,
      DIM,
      THREADS_PER_LINE,
      EMB_ON_LOCAL_MEM,
      QUERY_ON_LOCAL_MEM>(embeddings, query);
  return cuda_mm_types<T>::final_result(result);
}

template <typename D>
__inline__ __device__ decltype(auto) add_result(D x, D y) {
  return x + y;
}

#ifdef __HIP_PLATFORM_AMD__
template <>
__inline__ __device__ decltype(auto) add_result<__hip_bfloat162>(
    __hip_bfloat162 x,
    __hip_bfloat162 y) {
  return __hadd2(x, y);
}
#endif

template <
    typename T,
    int DIM,
    int THREADS_PER_LINE,
    bool EMB_ON_LOCAL_MEM,
    bool QUERY_ON_LOCAL_MEM = false>
__inline__ __device__ decltype(auto) dotproduct(
    const T* embeddings,
    const T* query,
    std::enable_if_t<(DIM * sizeof(T) > 256) && DIM % (256 / sizeof(T)) == 0>* =
        nullptr) {
  static constexpr uint64_t UNROLL_MAX_UNIT = 256 / sizeof(T);
  static_assert(DIM % UNROLL_MAX_UNIT == 0);

  auto result = cuda_mm_types<T>::init();
#pragma unroll 1
  for (int i = 0; i < DIM / UNROLL_MAX_UNIT; ++i) {
    result = add_result(
        result,
        dotproduct_block<
            T,
            UNROLL_MAX_UNIT,
            THREADS_PER_LINE,
            EMB_ON_LOCAL_MEM,
            QUERY_ON_LOCAL_MEM>(
            embeddings + i * UNROLL_MAX_UNIT, query + i * UNROLL_MAX_UNIT));
  }
  return cuda_mm_types<T>::final_result(result);
}

template <
    typename T,
    int DIM,
    int THREADS_PER_LINE,
    bool EMB_ON_LOCAL_MEM,
    bool QUERY_ON_LOCAL_MEM = false>
__inline__ __device__ decltype(auto) dotproduct(
    const T* embeddings,
    const T* query,
    std::enable_if_t<(DIM * sizeof(T) > 256) && DIM % (256 / sizeof(T)) != 0>* =
        nullptr) {
  static constexpr uint64_t UNROLL_MAX_UNIT = 384 / sizeof(T);
  static_assert(DIM % UNROLL_MAX_UNIT == 0);

  auto result = cuda_mm_types<T>::init();
#pragma unroll 1
  for (int i = 0; i < DIM / UNROLL_MAX_UNIT; ++i) {
    result = add_result(
        result,
        dotproduct_block<
            T,
            UNROLL_MAX_UNIT,
            THREADS_PER_LINE,
            EMB_ON_LOCAL_MEM,
            QUERY_ON_LOCAL_MEM>(
            embeddings + i * UNROLL_MAX_UNIT, query + i * UNROLL_MAX_UNIT));
  }
  return cuda_mm_types<T>::final_result(result);
}

template <
    typename T,
    typename RETURN_T,
    int DIM,
    int THREADS_PER_LINE,
    bool EMB_ON_LOCAL_MEM = false,
    typename DIVISOR_T = int32_t,
    bool QUERY_ON_LOCAL_MEM = false>
__inline__ __device__ RETURN_T get_score(
    const T* embeddings,
    const T* queries_ptr,
    DIVISOR_T divisor_for_int8) {
  auto score = dotproduct<
      T,
      DIM,
      THREADS_PER_LINE,
      EMB_ON_LOCAL_MEM,
      QUERY_ON_LOCAL_MEM>(embeddings, queries_ptr);
  return post_process<decltype(score), RETURN_T, DIVISOR_T>(
      score, divisor_for_int8);
}

} // namespace simple_index_mm

#define EMBEDDING_DIMS_ENUM(                                            \
    EMB_ATEN_T,                                                         \
    EMB_NATIVE_T,                                                       \
    RET_ATEN_T,                                                         \
    RET_NATIVE_T,                                                       \
    INVOKE,                                                             \
    DIM_ITER_FUNCTOR)                                                   \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 16, INVOKE)   \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 32, INVOKE)   \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 64, INVOKE)   \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 96, INVOKE)   \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 128, INVOKE)  \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 192, INVOKE)  \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 256, INVOKE)  \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 384, INVOKE)  \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 512, INVOKE)  \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 768, INVOKE)  \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 1024, INVOKE) \
  DIM_ITER_FUNCTOR(                                                     \
      EMB_ATEN_T, EMB_NATIVE_T, RET_ATEN_T, RET_NATIVE_T, 1280, INVOKE)

} // namespace st::ops
