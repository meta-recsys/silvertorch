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

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/macros/Macros.h>
#include <cuda.h> // @manual
#include <algorithm>
#include "faster_repeat_interleave.cuh"

namespace st::ops::faster_repeat_interleave {

using at::Tensor;

namespace {

template <typename repeat_t, typename consecutive_idx_t>
__global__ static void
consecutive_index_repeat_interleave_with_cumsum_compute_kernel(
    const consecutive_idx_t idx_start,
    const repeat_t* repeat_cumsum_ptr,
    consecutive_idx_t* result_ptr,
    int64_t size,
    int64_t result_size) {
  CUDA_KERNEL_ASSERT(
      result_size == repeat_cumsum_ptr[size - 1] &&
      "result_size must equal repeat_cumsum total");
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (blockDim.x * gridDim.x) / C10_WARP_SIZE;
  int warp_id = idx / C10_WARP_SIZE;
  int tid_in_warp = idx % C10_WARP_SIZE;
  for (int64_t i = warp_id; i < size; i += stride) {
    repeat_t end = repeat_cumsum_ptr[i];
    repeat_t start = (i == 0 ? 0 : repeat_cumsum_ptr[i - 1]);
    CUDA_KERNEL_ASSERT(end >= start && "repeat_cumsum must be non-decreasing");
    for (int64_t j = start + tid_in_warp; j < end; j += C10_WARP_SIZE) {
      result_ptr[j] = idx_start + static_cast<consecutive_idx_t>(i);
    }
  }
}

} // namespace

void consecutive_int64_repeat_interleave_with_cumsum_cuda_raw(
    const int64_t repeats_size,
    const int64_t* repeats_cumsum,
    const int64_t start,
    const int64_t output_size,
    int64_t* output,
    const c10::cuda::CUDAStream& stream) {
  int64_t block = 512;
  int64_t warps_per_block = block / C10_WARP_SIZE;
  int64_t grid = std::min<int64_t>(
      (repeats_size + warps_per_block - 1) / warps_per_block, 2048L);
  consecutive_index_repeat_interleave_with_cumsum_compute_kernel<
      int64_t,
      int64_t><<<grid, block, 0, stream>>>(
      start, repeats_cumsum, output, repeats_size, output_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace st::ops::faster_repeat_interleave
