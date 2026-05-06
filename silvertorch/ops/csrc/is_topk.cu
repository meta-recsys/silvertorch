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
#include <ATen/ceil_div.h> // @manual=//caffe2:ATen-cpu
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h> // @manual
#include <torch/torch.h> // @manual
#include <ATen/cuda/DeviceUtils.cuh>

namespace st::ops::is_topk {

using at::Tensor;

namespace {

// Per-row kernel: find the k-th value via sorting, then write 1/0 mask.
// This is a self-contained alternative to the jagged_scores_topk
// dependency used in the upstream kernel.
template <typename SCORE_T>
__global__ void gather_is_topk(
    SCORE_T* scores,
    SCORE_T* kth_values,
    int64_t row_count,
    int64_t col_count,
    bool largest,
    SCORE_T* is_topk_output,
    TORCH_DSA_KERNEL_ARGS) {
  auto total_elements = row_count * col_count;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_elements;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    auto row_idx = idx / col_count;

    float score_val = static_cast<float>(scores[idx]);
    float kth_val = static_cast<float>(kth_values[row_idx]);

    bool result;
    if (largest) {
      result = score_val >= kth_val;
    } else {
      result = score_val <= kth_val;
    }

    is_topk_output[idx] = static_cast<SCORE_T>(result);
  }
}

#define AT_DISPATCH_SCORE_TYPES(TYPE, NAME, ...)              \
  AT_DISPATCH_SWITCH(                                         \
      TYPE,                                                   \
      NAME,                                                   \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                        \
          at::ScalarType::Float, score_t, __VA_ARGS__)        \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                    \
              at::ScalarType::BFloat16, score_t, __VA_ARGS__) \
              AT_PRIVATE_CASE_TYPE_USING_HINT(                \
                  at::ScalarType::Half, score_t, __VA_ARGS__))

// Find the k-th value per row using at::sort.  Self-contained -- no
// dependency on jagged_scores_topk_cuda.
void get_kth_values_2d(
    const Tensor& scores,
    const Tensor& ks,
    bool largest,
    Tensor& kth_values) {
  // Sort each row (descending if largest, ascending if smallest).
  auto [sorted, _indices] = scores.sort(/*dim=*/1, /*descending=*/largest);

  // ks is per-row; clamp to valid range and gather the k-th element.
  auto ks_clamped =
      (ks - 1).clamp(0, scores.size(1) - 1).to(at::kLong).unsqueeze(1);
  auto gathered = sorted.gather(/*dim=*/1, ks_clamped).squeeze(1);
  kth_values.copy_(gathered);
}

} // namespace

void is_topk_cuda(
    const Tensor& scores,
    const Tensor& ks,
    Tensor& is_topk_output,
    const bool largest) {
  TORCH_CHECK(scores.is_cuda(), "scores must be on CUDA");
  TORCH_CHECK(ks.is_cuda(), "ks must be on CUDA");
  TORCH_CHECK(is_topk_output.is_cuda(), "is_topk_output must be on CUDA");
  TORCH_CHECK(scores.dim() == 2, "scores must be 2D tensor");
  TORCH_CHECK(ks.dim() == 1, "ks must be 1D tensor");
  TORCH_CHECK(
      ks.size(0) == scores.size(0), "ks size must match scores batch size");
  TORCH_CHECK(
      is_topk_output.is_contiguous(), "is_topk_output must be contiguous");
  TORCH_CHECK(
      is_topk_output.size(0) == scores.size(0) &&
          is_topk_output.size(1) == scores.size(1),
      "is_topk_output must have same shape as scores");

  auto kth_values = at::empty(
      scores.size(0),
      at::TensorOptions().dtype(scores.dtype()).device(scores.device()));

  get_kth_values_2d(scores, ks, largest, kth_values);

  constexpr int64_t kThreadsPerBlock = 256;
  auto total_elements = scores.size(0) * scores.size(1);
  auto grid_size = at::ceil_div(total_elements, kThreadsPerBlock);

  AT_DISPATCH_SCORE_TYPES(scores.scalar_type(), "gather_is_topk", [&]() {
    TORCH_DSA_KERNEL_LAUNCH(
        (gather_is_topk<score_t>),
        grid_size,
        kThreadsPerBlock,
        0,
        at::cuda::getCurrentCUDAStream(),
        scores.data_ptr<score_t>(),
        kth_values.data_ptr<score_t>(),
        scores.size(0),
        scores.size(1),
        largest,
        is_topk_output.data_ptr<score_t>());
  });
}

#undef AT_DISPATCH_SCORE_TYPES

TORCH_LIBRARY_IMPL(st, CUDA, m) {
  m.impl(
      "is_topk",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(is_topk_cuda)));
}

} // namespace st::ops::is_topk
