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

#include <torch/library.h> // @manual
#include <torch/torch.h> // @manual

namespace st::ops::is_topk {

using at::Tensor;

// is_topk: For each row in a 2-D scores tensor, mark the top-k elements
// with 1 and all others with 0.  k is per-row (given by the 1-D ks tensor).
//
// The op writes into a pre-allocated output tensor (same shape/dtype as
// scores) instead of returning a new tensor so the caller controls memory.
//
// Args:
//   scores:  2-D tensor of shape (B, N). Supported dtypes: float32,
//            float16, bfloat16.
//   ks:      1-D int tensor of shape (B,). ks[i] is the number of top
//            elements to select in row i.  When ks[i] >= N every element
//            in that row is selected.
//   is_topk_output:  2-D tensor of shape (B, N), same dtype as scores,
//            pre-allocated by the caller.  Will be filled with 1.0 for
//            selected positions and 0.0 otherwise.
//   largest: if true, select the largest k; if false, select the smallest k.
//
// Returns: nothing (output written in-place to is_topk_output).

static void is_topk_cpu(
    const Tensor& scores,
    const Tensor& ks,
    Tensor& is_topk_output,
    bool largest) {
  TORCH_CHECK(scores.dim() == 2, "scores must be 2D");
  TORCH_CHECK(ks.dim() == 1, "ks must be 1D");
  TORCH_CHECK(ks.size(0) == scores.size(0), "ks batch size must match scores");

  const int64_t batch_size = scores.size(0);
  const int64_t n = scores.size(1);

  is_topk_output.zero_();

  auto scores_f = scores.to(at::kFloat);
  auto output_f = is_topk_output.to(at::kFloat);

  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t k = ks[b].item<int64_t>();
    if (k >= n) {
      is_topk_output[b].fill_(1);
      continue;
    }
    if (k <= 0) {
      continue;
    }
    auto row = scores_f[b];
    auto [sorted_vals, sorted_idx] =
        row.sort(/*stable=*/false, /*dim=*/0, /*descending=*/largest);
    auto kth_val = sorted_vals[k - 1].item<float>();

    for (int64_t j = 0; j < n; ++j) {
      float s = scores_f[b][j].item<float>();
      bool selected = largest ? (s >= kth_val) : (s <= kth_val);
      if (selected) {
        is_topk_output[b][j] = 1;
      }
    }
  }
}

TORCH_LIBRARY_FRAGMENT(st, m) {
  m.def(
      "is_topk("
      "Tensor scores, "
      "Tensor ks, "
      "Tensor is_topk_output, "
      "bool largest = True) "
      "-> ()");
}

TORCH_LIBRARY_IMPL(st, CPU, m) {
  m.impl(
      "is_topk", torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(is_topk_cpu)));
}

} // namespace st::ops::is_topk
