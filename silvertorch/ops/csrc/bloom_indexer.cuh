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

#include <ATen/ATen.h>
#include <tuple>

namespace st {
namespace ops {
namespace bloom_indexer {

using at::Tensor;

std::tuple<Tensor, Tensor> bloom_index_build_cuda(
    const Tensor& feature_ids,
    const Tensor& feature_offsets,
    const Tensor& feature_values,
    double b_multiplier,
    int64_t k,
    bool fast_build = false);

} // namespace bloom_indexer
} // namespace ops
} // namespace st
