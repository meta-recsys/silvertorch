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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "expression_query_parser.h"

namespace st::ops::expression_query_parser {

using at::Tensor;

namespace {

std::tuple<int64_t, std::vector<Tensor>> parse_expression_query_batch_cuda(
    const c10::List<std::string>& expressions,
    const Tensor& silvertorch_ks,
    int64_t bloom_hash_k,
    bool return_query_plan,
    int64_t max_sub_queries) {
  std::vector<std::string> expr_strs;
  expr_strs.reserve(expressions.size());
  for (size_t i = 0; i < expressions.size(); ++i) {
    expr_strs.push_back(expressions.get(i));
  }
  return parse_expression_query_batch_cpu(
      expr_strs,
      silvertorch_ks.cpu(),
      bloom_hash_k,
      return_query_plan,
      max_sub_queries);
}

} // namespace

TORCH_LIBRARY_IMPL(st, CUDA, m) {
  m.impl(
      "parse_expression_query_batch",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(parse_expression_query_batch_cuda)));
}

} // namespace st::ops::expression_query_parser
