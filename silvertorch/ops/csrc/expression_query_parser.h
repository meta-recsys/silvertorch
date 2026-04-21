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
#include <string>
#include <tuple>
#include <vector>

namespace st::ops::expression_query_parser {

// Parse a batch of text expressions directly into query plans.
// Output format is identical to parse_filter_query_batch_cpu.
//
// Expression syntax:
//   feature term:  id:value  or  id:value:weight
//   operators:     AND (&), OR (|), NOT (!)
//   grouping:      parentheses ()
//   precedence:    NOT > AND > OR
//
// Example: "(1:100 AND 2:200) OR NOT 3:300"
std::tuple<int64_t, std::vector<at::Tensor>> parse_expression_query_batch_cpu(
    const std::vector<std::string>& expressions,
    const at::Tensor& silvertorch_ks,
    int64_t bloom_hash_k,
    bool return_query_plan,
    int64_t max_sub_queries);

} // namespace st::ops::expression_query_parser
