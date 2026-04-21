# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import silvertorch.ops._load_ops  # noqa: F401
import torch
import torch.nn as nn


class FilterQueryParserModule(nn.Module):
    """Module wrapping expression query parser for use in model pipelines.

    Parses text expressions into query plan tensors that can be used with
    BloomIndexSearchModule.

    Args:
        hash_k: Number of pre-calculated one-bit hashes.
        max_sub_queries: Maximum number of sub-queries per search.
    """

    def __init__(
        self,
        hash_k: int,
        max_sub_queries: int = 5,
    ) -> None:
        super().__init__()
        self.hash_k = hash_k
        self.max_sub_queries = max_sub_queries

    def forward(
        self,
        expressions: list[str],
        silvertorch_ks: torch.Tensor,
    ) -> tuple[int, list[torch.Tensor]]:
        """Parse text expressions into query plans.

        Args:
            expressions: List of expression strings to parse.
            silvertorch_ks: Per-query k values. Dtype int64.

        Returns:
            Tuple of (max_stack_size, [plans_data, plans_offsets]).
        """
        return torch.ops.st.parse_expression_query_batch(
            expressions,
            silvertorch_ks,
            self.hash_k,
            True,
            self.max_sub_queries,
        )
