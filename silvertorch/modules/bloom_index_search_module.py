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


class BloomIndexSearchModule(nn.Module):
    """Module wrapping bloom index search for use in model pipelines.

    Stores a bloom index as a buffer and provides a search method that takes
    pre-encoded query plan tensors.

    Args:
        k: Number of hash functions for bloom filter lookup.
        hash_k: Number of pre-calculated one-bit hashes.
    """

    def __init__(self, k: int, hash_k: int) -> None:
        super().__init__()
        self.k = k
        self.hash_k = hash_k
        self.register_buffer("bloom_index", torch.empty(0, dtype=torch.int64))
        self.register_buffer(
            "bloom_bundle_b_offsets", torch.empty(0, dtype=torch.int64)
        )

    def set_bloom_index(self, bloom_index: torch.Tensor) -> None:
        self.bloom_index = bloom_index

    def set_bloom_bundle_b_offsets(self, bloom_bundle_b_offsets: torch.Tensor) -> None:
        self.bloom_bundle_b_offsets = bloom_bundle_b_offsets

    def forward(
        self,
        bloom_query_plans_data: torch.Tensor,
        bloom_query_plans_offsets: torch.Tensor,
        return_bool_mask: bool = True,
    ) -> torch.Tensor:
        """Search the bloom index with pre-encoded query plans.

        Args:
            bloom_query_plans_data: Encoded query plan data tensor (kChar).
            bloom_query_plans_offsets: Encoded query plan offsets tensor (kInt64).
            return_bool_mask: If True, return bool mask; otherwise return bitmask.

        Returns:
            Result tensor with matching documents.
        """
        return torch.ops.st.bloom_index_search_batch(
            self.bloom_index,
            self.bloom_bundle_b_offsets,
            bloom_query_plans_data,
            bloom_query_plans_offsets,
            self.k,
            self.hash_k,
            return_bool_mask,
        )
