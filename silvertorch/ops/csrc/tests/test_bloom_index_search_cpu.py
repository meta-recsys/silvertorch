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

import unittest

import silvertorch.ops._load_ops  # noqa: F401
import torch


def _build_index_and_search(
    num_docs: int,
    num_features: int,
    expression: str,
    k: int = 3,
    hash_k: int = 7,
    b_multiplier: float = 2.0,
    return_bool_mask: bool = True,
) -> torch.Tensor:
    """Build bloom index from documents, parse expression, and search."""
    feature_ids = torch.arange(num_features, dtype=torch.int32)
    values = []
    offsets = [0]
    for doc in range(num_docs):
        for feat in range(num_features):
            values.append(doc * 100 + feat)
            offsets.append(len(values))

    feature_values = torch.tensor(values, dtype=torch.long)
    feature_offsets = torch.tensor(offsets, dtype=torch.long)

    bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
        feature_ids, feature_offsets, feature_values, b_multiplier, hash_k
    )

    silvertorch_ks = torch.ones(1, dtype=torch.long)
    _, tensors = torch.ops.st.parse_expression_query_batch(
        [expression], silvertorch_ks, hash_k, True, 5
    )

    return torch.ops.st.bloom_index_search_batch(
        bloom_index,
        bundle_b_offsets,
        tensors[0],
        tensors[1],
        k,
        hash_k,
        return_bool_mask,
    )


class TestBloomIndexSearchCpu(unittest.TestCase):
    def test_end_to_end_single_query(self) -> None:
        """Build index → parse → search, verify output shape."""
        # Document 0 has feature (0, 0), document 1 has feature (0, 100), etc.
        result = _build_index_and_search(10, 2, "0:0", return_bool_mask=True)
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result.size(0), 1)
        self.assertEqual(result.dtype, torch.bool)

    def test_end_to_end_bitmask(self) -> None:
        result = _build_index_and_search(10, 2, "0:0", return_bool_mask=False)
        self.assertEqual(result.dtype, torch.long)
        self.assertEqual(result.size(0), 1)

    def test_end_to_end_and_query(self) -> None:
        result = _build_index_and_search(10, 3, "0:0 AND 1:1")
        self.assertEqual(result.size(0), 1)

    def test_end_to_end_or_query(self) -> None:
        result = _build_index_and_search(10, 3, "0:0 OR 1:1")
        self.assertEqual(result.size(0), 1)

    def test_end_to_end_not_query(self) -> None:
        result = _build_index_and_search(10, 3, "NOT 0:0")
        self.assertEqual(result.size(0), 1)

    def test_end_to_end_batch_queries(self) -> None:
        """Multiple queries in one batch."""
        num_docs = 10
        num_features = 3
        hash_k = 7
        k = 3

        feature_ids = torch.arange(num_features, dtype=torch.int32)
        values = []
        offsets = [0]
        for doc in range(num_docs):
            for feat in range(num_features):
                values.append(doc * 100 + feat)
                offsets.append(len(values))
        feature_values = torch.tensor(values, dtype=torch.long)
        feature_offsets = torch.tensor(offsets, dtype=torch.long)

        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, hash_k
        )

        silvertorch_ks = torch.ones(1, dtype=torch.long)
        _, tensors = torch.ops.st.parse_expression_query_batch(
            ["0:0", "1:1", "0:0 AND 1:1"], silvertorch_ks, hash_k, True, 5
        )

        result = torch.ops.st.bloom_index_search_batch(
            bloom_index,
            bundle_b_offsets,
            tensors[0],
            tensors[1],
            k,
            hash_k,
            True,
        )

        self.assertEqual(result.size(0), 3)

    def test_generate_column_info(self) -> None:
        """Test generate_column_info_for_clusters op."""
        offsets = torch.tensor([0, 100, 200], dtype=torch.long)
        lengths = torch.tensor([100, 50, 80], dtype=torch.long)

        column_counts, start_column_ids, first_item_offsets = (
            torch.ops.st.generate_column_info_for_clusters(offsets, lengths)
        )

        self.assertEqual(column_counts.dtype, torch.int32)
        self.assertEqual(start_column_ids.dtype, torch.int32)
        self.assertEqual(column_counts.numel(), 3)


if __name__ == "__main__":
    unittest.main()
