# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import unittest
from unittest import mock

import torch
from silvertorch.ops.clustered_ann import (
    clustered_ann_search,
    selected_cluster_capacity,
)


class TestSelectedClusterCapacity(unittest.TestCase):
    def test_uses_largest_per_query_sum(self) -> None:
        lengths = torch.tensor([[3, 5, 7], [11, 2, 1]], dtype=torch.int64)
        self.assertEqual(selected_cluster_capacity(lengths), 15)

    def test_supports_non_contiguous_input(self) -> None:
        lengths = torch.tensor([[3, 11], [5, 2], [7, 1]], dtype=torch.int64).T
        self.assertFalse(lengths.is_contiguous())
        self.assertEqual(selected_cluster_capacity(lengths), 15)

    def test_empty_batch_has_zero_capacity(self) -> None:
        lengths = torch.empty((0, 4), dtype=torch.int64)
        self.assertEqual(selected_cluster_capacity(lengths), 0)

    def test_requires_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D"):
            selected_cluster_capacity(torch.tensor([1, 2, 3]))


class TestClusteredAnnSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.offsets = torch.tensor([0, 3, 8, 15], dtype=torch.int64)
        self.cluster_ids = torch.tensor([[0, 2], [1, 2]], dtype=torch.int64)
        self.lengths = torch.tensor([[3, 7], [5, 7]], dtype=torch.int64)
        self.embeddings = torch.randn(15, 16)
        self.queries = torch.randn(2, 16)

    @mock.patch("torch.ops.st.fused_kmean_ann", create=True)
    def test_computes_tight_capacity(self, op: mock.Mock) -> None:
        expected = (torch.empty(2, 12), torch.empty(2, 12, dtype=torch.int32))
        op.return_value = expected

        actual = clustered_ann_search(
            self.offsets,
            self.cluster_ids,
            self.lengths,
            self.embeddings,
            self.queries,
        )

        self.assertIs(actual, expected)
        self.assertEqual(op.call_args.args[5], 12)

    @mock.patch("torch.ops.st.fused_kmean_ann", create=True)
    def test_explicit_capacity_avoids_dynamic_calculation(
        self, op: mock.Mock
    ) -> None:
        op.return_value = (torch.empty(0), torch.empty(0, dtype=torch.int32))
        with mock.patch(
            "silvertorch.ops.clustered_ann.selected_cluster_capacity"
        ) as capacity:
            clustered_ann_search(
                self.offsets,
                self.cluster_ids,
                self.lengths,
                self.embeddings,
                self.queries,
                max_tensor_size_per_row=32,
            )

        capacity.assert_not_called()
        self.assertEqual(op.call_args.args[5], 32)

    def test_rejects_negative_explicit_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            clustered_ann_search(
                self.offsets,
                self.cluster_ids,
                self.lengths,
                self.embeddings,
                self.queries,
                max_tensor_size_per_row=-1,
            )


if __name__ == "__main__":
    unittest.main()
