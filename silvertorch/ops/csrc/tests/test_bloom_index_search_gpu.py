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

# Keep these on single lines so the @oss-disable sed transform commenting works.
# fmt: off
# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:bloom_indexer_gpu")
# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:bloom_index_search_gpu")
# fmt: on


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestBloomIndexSearchGpu(unittest.TestCase):
    def _make_features(
        self, num_docs: int, num_features: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_ids = torch.arange(num_features, dtype=torch.int32)
        values = []
        offsets = [0]
        for doc in range(num_docs):
            for feat in range(num_features):
                values.append(doc * 100 + feat)
                offsets.append(len(values))
        feature_values = torch.tensor(values, dtype=torch.long)
        feature_offsets = torch.tensor(offsets, dtype=torch.long)
        return feature_ids, feature_offsets, feature_values

    def test_search_gpu_basic(self) -> None:
        """Build on GPU, parse on CPU, search on GPU."""
        hash_k = 7
        k = 3
        feature_ids, feature_offsets, feature_values = self._make_features(10, 3)

        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids.cuda(),
            feature_offsets.cuda(),
            feature_values.cuda(),
            2.0,
            hash_k,
        )

        silvertorch_ks = torch.ones(1, dtype=torch.long)
        _, tensors = torch.ops.st.parse_expression_query_batch(
            ["0:0"], silvertorch_ks, hash_k, True, 5
        )

        # Plans are on CPU, move to GPU for search
        plans_data = tensors[0].cuda()
        plans_offsets = tensors[1].cuda()

        result = torch.ops.st.bloom_index_search_batch(
            bloom_index, bundle_b_offsets, plans_data, plans_offsets, k, hash_k, True
        )

        self.assertTrue(result.is_cuda)
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result.size(0), 1)
        self.assertEqual(result.dtype, torch.bool)

    def test_search_gpu_batch(self) -> None:
        """Batch queries on GPU."""
        hash_k = 7
        k = 3
        feature_ids, feature_offsets, feature_values = self._make_features(10, 3)

        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids.cuda(),
            feature_offsets.cuda(),
            feature_values.cuda(),
            2.0,
            hash_k,
        )

        silvertorch_ks = torch.ones(1, dtype=torch.long)
        _, tensors = torch.ops.st.parse_expression_query_batch(
            ["0:0", "1:1", "0:0 AND 1:1"], silvertorch_ks, hash_k, True, 5
        )

        result = torch.ops.st.bloom_index_search_batch(
            bloom_index,
            bundle_b_offsets,
            tensors[0].cuda(),
            tensors[1].cuda(),
            k,
            hash_k,
            True,
        )

        self.assertTrue(result.is_cuda)
        self.assertEqual(result.size(0), 3)

    def test_search_cpu_gpu_consistent(self) -> None:
        """CPU and GPU search should produce identical results."""
        hash_k = 7
        k = 3
        feature_ids, feature_offsets, feature_values = self._make_features(10, 2)

        # Build on CPU
        bloom_cpu, offsets_cpu = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, hash_k
        )

        silvertorch_ks = torch.ones(1, dtype=torch.long)
        _, tensors = torch.ops.st.parse_expression_query_batch(
            ["0:0"], silvertorch_ks, hash_k, True, 5
        )

        result_cpu = torch.ops.st.bloom_index_search_batch(
            bloom_cpu, offsets_cpu, tensors[0], tensors[1], k, hash_k, True
        )

        # Search on GPU with same index
        result_gpu = torch.ops.st.bloom_index_search_batch(
            bloom_cpu.cuda(),
            offsets_cpu.cuda(),
            tensors[0].cuda(),
            tensors[1].cuda(),
            k,
            hash_k,
            True,
        )

        self.assertTrue(torch.equal(result_cpu, result_gpu.cpu()))

    def test_generate_column_info_gpu(self) -> None:
        offsets = torch.tensor([0, 100, 200], dtype=torch.long).cuda()
        lengths = torch.tensor([100, 50, 80], dtype=torch.long).cuda()

        column_counts, start_column_ids, first_item_offsets = (
            torch.ops.st.generate_column_info_for_clusters(offsets, lengths)
        )

        self.assertTrue(column_counts.is_cuda)
        self.assertEqual(column_counts.dtype, torch.int32)
        self.assertEqual(column_counts.numel(), 3)


if __name__ == "__main__":
    unittest.main()
