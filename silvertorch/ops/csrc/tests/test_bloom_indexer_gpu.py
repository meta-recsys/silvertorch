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

# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:bloom_indexer_gpu")

COL_BUNDLE_SIZE = 32


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestBloomIndexerGpu(unittest.TestCase):
    def _make_simple_features(
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

    def test_build_bundled_gpu(self) -> None:
        feature_ids, feature_offsets, feature_values = self._make_simple_features(10, 3)
        feature_ids = feature_ids.cuda()
        feature_offsets = feature_offsets.cuda()
        feature_values = feature_values.cuda()

        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, 7
        )

        self.assertTrue(bloom_index.is_cuda)
        self.assertEqual(bloom_index.dim(), 1)
        self.assertEqual(bundle_b_offsets[0].item(), 0)
        self.assertGreater(bundle_b_offsets[-1].item(), 0)

    def test_build_bundled_cpu_gpu_match(self) -> None:
        """CPU and GPU build should produce identical bloom indices."""
        feature_ids, feature_offsets, feature_values = self._make_simple_features(10, 3)
        bloom_cpu, offsets_cpu = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, 7
        )

        bloom_gpu, offsets_gpu = torch.ops.st.bloom_index_build(
            feature_ids.cuda(),
            feature_offsets.cuda(),
            feature_values.cuda(),
            2.0,
            7,
        )

        self.assertTrue(torch.equal(bloom_cpu, bloom_gpu.cpu()))
        self.assertTrue(torch.equal(offsets_cpu, offsets_gpu.cpu()))


if __name__ == "__main__":
    unittest.main()
