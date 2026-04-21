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

# Bloom index constants
COL_BUNDLE_SIZE = 32


class TestBloomIndexerCpu(unittest.TestCase):
    def _make_simple_features(
        self, num_docs: int, num_features: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create simple feature data for num_docs documents with num_features each.

        Each document i has features (feature_id=j, value=i*100+j) for j in [0, num_features).
        """
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

    def test_build_bundled_basic(self) -> None:
        """Test basic bloom index build with bundled format."""
        num_docs = 10
        num_features = 3
        feature_ids, feature_offsets, feature_values = self._make_simple_features(
            num_docs, num_features
        )

        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, 7
        )

        self.assertEqual(bloom_index.dtype, torch.long)
        self.assertEqual(bloom_index.dim(), 1)
        self.assertEqual(bundle_b_offsets.dtype, torch.long)
        self.assertEqual(bundle_b_offsets.dim(), 1)
        # First offset should be 0
        self.assertEqual(bundle_b_offsets[0].item(), 0)
        # Last offset should be > 0
        self.assertGreater(bundle_b_offsets[-1].item(), 0)

    def test_build_bundled_single_doc(self) -> None:
        """Single document should work."""
        feature_ids = torch.tensor([0], dtype=torch.int32)
        feature_offsets = torch.tensor([0, 1], dtype=torch.long)
        feature_values = torch.tensor([42], dtype=torch.long)

        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, 3
        )

        self.assertEqual(bloom_index.dim(), 1)
        self.assertGreater(bloom_index.numel(), 0)

    def test_build_bundled_many_docs(self) -> None:
        """Test with enough documents to span multiple column bundles."""
        num_docs = COL_BUNDLE_SIZE * 64 * 2 + 10  # >2 bundles
        feature_ids = torch.tensor([0], dtype=torch.int32)
        feature_offsets = torch.zeros(num_docs + 1, dtype=torch.long)
        # Each document has one feature value
        for i in range(num_docs):
            feature_offsets[i + 1] = i + 1
        feature_values = torch.arange(num_docs, dtype=torch.long)

        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, 3
        )

        # Should have multiple bundles
        self.assertGreater(bundle_b_offsets.numel(), 2)

    def test_build_bundled_different_k(self) -> None:
        """Different k values should produce valid indices."""
        feature_ids, feature_offsets, feature_values = self._make_simple_features(5, 2)
        for k in [1, 3, 7, 10]:
            bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
                feature_ids, feature_offsets, feature_values, 2.0, k
            )
            self.assertEqual(bloom_index.dim(), 1)
            self.assertGreater(bloom_index.numel(), 0)


if __name__ == "__main__":
    unittest.main()
