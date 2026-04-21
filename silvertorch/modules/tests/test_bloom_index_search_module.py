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
from silvertorch.modules.bloom_index_search_module import BloomIndexSearchModule
from silvertorch.modules.bloom_index_search_module_builder import (
    BloomIndexSearchModuleBuilder,
)


class TestBloomIndexSearchModule(unittest.TestCase):
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
        return (
            feature_ids,
            torch.tensor(offsets, dtype=torch.long),
            torch.tensor(values, dtype=torch.long),
        )

    def _parse_expressions(
        self, expressions: list[str], hash_k: int = 7
    ) -> tuple[torch.Tensor, torch.Tensor]:
        silvertorch_ks = torch.ones(1, dtype=torch.long)
        _, tensors = torch.ops.st.parse_expression_query_batch(
            expressions, silvertorch_ks, hash_k, True, 5
        )
        return tensors[0], tensors[1]

    def test_module_forward(self) -> None:
        k, hash_k = 3, 7
        feature_ids, feature_offsets, feature_values = self._make_features(10, 3)
        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, hash_k
        )

        module = BloomIndexSearchModule(k=k, hash_k=hash_k)
        module.set_bloom_index(bloom_index)
        module.set_bloom_bundle_b_offsets(bundle_b_offsets)
        module.eval()

        plans_data, plans_offsets = self._parse_expressions(["0:0"])
        result = module(plans_data, plans_offsets)
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result.size(0), 1)

    def test_module_bitmask(self) -> None:
        k, hash_k = 3, 7
        feature_ids, feature_offsets, feature_values = self._make_features(10, 2)
        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, hash_k
        )

        module = BloomIndexSearchModule(k=k, hash_k=hash_k)
        module.set_bloom_index(bloom_index)
        module.set_bloom_bundle_b_offsets(bundle_b_offsets)
        module.eval()

        plans_data, plans_offsets = self._parse_expressions(["0:0"])
        result = module(plans_data, plans_offsets, return_bool_mask=False)
        self.assertEqual(result.dtype, torch.long)

    def test_module_batch(self) -> None:
        k, hash_k = 3, 7
        feature_ids, feature_offsets, feature_values = self._make_features(10, 3)
        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, hash_k
        )

        module = BloomIndexSearchModule(k=k, hash_k=hash_k)
        module.set_bloom_index(bloom_index)
        module.set_bloom_bundle_b_offsets(bundle_b_offsets)
        module.eval()

        plans_data, plans_offsets = self._parse_expressions(
            ["0:0", "1:1", "0:0 AND 1:1"]
        )
        result = module(plans_data, plans_offsets)
        self.assertEqual(result.size(0), 3)


class TestBloomIndexSearchModuleBuilder(unittest.TestCase):
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
        return (
            feature_ids,
            torch.tensor(offsets, dtype=torch.long),
            torch.tensor(values, dtype=torch.long),
        )

    def test_build_from_features(self) -> None:
        feature_ids, feature_offsets, feature_values = self._make_features(10, 3)
        module = (
            BloomIndexSearchModuleBuilder(k=3, hash_k=7)
            .set_feature_data(feature_ids, feature_offsets, feature_values)
            .set_b_multiplier(2.0)
            .build()
        )
        self.assertIsInstance(module, BloomIndexSearchModule)
        self.assertGreater(module.bloom_index.numel(), 0)

    def test_build_from_prebuilt(self) -> None:
        feature_ids, feature_offsets, feature_values = self._make_features(10, 3)
        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids, feature_offsets, feature_values, 2.0, 7
        )
        module = (
            BloomIndexSearchModuleBuilder(k=3, hash_k=7)
            .set_bloom_index(bloom_index)
            .set_bloom_bundle_b_offsets(bundle_b_offsets)
            .build()
        )
        self.assertTrue(torch.equal(module.bloom_index, bloom_index))

    def test_build_missing_data_raises(self) -> None:
        with self.assertRaises(ValueError):
            BloomIndexSearchModuleBuilder(k=3, hash_k=7).build()

    def test_build_missing_offsets_raises(self) -> None:
        with self.assertRaises(ValueError):
            (
                BloomIndexSearchModuleBuilder(k=3, hash_k=7)
                .set_bloom_index(torch.zeros(100, dtype=torch.long))
                .build()
            )


if __name__ == "__main__":
    unittest.main()
