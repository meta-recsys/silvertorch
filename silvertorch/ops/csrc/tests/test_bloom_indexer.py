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

"""Tests for bloom_index_build op -- CPU and GPU cross-validated."""

import unittest

import silvertorch.ops._load_ops  # noqa: F401
import torch

# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:bloom_indexer")
# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:bloom_indexer_gpu")

HAS_CUDA = torch.cuda.is_available()
COL_BUNDLE_SIZE = 32


def _make_simple_features(
    num_docs: int, num_features: int
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


class TestBloomIndexer(unittest.TestCase):
    """Each test builds on CPU and verifies, then on GPU and verifies,
    then asserts CPU == GPU."""

    def _build_on_device(
        self,
        device: str,
        feature_ids: torch.Tensor,
        feature_offsets: torch.Tensor,
        feature_values: torch.Tensor,
        b_multiplier: float,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.st.bloom_index_build(
            feature_ids.to(device),
            feature_offsets.to(device),
            feature_values.to(device),
            b_multiplier,
            k,
        )

    def _verify_basic_properties(
        self,
        bloom_index: torch.Tensor,
        bundle_b_offsets: torch.Tensor,
        device: str,
    ) -> None:
        self.assertEqual(bloom_index.dtype, torch.long)
        self.assertEqual(bloom_index.dim(), 1)
        self.assertGreater(bloom_index.numel(), 0)
        self.assertEqual(bundle_b_offsets.dtype, torch.long)
        self.assertEqual(bundle_b_offsets.dim(), 1)
        self.assertEqual(bundle_b_offsets[0].item(), 0, f"[{device}]")
        self.assertGreater(bundle_b_offsets[-1].item(), 0, f"[{device}]")

    def _run_and_compare(
        self,
        feature_ids: torch.Tensor,
        feature_offsets: torch.Tensor,
        feature_values: torch.Tensor,
        b_multiplier: float = 2.0,
        k: int = 7,
    ) -> None:
        bloom_cpu, offsets_cpu = self._build_on_device(
            "cpu", feature_ids, feature_offsets, feature_values, b_multiplier, k
        )
        self._verify_basic_properties(bloom_cpu, offsets_cpu, "cpu")

        if HAS_CUDA:
            bloom_gpu, offsets_gpu = self._build_on_device(
                "cuda",
                feature_ids,
                feature_offsets,
                feature_values,
                b_multiplier,
                k,
            )
            self._verify_basic_properties(bloom_gpu, offsets_gpu, "cuda")
            torch.testing.assert_close(
                bloom_cpu, bloom_gpu.cpu(), msg=lambda m: f"bloom_index mismatch: {m}"
            )
            torch.testing.assert_close(
                offsets_cpu,
                offsets_gpu.cpu(),
                msg=lambda m: f"bundle_b_offsets mismatch: {m}",
            )

    def test_build_basic(self) -> None:
        feature_ids, feature_offsets, feature_values = _make_simple_features(10, 3)
        self._run_and_compare(feature_ids, feature_offsets, feature_values)

    def test_build_single_doc(self) -> None:
        feature_ids = torch.tensor([0], dtype=torch.int32)
        feature_offsets = torch.tensor([0, 1], dtype=torch.long)
        feature_values = torch.tensor([42], dtype=torch.long)
        self._run_and_compare(feature_ids, feature_offsets, feature_values, k=3)

    def test_build_many_docs(self) -> None:
        """Enough documents to span multiple column bundles."""
        num_docs = COL_BUNDLE_SIZE * 64 * 2 + 10
        feature_ids = torch.tensor([0], dtype=torch.int32)
        feature_offsets = torch.zeros(num_docs + 1, dtype=torch.long)
        for i in range(num_docs):
            feature_offsets[i + 1] = i + 1
        feature_values = torch.arange(num_docs, dtype=torch.long)

        bloom_cpu, offsets_cpu = self._build_on_device(
            "cpu", feature_ids, feature_offsets, feature_values, 2.0, 3
        )
        self.assertGreater(offsets_cpu.numel(), 2, "[cpu] should have multiple bundles")

        if HAS_CUDA:
            bloom_gpu, offsets_gpu = self._build_on_device(
                "cuda", feature_ids, feature_offsets, feature_values, 2.0, 3
            )
            self.assertGreater(
                offsets_gpu.numel(), 2, "[cuda] should have multiple bundles"
            )
            torch.testing.assert_close(bloom_cpu, bloom_gpu.cpu())
            torch.testing.assert_close(offsets_cpu, offsets_gpu.cpu())

    def test_build_different_k(self) -> None:
        feature_ids, feature_offsets, feature_values = _make_simple_features(5, 2)
        for k in [1, 3, 7, 10]:
            self._run_and_compare(feature_ids, feature_offsets, feature_values, k=k)
