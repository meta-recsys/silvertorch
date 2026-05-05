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

"""Tests for bloom_index_search_batch and generate_column_info_for_clusters
-- CPU and GPU cross-validated."""

import unittest

import silvertorch.ops._load_ops  # noqa: F401
import torch

# fmt: off
# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:bloom_indexer")
# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:bloom_indexer_gpu")
# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:bloom_index_search")
# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:bloom_index_search_gpu")
# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:expression_query_parser")
# fmt: on

HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(
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


def _build_index(
    feature_ids: torch.Tensor,
    feature_offsets: torch.Tensor,
    feature_values: torch.Tensor,
    device: str = "cpu",
    hash_k: int = 7,
    b_multiplier: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.st.bloom_index_build(
        feature_ids.to(device),
        feature_offsets.to(device),
        feature_values.to(device),
        b_multiplier,
        hash_k,
    )


def _parse_queries(
    expressions: list[str], hash_k: int = 7
) -> tuple[torch.Tensor, torch.Tensor]:
    silvertorch_ks = torch.ones(1, dtype=torch.long)
    _, tensors = torch.ops.st.parse_expression_query_batch(
        expressions, silvertorch_ks, hash_k, True, 5
    )
    return tensors[0], tensors[1]


def _search(
    bloom_index: torch.Tensor,
    bundle_b_offsets: torch.Tensor,
    plans_data: torch.Tensor,
    plans_offsets: torch.Tensor,
    device: str = "cpu",
    k: int = 3,
    hash_k: int = 7,
    return_bool_mask: bool = True,
) -> torch.Tensor:
    return torch.ops.st.bloom_index_search_batch(
        bloom_index.to(device),
        bundle_b_offsets.to(device),
        plans_data.to(device),
        plans_offsets.to(device),
        k,
        hash_k,
        return_bool_mask,
    )


def _build_and_search(
    num_docs: int,
    num_features: int,
    expression: str,
    device: str = "cpu",
    k: int = 3,
    hash_k: int = 7,
    return_bool_mask: bool = True,
) -> torch.Tensor:
    """Full pipeline: build index -> parse expression -> search, all on one device."""
    feature_ids, feature_offsets, feature_values = _make_features(
        num_docs, num_features
    )
    bloom_index, bundle_b_offsets = _build_index(
        feature_ids, feature_offsets, feature_values, device, hash_k
    )
    plans_data, plans_offsets = _parse_queries([expression], hash_k)
    return _search(
        bloom_index,
        bundle_b_offsets,
        plans_data,
        plans_offsets,
        device,
        k,
        hash_k,
        return_bool_mask,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBloomIndexSearch(unittest.TestCase):
    """Each test runs on CPU and GPU, then asserts identical results."""

    def _run_on_both(
        self,
        num_docs: int,
        num_features: int,
        expressions: list[str],
        k: int = 3,
        hash_k: int = 7,
        return_bool_mask: bool = True,
    ) -> torch.Tensor:
        """Run the full build-parse-search pipeline on CPU (and GPU if available).
        Asserts that shapes, dtypes, and values match across devices.
        Returns the CPU result."""
        feature_ids, feature_offsets, feature_values = _make_features(
            num_docs, num_features
        )
        # Build index on CPU (used for both CPU search and as reference for GPU)
        bloom_cpu, offsets_cpu = _build_index(
            feature_ids, feature_offsets, feature_values, "cpu", hash_k
        )
        plans_data, plans_offsets = _parse_queries(expressions, hash_k)

        result_cpu = _search(
            bloom_cpu,
            offsets_cpu,
            plans_data,
            plans_offsets,
            "cpu",
            k,
            hash_k,
            return_bool_mask,
        )

        if return_bool_mask:
            self.assertEqual(result_cpu.dtype, torch.bool, "[cpu] dtype")
        else:
            self.assertEqual(result_cpu.dtype, torch.long, "[cpu] dtype")
        self.assertEqual(result_cpu.size(0), len(expressions), "[cpu] batch dim")

        if HAS_CUDA:
            result_gpu = _search(
                bloom_cpu,
                offsets_cpu,
                plans_data,
                plans_offsets,
                "cuda",
                k,
                hash_k,
                return_bool_mask,
            )
            self.assertEqual(result_gpu.size(0), len(expressions), "[cuda] batch dim")
            torch.testing.assert_close(
                result_cpu,
                result_gpu.cpu(),
                msg=lambda m: f"CPU/GPU mismatch: {m}",
            )

        return result_cpu

    # ---- single query tests ----

    def test_single_query_bool(self) -> None:
        result = self._run_on_both(10, 2, ["0:0"], return_bool_mask=True)
        self.assertEqual(result.dim(), 2)

    def test_single_query_bitmask(self) -> None:
        result = self._run_on_both(10, 2, ["0:0"], return_bool_mask=False)
        self.assertEqual(result.dtype, torch.long)

    def test_and_query(self) -> None:
        self._run_on_both(10, 3, ["0:0 AND 1:1"])

    def test_or_query(self) -> None:
        self._run_on_both(10, 3, ["0:0 OR 1:1"])

    def test_not_query(self) -> None:
        self._run_on_both(10, 3, ["NOT 0:0"])

    # ---- batch query tests ----

    def test_batch_queries(self) -> None:
        result = self._run_on_both(10, 3, ["0:0", "1:1", "0:0 AND 1:1"])
        self.assertEqual(result.size(0), 3)

    # ---- generate_column_info tests ----

    def test_generate_column_info(self) -> None:
        offsets = torch.tensor([0, 100, 200], dtype=torch.long)
        lengths = torch.tensor([100, 50, 80], dtype=torch.long)

        column_counts_cpu, start_ids_cpu, first_offsets_cpu = (
            torch.ops.st.generate_column_info_for_clusters(offsets, lengths)
        )
        self.assertEqual(column_counts_cpu.dtype, torch.int32)
        self.assertEqual(start_ids_cpu.dtype, torch.int32)
        self.assertEqual(column_counts_cpu.numel(), 3)

        if HAS_CUDA:
            column_counts_gpu, start_ids_gpu, first_offsets_gpu = (
                torch.ops.st.generate_column_info_for_clusters(
                    offsets.cuda(), lengths.cuda()
                )
            )
            self.assertTrue(column_counts_gpu.is_cuda)
            torch.testing.assert_close(column_counts_cpu, column_counts_gpu.cpu())
            torch.testing.assert_close(start_ids_cpu, start_ids_gpu.cpu())
            torch.testing.assert_close(first_offsets_cpu, first_offsets_gpu.cpu())

    # ---- GPU-specific build + search test ----

    def test_build_on_gpu_search_on_gpu(self) -> None:
        """Build the index directly on GPU, then search on GPU."""
        if not HAS_CUDA:
            self.skipTest("CUDA not available")

        feature_ids, feature_offsets, feature_values = _make_features(10, 3)
        bloom_gpu, offsets_gpu = _build_index(
            feature_ids, feature_offsets, feature_values, "cuda"
        )
        plans_data, plans_offsets = _parse_queries(["0:0"])
        result = _search(bloom_gpu, offsets_gpu, plans_data, plans_offsets, "cuda")
        self.assertTrue(result.is_cuda)
        self.assertEqual(result.size(0), 1)

        # Cross-check: build on CPU, search on CPU, compare
        bloom_cpu, offsets_cpu = _build_index(
            feature_ids, feature_offsets, feature_values, "cpu"
        )
        result_cpu = _search(bloom_cpu, offsets_cpu, plans_data, plans_offsets, "cpu")
        torch.testing.assert_close(result_cpu, result.cpu())
