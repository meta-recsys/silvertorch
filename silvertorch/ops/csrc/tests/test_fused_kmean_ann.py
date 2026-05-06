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

"""Tests for fused_kmean_ann and knn_expand_cluster_offset_batch ops.

Each test computes expected results independently, then verifies both the
CPU and GPU implementations produce matching outputs.
"""

import unittest

import silvertorch.ops._load_ops  # noqa: F401
import torch

# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:fused_kmean_ann")
# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:fused_kmean_ann_gpu")

HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_valid(
    scores: torch.Tensor, indices: torch.Tensor, invalid_index: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract non-padding entries per row, sort by doc index, flatten."""
    all_scores = []
    all_indices = []
    for b in range(indices.size(0)):
        mask = indices[b] != invalid_index
        row_idx = indices[b][mask]
        row_sc = scores[b][mask]
        order = row_idx.argsort()
        all_indices.append(row_idx[order])
        all_scores.append(row_sc[order])
    return torch.cat(all_scores), torch.cat(all_indices)


def _compute_expected_scores(
    cluster_offsets: torch.Tensor,
    cluster_ids: torch.Tensor,
    cluster_length: torch.Tensor,
    embeddings: torch.Tensor,
    queries: torch.Tensor,
    divisor_for_int8: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation: compute dot-product scores on CPU.

    Returns flattened (scores, doc_indices) sorted by doc index, with no
    padding -- only valid entries.
    """
    is_int8 = embeddings.dtype == torch.int8
    batch_size = cluster_ids.size(0)
    n_probes = cluster_ids.size(1)

    all_scores = []
    all_indices = []
    for b in range(batch_size):
        row_scores = []
        row_indices = []
        for p in range(n_probes):
            cid = int(cluster_ids[b, p].item())
            clen = int(cluster_length[b, p].item())
            doc_start = int(cluster_offsets[cid].item())
            for i in range(clen):
                doc_idx = doc_start + i
                if is_int8:
                    e = embeddings[doc_idx].to(torch.int32)
                    q = queries[b].to(torch.int32)
                    raw = (e * q).sum().item()
                    if divisor_for_int8 != -1:
                        row_scores.append(float(raw) / float(divisor_for_int8))
                    else:
                        row_scores.append(raw)
                else:
                    e = embeddings[doc_idx].to(torch.float64)
                    q = queries[b].to(torch.float64)
                    row_scores.append((e * q).sum().item())
                row_indices.append(doc_idx)
        order = sorted(range(len(row_indices)), key=lambda k: row_indices[k])
        all_scores.extend([row_scores[k] for k in order])
        all_indices.extend([row_indices[k] for k in order])

    if is_int8:
        if divisor_for_int8 != -1:
            score_t = torch.tensor(all_scores, dtype=torch.float16)
        else:
            score_t = torch.tensor(all_scores, dtype=torch.int32)
    elif embeddings.dtype == torch.float16:
        score_t = torch.tensor(all_scores, dtype=torch.float16)
    elif embeddings.dtype == torch.bfloat16:
        score_t = torch.tensor(all_scores, dtype=torch.bfloat16)
    else:
        score_t = torch.tensor(all_scores, dtype=torch.float32)

    idx_t = torch.tensor(all_indices, dtype=torch.int32)
    return score_t, idx_t


def _run_on_device(
    device: str,
    cluster_offsets: torch.Tensor,
    cluster_ids: torch.Tensor,
    cluster_length: torch.Tensor,
    embeddings: torch.Tensor,
    queries: torch.Tensor,
    max_size: int,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.st.fused_kmean_ann(
        cluster_offsets.to(device),
        cluster_ids.to(device),
        cluster_length.to(device),
        embeddings.to(device),
        queries.to(device),
        max_size,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestOpsRegistered(unittest.TestCase):
    def test_fused_kmean_ann_registered(self) -> None:
        self.assertTrue(hasattr(torch.ops.st, "fused_kmean_ann"))


# ---------------------------------------------------------------------------
# fused_kmean_ann tests
# ---------------------------------------------------------------------------


class TestFusedKmeanAnn(unittest.TestCase):
    """Each test verifies CPU and GPU against independently computed expected
    results, ensuring both implementations are correct and agree."""

    def _verify_on_device(
        self,
        device: str,
        cluster_offsets: torch.Tensor,
        cluster_ids: torch.Tensor,
        cluster_length: torch.Tensor,
        embeddings: torch.Tensor,
        queries: torch.Tensor,
        max_size: int,
        expected_scores: torch.Tensor,
        expected_indices: torch.Tensor,
        atol: float,
        rtol: float,
        **kwargs,
    ) -> None:
        scores, indices = _run_on_device(
            device,
            cluster_offsets,
            cluster_ids,
            cluster_length,
            embeddings,
            queries,
            max_size,
            **kwargs,
        )
        scores = scores.cpu()
        indices = indices.cpu()

        invalid = kwargs.get("invalid_index_value", -1)
        sc, idx = _extract_valid(scores, indices, invalid)
        self.assertEqual(
            idx.numel(),
            expected_indices.numel(),
            f"[{device}] valid count mismatch: got {idx.numel()}, "
            f"expected {expected_indices.numel()}",
        )
        torch.testing.assert_close(
            idx, expected_indices, msg=lambda m: f"[{device}] indices: {m}"
        )
        torch.testing.assert_close(
            sc,
            expected_scores,
            atol=atol,
            rtol=rtol,
            msg=lambda m: f"[{device}] scores: {m}",
        )

    def _run_cpu_and_gpu(
        self,
        cluster_offsets: torch.Tensor,
        cluster_ids: torch.Tensor,
        cluster_length: torch.Tensor,
        embeddings: torch.Tensor,
        queries: torch.Tensor,
        max_size: int,
        expected_scores: torch.Tensor,
        expected_indices: torch.Tensor,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        **kwargs,
    ) -> None:
        self._verify_on_device(
            "cpu",
            cluster_offsets,
            cluster_ids,
            cluster_length,
            embeddings,
            queries,
            max_size,
            expected_scores,
            expected_indices,
            atol,
            rtol,
            **kwargs,
        )
        if HAS_CUDA:
            self._verify_on_device(
                "cuda",
                cluster_offsets,
                cluster_ids,
                cluster_length,
                embeddings,
                queries,
                max_size,
                expected_scores,
                expected_indices,
                atol,
                rtol,
                **kwargs,
            )

    def _run_random(
        self,
        n_clusters: int,
        cluster_size: int,
        dim: int,
        batch_size: int,
        n_probes: int,
        embedding_dtype: torch.dtype = torch.float32,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        **kwargs,
    ) -> None:
        """Generate random data, compute expected, then verify CPU + GPU."""
        total_docs = n_clusters * cluster_size
        cluster_offsets = torch.arange(
            0, total_docs + 1, cluster_size, dtype=torch.long
        )
        if embedding_dtype == torch.int8:
            embeddings = torch.randint(-128, 127, (total_docs, dim), dtype=torch.int8)
            queries = torch.randint(-128, 127, (batch_size, dim), dtype=torch.int8)
        else:
            embeddings = torch.randn(total_docs, dim, dtype=embedding_dtype)
            queries = torch.randn(batch_size, dim, dtype=embedding_dtype)
        cluster_ids = torch.randint(0, n_clusters, (batch_size, n_probes)).long()
        cluster_length = torch.full(
            (batch_size, n_probes), cluster_size, dtype=torch.long
        )
        max_size = n_probes * cluster_size

        divisor = kwargs.get("divisor_for_int8", -1)
        expected_scores, expected_indices = _compute_expected_scores(
            cluster_offsets, cluster_ids, cluster_length, embeddings, queries, divisor
        )

        self._run_cpu_and_gpu(
            cluster_offsets,
            cluster_ids,
            cluster_length,
            embeddings,
            queries,
            max_size,
            expected_scores,
            expected_indices,
            atol,
            rtol,
            **kwargs,
        )

    # ---- deterministic correctness tests ----

    def test_correctness_float32_manual(self) -> None:
        """Hand-crafted data with known dot products."""
        cluster_offsets = torch.tensor([0, 3, 6], dtype=torch.long)
        embeddings = torch.randn(6, 16, dtype=torch.float32)
        queries = torch.randn(1, 16, dtype=torch.float32)
        cluster_ids = torch.tensor([[0, 1]], dtype=torch.long)
        cluster_length = torch.tensor([[3, 3]], dtype=torch.long)

        expected_scores, expected_indices = _compute_expected_scores(
            cluster_offsets, cluster_ids, cluster_length, embeddings, queries
        )

        self._run_cpu_and_gpu(
            cluster_offsets,
            cluster_ids,
            cluster_length,
            embeddings,
            queries,
            6,
            expected_scores,
            expected_indices,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_correctness_int8_with_divisor(self) -> None:
        # Use dim=16 (minimum GPU-supported dim). Pad values with zeros.
        cluster_offsets = torch.tensor([0, 2], dtype=torch.long)
        emb = torch.zeros(2, 16, dtype=torch.int8)
        emb[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        emb[1, :4] = torch.tensor([-1, -2, -3, -4], dtype=torch.int8)
        qry = torch.zeros(1, 16, dtype=torch.int8)
        qry[0, :4] = torch.tensor([10, 20, 30, 40], dtype=torch.int8)
        cluster_ids = torch.tensor([[0]], dtype=torch.long)
        cluster_length = torch.tensor([[2]], dtype=torch.long)

        # doc0: 1*10+2*20+3*30+4*40+0*..=300 -> 300/100=3.0
        # doc1: -300/100=-3.0
        expected_scores = torch.tensor([3.0, -3.0], dtype=torch.float16)
        expected_indices = torch.tensor([0, 1], dtype=torch.int32)

        self._run_cpu_and_gpu(
            cluster_offsets,
            cluster_ids,
            cluster_length,
            emb,
            qry,
            2,
            expected_scores,
            expected_indices,
            atol=0.01,
            rtol=0.01,
            divisor_for_int8=100,
        )

    def test_correctness_int8_no_divisor(self) -> None:
        cluster_offsets = torch.tensor([0, 2], dtype=torch.long)
        emb = torch.zeros(2, 16, dtype=torch.int8)
        emb[0, :2] = torch.tensor([1, 2], dtype=torch.int8)
        emb[1, :2] = torch.tensor([3, 4], dtype=torch.int8)
        qry = torch.zeros(1, 16, dtype=torch.int8)
        qry[0, :2] = torch.tensor([5, 6], dtype=torch.int8)
        cluster_ids = torch.tensor([[0]], dtype=torch.long)
        cluster_length = torch.tensor([[2]], dtype=torch.long)

        # doc0: 1*5+2*6=17, doc1: 3*5+4*6=39
        expected_scores = torch.tensor([17, 39], dtype=torch.int32)
        expected_indices = torch.tensor([0, 1], dtype=torch.int32)

        self._run_cpu_and_gpu(
            cluster_offsets,
            cluster_ids,
            cluster_length,
            emb,
            qry,
            2,
            expected_scores,
            expected_indices,
            atol=0,
            rtol=0,
            divisor_for_int8=-1,
        )

    def test_indices_are_global_doc_ids(self) -> None:
        cluster_offsets = torch.tensor([0, 3, 7], dtype=torch.long)
        embeddings = torch.randn(7, 16, dtype=torch.float32)
        queries = torch.randn(1, 16, dtype=torch.float32)
        cluster_ids = torch.tensor([[1]], dtype=torch.long)
        cluster_length = torch.tensor([[4]], dtype=torch.long)

        expected_scores, expected_indices = _compute_expected_scores(
            cluster_offsets, cluster_ids, cluster_length, embeddings, queries
        )
        torch.testing.assert_close(
            expected_indices, torch.tensor([3, 4, 5, 6], dtype=torch.int32)
        )

        self._run_cpu_and_gpu(
            cluster_offsets,
            cluster_ids,
            cluster_length,
            embeddings,
            queries,
            4,
            expected_scores,
            expected_indices,
        )

    def test_multi_probe_ordering(self) -> None:
        cluster_offsets = torch.tensor([0, 2, 5], dtype=torch.long)
        embeddings = torch.randn(5, 16, dtype=torch.float32)
        queries = torch.randn(1, 16, dtype=torch.float32)
        cluster_ids = torch.tensor([[0, 1]], dtype=torch.long)
        cluster_length = torch.tensor([[2, 3]], dtype=torch.long)

        expected_scores, expected_indices = _compute_expected_scores(
            cluster_offsets, cluster_ids, cluster_length, embeddings, queries
        )

        self._run_cpu_and_gpu(
            cluster_offsets,
            cluster_ids,
            cluster_length,
            embeddings,
            queries,
            5,
            expected_scores,
            expected_indices,
        )

    def test_padding_values(self) -> None:
        """Padding positions: scores=-FLT_MAX, indices=invalid_index_value."""
        cluster_offsets = torch.tensor([0, 2], dtype=torch.long)
        embeddings = torch.randn(2, 16, dtype=torch.float32)
        queries = torch.randn(1, 16, dtype=torch.float32)
        cluster_ids = torch.tensor([[0]], dtype=torch.long)
        cluster_length = torch.tensor([[2]], dtype=torch.long)

        for device in ["cpu", "cuda"] if HAS_CUDA else ["cpu"]:
            scores, indices = _run_on_device(
                device,
                cluster_offsets,
                cluster_ids,
                cluster_length,
                embeddings,
                queries,
                32,
                invalid_index_value=-99,
            )
            scores = scores.cpu()
            indices = indices.cpu()
            self.assertGreater(scores[0, 0].item(), -1e30)
            self.assertGreater(scores[0, 1].item(), -1e30)
            for c in range(2, scores.size(1)):
                self.assertEqual(indices[0, c].item(), -99, f"[{device}] col {c}")

    # ---- randomized float32 tests ----

    def test_float32_single_cluster(self) -> None:
        self._run_random(
            n_clusters=1, cluster_size=32, dim=64, batch_size=1, n_probes=1
        )

    def test_float32_multi_probe(self) -> None:
        self._run_random(
            n_clusters=5, cluster_size=32, dim=64, batch_size=2, n_probes=3
        )

    def test_float32_small_cluster(self) -> None:
        self._run_random(
            n_clusters=3, cluster_size=10, dim=32, batch_size=2, n_probes=2
        )

    def test_float32_large_cluster(self) -> None:
        self._run_random(
            n_clusters=4, cluster_size=100, dim=64, batch_size=2, n_probes=2
        )

    def test_float32_dim_128(self) -> None:
        self._run_random(
            n_clusters=3, cluster_size=32, dim=128, batch_size=2, n_probes=2
        )

    def test_float32_dim_256(self) -> None:
        self._run_random(
            n_clusters=3, cluster_size=32, dim=256, batch_size=1, n_probes=2
        )

    def test_float32_cluster_size_1(self) -> None:
        self._run_random(n_clusters=4, cluster_size=1, dim=32, batch_size=2, n_probes=3)

    def test_float32_cluster_size_31(self) -> None:
        self._run_random(
            n_clusters=3, cluster_size=31, dim=32, batch_size=1, n_probes=2
        )

    def test_float32_cluster_size_33(self) -> None:
        self._run_random(
            n_clusters=3, cluster_size=33, dim=32, batch_size=1, n_probes=2
        )

    def test_float32_batch_4(self) -> None:
        self._run_random(
            n_clusters=5, cluster_size=50, dim=64, batch_size=4, n_probes=3
        )

    # ---- float16 tests ----

    def test_half_basic(self) -> None:
        self._run_random(
            n_clusters=3,
            cluster_size=32,
            dim=64,
            batch_size=2,
            n_probes=2,
            embedding_dtype=torch.float16,
            atol=0.05,
            rtol=0.05,
        )

    def test_half_large_cluster(self) -> None:
        self._run_random(
            n_clusters=4,
            cluster_size=100,
            dim=64,
            batch_size=2,
            n_probes=2,
            embedding_dtype=torch.float16,
            atol=0.1,
            rtol=0.1,
        )

    # ---- int8 tests ----

    def test_int8_with_divisor_random(self) -> None:
        self._run_random(
            n_clusters=3,
            cluster_size=32,
            dim=64,
            batch_size=2,
            n_probes=2,
            embedding_dtype=torch.int8,
            divisor_for_int8=127,
            atol=0.02,
            rtol=0.02,
        )

    def test_int8_no_divisor_random(self) -> None:
        self._run_random(
            n_clusters=3,
            cluster_size=32,
            dim=64,
            batch_size=1,
            n_probes=2,
            embedding_dtype=torch.int8,
            divisor_for_int8=-1,
            atol=0,
            rtol=0,
        )

    def test_int8_small_cluster(self) -> None:
        self._run_random(
            n_clusters=5,
            cluster_size=10,
            dim=32,
            batch_size=2,
            n_probes=3,
            embedding_dtype=torch.int8,
            divisor_for_int8=64,
            atol=0.02,
            rtol=0.02,
        )

    # ---- dtype output tests ----

    def test_dtype_float32(self) -> None:
        scores, _ = _run_on_device("cpu", *self._quick_data(torch.float32), 8)
        self.assertEqual(scores.dtype, torch.float32)

    def test_dtype_float16(self) -> None:
        scores, _ = _run_on_device("cpu", *self._quick_data(torch.float16), 8)
        self.assertEqual(scores.dtype, torch.float16)

    def test_dtype_int8_no_divisor(self) -> None:
        scores, _ = _run_on_device(
            "cpu", *self._quick_data(torch.int8), 8, divisor_for_int8=-1
        )
        self.assertEqual(scores.dtype, torch.int32)

    def test_dtype_int8_with_divisor(self) -> None:
        scores, _ = _run_on_device(
            "cpu", *self._quick_data(torch.int8), 8, divisor_for_int8=127
        )
        self.assertEqual(scores.dtype, torch.float16)

    def _quick_data(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cluster_offsets = torch.tensor([0, 4, 8], dtype=torch.long)
        cluster_ids = torch.tensor([[0, 1]], dtype=torch.long)
        cluster_length = torch.tensor([[4, 4]], dtype=torch.long)
        if dtype == torch.int8:
            embeddings = torch.randint(-128, 127, (8, 16), dtype=torch.int8)
            queries = torch.randint(-128, 127, (1, 16), dtype=torch.int8)
        else:
            embeddings = torch.randn(8, 16, dtype=dtype)
            queries = torch.randn(1, 16, dtype=dtype)
        return cluster_offsets, cluster_ids, cluster_length, embeddings, queries
