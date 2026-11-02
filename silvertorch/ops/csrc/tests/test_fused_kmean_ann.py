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

import os
import subprocess
import sys
import tempfile
import unittest

import silvertorch.ops._load_ops  # noqa: F401
import torch

# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:fused_kmean_ann")
# @oss-disable[end= ]: torch.ops.load_library("//silvertorch/oss/ops/csrc:fused_kmean_ann_gpu")

HAS_CUDA = torch.cuda.is_available()


_CUDA_PARITY_SUBPROCESS = r"""
import sys

import silvertorch.ops._load_ops  # noqa: F401
import torch

torch.manual_seed(20261101)
cluster_lengths_1d = torch.tensor(
    [1, 7, 28, 31, 32, 33, 45, 64, 70, 100], dtype=torch.long
)
cluster_offsets = torch.cat(
    [torch.zeros(1, dtype=torch.long), cluster_lengths_1d.cumsum(0)]
)
cluster_ids = torch.stack(
    [
        torch.arange(cluster_lengths_1d.numel()),
        torch.arange(cluster_lengths_1d.numel() - 1, -1, -1),
    ]
)
cluster_lengths = cluster_lengths_1d[cluster_ids]
embeddings = torch.randint(
    -128, 128, (int(cluster_offsets[-1]), 256), dtype=torch.int8
)
queries = torch.randint(-128, 128, (2, 256), dtype=torch.int8)
max_size = int(cluster_lengths.sum(dim=1).max()) + 17

scores, indices = torch.ops.st.fused_kmean_ann(
    cluster_offsets.cuda(),
    cluster_ids.cuda(),
    cluster_lengths.cuda(),
    embeddings.cuda(),
    queries.cuda(),
    max_size,
    invalid_index_value=-99,
    divisor_for_int8=-1,
)
torch.cuda.synchronize()
torch.save((scores.cpu(), indices.cpu()), sys.argv[1])
"""


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

    def test_cuda_graph_op_preserves_original_schema(self) -> None:
        self.assertTrue(hasattr(torch.ops.st, "fused_kmean_ann_cuda_graph"))
        original_schema = str(torch.ops.st.fused_kmean_ann.default._schema)
        self.assertNotIn("use_cuda_graph_path", original_schema)
        self.assertIn("-> (Tensor, Tensor)", original_schema)

        graph_schema = str(torch.ops.st.fused_kmean_ann_cuda_graph.default._schema)
        self.assertIn("-> (Tensor, Tensor, Tensor)", graph_schema)


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

    def _run_mixed_cluster_int8(self, dim: int, divisor_for_int8: int) -> None:
        cluster_lengths_1d = torch.tensor(
            [1, 7, 28, 31, 32, 33, 45, 64, 70, 100], dtype=torch.long
        )
        cluster_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long), cluster_lengths_1d.cumsum(0)]
        )
        cluster_ids = torch.stack(
            [
                torch.arange(cluster_lengths_1d.numel()),
                torch.arange(cluster_lengths_1d.numel() - 1, -1, -1),
            ]
        )
        cluster_lengths = cluster_lengths_1d[cluster_ids]
        embeddings = torch.randint(
            -128,
            128,
            (int(cluster_offsets[-1]), dim),
            dtype=torch.int8,
        )
        queries = torch.randint(-128, 128, (2, dim), dtype=torch.int8)
        max_size = int(cluster_lengths.sum(dim=1).max()) + 17
        expected_scores, expected_indices = _compute_expected_scores(
            cluster_offsets,
            cluster_ids,
            cluster_lengths,
            embeddings,
            queries,
            divisor_for_int8,
        )

        exact = divisor_for_int8 == -1
        self._run_cpu_and_gpu(
            cluster_offsets,
            cluster_ids,
            cluster_lengths,
            embeddings,
            queries,
            max_size,
            expected_scores,
            expected_indices,
            atol=0 if exact else 0.02,
            rtol=0 if exact else 0.02,
            invalid_index_value=-99,
            divisor_for_int8=divisor_for_int8,
        )

    def test_int8_mixed_cluster_sizes_dim_128_to_512(self) -> None:
        for dim in (128, 256, 384, 512):
            for divisor_for_int8 in (-1, 127):
                with self.subTest(dim=dim, divisor=divisor_for_int8):
                    self._run_mixed_cluster_int8(dim, divisor_for_int8)

    def test_int8_clusters_all_warp_aligned(self) -> None:
        cluster_lengths_1d = torch.tensor([32, 64, 96], dtype=torch.long)
        cluster_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long), cluster_lengths_1d.cumsum(0)]
        )
        cluster_ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)
        cluster_lengths = cluster_lengths_1d[cluster_ids]
        embeddings = torch.randint(
            -128, 128, (int(cluster_offsets[-1]), 256), dtype=torch.int8
        )
        queries = torch.randint(-128, 128, (2, 256), dtype=torch.int8)
        expected_scores, expected_indices = _compute_expected_scores(
            cluster_offsets,
            cluster_ids,
            cluster_lengths,
            embeddings,
            queries,
            divisor_for_int8=-1,
        )

        self._run_cpu_and_gpu(
            cluster_offsets,
            cluster_ids,
            cluster_lengths,
            embeddings,
            queries,
            int(cluster_lengths.sum(dim=1).max()),
            expected_scores,
            expected_indices,
            atol=0,
            rtol=0,
            divisor_for_int8=-1,
        )

    def test_cooperative_int8_scan_matches_opt_out_raw_tensors(self) -> None:
        if not HAS_CUDA:
            self.skipTest("CUDA not available")

        variants = {
            "old": {
                "SILVERTORCH_DISABLE_COOPERATIVE_INT8_SCAN": "1",
            },
            "cooperative": {
                "SILVERTORCH_DISABLE_COOPERATIVE_INT8_SCAN": "0",
            },
        }
        outputs = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, overrides in variants.items():
                output_path = os.path.join(tmpdir, f"{name}.pt")
                env = os.environ.copy()
                env.pop("SILVERTORCH_DISABLE_COOPERATIVE_INT8_SCAN", None)
                env.update(overrides)
                subprocess.run(
                    [sys.executable, "-c", _CUDA_PARITY_SUBPROCESS, output_path],
                    env=env,
                    check=True,
                )
                outputs[name] = torch.load(
                    output_path, map_location="cpu", weights_only=True
                )

        old_scores, old_indices = outputs["old"]
        scores, indices = outputs["cooperative"]
        self.assertTrue(
            torch.equal(scores, old_scores), "cooperative scores differ from old"
        )
        self.assertTrue(
            torch.equal(indices, old_indices), "cooperative indices differ from old"
        )

    def test_int8_dim_256_per_embedding_scale_falls_back(self) -> None:
        if not HAS_CUDA:
            self.skipTest("CUDA not available")

        dim = 256
        cluster_offsets = torch.tensor([0, 33], dtype=torch.long)
        cluster_ids = torch.tensor([[0]], dtype=torch.long)
        cluster_lengths = torch.tensor([[33]], dtype=torch.long)
        # Keep the raw dot products representable as half: the established
        # per-embedding-scale path converts to half before dividing.
        embeddings = torch.randint(-4, 5, (33, dim), dtype=torch.int8)
        queries = torch.randint(-4, 5, (1, dim), dtype=torch.int8)
        per_embedding_scale = torch.full((33,), 127, dtype=torch.float16)
        expected_scores, expected_indices = _compute_expected_scores(
            cluster_offsets,
            cluster_ids,
            cluster_lengths,
            embeddings,
            queries,
            divisor_for_int8=127,
        )

        scores, indices = _run_on_device(
            "cuda",
            cluster_offsets,
            cluster_ids,
            cluster_lengths,
            embeddings,
            queries,
            33,
            per_embedding_scale=per_embedding_scale.cuda(),
        )
        scores, indices = _extract_valid(scores.cpu(), indices.cpu())
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(
            scores, expected_scores, atol=0.02, rtol=0.02
        )

    def _assert_cuda_graph_scan_parity(
        self,
        dtype: torch.dtype,
        dim: int,
        *,
        use_filter: bool = False,
        divisor_for_int8: int = -1,
        use_per_embedding_scale: bool = False,
    ) -> None:
        cluster_lengths_1d = torch.tensor(
            [0, 1, 31, 32, 33, 63, 64, 65], dtype=torch.long, device="cuda"
        )
        cluster_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device="cuda"),
                cluster_lengths_1d.cumsum(0),
            ]
        )
        cluster_ids = torch.stack(
            [
                torch.arange(8, device="cuda"),
                torch.arange(7, -1, -1, device="cuda"),
            ]
        )
        cluster_lengths = cluster_lengths_1d[cluster_ids]
        total_docs = int(cluster_offsets[-1])
        if dtype == torch.int8:
            low, high = (-4, 5) if use_per_embedding_scale else (-128, 128)
            embeddings = torch.randint(
                low, high, (total_docs, dim), dtype=dtype, device="cuda"
            )
            queries = torch.randint(
                low, high, (2, dim), dtype=dtype, device="cuda"
            )
        else:
            embeddings = torch.randn(total_docs, dim, dtype=dtype, device="cuda")
            queries = torch.randn(2, dim, dtype=dtype, device="cuda")
        capacity = int(cluster_lengths.sum(dim=1).max()) + 17
        kwargs = {"invalid_index_value": -99}
        if divisor_for_int8 != -1:
            kwargs["divisor_for_int8"] = divisor_for_int8
        if use_per_embedding_scale:
            kwargs["per_embedding_scale"] = torch.full(
                (total_docs,), 127, dtype=torch.float16, device="cuda"
            )
        if use_filter:
            words = (total_docs + 63) // 64
            kwargs["filtering_bit_mask"] = torch.full(
                (2, words),
                -6148914691236517206,  # 0xAAAAAAAAAAAAAAAA
                dtype=torch.long,
                device="cuda",
            )

        expected_scores, expected_indices = torch.ops.st.fused_kmean_ann(
            cluster_offsets,
            cluster_ids,
            cluster_lengths,
            embeddings,
            queries,
            capacity,
            **kwargs,
        )
        scores, indices, overflow = torch.ops.st.fused_kmean_ann_cuda_graph(
            cluster_offsets,
            cluster_ids,
            cluster_lengths,
            embeddings,
            queries,
            capacity,
            **kwargs,
        )
        torch.cuda.synchronize()
        self.assertEqual(overflow.item(), 0)
        torch.testing.assert_close(scores, expected_scores, atol=0, rtol=0)
        torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)

        def graph_call():
            return torch.ops.st.fused_kmean_ann_cuda_graph(
                cluster_offsets,
                cluster_ids,
                cluster_lengths,
                embeddings,
                queries,
                capacity,
                **kwargs,
            )

        graph, (graph_scores, graph_indices, graph_overflow) = (
            self._capture_scan(graph_call)
        )
        for _ in range(3):
            if dtype == torch.int8:
                queries.copy_(torch.randint_like(queries, low, high))
            else:
                queries.normal_()
            expected_scores, expected_indices = torch.ops.st.fused_kmean_ann(
                cluster_offsets,
                cluster_ids,
                cluster_lengths,
                embeddings,
                queries,
                capacity,
                **kwargs,
            )
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(graph_overflow.item(), 0)
            torch.testing.assert_close(
                graph_scores, expected_scores, atol=0, rtol=0
            )
            torch.testing.assert_close(
                graph_indices, expected_indices, atol=0, rtol=0
            )

    def _capture_scan(self, call):
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
        return graph, output

    def test_cuda_graph_scan_matches_eager(self) -> None:
        if not HAS_CUDA or torch.version.hip:
            self.skipTest("NVIDIA CUDA not available")

        cases = (
            (torch.int8, 64, False, -1, False),
            (torch.int8, 128, False, -1, False),
            (torch.int8, 256, False, 127, False),
            (torch.int8, 256, True, -1, False),
            (torch.int8, 256, False, -1, True),
            (torch.float32, 32, False, -1, False),
            (torch.float32, 128, True, -1, False),
        )
        for dtype, dim, use_filter, divisor, per_scale in cases:
            with self.subTest(
                dtype=dtype,
                dim=dim,
                filter=use_filter,
                divisor=divisor,
                per_scale=per_scale,
            ):
                self._assert_cuda_graph_scan_parity(
                    dtype,
                    dim,
                    use_filter=use_filter,
                    divisor_for_int8=divisor,
                    use_per_embedding_scale=per_scale,
                )

    def test_cuda_graph_scan_empty_selection(self) -> None:
        if not HAS_CUDA or torch.version.hip:
            self.skipTest("NVIDIA CUDA not available")

        offsets = torch.tensor([0, 1], dtype=torch.long, device="cuda")
        cluster_ids = torch.empty((2, 0), dtype=torch.long, device="cuda")
        lengths = torch.empty((2, 0), dtype=torch.long, device="cuda")
        embeddings = torch.randn(1, 32, device="cuda")
        queries = torch.randn(2, 32, device="cuda")
        scores, indices, overflow = torch.ops.st.fused_kmean_ann_cuda_graph(
            offsets,
            cluster_ids,
            lengths,
            embeddings,
            queries,
            17,
            invalid_index_value=-99,
        )
        torch.cuda.synchronize()
        self.assertEqual(overflow.item(), 0)
        self.assertTrue(torch.all(indices == -99))
        self.assertEqual(scores.shape, (2, 32))

        def call():
            return torch.ops.st.fused_kmean_ann_cuda_graph(
                offsets, cluster_ids, lengths, embeddings, queries, 17,
                invalid_index_value=-99,
            )

        graph, (graph_scores, graph_indices, graph_overflow) = self._capture_scan(call)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(graph_overflow.item(), 0)
        torch.testing.assert_close(graph_scores, scores, atol=0, rtol=0)
        torch.testing.assert_close(graph_indices, indices, atol=0, rtol=0)

    def test_cuda_graph_scan_replays_different_rows(self) -> None:
        if not HAS_CUDA or torch.version.hip:
            self.skipTest("NVIDIA CUDA not available")

        lengths_1d = torch.tensor(
            [0, 1, 31, 32, 33, 63, 64, 65], dtype=torch.long, device="cuda"
        )
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device="cuda"), lengths_1d.cumsum(0)]
        )
        embeddings = torch.randint(
            -128,
            128,
            (int(lengths_1d.sum()), 256),
            dtype=torch.int8,
            device="cuda",
        )
        cluster_ids = torch.stack(
            [torch.arange(8, device="cuda"), torch.arange(7, -1, -1, device="cuda")]
        )
        cluster_lengths = lengths_1d[cluster_ids]
        queries = torch.randint(-128, 128, (2, 256), dtype=torch.int8, device="cuda")
        capacity = 8 * int(lengths_1d.max())

        def graph_call() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return torch.ops.st.fused_kmean_ann_cuda_graph(
                offsets,
                cluster_ids,
                cluster_lengths,
                embeddings,
                queries,
                capacity,
            )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                graph_call()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_scores, graph_indices, graph_overflow = graph_call()

        selections = (
            ([7, 6, 5, 4, 3, 2, 1, 0], [0, 2, 4, 6, 1, 3, 5, 7]),
            ([1, 3, 5, 7, 0, 2, 4, 6], [7, 7, 6, 6, 5, 4, 3, 2]),
            ([0, 0, 1, 1, 2, 2, 3, 3], [7, 5, 3, 1, 6, 4, 2, 0]),
        )
        for row0, row1 in selections:
            replacement = torch.tensor([row0, row1], device="cuda")
            cluster_ids.copy_(replacement)
            cluster_lengths.copy_(lengths_1d[replacement])
            queries.copy_(torch.randint_like(queries, -128, 128))
            expected_scores, expected_indices = torch.ops.st.fused_kmean_ann(
                offsets,
                cluster_ids,
                cluster_lengths,
                embeddings,
                queries,
                capacity,
            )
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(graph_overflow.item(), 0)
            torch.testing.assert_close(
                graph_scores, expected_scores, atol=0, rtol=0
            )
            torch.testing.assert_close(
                graph_indices, expected_indices, atol=0, rtol=0
            )

    def test_cuda_graph_scan_capacity_boundary_and_overflow(self) -> None:
        if not HAS_CUDA or torch.version.hip:
            self.skipTest("NVIDIA CUDA not available")

        offsets = torch.tensor([0, 32, 64], dtype=torch.long, device="cuda")
        cluster_ids = torch.tensor([[0, 1]], dtype=torch.long, device="cuda")
        lengths = torch.tensor([[32, 32]], dtype=torch.long, device="cuda")
        embeddings = torch.randint(
            -128, 128, (64, 128), dtype=torch.int8, device="cuda"
        )
        queries = torch.randint(-128, 128, (1, 128), dtype=torch.int8, device="cuda")
        expected_scores, expected_indices = torch.ops.st.fused_kmean_ann(
            offsets, cluster_ids, lengths, embeddings, queries, 64
        )
        scores, indices, overflow = torch.ops.st.fused_kmean_ann_cuda_graph(
            offsets, cluster_ids, lengths, embeddings, queries, 64
        )
        torch.cuda.synchronize()
        self.assertEqual(overflow.item(), 0)
        torch.testing.assert_close(scores, expected_scores, atol=0, rtol=0)
        torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)

        _, truncated_indices, overflow = torch.ops.st.fused_kmean_ann_cuda_graph(
            offsets, cluster_ids, lengths, embeddings, queries, 1
        )
        torch.cuda.synchronize()
        self.assertEqual(overflow.item(), 1)
        self.assertTrue(torch.all(truncated_indices[0] >= 0))

        # Overflow reporting must not poison the CUDA context.
        _, _, recovered_overflow = torch.ops.st.fused_kmean_ann_cuda_graph(
            offsets, cluster_ids, lengths, embeddings, queries, 64
        )
        torch.cuda.synchronize()
        self.assertEqual(recovered_overflow.item(), 0)

    def test_cuda_graph_scan_overflow_flag_resets_on_replay(self) -> None:
        if not HAS_CUDA or torch.version.hip:
            self.skipTest("NVIDIA CUDA not available")

        offsets = torch.tensor([0, 32, 96], dtype=torch.long, device="cuda")
        cluster_ids = torch.tensor([[1]], dtype=torch.long, device="cuda")
        lengths = torch.tensor([[64]], dtype=torch.long, device="cuda")
        embeddings = torch.randint(
            -128, 128, (96, 128), dtype=torch.int8, device="cuda"
        )
        queries = torch.randint(-128, 128, (1, 128), dtype=torch.int8, device="cuda")

        def graph_call() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return torch.ops.st.fused_kmean_ann_cuda_graph(
                offsets, cluster_ids, lengths, embeddings, queries, 32
            )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                graph_call()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, _, overflow = graph_call()

        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(overflow.item(), 1)

        cluster_ids.fill_(0)
        lengths.fill_(32)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(overflow.item(), 0)

        cluster_ids.fill_(1)
        lengths.fill_(64)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(overflow.item(), 1)

    def test_cuda_graph_scan_tail_overflow(self) -> None:
        if not HAS_CUDA or torch.version.hip:
            self.skipTest("NVIDIA CUDA not available")

        cases = (
            ([33, 45], [[0, 1]]),
            # The same warp sees an invalid slot from row 0 and valid slots
            # from row 1; every eight-lane group must still shuffle.
            ([31, 2, 1, 0], [[0, 1], [2, 3]]),
        )
        for sizes, selected in cases:
            with self.subTest(sizes=sizes):
                sizes = torch.tensor(sizes, dtype=torch.long, device="cuda")
                offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)])
                ids = torch.tensor(selected, dtype=torch.long, device="cuda")
                lengths = sizes[ids]
                embeddings = torch.randint(
                    -128, 128, (int(offsets[-1]), 128),
                    dtype=torch.int8, device="cuda",
                )
                queries = torch.randint(
                    -128, 128, (ids.size(0), 128),
                    dtype=torch.int8, device="cuda",
                )

                def call():
                    return torch.ops.st.fused_kmean_ann_cuda_graph(
                        offsets, ids, lengths, embeddings, queries, 32
                    )

                graph, (scores, indices, overflow) = self._capture_scan(call)
                for _ in range(3):
                    queries.copy_(torch.randint_like(queries, -128, 128))
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertEqual(overflow.item(), 1)
                    torch.testing.assert_close(
                        indices[0],
                        torch.arange(32, dtype=torch.int32, device="cuda"),
                        atol=0,
                        rtol=0,
                    )
                    for row in range(ids.size(0)):
                        valid = indices[row] != -1
                        docs = indices[row, valid].long()
                        self.assertTrue(torch.all(docs >= 0))
                        self.assertTrue(torch.all(docs < embeddings.size(0)))
                        expected = (
                            embeddings[docs].int() * queries[row].int()
                        ).sum(dim=1, dtype=torch.int32)
                        torch.testing.assert_close(
                            scores[row, valid], expected, atol=0, rtol=0
                        )
                    if ids.size(0) == 2:
                        self.assertEqual(indices[1, 0].item(), 33)

    def test_cuda_graph_scan_filter_row_mapping(self) -> None:
        if not HAS_CUDA or torch.version.hip:
            self.skipTest("NVIDIA CUDA not available")

        offsets = torch.tensor([0, 33, 78], device="cuda")
        ids = torch.tensor([[0], [1]], device="cuda")
        lengths = torch.tensor([[33], [45]], device="cuda")
        embeddings = torch.randint(
            -128, 128, (78, 128), dtype=torch.int8, device="cuda"
        )
        queries = torch.randint(
            -128, 128, (2, 128), dtype=torch.int8, device="cuda"
        )
        masks = torch.tensor(
            [[-1, -1], [0, 0], [-6148914691236517206] * 2],
            dtype=torch.long, device="cuda",
        )
        mapping = torch.tensor([2, 0], device="cuda")

        def call():
            return torch.ops.st.fused_kmean_ann_cuda_graph(
                offsets, ids, lengths, embeddings, queries, 64,
                filtering_bit_mask=masks, filtering_bit_index=mapping,
            )

        graph, (scores, indices, overflow) = self._capture_scan(call)
        for rows in ([2, 0], [1, 2], [0, 1]):
            mapping.copy_(mapping.new_tensor(rows))
            expected_scores, expected_indices = torch.ops.st.fused_kmean_ann(
                offsets, ids, lengths, embeddings, queries, 64,
                filtering_bit_mask=masks[mapping].contiguous(),
            )
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(overflow.item(), 0)
            torch.testing.assert_close(scores, expected_scores, atol=0, rtol=0)
            torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)

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


# ---------------------------------------------------------------------------
# fused_kmean_ann_with_partial_masks tests
# ---------------------------------------------------------------------------


class TestFusedKmeanAnnWithPartialMasks(unittest.TestCase):
    """Tests for fused_kmean_ann_with_partial_masks op."""

    def test_registered(self) -> None:
        self.assertTrue(hasattr(torch.ops.st, "fused_kmean_ann_with_partial_masks"))
        self.assertTrue(
            hasattr(torch.ops.st, "fused_kmean_ann_with_partial_masks_multiple")
        )

    def _make_partial_masks(
        self,
        cluster_offsets: torch.Tensor,
        cluster_ids: torch.Tensor,
        cluster_length: torch.Tensor,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all-pass partial masks (all bits set) for testing."""
        batch_size = cluster_ids.size(0)
        num_probes = cluster_ids.size(1)

        cumsum_list = []
        offset_list = []
        mask_list = []
        col_acc = 0
        for b in range(batch_size):
            for p in range(num_probes):
                cid = int(cluster_ids[b, p].item())
                clen = int(cluster_length[b, p].item())
                doc_start = int(cluster_offsets[cid].item())
                offset_in_col = doc_start % 64
                num_cols = (clen + offset_in_col + 63) // 64 if clen > 0 else 0
                col_acc += num_cols
                cumsum_list.append(col_acc)
                offset_list.append(offset_in_col)
                for _ in range(num_cols):
                    mask_list.append(-1)

        col_cumsum = torch.tensor(cumsum_list, dtype=torch.int32, device=device)
        first_offsets = torch.tensor(offset_list, dtype=torch.int8, device=device)
        col_masks = torch.tensor(mask_list, dtype=torch.long, device=device)
        return col_cumsum, first_offsets, col_masks

    def _run_multiple(
        self,
        device: str,
        queries: torch.Tensor,
        list_cluster_offsets: list[torch.Tensor],
        list_cluster_ids: list[torch.Tensor],
        list_cluster_lengths: list[torch.Tensor],
        list_embeddings: list[torch.Tensor],
        list_max_sizes: list[int],
        list_col_cumsum: list[torch.Tensor],
        list_first_offsets: list[torch.Tensor],
        list_col_masks: list[torch.Tensor],
        fuse: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return torch.ops.st.fused_kmean_ann_with_partial_masks_multiple(
            queries.to(device),
            [t.to(device) for t in list_cluster_offsets],
            [t.to(device) for t in list_cluster_ids],
            [t.to(device) for t in list_cluster_lengths],
            [t.to(device) for t in list_embeddings],
            list_max_sizes,
            [t.to(device) for t in list_col_cumsum],
            [t.to(device) for t in list_first_offsets],
            [t.to(device) for t in list_col_masks],
            -1,
            -1,
            None,
            None,
            fuse,
        )

    def _verify_results(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        expected_indices: torch.Tensor,
        expected_scores: torch.Tensor,
    ) -> None:
        sorted_indices, sort_order = indices.sort(dim=1, descending=True)
        sorted_scores = torch.gather(scores, 1, sort_order)
        sorted_expected_indices, expected_sort_order = expected_indices.sort(
            dim=1, descending=True
        )
        sorted_expected_scores = torch.gather(expected_scores, 1, expected_sort_order)
        torch.testing.assert_close(
            sorted_indices, sorted_expected_indices.to(torch.int32)
        )
        torch.testing.assert_close(
            sorted_scores,
            sorted_expected_scores,
            atol=0.005,
            rtol=0.01,
        )

    def _build_multiple_case(
        self,
        dim: int,
        batch_size: int,
        cluster_lengths_per_index: list[list[int]],
        dtype: torch.dtype = torch.float32,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[int],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        if dtype == torch.int8:
            queries = torch.randint(-16, 16, (batch_size, dim), dtype=dtype)
        else:
            queries = torch.randn(batch_size, dim, dtype=dtype)

        list_cluster_offsets = []
        list_cluster_ids = []
        list_cluster_lengths = []
        list_embeddings = []
        list_max_sizes = []
        list_col_cumsum = []
        list_first_offsets = []
        list_col_masks = []

        for index_id, cluster_lengths in enumerate(cluster_lengths_per_index):
            cluster_lengths_t = torch.tensor(cluster_lengths, dtype=torch.long)
            cluster_offsets = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long),
                    cluster_lengths_t.cumsum(0)[:-1],
                ]
            )
            total_docs = int(cluster_lengths_t.sum().item())
            if dtype == torch.int8:
                embeddings = torch.randint(-16, 16, (total_docs, dim), dtype=dtype)
            else:
                embeddings = torch.randn(total_docs, dim, dtype=dtype)
            n_clusters = len(cluster_lengths)
            n_probes = min(3, n_clusters)
            cluster_ids = (
                torch.arange(batch_size * n_probes, dtype=torch.long)
                .view(batch_size, n_probes)
                .add(index_id)
                .remainder(n_clusters)
            )
            selected_cluster_lengths = cluster_lengths_t[cluster_ids]
            max_size = int(selected_cluster_lengths.sum(-1).max().item())
            col_cumsum, first_offsets, col_masks = self._make_partial_masks(
                cluster_offsets,
                cluster_ids,
                selected_cluster_lengths,
            )

            list_cluster_offsets.append(cluster_offsets)
            list_cluster_ids.append(cluster_ids)
            list_cluster_lengths.append(selected_cluster_lengths)
            list_embeddings.append(embeddings)
            list_max_sizes.append(max_size)
            list_col_cumsum.append(col_cumsum)
            list_first_offsets.append(first_offsets)
            list_col_masks.append(col_masks)

        return (
            queries,
            list_cluster_offsets,
            list_cluster_ids,
            list_cluster_lengths,
            list_embeddings,
            list_max_sizes,
            list_col_cumsum,
            list_first_offsets,
            list_col_masks,
        )

    def test_cpu_basic(self) -> None:
        n_clusters = 3
        cluster_size = 32
        dim = 64
        total_docs = n_clusters * cluster_size
        cluster_offsets = torch.arange(
            0, total_docs + 1, cluster_size, dtype=torch.long
        )
        embeddings = torch.randn(total_docs, dim)
        queries = torch.randn(2, dim)
        cluster_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        cluster_length = torch.full((2, 2), cluster_size, dtype=torch.long)
        max_size = 2 * cluster_size

        col_cumsum, first_offsets, col_masks = self._make_partial_masks(
            cluster_offsets, cluster_ids, cluster_length
        )

        scores, indices = torch.ops.st.fused_kmean_ann_with_partial_masks(
            cluster_offsets,
            cluster_ids,
            cluster_length,
            embeddings,
            queries,
            max_size,
            col_cumsum,
            first_offsets,
            col_masks,
        )
        self.assertEqual(scores.size(0), 2)
        self.assertEqual(indices.size(0), 2)
        self.assertEqual(scores.dtype, torch.float32)
        self.assertEqual(indices.dtype, torch.int32)

    def test_gpu_runs(self) -> None:
        if not HAS_CUDA:
            self.skipTest("CUDA not available")

        n_clusters = 3
        cluster_size = 32
        dim = 64
        total_docs = n_clusters * cluster_size
        cluster_offsets = torch.arange(
            0, total_docs + 1, cluster_size, dtype=torch.long
        )
        embeddings = torch.randn(total_docs, dim)
        queries = torch.randn(2, dim)
        cluster_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        cluster_length = torch.full((2, 2), cluster_size, dtype=torch.long)
        max_size = 2 * cluster_size

        col_cumsum, first_offsets, col_masks = self._make_partial_masks(
            cluster_offsets, cluster_ids, cluster_length, "cuda"
        )

        scores, indices = torch.ops.st.fused_kmean_ann_with_partial_masks(
            cluster_offsets.cuda(),
            cluster_ids.cuda(),
            cluster_length.cuda(),
            embeddings.cuda(),
            queries.cuda(),
            max_size,
            col_cumsum,
            first_offsets,
            col_masks,
        )
        self.assertTrue(scores.is_cuda)
        self.assertEqual(scores.size(0), 2)

    def _run_int8_with_partial_masks(self, cluster_size: int, dim: int) -> None:
        if not HAS_CUDA:
            self.skipTest("CUDA not available")

        n_clusters = 2
        total_docs = n_clusters * cluster_size
        cluster_offsets = torch.arange(
            0, total_docs + 1, cluster_size, dtype=torch.long
        )
        embeddings = torch.randint(-128, 127, (total_docs, dim), dtype=torch.int8)
        queries = torch.randint(-128, 127, (1, dim), dtype=torch.int8)
        cluster_ids = torch.tensor([[0, 1]], dtype=torch.long)
        cluster_length = torch.full((1, 2), cluster_size, dtype=torch.long)
        max_size = 2 * cluster_size

        col_cumsum, first_offsets, col_masks = self._make_partial_masks(
            cluster_offsets, cluster_ids, cluster_length, "cuda"
        )

        scores, indices = torch.ops.st.fused_kmean_ann_with_partial_masks(
            cluster_offsets.cuda(),
            cluster_ids.cuda(),
            cluster_length.cuda(),
            embeddings.cuda(),
            queries.cuda(),
            max_size,
            col_cumsum,
            first_offsets,
            col_masks,
            divisor_for_int8=127,
        )
        self.assertEqual(scores.dtype, torch.float16)

    def test_int8_with_partial_masks(self) -> None:
        self._run_int8_with_partial_masks(cluster_size=32, dim=64)

    def test_int8_dim_256_with_partial_masks(self) -> None:
        self._run_int8_with_partial_masks(cluster_size=33, dim=256)

    def test_fused_kmean_ann_with_partial_masks_multiple_matches_individual_cpu(
        self,
    ) -> None:
        (
            queries,
            list_cluster_offsets,
            list_cluster_ids,
            list_cluster_lengths,
            list_embeddings,
            list_max_sizes,
            list_col_cumsum,
            list_first_offsets,
            list_col_masks,
        ) = self._build_multiple_case(
            dim=64,
            batch_size=2,
            cluster_lengths_per_index=[[31, 33, 8], [16, 32, 24], [20, 40, 12]],
        )

        multi_scores, multi_indices = self._run_multiple(
            "cpu",
            queries,
            list_cluster_offsets,
            list_cluster_ids,
            list_cluster_lengths,
            list_embeddings,
            list_max_sizes,
            list_col_cumsum,
            list_first_offsets,
            list_col_masks,
        )

        self.assertEqual(len(multi_scores), len(list_cluster_offsets))
        self.assertEqual(len(multi_indices), len(list_cluster_offsets))
        for i in range(len(list_cluster_offsets)):
            individual_scores, individual_indices = (
                torch.ops.st.fused_kmean_ann_with_partial_masks(
                    list_cluster_offsets[i],
                    list_cluster_ids[i],
                    list_cluster_lengths[i],
                    list_embeddings[i],
                    queries,
                    list_max_sizes[i],
                    list_col_cumsum[i],
                    list_first_offsets[i],
                    list_col_masks[i],
                )
            )
            self._verify_results(
                multi_indices[i],
                multi_scores[i],
                individual_indices,
                individual_scores,
            )

    def test_fused_kmean_ann_with_partial_masks_multiple_fuse_parity_gpu(
        self,
    ) -> None:
        if not HAS_CUDA:
            self.skipTest("CUDA not available")

        for dim, num_indices, batch_size in (
            (64, 1, 4),
            (96, 2, 4),
            (128, 4, 8),
        ):
            with self.subTest(dim=dim, num_indices=num_indices, batch_size=batch_size):
                cluster_lengths_per_index = [
                    [31 + i, 32, 33, 17 + i] for i in range(num_indices)
                ]
                (
                    queries,
                    list_cluster_offsets,
                    list_cluster_ids,
                    list_cluster_lengths,
                    list_embeddings,
                    list_max_sizes,
                    list_col_cumsum,
                    list_first_offsets,
                    list_col_masks,
                ) = self._build_multiple_case(
                    dim=dim,
                    batch_size=batch_size,
                    cluster_lengths_per_index=cluster_lengths_per_index,
                    dtype=torch.float16,
                )

                nonfused_scores, nonfused_indices = self._run_multiple(
                    "cuda",
                    queries,
                    list_cluster_offsets,
                    list_cluster_ids,
                    list_cluster_lengths,
                    list_embeddings,
                    list_max_sizes,
                    list_col_cumsum,
                    list_first_offsets,
                    list_col_masks,
                    fuse=False,
                )
                fused_scores, fused_indices = self._run_multiple(
                    "cuda",
                    queries,
                    list_cluster_offsets,
                    list_cluster_ids,
                    list_cluster_lengths,
                    list_embeddings,
                    list_max_sizes,
                    list_col_cumsum,
                    list_first_offsets,
                    list_col_masks,
                    fuse=True,
                )

                self.assertEqual(len(nonfused_scores), num_indices)
                self.assertEqual(len(fused_scores), num_indices)
                for i in range(num_indices):
                    self._verify_results(
                        fused_indices[i],
                        fused_scores[i],
                        nonfused_indices[i],
                        nonfused_scores[i],
                    )
                    individual_scores, individual_indices = (
                        torch.ops.st.fused_kmean_ann_with_partial_masks(
                            list_cluster_offsets[i].cuda(),
                            list_cluster_ids[i].cuda(),
                            list_cluster_lengths[i].cuda(),
                            list_embeddings[i].cuda(),
                            queries.cuda(),
                            list_max_sizes[i],
                            list_col_cumsum[i].cuda(),
                            list_first_offsets[i].cuda(),
                            list_col_masks[i].cuda(),
                        )
                    )
                    self._verify_results(
                        fused_indices[i],
                        fused_scores[i],
                        individual_indices,
                        individual_scores,
                    )
