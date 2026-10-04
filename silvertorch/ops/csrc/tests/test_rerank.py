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

import torch
from silvertorch.ops.rerank import rerank_scores_fp32


class TestRerankScoresFp32(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 1.0],
                [-1.0, 2.0, 0.0],
                [4.0, 1.0, -2.0],
            ],
            dtype=torch.float32,
        )
        self.queries = torch.tensor(
            [[2.0, -1.0, 0.5], [1.0, 2.0, -1.0]], dtype=torch.float32
        )

    def test_matches_reference(self) -> None:
        candidate_ids = torch.tensor([[0, 2, 3], [1, 0, 2]], dtype=torch.int32)
        actual = rerank_scores_fp32(
            self.embeddings, self.queries, candidate_ids
        )
        expected = torch.stack(
            [
                self.embeddings[candidate_ids[0].long()] @ self.queries[0],
                self.embeddings[candidate_ids[1].long()] @ self.queries[1],
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_invalid_ids_receive_negative_infinity(self) -> None:
        candidate_ids = torch.tensor([[0, -1, 4], [1, -2, 2]])
        actual = rerank_scores_fp32(
            self.embeddings, self.queries, candidate_ids
        )
        self.assertTrue(torch.isneginf(actual[0, 1]))
        self.assertTrue(torch.isneginf(actual[0, 2]))
        self.assertTrue(torch.isneginf(actual[1, 1]))
        self.assertTrue(torch.isfinite(actual[0, 0]))

    def test_supports_non_contiguous_candidate_ids(self) -> None:
        candidate_ids = torch.tensor([[0, 3, 2, 1], [1, 2, 0, 3]])[:, ::2]
        self.assertFalse(candidate_ids.is_contiguous())
        actual = rerank_scores_fp32(
            self.embeddings, self.queries, candidate_ids
        )
        self.assertEqual(actual.shape, candidate_ids.shape)

    def test_requires_fp32_inputs(self) -> None:
        with self.assertRaisesRegex(TypeError, "torch.float32"):
            rerank_scores_fp32(
                self.embeddings.half(),
                self.queries,
                torch.tensor([[0], [1]]),
            )

    def test_requires_matching_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            rerank_scores_fp32(
                self.embeddings,
                self.queries,
                torch.tensor([[0, 1]]),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_compiled_cuda_matches_eager(self) -> None:
        embeddings = self.embeddings.cuda()
        queries = self.queries.cuda()
        candidate_ids = torch.tensor(
            [[0, 2, -1], [1, 3, 2]], dtype=torch.int32, device="cuda"
        )
        eager = rerank_scores_fp32(embeddings, queries, candidate_ids)
        compiled = rerank_scores_fp32(
            embeddings, queries, candidate_ids, use_compile=True
        )
        torch.testing.assert_close(compiled, eager)


if __name__ == "__main__":
    unittest.main()
