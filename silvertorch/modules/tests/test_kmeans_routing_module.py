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

import torch
import torch.nn.functional as F
from silvertorch.modules.kmeans_routing_module import KMeansRoutingModule


class TestKMeansRoutingModule(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20261101)
        self.centroids = torch.randn(128, 32, dtype=torch.float32)
        self.queries = torch.randn(4, 32, dtype=torch.float32)

    def _manual_route(
        self, queries: torch.Tensor, centroids: torch.Tensor, nprobe: int
    ) -> torch.Tensor:
        scores = torch.mm(F.normalize(queries, dim=1), centroids.t())
        return torch.topk(scores, nprobe, dim=1, sorted=True).indices

    def test_fp32_is_default_and_matches_existing_composition(self) -> None:
        module = KMeansRoutingModule(self.centroids, nprobe=8)
        actual = module(self.queries)
        expected = self._manual_route(self.queries, self.centroids, 8)

        self.assertFalse(module.use_fp16_routing)
        self.assertIs(module.centroids, self.centroids)
        self.assertEqual(module.centroids.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_fp16_matches_manual_lower_precision_routing(self) -> None:
        module = KMeansRoutingModule(
            self.centroids, nprobe=8, use_fp16_routing=True
        )
        actual = module(self.queries)
        expected = torch.topk(
            torch.mm(
                F.normalize(self.queries, dim=1).half(),
                self.centroids.half().t(),
            ),
            8,
            dim=1,
            sorted=True,
        ).indices

        manual_scores = torch.mm(
            F.normalize(self.queries, dim=1).half(), self.centroids.half().t()
        )
        actual_scores = torch.gather(manual_scores, 1, actual)

        self.assertEqual(module.centroids.dtype, torch.float16)
        self.assertEqual(actual.dtype, torch.int64)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.all(actual_scores[:, :-1] >= actual_scores[:, 1:]))

    def test_fp16_owns_only_one_half_precision_centroid_buffer(self) -> None:
        module = KMeansRoutingModule(
            self.centroids, nprobe=8, use_fp16_routing=True
        )
        buffers = list(module.buffers())
        self.assertEqual(len(buffers), 1)
        self.assertEqual(buffers[0].dtype, torch.float16)
        self.assertEqual(
            buffers[0].numel() * buffers[0].element_size(),
            self.centroids.numel() * 2,
        )

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D"):
            KMeansRoutingModule(torch.randn(32), nprobe=1)
        with self.assertRaisesRegex(TypeError, "float32"):
            KMeansRoutingModule(self.centroids.half(), nprobe=1)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            KMeansRoutingModule(self.centroids.t(), nprobe=1)
        with self.assertRaisesRegex(ValueError, "nprobe"):
            KMeansRoutingModule(self.centroids, nprobe=0)
        with self.assertRaisesRegex(ValueError, "nprobe"):
            KMeansRoutingModule(self.centroids, nprobe=129)

        module = KMeansRoutingModule(self.centroids, nprobe=8)
        with self.assertRaisesRegex(ValueError, "2D"):
            module(torch.randn(32))
        with self.assertRaisesRegex(TypeError, "float32"):
            module(self.queries.half())
        with self.assertRaisesRegex(ValueError, "dimension"):
            module(torch.randn(4, 16))

    def test_cuda_graph_replays_with_new_queries(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        module = KMeansRoutingModule(
            self.centroids.cuda(), nprobe=8, use_fp16_routing=True
        )
        static_queries = self.queries.cuda()

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                module(static_queries)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_ids = module(static_queries)

        for _ in range(3):
            replacement = torch.randn_like(static_queries)
            expected = module(replacement)
            static_queries.copy_(replacement)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(graph_ids, expected))


if __name__ == "__main__":
    unittest.main()
