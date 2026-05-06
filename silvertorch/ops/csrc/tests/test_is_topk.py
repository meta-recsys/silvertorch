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

"""Tests for is_topk op -- CPU and GPU cross-validated."""

import unittest

import silvertorch.ops._load_ops  # noqa: F401
import torch

torch.ops.load_library("//silvertorch/oss/ops/csrc:is_topk")
torch.ops.load_library("//silvertorch/oss/ops/csrc:is_topk_gpu")

HAS_CUDA = torch.cuda.is_available()


def _is_topk_reference(
    scores: torch.Tensor, ks: torch.Tensor, largest: bool
) -> torch.Tensor:
    """Pure-Python reference: double argsort to produce a boolean mask."""
    batch_size, n = scores.shape
    result = torch.zeros_like(scores)
    for b in range(batch_size):
        k = min(int(ks[b].item()), n)
        if k <= 0:
            continue
        row = scores[b].float()
        _, idx = row.sort(descending=largest)
        result[b][idx[:k]] = 1.0
    return result


class TestIsTopk(unittest.TestCase):
    """Each test verifies CPU and GPU against the same expected results."""

    def _run_on_device(
        self,
        device: str,
        scores: torch.Tensor,
        ks: torch.Tensor,
        largest: bool = True,
    ) -> torch.Tensor:
        s = scores.to(device)
        k = ks.to(device)
        output = torch.zeros_like(s)
        torch.ops.st.is_topk(s, k, output, largest)
        return output.cpu()

    def _verify_on_devices(
        self,
        scores: torch.Tensor,
        ks: torch.Tensor,
        expected: torch.Tensor,
        largest: bool = True,
    ) -> None:
        result_cpu = self._run_on_device("cpu", scores, ks, largest)
        torch.testing.assert_close(result_cpu, expected, msg=lambda m: f"[cpu] {m}")

        if HAS_CUDA:
            result_gpu = self._run_on_device("cuda", scores, ks, largest)
            torch.testing.assert_close(
                result_gpu, expected, msg=lambda m: f"[cuda] {m}"
            )

    # ---- deterministic tests ----

    def test_simple_largest(self) -> None:
        scores = torch.tensor(
            [
                [0.5, 0.4, 0.8, 1.2],
                [1.8, 1.9, 0.4, 0.3],
                [0.8, 0.7, 0.9, 0.6],
            ],
            dtype=torch.float32,
        )
        ks = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self._verify_on_devices(scores, ks, expected, largest=True)

    def test_simple_smallest(self) -> None:
        scores = torch.tensor(
            [
                [0.5, 0.4, 0.8, 1.2],
                [1.8, 1.9, 0.4, 0.3],
                [0.8, 0.7, 0.9, 0.6],
            ],
            dtype=torch.float32,
        )
        ks = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self._verify_on_devices(scores, ks, expected, largest=False)

    def test_k_greater_than_n(self) -> None:
        """When k >= n, all elements should be selected."""
        scores = torch.rand(2, 3, dtype=torch.float32)
        ks = torch.tensor([5, 10], dtype=torch.int64)
        expected = torch.ones_like(scores)
        self._verify_on_devices(scores, ks, expected, largest=True)

    def test_k_equals_n(self) -> None:
        scores = torch.rand(2, 4, dtype=torch.float32)
        ks = torch.tensor([4, 4], dtype=torch.int64)
        expected = torch.ones_like(scores)
        self._verify_on_devices(scores, ks, expected, largest=True)

    # ---- randomized cross-device tests ----

    def _run_random(
        self,
        batch_size: int,
        n: int,
        dtype: torch.dtype = torch.float32,
        largest: bool = True,
    ) -> None:
        scores = torch.randn(batch_size, n, dtype=dtype)
        # Use unique scores to avoid tie-breaking ambiguity
        scores = scores + torch.arange(n).float() * 1e-6
        ks = torch.randint(1, n + 1, (batch_size,), dtype=torch.int64)
        expected = _is_topk_reference(scores, ks, largest)
        self._verify_on_devices(scores, ks, expected, largest)

    def test_random_float32_largest(self) -> None:
        self._run_random(batch_size=5, n=100, dtype=torch.float32, largest=True)

    def test_random_float32_smallest(self) -> None:
        self._run_random(batch_size=5, n=100, dtype=torch.float32, largest=False)

    def test_random_float16(self) -> None:
        self._run_random(batch_size=3, n=50, dtype=torch.float16, largest=True)

    def test_random_bfloat16(self) -> None:
        self._run_random(batch_size=3, n=50, dtype=torch.bfloat16, largest=True)

    def test_random_large(self) -> None:
        self._run_random(batch_size=4, n=1000, dtype=torch.float32, largest=True)

    def test_random_single_row(self) -> None:
        self._run_random(batch_size=1, n=200, dtype=torch.float32, largest=True)

    # ---- registration test ----

    def test_registered(self) -> None:
        self.assertTrue(hasattr(torch.ops.st, "is_topk"))
