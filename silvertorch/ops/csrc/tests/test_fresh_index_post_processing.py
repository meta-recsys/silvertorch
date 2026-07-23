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

torch.ops.load_library(
    "//silvertorch/oss/ops/csrc:fresh_index_post_processing"
# @oss-disable[end= ]: )
torch.ops.load_library(
    "//silvertorch/oss/ops/csrc:fresh_index_post_processing_gpu"
# @oss-disable[end= ]: )

HAS_CUDA = torch.cuda.is_available()


class FreshIndexPostProcessingTest(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertTrue(
            hasattr(torch.ops.st, "take_top_k_and_gather_from_main_and_fresh")
        )

    def _build_inputs(
        self, device: torch.device, index_dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # All merged scores per row are distinct so the top-k selection is
        # unambiguous across both devices.
        main_scores = torch.tensor(
            [
                [0.90, 0.10, 0.70, 0.80, 0.40],
                [0.15, 0.28, 0.85, 0.05, 0.38],
            ],
            dtype=torch.float16,
            device=device,
        )
        fresh_scores = torch.tensor(
            [
                [0.95, 0.20, 0.30, 0.60],
                [0.62, 0.99, 0.48, 0.31],
            ],
            dtype=torch.float16,
            device=device,
        )
        main_indices = torch.tensor(
            [
                [2, 5, 4, 3, 0],
                [6, 2, 3, 1, 0],
            ],
            dtype=index_dtype,
            device=device,
        )
        fresh_indices = torch.tensor(
            [
                [1, 0, 2, 3],
                [4, 1, 3, 0],
            ],
            dtype=index_dtype,
            device=device,
        )
        main_item_ids = torch.tensor(
            [10, 20, 30, 40, 50, 60, 70], dtype=torch.int64, device=device
        )
        fresh_item_ids = torch.tensor(
            [100, 200, 300, 400, 500], dtype=torch.int64, device=device
        )
        vecs = [
            [float(i), float(i) + 0.5, float(i) - 0.5, float(i) * 2.0] for i in range(7)
        ]
        main_embeddings = torch.tensor(vecs, dtype=torch.bfloat16, device=device)
        fresh_embeddings = torch.tensor(vecs[:5], dtype=torch.bfloat16, device=device)
        return (
            main_scores,
            fresh_scores,
            main_indices,
            fresh_indices,
            main_item_ids,
            fresh_item_ids,
            main_embeddings,
            fresh_embeddings,
        )

    def _gather_reference(
        self,
        top_indices_indices: torch.Tensor,
        main_indices: torch.Tensor,
        fresh_indices: torch.Tensor,
        main_data: torch.Tensor,
        fresh_data: torch.Tensor,
        fresh_vs_full_div_index: int,
    ) -> torch.Tensor:
        gathered = []
        for row in range(top_indices_indices.size(0)):
            for col in range(top_indices_indices.size(1)):
                top_index = int(top_indices_indices[row, col].item())
                if top_index < fresh_vs_full_div_index:
                    src_index = int(fresh_indices[row, top_index].item())
                    if src_index < 0 or src_index >= fresh_data.size(0):
                        src_index = 0
                    gathered.append(fresh_data[src_index])
                else:
                    src_index = int(
                        main_indices[row, top_index - fresh_vs_full_div_index].item()
                    )
                    if src_index < 0 or src_index >= main_data.size(0):
                        src_index = 0
                    gathered.append(main_data[src_index])

        if main_data.dim() == 2:
            return torch.stack(gathered).reshape(-1, main_data.size(1))
        return torch.stack(gathered).reshape(-1)

    def _reference_top_k_and_gather(
        self,
        main_scores: torch.Tensor,
        fresh_scores: torch.Tensor,
        main_indices: torch.Tensor,
        fresh_indices: torch.Tensor,
        k: int,
        keys: list[str],
        list_main_data: list[torch.Tensor],
        list_fresh_data: list[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        fresh_vs_full_div_index = fresh_scores.size(1)
        merged_scores = torch.cat([fresh_scores, main_scores], dim=1)
        k_clamped = min(k, merged_scores.size(1))
        ref_top_scores, ref_top_indices_indices = merged_scores.topk(
            k_clamped, largest=True, sorted=True
        )
        ref_is_from_fresh = ref_top_indices_indices < fresh_vs_full_div_index
        ref_gathered = {}
        for key, main_data, fresh_data in zip(
            keys, list_main_data, list_fresh_data, strict=True
        ):
            ref_gathered[key] = self._gather_reference(
                ref_top_indices_indices,
                main_indices,
                fresh_indices,
                main_data,
                fresh_data,
                fresh_vs_full_div_index,
            )
        return (
            ref_top_scores,
            ref_top_indices_indices,
            ref_is_from_fresh,
            ref_gathered,
        )

    def _check_take_top_k_and_gather(self, device: torch.device) -> None:
        keys = ["item_ids", "embeddings"]
        for kernel_version in [1, 2]:
            for index_dtype in [torch.int32, torch.int64]:
                for k in [3, 7, 20]:
                    (
                        main_scores,
                        fresh_scores,
                        main_indices,
                        fresh_indices,
                        main_item_ids,
                        fresh_item_ids,
                        main_embeddings,
                        fresh_embeddings,
                    ) = self._build_inputs(device, index_dtype)
                    list_main_data = [main_item_ids, main_embeddings]
                    list_fresh_data = [fresh_item_ids, fresh_embeddings]

                    (
                        top_scores,
                        top_indices_indices,
                        is_from_fresh,
                        gathered,
                    ) = torch.ops.st.take_top_k_and_gather_from_main_and_fresh(
                        main_scores=main_scores,
                        fresh_scores=fresh_scores,
                        main_indices=main_indices,
                        fresh_indices=fresh_indices,
                        k=k,
                        keys=keys,
                        list_main_data=list_main_data,
                        list_fresh_data=list_fresh_data,
                        kernel_version=kernel_version,
                    )

                    (
                        ref_top_scores,
                        ref_top_indices_indices,
                        ref_is_from_fresh,
                        ref_gathered,
                    ) = self._reference_top_k_and_gather(
                        main_scores,
                        fresh_scores,
                        main_indices,
                        fresh_indices,
                        k,
                        keys,
                        list_main_data,
                        list_fresh_data,
                    )

                    msg = (
                        f"kernel_version={kernel_version}, index_dtype={index_dtype}, "
                        f"k={k}, device={device}"
                    )
                    self.assertTrue(torch.equal(top_scores, ref_top_scores), msg)
                    self.assertTrue(
                        torch.equal(top_indices_indices, ref_top_indices_indices), msg
                    )
                    self.assertTrue(torch.equal(is_from_fresh, ref_is_from_fresh), msg)
                    self.assertCountEqual(list(gathered.keys()), keys)
                    for key in keys:
                        self.assertTrue(
                            torch.equal(gathered[key], ref_gathered[key]),
                            f"gather mismatch for key={key}, {msg}",
                        )

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    def test_take_top_k_and_gather_from_main_and_fresh_gpu(self) -> None:
        self._check_take_top_k_and_gather(torch.device("cuda"))

    def test_take_top_k_and_gather_from_main_and_fresh_cpu(self) -> None:
        self._check_take_top_k_and_gather(torch.device("cpu"))
