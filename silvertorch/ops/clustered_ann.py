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

from typing import Optional

import torch


def selected_cluster_capacity(selected_cluster_lengths: torch.Tensor) -> int:
    """Return the exact output width required by a batch of selected clusters.

    ``fused_kmean_ann`` writes one candidate slot for every document in every
    selected cluster.  Its output therefore only needs to be as wide as the
    largest per-query sum of selected cluster lengths.

    This function calls ``Tensor.item()``.  For a CUDA tensor that introduces
    a device-to-host synchronization, so latency-sensitive and CUDA Graph
    callers should compute a static conservative capacity once and pass it to
    :func:`clustered_ann_search` instead.

    Args:
        selected_cluster_lengths: Tensor of shape ``(batch, num_probes)``.

    Returns:
        The required candidate capacity.  An empty batch has capacity zero.
    """
    if selected_cluster_lengths.dim() != 2:
        raise ValueError("selected_cluster_lengths must be a 2-D tensor")
    if selected_cluster_lengths.size(0) == 0:
        return 0
    return int(selected_cluster_lengths.sum(dim=1).max().item())


def clustered_ann_search(
    cluster_offsets: torch.Tensor,
    selected_cluster_ids: torch.Tensor,
    selected_cluster_lengths: torch.Tensor,
    embeddings: torch.Tensor,
    queries: torch.Tensor,
    max_tensor_size_per_row: Optional[int] = None,
    filtering_bit_mask: Optional[torch.Tensor] = None,
    invalid_index_value: int = -1,
    divisor_for_int8: int = -1,
    filtering_bit_index: Optional[torch.Tensor] = None,
    per_embedding_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run clustered ANN search with an optional tight candidate capacity.

    When ``max_tensor_size_per_row`` is omitted, the helper computes the exact
    capacity required by the selected clusters in this batch.  Passing an
    explicit value preserves a static allocation strategy and avoids the
    synchronization described by :func:`selected_cluster_capacity`.

    Filtering does not reduce the required capacity: the low-level operator
    retains slots for filtered documents, so capacity is always based on the
    complete selected cluster lengths.

    All arguments other than ``max_tensor_size_per_row`` are forwarded to
    ``torch.ops.st.fused_kmean_ann`` unchanged.
    """
    if max_tensor_size_per_row is None:
        max_tensor_size_per_row = selected_cluster_capacity(
            selected_cluster_lengths
        )
    if max_tensor_size_per_row < 0:
        raise ValueError("max_tensor_size_per_row must be non-negative")

    if not hasattr(torch.ops.st, "fused_kmean_ann"):
        import silvertorch.ops._load_ops  # noqa: F401

    return torch.ops.st.fused_kmean_ann(
        cluster_offsets,
        selected_cluster_ids,
        selected_cluster_lengths,
        embeddings,
        queries,
        max_tensor_size_per_row,
        filtering_bit_mask,
        invalid_index_value,
        divisor_for_int8,
        filtering_bit_index,
        per_embedding_scale,
    )
