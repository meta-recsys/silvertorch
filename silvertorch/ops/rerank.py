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

from functools import lru_cache
from typing import Callable

import torch


def _rerank_scores_fp32_impl(
    embeddings: torch.Tensor,
    queries: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    valid = candidate_ids.ge(0) & candidate_ids.lt(embeddings.size(0))
    safe_ids = candidate_ids.clamp(0, embeddings.size(0) - 1).to(torch.int64)
    scores = (embeddings[safe_ids] * queries[:, None, :]).sum(dim=2)
    return scores.masked_fill(~valid, -torch.inf)


@lru_cache(maxsize=1)
def _compiled_reranker() -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]:
    return torch.compile(_rerank_scores_fp32_impl, fullgraph=True, dynamic=True)


def rerank_scores_fp32(
    embeddings: torch.Tensor,
    queries: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    use_compile: bool = False,
) -> torch.Tensor:
    """Compute FP32 dot products for selected candidate IDs.

    The eager path is the portable default.  With ``use_compile=True``,
    ``torch.compile`` can fuse candidate gather, multiplication, reduction,
    and invalid-ID masking so the ``(batch, candidates, dimension)`` gathered
    vectors and multiplication result do not need to be materialized as
    separate tensors.  The first compiled call includes compilation latency;
    warm the helper before measuring steady-state performance.

    Candidate IDs outside ``[0, embeddings.size(0))`` receive a score of
    negative infinity.  This includes the ``-1`` padding emitted by clustered
    ANN operators and prevents padding from being returned as a valid result.

    Args:
        embeddings: FP32 tensor of shape ``(num_embeddings, dimension)``.
        queries: FP32 tensor of shape ``(batch, dimension)``.
        candidate_ids: int32 or int64 tensor of shape ``(batch, candidates)``.
        use_compile: Use the lazily created ``torch.compile`` implementation.

    Returns:
        FP32 scores of shape ``(batch, candidates)``.
    """
    if embeddings.dim() != 2:
        raise ValueError("embeddings must be a 2-D tensor")
    if queries.dim() != 2:
        raise ValueError("queries must be a 2-D tensor")
    if candidate_ids.dim() != 2:
        raise ValueError("candidate_ids must be a 2-D tensor")
    if embeddings.dtype != torch.float32 or queries.dtype != torch.float32:
        raise TypeError("embeddings and queries must use torch.float32")
    if candidate_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("candidate_ids must use torch.int32 or torch.int64")
    if embeddings.device != queries.device or embeddings.device != candidate_ids.device:
        raise ValueError("embeddings, queries, and candidate_ids must share a device")
    if embeddings.size(1) != queries.size(1):
        raise ValueError("embedding and query dimensions must match")
    if queries.size(0) != candidate_ids.size(0):
        raise ValueError("query and candidate batch sizes must match")
    if embeddings.size(0) == 0 and candidate_ids.numel() != 0:
        raise ValueError("embeddings cannot be empty when candidates are present")

    implementation = _compiled_reranker() if use_compile else _rerank_scores_fp32_impl
    return implementation(embeddings, queries, candidate_ids)
