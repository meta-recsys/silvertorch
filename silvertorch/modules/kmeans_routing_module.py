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

import torch
import torch.nn as nn
import torch.nn.functional as F


class KMeansRoutingModule(nn.Module):
    """Route queries to their highest-scoring K-means centroids.

    Queries are L2-normalized before the centroid matrix multiplication. By
    default, routing remains entirely in FP32. ``use_fp16_routing=True`` is an
    explicit, construction-time optimization that stores the routing centroid
    matrix in FP16 and casts normalized queries immediately before the matrix
    multiplication. The returned cluster IDs are always int64.

    Keeping the precision choice static makes both paths CUDA-graph capturable
    and avoids any copy or cast in the default FP32 path.

    PyTorch may use reduced-precision accumulation for FP16 matrix
    multiplication when
    ``torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`` is
    enabled. This can introduce additional routing-score error around ties.

    Args:
        centroids: Contiguous FP32 tensor shaped ``(num_clusters, dimension)``.
        nprobe: Number of clusters to select for each query.
        use_fp16_routing: Use FP16 for only the routing matrix multiplication
            and top-k. Defaults to False.
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        nprobe: int,
        use_fp16_routing: bool = False,
    ) -> None:
        super().__init__()
        if centroids.dim() != 2:
            raise ValueError("centroids must be a 2D tensor")
        if centroids.dtype != torch.float32:
            raise TypeError("centroids must have dtype torch.float32")
        if not centroids.is_contiguous():
            raise ValueError("centroids must be contiguous")
        if nprobe <= 0 or nprobe > centroids.size(0):
            raise ValueError(
                f"nprobe must be in [1, {centroids.size(0)}], got {nprobe}"
            )

        self.nprobe = nprobe
        self.use_fp16_routing = use_fp16_routing
        routing_centroids = centroids.half() if use_fp16_routing else centroids
        self.register_buffer("centroids", routing_centroids)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """Return the selected centroid IDs for each query.

        Args:
            queries: FP32 tensor shaped ``(batch, dimension)``.

        Returns:
            Int64 tensor shaped ``(batch, nprobe)``.
        """
        if queries.dim() != 2:
            raise ValueError("queries must be a 2D tensor")
        if queries.dtype != torch.float32:
            raise TypeError("queries must have dtype torch.float32")
        if queries.size(1) != self.centroids.size(1):
            raise ValueError(
                "query dimension must match the centroid dimension: "
                f"{queries.size(1)} != {self.centroids.size(1)}"
            )
        if queries.device != self.centroids.device:
            raise ValueError("queries and centroids must be on the same device")

        normalized_queries = F.normalize(queries, dim=1)
        if self.use_fp16_routing:
            normalized_queries = normalized_queries.half()
        routing_scores = torch.mm(normalized_queries, self.centroids.t())
        return torch.topk(routing_scores, self.nprobe, dim=1, sorted=True).indices
