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

import silvertorch.ops._load_ops  # noqa: F401
import torch
from silvertorch.modules.bloom_index_search_module import BloomIndexSearchModule


class BloomIndexSearchModuleBuilder:
    """Builder for constructing BloomIndexSearchModule instances.

    Example::

        module = (
            BloomIndexSearchModuleBuilder(k=3, hash_k=7)
            .set_feature_data(feature_ids, feature_offsets, feature_values)
            .set_b_multiplier(2.0)
            .build()
        )

    Or with a pre-built bloom index::

        module = (
            BloomIndexSearchModuleBuilder(k=3, hash_k=7)
            .set_bloom_index(bloom_index)
            .set_bloom_bundle_b_offsets(bloom_bundle_b_offsets)
            .build()
        )

    Args:
        k: Number of hash functions for bloom filter lookup.
        hash_k: Number of pre-calculated one-bit hashes.
    """

    def __init__(self, k: int, hash_k: int) -> None:
        self._k = k
        self._hash_k = hash_k
        self._bloom_index: torch.Tensor | None = None
        self._bloom_bundle_b_offsets: torch.Tensor | None = None
        self._feature_ids: torch.Tensor | None = None
        self._feature_offsets: torch.Tensor | None = None
        self._feature_values: torch.Tensor | None = None
        self._b_multiplier: float = 2.0
        self._device: torch.device | None = None

    def set_bloom_index(
        self, bloom_index: torch.Tensor
    ) -> "BloomIndexSearchModuleBuilder":
        self._bloom_index = bloom_index
        return self

    def set_bloom_bundle_b_offsets(
        self, bloom_bundle_b_offsets: torch.Tensor
    ) -> "BloomIndexSearchModuleBuilder":
        self._bloom_bundle_b_offsets = bloom_bundle_b_offsets
        return self

    def set_feature_data(
        self,
        feature_ids: torch.Tensor,
        feature_offsets: torch.Tensor,
        feature_values: torch.Tensor,
    ) -> "BloomIndexSearchModuleBuilder":
        self._feature_ids = feature_ids
        self._feature_offsets = feature_offsets
        self._feature_values = feature_values
        return self

    def set_b_multiplier(self, b_multiplier: float) -> "BloomIndexSearchModuleBuilder":
        self._b_multiplier = b_multiplier
        return self

    def set_device(self, device: torch.device) -> "BloomIndexSearchModuleBuilder":
        self._device = device
        return self

    def build(self) -> BloomIndexSearchModule:
        if self._feature_ids is not None:
            bloom_index, bloom_bundle_b_offsets = torch.ops.st.bloom_index_build(
                self._feature_ids,
                self._feature_offsets,
                self._feature_values,
                self._b_multiplier,
                self._hash_k,
            )
        elif self._bloom_index is not None:
            bloom_index = self._bloom_index
            if self._bloom_bundle_b_offsets is None:
                raise ValueError(
                    "bloom_bundle_b_offsets must be set when using a pre-built bloom_index"
                )
            bloom_bundle_b_offsets = self._bloom_bundle_b_offsets
        else:
            raise ValueError(
                "Either feature data (via set_feature_data) or bloom_index "
                "(via set_bloom_index) must be set before building"
            )

        module = BloomIndexSearchModule(k=self._k, hash_k=self._hash_k)
        module.set_bloom_index(bloom_index)
        module.set_bloom_bundle_b_offsets(bloom_bundle_b_offsets)

        if self._device is not None:
            module = module.to(self._device)

        return module
