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


def _make_features(
    num_docs: int,
    features_per_doc: list[list[tuple[int, int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build feature tensors from per-document feature lists.

    Args:
        num_docs: number of documents
        features_per_doc: list of [(feature_id, feature_value), ...] per doc.
            All documents must reference the same set of feature_ids.

    Returns:
        feature_ids, feature_offsets, feature_values
    """
    all_feat_ids = sorted({fid for doc in features_per_doc for fid, _ in doc})
    feature_ids = torch.tensor(all_feat_ids, dtype=torch.int32)
    num_features = len(all_feat_ids)

    # Build per-(doc, feature_id) value lists
    doc_feat_values: list[list[list[int]]] = []
    for doc_feats in features_per_doc:
        feat_map: dict[int, list[int]] = {fid: [] for fid in all_feat_ids}
        for fid, fval in doc_feats:
            feat_map[fid].append(fval)
        doc_feat_values.append([sorted(feat_map[fid]) for fid in all_feat_ids])

    values = []
    offsets = [0]
    for doc_idx in range(num_docs):
        for feat_idx in range(num_features):
            vals = doc_feat_values[doc_idx][feat_idx]
            values.extend(vals)
            offsets.append(len(values))

    feature_values = torch.tensor(values, dtype=torch.long)
    feature_offsets = torch.tensor(offsets, dtype=torch.long)
    return feature_ids, feature_offsets, feature_values


def _build_and_search(
    features_per_doc: list[list[tuple[int, int]]],
    expressions: list[str],
    k: int = 3,
    hash_k: int = 7,
    b_multiplier: float = 2.0,
    return_bool_mask: bool = True,
) -> torch.Tensor:
    """End-to-end: build bloom index → parse expressions → search."""
    num_docs = len(features_per_doc)
    feature_ids, feature_offsets, feature_values = _make_features(
        num_docs, features_per_doc
    )

    bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
        feature_ids, feature_offsets, feature_values, b_multiplier, hash_k
    )

    silvertorch_ks = torch.ones(1, dtype=torch.long)
    _, tensors = torch.ops.st.parse_expression_query_batch(
        expressions, silvertorch_ks, hash_k, True, 5
    )

    return torch.ops.st.bloom_index_search_batch(
        bloom_index,
        bundle_b_offsets,
        tensors[0],
        tensors[1],
        k,
        hash_k,
        return_bool_mask,
    )


class TestBloomSearchIntegration(unittest.TestCase):
    """End-to-end integration tests: bloom_indexer → expression_parser → search."""

    def test_single_feature_match(self) -> None:
        # Doc 0 has feature (1, 100)
        result = _build_and_search(
            features_per_doc=[[(1, 100)]],
            expressions=["1:100"],
        )
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result.size(0), 1)  # 1 query
        self.assertTrue(result[0, 0].item())  # doc 0 should match

    def test_no_match(self) -> None:
        # Doc 0 has feature (1, 100), query for (2, 200) which doesn't exist
        result = _build_and_search(
            features_per_doc=[[(1, 100)]],
            expressions=["2:200"],
            return_bool_mask=False,
        )
        self.assertEqual(result.dtype, torch.long)
        self.assertEqual(result.size(0), 1)

    def test_and_query(self) -> None:
        # Doc 0 has both features
        result = _build_and_search(
            features_per_doc=[[(1, 100), (2, 200)]],
            expressions=["1:100 AND 2:200"],
        )
        self.assertTrue(result[0, 0].item())

    def test_or_query(self) -> None:
        # Doc 0 has only feature (1, 100), OR should still match
        result = _build_and_search(
            features_per_doc=[[(1, 100)]],
            expressions=["1:100 OR 2:200"],
        )
        self.assertTrue(result[0, 0].item())

    def test_not_query(self) -> None:
        # Doc 0 has feature (1, 100), NOT 2:200 should match (2:200 not present)
        result = _build_and_search(
            features_per_doc=[[(1, 100)]],
            expressions=["NOT 2:200"],
        )
        self.assertTrue(result[0, 0].item())

    def test_multi_doc(self) -> None:
        # Multiple documents with different features
        result = _build_and_search(
            features_per_doc=[
                [(1, 100), (2, 200)],  # doc 0: has both
                [(1, 100)],  # doc 1: has only (1, 100)
                [(2, 200)],  # doc 2: has only (2, 200)
            ],
            expressions=["1:100 AND 2:200"],
        )
        self.assertEqual(result.size(0), 1)  # 1 query

    def test_batch_queries(self) -> None:
        result = _build_and_search(
            features_per_doc=[[(1, 100), (2, 200)]],
            expressions=["1:100", "2:200", "1:100 AND 2:200"],
        )
        self.assertEqual(result.size(0), 3)

    def test_output_bool_mask(self) -> None:
        result = _build_and_search(
            features_per_doc=[[(1, 100)]],
            expressions=["1:100"],
            return_bool_mask=True,
        )
        self.assertEqual(result.dtype, torch.bool)

    def test_output_bitmask(self) -> None:
        result = _build_and_search(
            features_per_doc=[[(1, 100)]],
            expressions=["1:100"],
            return_bool_mask=False,
        )
        self.assertEqual(result.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
