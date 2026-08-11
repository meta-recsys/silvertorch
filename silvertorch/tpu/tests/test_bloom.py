# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

# pyre-unsafe

import unittest

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:
    jax = None
    jnp = None


@unittest.skipIf(jax is None, "JAX is not installed")
class TestTpuBloom(unittest.TestCase):
    def _build_index(self):
        from ..bloom import build_bloom_index

        feature_ids = [1, 2]
        feature_values = [
            100,  # doc 0, language: en
            200,
            201,  # doc 0, category: music, pop
            100,
            101,  # doc 1, language: en, es
            200,  # doc 1, category: music
            102,  # doc 2, language: ja
            202,  # doc 2, category: news
            100,  # doc 3, language: en
            200,
            203,  # doc 3, category: music, rock
        ]
        feature_offsets = [0, 1, 3, 5, 6, 7, 8, 9, 11]
        return build_bloom_index(
            feature_ids,
            feature_offsets,
            feature_values,
            b_multiplier=10.0,
            k=4,
        )

    def test_search_matches_expected_mask_on_current_backend(self) -> None:
        from ..bloom import search

        index = self._build_index()
        mask = search(
            index,
            [
                "1:100",
                "1:100 AND NOT 2:201",
                "1:102 OR 2:203",
            ],
        )
        mask.block_until_ready()

        expected = [
            [True, True, False, True],
            [False, True, False, True],
            [False, False, True, True],
        ]
        self.assertEqual(jax.device_get(mask).tolist(), expected)

    def test_empty_expression_matches_all_docs(self) -> None:
        from ..bloom import search

        index = self._build_index()
        mask = search(index, [""])
        mask.block_until_ready()

        self.assertEqual(jax.device_get(mask).tolist(), [[True, True, True, True]])

    def test_empty_query_batch_has_expected_shape(self) -> None:
        from ..bloom import search

        index = self._build_index()
        mask = search(index, [])

        self.assertEqual(mask.shape, (0, 4))
        self.assertEqual(mask.dtype, jnp.bool_)

    def test_invalid_offsets_raise(self) -> None:
        from ..bloom import build_bloom_index

        with self.assertRaises(ValueError):
            build_bloom_index(
                feature_ids=[1, 2],
                feature_offsets=[0, 1, 2],
                feature_values=[100],
            )


if __name__ == "__main__":
    unittest.main()
