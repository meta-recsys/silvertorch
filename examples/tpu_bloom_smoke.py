#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Run a small SilverTorch-style Bloom filter workload on JAX/TPU."""

from __future__ import annotations

import jax
import jax.numpy as jnp
# @oss-disable[end= ]: from silvertorch.silvertorch.silvertorch.tpu import bloom

from silvertorch.tpu import bloom # @oss-enable


def main() -> None:
    print(f"backend: {jax.default_backend()}", flush=True)
    print(f"devices: {jax.local_devices()}", flush=True)
    if jax.default_backend() != "tpu":
        raise RuntimeError("Expected to run on TPU; JAX did not see TPU devices")

    # Same corpus shape as README example.
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

    index = bloom.build_bloom_index(
        feature_ids,
        feature_offsets,
        feature_values,
        b_multiplier=10.0,
        k=4,
        device=jax.local_devices()[0],
    )

    queries = [
        "1:100",
        "1:100 AND NOT 2:201",
        "1:102 OR 2:203",
    ]
    mask = bloom.search(index, queries)
    mask.block_until_ready()

    print(f"signature shape: {index.signatures.shape}", flush=True)
    print(f"result shape: {mask.shape}", flush=True)
    for query, row in zip(queries, mask):
        hits = jnp.nonzero(row, size=row.shape[0], fill_value=-1)[0]
        hits = [int(x) for x in hits.tolist() if int(x) >= 0]
        print(f"{query!r:30} -> docs {hits}", flush=True)

    expected = jnp.asarray(
        [
            [True, True, False, True],
            [False, True, False, True],
            [False, False, True, True],
        ],
        dtype=jnp.bool_,
    )
    if not bool(jnp.all(mask == expected)):
        raise AssertionError(f"Unexpected mask:\n{mask}\nexpected:\n{expected}")
    print("SUCCESS: JAX/TPU SilverTorch Bloom smoke test passed", flush=True)


if __name__ == "__main__":
    main()
