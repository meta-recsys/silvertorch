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

"""JAX/XLA Bloom candidate filter prototype for TPU.

This is not a binding to the existing packed C++/CUDA `torch.ops.st.*` Bloom
index. The production SilverTorch layout stores per-bundle packed uint64 masks
and relies on custom kernels. That layout is not directly usable on TPU today.

This module uses an XLA-friendly fixed-width boolean signature matrix:

    signatures[doc_id, bit_position] -> bool

The expression language mirrors the public SilverTorch examples
(`feature_id:value`, `AND`, `OR`, `NOT`, parentheses). Building is host-side
Python for now; searching is JAX and runs on whatever JAX backend owns the
signature array, including TPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast, Sequence

import jax
import jax.numpy as jnp
import numpy as np


_MASK32 = 0xFFFFFFFF
_MIX_A = 0x9E3779B1
_MIX_B = 0x85EBCA6B
_MIX_C = 0xC2B2AE35


@dataclass(frozen=True)
class JaxBloomIndex:
    """TPU-native Bloom index representation.

    Attributes:
        signatures: Boolean JAX array with shape `[num_docs, bloom_size]`.
        bloom_size: Number of Bloom positions in each document signature.
        k: Number of hash probes used to index and query each term.
        num_docs: Number of real documents in `signatures`.
    """

    signatures: jax.Array
    bloom_size: int
    k: int
    num_docs: int


@dataclass(frozen=True)
class _Term:
    feature_id: int
    feature_value: int


@dataclass(frozen=True)
class _Not:
    child: object


@dataclass(frozen=True)
class _And:
    children: tuple[object, ...]


@dataclass(frozen=True)
class _Or:
    children: tuple[object, ...]


@dataclass(frozen=True)
class _Empty:
    pass


@dataclass(frozen=True)
class _PreparedFeatures:
    ids: list[int]
    offsets: list[int]
    values: list[int]
    num_docs: int
    num_features: int


def _as_int_list(values: Sequence[int] | np.ndarray | jax.Array) -> list[int]:
    if not isinstance(values, Sequence):
        values = cast(Any, values).tolist()
    return [int(v) for v in values]


def _mix32_py(x: int) -> int:
    x &= _MASK32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & _MASK32
    x ^= x >> 15
    x = (x * 0x846CA68B) & _MASK32
    x ^= x >> 16
    return x & _MASK32


def _hash_one_py(feature_id: int, feature_value: int, seed: int) -> int:
    x = (
        ((feature_id & _MASK32) * _MIX_A)
        ^ ((feature_value & _MASK32) * _MIX_B)
        ^ ((seed & _MASK32) * _MIX_C)
    )
    return _mix32_py(x)


def _hash_positions_py(
    feature_id: int,
    feature_value: int,
    k: int,
    bloom_size: int,
) -> list[int]:
    return [
        _hash_one_py(feature_id, feature_value, seed) % bloom_size for seed in range(k)
    ]


def _mix32_jax(x: jax.Array) -> jax.Array:
    x = x.astype(jnp.uint32)
    x = x ^ (x >> jnp.uint32(16))
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> jnp.uint32(15))
    x = x * jnp.uint32(0x846CA68B)
    x = x ^ (x >> jnp.uint32(16))
    return x


def _hash_positions_jax(
    feature_id: int,
    feature_value: int,
    k: int,
    bloom_size: int,
) -> jax.Array:
    seeds = jnp.arange(k, dtype=jnp.uint32)
    x = (
        jnp.uint32(feature_id & _MASK32) * jnp.uint32(_MIX_A)
        ^ jnp.uint32(feature_value & _MASK32) * jnp.uint32(_MIX_B)
        ^ seeds * jnp.uint32(_MIX_C)
    )
    return (jnp.mod(_mix32_jax(x), jnp.uint32(bloom_size))).astype(jnp.int32)


def _prepare_features(
    feature_ids: Sequence[int] | np.ndarray | jax.Array,
    feature_offsets: Sequence[int] | np.ndarray | jax.Array,
    feature_values: Sequence[int] | np.ndarray | jax.Array,
) -> _PreparedFeatures:
    ids = _as_int_list(feature_ids)
    offsets = _as_int_list(feature_offsets)
    values = _as_int_list(feature_values)

    if not ids:
        raise ValueError("feature_ids must not be empty")
    if (len(offsets) - 1) % len(ids) != 0:
        raise ValueError("feature_offsets length must be num_docs * num_features + 1")
    if offsets[-1] != len(values):
        raise ValueError("feature_offsets[-1] must equal len(feature_values)")

    num_features = len(ids)
    num_docs = (len(offsets) - 1) // num_features
    return _PreparedFeatures(ids, offsets, values, num_docs, num_features)


def _max_terms_per_doc(features: _PreparedFeatures) -> int:
    max_terms = 1
    for doc_id in range(features.num_docs):
        terms = 0
        for feature_col in range(features.num_features):
            offset_idx = doc_id * features.num_features + feature_col
            terms += features.offsets[offset_idx + 1] - features.offsets[offset_idx]
        max_terms = max(max_terms, terms)
    return max_terms


def _build_signatures(
    features: _PreparedFeatures,
    bloom_size: int,
    k: int,
) -> np.ndarray:
    signatures = np.zeros((features.num_docs, bloom_size), dtype=np.bool_)

    for doc_id in range(features.num_docs):
        for feature_col, feature_id in enumerate(features.ids):
            offset_idx = doc_id * features.num_features + feature_col
            start = features.offsets[offset_idx]
            end = features.offsets[offset_idx + 1]
            for value_idx in range(start, end):
                positions = _hash_positions_py(
                    feature_id,
                    features.values[value_idx],
                    k,
                    bloom_size,
                )
                for pos in positions:
                    signatures[doc_id, pos] = True

    return signatures


def build_bloom_index(
    feature_ids: Sequence[int] | np.ndarray | jax.Array,
    feature_offsets: Sequence[int] | np.ndarray | jax.Array,
    feature_values: Sequence[int] | np.ndarray | jax.Array,
    b_multiplier: float = 5.0,
    k: int = 3,
    device: Any = None,
) -> JaxBloomIndex:
    """Build a TPU-native Bloom signature matrix from SilverTorch feature data.

    Args:
        feature_ids: Feature schema, shape `[num_features]`.
        feature_offsets: Jagged offsets, shape `[num_docs * num_features + 1]`.
        feature_values: Flattened feature values.
        b_multiplier: Controls Bloom width. Higher means fewer false positives.
        k: Number of hash probes for each term.
        device: Optional JAX device to place the signature matrix on.

    Returns:
        JaxBloomIndex suitable for `search`.
    """
    if b_multiplier <= 1.0:
        raise ValueError("b_multiplier must be greater than 1.0")
    if k <= 0:
        raise ValueError("k must be positive")

    features = _prepare_features(feature_ids, feature_offsets, feature_values)
    max_terms_per_doc = _max_terms_per_doc(features)
    bloom_size = max(1, int(math.ceil(max_terms_per_doc * k * b_multiplier)))

    signatures = _build_signatures(features, bloom_size, k)
    jax_signatures = jnp.asarray(signatures)
    if device is not None:
        jax_signatures = jax.device_put(jax_signatures, device)
    return JaxBloomIndex(jax_signatures, bloom_size, k, features.num_docs)


class _Tokenizer:
    def __init__(self, expression: str) -> None:
        spaced = (
            expression.replace("(", " ( ")
            .replace(")", " ) ")
            .replace("&", " AND ")
            .replace("|", " OR ")
            .replace("!", " NOT ")
        )
        self._tokens = spaced.split()
        self._pos = 0

    def peek(self) -> str | None:
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def next(self) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return token


class _Parser:
    def __init__(self, expression: str) -> None:
        self._tokens = _Tokenizer(expression)

    def parse(self) -> object:
        if self._tokens.peek() is None:
            return _Empty()
        node = self._parse_or()
        if self._tokens.peek() is not None:
            raise ValueError(f"Unexpected token {self._tokens.peek()!r}")
        return node

    def _parse_or(self) -> object:
        children = [self._parse_and()]
        while self._tokens.peek() == "OR":
            self._tokens.next()
            children.append(self._parse_and())
        if len(children) == 1:
            return children[0]
        return _Or(tuple(children))

    def _parse_and(self) -> object:
        children = [self._parse_unary()]
        while self._tokens.peek() == "AND":
            self._tokens.next()
            children.append(self._parse_unary())
        if len(children) == 1:
            return children[0]
        return _And(tuple(children))

    def _parse_unary(self) -> object:
        if self._tokens.peek() == "NOT":
            self._tokens.next()
            return _Not(self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> object:
        if self._tokens.peek() == "(":
            self._tokens.next()
            node = self._parse_or()
            if self._tokens.next() != ")":
                raise ValueError("Expected ')'")
            return node
        return self._parse_term()

    def _parse_term(self) -> object:
        token = self._tokens.next()
        parts = token.split(":")
        if len(parts) < 2:
            raise ValueError(f"Expected feature_id:feature_value, got {token!r}")
        return _Term(int(parts[0]), int(parts[1]))


def _parse_expression(expression: str) -> object:
    return _Parser(expression.strip()).parse()


def _eval(node: object, index: JaxBloomIndex) -> jax.Array:
    if isinstance(node, _Empty):
        return jnp.ones((index.num_docs,), dtype=jnp.bool_)
    if isinstance(node, _Term):
        positions = _hash_positions_jax(
            node.feature_id,
            node.feature_value,
            index.k,
            index.bloom_size,
        )
        return jnp.all(index.signatures[:, positions], axis=1)
    if isinstance(node, _Not):
        return jnp.logical_not(_eval(node.child, index))
    if isinstance(node, _And):
        result = jnp.ones((index.num_docs,), dtype=jnp.bool_)
        for child in node.children:
            result = jnp.logical_and(result, _eval(child, index))
        return result
    if isinstance(node, _Or):
        result = jnp.zeros((index.num_docs,), dtype=jnp.bool_)
        for child in node.children:
            result = jnp.logical_or(result, _eval(child, index))
        return result
    raise TypeError(f"Unsupported expression node: {type(node).__name__}")


def search(index: JaxBloomIndex, expressions: Sequence[str]) -> jax.Array:
    """Evaluate expression strings against a JaxBloomIndex.

    Returns a boolean JAX array with shape `[num_queries, num_docs]`.
    """
    rows = [_eval(_parse_expression(expr), index) for expr in expressions]
    if not rows:
        return jnp.zeros((0, index.num_docs), dtype=jnp.bool_)
    return jnp.stack(rows, axis=0)
