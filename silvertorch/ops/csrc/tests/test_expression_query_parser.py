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


def _parse(
    expression: str, hash_k: int = 7, max_sub_queries: int = 5
) -> tuple[int, list[torch.Tensor]]:
    silvertorch_ks = torch.ones(1, dtype=torch.long)
    return torch.ops.st.parse_expression_query_batch(
        [expression], silvertorch_ks, hash_k, True, max_sub_queries
    )


def _parse_batch(
    expressions: list[str], hash_k: int = 7
) -> tuple[int, list[torch.Tensor]]:
    silvertorch_ks = torch.ones(1, dtype=torch.long)
    return torch.ops.st.parse_expression_query_batch(
        expressions, silvertorch_ks, hash_k, True, 5
    )


class TestExpressionQueryParser(unittest.TestCase):
    # --- Single term ---

    def test_single_term(self) -> None:
        max_stack, tensors = _parse("42:100")
        self.assertEqual(max_stack, 1)
        self.assertEqual(len(tensors), 2)
        self.assertGreater(tensors[0].numel(), 0)

    # --- Operator tests ---

    def test_and(self) -> None:
        max_stack, tensors = _parse("1:10 AND 2:20")
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_triple_and(self) -> None:
        max_stack, tensors = _parse("1:10 & 2:20 & 3:30")
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_or(self) -> None:
        max_stack, tensors = _parse("1:10 OR 2:20")
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_not(self) -> None:
        max_stack, tensors = _parse("NOT 1:10")
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_nested(self) -> None:
        max_stack, tensors = _parse("(1:100 AND 2:200) OR NOT 3:300")
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_complex(self) -> None:
        max_stack, tensors = _parse("(1:100 & 2:200) | !(3:300 & 4:400)")
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_double_not(self) -> None:
        max_stack, tensors = _parse("NOT NOT 1:10")
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_negative_values(self) -> None:
        max_stack, tensors = _parse("-1:-200")
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_empty_expression(self) -> None:
        max_stack, tensors = _parse("")
        self.assertEqual(len(tensors), 2)

    # --- Precedence tests ---

    def test_precedence_not_over_and(self) -> None:
        """NOT 1:10 AND 2:20 should parse as (NOT 1:10) AND 2:20."""
        _, t1 = _parse("NOT 1:10 AND 2:20")
        _, t2 = _parse("(NOT 1:10) AND 2:20")
        self.assertTrue(t1[0].equal(t2[0]))
        self.assertTrue(t1[1].equal(t2[1]))

    def test_precedence_and_over_or(self) -> None:
        """1:10 OR 2:20 AND 3:30 should parse as 1:10 OR (2:20 AND 3:30)."""
        _, t1 = _parse("1:10 OR 2:20 AND 3:30")
        _, t2 = _parse("1:10 OR (2:20 AND 3:30)")
        self.assertTrue(t1[0].equal(t2[0]))
        self.assertTrue(t1[1].equal(t2[1]))

    def test_parentheses_override_precedence(self) -> None:
        """(1:10 OR 2:20) AND 3:30 differs from default precedence."""
        _, t_parens = _parse("(1:10 OR 2:20) AND 3:30")
        _, t_default = _parse("1:10 OR 2:20 AND 3:30")
        # Should differ because parens change grouping
        self.assertFalse(t_parens[0].equal(t_default[0]))

    # --- Batch test ---

    def test_batch_parsing(self) -> None:
        max_stack, tensors = _parse_batch(["1:100", "1:10 AND 2:20", "NOT 3:30"])
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)
        self.assertGreater(tensors[0].numel(), 0)
        self.assertGreater(tensors[1].numel(), 0)

    # --- Error handling ---

    def test_error_missing_value(self) -> None:
        with self.assertRaises(RuntimeError):
            _parse("42")

    def test_error_missing_close_paren(self) -> None:
        with self.assertRaises(RuntimeError):
            _parse("(1:10 AND 2:20")

    def test_error_unknown_keyword(self) -> None:
        with self.assertRaises(RuntimeError):
            _parse("1:10 XOR 2:20")

    def test_error_trailing_operator(self) -> None:
        with self.assertRaises(RuntimeError):
            _parse("1:10 AND")

    def test_error_trailing_token(self) -> None:
        with self.assertRaises(RuntimeError):
            _parse("1:10 2:20")

    # --- Different k values ---

    def test_different_k_values(self) -> None:
        for k in [1, 3, 5, 10]:
            max_stack, tensors = _parse("1:100 AND 2:200", hash_k=k)
            self.assertGreater(max_stack, 0, f"Failed for k={k}")
            self.assertEqual(len(tensors), 2, f"Failed for k={k}")

    # --- Batching tests ---

    def test_and_batching_overflow(self) -> None:
        max_stack, tensors = _parse(
            "1:10 & 2:20 & 3:30 & 4:40 & 5:50", max_sub_queries=3
        )
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_or_batching_overflow(self) -> None:
        max_stack, tensors = _parse(
            "1:10 | 2:20 | 3:30 | 4:40 | 5:50", max_sub_queries=2
        )
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_nested_batching(self) -> None:
        max_stack, tensors = _parse(
            "(1:10 | 2:20 | 3:30 | 4:40) & 5:50", max_sub_queries=2
        )
        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    # --- Roundtrip consistency ---

    def test_roundtrip_consistency(self) -> None:
        _, t1 = _parse("(1:100 AND 2:200) OR NOT 3:300")
        _, t2 = _parse("(1:100 AND 2:200) OR NOT 3:300")
        self.assertTrue(t1[0].equal(t2[0]))
        self.assertTrue(t1[1].equal(t2[1]))


if __name__ == "__main__":
    unittest.main()
