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
from silvertorch.modules.filter_query_parser_module import FilterQueryParserModule
from silvertorch.modules.filter_query_parser_module_builder import (
    FilterQueryParserModuleBuilder,
)


class TestFilterQueryParserModule(unittest.TestCase):
    def test_forward_single(self) -> None:
        module = FilterQueryParserModule(hash_k=7)
        module.eval()

        silvertorch_ks = torch.ones(1, dtype=torch.long)
        max_stack, tensors = module(["1:100"], silvertorch_ks)

        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)
        self.assertGreater(tensors[0].numel(), 0)  # plans_data
        self.assertGreater(tensors[1].numel(), 0)  # plans_offsets

    def test_forward_batch(self) -> None:
        module = FilterQueryParserModule(hash_k=7)
        module.eval()

        silvertorch_ks = torch.ones(1, dtype=torch.long)
        max_stack, tensors = module(
            ["1:100", "1:10 AND 2:20", "NOT 3:30"], silvertorch_ks
        )

        self.assertGreater(max_stack, 0)
        self.assertEqual(len(tensors), 2)

    def test_forward_and_query(self) -> None:
        module = FilterQueryParserModule(hash_k=7)
        module.eval()

        silvertorch_ks = torch.ones(1, dtype=torch.long)
        max_stack, tensors = module(["1:100 AND 2:200"], silvertorch_ks)

        self.assertGreater(max_stack, 0)

    def test_forward_empty(self) -> None:
        module = FilterQueryParserModule(hash_k=7)
        module.eval()

        silvertorch_ks = torch.ones(1, dtype=torch.long)
        max_stack, tensors = module([""], silvertorch_ks)

        self.assertEqual(len(tensors), 2)


class TestFilterQueryParserModuleBuilder(unittest.TestCase):
    def test_build(self) -> None:
        module = FilterQueryParserModuleBuilder(hash_k=7).build()
        self.assertIsInstance(module, FilterQueryParserModule)
        self.assertFalse(module.training)

    def test_build_custom_params(self) -> None:
        module = FilterQueryParserModuleBuilder(hash_k=5, max_sub_queries=10).build()
        self.assertEqual(module.hash_k, 5)
        self.assertEqual(module.max_sub_queries, 10)


if __name__ == "__main__":
    unittest.main()
