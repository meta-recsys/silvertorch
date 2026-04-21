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

"""Tests that the OSS package structure would work after ShipIt transformation.

Verifies that all modules that OSS users need can be imported under the
silvertorch.oss.* namespace (which ShipIt rewrites to silvertorch.*).
Also verifies that ops are registered and callable after loading.
"""

import os
import unittest

import silvertorch.ops._load_ops  # noqa: F401
import torch


class TestOssModuleImports(unittest.TestCase):
    """Verify all public modules can be imported."""

    def test_import_bloom_index_search_module(self) -> None:
        from silvertorch.modules.bloom_index_search_module import (
            BloomIndexSearchModule,
        )

        self.assertTrue(issubclass(BloomIndexSearchModule, torch.nn.Module))

    def test_import_bloom_index_search_module_builder(self) -> None:
        from silvertorch.modules.bloom_index_search_module_builder import (
            BloomIndexSearchModuleBuilder,
        )

        self.assertTrue(callable(BloomIndexSearchModuleBuilder))

    def test_import_filter_query_parser_module(self) -> None:
        from silvertorch.modules.filter_query_parser_module import (
            FilterQueryParserModule,
        )

        self.assertTrue(issubclass(FilterQueryParserModule, torch.nn.Module))

    def test_import_filter_query_parser_module_builder(self) -> None:
        from silvertorch.modules.filter_query_parser_module_builder import (
            FilterQueryParserModuleBuilder,
        )

        self.assertTrue(callable(FilterQueryParserModuleBuilder))

    def test_modules_init_exports(self) -> None:
        from silvertorch.modules import (
            BloomIndexSearchModule,
            BloomIndexSearchModuleBuilder,
            FilterQueryParserModule,
            FilterQueryParserModuleBuilder,
        )

        self.assertIsNotNone(BloomIndexSearchModule)
        self.assertIsNotNone(BloomIndexSearchModuleBuilder)
        self.assertIsNotNone(FilterQueryParserModule)
        self.assertIsNotNone(FilterQueryParserModuleBuilder)


class TestOssOpsRegistered(unittest.TestCase):
    """Verify all torch.ops.st.* ops are registered after _load_ops."""

    def test_bloom_index_build_registered(self) -> None:
        self.assertTrue(hasattr(torch.ops.st, "bloom_index_build"))

    def test_bloom_index_search_batch_registered(self) -> None:
        self.assertTrue(hasattr(torch.ops.st, "bloom_index_search_batch"))

    def test_parse_expression_query_batch_registered(self) -> None:
        self.assertTrue(hasattr(torch.ops.st, "parse_expression_query_batch"))

    def test_generate_column_info_registered(self) -> None:
        self.assertTrue(hasattr(torch.ops.st, "generate_column_info_for_clusters"))


class TestSetupPyStructure(unittest.TestCase):
    """Verify setup.py has correct package configuration."""

    def test_setup_py_has_package_dir(self) -> None:
        """setup.py must have package_dir mapping for OSS install.

        In the internal repo, setup.py lives next to the silvertorch.oss package.
        After ShipIt sync, setup.py lives at the OSS repo root (one level above
        the silvertorch package). Try the internal layout first; fall back to
        looking up two levels from this test file.
        """
        try:
            import silvertorch.oss

            search_dir = os.path.dirname(silvertorch.oss.__file__)
        except ImportError:
            # OSS layout: setup.py is two levels up from this test file
            # (silvertorch/ops/csrc/tests/ -> silvertorch/ -> repo root)
            search_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            )
        setup_path = os.path.join(search_dir, "setup.py")
        if not os.path.isfile(setup_path):
            self.skipTest(f"setup.py not found at {setup_path}")

        with open(setup_path) as f:
            content = f.read()
        self.assertIn("package_dir", content)
        self.assertIn('"silvertorch.ops"', content)
        self.assertIn('"silvertorch.modules"', content)
