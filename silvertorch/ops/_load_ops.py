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

"""Loads silvertorch C++ ops that register torch.ops.st.* operators."""

import importlib

import torch  # noqa: F401

# @oss-disable[end= ]: _BUCK_TARGETS = [
    # @oss-disable[end= ]: "//silvertorch/oss/ops/csrc:bloom_indexer",
    # @oss-disable[end= ]: "//silvertorch/oss/ops/csrc:bloom_index_search",
    # @oss-disable[end= ]: "//silvertorch/oss/ops/csrc:expression_query_parser",
# @oss-disable[end= ]: ]

try:
    # silvertorch._C is the OSS pip-installed extension. In internal BUCK
    # builds this raises ImportError and we fall through to load_library.
    # Using importlib (instead of `import silvertorch._C`) hides the import
    # from autodeps so it doesn't try to find an owner for the OSS-only target.
    importlib.import_module("silvertorch._C")
except ImportError as _e:
    # @oss-disable[end= ]: for _target in _BUCK_TARGETS:
        # @oss-disable[end= ]: torch.ops.load_library(_target)
    if not hasattr(torch.ops, "st") or not hasattr(torch.ops.st, "bloom_index_build"):
        raise ImportError(
            "silvertorch C++ ops not found. "
            "Please install silvertorch: pip install silvertorch"
        ) from _e
