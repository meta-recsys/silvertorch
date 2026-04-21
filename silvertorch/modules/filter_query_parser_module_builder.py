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

from silvertorch.modules.filter_query_parser_module import FilterQueryParserModule


class FilterQueryParserModuleBuilder:
    """Builder for constructing FilterQueryParserModule instances.

    Example::

        module = FilterQueryParserModuleBuilder(hash_k=7).build()

    Args:
        hash_k: Number of pre-calculated one-bit hashes.
        max_sub_queries: Maximum number of sub-queries per search.
    """

    def __init__(
        self,
        hash_k: int,
        max_sub_queries: int = 5,
    ) -> None:
        self._hash_k = hash_k
        self._max_sub_queries = max_sub_queries

    def build(self) -> FilterQueryParserModule:
        module = FilterQueryParserModule(
            hash_k=self._hash_k,
            max_sub_queries=self._max_sub_queries,
        )
        module.eval()
        return module
