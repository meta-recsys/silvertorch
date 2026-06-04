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

"""Fetch MovieLens 32M and run the SilverTorch smoke-test benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

FBCODE_ROOT = Path(__file__).resolve().parents[4]
if __package__ in (None, ""):
    if str(FBCODE_ROOT) not in sys.path:
        sys.path.insert(0, str(FBCODE_ROOT))
    __package__ = "silvertorch.silvertorch.benchmark.movielens"

from .movielens_small import main as run_movielens_benchmark


def main(argv: list[str] | None = None) -> None:
    run_movielens_benchmark(argv, default_preset="32m")


if __name__ == "__main__":
    main(sys.argv[1:])
