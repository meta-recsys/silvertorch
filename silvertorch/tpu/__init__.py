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

"""Experimental JAX/TPU implementations.

These modules are intentionally separate from the PyTorch custom operators.
The existing `torch.ops.st.*` extension has CPU/CUDA kernels and does not
lower to TPU/XLA.
"""

try:
    from .bloom import build_bloom_index, JaxBloomIndex, search
except ModuleNotFoundError as error:
    if error.name != "jax":
        raise

    missing_jax_error = error

    def _missing_jax(*args, **kwargs):
        raise ModuleNotFoundError(
            "silvertorch.tpu requires JAX. Install it with "
            "`python -m pip install 'jax[cpu]'` for CPU testing or use the "
            "JAX TPU runtime image for TPU execution."
        ) from missing_jax_error

    JaxBloomIndex = None
    build_bloom_index = _missing_jax
    search = _missing_jax

__all__ = ["JaxBloomIndex", "build_bloom_index", "search"]
