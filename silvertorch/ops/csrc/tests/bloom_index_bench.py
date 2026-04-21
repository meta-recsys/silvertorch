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

import argparse
import random

import silvertorch.ops._load_ops  # noqa: F401
import torch
import torch.utils.benchmark as benchmark

_SEED = 12345
_ITERATIONS = 100


def _generate_features(
    num_documents: int,
    num_features: int,
    num_values: int,
    num_feature_values: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random feature data for bloom index building."""
    torch.manual_seed(_SEED)
    feature_ids = torch.arange(1, num_features + 1, dtype=torch.int32)

    feature_offsets = torch.full(
        (num_documents * num_features + 1,),
        num_feature_values,
        dtype=torch.int64,
    )
    feature_offsets[0] = 0
    feature_offsets = torch.cumsum(feature_offsets, dim=0)

    feature_values = torch.rand((num_documents * num_features, num_values))
    feature_values = torch.argsort(feature_values, dim=1)[:, :num_feature_values] + 1
    feature_values = feature_values.sort(dim=1)[0].reshape(-1)

    return feature_ids, feature_offsets, feature_values


def _generate_expression_queries(
    num_features: int,
    num_values: int,
    num_feature_values: int,
    num_or: int = 2,
) -> list[str]:
    """Generate random expression queries."""
    random.seed(_SEED)
    and_terms = []
    count = num_or
    while count > 0:
        for feature in range(1, num_features + 1):
            values = list(range(1, num_values + 1))
            random.shuffle(values)
            values = values[:num_feature_values]
            or_expr = " OR ".join(f"{feature}:{v}" for v in values)
            and_terms.append(f"({or_expr})")
            count -= 1
            if count <= 0:
                break
    return [" AND ".join(and_terms)]


def _run_build_benchmark(
    feature_ids: torch.Tensor,
    feature_offsets: torch.Tensor,
    feature_values: torch.Tensor,
    b_multiplier: float,
    k: int,
    num_threads: int,
) -> float:
    """Benchmark bloom index build."""
    bench = benchmark.Timer(
        stmt="torch.ops.st.bloom_index_build("
        "feature_ids, feature_offsets, feature_values, b_multiplier, k)",
        setup=f"""
import os
os.environ["OMP_NUM_THREADS"] = str({num_threads})
torch.set_num_threads({num_threads})
""",
        globals={
            "torch": torch,
            "feature_ids": feature_ids,
            "feature_offsets": feature_offsets,
            "feature_values": feature_values,
            "b_multiplier": b_multiplier,
            "k": k,
        },
    )
    result = bench.timeit(_ITERATIONS)
    return result.mean


def _run_search_benchmark(
    bloom_index: torch.Tensor,
    bundle_b_offsets: torch.Tensor,
    plans_data: torch.Tensor,
    plans_offsets: torch.Tensor,
    k: int,
    hash_k: int,
    return_bool_mask: bool,
    num_threads: int,
) -> float:
    """Benchmark bloom index search."""
    bench = benchmark.Timer(
        stmt="torch.ops.st.bloom_index_search_batch("
        "bloom_index, bundle_b_offsets, plans_data, plans_offsets, "
        "k, hash_k, return_bool_mask)",
        setup=f"""
import os
os.environ["OMP_NUM_THREADS"] = str({num_threads})
torch.set_num_threads({num_threads})
""",
        globals={
            "torch": torch,
            "bloom_index": bloom_index,
            "bundle_b_offsets": bundle_b_offsets,
            "plans_data": plans_data,
            "plans_offsets": plans_offsets,
            "k": k,
            "hash_k": hash_k,
            "return_bool_mask": return_bool_mask,
        },
    )
    result = bench.timeit(_ITERATIONS)
    return result.mean


def _run_parse_benchmark(
    expressions: list[str],
    hash_k: int,
    num_threads: int,
) -> float:
    """Benchmark expression query parsing."""
    silvertorch_ks = torch.ones(1, dtype=torch.long)
    bench = benchmark.Timer(
        stmt="torch.ops.st.parse_expression_query_batch("
        "expressions, silvertorch_ks, hash_k, True, 5)",
        setup=f"""
import os
os.environ["OMP_NUM_THREADS"] = str({num_threads})
torch.set_num_threads({num_threads})
""",
        globals={
            "torch": torch,
            "expressions": expressions,
            "silvertorch_ks": silvertorch_ks,
            "hash_k": hash_k,
        },
    )
    result = bench.timeit(_ITERATIONS)
    return result.mean


def _run_benchmark(
    num_documents: list[int],
    num_index_features: int,
    num_index_values: int,
    num_index_feature_values: int,
    num_query_features: int,
    num_query_values: int,
    num_query_feature_values: int,
    b_multiplier: float,
    bloom_index_k: int,
    batch_sizes: list[int],
    num_or: list[int],
) -> None:
    print("=" * 60)
    print("BLOOM INDEX BENCHMARK")
    print("=" * 60)

    num_threads = torch.get_num_threads()

    for num_doc in num_documents:
        print(f"\n--- {num_doc} documents ---")

        feature_ids, feature_offsets, feature_values = _generate_features(
            num_doc, num_index_features, num_index_values, num_index_feature_values
        )

        # Benchmark build
        build_time = _run_build_benchmark(
            feature_ids,
            feature_offsets,
            feature_values,
            b_multiplier,
            bloom_index_k,
            num_threads,
        )
        print(f"  Build: {build_time * 1000:.2f} ms")

        # Build index for search benchmarks
        bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
            feature_ids,
            feature_offsets,
            feature_values,
            b_multiplier,
            bloom_index_k,
        )
        print(
            f"  Index size: {bloom_index.numel()} int64s "
            f"({bloom_index.numel() * 8 / 1024 / 1024:.1f} MB)"
        )

        for num_or_val in num_or:
            expressions = _generate_expression_queries(
                num_query_features,
                num_query_values,
                num_query_feature_values,
                num_or_val,
            )

            # Benchmark parse
            parse_time = _run_parse_benchmark(
                expressions,
                bloom_index_k,
                num_threads,
            )
            print(f"  Parse (num_or={num_or_val}): {parse_time * 1000:.2f} ms")

            silvertorch_ks = torch.ones(1, dtype=torch.long)
            _, tensors = torch.ops.st.parse_expression_query_batch(
                expressions,
                silvertorch_ks,
                bloom_index_k,
                True,
                5,
            )
            plans_data, plans_offsets = tensors

            for batch_size in batch_sizes:
                # Replicate plans for batch
                batch_data = plans_data
                batch_offsets = plans_offsets
                if batch_size > 1:
                    data_list = [plans_data] * batch_size
                    offset_list = []
                    data_offset = 0
                    for i in range(batch_size):
                        shifted = plans_offsets.clone()
                        if i > 0:
                            shifted = shifted + data_offset
                        offset_list.append(shifted)
                        data_offset += plans_data.numel()
                    batch_data = torch.cat(data_list)
                    batch_offsets = torch.cat(offset_list)

                for return_bool_mask in [True, False]:
                    search_time = _run_search_benchmark(
                        bloom_index,
                        bundle_b_offsets,
                        batch_data,
                        batch_offsets,
                        bloom_index_k,
                        bloom_index_k,
                        return_bool_mask,
                        num_threads,
                    )
                    mask_str = "bool" if return_bool_mask else "bits"
                    print(
                        f"  Search (or={num_or_val}, batch={batch_size}, "
                        f"{mask_str}): {search_time * 1000:.2f} ms"
                    )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bloom index benchmark")
    parser.add_argument(
        "--num_documents",
        nargs="*",
        type=int,
        default=[100000, 1000000],
    )
    parser.add_argument("--num_index_features", type=int, default=10)
    parser.add_argument("--num_index_values", type=int, default=10)
    parser.add_argument("--num_index_feature_values", type=int, default=5)
    parser.add_argument("--num_query_features", type=int, default=10)
    parser.add_argument("--num_query_values", type=int, default=10)
    parser.add_argument("--num_query_feature_values", type=int, default=5)
    parser.add_argument("--b_multiplier", type=float, default=2.0)
    parser.add_argument("--bloom_index_k", type=int, default=5)
    parser.add_argument("--batch_size", nargs="*", type=int, default=[1, 8])
    parser.add_argument("--num_or", nargs="*", type=int, default=[2, 10])

    args = parser.parse_args()

    _run_benchmark(
        args.num_documents,
        args.num_index_features,
        args.num_index_values,
        args.num_index_feature_values,
        args.num_query_features,
        args.num_query_values,
        args.num_query_feature_values,
        args.b_multiplier,
        args.bloom_index_k,
        args.batch_size,
        args.num_or,
    )
