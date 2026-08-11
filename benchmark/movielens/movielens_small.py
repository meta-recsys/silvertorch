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

"""Benchmark SilverTorch bloom-index ops on MovieLens metadata.

This script uses the public MovieLens CSV datasets (for example
``ml-latest-small`` or ``ml-latest``) and converts movie metadata into the
feature tensor layout expected by SilverTorch:

- feature id ``1``: movie genres (multi-value)
- feature id ``2``: release year (single value when present)
- feature id ``3``: normalized user tags from ``tags.csv`` (optional)

The parser runs on CPU; build and search can run on CPU or CUDA.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast, TypeVar

import silvertorch.ops._load_ops  # noqa: F401
import torch

SEED = 12345
FBCODE_ROOT = Path(__file__).resolve().parents[4]
if __package__ in (None, ""):
    if str(FBCODE_ROOT) not in sys.path:
        sys.path.insert(0, str(FBCODE_ROOT))
    __package__ = "silvertorch.silvertorch.benchmark.movielens"

from .dataloader import (
    DATASET_URLS,
    DEFAULT_DATA_ROOT,
    FEATURE_GENRE,
    FEATURE_TAG,
    FEATURE_YEAR,
    load_corpus,
    MovieLensCorpus,
    resolve_dataset_dir,
)


@dataclass(frozen=True)
class QuerySpec:
    expression: str
    description: str


@dataclass(frozen=True)
class BenchmarkStats:
    mean_ms: float
    median_ms: float
    min_ms: float


@dataclass(frozen=True)
class BenchmarkPreset:
    dataset_name: str
    replication_factor: int
    include_tags: bool
    num_queries: int
    description: str


PRESETS = {
    "small": BenchmarkPreset(
        dataset_name="ml-latest-small",
        replication_factor=16,
        include_tags=False,
        num_queries=128,
        description="MovieLens latest small benchmark preset.",
    ),
    "32m": BenchmarkPreset(
        dataset_name="ml-32m",
        replication_factor=1,
        include_tags=False,
        num_queries=256,
        description="MovieLens 32M smoke-test and benchmark preset.",
    ),
}

MeasureResult = TypeVar("MeasureResult")


def _parse_args(
    argv: list[str] | None = None,
    *,
    default_preset: str = "small",
) -> argparse.Namespace:
    preset = PRESETS[default_preset]
    parser = argparse.ArgumentParser(
        description=preset.description,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to an extracted MovieLens CSV dataset directory.",
    )
    parser.add_argument(
        "--dataset-name",
        choices=sorted(DATASET_URLS),
        default=preset.dataset_name,
        help="Dataset to download into benchmark/data when --download is used.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for downloaded datasets.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset if it is missing.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Where to run build/search. The parser always runs on CPU.",
    )
    parser.add_argument(
        "--replication-factor",
        type=int,
        default=preset.replication_factor,
        help="Repeat the base MovieLens corpus N times to scale the benchmark.",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=None,
        help="Cap the number of base movies loaded before replication.",
    )
    parser.add_argument(
        "--include-tags",
        action="store_true",
        default=preset.include_tags,
        help="Include normalized tags from tags.csv as a third feature.",
    )
    parser.add_argument(
        "--no-include-tags",
        action="store_false",
        dest="include_tags",
        help="Disable tag loading even if the preset enables it.",
    )
    parser.add_argument(
        "--min-tag-frequency",
        type=int,
        default=2,
        help="Only keep normalized tags seen at least this many times globally.",
    )
    parser.add_argument(
        "--max-tags-per-movie",
        type=int,
        default=3,
        help="Cap the number of kept tags per movie.",
    )
    parser.add_argument("--k", type=int, default=3, help="Bloom index K.")
    parser.add_argument(
        "--hash-k",
        type=int,
        default=7,
        help="Parser/search HASH_K. Must be >= --k.",
    )
    parser.add_argument(
        "--b-multiplier",
        type=float,
        default=5.0,
        help="Bloom width multiplier passed to bloom_index_build.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=preset.num_queries,
        help="Number of benchmark queries in the search batch.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of warmup iterations per benchmark stage.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of measured iterations per benchmark stage.",
    )
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=5,
        help="How many sample queries to print with hit counts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for query generation.",
    )
    return parser.parse_args(argv)


def _choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device_arg


def _sync_for_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _measure(
    fn: Callable[[], MeasureResult],
    *,
    device: str,
    warmup_iters: int,
    iters: int,
) -> tuple[BenchmarkStats, MeasureResult]:
    result: MeasureResult | None = None
    for _ in range(warmup_iters):
        result = fn()
        _sync_for_device(device)

    timings_s = []
    for _ in range(iters):
        _sync_for_device(device)
        start = time.perf_counter()
        result = fn()
        _sync_for_device(device)
        timings_s.append(time.perf_counter() - start)

    timings_ms = [value * 1000.0 for value in timings_s]
    stats = BenchmarkStats(
        mean_ms=statistics.mean(timings_ms),
        median_ms=statistics.median(timings_ms),
        min_ms=min(timings_ms),
    )
    if result is None:
        raise RuntimeError("Benchmark did not execute any iterations")
    return stats, result


def _generate_query_specs(
    corpus: MovieLensCorpus,
    num_queries: int,
    *,
    seed: int,
) -> list[QuerySpec]:
    rng = random.Random(seed)
    docs_with_year = [doc for doc in corpus.documents if doc.year is not None]
    docs_with_tags = [doc for doc in corpus.documents if doc.tags]
    docs_with_multi_genre = [doc for doc in corpus.documents if len(doc.genres) >= 2]

    if not corpus.documents:
        raise ValueError("No documents available for query generation")
    if not corpus.all_genres:
        raise ValueError("MovieLens corpus has no genres to query")

    queries: list[QuerySpec] = []
    while len(queries) < num_queries:
        choice = rng.choice(
            ["genre", "genre_and_year", "genre_or_genre", "genre_not", "tag_and_genre"]
        )

        if choice == "genre":
            doc = rng.choice(corpus.documents)
            if not doc.genres:
                continue
            genre = rng.choice(doc.genres)
            label = corpus.genre_labels[genre]
            queries.append(QuerySpec(f"{FEATURE_GENRE}:{genre}", f"genre={label}"))
            continue

        if choice == "genre_and_year":
            if not docs_with_year:
                continue
            doc = rng.choice(docs_with_year)
            if not doc.genres:
                continue
            genre = rng.choice(doc.genres)
            label = corpus.genre_labels[genre]
            queries.append(
                QuerySpec(
                    f"{FEATURE_GENRE}:{genre} AND {FEATURE_YEAR}:{doc.year}",
                    f"genre={label} AND year={doc.year}",
                )
            )
            continue

        if choice == "genre_or_genre":
            if docs_with_multi_genre:
                doc = rng.choice(docs_with_multi_genre)
                genre_a, genre_b = rng.sample(list(doc.genres), k=2)
            else:
                doc = rng.choice(corpus.documents)
                if not doc.genres:
                    continue
                genre_a = rng.choice(doc.genres)
                other_choices = [g for g in corpus.all_genres if g != genre_a]
                if not other_choices:
                    continue
                genre_b = rng.choice(other_choices)

            if doc.year is not None:
                queries.append(
                    QuerySpec(
                        (
                            f"({FEATURE_GENRE}:{genre_a} OR {FEATURE_GENRE}:{genre_b}) "
                            f"AND {FEATURE_YEAR}:{doc.year}"
                        ),
                        (
                            f"(genre={corpus.genre_labels[genre_a]} OR "
                            f"genre={corpus.genre_labels[genre_b]}) AND year={doc.year}"
                        ),
                    )
                )
            else:
                queries.append(
                    QuerySpec(
                        f"{FEATURE_GENRE}:{genre_a} OR {FEATURE_GENRE}:{genre_b}",
                        (
                            f"genre={corpus.genre_labels[genre_a]} OR "
                            f"genre={corpus.genre_labels[genre_b]}"
                        ),
                    )
                )
            continue

        if choice == "genre_not":
            doc = rng.choice(corpus.documents)
            if not doc.genres:
                continue
            genre = rng.choice(doc.genres)
            excluded_choices = [g for g in corpus.all_genres if g not in doc.genres]
            if not excluded_choices:
                continue
            excluded = rng.choice(excluded_choices)
            queries.append(
                QuerySpec(
                    f"{FEATURE_GENRE}:{genre} AND NOT {FEATURE_GENRE}:{excluded}",
                    (
                        f"genre={corpus.genre_labels[genre]} AND NOT "
                        f"genre={corpus.genre_labels[excluded]}"
                    ),
                )
            )
            continue

        if choice == "tag_and_genre":
            if not corpus.has_tags or not docs_with_tags:
                continue
            doc = rng.choice(docs_with_tags)
            if not doc.genres:
                continue
            tag = rng.choice(doc.tags)
            genre = rng.choice(doc.genres)
            queries.append(
                QuerySpec(
                    f"{FEATURE_TAG}:{tag} AND {FEATURE_GENRE}:{genre}",
                    (
                        f"tag={corpus.tag_labels[tag]} AND "
                        f"genre={corpus.genre_labels[genre]}"
                    ),
                )
            )

    if not queries:
        raise ValueError("No queries generated")
    return queries


def _size_mb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024.0 * 1024.0)


def _print_summary(
    dataset_dir: Path,
    device: str,
    corpus: MovieLensCorpus,
    build_stats: BenchmarkStats,
    parse_stats: BenchmarkStats,
    search_stats: BenchmarkStats,
    bloom_index: torch.Tensor,
    plans_data: torch.Tensor,
    queries: list[QuerySpec],
    result_mask: torch.Tensor,
    sample_queries: int,
) -> None:
    print("=" * 72)
    print("SILVERTORCH MOVIELENS BENCHMARK")
    print("=" * 72)
    print(f"Dataset:          {dataset_dir}")
    print(f"Device:           {device}")
    print(f"Base movies:      {corpus.base_num_docs}")
    print(f"Indexed docs:     {corpus.num_docs}")
    print(f"Features/doc:     {corpus.feature_ids.numel()}")
    print(f"Tag feature:      {'enabled' if corpus.has_tags else 'disabled'}")
    print(f"Genre values:     {len(corpus.genre_labels)}")
    print(f"Year values:      {len(corpus.all_years)}")
    print(f"Tag values:       {len(corpus.tag_labels)}")
    print(
        f"Index size:       {bloom_index.numel()} int64s ({_size_mb(bloom_index):.2f} MB)"
    )
    print(
        f"Plan size:        {plans_data.numel()} int64s ({_size_mb(plans_data):.2f} MB)"
    )
    print()
    print("Timing (mean / median / min)")
    print(
        f"  Build:          {build_stats.mean_ms:.2f} / {build_stats.median_ms:.2f} / {build_stats.min_ms:.2f} ms"
    )
    print(
        f"  Parse:          {parse_stats.mean_ms:.2f} / {parse_stats.median_ms:.2f} / {parse_stats.min_ms:.2f} ms"
    )
    print(
        f"  Search batch:   {search_stats.mean_ms:.2f} / {search_stats.median_ms:.2f} / {search_stats.min_ms:.2f} ms"
    )
    print()
    print("Sample queries")
    visible_docs = result_mask[:, : corpus.num_docs]
    for idx, query in enumerate(queries[:sample_queries]):
        hit_count = int(visible_docs[idx].sum().item())
        print(f"  {idx + 1}. {query.description}")
        print(f"     expr={query.expression}")
        print(f"     hits={hit_count}")


def _run_smoke_checks(
    corpus: MovieLensCorpus,
    queries: list[QuerySpec],
    bloom_index: torch.Tensor,
    result_mask: torch.Tensor,
) -> None:
    if not hasattr(torch.ops, "st") or not hasattr(torch.ops.st, "bloom_index_build"):
        raise RuntimeError("torch.ops.st.bloom_index_build is not registered")
    if not hasattr(torch.ops.st, "bloom_index_search_batch"):
        raise RuntimeError("torch.ops.st.bloom_index_search_batch is not registered")
    if bloom_index.numel() == 0:
        raise RuntimeError("bloom_index_build returned an empty index")

    visible_docs = result_mask[:, : corpus.num_docs]
    if visible_docs.size(0) != len(queries):
        raise RuntimeError(
            f"search result batch size {visible_docs.size(0)} did not match {len(queries)} queries"
        )
    if visible_docs.size(1) != corpus.num_docs:
        raise RuntimeError(
            f"search result width {visible_docs.size(1)} did not match {corpus.num_docs} indexed docs"
        )
    if not bool(visible_docs.any(dim=1).all().item()):
        raise RuntimeError(
            "at least one generated query returned zero hits; package smoke test failed"
        )


def main(
    argv: list[str] | None = None,
    *,
    default_preset: str = "small",
) -> None:
    args = _parse_args(argv, default_preset=default_preset)
    if args.hash_k < args.k:
        raise ValueError("--hash-k must be >= --k")

    dataset_dir = resolve_dataset_dir(
        dataset_dir=args.dataset_dir,
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        download=args.download,
    )
    device = _choose_device(args.device)

    corpus = load_corpus(
        dataset_dir=dataset_dir,
        include_tags=args.include_tags,
        min_tag_frequency=args.min_tag_frequency,
        max_tags_per_movie=args.max_tags_per_movie,
        max_movies=args.max_movies,
        replication_factor=args.replication_factor,
    )
    queries = _generate_query_specs(
        corpus,
        args.num_queries,
        seed=args.seed,
    )

    feature_ids = corpus.feature_ids.to(device)
    feature_offsets = corpus.feature_offsets.to(device)
    feature_values = corpus.feature_values.to(device)
    silvertorch_ks = torch.full((len(queries),), args.k, dtype=torch.long)
    expressions = [query.expression for query in queries]

    build_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]] = lambda: cast(
        tuple[torch.Tensor, torch.Tensor],
        torch.ops.st.bloom_index_build(
            feature_ids,
            feature_offsets,
            feature_values,
            args.b_multiplier,
            args.k,
        ),
    )
    build_stats, build_result = _measure(
        build_fn,
        device=device,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )
    bloom_index, bundle_b_offsets = build_result

    parse_fn: Callable[[], tuple[object, tuple[torch.Tensor, torch.Tensor]]] = (
        lambda: cast(
            tuple[object, tuple[torch.Tensor, torch.Tensor]],
            torch.ops.st.parse_expression_query_batch(
                expressions,
                silvertorch_ks,
                args.hash_k,
                True,
                8,
            ),
        )
    )
    parse_stats, parse_result = _measure(
        parse_fn,
        device="cpu",
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )
    _, tensors = parse_result
    plans_data, plans_offsets = tensors

    if device == "cuda":
        plans_data = plans_data.cuda()
        plans_offsets = plans_offsets.cuda()

    search_fn: Callable[[], torch.Tensor] = lambda: cast(  # noqa: E731
        torch.Tensor,
        torch.ops.st.bloom_index_search_batch(
            bloom_index,
            bundle_b_offsets,
            plans_data,
            plans_offsets,
            args.k,
            args.hash_k,
            True,
        ),
    )
    search_stats, search_result = _measure(
        search_fn,
        device=device,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )

    result_mask = (
        search_result.cpu() if search_result.device.type == "cuda" else search_result
    )
    _run_smoke_checks(corpus, queries, bloom_index, result_mask)
    _print_summary(
        dataset_dir=dataset_dir,
        device=device,
        corpus=corpus,
        build_stats=build_stats,
        parse_stats=parse_stats,
        search_stats=search_stats,
        bloom_index=bloom_index,
        plans_data=plans_data,
        queries=queries,
        result_mask=result_mask,
        sample_queries=args.sample_queries,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
