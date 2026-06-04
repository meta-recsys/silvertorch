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

"""Shared MovieLens downloader and tensor loader for benchmarks."""

from __future__ import annotations

import csv
import re
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

DATASET_URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    "ml-latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
}
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

FEATURE_GENRE = 1
FEATURE_YEAR = 2
FEATURE_TAG = 3

TAG_TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


@dataclass(frozen=True)
class MovieDocument:
    movie_id: int
    title: str
    genres: tuple[int, ...]
    year: int | None
    tags: tuple[int, ...]


@dataclass(frozen=True)
class MovieLensCorpus:
    feature_ids: torch.Tensor
    feature_offsets: torch.Tensor
    feature_values: torch.Tensor
    documents: list[MovieDocument]
    num_docs: int
    base_num_docs: int
    has_tags: bool
    genre_labels: dict[int, str]
    tag_labels: dict[int, str]
    all_genres: list[int]
    all_years: list[int]
    all_tags: list[int]


def _normalize_tag(raw_tag: str) -> str | None:
    tokens = TAG_TOKEN_RE.findall(raw_tag.lower())
    if not tokens:
        return None
    return " ".join(tokens)


def _extract_year(title: str) -> int | None:
    match = YEAR_RE.search(title)
    return int(match.group(1)) if match is not None else None


def resolve_dataset_dir(
    *,
    dataset_dir: Path | None,
    data_root: Path,
    dataset_name: str,
    download: bool,
) -> Path:
    if dataset_dir is not None:
        resolved_dir = dataset_dir.expanduser().resolve()
        if not resolved_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {resolved_dir}")
        return resolved_dir

    resolved_root = data_root.expanduser().resolve()
    resolved_dir = resolved_root / dataset_name
    if resolved_dir.is_dir():
        return resolved_dir

    if not download:
        raise FileNotFoundError(
            "MovieLens dataset not found. Pass --download or set --dataset-dir."
        )

    return download_dataset(dataset_name, resolved_root)


def download_dataset(dataset_name: str, data_root: Path) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_root / dataset_name
    if dataset_dir.is_dir():
        return dataset_dir

    url = DATASET_URLS[dataset_name]
    archive_path = data_root / f"{dataset_name}.zip"
    print(f"Downloading {dataset_name} from {url}")
    try:
        urllib.request.urlretrieve(url, archive_path)
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Failed to download MovieLens. Pass --dataset-dir to a local extracted "
            "MovieLens CSV dataset if download is unavailable."
        ) from error

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(data_root)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Expected extracted dataset directory at {dataset_dir}, but it was not created."
        )
    return dataset_dir


def _load_tag_metadata(
    tags_path: Path,
    min_tag_frequency: int,
    max_tags_per_movie: int,
) -> dict[int, tuple[str, ...]]:
    per_movie_tags: dict[int, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()

    with tags_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_tag = row.get("tag")
            movie_id = row.get("movieId")
            if raw_tag is None or movie_id is None:
                continue
            tag = _normalize_tag(raw_tag)
            if tag is None:
                continue
            movie_key = int(movie_id)
            per_movie_tags[movie_key][tag] += 1
            global_counts[tag] += 1

    allowed_tags = {
        tag for tag, count in global_counts.items() if count >= min_tag_frequency
    }

    filtered: dict[int, tuple[str, ...]] = {}
    for movie_id, counts in per_movie_tags.items():
        kept = [
            tag
            for tag, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if tag in allowed_tags
        ][:max_tags_per_movie]
        filtered[movie_id] = tuple(kept)
    return filtered


def load_corpus(
    dataset_dir: Path,
    *,
    include_tags: bool,
    min_tag_frequency: int,
    max_tags_per_movie: int,
    max_movies: int | None,
    replication_factor: int,
) -> MovieLensCorpus:
    if replication_factor < 1:
        raise ValueError("--replication-factor must be >= 1")

    movies_path = dataset_dir / "movies.csv"
    if not movies_path.is_file():
        raise FileNotFoundError(
            f"{dataset_dir} does not look like a CSV MovieLens dataset: missing movies.csv"
        )

    tag_strings_by_movie: dict[int, tuple[str, ...]] = {}
    tags_path = dataset_dir / "tags.csv"
    if include_tags and tags_path.is_file():
        tag_strings_by_movie = _load_tag_metadata(
            tags_path,
            min_tag_frequency=min_tag_frequency,
            max_tags_per_movie=max_tags_per_movie,
        )

    raw_movies: list[tuple[int, str, tuple[str, ...], int | None, tuple[str, ...]]] = []
    all_genres: set[str] = set()
    all_tags: set[str] = set()

    with movies_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            movie_id = int(row["movieId"])
            title = row["title"]
            genres_raw = row["genres"]
            genres = (
                ()
                if genres_raw == "(no genres listed)"
                else tuple(sorted(genres_raw.split("|")))
            )
            tags = tag_strings_by_movie.get(movie_id, ())
            year = _extract_year(title)

            all_genres.update(genres)
            all_tags.update(tags)
            raw_movies.append((movie_id, title, genres, year, tags))

            if max_movies is not None and len(raw_movies) >= max_movies:
                break

    if not raw_movies:
        raise ValueError(f"No movies were loaded from {movies_path}")

    genre_to_value = {genre: idx + 1 for idx, genre in enumerate(sorted(all_genres))}
    tag_to_value = {tag: idx + 1 for idx, tag in enumerate(sorted(all_tags))}

    documents: list[MovieDocument] = []
    for movie_id, title, genres, year, tags in raw_movies:
        documents.append(
            MovieDocument(
                movie_id=movie_id,
                title=title,
                genres=tuple(genre_to_value[g] for g in genres),
                year=year,
                tags=tuple(tag_to_value[tag] for tag in tags if tag in tag_to_value),
            )
        )

    has_tags = any(doc.tags for doc in documents)
    feature_id_list = [FEATURE_GENRE, FEATURE_YEAR]
    if has_tags:
        feature_id_list.append(FEATURE_TAG)

    values: list[int] = []
    offsets = [0]
    for _ in range(replication_factor):
        for doc in documents:
            values.extend(doc.genres)
            offsets.append(len(values))

            if doc.year is not None:
                values.append(doc.year)
            offsets.append(len(values))

            if has_tags:
                values.extend(doc.tags)
                offsets.append(len(values))

    return MovieLensCorpus(
        feature_ids=torch.tensor(feature_id_list, dtype=torch.int32),
        feature_offsets=torch.tensor(offsets, dtype=torch.long),
        feature_values=torch.tensor(values, dtype=torch.long),
        documents=documents,
        num_docs=len(documents) * replication_factor,
        base_num_docs=len(documents),
        has_tags=has_tags,
        genre_labels={value: genre for genre, value in genre_to_value.items()},
        tag_labels={value: tag for tag, value in tag_to_value.items()},
        all_genres=sorted(genre_to_value.values()),
        all_years=sorted({doc.year for doc in documents if doc.year is not None}),
        all_tags=sorted(tag_to_value.values()),
    )
