███████╗██╗██╗     ██╗   ██╗███████╗██████╗ ████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗
██╔════╝██║██║     ██║   ██║██╔════╝██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║
███████╗██║██║      ██║ ██║ █████╗  ██████╔╝   ██║   ██║   ██║██████╔╝██║     ███████║
╚════██║██║██║      ██║ ██║ ██╔══╝  ██╔══██╗   ██║   ██║   ██║██╔══██╗██║     ██╔══██║
███████║██║███████╗ ╚████╔╝ ███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║
╚══════╝╚═╝╚══════╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

# SilverTorch

GPU Retrieval Library for **Recommendation • GenAI RAG • Search Systems**

SilverTorch is a high-performance open-source GPU retrieval engine designed for
large-scale recommender systems, generative retrieval (RAG), and vector search —
optimized for latency, throughput, and GPU cost efficiency.

- **Version:** 1.0.0
- **Contact:** bixue@meta.com, leich@meta.com
- **License:** Apache 2.0

---

## Installation

SilverTorch builds a PyTorch C++/CUDA extension (`silvertorch._C`) from source.
The **single most common install failure is a CUDA / PyTorch version mismatch** —
PyTorch wheels are pinned to a specific CUDA toolkit version, and your local
`nvcc` must match. The 4-step flow below avoids that.

### 1. Tested matrix

These combinations are verified end-to-end (build + 73-test suite + benchmark).

| Python    | PyTorch       | CUDA toolkit | Notes                       |
|-----------|---------------|--------------|-----------------------------|
| 3.10–3.11 | 2.4 – 2.6     | 12.1 / 12.4  | Conservative                |
| 3.11–3.12 | **2.7 – 2.10**| **12.8**     | **Recommended**             |
| 3.13      | 2.10+         | 12.8         | Experimental                |

PyTorch 2.11+ paired with CUDA 13.x is **not yet supported** — the bundled
CCCL 3.x headers conflict with PyTorch's `at_cuda_detail::cub` shim. Use
PyTorch ≤ 2.10 with CUDA 12.x instead.

### 2. Install (4 steps)

```bash
# 2.1  Point CUDA_HOME at your CUDA toolkit. nvcc must be on PATH and its
#      version MUST match the torch wheel you install in step 2.2.
export CUDA_HOME=/usr/local/cuda-12.8        # adjust to your install
export PATH=$CUDA_HOME/bin:$PATH

# 2.2  Install a PyTorch wheel built against the SAME CUDA version.
#      Index URLs: cu121 → CUDA 12.1, cu124 → CUDA 12.4, cu128 → CUDA 12.8
pip install "torch>=2.7,<2.11" --index-url https://download.pytorch.org/whl/cu128

# 2.3  Install ninja (cuts the C++/CUDA build to roughly 30 seconds).
pip install ninja

# 2.4  Build SilverTorch against the torch you just installed.
#      --no-build-isolation is REQUIRED — without it, pip creates a fresh build
#      env and pulls the "latest" torch, re-introducing the version mismatch.
pip install --no-build-isolation silvertorch
```

For local development from a checkout:

```bash
git clone https://github.com/<org>/silvertorch.git
cd silvertorch
pip install --no-build-isolation -e .
```

### 3. Verify the install

```python
import silvertorch.ops._load_ops  # loads silvertorch._C, registers torch.ops.st.*
import torch
assert hasattr(torch.ops.st, "bloom_index_build")
assert hasattr(torch.ops.st, "bloom_index_search_batch")
print("SilverTorch ready —", "GPU" if torch.cuda.is_available() else "CPU only")
```

Run the test suite (expect 73 passed):

```bash
pytest silvertorch/
```

### 4. CPU-only build

If `torch.cuda.is_available()` returns False at install time, `setup.py`
automatically builds a CPU-only extension. The bloom-index search ops still
work; only the CUDA-accelerated kernels are unavailable.

---

## Quick start

SilverTorch's bloom index lets you boolean-filter millions of documents on
GPU in a few microseconds per query. A document is a small set of
`(feature_id, feature_value)` pairs — for example, a video tagged with
`(language, "en")`, `(category, "music")`, `(creator_id, 12345)`. You build
a packed bloom index once, then evaluate boolean queries (`AND`, `OR`,
`NOT`, parens) over millions of documents in a single fused kernel.

### Example 1 — Low-level ops API (recommended for production pipelines)

```python
import silvertorch.ops._load_ops  # noqa: F401  registers torch.ops.st.*
import torch

# ----- 1. Describe a small corpus of 4 documents over 2 features -----
# Feature schema (column order matters):
#   feature_ids[0] = 1  → "language"
#   feature_ids[1] = 2  → "category"
#
# Documents:
#   doc 0: language={en},        category={music, pop}
#   doc 1: language={en, es},    category={music}
#   doc 2: language={ja},        category={news}
#   doc 3: language={en},        category={music, rock}
feature_ids = torch.tensor([1, 2], dtype=torch.int32)

# Jagged layout: feature_offsets has (num_docs * num_features + 1) entries.
# For document d, feature f, the values are
#   feature_values[feature_offsets[d * num_features + f]
#                : feature_offsets[d * num_features + f + 1]]
feature_values = torch.tensor(
    [
        100,                # doc 0, language: en
        200, 201,           # doc 0, category: music, pop
        100, 101,           # doc 1, language: en, es
        200,                # doc 1, category: music
        102,                # doc 2, language: ja
        202,                # doc 2, category: news
        100,                # doc 3, language: en
        200, 203,           # doc 3, category: music, rock
    ],
    dtype=torch.long,
)
feature_offsets = torch.tensor(
    [0, 1, 3, 5, 6, 7, 8, 9, 11], dtype=torch.long
)

# ----- 2. Build the bloom index -----
K = 3        # number of hash positions per term (more = lower false-positive rate)
HASH_K = 7   # number of pre-computed hashes (used by V2 indexer; HASH_K >= K)
B_MULTIPLIER = 5.0  # bloom width = max_features_per_doc * K * B_MULTIPLIER
                    # Higher = fewer false positives, more memory.
                    # Rule of thumb: start at 5–10 for production corpora.

bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
    feature_ids,
    feature_offsets,
    feature_values,
    B_MULTIPLIER,
    K,
)

# ----- 3. Parse boolean queries -----
# Query syntax:  feature_id:value joined with AND / OR / NOT / parentheses.
queries = [
    "1:100",                            # docs in English
    "1:100 AND NOT 2:201",              # English, but NOT tagged 'pop'
    "1:102 OR 2:203",                   # Japanese OR rock
]
silvertorch_ks = torch.full((len(queries),), K, dtype=torch.long)

_, [plans_data, plans_offsets] = torch.ops.st.parse_expression_query_batch(
    queries,
    silvertorch_ks,
    HASH_K,
    True,   # return_query_plan
    5,      # max_sub_queries
)

# ----- 4. Search the index -----
# Returns a [num_queries, num_docs] bool mask.
mask = torch.ops.st.bloom_index_search_batch(
    bloom_index,
    bundle_b_offsets,
    plans_data,
    plans_offsets,
    K,
    HASH_K,
    True,  # return_bool_mask (False → packed uint64 bitmask)
)

num_docs = feature_offsets.numel() // feature_ids.numel()
for q, m in zip(queries, mask[:, :num_docs]):
    hits = m.nonzero().flatten().tolist()
    print(f"{q!r:30} → docs {hits}")
# '1:100'                        → docs [0, 1, 3]
# '1:100 AND NOT 2:201'          → docs [1, 3]
# '1:102 OR 2:203'               → docs [2, 3]
```

> **Bloom is a candidate filter, not an exact one.** The mask returned by
> `bloom_index_search_batch` is guaranteed to contain every true match, but
> may include a few false positives — that's the bloom-filter trade-off. To
> reduce the false-positive rate, increase `K` and/or `B_MULTIPLIER` (they
> trade off against index size and build time). For exact filtering, treat
> the mask as a candidate set and verify each candidate against your source
> of truth.

To run on GPU, move the input tensors to CUDA before calling
`bloom_index_build` and `bloom_index_search_batch`. The query parser stays
on CPU; only `plans_data` and `plans_offsets` need `.cuda()` before the
search call.

### Example 2 — High-level Module API (recommended for `nn.Module` pipelines)

The same flow wrapped in `torch.nn.Module`s, ready to plug into a serving
graph or `torch.compile`-traced model. This snippet is self-contained — it
re-defines the same 4-document corpus from Example 1 so you can run it
standalone:

```python
import silvertorch.ops._load_ops  # noqa: F401
import torch
from silvertorch.modules import (
    BloomIndexSearchModule,
    FilterQueryParserModule,
)

# Same 4-doc / 2-feature corpus as Example 1.
feature_ids = torch.tensor([1, 2], dtype=torch.int32)
feature_values = torch.tensor(
    [100, 200, 201, 100, 101, 200, 102, 202, 100, 200, 203],
    dtype=torch.long,
)
feature_offsets = torch.tensor(
    [0, 1, 3, 5, 6, 7, 8, 9, 11], dtype=torch.long
)

K, HASH_K = 3, 7

# 1. Build the index (using the ops API as in Example 1). Use B_MULTIPLIER=5.0
#    or higher to keep the bloom-filter false-positive rate low — see the
#    note under Example 1.
bloom_index, bundle_b_offsets = torch.ops.st.bloom_index_build(
    feature_ids, feature_offsets, feature_values, 5.0, K
)

# 2. Wrap the index as a Module and stash it as a buffer.
search = BloomIndexSearchModule(k=K, hash_k=HASH_K)
search.set_bloom_index(bloom_index)
search.set_bloom_bundle_b_offsets(bundle_b_offsets)

# 3. Wrap the parser; it has no state.
parser = FilterQueryParserModule(hash_k=HASH_K, max_sub_queries=5)

# 4. Run end-to-end as a normal nn.Module.
queries = ["1:100 AND 2:200", "1:102"]
silvertorch_ks = torch.full((len(queries),), K, dtype=torch.long)

_, [plans_data, plans_offsets] = parser(queries, silvertorch_ks)
mask = search(plans_data, plans_offsets, return_bool_mask=True)
print(mask.shape, mask.dtype)   # torch.Size([2, 2048]) torch.bool
# Mask is sized to the bundle-aligned doc space (rounded up to a multiple of
# 32 docs * 64 bits per uint64). The first `num_docs` positions are real;
# the rest are padding. Slice with `mask[:, :num_docs]` for the true matches:
num_docs = feature_offsets.numel() // feature_ids.numel()
for q, m in zip(queries, mask[:, :num_docs]):
    print(f"{q!r:25} -> docs {m.nonzero().flatten().tolist()}")
# '1:100 AND 2:200'         -> docs [0, 1, 3]
# '1:102'                   -> docs [2]
```

For end-to-end CPU/GPU consistency tests and batched-query examples, see
`silvertorch/ops/csrc/tests/test_bloom_search_integration.py` and
`silvertorch/ops/csrc/tests/bloom_index_bench.py`.

---

## Troubleshooting

**`The detected CUDA version (X.Y) mismatches the version that was used to compile PyTorch (A.B)`**
Your `nvcc` (from `CUDA_HOME` / `PATH`) does not match the CUDA the torch wheel
was built against. Re-do step 2.1 with the right `CUDA_HOME`, or reinstall
torch from the matching `--index-url` (step 2.2).

**`namespace "at_cuda_detail::cub" has no member ...`**
You are on PyTorch 2.11+ with CUDA 13. Downgrade to PyTorch ≤ 2.10 with
CUDA 12.x — see the tested matrix above.

**`ModuleNotFoundError: No module named 'torch'` during `pip install .`**
You omitted `--no-build-isolation` and pip's isolated build env did not pick
up your local torch. Rerun with `--no-build-isolation` after step 2.2.

**`Attempted to use ninja as the BuildExtension backend but we could not find ninja`**
Cosmetic warning. `pip install ninja` cuts build time roughly 5×.

**`ninja: error: build.ninja:NN: multiple rules generate <name>.o`**
You added a new pair of `name.cpp` + `name.cu` files. Rename the `.cu` to
`name_cuda.cu` (PyTorch ≤ 2.5 strips file extensions when computing object
filenames, so same-base-name pairs collide).

---
© 2025 The Silvertorch Project | Open Source
