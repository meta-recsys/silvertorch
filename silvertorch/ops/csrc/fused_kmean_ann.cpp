// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/library.h> // @manual
#include <torch/torch.h> // @manual

#include <cfloat>
#include <cmath>
#include <limits>

namespace st::ops::fused_kmean_ann {

using at::Tensor;

namespace {

// Determine score dtype: for int8 embeddings the score type depends on
// whether a divisor/scale is provided; for all other types score type
// matches the embedding type.
at::ScalarType resolve_score_dtype(
    at::ScalarType emb_dtype,
    int64_t divisor_for_int8,
    bool has_per_embedding_scale) {
  if (emb_dtype != at::kChar) {
    return emb_dtype;
  }
  return (divisor_for_int8 != -1 || has_per_embedding_scale) ? at::kHalf
                                                             : at::kInt;
}

// Compute a single dot product between query[0..dim) and emb[0..dim) in
// float64 so the reference is as accurate as possible regardless of the
// storage type.
double dot_f64(const Tensor& emb_row, const Tensor& query_row) {
  auto e = emb_row.to(at::kDouble).contiguous();
  auto q = query_row.to(at::kDouble).contiguous();
  const double* ep = e.data_ptr<double>();
  const double* qp = q.data_ptr<double>();
  double acc = 0.0;
  for (int64_t d = 0; d < e.numel(); ++d) {
    acc += ep[d] * qp[d];
  }
  return acc;
}

// For int8 embeddings the raw dot product is an integer sum; reproduce
// that exactly so the CPU reference matches the GPU dp4a path.
int32_t dot_int8(const Tensor& emb_row, const Tensor& query_row) {
  auto e = emb_row.contiguous();
  auto q = query_row.contiguous();
  const int8_t* ep = e.data_ptr<int8_t>();
  const int8_t* qp = q.data_ptr<int8_t>();
  int32_t acc = 0;
  for (int64_t d = 0; d < e.numel(); ++d) {
    acc += static_cast<int32_t>(ep[d]) * static_cast<int32_t>(qp[d]);
  }
  return acc;
}

bool check_filtering_bit(
    const int64_t* mask_ptr,
    int64_t mask_columns,
    int64_t doc_index) {
  int64_t word = doc_index / 64;
  int64_t bit = doc_index % 64;
  if (word >= mask_columns) {
    return true;
  }
  return (static_cast<uint64_t>(mask_ptr[word]) >> bit) & 1ULL;
}

} // namespace

static std::tuple<Tensor, Tensor> fused_kmean_ann_cpu(
    const Tensor& cluster_offsets,
    const Tensor& cluster_ids,
    const Tensor& cluster_length,
    const Tensor& embeddings,
    const Tensor& queries,
    int64_t max_tensor_size_per_row,
    const std::optional<Tensor>& filtering_bit_mask,
    int64_t invalid_index_value,
    int64_t divisor_for_int8,
    const std::optional<Tensor>& filtering_bit_index,
    const std::optional<Tensor>& per_embedding_scale) {
  TORCH_CHECK(cluster_offsets.dim() == 1);
  TORCH_CHECK(cluster_ids.dim() == 2);
  TORCH_CHECK(cluster_length.dim() == 2);
  TORCH_CHECK(embeddings.dim() == 2);
  TORCH_CHECK(queries.dim() == 2);
  TORCH_CHECK(cluster_ids.size(0) == cluster_length.size(0));
  TORCH_CHECK(cluster_ids.size(0) == queries.size(0));

  // Match GPU behavior: round up to warp-size boundary.
  constexpr int64_t kWarpThreadCount = 32;
  max_tensor_size_per_row = (max_tensor_size_per_row + kWarpThreadCount - 1) /
      kWarpThreadCount * kWarpThreadCount;

  const int64_t batch_size = cluster_ids.size(0);
  const int64_t num_probes = cluster_ids.size(1);
  const bool is_int8 = embeddings.scalar_type() == at::kChar;
  const bool has_scale = per_embedding_scale.has_value();
  const at::ScalarType score_dtype = resolve_score_dtype(
      embeddings.scalar_type(), divisor_for_int8, has_scale);

  auto offsets_a = cluster_offsets.accessor<int64_t, 1>();
  auto ids_a = cluster_ids.accessor<int64_t, 2>();
  auto len_a = cluster_length.accessor<int64_t, 2>();

  // Filtering bitmask pointers (optional).
  const int64_t* filter_ptr = nullptr;
  int64_t filter_cols = 0;
  if (filtering_bit_mask) {
    filter_ptr = filtering_bit_mask->data_ptr<int64_t>();
    filter_cols = filtering_bit_mask->size(1);
  }
  const int64_t* filter_index_ptr = nullptr;
  if (filtering_bit_index) {
    filter_index_ptr = filtering_bit_index->data_ptr<int64_t>();
  }

  // Per-embedding scale (optional, float16).
  const at::Half* scale_ptr = nullptr;
  if (has_scale) {
    scale_ptr = per_embedding_scale->data_ptr<at::Half>();
  }

  // Allocate output: scores filled with dtype-minimum, indices with
  // invalid_index_value.
  Tensor scores;
  if (score_dtype == at::kInt) {
    scores = at::full(
        {batch_size, max_tensor_size_per_row},
        std::numeric_limits<int32_t>::min(),
        c10::TensorOptions().dtype(at::kInt));
  } else if (score_dtype == at::kHalf) {
    scores = at::full(
        {batch_size, max_tensor_size_per_row},
        -65504.0f,
        c10::TensorOptions().dtype(at::kHalf));
  } else if (score_dtype == at::kBFloat16) {
    scores = at::full(
        {batch_size, max_tensor_size_per_row},
        -FLT_MAX,
        c10::TensorOptions().dtype(at::kBFloat16));
  } else {
    scores = at::full(
        {batch_size, max_tensor_size_per_row},
        -FLT_MAX,
        c10::TensorOptions().dtype(at::kFloat));
  }

  Tensor indices = at::full(
      {batch_size, max_tensor_size_per_row},
      static_cast<int32_t>(invalid_index_value),
      c10::TensorOptions().dtype(at::kInt));

  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t write_col = 0;
    const auto query_row = queries[b];

    // Resolve which row of the bitmask to use for this query.
    int64_t filter_row = b;
    if (filter_index_ptr) {
      filter_row = filter_index_ptr[b];
    }

    for (int64_t p = 0; p < num_probes; ++p) {
      int64_t cid = ids_a[b][p];
      int64_t clen = len_a[b][p];
      int64_t doc_start = offsets_a[cid];

      for (int64_t i = 0; i < clen && write_col < max_tensor_size_per_row;
           ++i) {
        int64_t doc_idx = doc_start + i;

        // Apply filtering bitmask if present.
        if (filter_ptr) {
          const int64_t* row_mask = filter_ptr + filter_row * filter_cols;
          if (!check_filtering_bit(row_mask, filter_cols, doc_idx)) {
            continue;
          }
        }

        const auto emb_row = embeddings[doc_idx];

        if (is_int8) {
          int32_t raw = dot_int8(emb_row, query_row);
          if (has_scale) {
            float s = static_cast<float>(scale_ptr[doc_idx]);
            scores[b][write_col] = at::Half(static_cast<float>(raw) / s);
          } else if (divisor_for_int8 != -1) {
            scores[b][write_col] = at::Half(
                static_cast<float>(raw) / static_cast<float>(divisor_for_int8));
          } else {
            scores[b][write_col] = raw;
          }
        } else {
          double s = dot_f64(emb_row, query_row);
          if (score_dtype == at::kFloat) {
            scores[b][write_col] = static_cast<float>(s);
          } else if (score_dtype == at::kHalf) {
            scores[b][write_col] = at::Half(static_cast<float>(s));
          } else if (score_dtype == at::kBFloat16) {
            scores[b][write_col] = at::BFloat16(static_cast<float>(s));
          }
        }

        indices[b][write_col] = static_cast<int32_t>(doc_idx);
        ++write_col;
      }
    }
  }

  return std::make_tuple(std::move(scores), std::move(indices));
}

TORCH_LIBRARY_FRAGMENT(st, m) {
  // fused_kmean_ann: Fused KMeans-based Approximate Nearest Neighbor search.
  //
  // This is the core retrieval kernel for clustered KNN indices. It fuses
  // two operations that would otherwise require separate kernel launches:
  //   1. Cluster offset lookup -- expand selected cluster IDs into document
  //      index ranges using the CSR-format cluster_offsets array.
  //   2. Batched dot-product scoring -- compute the dot product between each
  //      query and every document embedding within the selected clusters.
  //
  // The kernel is designed for GPU execution using CUDA warp-level
  // parallelism. Each cluster's documents are partitioned into:
  //   - "warp payloads": groups of 32 documents processed by a full warp.
  //   - "remaining payloads": leftover documents (< 32) processed individually.
  // This two-phase approach maximizes GPU utilization by aligning work to
  // warp boundaries.
  //
  // Index build flow (produces the inputs to this op):
  //   1. Run KMeans clustering on item embeddings to get centroids + cluster
  //      assignments.
  //   2. Sort items by cluster assignment; compute cluster_offsets as a
  //      prefix-sum of cluster sizes (CSR format).
  //   3. At query time, find the nearest centroids (probes) for each query,
  //      gather the corresponding cluster_ids and cluster_length tensors.
  //   4. Call this op to score all documents within the selected clusters.
  //
  // Supported embedding dtypes: float32, float16, bfloat16, int8.
  // For int8 embeddings, the dot-product accumulates in int32 using dp4a
  // instructions; the result is optionally scaled by divisor_for_int8 or
  // per_embedding_scale to produce half-precision scores.
  //
  // Args:
  //   cluster_offsets: 1-D tensor of shape (num_clusters + 1). CSR-format
  //     boundaries of the embedding table grouped by cluster. E.g., clusters
  //     of sizes [3, 5, 7] yield cluster_offsets = [0, 3, 8, 15].
  //   cluster_ids: 2-D tensor of shape (B, num_probes). For each query in
  //     the batch, which cluster IDs to search.
  //   cluster_length: 2-D tensor of shape (B, num_probes). Number of
  //     documents in each selected cluster (must match cluster_ids shape).
  //   embeddings: 2-D tensor of shape (total_docs, DIM). The full embedding
  //     table, sorted by cluster assignment so that documents in the same
  //     cluster are contiguous. DIM must be one of {16, 32, 64, 96, 128,
  //     192, 256, 384, 512, 768, 1024, 1280}.
  //   queries: 2-D tensor of shape (B, DIM). Query embeddings; batch size B
  //     must match cluster_ids.size(0), and DIM must match embeddings.size(1).
  //   max_tensor_size_per_row: int. Maximum number of candidate documents per
  //     query (= sum of cluster lengths across all probes for the largest
  //     query). Determines the output width; shorter rows are padded.
  //   filtering_bit_mask: optional 2-D int64 tensor of shape (B, num_words).
  //     Per-query bitmask where bit j indicates whether document j passes
  //     the filter. Documents failing the filter are skipped (not scored).
  //   invalid_index_value: int (default -1). Padding value for the indices
  //     tensor in positions beyond the actual number of documents.
  //   divisor_for_int8: int (default -1). When embeddings are int8 and this
  //     value != -1, each int32 dot-product is divided by this value and
  //     stored as float16. When -1, raw int32 scores are returned.
  //   filtering_bit_index: optional 1-D int64 tensor of shape (B). Maps
  //     each query row to its corresponding row in filtering_bit_mask,
  //     enabling shared bitmasks across queries.
  //   per_embedding_scale: optional 1-D float16 tensor of shape (total_docs).
  //     Per-document scale factor. When present, the int8 dot-product for
  //     document i is divided by per_embedding_scale[i] instead of
  //     divisor_for_int8. Ignored for non-int8 embeddings.
  //
  // Returns:
  //   A tuple of two tensors, both of shape (B, max_tensor_size_per_row):
  //     scores: dot-product scores. Dtype matches the embedding dtype for
  //       float/half/bfloat16; for int8 it is int32 (no divisor) or float16
  //       (with divisor or per_embedding_scale). Padding positions are filled
  //       with the minimum representable value of the score dtype.
  //     indices: int32 document indices into the embedding table. Padding
  //       positions are filled with invalid_index_value.
  //
  // Example:
  //   # 3 clusters with 3, 5, 7 documents; embedding dim = 64
  //   cluster_offsets = [0, 3, 8, 15]             # shape (4,)
  //   embeddings = randn(15, 64)                  # shape (15, 64)
  //   queries = randn(2, 64)                      # shape (2, 64), batch=2
  //   cluster_ids = [[0, 2], [1, 2]]              # shape (2, 2), 2 probes
  //   cluster_length = [[3, 7], [5, 7]]           # shape (2, 2)
  //   max_tensor_size_per_row = 12                # max(3+7, 5+7) = 12
  //
  //   scores, indices = st.fused_kmean_ann(
  //       cluster_offsets, cluster_ids, cluster_length,
  //       embeddings, queries, max_tensor_size_per_row)
  //   # scores.shape  == (2, 12)  -- padded with -FLT_MAX
  //   # indices.shape == (2, 12)  -- padded with -1
  m.def(
      "fused_kmean_ann("
      "Tensor cluster_offsets, "
      "Tensor cluster_ids, "
      "Tensor cluster_length, "
      "Tensor embeddings, "
      "Tensor queries, "
      "int max_tensor_size_per_row, "
      "Tensor? filtering_bit_mask=None, "
      "int invalid_index_value=-1,"
      "int divisor_for_int8=-1,"
      "Tensor? filtering_bit_index=None, "
      "Tensor? per_embedding_scale=None "
      ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(st, CPU, m) {
  m.impl(
      "fused_kmean_ann",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(fused_kmean_ann_cpu)));
}

} // namespace st::ops::fused_kmean_ann
