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

#pragma once

#include <array>
#include <tuple>

#include <ATen/ATen.h>
#include "bloom_index_util.h"

namespace st {
namespace ops {
namespace bloom_index {

#if defined(BLOOM_INDEX_CUDA)
#define BLOOM_INDEX_DEVICE_HOST_INLINE static __device__ __host__ __inline__
#else
#define BLOOM_INDEX_DEVICE_HOST_INLINE static inline
#endif

/***
   BIT Array arrangement (use two bytes as example):
   BYTE:         0                        1
   BIT:  7  6  5  4  3  2  1  0  |  7  6  5  4  3  2  1  0
   SEQ:  7  6  5  4  3  2  1  0     15 14 13 12 11 10 9  8
***/

template <bool is_bloom_v2>
struct BIndexType;

template <>
struct BIndexType<true> {
  using type = int32_t;
};

template <>
struct BIndexType<false> {
  using type = int16_t;
};

// NOLINTNEXTLINE:
BLOOM_INDEX_DEVICE_HOST_INLINE int64_t round_bits_to_bytes(int64_t b) {
  return (b + 8 - 1) / 8;
}

// NOLINTNEXTLINE:
BLOOM_INDEX_DEVICE_HOST_INLINE uint64_t
murmur_hash3_2x64(const uint64_t x, const uint64_t y, const uint64_t seed) {
  const uint64_t c1 = 0x87c37b91114253d5;
  const uint64_t c2 = 0x4cf5ad432745937f;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  // First 64-bit block
  uint64_t k1 = x;
  k1 *= c1;
  k1 = (k1 << 31) | (k1 >> (64 - 31));
  k1 *= c2;
  h1 ^= k1;
  h1 = (h1 << 27) | (h1 >> (64 - 27));
  h1 += h2;
  h1 = h1 * 5 + 0x52dce729;

  // Second 64-bit block
  uint64_t k2 = y;
  k2 *= c2;
  k2 = (k2 << 33) | (k2 >> (64 - 33));
  k2 *= c1;
  h2 ^= k2;
  h2 = (h2 << 31) | (h2 >> (64 - 31));
  h2 += h1;
  h2 = h2 * 5 + 0x38495ab5;

  // Finalization
  h1 ^= 16;
  h2 ^= 16;
  h1 += h2;
  h2 += h1;
  h1 ^= h1 >> 33;
  h1 *= 0xff51afd7ed558ccd;
  h1 ^= h1 >> 33;
  h1 *= 0xc4ceb9fe1a85ec53;
  h1 ^= h1 >> 33;
  h2 ^= h2 >> 33;
  h2 *= 0xff51afd7ed558ccd;
  h2 ^= h2 >> 33;
  h2 *= 0xc4ceb9fe1a85ec53;
  h2 ^= h2 >> 33;
  h1 += h2;
  h2 += h1;

  return h1 ^ h2;
}

// NOLINTNEXTLINE:
BLOOM_INDEX_DEVICE_HOST_INLINE void set_bit(
    int64_t b_p,
    int8_t* document_signature_ptr) {
  // NOLINTNEXTLINE: narrowing conversion from 'int' to signed type 'int8_t'
  *(document_signature_ptr + b_p / 8) |= static_cast<int8_t>(1) << (b_p % 8);
}

// NOLINTNEXTLINE:
BLOOM_INDEX_DEVICE_HOST_INLINE bool get_bit(
    const int8_t* document_signature_ptr,
    int64_t b_p) {
  return (*(document_signature_ptr + b_p / 8) &
          (static_cast<int8_t>(1) << (b_p % 8))) != 0;
}

// NOLINTNEXTLINE:
BLOOM_INDEX_DEVICE_HOST_INLINE bool get_bit_32_bit_mask(
    uint32_t bloom_index_bit_mask,
    int64_t pos) {
  //     bloom index bit mask output.
  //     lower doc id put at higher bits in unit32_t, for example:
  //     |< high                                       low >|
  //      Doc0, Doc1, Doc2 ...... Doc 28, Doc29, Doc30, Doc31
  static constexpr uint32_t C_MASK = static_cast<uint32_t>(1) << 31;
  return (bloom_index_bit_mask & (C_MASK >> (pos % 32))) != 0;
}

// NOLINTNEXTLINE:
BLOOM_INDEX_DEVICE_HOST_INLINE bool get_bit_64_bit_mask(
    uint64_t bloom_index_bit_mask,
    int64_t pos) {
  //     bloom index bit mask output.
  //     lower doc id put at higher bits in unit64_t, for example:
  //     |< high                                       low >|
  //      Doc0, Doc1, Doc2 ...... Doc 60, Doc61, Doc62, Doc63
  static constexpr uint64_t C_MASK = 1ULL << 63;
  return (bloom_index_bit_mask & (C_MASK >> (pos % 64))) != 0;
}

// NOLINTNEXTLINE:
BLOOM_INDEX_DEVICE_HOST_INLINE bool get_bit_64_bit_mask(
    const uint64_t* bloom_index_bit_mask,
    int64_t pos) {
  return get_bit_64_bit_mask(*(bloom_index_bit_mask + pos / 64), pos);
}

// NOLINTNEXTLINE:
BLOOM_INDEX_DEVICE_HOST_INLINE uint32_t
get_next_32_bit_mask(const uint64_t* bloom_index_bit_mask, int64_t pos) {
  //     bloom index bit mask output.
  //     lower doc id put at higher bits in unit64_t, for example:
  //     |< high                                       low >|
  //      Doc0, Doc1, Doc2 ...... Doc 60, Doc61, Doc62, Doc63
  static constexpr uint64_t C_MASK = UINT64_MAX;

  int64_t pos_in_uint64 = (pos % 64);
  const uint64_t* target_bit_mask_ptr = bloom_index_bit_mask + pos / 64;

  if (pos_in_uint64 <= 32) {
    return static_cast<uint32_t>(
        (*target_bit_mask_ptr & (C_MASK >> pos_in_uint64)) >>
        (32 - pos_in_uint64));
  }

  // bits spread on two uint64_t.
  uint32_t return_value =
      static_cast<uint32_t>(*target_bit_mask_ptr & (C_MASK >> pos_in_uint64))
      << (pos_in_uint64 - 32);
  return_value |=
      static_cast<uint32_t>(*(target_bit_mask_ptr + 1) >> (96 - pos_in_uint64));
  return return_value;
}

// NOLINTNEXTLINE:
template <typename B_INDEX_TYPE>
BLOOM_INDEX_DEVICE_HOST_INLINE void assign_one_bits_position(
    int64_t feature_id,
    int64_t feature_value,
    int64_t b,
    int64_t k,
    B_INDEX_TYPE* doc_one_bits_positions) {
  uint64_t i = 0;
  uint64_t seed = 0;
  while (i < k) {
    uint64_t hash = murmur_hash3_2x64(feature_id, feature_value, seed++);
    uint64_t b_p = hash % b;
    bool duplicate_b_p = false;
    for (uint64_t j = 0; j < i; ++j) {
      if (b_p == static_cast<uint64_t>(doc_one_bits_positions[j])) {
        duplicate_b_p = true;
      }
    }
    if (duplicate_b_p) {
      continue;
    }

    doc_one_bits_positions[i++] = static_cast<B_INDEX_TYPE>(b_p);
  }
}

// NOLINTNEXTLINE:
BLOOM_INDEX_DEVICE_HOST_INLINE void prefetch_one_bits_hashes(
    int64_t feature_id,
    int64_t feature_value,
    int64_t relaxed_k,
    typename QueryPlanOneBitsPType<true>::type* doc_one_bits_hashes) {
  uint64_t i = 0;
  uint64_t seed = 0;
  while (i < relaxed_k) {
    doc_one_bits_hashes[i++] =
        murmur_hash3_2x64(feature_id, feature_value, seed++);
  }
}

// NOLINTNEXTLINE:
template <typename B_INDEX_TYPE>
BLOOM_INDEX_DEVICE_HOST_INLINE void assign_document_signature(
    int64_t feature_id,
    int64_t feature_value,
    int64_t b,
    int64_t k,
    int8_t* document_signature_ptr) {
  std::array<B_INDEX_TYPE, MAX_K> used_b_p = {0};
  assign_one_bits_position<B_INDEX_TYPE>(
      feature_id, feature_value, b, k, used_b_p.data());
  for (int64_t i = 0; i < k; ++i) {
    set_bit(static_cast<int64_t>(used_b_p[i]), document_signature_ptr);
  }
}

BLOOM_INDEX_DEVICE_HOST_INLINE bool is_document_valid_with_full_mask(
    const uint64_t* filtering_bitmask_ptr,
    const int64_t* filtering_bit_mask_index_ptr,
    const int64_t filtering_bitmask_column_size,
    const int64_t query_id,
    const int64_t item_id) {
  if (filtering_bitmask_ptr != nullptr) {
    int64_t row = filtering_bit_mask_index_ptr == nullptr
        ? query_id
        : filtering_bit_mask_index_ptr[query_id];
    return get_bit_64_bit_mask(
        filtering_bitmask_ptr
            [row * filtering_bitmask_column_size + item_id / C_BITS_IN_UINT64],
        item_id);
  }
  return true;
}

BLOOM_INDEX_DEVICE_HOST_INLINE bool is_document_valid_with_partial_mask(
    const int32_t* column_counts_cumsum_ptr,
    const int8_t* first_item_offset_in_column_ptr,
    const uint64_t* column_masks_ptr,
    const int64_t cluster_index,
    const int64_t doc_offset_in_cluster) {
  int64_t cluster_start_column_index =
      cluster_index == 0 ? 0 : column_counts_cumsum_ptr[cluster_index - 1];
  return get_bit_64_bit_mask(
      column_masks_ptr + cluster_start_column_index,
      first_item_offset_in_column_ptr[cluster_index] + doc_offset_in_cluster);
}

#undef BLOOM_INDEX_DEVICE_HOST_INLINE

// bloom_index ops declarations:
/*
 * Computes masks indicating what are the bloom index columns that the selected
 *   clusters are going to touch.
 *
 * @param bloom_index
 *    The bloom index that we are going to query.
 *
 * @param selected_cluster_offsets
 *    A [Batches x Queries] 2-D tensor representing the KNN clusters being
 * selected.
 *
 * @param selected_cluster_lengths
 *    A [Batches x Queries] 2-D tensor containing lengths for each selected
 * clusters.
 *
 * @returns
 *    A 1-D tensor that has the same number of elements as the number of columns
 *    in the bloom index. Each element can be interpreted as boolean to tell if
 *    the column is being selected in the query or not.
 */
// NOLINTNEXTLINE:
at::Tensor generate_bloom_column_mask_for_selected_clusters(
    const at::Tensor& bloom_index,
    const at::Tensor& selected_cluster_offsets,
    const at::Tensor& selected_cluster_lengths);

/*
 * This function generations column_counts, start_column_ids,
 * first_item_offsets_in_column based on the selected clusters information.
 */
// NOLINTNEXTLINE:
std::tuple<at::Tensor, at::Tensor, at::Tensor>
generate_column_info_for_clusters(
    const at::Tensor& selected_cluster_offsets,
    const at::Tensor& selected_cluster_lengths);

/*
 * generate following 4 tensors from the inputs:
 *   column_counts: column count for each cluster
 *   start_column_ids: start column id for each cluster
 *   start_column_carryovers: how many carryover bits in first col for
 *                            each cluster
 *   end_column_remains: how many remaining bits in last col for
 *                       each cluster
 * this is for co-design with Knn for jagged data flow.
 * these output tensors will be used to generate mask information
 * which will be used for knn jagged output kernel.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
// NOLINTNEXTLINE:
generate_cluster_column_info_for_jagged_flow(
    const at::Tensor& selected_cluster_offsets,
    const at::Tensor& selected_cluster_lengths);

// NOLINTNEXTLINE:
void mask_marginal_bits_from_column_response(
    at::Tensor& column_masks,
    at::Tensor& column_bit_set_counts,
    const at::Tensor& column_counts_cumsum,
    const at::Tensor& start_column_carryovers,
    const at::Tensor& end_column_remains);
} // namespace bloom_index
} // namespace ops
} // namespace st
