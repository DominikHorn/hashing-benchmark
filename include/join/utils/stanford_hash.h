#pragma once

// This is a slightly modified version of
// hashing.cc from the Stanford FutureData index baselines repo.
// Original copyright:  Copyright (c) 2017-present Peter Bailis, Kai Sheng Tai, Pratiksha Thaker, Matei Zaharia
// MIT License

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <random>
#include <vector>

#include "utils/math.h"

const uint32_t INVALID_KEY = 0xffffffff;     // An unusable key we'll treat as a sentinel value
const uint32_t bckt_size = 8;              // Bucket size for cuckoo hash
const double LOAD_FACTOR = 0.99;             // Load factor used for hash tables

// Finalization step of Murmur3 hash
uint32_t hash32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x85ebca6b;
  value ^= value >> 13;
  value *= 0xc2b2ae35;
  value ^= value >> 16;
  return value;
}

// Fast alternative to modulo from Daniel Lemire
//uint32_t alt_mod(uint32_t x, uint32_t n) {
//  return ((uint64_t) x * (uint64_t) n) >> 32 ;
//}

// A bucketed cuckoo hash map with keys of type uint32_t and values of type V
template <typename V>
class CuckooHashMap {
public:
  struct SearchResult {
    bool found;
    V value;
  };

private:
  struct Bucket {
    uint32_t keys[bckt_size] __attribute__((aligned(32)));
    V values[bckt_size];
  };

  Bucket *buckets_;
  uint32_t num_buckets_;  // Total number of buckets
  uint32_t size_;         // Number of entries filled
  std::mt19937 rand_;          // RNG for moving items around
  V uninitialized_value_;

public:
  CuckooHashMap(uint32_t capacity): size_(0) {
    num_buckets_ = (capacity + bckt_size - 1) / bckt_size;
    int r = posix_memalign((void **) &buckets_, 32, num_buckets_ * sizeof(Bucket));
    if (r != 0) cout << "could not memalign in cuckoo hash map \n";
    for (uint32_t i = 0; i < num_buckets_; i++) {
      for (size_t j = 0; j < bckt_size; j++) {
        buckets_[i].keys[j] = INVALID_KEY;
      }
    }
  }

  ~CuckooHashMap() {
    free(buckets_);
  }

  SearchResult get(uint32_t key) const {
    uint32_t hash = hash32(key);
    uint32_t i1 = alt_mod(hash, num_buckets_);
    Bucket *b1 = &buckets_[i1];

    __m256i vkey = _mm256_set1_epi32(key);
    __m256i vbucket = _mm256_load_si256((const __m256i *) &b1->keys);
    __m256i cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    int mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      return { true, b1->values[index] };
    }

    uint32_t i2 = alt_mod(hash32(key ^ hash), num_buckets_);
    if (i2 == i1) {
      i2 = (i1 == num_buckets_ - 1) ? 0 : i1 + 1;
    }
    Bucket *b2 = &buckets_[i2];

    vbucket = _mm256_load_si256((const __m256i *) &b2->keys);
    cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      return { true, b2->values[index] };
    }

    return { false, uninitialized_value_ };
  }

  void insert(uint32_t key, V value) {
    insert(key, value, false);
  }

  uint32_t size() {
    return size_;
  }

  uint64_t size_bytes() const {
    return num_buckets_ * sizeof(Bucket);
  }

private:
  // Insert a key into the table if it's not already inside it;
  // if this is a re-insert, we won't increase the size_ field.
  void insert(uint32_t key, V value, bool is_reinsert) {
    uint32_t hash = hash32(key);
    uint32_t i1 = alt_mod(hash, num_buckets_);
    uint32_t i2 = alt_mod(hash32(key ^ hash), num_buckets_);
    if (i2 == i1) {
      i2 = (i1 == num_buckets_ - 1) ? 0 : i1 + 1;
    }

    Bucket *b1 = &buckets_[i1];
    Bucket *b2 = &buckets_[i2];

    // Update old value if the key is already in the table
    __m256i vkey = _mm256_set1_epi32(key);
    __m256i vbucket = _mm256_load_si256((const __m256i *) &b1->keys);
    __m256i cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    int mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      b1->values[index] = value;
      return;
    }

    vbucket = _mm256_load_si256((const __m256i *) &b2->keys);
    cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      b2->values[index] = value;
      return;
    }

    if (!is_reinsert) {
      size_++;
    }

    size_t count1 = 0;
    for (size_t i = 0; i < bckt_size; i++) {
      count1 += (b1->keys[i] != INVALID_KEY ? 1 : 0);
    }
    size_t count2 = 0;
    for (size_t i = 0; i < bckt_size; i++) {
      count2 += (b2->keys[i] != INVALID_KEY ? 1 : 0);
    }

    if (count1 <= count2 && count1 < bckt_size) {
      // Add it into bucket 1
      b1->keys[count1] = key;
      b1->values[count1] = value;
    } else if (count2 < bckt_size) {
      // Add it into bucket 2
      b2->keys[count2] = key;
      b2->values[count2] = value;
    } else {
      // Both buckets are full; evict a random item from one of them
      assert(count1 == bckt_size);
      assert(count2 == bckt_size);

      Bucket *victim_bucket = b1;
      if (rand_() % 2 == 0) {
        victim_bucket = b2;
      }
      uint32_t victim_index = rand_() % bckt_size;
      uint32_t old_key = victim_bucket->keys[victim_index];
      V old_value = victim_bucket->values[victim_index];
      victim_bucket->keys[victim_index] = key;
      victim_bucket->values[victim_index] = value;
      insert(old_key, old_value, true);
    }
  }
};
