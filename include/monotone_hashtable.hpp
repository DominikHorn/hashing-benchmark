#pragma once

#include <immintrin.h>

#include <array>
#include <cstddef>
#include <learned_hashing.hpp>
#include <limits>
#include <string>

#include "include/convenience/builtins.hpp"
#include "include/support.hpp"

namespace masters_thesis {
template <class Key, class Payload, size_t BucketSize,
          class Model = learned_hashing::MonotoneRMIHash<Key, 1000000>,
          Key Sentinel = std::numeric_limits<Key>::max()>
class MonotoneHashtable {
  struct Bucket {
    std::array<Key, BucketSize> keys;
    std::array<Payload, BucketSize> payloads;
    Bucket* next = nullptr;

    Bucket() {
      // Sentinel value in each slot per default
      std::fill(keys.begin(), keys.end(), Sentinel);
    }

    void insert(const Key& key, const Payload& payload) {
      Bucket* previous = this;

      for (Bucket* current = previous; current != nullptr;
           current = current->next) {
        for (size_t i = 0; i < BucketSize; i++) {
          if (current->keys[i] == Sentinel) {
            current->keys[i] = key;
            current->payloads[i] = payload;
            return;
          }
        }

        previous = current;
      }

      // static var will be shared by all instances
      static support::Tape<Bucket> tape{};

      previous->next = tape.alloc();
      previous->next->insert(key, payload);
    }

    size_t byte_size() const {
      return sizeof(Bucket) + (next != nullptr ? next->byte_size() : 0);
    }
  };
  /// directory of buckets
  std::vector<Bucket> buckets;

  /// model for predicting the correct index
  const Model model;

 public:
  /**
   * Constructs a MonotoneHashtable given a *sorted* list of keys
   * together with their corresponding payloads
   */
  MonotoneHashtable(const std::vector<Key>& keys,
                    const std::vector<Payload>& payloads)
      : buckets(1 + keys.size() / BucketSize),
        model(keys.begin(), keys.end(), buckets.size()) {
    // insert all keys according to model
    for (size_t i = 0; i < keys.size(); i++) insert(keys[i], payloads[i]);
  }

  forceinline void insert(const Key& key, const Payload& payload) {
    const auto index = model(key);
    assert(index >= 0);
    assert(index < buckets.size());
    buckets[index].insert(key, payload);
  }

  forceinline Payload lookup(const Key& key) const {
    const size_t directory_ind = model(key);

    // since BucketSize is a template arg and therefore compile-time static,
    // compiler will recognize that all branches of this if/else but one can be
    // eliminated during optimization, therefore making this a 0 runtime cost
    // specialization
#ifdef __AVX512F__
    if constexpr (BucketSize == 8 && sizeof(Key) == 8) {
      for (auto bucket = &buckets[directory_ind]; bucket != nullptr;) {
        auto vkey = _mm512_set1_epi64(key);
        auto vbucket = _mm512_loadu_si512((const __m512i*)&bucket->keys);
        auto mask = _mm512_cmpeq_epi64_mask(vkey, vbucket);

        if (mask != 0) {
          int index = __builtin_ctz(mask);
          assert(index >= 0);
          assert(index < BucketSize);
          return bucket->payloads[index];
        }

        bucket = bucket->next;
      }

      return std::numeric_limits<Payload>::max();
    }
#endif
#ifdef __AVX2__
    if constexpr (BucketSize == 8 && sizeof(Key) == 4) {
      for (auto bucket = &buckets[directory_ind]; bucket != nullptr;) {
        auto vkey = _mm256_set1_epi32(key);
        auto vbucket = _mm256_loadu_si256((const __m256i*)&bucket->keys);
        auto cmp = _mm256_cmpeq_epi32(vkey, vbucket);
        int mask = _mm256_movemask_epi8(cmp);
        if (mask != 0) {
          int index = __builtin_ctz(mask) >> 2;
          assert(index >= 0);
          assert(index < BucketSize);
          return bucket->payloads[index];
        }

        bucket = bucket->next;
      }
      return std::numeric_limits<Payload>::max();
    }
    if constexpr (BucketSize == 4 && sizeof(Key) == 8) {
      for (auto bucket = &buckets[directory_ind]; bucket != nullptr;) {
        auto vkey = _mm256_set1_epi64x(key);
        auto vbucket = _mm256_loadu_si256((const __m256i*)&bucket->keys);
        auto cmp = _mm256_cmpeq_epi64(vkey, vbucket);
        int mask = _mm256_movemask_epi8(cmp);
        if (mask != 0) {
          int index = __builtin_ctz(mask) >> 3;
          assert(index >= 0);
          assert(index < BucketSize);
          return bucket->payloads[index];
        }

        bucket = bucket->next;
      }
      return std::numeric_limits<Payload>::max();
    }
#endif

    // Generic non-SIMD algorithm. Note that a smart compiler might vectorize
    // this nested loop construction anyways.
    for (auto bucket = &buckets[directory_ind]; bucket != nullptr;) {
      for (size_t i = 0; i < BucketSize; i++) {
        const auto& current_key = bucket->keys[i];
        if (current_key == Sentinel) break;
        if (current_key == key) return bucket->payloads[i];
      }
      bucket = bucket->next;
    }

    return std::numeric_limits<Payload>::max();
  }

  std::string name() {
    return "MonotoneHashtable<" + std::to_string(sizeof(Key)) + ", " +
           std::to_string(sizeof(Payload)) + ", " + std::to_string(BucketSize) +
           ", " + model.name() + ">";
  }

  size_t directory_byte_size() const {
    size_t directory_bytesize = sizeof(decltype(buckets));
    for (const auto& bucket : buckets) directory_bytesize += bucket.byte_size();
    return directory_bytesize;
  }

  size_t model_byte_size() const { return model.byte_size(); }

  size_t byte_size() const { return model_byte_size() + directory_byte_size(); }
};
}  // namespace masters_thesis
