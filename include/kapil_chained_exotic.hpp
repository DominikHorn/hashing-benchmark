#pragma once


#include <exotic_hashing.hpp>
#include <learned_hashing.hpp>

// Order is important
#include "include/convenience/builtins.hpp"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <learned_hashing.hpp>
#include <limits>
#include <string>
#include <utility>

#include "include/convenience/builtins.hpp"
#include "include/support.hpp"

namespace masters_thesis {
template <class Key, class Payload, size_t BucketSize,
          class MMPHF,
          bool ManualPrefetch = false,
          Key Sentinel = std::numeric_limits<Key>::max()>
class KapilChainedExoticHashTable {
    
    public:
     MMPHF mmphf;
    //   const HashFn hashfn;  

  struct Bucket {
    std::array<Key, BucketSize> keys;
    std::array<Payload, BucketSize> payloads;
    Bucket* next = nullptr;

    Bucket() {
      // Sentinel value in each slot per default
      std::fill(keys.begin(), keys.end(), Sentinel);
    }

    void insert(const Key& key, const Payload& payload,
                support::Tape<Bucket>& tape) {
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
      previous->next = tape.alloc();
      previous->next->insert(key, payload, tape);
    }

    size_t byte_size() const {
      return sizeof(Bucket) + (next != nullptr ? next->byte_size() : 0);
    }
  };

  /// directory of buckets
  std::vector<Bucket> buckets;

  /// mmphf for predicting the correct index
//   Model mmphf;

  /// allocator for buckets
  std::unique_ptr<support::Tape<Bucket>> tape;

  /**
   * Inserts a given (key,payload) tuple into the hashtable.
   *
   * Note that this function has to be private for now since
   * mmphf retraining will be necessary if this is used as a
   * construction interface.
   */
  forceinline void insert(const Key& key, const Payload& payload) {
    const auto index = mmphf(key);
    assert(index >= 0);
    assert(index < buckets.size());
    buckets[index].insert(key, payload, *tape);
  }

 public:
  KapilChainedExoticHashTable() = default;

  /**
   * Constructs a KapilChainedExoticHashTable given a list of keys
   * together with their corresponding payloads
   */
  KapilChainedExoticHashTable(std::vector<std::pair<Key, Payload>> data)
      : buckets((1 + data.size()*1.0) / BucketSize),
        tape(std::make_unique<support::Tape<Bucket>>()) {
    // ensure data is sorted
    std::sort(data.begin(), data.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // obtain list of keys -> necessary for mmphf training
    std::vector<Key> keys;
    keys.reserve(data.size());
    std::transform(data.begin(), data.end(), std::back_inserter(keys),
                   [](const auto& p) { return p.first; });

    // train mmphf on sorted data
    mmphf.construct(keys.begin(), keys.end());

    // insert all keys according to mmphf prediction.
    // since we sorted above, this will permit further
    // optimizations during lookup etc & enable implementing
    // efficient iterators in the first place.
    for (const auto& d : data) insert(d.first, d.second);
  }

  class Iterator {
    size_t directory_ind, bucket_ind;
    Bucket const* current_bucket;
    const KapilChainedExoticHashTable& hashtable;

    Iterator(size_t directory_ind, size_t bucket_ind, Bucket const* bucket,
             const KapilChainedExoticHashTable& hashtable)
        : directory_ind(directory_ind),
          bucket_ind(bucket_ind),
          current_bucket(bucket),
          hashtable(hashtable) {}

   public:
    forceinline const Key& key() const {
      assert(current_bucket != nullptr);
      return current_bucket->keys[bucket_ind];
    }

    forceinline const Payload& payload() const {
      assert(current_bucket != nullptr);
      return current_bucket->payloads[bucket_ind];
    }

    forceinline Iterator& operator++() {
      assert(current_bucket != nullptr);

      // prefetch next bucket during iteration. If there is no
      // next bucket, then prefetch the next directory slot
      if (current_bucket->next != nullptr)
        prefetch(current_bucket->next, 0, 0);
      else if (directory_ind + 1 < hashtable.buckets.size())
        prefetch(&hashtable.buckets[directory_ind + 1], 0, 0);

      // since data was inserted in sorted order,
      // simply advancing to next slot does the trick
      bucket_ind++;

      // switch to next bucket
      if (bucket_ind >= BucketSize ||
          current_bucket->keys[bucket_ind] == Sentinel) {
        bucket_ind = 0;
        current_bucket = current_bucket->next;

        // switch to next bucket chain in directory
        if (current_bucket == nullptr &&
            ++directory_ind < hashtable.buckets.size())
          current_bucket = &hashtable.buckets[directory_ind];
      }

      return *this;
    }

    // // TODO(dominik): support postfix increment
    // forceinline Iterator operator++(int) {
    //   Iterator old = *this;
    //   ++this;
    //   return old;
    // }

    forceinline bool operator==(const Iterator& other) const {
      return directory_ind == other.directory_ind &&
             bucket_ind == other.bucket_ind &&
             current_bucket == other.current_bucket &&
             &hashtable == &other.hashtable;
    }

    friend class KapilChainedExoticHashTable;
  };

  /**
   * Past the end iterator, use like usual in stl
   */
  forceinline Iterator end() const {
    return {buckets.size(), 0, nullptr, *this};
  }

  /**
   * Returns an iterator pointing to the payload for a given key
   * or end() if no such key could be found
   *
   * @param key the key to search
   */
  forceinline Iterator operator[](const Key& key) const {
    assert(key != Sentinel);

    // will become NOOP at compile time if ManualPrefetch == false
    // const auto prefetch_next = [](const auto& bucket) {
    //   if constexpr (ManualPrefetch)
    //     // manually prefetch next if != nullptr;
    //     if (bucket->next != nullptr) prefetch(bucket->next, 0, 0);
    // };

    // obtain directory bucket
    const size_t directory_ind = mmphf(key);
    auto bucket = &buckets[directory_ind];


    return {directory_ind, 0, bucket, *this};

    // prefetch_next(bucket);

    // since BucketSize is a template arg and therefore compile-time static,
    // compiler will recognize that all branches of this if/else but one can
    // be eliminated during optimization, therefore making this a 0 runtime
    // cost specialization
// #ifdef __AVX512F__
//     if constexpr (BucketSize == 8 && sizeof(Key) == 8) {
//       while (bucket != nullptr) {
//         auto vkey = _mm512_set1_epi64(key);
//         auto vbucket = _mm512_loadu_si512((const __m512i*)&bucket->keys);
//         auto mask = _mm512_cmpeq_epi64_mask(vkey, vbucket);

//         if (mask != 0) {
//           size_t index = __builtin_ctz(mask);
//           assert(index >= 0);
//           assert(index < BucketSize);
//           return {directory_ind, index, bucket, *this};
//         }

//         bucket = bucket->next;
//         prefetch_next(bucket);
//       }

//       return end();
//     }
// #endif
// #ifdef __AVX2__
//     if constexpr (BucketSize == 8 && sizeof(Key) == 4) {
//       while (bucket != nullptr) {
//         auto vkey = _mm256_set1_epi32(key);
//         auto vbucket = _mm256_loadu_si256((const __m256i*)&bucket->keys);
//         auto cmp = _mm256_cmpeq_epi32(vkey, vbucket);
//         int mask = _mm256_movemask_epi8(cmp);
//         if (mask != 0) {
//           size_t index = __builtin_ctz(mask) >> 2;
//           assert(index >= 0);
//           assert(index < BucketSize);
//           return {directory_ind, index, bucket, *this};
//         }

//         bucket = bucket->next;
//         prefetch_next(bucket);
//       }
//       return end();
//     }
//     if constexpr (BucketSize == 4 && sizeof(Key) == 8) {
//       while (bucket != nullptr) {
//         auto vkey = _mm256_set1_epi64x(key);
//         auto vbucket = _mm256_loadu_si256((const __m256i*)&bucket->keys);
//         auto cmp = _mm256_cmpeq_epi64(vkey, vbucket);
//         int mask = _mm256_movemask_epi8(cmp);
//         if (mask != 0) {
//           size_t index = __builtin_ctz(mask) >> 3;
//           assert(index >= 0);
//           assert(index < BucketSize);
//           return {directory_ind, index, bucket, *this};
//         }

//         bucket = bucket->next;
//         prefetch_next(bucket);
//       }
//       return end();
//     }
// #endif

    // Generic non-SIMD algorithm. Note that a smart compiler might vectorize
    // this nested loop construction anyways.
    while (bucket != nullptr) {
      for (size_t i = 0; i < BucketSize; i++) {
        const auto& current_key = bucket->keys[i];
        if (current_key == Sentinel) break;
        if (current_key == key) return {directory_ind, i, bucket, *this};
      }

      bucket = bucket->next;
    //   prefetch_next(bucket);
    }

    return end();
  }

  std::string name() {
    std::string prefix = (ManualPrefetch ? "Prefetched" : "");
    return prefix + "KapilChainedExoticHashTable<" + std::to_string(sizeof(Key)) + ", " +
           std::to_string(sizeof(Payload)) + ", " + std::to_string(BucketSize) +
           ", " + mmphf.name() + ">";
  }

  size_t directory_byte_size() const {
    size_t directory_bytesize = sizeof(decltype(buckets));
    for (const auto& bucket : buckets) directory_bytesize += bucket.byte_size();
    return directory_bytesize;
  }

  //kapil_change: assuming mmphf size to be zero  
  size_t mmphf_byte_size() const { return mmphf.byte_size(); }

  size_t byte_size() const { return mmphf_byte_size() + directory_byte_size(); }
};
}  // namespace masters_thesis
