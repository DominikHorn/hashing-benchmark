#pragma once

#include <immintrin.h>

#include <array>
#include <cstddef>
#include <learned_hashing.hpp>
#include <limits>

// TODO: this include is fishy/magic
#include "include/convenience/builtins.hpp"

template <class Key, class Payload, size_t BucketSize,
          size_t SecondLevelModelCount,
          Key Sentinel = std::numeric_limits<Key>::max()>
struct RMIHashtable {
  struct Tape;

  struct Bucket {
    std::array<Key, BucketSize> keys;
    std::array<Payload, BucketSize> payloads;
    Bucket* next = nullptr;

    Bucket() {
      // Sentinel value in each slot per default
      std::fill(keys.begin(), keys.end(), Sentinel);
    }

    forceinline size_t byte_size() const {
      return sizeof(Bucket) + (next != nullptr ? next->byte_size() : 0);
    }

    void insert(const Key& key, const Payload& payload, Tape& tape) {
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

      previous->next = tape.new_bucket();
      previous->next->insert(key, payload, tape);
    }
  };

  struct Tape {
    std::vector<Bucket*> begins;
    size_t index;
    size_t size;

    ~Tape() {
      for (auto begin : begins) delete[] begin;
    }

    forceinline Bucket* new_bucket(size_t tape_size = 10000000) {
      if (unlikely(index == size || begins.size() == 0 ||
                   begins[begins.size() - 1] == nullptr)) {
        begins.push_back(new Bucket[tape_size]);
        index = 0;
        size = tape_size;
      }

      return &begins[begins.size() - 1][index++];
    }
  };

  RMIHashtable(const std::vector<Key>& dataset,
               const std::vector<Payload>& payloads)
      : buckets(bucket_cnt(dataset)),
        rmi(dataset.begin(), dataset.end(), bucket_cnt(dataset)) {
    const size_t notify_at = dataset.size() / 100;
    size_t notify = 0;
    // insert all keys exactly where model tells us to
    for (size_t i = 0; i < dataset.size(); i++) {
      const auto& key = dataset[i];
      const auto& payload = payloads[i];
      insert(key, payload);

      notify++;
      if (notify >= notify_at) {
        std::cout << "." << std::flush;
        notify -= notify_at;
      }
    }
    std::cout << " ";
  }

  forceinline void insert(const Key& key, const Payload& payload) {
    const auto index = rmi(key);
    assert(index >= 0);
    assert(index < buckets.size());

    buckets[index].insert(key, payload, tape);
  }

  forceinline Payload lookup(const Key& key) const {
    // since BucketSize is a template arg and therefore compile-time static,
    // compiler will hopefully recognize that all branches of this if/else but
    // one can be eliminated during optimization, therefore allowing for 0 cost
    // specialization/simdiization
#ifdef __AVX512F__
    if (BucketSize == 8) {
      for (auto bucket = &buckets[rmi(key)]; bucket != nullptr;) {
        // prefetch the next bucket before examining keys to hide latency
        // of traversing the bucket chain
        prefetchit(bucket->next, 0, 3);

        __m512i vkey = _mm512_set1_epi64(key);
        __m512i vbucket = _mm512_load_si512((const __m512i*)&bucket->keys);
        auto mask = _mm512_cmpeq_epi64_mask(vkey, vbucket);

        if (mask != 0) {
          int index = __builtin_ctz(mask);
          assert(index >= 0);
          assert(index < BucketSize);
          return bucket->payloads[index];
        }

        bucket = bucket->next;
      }
#else
#warning "Missing AVX512 support for vectorized BucketSize=8 lookups"
    if (false) {
#endif
    } else {
      const auto buckets_size = buckets.size();

      // all keys are placed exactly where model tells us
      for (size_t bucket_ind = rmi(key); bucket_ind < buckets_size;
           bucket_ind++) {
        for (auto bucket = &buckets[bucket_ind]; bucket != nullptr;) {
          // prefetch the next bucket before examining keys to hide latency
          // of traversing the bucket chain
          prefetchit(bucket->next, 0, 3);

          for (size_t i = 0; i < BucketSize; i++) {
            const auto& current_key = bucket->keys[i];
            if (current_key == Sentinel) break;
            if (current_key == key) {
              return bucket->payloads[i];
            }
          }
          bucket = bucket->next;
        }
      }
    }

    return std::numeric_limits<Payload>::max();
  }

  forceinline size_t size() const { return buckets.size(); }

  forceinline size_t directory_byte_size() const {
    size_t directory_bytesize = sizeof(buckets);
    for (const auto& bucket : buckets) directory_bytesize += bucket.byte_size();
    return directory_bytesize;
  }

  forceinline size_t rmi_byte_size() const { return rmi.byte_size(); }

  forceinline size_t byte_size() const {
    return rmi_byte_size() + directory_byte_size();
  }

 private:
  std::vector<Bucket> buckets;
  Tape tape;
  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi;

  template <class T>
  static size_t bucket_cnt(const std::vector<T>& dataset) {
    return 1 + dataset.size() / BucketSize;
  }
};
