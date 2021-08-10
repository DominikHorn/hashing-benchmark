#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <hashing.hpp>
#include <hashtable.hpp>
#include <iostream>
#include <learned_hashing.hpp>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "include/convenience/builtins.hpp"
#include "support/datasets.hpp"

namespace _ {
using Key = std::uint64_t;
using Payload = std::uint64_t;

const std::vector<std::int64_t> intervals{
    1 /*, 2, 4, 8, 16, 32, 64, 128, 256*/};
const size_t gen_dataset_size = 100000000;
const std::vector<std::int64_t> datasets{
    dataset::ID::SEQUENTIAL, dataset::ID::GAPPED_10, dataset::ID::UNIFORM,
    dataset::ID::FB, /*dataset::ID::OSM,*/ dataset::ID::WIKI};

std::random_device rd;
std::default_random_engine rng(rd());

static void SortedArrayRangeLookupBinarySearch(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  const auto did = static_cast<dataset::ID>(state.range(1));
  auto dataset = dataset::load_cached(did, gen_dataset_size);

  state.counters["dataset_size"] = dataset.size();
  state.SetLabel(dataset::name(did));

  if (dataset.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);
  for (auto _ : state) {
    const auto lower = dataset[dist(rng)];
    const auto upper = lower + interval_size;

    std::vector<Payload> result;
    for (auto iter = std::lower_bound(dataset.begin(), dataset.end(), lower);
         iter < dataset.end() && *iter < upper; iter++)
      result.push_back(*iter - 1);

    benchmark::DoNotOptimize(result.data());
  }
}

template <size_t SecondLevelModelCount>
static void SortedArrayRangeLookupRMI(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  const auto did = static_cast<dataset::ID>(state.range(1));

  std::cout << "(0) loading dataset" << std::endl;
  auto dataset = dataset::load_cached(did, gen_dataset_size);

  std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);

  state.counters["dataset_size"] = dataset.size();
  state.SetLabel(dataset::name(did));

  if (dataset.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  std::cout << "(1) sampling data" << std::endl;

  std::vector<decltype(dataset)::value_type> sample(dataset.size() / 100);
  for (size_t i = 0; i < sample.size(); i++) sample[i] = dataset[dist(rng)];
  sample.push_back(*std::min_element(dataset.begin(), dataset.end()));
  sample.push_back(*std::max_element(dataset.begin(), dataset.end()));
  dataset::deduplicate_and_sort(sample);

  std::cout << "(2) building rmi" << std::endl;
  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi(
      sample.begin(), sample.end(), dataset.size());

  std::cout << "(3) finding max error" << std::endl;

  // determine maximum model error
  size_t max_error = 0;
  size_t notify_at = dataset.size() / 100;
  for (size_t i = 0; i < dataset.size(); i++) {
    const auto pred = rmi(dataset[i]);

    max_error = std::max(max_error, pred > i ? pred - i : i - pred);

    if (i % notify_at == 0) std::cout << "." << std::flush;
  }
  std::cout << std::endl;

  std::cout << "\t-> max_error: " << max_error << std::endl
            << "(4) benchmarking" << std::endl;

  for (auto _ : state) {
    const auto lower = dataset[dist(rng)];
    const auto upper = lower + interval_size;

    std::vector<Payload> result;

    // Interval is determined by max error
    const auto pred_ind = rmi(lower);
    const auto begin_iter =
        dataset.begin() + (pred_ind > max_error) * (pred_ind - max_error);
    const auto end_iter =
        dataset.begin() + std::min(pred_ind + max_error, dataset.size() - 1);

    for (auto iter = std::lower_bound(begin_iter, end_iter, lower);
         iter < dataset.end() && *iter < upper; iter++)
      result.push_back(*iter - 1);

    benchmark::DoNotOptimize(result.data());
  }

  std::cout << "\t-> done" << std::endl;
}

const Key Sentinel = std::numeric_limits<Key>::max();
template <size_t BucketSize>
struct Bucket {
  std::array<Key, BucketSize> keys;
  Bucket* next = nullptr;

  struct Tape {
    std::vector<Bucket*> begins;
    size_t index;
    size_t size;

    ~Tape() {
      for (auto begin : begins) delete[] begin;
    }

    forceinline Bucket* new_bucket(size_t tape_size = 1000000) {
      if (unlikely(index == size || begins.size() == 0 ||
                   begins[begins.size() - 1] == nullptr)) {
        begins.push_back(new Bucket[tape_size]);
        index = 0;
        size = tape_size;
      }

      return &begins[begins.size() - 1][index++];
    }
  };

  Bucket() {
    // Sentinel value in each slot per default
    std::fill(keys.begin(), keys.end(), Sentinel);
  }

  forceinline void insert(const Key& key, Tape& tape) {
    Bucket* previous = this;

    for (Bucket* current = previous; current != nullptr;
         current = current->next) {
      for (size_t i = 0; i < BucketSize; i++) {
        if (current->keys[i] == Sentinel) {
          current->keys[i] = key;
          return;
        }
      }

      previous = current;
    }

    previous->next = tape.new_bucket();
    previous->next->insert(key, tape);
  }
} packit;

template <size_t SecondLevelModelCount, size_t BucketSize>
static void BucketsRangeLookupRMI(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  const auto did = static_cast<dataset::ID>(state.range(1));

  std::cout << "(0) loading dataset" << std::endl;
  auto dataset = dataset::load_cached(did, gen_dataset_size);

  std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);

  state.counters["dataset_size"] = dataset.size();
  state.SetLabel(dataset::name(did));

  if (dataset.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  std::cout << "(1) sampling data" << std::endl;
  std::vector<decltype(dataset)::value_type> sample(dataset.size() / 100);
  for (size_t i = 0; i < sample.size(); i++) sample[i] = dataset[dist(rng)];
  sample.push_back(*std::min_element(dataset.begin(), dataset.end()));
  sample.push_back(*std::max_element(dataset.begin(), dataset.end()));
  dataset::deduplicate_and_sort(sample);

  std::vector<Bucket<BucketSize>> buckets(
      dataset.size());  // TODO: load factors?

  std::cout << "(2) building rmi" << std::endl;
  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi(
      sample.begin(), sample.end(), buckets.size());
  std::cout << "(3) inserting keys" << std::endl;

  // insert all keys exactly where model tells us to
  size_t notify_at = dataset.size() / 100;
  typename Bucket<BucketSize>::Tape tape;
  for (size_t i = 0; i < dataset.size(); i++) {
    const auto key = dataset[i];
    const auto ind = rmi(key);
    buckets[ind].insert(key, tape);

    if (i % notify_at == 0) std::cout << "." << std::flush;
  }
  std::cout << std::endl;

  std::cout << "(4) benchmarking" << std::endl;
  for (auto _ : state) {
    const auto lower = dataset[dist(rng)];
    const auto upper = lower + interval_size;

    std::vector<Payload> result;

    bool upper_encountered = false;
    for (size_t bucket_ind = rmi(lower);
         !upper_encountered && bucket_ind < buckets.size(); bucket_ind++) {
      for (auto* b = &buckets[bucket_ind]; b != nullptr; b = b->next) {
        for (size_t i = 0; i < BucketSize; i++) {
          const auto& current_key = b->keys[i];
          if (current_key == Sentinel) {
            b = nullptr;
            break;
          }
          if (current_key >= upper) {
            upper_encountered = true;
            continue;  // Don't assume data was inserted in sorted order
          }
          if (current_key < lower)
            continue;  // Don't assume data was inserted in sorted order

          result.push_back(current_key - 1);
        }

        if (b == nullptr) break;
      }
    }

    assert(result.size() <= interval_size);

    benchmark::DoNotOptimize(result.data());
  }
  std::cout << "\t-> done" << std::endl;
}

#define __BENCHMARK_TWO_PARAM(fun, model_size, bucket_size) \
  BENCHMARK_TEMPLATE(fun, model_size, bucket_size)          \
      ->ArgsProduct({intervals, datasets});

#define _BENCHMARK_TWO_PARAM(fun, model_size) \
  __BENCHMARK_TWO_PARAM(fun, model_size, 1)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 2)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 8)
#define BENCHMARK_TWO_PARAM(fun)  \
  _BENCHMARK_TWO_PARAM(fun, 10)   \
  _BENCHMARK_TWO_PARAM(fun, 1000) \
  _BENCHMARK_TWO_PARAM(fun, 100000)

// BENCHMARK(SortedArrayRangeLookupBinarySearch)
//    ->ArgsProduct({intervals, datasets});

BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMI, 10)
    ->ArgsProduct({intervals, datasets});
BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMI, 1000)
    ->ArgsProduct({intervals, datasets});
BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMI, 100000)
    ->ArgsProduct({intervals, datasets});

BENCHMARK_TWO_PARAM(BucketsRangeLookupRMI);
}  // namespace _

