#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <hashing.hpp>
#include <hashtable.hpp>
#include <iostream>
#include <learned_hashing.hpp>
#include <limits>
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
    dataset::ID::FB,         dataset::ID::OSM,       dataset::ID::WIKI};

std::random_device rd;
std::default_random_engine rng(rd());

static void BM_ShuffleArray(benchmark::State& state) {
  auto dataset =
      dataset::load_cached(dataset::ID::SEQUENTIAL, gen_dataset_size);
  for (auto _ : state) {
    dataset::shuffle(dataset);
    benchmark::DoNotOptimize(dataset.data());
  }
}

static void BM_ShuffleAndSortArray(benchmark::State& state) {
  auto dataset =
      dataset::load_cached(dataset::ID::SEQUENTIAL, gen_dataset_size);
  for (auto _ : state) {
    dataset::shuffle(dataset);
    std::sort(dataset.begin(), dataset.end());
    benchmark::DoNotOptimize(dataset.data());
  }
}

static void BM_SortedArrayRangeLookupBinarySearch(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  auto dataset = dataset::load_cached(static_cast<dataset::ID>(state.range(1)),
                                      gen_dataset_size);

  if (dataset.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  // We must sort the entire array ;(
  std::sort(dataset.begin(), dataset.end());

  const auto min_key = *std::min_element(dataset.begin(), dataset.end());
  const auto max_key = *std::max_element(dataset.begin(), dataset.end());
  std::uniform_int_distribution<size_t> dist(min_key, max_key);

  for (auto _ : state) {
    const auto lower = dist(rng);
    const auto upper = lower + interval_size;

    std::vector<Payload> result;
    for (auto iter = std::lower_bound(dataset.begin(), dataset.end(), lower);
         iter < dataset.end() && *iter < upper; iter++)
      result.push_back(*iter - 1);

    benchmark::DoNotOptimize(result.data());
  }
}

template <size_t SecondLevelModelCount>
static void BM_SortedArrayRangeLookupRMI(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  const auto dataset = dataset::load_cached(
      static_cast<dataset::ID>(state.range(1)), gen_dataset_size);

  if (dataset.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  const auto min_key = *std::min_element(dataset.begin(), dataset.end());
  const auto max_key = *std::max_element(dataset.begin(), dataset.end());
  std::uniform_int_distribution<size_t> dist(min_key, max_key);

  // build model based on data sample. assume data is random shuffled (which it
  // is) to compactify this code
  std::vector<decltype(dataset)::value_type> sample(
      dataset.begin(), dataset.begin() + dataset.size() / 100);
  std::sort(sample.begin(), sample.end());
  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi(
      sample.begin(), sample.end(), dataset.size() - 1);

  // determine maximum model error
  size_t max_error = 0;
  for (const auto& key : dataset) {
    const auto pred_ind = rmi(key);
    size_t actual_ind = pred_ind;
    while (actual_ind > 0 && dataset[actual_ind] > key) actual_ind--;
    while (actual_ind + 1 < dataset.size() && dataset[actual_ind] < key)
      actual_ind++;

    max_error =
        std::max(max_error, pred_ind > actual_ind ? pred_ind - actual_ind
                                                  : actual_ind - pred_ind);
  }

  for (auto _ : state) {
    const auto lower = dist(rng);
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
}

const Key Sentinel = std::numeric_limits<Key>::max();
template <size_t BucketSize>
struct Bucket {
  std::array<Key, BucketSize> keys;

  Bucket* next = nullptr;

  Bucket() {
    // Sentinel value in each slot per default
    std::fill(keys.begin(), keys.end(), Sentinel);
  }

  forceinline void insert(const Key& key) {
    for (size_t i = 0; i < BucketSize; i++) {
      if (keys[i] == Sentinel) {
        keys[i] = key;
        return;
      }
    }

    if (next == nullptr) next = new Bucket();
    next->insert(key);
  }
} packit;

template <size_t SecondLevelModelCount, size_t BucketSize>
static void BM_BucketsRangeLookupRMI(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  const auto dataset = dataset::load_cached(
      static_cast<dataset::ID>(state.range(1)), gen_dataset_size);

  if (dataset.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  const auto min_key = *std::min_element(dataset.begin(), dataset.end());
  const auto max_key = *std::max_element(dataset.begin(), dataset.end());
  std::uniform_int_distribution<size_t> dist(min_key, max_key);

  std::vector<Bucket<BucketSize>> buckets(
      dataset.size());  // TODO: load factors?

  // build model based on data sample. assume data is random shuffled (which it
  // is) to compactify this code
  std::vector<decltype(dataset)::value_type> sample(
      dataset.begin(), dataset.begin() + dataset.size() / 100);
  std::sort(sample.begin(), sample.end());
  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi(
      sample.begin(), sample.end(), buckets.size() - 1);

  // insert all keys exactly where model tells us to
  for (const auto& key : dataset) buckets[rmi(key)].insert(key);

  for (auto _ : state) {
    const auto lower = dist(rng);
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
}

#define __BENCHMARK_TWO_PARAM(fun, model_size, bucket_size) \
  BENCHMARK_TEMPLATE(fun, model_size, bucket_size)          \
      ->ArgsProduct({intervals, datasets});

#define _BENCHMARK_TWO_PARAM(fun, model_size) \
  __BENCHMARK_TWO_PARAM(fun, model_size, 1)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 2)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 4)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 8)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 16)  \
  __BENCHMARK_TWO_PARAM(fun, model_size, 32)
#define BENCHMARK_TWO_PARAM(fun)     \
  _BENCHMARK_TWO_PARAM(fun, 10)      \
  _BENCHMARK_TWO_PARAM(fun, 100)     \
  _BENCHMARK_TWO_PARAM(fun, 10000)   \
  _BENCHMARK_TWO_PARAM(fun, 1000000) \
  _BENCHMARK_TWO_PARAM(fun, 10000000)

BENCHMARK(BM_ShuffleArray);
BENCHMARK(BM_ShuffleAndSortArray);

BENCHMARK(BM_SortedArrayRangeLookupBinarySearch)
    ->ArgsProduct({intervals, datasets});

BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 10)
    ->ArgsProduct({intervals, datasets});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 100)
    ->ArgsProduct({intervals, datasets});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 10000)
    ->ArgsProduct({intervals, datasets});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 1000000)
    ->ArgsProduct({intervals, datasets});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 10000000)
    ->ArgsProduct({intervals, datasets});

BENCHMARK_TWO_PARAM(BM_BucketsRangeLookupRMI);
}  // namespace _

