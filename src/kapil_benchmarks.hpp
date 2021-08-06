#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <hashing.hpp>
#include <hashtable.hpp>
#include <iostream>
#include <learned_hashing.hpp>
#include <limits>
#include <random>
#include <vector>

namespace _ {
using Key = std::uint64_t;
using Payload = std::uint64_t;

const size_t dataset_size = 1000000;
const size_t interval_min = 1;
const size_t interval_max = dataset_size / 2;
const size_t dataset_min = 1;
const size_t dataset_max = 2;

std::random_device rd;
std::default_random_engine rng(rd());

static std::vector<Key> get_dataset(size_t ind) {
  // TODO: implement SOSD datasets
  static std::vector<Key> ds_uniform_random;
  static std::vector<Key> ds_sequential;

  switch (ind) {
    case 1:
      if (ds_uniform_random.empty()) {
        ds_uniform_random.resize(dataset_size);
        std::uniform_int_distribution<size_t> dist(0, 99);
        for (size_t idx = 0, num = 0; idx < ds_uniform_random.size(); idx++) {
          do num++;
          while (dist(rng) < 90);
          ds_uniform_random[idx] = num;
        }
      }
      return ds_uniform_random;
    case 2:
      if (ds_sequential.empty()) {
        ds_sequential.resize(dataset_size);
        Key k = 2000;
        for (size_t i = 0; i < ds_sequential.size(); i++, k++)
          ds_sequential[i] = k;
      }
      return ds_sequential;
  }

  // TODO: default error (?)
  assert(false);
  return {};
}

static void BM_SortedArrayRangeLookupBinarySearch(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  const auto dataset = get_dataset(state.range(1));

  const auto min_key = dataset[0];
  const auto max_key = dataset[dataset.size() - 1];
  std::uniform_int_distribution<size_t> dist(min_key, max_key);

  for (auto _ : state) {
    const auto lower = dist(rng);
    const auto upper = lower + interval_size;

    std::vector<Payload> result;
    for (auto iter = std::lower_bound(dataset.begin(), dataset.end(), lower);
         iter < dataset.end() && *iter < upper; iter++)
      result.push_back(*iter - 1);  // TODO: actual payload lookup?

    benchmark::DoNotOptimize(result.data());
  }
}

template <size_t SecondLevelModelCount>
static void BM_SortedArrayRangeLookupRMI(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  const auto dataset = get_dataset(state.range(1));

  const auto min_key = dataset[0];
  const auto max_key = dataset[dataset.size() - 1];
  std::uniform_int_distribution<size_t> dist(min_key, max_key);

  // determine maximum model error
  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi(
      dataset.begin(), dataset.end(), dataset.size());
  size_t max_error = 0;
  for (const auto& key : dataset) {
    const auto pred_ind = rmi(key);
    size_t actual_ind = pred_ind;
    while (dataset[actual_ind] > key) actual_ind--;
    while (dataset[actual_ind] < key) actual_ind++;

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
      result.push_back(*iter - 1);  // TODO: actual payload lookup?

    benchmark::DoNotOptimize(result.data());
  }
}

const Key Sentinel = std::numeric_limits<Key>::max();
template <size_t BucketSize>
struct Bucket {
  std::array<Key, BucketSize> keys;
  // std::array<Payload, BucketSize> payloads; // TODO: payloads?

  Bucket* next = nullptr;

  Bucket() {
    // Sentinel value in each slot per default
    std::fill(keys.begin(), keys.end(), Sentinel);
  }

  forceinline void insert(const Key& key) {
    for (size_t i = 0; i < BucketSize; i++) {
      if (keys[i] == Sentinel) {
        keys[i] = key;
        // b->payloads[i] = key-1; // TODO: payloads?
        return;
      }
    }

    if (next == nullptr) next = new Bucket();
    next->insert(key);
  }
};

template <size_t SecondLevelModelCount, size_t BucketSize>
static void BM_BucketsRangeLookupRMI(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  const auto dataset = get_dataset(state.range(1));

  const auto min_key = dataset[0];
  const auto max_key = dataset[dataset.size() - 1];
  std::uniform_int_distribution<size_t> dist(min_key, max_key);

  std::vector<Bucket<BucketSize>> buckets(
      dataset.size());  // TODO: load factors?

  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi(
      dataset.begin(), dataset.end(), dataset.size());

  // insert all keys
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

          result.push_back(current_key - 1);
        }

        if (b == nullptr) break;
      }
    }

    benchmark::DoNotOptimize(result.data());
  }
}

#define __BENCHMARK_TWO_PARAM(fun, model_size, bucket_size) \
  BENCHMARK_TEMPLATE(fun, model_size, bucket_size)          \
      ->RangeMultiplier(2)                                  \
      ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
#define _BENCHMARK_TWO_PARAM(fun, model_size) \
  __BENCHMARK_TWO_PARAM(fun, model_size, 1)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 2)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 4)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 8)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 16)  \
  __BENCHMARK_TWO_PARAM(fun, model_size, 32)  \
  __BENCHMARK_TWO_PARAM(fun, model_size, 64)  \
  __BENCHMARK_TWO_PARAM(fun, model_size, 128) \
  __BENCHMARK_TWO_PARAM(fun, model_size, 256)
#define BENCHMARK_TWO_PARAM(fun)     \
  _BENCHMARK_TWO_PARAM(fun, 10)      \
  _BENCHMARK_TWO_PARAM(fun, 100)     \
  _BENCHMARK_TWO_PARAM(fun, 10000)   \
  _BENCHMARK_TWO_PARAM(fun, 1000000) \
  _BENCHMARK_TWO_PARAM(fun, 10000000)

BENCHMARK(BM_SortedArrayRangeLookupBinarySearch)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 10)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 100)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 10000)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 1000000)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 10000000)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
BENCHMARK_TWO_PARAM(BM_BucketsRangeLookupRMI);
}  // namespace _

