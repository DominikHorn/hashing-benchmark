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

static void ShuffleArray(benchmark::State& state) {
  auto dataset =
      dataset::load_cached(dataset::ID::SEQUENTIAL, gen_dataset_size);
  for (auto _ : state) {
    dataset::shuffle(dataset);
    benchmark::DoNotOptimize(dataset.data());
  }
}

static void ShuffleAndSortArray(benchmark::State& state) {
  auto dataset =
      dataset::load_cached(dataset::ID::SEQUENTIAL, gen_dataset_size);
  for (auto _ : state) {
    dataset::shuffle(dataset);
    std::sort(dataset.begin(), dataset.end());
    benchmark::DoNotOptimize(dataset.data());
  }
}

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

  // We must sort the entire array ;(
  std::sort(dataset.begin(), dataset.end());

  std::uniform_int_distribution<size_t> dist(0, dataset.size());
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
  auto dataset = dataset::load_cached(did, gen_dataset_size);

  state.counters["dataset_size"] = dataset.size();
  state.SetLabel(dataset::name(did));

  if (dataset.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  // build model based on data sample. assume data is random shuffled (which it
  // is) to compactify this code
  std::vector<decltype(dataset)::value_type> sample(
      dataset.begin(), dataset.begin() + dataset.size() / 100);
  std::sort(sample.begin(), sample.end());
  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi(
      sample.begin(), sample.end(), dataset.size() - 1);

  // We must sort the entire array ;(
  std::sort(dataset.begin(), dataset.end());

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

  std::cout << "max_error: " << max_error << std::endl;

  std::uniform_int_distribution<size_t> dist(0, dataset.size());
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
    for (size_t i = 0; i < BucketSize; i++) {
      if (keys[i] == Sentinel) {
        keys[i] = key;
        return;
      }
    }

    if (next == nullptr) next = tape.new_bucket();
    next->insert(key, tape);
  }
} packit;

template <size_t SecondLevelModelCount, size_t BucketSize>
static void BucketsRangeLookupRMI(benchmark::State& state) {
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

  std::vector<Bucket<BucketSize>> buckets(
      dataset.size());  // TODO: load factors?

  // sample data. assume t is random shuffled (which it
  // is) to compactify this code
  std::vector<decltype(dataset)::value_type> sample(
      dataset.begin(), dataset.begin() + dataset.size() / 100);

  // build model (sorted input!)
  std::sort(sample.begin(), sample.end());
  std::cout << "(1) building rmi" << std::endl;
  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi(
      sample.begin(), sample.end(), buckets.size() - 1);
  std::cout << "(2) inserting keys" << std::endl;

  // insert all keys exactly where model tells us to
  typename Bucket<BucketSize>::Tape tape;
  for (const auto& key : dataset) buckets[rmi(key)].insert(key, tape);

  std::cout << "(3) probing" << std::endl;
  std::uniform_int_distribution<size_t> dist(0, dataset.size());
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
}

#define __BENCHMARK_TWO_PARAM(fun, model_size, bucket_size) \
  BENCHMARK_TEMPLATE(fun, model_size, bucket_size)          \
      ->ArgsProduct({intervals, datasets});

#define _BENCHMARK_TWO_PARAM(fun, model_size) \
  __BENCHMARK_TWO_PARAM(fun, model_size, 1)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 2)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 4)   \
  __BENCHMARK_TWO_PARAM(fun, model_size, 16)
#define BENCHMARK_TWO_PARAM(fun)   \
  _BENCHMARK_TWO_PARAM(fun, 0)    \
  _BENCHMARK_TWO_PARAM(fun, 10)    \
  _BENCHMARK_TWO_PARAM(fun, 1000)   \
  _BENCHMARK_TWO_PARAM(fun, 100000) \
  _BENCHMARK_TWO_PARAM(fun, 10000000)

BENCHMARK(ShuffleArray);
BENCHMARK(ShuffleAndSortArray);

BENCHMARK(SortedArrayRangeLookupBinarySearch)
    ->ArgsProduct({intervals, datasets});

BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMI, 0)
    ->ArgsProduct({intervals, datasets});
BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMI, 100)
    ->ArgsProduct({intervals, datasets});
BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMI, 1000)
    ->ArgsProduct({intervals, datasets});
BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMI, 100000)
    ->ArgsProduct({intervals, datasets});
BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMI, 10000000)
    ->ArgsProduct({intervals, datasets});

BENCHMARK_TWO_PARAM(BucketsRangeLookupRMI);
}  // namespace _

