#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <hashing.hpp>
#include <hashtable.hpp>
#include <iostream>
#include <iterator>
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
const size_t gen_dataset_size = 200000000;
const std::vector<std::int64_t> datasets{
    dataset::ID::SEQUENTIAL, dataset::ID::GAPPED_10, dataset::ID::UNIFORM,
    dataset::ID::FB,         dataset::ID::OSM,       dataset::ID::WIKI};

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

  std::cout << "(0) shuffling dataset for probing" << std::endl;
  auto shuffled_dataset = dataset;
  std::shuffle(shuffled_dataset.begin(), shuffled_dataset.end(), rng);

  std::cout << "(1) benchmarking" << std::endl;
  size_t i = 0;
  for (auto _ : state) {
    while (unlikely(i >= shuffled_dataset.size())) i -= shuffled_dataset.size();

    const auto lower = shuffled_dataset[i++];
    const auto upper = lower + interval_size;

    std::vector<Payload> result;
    for (auto iter = std::lower_bound(dataset.begin(), dataset.end(), lower);
         iter < dataset.end() && *iter < upper; iter++)
      result.push_back(*iter - 1);

    benchmark::DoNotOptimize(result.data());
  }
}

struct SequentialRangeLookup {
  template <class T, class Predictor>
  SequentialRangeLookup(const std::vector<T>& dataset,
                        const Predictor& predictor) {
    UNUSED(dataset);
    UNUSED(predictor);
  }

  template <class T>
  forceinline std::vector<T> operator()(size_t pred_ind, size_t lower,
                                        size_t upper,
                                        const std::vector<T>& dataset) const {
    size_t lb = pred_ind;
    if (dataset[lb] > lower)
      while (lb > 0 && dataset[lb] > lower) lb--;
    if (dataset[lb] < lower) {
      while (lb < dataset.size() && dataset[lb] < lower) lb++;
      lb--;
    }

    assert(lb >= 0);
    assert(lb < dataset.size());

    std::vector<T> result;
    for (size_t i = lb; i < dataset.size() && dataset[i] < upper; i++)
      result.push_back(dataset[i] - 1);

    return result;
  }
};

struct ExponentialRangeLookup {
  template <class T, class Predictor>
  ExponentialRangeLookup(const std::vector<T>& dataset,
                         const Predictor& predictor) {
    UNUSED(dataset);
    UNUSED(predictor);
  }

  template <class T>
  forceinline std::vector<T> operator()(size_t pred_ind, size_t lower,
                                        size_t upper,
                                        const std::vector<T>& dataset) const {
    typename std::vector<T>::const_iterator interval_start, interval_end;

    size_t next_err = 1;
    if (dataset[pred_ind] < lower) {
      size_t err = 0;
      while (err + pred_ind < dataset.size() &&
             dataset[pred_ind + err] < lower) {
        err = next_err;
        next_err *= 2;
      }
      err = std::min(err, dataset.size() - pred_ind);

      interval_start = dataset.begin() + pred_ind;
      // +1 since lower_bound searches up to *excluding*
      interval_end = dataset.begin() + pred_ind + err;
    } else {
      size_t err = 0;
      while (err < pred_ind && dataset[pred_ind - err] < lower) {
        err = next_err;
        next_err *= 2;
      }
      err = std::min(err, pred_ind);

      interval_start = dataset.begin() + pred_ind - err;
      // +1 since lower_bound searches up to *excluding*
      interval_end = dataset.begin() + pred_ind;
    }

    assert(interval_start >= dataset.begin());
    assert(interval_start < dataset.end());

    assert(interval_end >= dataset.begin());
    assert(interval_end < dataset.end());

    std::vector<T> result;
    for (auto iter = std::lower_bound(interval_start, interval_end, lower);
         iter < dataset.end() && *iter < upper; iter++)
      result.push_back(*iter - 1);

    return result;
  }
};

struct BinaryRangeLookup {
  size_t max_error = 0;

  template <class T, class Predictor>
  BinaryRangeLookup(const std::vector<T>& dataset, const Predictor& predictor) {
    for (size_t i = 0; i < dataset.size(); i++) {
      const size_t pred = predictor(dataset[i]);
      max_error = std::max(max_error, pred >= i ? pred - i : i - pred);
    }
  }

  template <class T>
  forceinline std::vector<T> operator()(size_t pred_ind, size_t lower,
                                        size_t upper,
                                        const std::vector<T>& dataset) const {
    // compute interval bounds
    auto interval_start =
        dataset.begin() + (pred_ind > max_error) * (pred_ind - max_error);
    // +1 since std::lower_bound searches up to excluding upper bound
    auto interval_end = dataset.begin() +
                        std::min(pred_ind + max_error, dataset.size() - 1) + 1;

    assert(interval_start >= dataset.begin());
    assert(interval_start < dataset.end());

    assert(interval_end >= dataset.begin());
    assert(interval_end < dataset.end());

    std::vector<T> result;
    for (auto iter = std::lower_bound(interval_start, interval_end, lower);
         iter < dataset.end() && *iter < upper; iter++)
      result.push_back(*iter - 1);

    return result;
  }
};

template <size_t SecondLevelModelCount, class RangeLookup>
static void SortedArrayRangeLookupRMITemplate(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  const auto did = static_cast<dataset::ID>(state.range(1));

  std::cout << "(0) loading dataset" << std::endl;
  auto dataset = dataset::load_cached(did, gen_dataset_size);

  if (dataset.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  // generic information
  state.counters["dataset_size"] = dataset.size();
  state.SetLabel(dataset::name(did));

  std::cout << "(1) building rmi" << std::endl;
  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi(
      dataset.begin(), dataset.end(), dataset.size());

  // measure bytesize
  state.counters["sorted_array_bytesize"] =
      sizeof(dataset) + dataset.size() * sizeof(decltype(dataset)::value_type);

  // measure rmi size in bytes
  state.counters["rmi_bytesize"] = rmi.byte_size();

  // construct range lookup method
  const RangeLookup range_lookup(dataset, rmi);

  std::cout << "(2) shuffling dataset for probing" << std::endl;
  auto shuffled_dataset = dataset;
  std::shuffle(shuffled_dataset.begin(), shuffled_dataset.end(), rng);

  size_t i = 0;
  std::cout << "(3) benchmarking" << std::endl;
  for (auto _ : state) {
    while (unlikely(i >= shuffled_dataset.size())) i -= shuffled_dataset.size();

    const auto lower = shuffled_dataset[i++];
    const auto upper = lower + interval_size;

    // Find lower bound sequentially searching starting at pred_ind
    const size_t pred_ind = rmi(lower);
    std::vector<Payload> result = range_lookup(pred_ind, lower, upper, dataset);

    assert(result.size() >= 1);
    assert(result.size() <= interval_size);
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

  ~Bucket() {
    if (next != nullptr) delete next;
  }

  forceinline size_t byte_size() const {
    if (next != nullptr) return sizeof(Bucket<BucketSize>) + next->byte_size();
    return sizeof(Bucket<BucketSize>);
  }

  forceinline void insert(const Key& key) {
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

    previous->next = new Bucket;
    previous->next->insert(key);
  }
} packit;

template <size_t SecondLevelModelCount, size_t BucketSize>
static void BucketsRangeLookupRMI(benchmark::State& state) {
  const size_t interval_size = state.range(0);
  const auto did = static_cast<dataset::ID>(state.range(1));

  std::cout << "(0) loading dataset" << std::endl;
  auto dataset = dataset::load_cached(did, gen_dataset_size);

  if (dataset.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  state.counters["dataset_size"] = dataset.size();
  state.SetLabel(dataset::name(did));

  // initialize directory
  std::vector<Bucket<BucketSize>> buckets(dataset.size() / BucketSize);

  std::cout << "(1) building rmi" << std::endl;
  const learned_hashing::RMIHash<Key, SecondLevelModelCount> rmi(
      dataset.begin(), dataset.end(), buckets.size());

  // insert all keys exactly where model tells us to
  size_t notify_at = dataset.size() / 100;
  std::cout << "(2) shuffling dataset for insertion" << std::endl;
  auto shuffled_dataset = dataset;
  std::shuffle(shuffled_dataset.begin(), shuffled_dataset.end(), rng);
  std::cout << "(3) inserting keys" << std::endl;
  for (size_t i = 0; i < dataset.size(); i++) {
    const auto key = dataset[i];
    const auto ind = rmi(key);
    buckets[ind].insert(key);
    if (i % notify_at == 0) std::cout << "." << std::flush;
  }
  std::cout << std::endl;

  // measure directories's byte size
  size_t directory_bytesize = sizeof(buckets);
  for (const auto& bucket : buckets) directory_bytesize += bucket.byte_size();
  state.counters["directory_bytesize"] = directory_bytesize;

  // measure rmi size in bytes
  state.counters["rmi_bytesize"] = rmi.byte_size();

  std::cout << "(3) shuffling dataset for probing" << std::endl;
  std::shuffle(shuffled_dataset.begin(), shuffled_dataset.end(), rng);

  size_t i = 0;
  std::cout << "(4) benchmarking" << std::endl;
  for (auto _ : state) {
    while (unlikely(i >= shuffled_dataset.size())) i -= shuffled_dataset.size();

    const auto lower = shuffled_dataset[i++];
    const auto upper = lower + interval_size;

    std::vector<Payload> result;

    bool search_finished = false;
    for (size_t bucket_ind = rmi(lower);
         !search_finished && bucket_ind < buckets.size(); bucket_ind++) {
      for (auto* b = &buckets[bucket_ind]; b != nullptr; b = b->next) {
        for (size_t i = 0; i < BucketSize; i++) {
          const auto& current_key = b->keys[i];
          // assume bucket chain is finisehd & break early
          if (current_key == Sentinel) {
            b = nullptr;
            break;
          }

          // don't assume data was inserted in sorted order
          if (current_key >= upper) {
            search_finished = true;
            continue;
          }

          // don't assume data was inserted in sorted order
          if (current_key < lower) continue;

          result.push_back(current_key - 1);
        }

        // assume data was inserted in sorted order, i.e., bucket
        // chain does not have to be traversed further
        if (search_finished || b == nullptr) break;
      }
    }

    assert(result.size() >= 1);
    assert(result.size() <= interval_size);
    benchmark::DoNotOptimize(result.data());
  }
}

#define __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, bucket_size) \
  BENCHMARK_TEMPLATE(fun, model_size, bucket_size)                     \
      ->ArgsProduct({intervals, datasets});

#define _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size) \
  __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, 1)   \
  __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, 2)   \
  __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, 8)
#define BENCHMARK_BUCKETS_RANGE_LOOKUP(fun)     \
  _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, 1000)    \
  _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, 100000)  \
  _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, 1000000) \
  _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, 10000000)

BENCHMARK(SortedArrayRangeLookupBinarySearch)
    ->ArgsProduct({intervals, datasets});

#define BENCHMARK_SORTED_RANGE_LOOKUP_RMI(LookupMethod)                        \
  BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMITemplate, 1000, LookupMethod)    \
      ->ArgsProduct({intervals, datasets});                                    \
  BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMITemplate, 100000, LookupMethod)  \
      ->ArgsProduct({intervals, datasets});                                    \
  BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMITemplate, 1000000, LookupMethod) \
      ->ArgsProduct({intervals, datasets});                                    \
  BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMITemplate, 10000000, LookupMethod)

BENCHMARK_BUCKETS_RANGE_LOOKUP(BucketsRangeLookupRMI);

BENCHMARK_SORTED_RANGE_LOOKUP_RMI(BinaryRangeLookup);
BENCHMARK_SORTED_RANGE_LOOKUP_RMI(ExponentialRangeLookup);
BENCHMARK_SORTED_RANGE_LOOKUP_RMI(SequentialRangeLookup);

}  // namespace _

