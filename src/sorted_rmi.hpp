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

const size_t gen_dataset_size = 200000000;
const std::vector<std::int64_t> datasets{
    dataset::ID::SEQUENTIAL, /*dataset::ID::GAPPED_10,*/ dataset::ID::UNIFORM,
    dataset::ID::FB, dataset::ID::OSM, dataset::ID::WIKI};

std::random_device rd;
std::default_random_engine rng(rd());

static void SortedArrayRangeLookupBinarySearch(benchmark::State& state) {
  const auto did = static_cast<dataset::ID>(state.range(0));
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

  std::cout << "(1) generating payloads" << std::endl;
  auto payloads = dataset;
  std::shuffle(payloads.begin(), payloads.end(), rng);

  std::cout << "(2) benchmarking" << std::endl;
  size_t i = 0;
  for (auto _ : state) {
    while (unlikely(i >= shuffled_dataset.size())) i -= shuffled_dataset.size();

    const auto searched = shuffled_dataset[i++];

    const auto iter =
        std::lower_bound(dataset.begin(), dataset.end(), searched);
    const Payload payload = iter < dataset.end()
                                ? payloads[std::distance(dataset.begin(), iter)]
                                : std::numeric_limits<Payload>::max();
    benchmark::DoNotOptimize(payload);
  }
}

struct SequentialRangeLookup {
  template <class T, class Predictor>
  SequentialRangeLookup(const std::vector<T>& dataset,
                        const Predictor& predictor) {
    UNUSED(dataset);
    UNUSED(predictor);
  }

  forceinline size_t operator()(size_t pred_ind, Key searched,
                                const std::vector<Key>& dataset) const {
    const auto last_ind = dataset.size() - 1;
    size_t lb = pred_ind;

    while (lb > 0 && dataset[lb] > searched) lb--;
    while (lb < last_ind && dataset[lb] < searched) lb++;

    assert(lb >= 0);
    assert(lb <= last_ind);

    return lb;
  }
};

struct ExponentialRangeLookup {
  template <class T, class Predictor>
  ExponentialRangeLookup(const std::vector<T>& dataset,
                         const Predictor& predictor) {
    UNUSED(dataset);
    UNUSED(predictor);
  }

  forceinline size_t operator()(size_t pred_ind, Key searched,
                                const std::vector<Key>& dataset) const {
    typename std::vector<Key>::const_iterator interval_start, interval_end;
    const auto dataset_size = dataset.size();

    size_t next_err = 1;
    if (dataset[pred_ind] < searched) {
      size_t err = 0;
      while (err + pred_ind < dataset_size &&
             dataset[err + pred_ind] < searched) {
        err = next_err;
        next_err *= 2;
      }
      err = std::min(err, dataset_size - pred_ind);

      interval_start = dataset.begin() + pred_ind;
      // +1 since lower_bound searches up to *excluding*
      interval_end = dataset.begin() + pred_ind + err;
    } else {
      size_t err = 0;
      while (err < pred_ind && dataset[pred_ind - err] < searched) {
        err = next_err;
        next_err *= 2;
      }
      err = std::min(err, pred_ind);

      interval_start = dataset.begin() + pred_ind - err;
      // +1 since lower_bound searches up to *excluding*
      interval_end = dataset.begin() + pred_ind;
    }

    assert(interval_start >= dataset.begin());
    assert(interval_end >= interval_start);
    assert(interval_end < dataset.end());

    return std::distance(
        dataset.begin(),
        std::lower_bound(interval_start, interval_end, searched));
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

  forceinline Payload operator()(size_t pred_ind, Key searched,
                                 const std::vector<Key>& dataset) const {
    // compute interval bounds
    auto interval_start =
        dataset.begin() + (pred_ind > max_error) * (pred_ind - max_error);
    // +1 since std::lower_bound searches up to excluding upper bound
    auto interval_end = dataset.begin() +
                        std::min(pred_ind + max_error, dataset.size() - 1) + 1;

    assert(interval_start >= dataset.begin());
    assert(interval_end >= interval_start);
    assert(interval_end < dataset.end());

    return std::distance(
        dataset.begin(),
        std::lower_bound(interval_start, interval_end, searched));
  }
};

template <size_t SecondLevelModelCount, class RangeLookup>
static void SortedArrayRangeLookupRMITemplate(benchmark::State& state) {
  const auto did = static_cast<dataset::ID>(state.range(0));

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

  std::cout << "(3) generating payloads" << std::endl;
  auto payloads = dataset;
  std::shuffle(payloads.begin(), payloads.end(), rng);

  size_t i = 0;
  std::cout << "(4) benchmarking" << std::endl;
  for (auto _ : state) {
    while (unlikely(i >= shuffled_dataset.size())) i -= shuffled_dataset.size();

    const auto searched = shuffled_dataset[i++];

    const size_t payload_ind = range_lookup(rmi(searched), searched, dataset);
    const Payload payload = payload_ind < payloads.size()
                                ? payloads[payload_ind]
                                : std::numeric_limits<Payload>::max();
    benchmark::DoNotOptimize(payload);
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
  const auto did = static_cast<dataset::ID>(state.range(0));

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

  std::cout << "(4) generating payloads" << std::endl;
  auto payloads = dataset;
  std::shuffle(payloads.begin(), payloads.end(), rng);

  size_t i = 0;
  const auto buckets_size = buckets.size();
  const auto dataset_size = dataset.size();
  std::cout << "(5) benchmarking" << std::endl;
  for (auto _ : state) {
    while (unlikely(i >= dataset_size)) i -= dataset_size;

    const auto searched = shuffled_dataset[i++];
    auto payload = std::numeric_limits<Payload>::max();

    // all keys are placed exactly where model tells us
    for (size_t bucket_ind = rmi(searched); bucket_ind < buckets_size;
         bucket_ind++) {
      auto bucket = &buckets[bucket_ind];
      // TODO: implement SOSD vectorized lookup! -> has to be specialized for
      // each bucket size :(
      for (size_t i = 0; i < BucketSize; i++) {
        const auto& current_key = bucket->keys[i];
        if (current_key == Sentinel) break;
        if (current_key == searched) {
          payload = current_key - 1;
          goto search_finished;
        }
      }
    }
  search_finished:

    benchmark::DoNotOptimize(payload);
  }
}

#define __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, bucket_size) \
  BENCHMARK_TEMPLATE(fun, model_size, bucket_size)->ArgsProduct({datasets});

#define _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size) \
  __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, 1)   \
  __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, 2)   \
  __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, 8)
#define BENCHMARK_BUCKETS_RANGE_LOOKUP(fun)    \
  _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, 1000)   \
  _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, 100000) \
  _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, 10000000)

#define BENCHMARK_SORTED_RANGE_LOOKUP_RMI(LookupMethod)                       \
  BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMITemplate, 1000, LookupMethod)   \
      ->ArgsProduct({datasets});                                              \
  BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMITemplate, 100000, LookupMethod) \
      ->ArgsProduct({datasets});                                              \
  BENCHMARK_TEMPLATE(SortedArrayRangeLookupRMITemplate, 10000000,             \
                     LookupMethod)                                            \
      ->ArgsProduct({datasets});

BENCHMARK(SortedArrayRangeLookupBinarySearch)->ArgsProduct({datasets});

BENCHMARK_BUCKETS_RANGE_LOOKUP(BucketsRangeLookupRMI);

BENCHMARK_SORTED_RANGE_LOOKUP_RMI(BinaryRangeLookup);
BENCHMARK_SORTED_RANGE_LOOKUP_RMI(ExponentialRangeLookup);
BENCHMARK_SORTED_RANGE_LOOKUP_RMI(SequentialRangeLookup);

}  // namespace _

