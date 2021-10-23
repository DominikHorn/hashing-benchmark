#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <hashing.hpp>
#include <hashtable.hpp>
#include <iostream>
#include <iterator>
#include <learned_hashing.hpp>
#include <limits>
#include <masters_thesis.hpp>
#include <ostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "../thirdparty/perfevent/PerfEvent.hpp"
#include "include/convenience/builtins.hpp"
#include "support/datasets.hpp"

namespace _ {
using Key = std::uint64_t;
using Payload = std::uint64_t;

const size_t gen_dataset_size = 200000000;
const std::vector<std::int64_t> datasets{
    dataset::ID::SEQUENTIAL, dataset::ID::UNIFORM, dataset::ID::WIKI,
    dataset::ID::OSM, dataset::ID::FB};

std::random_device rd;
std::default_random_engine rng(rd());

static void SortedArrayRangeLookupBinarySearch(benchmark::State& state) {
  const auto did = static_cast<dataset::ID>(state.range(0));
  auto dataset = dataset::load_cached<Key>(did, gen_dataset_size);

  state.counters["dataset_size"] = dataset.size();
  state.SetLabel(dataset::name(did));

  if (dataset.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  // generate payloads
  auto payloads = dataset;
  std::shuffle(payloads.begin(), payloads.end(), rng);

  // shuffle for probing
  auto shuffled_dataset = dataset;
  std::shuffle(shuffled_dataset.begin(), shuffled_dataset.end(), rng);

  // benchmark
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
    full_mem_barrier;
  }
}

// template <size_t SecondLevelModelCount, size_t BucketSize>
// static void BucketsRangeLookupRMI(benchmark::State& state) {
//   const auto did = static_cast<dataset::ID>(state.range(0));
//   auto dataset = dataset::load_cached<Key>(did, gen_dataset_size);
//
//   if (dataset.empty()) {
//     // otherwise google benchmark produces an error ;(
//     for (auto _ : state) {
//     }
//     return;
//   }
//
//   state.counters["dataset_size"] = dataset.size();
//   state.SetLabel(dataset::name(did));
//
//   // generate payloads
//   auto payloads = dataset;
//   std::shuffle(payloads.begin(), payloads.end(), rng);
//
//   // build hashtable
//   masters_thesis::MonotoneHashtable<Key, Payload, BucketSize> ht(dataset,
//                                                                  payloads);
//
//   // measure byte sizes
//   state.counters["directory_bytesize"] = ht.directory_byte_size();
//   state.counters["model_bytesize"] = ht.model_byte_size();
//
//   // shuffle data for probing
//   auto shuffled_dataset = dataset;
//   std::shuffle(shuffled_dataset.begin(), shuffled_dataset.end(), rng);
//
//   // benchmark
//   size_t i = 0;
//   const auto dataset_size = dataset.size();
//   for (auto _ : state) {
//     while (unlikely(i >= dataset_size)) i -= dataset_size;
//
//     const auto searched = shuffled_dataset[i++];
//     const auto payload = ht.lookup(searched);
//     benchmark::DoNotOptimize(payload);
//     full_mem_barrier;
//   }
// }
//
// #define __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, bucket_size) \
//   BENCHMARK_TEMPLATE(fun, model_size, bucket_size)->ArgsProduct({datasets});
//
// #define _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size) \
//   __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, 1)   \
//   __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, 2)   \
//   __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, 4)   \
//   __BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, model_size, 8)
// #define BENCHMARK_BUCKETS_RANGE_LOOKUP(fun)   \
//   _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, 100)   \
//   _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, 10000) \
//   _BENCHMARK_BUCKETS_RANGE_LOOKUP(fun, 1000000)
//
// BENCHMARK_BUCKETS_RANGE_LOOKUP(BucketsRangeLookupRMI);

BENCHMARK(SortedArrayRangeLookupBinarySearch);

}  // namespace _

