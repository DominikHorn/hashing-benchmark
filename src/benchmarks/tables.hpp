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
#include <type_traits>
#include <utility>
#include <vector>

#include "../../thirdparty/perfevent/PerfEvent.hpp"
#include "../support/datasets.hpp"
#include "../support/probing_set.hpp"
#include "include/convenience/builtins.hpp"
#include "include/mmphf/rank_hash.hpp"
#include "include/rmi.hpp"

namespace _ {
using Key = std::uint64_t;
using Payload = std::uint64_t;

const std::vector<std::int64_t> dataset_sizes{1000000, 10000000, 100000000};
const std::vector<std::int64_t> datasets{
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::SEQUENTIAL),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::GAPPED_10),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::UNIFORM),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::FB),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::NORMAL),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::OSM),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::WIKI)};
const std::vector<std::int64_t> probe_distributions{
    static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
        dataset::ProbingDistribution::UNIFORM),
    static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
        dataset::ProbingDistribution::EXPONENTIAL_RANDOM),
    static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
        dataset::ProbingDistribution::EXPONENTIAL_SORTED)};

template <class Table, size_t RangeSize>
static void TableProbe(benchmark::State& state) {
  std::random_device rd;
  std::default_random_engine rng(rd());

  // Extract variables
  const auto dataset_size = static_cast<size_t>(state.range(0));
  const auto did = static_cast<dataset::ID>(state.range(1));
  const auto probing_dist =
      static_cast<dataset::ProbingDistribution>(state.range(2));

  // Generate data (keys & payloads) & probing set
  std::vector<std::pair<Key, Payload>> data;
  data.reserve(dataset_size);
  std::vector<Key> probing_set;
  {
    auto keys = dataset::load_cached<Key>(did, dataset_size);

    std::transform(keys.begin(), keys.end(), std::back_inserter(data),
                   [](const Key& key) { return std::make_pair(key, key - 5); });

    probing_set = dataset::generate_probing_set(keys, probing_dist);
  }

  if (data.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  // build table
  Table table(data);

  size_t i = 0;
  for (auto _ : state) {
    while (unlikely(i >= probing_set.size())) i -= probing_set.size();
    const auto searched = probing_set[i++];

    // Lower bound lookup
    auto it = table[searched];

    // RangeSize == 0 -> only key lookup
    // RangeSize == 1 -> key + payload lookup
    // RangeSize  > 1 -> lb key + multiple payload lookups
    if constexpr (RangeSize == 0) {
      benchmark::DoNotOptimize(it);
      full_mem_barrier;
    } else if constexpr (RangeSize == 1) {
      const auto payload = *it;
      benchmark::DoNotOptimize(payload);
      full_mem_barrier;
    } else if constexpr (RangeSize > 1) {
      for (size_t i = 0; it != table.end() && i < RangeSize; i++, ++it) {
        const auto payload = *it;
        benchmark::DoNotOptimize(payload);
        full_mem_barrier;
      }
    }
  }

  // set counters (don't do this in inner loop to avoid tainting results)
  state.counters["table_bytes"] = table.byte_size();
  state.counters["table_directory_bytes"] = table.directory_byte_size();
  state.counters["table_bits_per_key"] = 8. * table.byte_size() / data.size();
  state.counters["data_elem_count"] = data.size();
  state.SetLabel(table.name() + ":" + dataset::name(did) + ":" +
                 dataset::name(probing_dist));
}

using namespace masters_thesis;

#define BM_LB(Table)                                                 \
  BENCHMARK_TEMPLATE(TableProbe, Table, 0)                           \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions}); \
  BENCHMARK_TEMPLATE(TableProbe, Table, 1)                           \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions});

#define BM_RANGE(Table)                                              \
  BENCHMARK_TEMPLATE(TableProbe, Table, 10)                          \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions}); \
  BENCHMARK_TEMPLATE(TableProbe, Table, 20)                          \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions});

#define BM(Table) \
  BM_LB(Table)    \
  BM_RANGE(Table)

#define BenchmarkMonotone(BucketSize, Model)              \
  using MonotoneHashtable##BucketSize##Model =            \
      MonotoneHashtable<Key, Payload, BucketSize, Model>; \
  BM_LB(MonotoneHashtable##BucketSize##Model);

#define BenchmarkNonMonotoneLB(BucketSize, Model)         \
  using MonotoneHashtable##BucketSize##Model =            \
      MonotoneHashtable<Key, Payload, BucketSize, Model>; \
  BM(MonotoneHashtable##BucketSize##Model);

#define BenchmarkMMPHFTable(MMPHF)                           \
  using MMPHFTable##MMPHF = MMPHFTable<Key, Payload, MMPHF>; \
  BM(MMPHFTable##MMPHF);

using MWHC = exotic_hashing::MWHC<Key>;
BenchmarkMMPHFTable(MWHC);

using CompressedMWHC = exotic_hashing::CompressedMWHC<Key>;
BenchmarkMMPHFTable(CompressedMWHC);

using RankHash = exotic_hashing::RankHash<Key>;
BenchmarkMMPHFTable(RankHash);

using MonotoneRMILearnedRank =
    exotic_hashing::LearnedRank<Key,
                                learned_hashing::MonotoneRMIHash<Key, 1000000>>;
BenchmarkMMPHFTable(MonotoneRMILearnedRank);

using RadixSplineLearnedRank =
    exotic_hashing::LearnedRank<Key, learned_hashing::RadixSplineHash<Key>>;
BenchmarkMMPHFTable(RadixSplineLearnedRank);

using RMI = learned_hashing::RMIHash<Key, 1000000>;
BenchmarkNonMonotoneLB(1, RMI);
BenchmarkNonMonotoneLB(4, RMI);

using MonotoneRMI = learned_hashing::MonotoneRMIHash<Key, 1000000>;
BenchmarkMonotone(1, MonotoneRMI);
BenchmarkMonotone(4, MonotoneRMI);

using MonotoneRadixSpline = learned_hashing::RadixSplineHash<Key>;
BenchmarkMonotone(1, MonotoneRadixSpline);
BenchmarkMonotone(4, MonotoneRadixSpline);

using CompressedMWHC = exotic_hashing::CompressedMWHC<Key>;
BenchmarkMMPHFTable(CompressedMWHC);

using CompressedRankHash = exotic_hashing::CompressedRankHash<Key>;
BenchmarkMMPHFTable(CompressedRankHash);

using MonotoneRMICompressedLearnedRank = exotic_hashing::CompressedLearnedRank<
    Key, learned_hashing::MonotoneRMIHash<Key, 1000000>>;
BenchmarkMMPHFTable(MonotoneRMICompressedLearnedRank);

using RadixSplineCompressedLearnedRank = exotic_hashing::CompressedLearnedRank<
    Key, learned_hashing::RadixSplineHash<Key>>;
BenchmarkMMPHFTable(RadixSplineCompressedLearnedRank);
}  // namespace _
