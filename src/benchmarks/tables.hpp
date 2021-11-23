#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
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
#include <string>
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

template <class Table>
static void Construction(benchmark::State& state) {
  std::random_device rd;
  std::default_random_engine rng(rd());

  // Extract variables
  const auto dataset_size = static_cast<size_t>(state.range(0));
  const auto did = static_cast<dataset::ID>(state.range(1));

  // Generate data (keys & payloads) & probing set
  std::vector<std::pair<Key, Payload>> data;
  data.reserve(dataset_size);
  {
    auto keys = dataset::load_cached<Key>(did, dataset_size);

    std::transform(keys.begin(), keys.end(), std::back_inserter(data),
                   [](const Key& key) { return std::make_pair(key, key - 5); });
  }

  if (data.empty()) {
    // otherwise google benchmark produces an error ;(
    for (auto _ : state) {
    }
    return;
  }

  // build table
  size_t total_bytes, directory_bytes;
  std::string name;
  for (auto _ : state) {
    Table table(data);

    total_bytes = table.byte_size();
    directory_bytes = table.directory_byte_size();
    name = table.name();
  }

  // set counters (don't do this in inner loop to avoid tainting results)
  state.counters["table_bytes"] = total_bytes;
  state.counters["table_directory_bytes"] = directory_bytes;
  state.counters["table_model_bytes"] = total_bytes - directory_bytes;
  state.counters["table_bits_per_key"] = 8. * total_bytes / data.size();
  state.counters["data_elem_count"] = data.size();
  state.SetLabel(name + ":" + dataset::name(did));
}

template <class Table, size_t RangeSize>
static void TableProbe(benchmark::State& state) {
  // Extract variables
  const auto dataset_size = static_cast<size_t>(state.range(0));
  const auto did = static_cast<dataset::ID>(state.range(1));
  const auto probing_dist =
      static_cast<dataset::ProbingDistribution>(state.range(2));

  static std::vector<std::pair<Key, Payload>> data{};
  static std::vector<Key> probing_set{};

  static Table table{};
  static std::string previous_signature = "";

  // google benchmark will run a benchmark function multiple times
  // to determine, amongst other things, the iteration count for
  // the benchmark loop. Technically, BM functions must be pure. However,
  // since this setup logic is very expensive, we cache setup based on
  // a unique signature containing all parameters.
  // NOTE: google benchmark's fixtures suffer from the same
  // 'execute setup multiple times' issue:
  // https://github.com/google/benchmark/issues/952
  std::string signature =
      std::string(typeid(Table).name()) + "_" + std::to_string(RangeSize) +
      "_" + std::to_string(dataset_size) + "_" + dataset::name(did) + "_" +
      dataset::name(probing_dist);
  if (previous_signature != signature) {
    std::cout << "performing setup... ";
    auto start = std::chrono::steady_clock::now();
    // Generate data (keys & payloads) & probing set
    data.clear();
    data.reserve(dataset_size);
    {
      auto keys = dataset::load_cached<Key>(did, dataset_size);

      std::transform(
          keys.begin(), keys.end(), std::back_inserter(data),
          [](const Key& key) { return std::make_pair(key, key - 5); });

      probing_set = dataset::generate_probing_set(keys, probing_dist);
    }

    if (data.empty()) {
      // otherwise google benchmark produces an error ;(
      for (auto _ : state) {
      }
      std::cout << "failed" << std::endl;
      return;
    }

    // build table
    table = Table(data);

    // measure time elapsed
    const auto delta = std::chrono::steady_clock::now() - start;
    std::cout << "succeeded in " << std::setw(9) << delta.count() << " seconds"
              << std::endl;
  }
  previous_signature = signature;

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
    } else if constexpr (RangeSize == 1) {
      const auto payload = *it;
      benchmark::DoNotOptimize(payload);
    } else if constexpr (RangeSize > 1) {
      Payload total = 0;
      for (size_t i = 0; it != table.end() && i < RangeSize; i++, ++it) {
        total += *it;
      }
      benchmark::DoNotOptimize(total);
    }
    full_mem_barrier;
  }

  // set counters (don't do this in inner loop to avoid tainting results)
  state.counters["table_bytes"] = table.byte_size();
  state.counters["table_directory_bytes"] = table.directory_byte_size();
  state.counters["table_bits_per_key"] = 8. * table.byte_size() / data.size();
  state.counters["data_elem_count"] = data.size();
  state.SetLabel(table.name() + ":" + dataset::name(did) + ":" +
                 dataset::name(probing_dist));
}

template <class Table>
static void TableMixedLookup(benchmark::State& state) {
  std::random_device rd;
  std::default_random_engine rng(rd());

  // Extract variables
  const auto dataset_size = static_cast<size_t>(state.range(0));
  const auto did = static_cast<dataset::ID>(state.range(1));
  const auto probing_dist =
      static_cast<dataset::ProbingDistribution>(state.range(2));
  const auto percentage_of_point_queries = static_cast<size_t>(state.range(3));

  static std::vector<std::pair<Key, Payload>> data{};
  static std::vector<Key> probing_set{};

  static Table table{};
  static std::string previous_signature = "";

  // google benchmark will run a benchmark function multiple times
  // to determine, amongst other things, the iteration count for
  // the benchmark loop. Technically, BM functions must be pure. However,
  // since this setup logic is very expensive, we cache setup based on
  // a unique signature containing all parameters.
  // NOTE: google benchmark's fixtures suffer from the same
  // 'execute setup multiple times' issue:
  // https://github.com/google/benchmark/issues/952
  std::string signature = std::string(typeid(Table).name()) + "_" +
                          std::to_string(percentage_of_point_queries) + "_" +
                          std::to_string(dataset_size) + "_" +
                          dataset::name(did) + "_" +
                          dataset::name(probing_dist);
  if (previous_signature != signature) {
    std::cout << "performing setup... ";
    auto start = std::chrono::steady_clock::now();

    // Generate data (keys & payloads) & probing set
    data.clear();
    data.reserve(dataset_size);
    {
      auto keys = dataset::load_cached<Key>(did, dataset_size);

      std::transform(
          keys.begin(), keys.end(), std::back_inserter(data),
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
    table = Table(data);

    // measure time elapsed
    const auto delta = std::chrono::steady_clock::now() - start;
    std::cout << "succeeded in " << std::setw(9) << delta.count() << " seconds"
              << std::endl;
  }
  previous_signature = signature;

  // distribution
  std::uniform_int_distribution<size_t> point_query_dist(1, 100);

  size_t i = 0;
  for (auto _ : state) {
    while (unlikely(i >= probing_set.size())) i -= probing_set.size();
    const auto searched = probing_set[i++];

    // Lower bound lookup
    auto it = table[searched];
    const auto lb_payload = *it;
    benchmark::DoNotOptimize(lb_payload);

    // Chance based perform full range scan
    if (point_query_dist(rng) > percentage_of_point_queries) {
      ++it;
      Payload total = 0;
      for (size_t i = 1; it != table.end() && i < 10; i++, ++it) {
        total += *it;
      }
      benchmark::DoNotOptimize(total);
    }

    full_mem_barrier;
  }

  // set counters (don't do this in inner loop to avoid tainting results)
  state.counters["table_bytes"] = table.byte_size();
  state.counters["point_lookup_percent"] =
      static_cast<double>(percentage_of_point_queries) / 100.0;
  state.counters["table_directory_bytes"] = table.directory_byte_size();
  state.counters["table_bits_per_key"] = 8. * table.byte_size() / data.size();
  state.counters["data_elem_count"] = data.size();
  state.SetLabel(table.name() + ":" + dataset::name(did) + ":" +
                 dataset::name(probing_dist));
}

using namespace masters_thesis;

#define BM(Table)                                                              \
  BENCHMARK_TEMPLATE(Construction, Table)                                      \
      ->ArgsProduct({dataset_sizes, datasets});                                \
  BENCHMARK_TEMPLATE(TableMixedLookup, Table)                                  \
      ->ArgsProduct(                                                           \
          {{100000000}, datasets, probe_distributions, {0, 25, 50, 75, 100}}); \
  BENCHMARK_TEMPLATE(TableProbe, Table, 0)                                     \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions});           \
  BENCHMARK_TEMPLATE(TableProbe, Table, 1)                                     \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions});           \
  BENCHMARK_TEMPLATE(TableProbe, Table, 10)                                    \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions});           \
  BENCHMARK_TEMPLATE(TableProbe, Table, 20)                                    \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions});

#define BenchmarkMonotone(BucketSize, Model)                    \
  using MonotoneHashtable##BucketSize##Model =                  \
      MonotoneHashtable<Key, Payload, BucketSize, Model>;       \
  BM(MonotoneHashtable##BucketSize##Model);                     \
  using PrefetchedMonotoneHashtable##BucketSize##Model =        \
      MonotoneHashtable<Key, Payload, BucketSize, Model, true>; \
  BM(PrefetchedMonotoneHashtable##BucketSize##Model);

#define BenchmarkMMPHFTable(MMPHF)                           \
  using MMPHFTable##MMPHF = MMPHFTable<Key, Payload, MMPHF>; \
  BM(MMPHFTable##MMPHF);

using MWHC = exotic_hashing::MWHC<Key>;
BenchmarkMMPHFTable(MWHC);

using CompressedMWHC = exotic_hashing::CompressedMWHC<Key>;
BenchmarkMMPHFTable(CompressedMWHC);

using RankHash = exotic_hashing::RankHash<Key>;
BenchmarkMMPHFTable(RankHash);

using FST = exotic_hashing::FastSuccinctTrie<Key>;
BenchmarkMMPHFTable(FST);

// using MonotoneRMILearnedRank =
//     exotic_hashing::LearnedRank<Key,
//                                 learned_hashing::MonotoneRMIHash<Key,
//                                 1000000>>;
// BenchmarkMMPHFTable(MonotoneRMILearnedRank);

using RadixSplineLearnedRank =
    exotic_hashing::LearnedRank<Key, learned_hashing::RadixSplineHash<Key>>;
BenchmarkMMPHFTable(RadixSplineLearnedRank);

using RadixSplineUnoptimizedLearnedRank =
    exotic_hashing::UnoptimizedLearnedRank<
        Key, learned_hashing::RadixSplineHash<Key>>;
BenchmarkMMPHFTable(RadixSplineUnoptimizedLearnedRank);

using RMI = learned_hashing::RMIHash<Key, 1000000>;
BenchmarkMonotone(1, RMI);
BenchmarkMonotone(4, RMI);

// using MonotoneRMI = learned_hashing::MonotoneRMIHash<Key, 1000000>;
// BenchmarkMonotone(1, MonotoneRMI);
// BenchmarkMonotone(4, MonotoneRMI);

using MonotoneRadixSpline = learned_hashing::RadixSplineHash<Key>;
BenchmarkMonotone(1, MonotoneRadixSpline);
BenchmarkMonotone(4, MonotoneRadixSpline);

using CompressedMWHC = exotic_hashing::CompressedMWHC<Key>;
BenchmarkMMPHFTable(CompressedMWHC);

using CompressedRankHash = exotic_hashing::CompressedRankHash<Key>;
BenchmarkMMPHFTable(CompressedRankHash);

// using MonotoneRMICompressedLearnedRank =
// exotic_hashing::CompressedLearnedRank<
//     Key, learned_hashing::MonotoneRMIHash<Key, 1000000>>;
// BenchmarkMMPHFTable(MonotoneRMICompressedLearnedRank);

using RadixSplineCompressedLearnedRank = exotic_hashing::CompressedLearnedRank<
    Key, learned_hashing::RadixSplineHash<Key>>;
BenchmarkMMPHFTable(RadixSplineCompressedLearnedRank);
}  // namespace _
