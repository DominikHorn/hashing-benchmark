#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
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
#include <string>

#include <stdio.h>
#include <stdlib.h>

#include "../../thirdparty/perfevent/PerfEvent.hpp"
#include "../support/datasets.hpp"
#include "../support/probing_set.hpp"
#include "include/convenience/builtins.hpp"
#include "include/mmphf/rank_hash.hpp"
#include "include/rmi.hpp"

namespace _ {
using Key = std::uint64_t;
using Payload = std::uint64_t;


// const std::vector<std::int64_t> dataset_sizes{100000000};
// const std::vector<std::int64_t> datasets{
//     static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::UNIFORM)
//     // static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::GAPPED_10),
//     // static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::SEQUENTIAL)
//     };
const std::vector<std::int64_t> probe_distributions{
    // static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
    //     dataset::ProbingDistribution::EXPONENTIAL_SORTED),
    // static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
    //     dataset::ProbingDistribution::EXPONENTIAL_RANDOM),
    static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
        dataset::ProbingDistribution::UNIFORM)};

const std::vector<std::int64_t> dataset_sizes{100000000};
const std::vector<std::int64_t> succ_probability{100};
const std::vector<std::int64_t> datasets{
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::SEQUENTIAL),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::GAPPED_10),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::UNIFORM),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::NORMAL),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::WIKI),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::FB)
    // static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::OSM)
    };

const std::vector<std::int64_t> variance_datasets{
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::UNIFORM),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::Variance_2),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::Variance_4),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::Variance_half),
    static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::Variance_quarter)
    // static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::OSM)
    };

// const std::vector<std::int64_t> probe_distributions{
//     static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
//         dataset::ProbingDistribution::UNIFORM),
//     static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
//         dataset::ProbingDistribution::EXPONENTIAL_RANDOM),
//     static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
//         dataset::ProbingDistribution::EXPONENTIAL_SORTED)};

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

std::string previous_signature = "";
std::vector<Key> probing_set{};
void* prev_table = nullptr;
std::function<void()> free_lambda = []() {};

template <class Table, size_t RangeSize>
static void TableProbe(benchmark::State& state) {
  // Extract variables
  const auto dataset_size = static_cast<size_t>(state.range(0));
  const auto did = static_cast<dataset::ID>(state.range(1));
  const auto probing_dist =
      static_cast<dataset::ProbingDistribution>(state.range(2));

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
    std::vector<std::pair<Key, Payload>> data{};
    data.reserve(dataset_size);
    {
      auto keys = dataset::load_cached<Key>(did, dataset_size);

      std::transform(
          keys.begin(), keys.end(), std::back_inserter(data),
          [](const Key& key) { return std::make_pair(key, key - 5); });
      int succ_probability=100;
      probing_set = dataset::generate_probing_set(keys, probing_dist,succ_probability);
    }

    if (data.empty()) {
      // otherwise google benchmark produces an error ;(
      for (auto _ : state) {
      }
      std::cout << "failed" << std::endl;
      return;
    }

    // build table
    if (prev_table != nullptr) free_lambda();
    prev_table = new Table(data);
    free_lambda = []() { delete ((Table*)prev_table); };

    // measure time elapsed
    const auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "succeeded in " << std::setw(9) << diff.count() << " seconds"
              << std::endl;
  }
  previous_signature = signature;

  assert(prev_table != nullptr);
  Table* table = (Table*)prev_table;

  size_t i = 0;
  for (auto _ : state) {
    while (unlikely(i >= probing_set.size())) i -= probing_set.size();
    const auto searched = probing_set[i++];

    // Lower bound lookup
    auto it = table->operator[](
        searched);  // TODO: does this generate a 'call' op? =>
                    // https://stackoverflow.com/questions/10631283/how-will-i-know-whether-inline-function-is-actually-replaced-at-the-place-where

    // RangeSize == 0 -> only key lookup
    // RangeSize == 1 -> key + payload lookup
    // RangeSize  > 1 -> lb key + multiple payload lookups
    if constexpr (RangeSize == 0) {
      benchmark::DoNotOptimize(it);
    } else if constexpr (RangeSize == 1) {
      const auto payload = it.payload();
      benchmark::DoNotOptimize(payload);
    } else if constexpr (RangeSize > 1) {
      Payload total = 0;
      for (size_t i = 0; it != table->end() && i < RangeSize; i++, ++it) {
        total ^= it.payload();
      }
      benchmark::DoNotOptimize(total);
    }
    full_mem_barrier;
  }

  // set counters (don't do this in inner loop to avoid tainting results)
  state.counters["table_bytes"] = table->byte_size();
  state.counters["table_directory_bytes"] = table->directory_byte_size();
  state.counters["table_bits_per_key"] = 8. * table->byte_size() / dataset_size;
  state.counters["data_elem_count"] = dataset_size;
  state.SetLabel(table->name() + ":" + dataset::name(did) + ":" +
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
    std::vector<std::pair<Key, Payload>> data{};
    data.reserve(dataset_size);
    {
      auto keys = dataset::load_cached<Key>(did, dataset_size);

      std::transform(
          keys.begin(), keys.end(), std::back_inserter(data),
          [](const Key& key) { return std::make_pair(key, key - 5); });
      int succ_probability=100;
      probing_set = dataset::generate_probing_set(keys, probing_dist,succ_probability);
    }

    if (data.empty()) {
      // otherwise google benchmark produces an error ;(
      for (auto _ : state) {
      }
      return;
    }

    // build table
    if (prev_table != nullptr) free_lambda();
    prev_table = new Table(data);
    free_lambda = []() { delete ((Table*)prev_table); };

    // measure time elapsed
    const auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "succeeded in " << std::setw(9) << diff.count() << " seconds"
              << std::endl;
  }
  previous_signature = signature;

  assert(prev_table != nullptr);
  Table* table = (Table*)prev_table;

  // distribution
  std::uniform_int_distribution<size_t> point_query_dist(1, 100);

  size_t i = 0;
  for (auto _ : state) {
    while (unlikely(i >= probing_set.size())) i -= probing_set.size();
    const auto searched = probing_set[i++];

    // Lower bound lookup
    auto it = table->operator[](searched);
    if (it != table->end()) {
      const auto lb_payload = it.payload();
      benchmark::DoNotOptimize(lb_payload);

      // Chance based perform full range scan
      if (point_query_dist(rng) > percentage_of_point_queries) {
        ++it;
        Payload total = 0;
        for (size_t i = 1; it != table->end() && i < 10; i++, ++it) {
          total ^= it.payload();
        }
        benchmark::DoNotOptimize(total);
      }
    }

    full_mem_barrier;
  }

  // set counters (don't do this in inner loop to avoid tainting results)
  state.counters["table_bytes"] = table->byte_size();
  state.counters["point_lookup_percent"] =
      static_cast<double>(percentage_of_point_queries) / 100.0;
  state.counters["table_directory_bytes"] = table->directory_byte_size();
  state.counters["table_bits_per_key"] = 8. * table->byte_size() / dataset_size;
  state.counters["data_elem_count"] = dataset_size;
  state.SetLabel(table->name() + ":" + dataset::name(did) + ":" +
                 dataset::name(probing_dist));
}



template <class Table,size_t RangeSize>
static void PointProbe(benchmark::State& state) {
  // Extract variables
  const auto dataset_size = static_cast<size_t>(state.range(0));
  const auto did = static_cast<dataset::ID>(state.range(1));
  const auto probing_dist =
      static_cast<dataset::ProbingDistribution>(state.range(2));
   const auto succ_probability =
      static_cast<size_t>(state.range(3));    

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
    std::vector<std::pair<Key, Payload>> data{};
    data.reserve(dataset_size);
    {
      auto keys = dataset::load_cached<Key>(did, dataset_size);

      std::transform(
          keys.begin(), keys.end(), std::back_inserter(data),
          [](const Key& key) { return std::make_pair(key, key - 5); });
      // int succ_probability=100;
      probing_set = dataset::generate_probing_set(keys, probing_dist,succ_probability);
    }

    if (data.empty()) {
      // otherwise google benchmark produces an error ;(
      for (auto _ : state) {
      }
      std::cout << "failed" << std::endl;
      return;
    }

    // build table
    if (prev_table != nullptr) free_lambda();
    prev_table = new Table(data);
    free_lambda = []() { delete ((Table*)prev_table); };

    // measure time elapsed
    const auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "succeeded in " << std::setw(9) << diff.count() << " seconds"
              << std::endl;
    
    

  }
  

  assert(prev_table != nullptr);
  Table* table = (Table*)prev_table;

  if (previous_signature != signature)
  {
    std::cout<<std::endl<<" Dataset Size: "<<std::to_string(dataset_size) <<" Dataset: "<< dataset::name(did)<<std::endl;
    table->print_data_statistics();
  }

  // std::cout<<"signature swap"<<std::endl;

  previous_signature = signature;  

  // std::cout<<"again?"<<std::endl;

  size_t i = 0;
  for (auto _ : state) {
    // while (unlikely(i >= probing_set.size())) i -= probing_set.size();
    const auto searched = probing_set[i%probing_set.size()];
    i++;

    // Lower bound lookup
    auto it = table->hash_val(searched);
    // auto it = table->operator[](
        // searched);  // TODO: does this generate a 'call' op? =>
                    // https://stackoverflow.com/questions/10631283/how-will-i-know-whether-inline-function-is-actually-replaced-at-the-place-where

    benchmark::DoNotOptimize(it);
    // __sync_synchronize();
    // full_mem_barrier;
  }

  // set counters (don't do this in inner loop to avoid tainting results)
  state.counters["table_bytes"] = table->byte_size();
  state.counters["table_directory_bytes"] = table->directory_byte_size();
  state.counters["table_bits_per_key"] = 8. * table->byte_size() / dataset_size;
  state.counters["data_elem_count"] = dataset_size;

  std::stringstream ss;
  ss << succ_probability;
  std::string temp = ss.str();
  state.SetLabel(table->name() + ":" + dataset::name(did) + ":" +
                 dataset::name(probing_dist)+":"+temp);
}



template <class Table,size_t RangeSize>
static void GapStats(benchmark::State& state) {
  // Extract variables
  const auto dataset_size = static_cast<size_t>(state.range(0));
  const auto did = static_cast<dataset::ID>(state.range(1));
  const auto probing_dist =
      static_cast<dataset::ProbingDistribution>(state.range(2));
   const auto succ_probability =
      static_cast<size_t>(state.range(3));    

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
    std::vector<std::pair<Key, Payload>> data{};
    data.reserve(dataset_size);
    {
      auto keys = dataset::load_cached<Key>(did, dataset_size);

      std::transform(
          keys.begin(), keys.end(), std::back_inserter(data),
          [](const Key& key) { return std::make_pair(key, key - 5); });
      // int succ_probability=100;
      probing_set = dataset::generate_probing_set(keys, probing_dist,succ_probability);
    }

    if (data.empty()) {
      // otherwise google benchmark produces an error ;(
      for (auto _ : state) {
      }
      std::cout << "failed" << std::endl;
      return;
    }

    // build table
    if (prev_table != nullptr) free_lambda();
    prev_table = new Table(data);
    free_lambda = []() { delete ((Table*)prev_table); };

    // measure time elapsed
    const auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "succeeded in " << std::setw(9) << diff.count() << " seconds"
              << std::endl;
    
    

  }
  

  assert(prev_table != nullptr);
  Table* table = (Table*)prev_table;

  if (previous_signature != signature)
  {
    std::cout<<std::endl<<" Dataset Size: "<<std::to_string(dataset_size) <<" Dataset: "<< dataset::name(did)<<std::endl;
    table->gap_stats();
  }

  // std::cout<<"signature swap"<<std::endl;

  previous_signature = signature;  

  // std::cout<<"again?"<<std::endl;

  size_t i = 0;
  for (auto _ : state) {
    // while (unlikely(i >= probing_set.size())) i -= probing_set.size();
    // const auto searched = probing_set[i%probing_set.size()];
    // i++;

    // Lower bound lookup
    auto it = table->useless_func();  // TODO: does this generate a 'call' op? =>
                    // https://stackoverflow.com/questions/10631283/how-will-i-know-whether-inline-function-is-actually-replaced-at-the-place-where

    benchmark::DoNotOptimize(it);
    __sync_synchronize();
    // full_mem_barrier;
  }

  // set counters (don't do this in inner loop to avoid tainting results)
  state.counters["table_bytes"] = table->byte_size();
  state.counters["table_directory_bytes"] = table->directory_byte_size();
  state.counters["table_bits_per_key"] = 8. * table->byte_size() / dataset_size;
  state.counters["data_elem_count"] = dataset_size;

  std::stringstream ss;
  ss << succ_probability;
  std::string temp = ss.str();
  state.SetLabel(table->name() + ":" + dataset::name(did) + ":" +
                 dataset::name(probing_dist)+":"+temp);
}

template <class Table,size_t RangeSize>
static void CollisionStats(benchmark::State& state) {
  // Extract variables
  const auto dataset_size = static_cast<size_t>(state.range(0));
  const auto did = static_cast<dataset::ID>(state.range(1));
  const auto probing_dist =
      static_cast<dataset::ProbingDistribution>(state.range(2));
   const auto succ_probability =
      static_cast<size_t>(state.range(3));    

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
    std::vector<std::pair<Key, Payload>> data{};
    data.reserve(dataset_size);
    {
      auto keys = dataset::load_cached<Key>(did, dataset_size);

      std::transform(
          keys.begin(), keys.end(), std::back_inserter(data),
          [](const Key& key) { return std::make_pair(key, key - 5); });
      // int succ_probability=100;
      probing_set = dataset::generate_probing_set(keys, probing_dist,succ_probability);
    }

    if (data.empty()) {
      // otherwise google benchmark produces an error ;(
      for (auto _ : state) {
      }
      std::cout << "failed" << std::endl;
      return;
    }

    // build table
    if (prev_table != nullptr) free_lambda();
    prev_table = new Table(data);
    free_lambda = []() { delete ((Table*)prev_table); };

    // measure time elapsed
    const auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "succeeded in " << std::setw(9) << diff.count() << " seconds"
              << std::endl;
    
    

  }
  

  assert(prev_table != nullptr);
  Table* table = (Table*)prev_table;

  if (previous_signature != signature)
  {
    std::cout<<std::endl<<" Dataset Size: "<<std::to_string(dataset_size) <<" Dataset: "<< dataset::name(did)<<std::endl;
    table->print_data_statistics();
  }

   if (previous_signature != signature)
  {
    std::cout<<"Probing set size is: "<<probing_set.size()<<std::endl;
    std::cout<<std::endl<<" Dataset Size: "<<std::to_string(dataset_size) <<" Dataset: "<< dataset::name(did)<<std::endl;
    // table->print_data_statistics();

   
      uint64_t total_sum=0;
     auto start = std::chrono::high_resolution_clock::now(); 

    for(int itr=0;itr<probing_set.size()*0.1;itr++)
    {
      const auto searched = probing_set[itr%probing_set.size()];
      // i++;
      total_sum+=table->hash_val(searched);
      // Lower bound lookup
    //  table->insert(searched,searched);  // TODO: does this generate a 'call' op? =>
                      // https://stackoverflow.com/questions/10631283/how-will-i-know-whether-inline-function-is-actually-replaced-at-the-place-where
      
      
      // __sync_synchronize();
    }

     auto stop = std::chrono::high_resolution_clock::now(); 
    // auto duration = duration_cast<milliseconds>(stop - start); 
    auto duration = duration_cast<std::chrono::nanoseconds>(stop - start); 
    std::cout << "Hash Computation is: "<< duration.count()*10.00/probing_set.size() << " nanoseconds" << std::endl;
    std::cout << "Total Sum: "<<total_sum<<std::endl;
  
  }

  // std::cout<<"signature swap"<<std::endl;

  previous_signature = signature;  

  // std::cout<<"again?"<<std::endl;

  size_t i = 0;
  for (auto _ : state) {
    // while (unlikely(i >= probing_set.size())) i -= probing_set.size();
    // const auto searched = probing_set[i%probing_set.size()];
    // i++;

    // // Lower bound lookup
    auto it = table->useless_func();  // TODO: does this generate a 'call' op? =>
    //                 // https://stackoverflow.com/questions/10631283/how-will-i-know-whether-inline-function-is-actually-replaced-at-the-place-where

    // benchmark::DoNotOptimize(it);
    // __sync_synchronize();
    // full_mem_barrier;
  }

  // set counters (don't do this in inner loop to avoid tainting results)
  state.counters["table_bytes"] = table->byte_size();
  state.counters["table_directory_bytes"] = table->directory_byte_size();
  state.counters["table_bits_per_key"] = 8. * table->byte_size() / dataset_size;
  state.counters["data_elem_count"] = dataset_size;

  std::stringstream ss;
  ss << succ_probability;
  std::string temp = ss.str();
  state.SetLabel(table->name() + ":" + dataset::name(did) + ":" +
                 dataset::name(probing_dist)+":"+temp);
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


#define KAPILBM(Table)                                                              \
  BENCHMARK_TEMPLATE(CollisionStats, Table, 0)                                     \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions,succ_probability});



#define KAPILGAPBM(Table)                                                              \
  BENCHMARK_TEMPLATE(GapStats, Table, 0)                                     \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions,succ_probability});

#define KAPILVarianceCollisionBM(Table)                                                              \
  BENCHMARK_TEMPLATE(CollisionStats, Table, 0)                                     \
      ->ArgsProduct({dataset_sizes, variance_datasets, probe_distributions,succ_probability});

#define KAPILCollisionBM(Table)                                                              \
  BENCHMARK_TEMPLATE(CollisionStats, Table, 0)                                     \
      ->ArgsProduct({dataset_sizes, datasets, probe_distributions,succ_probability});




// ############################## Chaining ##############################
// ############################## Chaining ##############################
// ############################## Chaining ##############################


#define BenchmarKapilChained(BucketSize,OverAlloc,HashFn)                           \
  using KapilChainedHashTable##BucketSize##OverAlloc##HashFn = KapilChainedHashTable<Key, Payload, BucketSize,OverAlloc, HashFn>; \
  KAPILBM(KapilChainedHashTable##BucketSize##OverAlloc##HashFn);


#define BenchmarKapilChainedExotic(BucketSize,OverAlloc,MMPHF)                           \
  using KapilChainedExoticHashTable##BucketSize##MMPHF = KapilChainedExoticHashTable<Key, Payload, BucketSize,OverAlloc, MMPHF>; \
  KAPILBM(KapilChainedExoticHashTable##BucketSize##MMPHF);

#define BenchmarKapilChainedModel(BucketSize,OverAlloc,Model)                           \
  using KapilChainedModelHashTable##BucketSize##OverAlloc##Model = KapilChainedModelHashTable<Key, Payload, BucketSize,OverAlloc, Model>; \
  KAPILBM(KapilChainedModelHashTable##BucketSize##OverAlloc##Model);


#define BenchmarKapilGAPChainedModel(BucketSize,OverAlloc,Model)                           \
  using KapilChainedModelHashTable##BucketSize##OverAlloc##Model = KapilChainedModelHashTable<Key, Payload, BucketSize,OverAlloc, Model>; \
  KAPILGAPBM(KapilChainedModelHashTable##BucketSize##OverAlloc##Model);

#define BenchmarKapilVarianceCollisionChainedModel(BucketSize,OverAlloc,Model)                           \
  using KapilVarianceChainedModelHashTable##BucketSize##OverAlloc##Model = KapilChainedModelHashTable<Key, Payload, BucketSize,OverAlloc, Model>; \
  KAPILVarianceCollisionBM(KapilVarianceChainedModelHashTable##BucketSize##OverAlloc##Model);

#define BenchmarKapilCollisionChainedModel(BucketSize,OverAlloc,Model)                           \
  using KapilChainedModelHashTable##BucketSize##OverAlloc##Model = KapilChainedModelHashTable<Key, Payload, BucketSize,OverAlloc, Model>; \
  KAPILCollisionBM(KapilChainedModelHashTable##BucketSize##OverAlloc##Model);  


#define BenchmarKapilCollisionChainedExotic(BucketSize,OverAlloc,Model)                           \
  using KapilChainedExoticHashTable##BucketSize##OverAlloc##Model = KapilChainedExoticHashTable<Key, Payload, BucketSize, Model>; \
  KAPILCollisionBM(KapilChainedExoticHashTable##BucketSize##OverAlloc##Model);  


/////////////////GAP EXPTS/////////////////
// using RMIHash = learned_hashing::RMIHash<std::uint64_t,1000>;
// BenchmarKapilGAPChainedModel(1,0,RMIHash);

/////////////////VARIANCE EXPTS/////////////////
// using RMIHash = learned_hashing::RMIHash<std::uint64_t,1000000>;
// BenchmarKapilVarianceCollisionChainedModel(1,10050,RMIHash);

// using RMIHash = learned_hashing::RMIHash<std::uint64_t,1000000>;
// BenchmarKapilVarianceCollisionChainedModel(1,10075,RMIHash);

// using RMIHash = learned_hashing::RMIHash<std::uint64_t,1000000>;
// BenchmarKapilVarianceCollisionChainedModel(1,0,RMIHash);

// using RMIHash = learned_hashing::RMIHash<std::uint64_t,1000000>;
// BenchmarKapilVarianceCollisionChainedModel(1,50,RMIHash);

// using RMIHash = learned_hashing::RMIHash<std::uint64_t,1000000>;
// BenchmarKapilVarianceCollisionChainedModel(1,100,RMIHash);

// using RMIHash = learned_hashing::RMIHash<std::uint64_t,1000000>;
// BenchmarKapilVarianceCollisionChainedModel(1,300,RMIHash);

// using RMIHash = learned_hashing::RMIHash<std::uint64_t,1000000>;
// BenchmarKapilVarianceCollisionChainedModel(1,700,RMIHash);


/////////////////MODEL SIZE EXPTS/////////////////

// using RMIHash1 = learned_hashing::RMIHash<std::uint64_t,1>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash1);

// using RMIHash2 = learned_hashing::RMIHash<std::uint64_t,10>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash2);

// using RMIHash3 = learned_hashing::RMIHash<std::uint64_t,25>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash3);

// using RMIHash4 = learned_hashing::RMIHash<std::uint64_t,100>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash4);

// using RMIHash5 = learned_hashing::RMIHash<std::uint64_t,1000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash5);

// using RMIHash6 = learned_hashing::RMIHash<std::uint64_t,10000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash6);

// using RMIHash7 = learned_hashing::RMIHash<std::uint64_t,100000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash7);


// using RMIHash8 = learned_hashing::RMIHash<std::uint64_t,1000000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash8);

// using RMIHash9 = learned_hashing::RMIHash<std::uint64_t,10000000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash9);

// using RMIHash10 = learned_hashing::RMIHash<std::uint64_t,50000000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash10);


// using RadixSplineHash1 = learned_hashing::RadixSplineHash<std::uint64_t,18,100000>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash1);

// using RadixSplineHash2 = learned_hashing::RadixSplineHash<std::uint64_t,18,1024>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash2);

// using RadixSplineHash3 = learned_hashing::RadixSplineHash<std::uint64_t,18,256>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash3);

// using RadixSplineHash4 = learned_hashing::RadixSplineHash<std::uint64_t,18,128>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash4);

// using RadixSplineHash5 = learned_hashing::RadixSplineHash<std::uint64_t,18,32>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash5);

// using RadixSplineHash6 = learned_hashing::RadixSplineHash<std::uint64_t,18,16>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash6);

// using RadixSplineHash7 = learned_hashing::RadixSplineHash<std::uint64_t,18,8>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash7);

// using RadixSplineHash8 = learned_hashing::RadixSplineHash<std::uint64_t,18,4>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash8);

// using RadixSplineHash9 = learned_hashing::RadixSplineHash<std::uint64_t,18,2>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash9);


// using PGMHash1 = learned_hashing::PGMHash<std::uint64_t,100000,100000,500000000,float>;
// BenchmarKapilCollisionChainedModel(1,0,PGMHash1);

// using PGMHash2 = learned_hashing::PGMHash<std::uint64_t,1024,1024,500000000,float>;
// BenchmarKapilCollisionChainedModel(1,0,PGMHash2);

// using PGMHash3 = learned_hashing::PGMHash<std::uint64_t,128,128,500000000,float>;
// BenchmarKapilCollisionChainedModel(1,0,PGMHash3);

// using PGMHash4 = learned_hashing::PGMHash<std::uint64_t,32,32,500000000,float>;
// BenchmarKapilCollisionChainedModel(1,0,PGMHash4);

// using PGMHash5 = learned_hashing::PGMHash<std::uint64_t,2,2,500000000,float>;
// BenchmarKapilCollisionChainedModel(1,0,PGMHash5);


/////////////////HASH COMPUTE EXPTS/////////////////

// using MURMUR = hashing::MurmurFinalizer<Key>;
// BenchmarKapilChained(1,0,MURMUR);

// using MultPrime64 = hashing::MultPrime64;
// BenchmarKapilChained(1,0,MultPrime64);

// using FibonacciPrime64 = hashing::FibonacciPrime64;
// BenchmarKapilChained(1,0,FibonacciPrime64);

// using AquaHash = hashing::AquaHash<Key>;
// BenchmarKapilChained(1,0,AquaHash);

// using XXHash3 = hashing::XXHash3<Key>;
// BenchmarKapilChained(1,0,XXHash3);

// using MWHC = exotic_hashing::MWHC<Key>;
// BenchmarKapilChainedExotic(1,20,MWHC);

// using BitMWHC = exotic_hashing::BitMWHC<Key>;
// BenchmarKapilChainedExotic(1,20,BitMWHC);

// using FST = exotic_hashing::FastSuccinctTrie<Data>;
// BenchmarKapilChainedExotic(1,20,FST);

// using CompressedMWHC = exotic_hashing::CompressedMWHC<Key>;
// BenchmarKapilChainedExotic(1,20,CompressedMWHC);

// using RankHash = exotic_hashing::RankHash<Key>;
// BenchmarKapilChainedExotic(1,20,RankHash);



// using RMIHash1 = learned_hashing::RMIHash<std::uint64_t,1>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash1);

// using RMIHash2 = learned_hashing::RMIHash<std::uint64_t,10>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash2);

// using RMIHash3 = learned_hashing::RMIHash<std::uint64_t,25>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash3);

// using RMIHash4 = learned_hashing::RMIHash<std::uint64_t,100>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash4);

// using RMIHash5 = learned_hashing::RMIHash<std::uint64_t,1000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash5);

// using RMIHash6 = learned_hashing::RMIHash<std::uint64_t,10000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash6);

// using RMIHash7 = learned_hashing::RMIHash<std::uint64_t,100000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash7);


// using RMIHash8 = learned_hashing::RMIHash<std::uint64_t,1000000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash8);

// using RMIHash9 = learned_hashing::RMIHash<std::uint64_t,10000000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash9);

// using RMIHash10 = learned_hashing::RMIHash<std::uint64_t,50000000>;
// BenchmarKapilCollisionChainedModel(1,0,RMIHash10);


// using RadixSplineHash1 = learned_hashing::RadixSplineHash<std::uint64_t,18,100000>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash1);

// using RadixSplineHash2 = learned_hashing::RadixSplineHash<std::uint64_t,18,1024>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash2);

// using RadixSplineHash3 = learned_hashing::RadixSplineHash<std::uint64_t,18,256>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash3);

// using RadixSplineHash4 = learned_hashing::RadixSplineHash<std::uint64_t,18,128>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash4);

// using RadixSplineHash5 = learned_hashing::RadixSplineHash<std::uint64_t,18,32>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash5);

// using RadixSplineHash6 = learned_hashing::RadixSplineHash<std::uint64_t,18,16>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash6);

// using RadixSplineHash7 = learned_hashing::RadixSplineHash<std::uint64_t,18,8>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash7);

// using RadixSplineHash8 = learned_hashing::RadixSplineHash<std::uint64_t,18,4>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash8);

// using RadixSplineHash9 = learned_hashing::RadixSplineHash<std::uint64_t,18,2>;
// BenchmarKapilCollisionChainedModel(1,0,RadixSplineHash9);








using PGMHash4 = learned_hashing::PGMHash<std::uint64_t,32,32,500000000,float>;
BenchmarKapilCollisionChainedModel(1,0,PGMHash4);

using PGMHash5 = learned_hashing::PGMHash<std::uint64_t,2,2,500000000,float>;
BenchmarKapilCollisionChainedModel(1,0,PGMHash5);

using PGMHash3 = learned_hashing::PGMHash<std::uint64_t,128,128,500000000,float>;
BenchmarKapilCollisionChainedModel(1,0,PGMHash3);

using PGMHash2 = learned_hashing::PGMHash<std::uint64_t,1024,1024,500000000,float>;
BenchmarKapilCollisionChainedModel(1,0,PGMHash2);

using PGMHash1 = learned_hashing::PGMHash<std::uint64_t,10000,10000,500000000,float>;
BenchmarKapilCollisionChainedModel(1,0,PGMHash1);



}  // namespace _

