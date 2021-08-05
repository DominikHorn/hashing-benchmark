#include <benchmark/benchmark.h>

#include <algorithm>
#include <hashtable.hpp>
#include <iostream>
#include <learned_hashing.hpp>
#include <limits>
#include <random>
#include <vector>

namespace _kapil_benchmark {
using Key = std::uint64_t;
using Payload = std::uint64_t;

const size_t dataset_size = 100000;
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
        for (size_t idx = 0, num = 0; idx < ds_uniform_random.size();
             idx++, num++) {
          while (dist(rng) < 90) num++;
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
      result.push_back(*iter - 1);

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
      result.push_back(*iter - 1);

    benchmark::DoNotOptimize(result.data());
  }
}

BENCHMARK(BM_SortedArrayRangeLookupBinarySearch)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});

BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 10)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 100)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 1000)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 100000)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
BENCHMARK_TEMPLATE(BM_SortedArrayRangeLookupRMI, 1000000)
    ->RangeMultiplier(2)
    ->Ranges({{interval_min, interval_max}, {dataset_min, dataset_max}});
}  // namespace _kapil_benchmark

