#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

static void BM_SortedArrayRangeLookup(benchmark::State& state) {
  using Key = std::uint64_t;
  using Payload = std::uint64_t;
  struct Slot {
    Key key = std::numeric_limits<Key>::max();
    Payload payload = 0;

    bool operator<(const Slot& rhs) const { return this->key < rhs.key; }
    bool operator<(const Key& rhs) const { return this->key < rhs; }
  } __attribute__((packed));

  std::vector<Slot> dataset(1000000);

  // Random generator
  std::random_device rd;
  std::default_random_engine rng(rd());

  // Generate dataset with random holes
  // TODO: dataset should be a variable benchmark parameter
  // TODO: use different datasets
  {
    std::uniform_int_distribution<size_t> dist(0, 99);
    for (size_t idx = 0, num = 0; idx < dataset.size(); idx++, num++) {
      while (dist(rng) < 90) num++;
      dataset[idx].key = num;
      dataset[idx].payload = num - 1;
    }
  }

  // Dataset is already sorted by construction, however,
  // this will not be the case for all datasets in the future
  std::sort(dataset.begin(), dataset.end());

  // Random lookups
  const auto min_key = dataset[0].key;
  const auto max_key = dataset[dataset.size() - 1].key;
  std::uniform_int_distribution<size_t> dist(min_key, max_key);
  // TODO: interval_size should be a benchmark parameter
  // TODO: use different interval sizes
  const size_t interval_size = 1000;

  const auto lower_bound = [&](const Key& val) {
    // TODO: use implementation other than (probably) binary search, e.g.,
    // linear interpolation search, exponential search etc
    return std::lower_bound(dataset.begin(), dataset.end(), val);
  };

  for (auto _ : state) {
    const auto lower = dist(rng);
    const auto upper = lower + interval_size;

    std::vector<Payload> result;
    for (auto iter = lower_bound(lower);
         // TODO: use key <= upper?
         iter < dataset.end() && iter->key < upper; iter++)
      result.push_back(iter->payload);

    benchmark::DoNotOptimize(result.data());
  }
}

BENCHMARK(BM_SortedArrayRangeLookup);

BENCHMARK_MAIN();
