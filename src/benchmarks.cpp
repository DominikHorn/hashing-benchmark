#include <benchmark/benchmark.h>

// TODO(dominik): these imports merely serve to dogfood importing these
// libraries (see if everything works)
#include <cstdint>
#include <exotic_hashing.hpp>

#include "sorted_rmi.hpp"
#include "support/datasets.hpp"

// BENCHMARK_MAIN();

int main() {
  auto dataset = dataset::load_cached(dataset::UNIFORM, 200000000);

  std::random_device rd;
  std::default_random_engine rng(rd());

  // generate payloads
  auto payloads = dataset;
  std::shuffle(payloads.begin(), payloads.end(), rng);

  // build hashtable
  RMIHashtable<std::uint64_t, std::uint64_t, 1, 0> ht(dataset, payloads);

  // benchmark
  PerfEvent e;
  e.startCounters();
  for (size_t i = 0; i < dataset.size(); i++) {
    const auto payload = ht.lookup(dataset[i]);
    benchmark::DoNotOptimize(payload);
  }
  e.stopCounters();
  e.printReport(std::cout, dataset.size());

  return 0;
}
