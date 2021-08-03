#include <benchmark/benchmark.h>

#include <iostream>

static void BM_SortedArrayRangeLookup(benchmark::State& state) {
  for (auto _ : state) {
    std::cout << "hello world" << std::endl;
  }
}

BENCHMARK(BM_SortedArrayRangeLookup);

BENCHMARK_MAIN();
