#include <benchmark/benchmark.h>

#include "sorted_rmi.hpp"

BENCHMARK_MAIN();

// #include <chrono>
// #include <cstdint>
//
// #include "../include/rmi_hashtable.hpp"
// #include "support/datasets.hpp"
//
// int main() {
//   using T = std::uint32_t;
//   auto dataset = dataset::load_cached<T>(dataset::ID::UNIFORM, 190000000);
//
//   std::random_device rd;
//   std::default_random_engine rng(rd());
//
//   // build hashtable
//   RMIHashtable<T, T, 8, 0> ht(dataset, dataset);
//
//   // probe order
//   auto probe_order = dataset;
//   std::shuffle(probe_order.begin(), probe_order.end(), rng);
//
//   for (size_t iter = 0; iter < 5; iter++) {
//     size_t hits = 0;
//     const auto start_time = std::chrono::steady_clock::now();
//     for (size_t i = 0; i < probe_order.size(); i++) {
//       hits += ht.lookup(probe_order[i]) == probe_order[i];
//     }
//     const auto end_time = std::chrono::steady_clock::now();
//     const auto delta_ns = static_cast<uint64_t>(
//         std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
//                                                              start_time)
//             .count());
//     std::cout << static_cast<double>(delta_ns) /
//                      static_cast<double>(dataset.size())
//               << "; hits: " << hits << std::endl;
//   }
//   return 0;
// }
