# pragma once

/* Algorithm utilities */

#include "emmintrin.h"
#include "immintrin.h"
#include "smmintrin.h"

#include <vector>

#include "utils/data_structures.h"

// Function adapted from https://github.com/gvinciguerra/rmi_pgm/blob/357acf668c22f927660d6ed11a15408f722ea348/main.cpp#L29.
// Authored by Giorgio Vinciguerra. 
template<class KeyType>
inline uint64_t branchless_binary_search(const std::vector<KeyType>& keys, const KeyType lookup_key, uint64_t start, uint64_t end);

template<class KeyType>
inline uint64_t branchless_binary_search(const std::vector<KeyType>& keys, const KeyType lookup_key, uint64_t start, uint64_t end)
{
    // Search for first occurrence of key.
    int n = end - start + 1; // `end` is inclusive.
    
//    if (n < 32) {
//      // Do linear search over narrowed range.
//      uint64_t current = start;
//      while (keys[current] < lookup_key) ++current;
//      return current;
//    }

    uint64_t lower = start;
    while (const int half = n / 2) {
      const int middle = lower + half;
      // Prefetch next two middles.
      __builtin_prefetch(&(keys[lower + half / 2]), 0, 0);
      __builtin_prefetch(&(keys[middle + half / 2]), 0, 0);
      lower = (keys[middle] <= lookup_key) ? middle : lower;
      n -= half;
    }

    // Scroll back to the first occurrence.
    while (lower > 0 && keys[lower - 1] == lookup_key) --lower;

    return lower;
}
