#pragma once

#include <cstdint>
#include <vector>

#include "include/convenience/builtins.hpp"

namespace masters_thesis::support {
/**
 * Tape aims to tremendously improve overall performance
 * for allocating small entities T. Usually, mallocing
 * each entity separately will most likely result in
 * high memory fragmentation which can affect runtime
 * performance due to poor locality.
 *
 * Another, obvious issue is the performance overhead
 * of mallocing each entity T separately.
 *
 * A better approach, i.e., the one implemented by Tape,
 * is to strategically overallocate and to subsequently
 * consume the additional available memory in future calls.
 */
template <class T>
class Tape {
  std::vector<T*> begins;
  size_t index;
  size_t size;

 public:
  constexpr Tape() = default;

  ~Tape() {
    for (auto begin : begins) delete[] begin;
  }

  /**
   * Similar to new, however aims to reduce malloc calls
   * by overallocating and then subsequently reusing additional
   * available memory for future calls
   *
   * @param next_tape_segment_size size of the next tape segment
   *  should a new allocation be neccessary. Defaults to a value such
   *  that a <= 4MB is allocated
   */
  forceinline T* alloc(
      size_t next_tape_segment_size = (4LLU * 1024LLU * 1024LLU) / sizeof(T)) {
    if (unlikely(index == size || begins.size() == 0 ||
                 begins[begins.size() - 1] == nullptr)) {
      begins.push_back(new T[next_tape_segment_size]);
      index = 0;
      size = next_tape_segment_size;
    }

    return &begins.back()[index++];
  }
};

}  // namespace masters_thesis::support
