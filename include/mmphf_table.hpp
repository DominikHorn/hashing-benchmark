#pragma once

#include <exotic_hashing.hpp>
#include <learned_hashing.hpp>

// Order is important
#include "include/convenience/builtins.hpp"

namespace masters_thesis {

template <class Key, class Payload,
          class MMPHF = exotic_hashing::LearnedRank<
              Key, learned_hashing::MonotoneRMIHash<Key, 1000000>>>
class MMPHFTable {
  std::vector<std::pair<Key, Payload>> payloads;
  MMPHF mmphf;

 public:
  MMPHFTable() noexcept = default;

  /**
   * Constructs a MMPHFtable given a list of keys
   * together with their corresponding payloads
   */
  MMPHFTable(std::vector<std::pair<Key, Payload>> data) {
    // ensure data is sorted
    std::sort(data.begin(), data.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // store list of payloads
    payloads = data;

    // obtain list of keys to construct mmphf
    std::vector<Key> keys;
    keys.reserve(data.size());
    std::transform(data.begin(), data.end(), std::back_inserter(keys),
                   [](const auto& p) { return p.first; });

    // construct mmphf on keys
    mmphf.construct(keys.begin(), keys.end());
  }

  class Iterator {
    size_t payloads_ind;
    const std::vector<std::pair<Key, Payload>>& payloads;

    Iterator(size_t i, const decltype(payloads)& payloads)
        : payloads_ind(i), payloads(payloads) {}

   public:
    forceinline const Key& key() const { return payloads[payloads_ind].first; }

    forceinline const Payload& payload() const {
      return payloads[payloads_ind].second;
    }

    forceinline Iterator& operator++() {
      assert(payloads_ind < payloads.size());
      payloads_ind++;
      return *this;
    }

    // // TODO(dominik): support postfix increment
    // forceinline Iterator operator++(int) {
    //   Iterator old = *this;
    //   ++this;
    //   return old;
    // }

    forceinline bool operator==(const Iterator& other) const {
      return payloads_ind == other.payloads_ind && &payloads == &other.payloads;
    }

    friend MMPHFTable;
  };

  /**
   * Past the end iterator, use like usual in stl
   */
  forceinline Iterator end() const { return {payloads.size(), payloads}; }

  forceinline Iterator operator[](const Key& key) const {
    const auto index = mmphf(key);
    if (payloads[index].first != key) return end();

    return {index, payloads};
  }

  std::string name() {
    return "MMPHFtable<" + std::to_string(sizeof(Key)) + ", " +
           std::to_string(sizeof(Payload)) + ", " + mmphf.name() + ">";
  }

  size_t mmphf_byte_size() const { return mmphf.byte_size(); }

  size_t directory_byte_size() const {
    return payloads.size() * sizeof(typename decltype(payloads)::value_type) +
           sizeof(decltype(payloads));
  }

  size_t byte_size() const { return mmphf_byte_size() + directory_byte_size(); }
};
}  // namespace masters_thesis
