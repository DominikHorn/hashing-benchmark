#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <masters_thesis.hpp>

TEST(MMPHFTable, Lookup) {
  using Key = std::uint64_t;
  using Payload = std::uint64_t;
  const std::vector<std::pair<Key, Payload>> data{{0, 3},  {1, 7},  {5, 13},
                                                  {8, 42}, {9, 69}, {10, 1337}};

  const masters_thesis::MMPHFTable<Key, Payload> ht(data);
  for (const auto& d : data) {
    const auto it = ht[d.first];
    EXPECT_TRUE(it != ht.end());

    const Payload payload = it.payload();
    EXPECT_EQ(payload, d.second);
  }
}

TEST(MMPHFTable, Iterate) {
  using Key = std::uint64_t;
  using Payload = std::uint64_t;
  const std::vector<std::pair<Key, Payload>> data{{0, 3},  {1, 7},  {5, 13},
                                                  {8, 42}, {9, 69}, {10, 1337}};

  const masters_thesis::MMPHFTable<Key, Payload> ht(data);
  for (size_t i = 0; i < data.size(); i++) {
    auto it = ht[data[i].first];
    for (size_t j = i; it != ht.end(); ++it, j++) {
      EXPECT_TRUE(it != ht.end());

      const Payload payload = it.payload();
      EXPECT_EQ(payload, data[j].second);
    }
  }
}
