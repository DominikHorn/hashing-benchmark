#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <masters_thesis.hpp>

namespace _ {
template <class Key, class Payload, class Hashtable>
static void LookupTest() {
  const std::vector<std::pair<Key, Payload>> data{{0, 3},  {1, 7},  {5, 13},
                                                  {8, 42}, {9, 69}, {10, 1337}};
  const std::vector<Key> non_keys{2, 3, 4, 6, 7};

  const Hashtable ht(data);
  for (const auto& d : data) {
    const auto it = ht[d.first];
    EXPECT_TRUE(it != ht.end());

    const Payload payload = it.payload();
    EXPECT_EQ(payload, d.second);
  }

  for (size_t i = 0; i < non_keys.size(); i++)
    EXPECT_TRUE(ht[non_keys[i]] == ht.end());
}
};  // namespace _

TEST(MonotoneHashtable, Lookup) {
  // BucketSize=1, 32bit keys
  _::LookupTest<
      std::uint32_t, std::uint32_t,
      masters_thesis::MonotoneHashtable<std::uint32_t, std::uint32_t, 1>>();

  // BucketSize=2, 32bit keys
  _::LookupTest<
      std::uint32_t, std::uint32_t,
      masters_thesis::MonotoneHashtable<std::uint32_t, std::uint32_t, 2>>();

  // BucketSize=4, 32bit keys
  _::LookupTest<
      std::uint32_t, std::uint32_t,
      masters_thesis::MonotoneHashtable<std::uint32_t, std::uint32_t, 4>>();

  // BucketSize=8, 32bit keys
  _::LookupTest<
      std::uint32_t, std::uint32_t,
      masters_thesis::MonotoneHashtable<std::uint32_t, std::uint32_t, 8>>();

  // BucketSize=1, 64bit keys
  _::LookupTest<
      std::uint64_t, std::uint64_t,
      masters_thesis::MonotoneHashtable<std::uint64_t, std::uint64_t, 1>>();

  // BucketSize=2, 64bit keys
  _::LookupTest<
      std::uint64_t, std::uint64_t,
      masters_thesis::MonotoneHashtable<std::uint64_t, std::uint64_t, 2>>();

  // BucketSize=4, 64bit keys
  _::LookupTest<
      std::uint64_t, std::uint64_t,
      masters_thesis::MonotoneHashtable<std::uint64_t, std::uint64_t, 4>>();

  // BucketSize=8, 64bit keys
  _::LookupTest<
      std::uint64_t, std::uint64_t,
      masters_thesis::MonotoneHashtable<std::uint64_t, std::uint64_t, 8>>();
}

TEST(MonotoneHashtable, Iterate) {
  using Key = std::uint32_t;
  using Payload = std::uint32_t;
  const std::vector<std::pair<Key, Payload>> data{{0, 3},  {1, 7},  {5, 13},
                                                  {8, 42}, {9, 69}, {10, 1337}};

  const masters_thesis::MonotoneHashtable<Key, Payload, 8> ht(data);
  for (size_t i = 0; i < data.size(); i++) {
    auto it = ht[data[i].first];
    for (size_t j = i; it != ht.end(); ++it, j++) {
      EXPECT_TRUE(it != ht.end());
      EXPECT_LT(j, data.size());

      const Payload payload = it.payload();
      EXPECT_EQ(payload, data[j].second);
    }
  }
}
