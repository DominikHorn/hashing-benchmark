#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <masters_thesis.hpp>

namespace _ {
template <class Key, class Payload, class Hashtable>
static void LookupTest() {
  const std::vector<Key> keys{0, 1, 5, 8, 9, 10};
  const std::vector<Key> non_keys{2, 3, 4, 6, 7};
  const std::vector<Payload> payloads{3, 7, 13, 42, 69, 1337};

  const Hashtable ht(keys, payloads);
  for (size_t i = 0; i < keys.size(); i++)
    EXPECT_EQ(ht.lookup(keys[i]), payloads[i]);

  for (size_t i = 0; i < non_keys.size(); i++)
    EXPECT_EQ(ht.lookup(non_keys[i]), std::numeric_limits<Payload>::max());
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
