#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "../include/rmi_hashtable.hpp"

TEST(RMIHashtable, Lookup1_32) {
  using Key = std::uint32_t;
  using Payload = std::uint32_t;
  const std::vector<Key> keys{0, 1, 5, 8, 9, 10};
  const std::vector<Key> non_keys{2, 3, 4, 6, 7};
  const std::vector<Payload> payloads{3, 7, 13, 42, 69, 1337};

  RMIHashtable<Key, Payload, 1, 100> ht(keys, payloads);
  for (size_t i = 0; i < keys.size(); i++)
    EXPECT_EQ(ht.lookup(keys[i]), payloads[i]);

  for (size_t i = 0; i < non_keys.size(); i++)
    EXPECT_EQ(ht.lookup(non_keys[i]), std::numeric_limits<Payload>::max());
}

TEST(RMIHashtable, Lookup2_32) {
  using Key = std::uint32_t;
  using Payload = std::uint32_t;
  const std::vector<Key> keys{0, 1, 5, 8, 9, 10};
  const std::vector<Key> non_keys{2, 3, 4, 6, 7};
  const std::vector<Payload> payloads{3, 7, 13, 42, 69, 1337};

  RMIHashtable<Key, Payload, 2, 100> ht(keys, payloads);
  for (size_t i = 0; i < keys.size(); i++)
    EXPECT_EQ(ht.lookup(keys[i]), payloads[i]);

  for (size_t i = 0; i < non_keys.size(); i++)
    EXPECT_EQ(ht.lookup(non_keys[i]), std::numeric_limits<Payload>::max());
}

TEST(RMIHashtable, Lookup4_32) {
  using Key = std::uint32_t;
  using Payload = std::uint32_t;
  const std::vector<Key> keys{0, 1, 5, 8, 9, 10};
  const std::vector<Key> non_keys{2, 3, 4, 6, 7};
  const std::vector<Payload> payloads{3, 7, 13, 42, 69, 1337};

  RMIHashtable<Key, Payload, 4, 100> ht(keys, payloads);
  for (size_t i = 0; i < keys.size(); i++)
    EXPECT_EQ(ht.lookup(keys[i]), payloads[i]);

  for (size_t i = 0; i < non_keys.size(); i++)
    EXPECT_EQ(ht.lookup(non_keys[i]), std::numeric_limits<Payload>::max());
}

TEST(RMIHashtable, Lookup8_32) {
  using Key = std::uint32_t;
  using Payload = std::uint32_t;
  const std::vector<Key> keys{0, 1, 5, 8, 9, 10};
  const std::vector<Key> non_keys{2, 3, 4, 6, 7};
  const std::vector<Payload> payloads{3, 7, 13, 42, 69, 1337};

  RMIHashtable<Key, Payload, 8, 100> ht(keys, payloads);
  for (size_t i = 0; i < keys.size(); i++)
    EXPECT_EQ(ht.lookup(keys[i]), payloads[i]);

  for (size_t i = 0; i < non_keys.size(); i++)
    EXPECT_EQ(ht.lookup(non_keys[i]), std::numeric_limits<Payload>::max());
}

TEST(RMIHashtable, Lookup1_64) {
  using Key = std::uint64_t;
  using Payload = std::uint64_t;
  const std::vector<Key> keys{0, 1, 5, 8, 9, 10};
  const std::vector<Key> non_keys{2, 3, 4, 6, 7};
  const std::vector<Payload> payloads{3, 7, 13, 42, 69, 1337};

  RMIHashtable<Key, Payload, 1, 100> ht(keys, payloads);
  for (size_t i = 0; i < keys.size(); i++)
    EXPECT_EQ(ht.lookup(keys[i]), payloads[i]);

  for (size_t i = 0; i < non_keys.size(); i++)
    EXPECT_EQ(ht.lookup(non_keys[i]), std::numeric_limits<Payload>::max());
}

TEST(RMIHashtable, Lookup2_64) {
  using Key = std::uint64_t;
  using Payload = std::uint64_t;
  const std::vector<Key> keys{0, 1, 5, 8, 9, 10};
  const std::vector<Key> non_keys{2, 3, 4, 6, 7};
  const std::vector<Payload> payloads{3, 7, 13, 42, 69, 1337};

  RMIHashtable<Key, Payload, 2, 100> ht(keys, payloads);
  for (size_t i = 0; i < keys.size(); i++)
    EXPECT_EQ(ht.lookup(keys[i]), payloads[i]);

  for (size_t i = 0; i < non_keys.size(); i++)
    EXPECT_EQ(ht.lookup(non_keys[i]), std::numeric_limits<Payload>::max());
}

TEST(RMIHashtable, Lookup4_64) {
  using Key = std::uint64_t;
  using Payload = std::uint64_t;
  const std::vector<Key> keys{0, 1, 5, 8, 9, 10};
  const std::vector<Key> non_keys{2, 3, 4, 6, 7};
  const std::vector<Payload> payloads{3, 7, 13, 42, 69, 1337};

  RMIHashtable<Key, Payload, 4, 100> ht(keys, payloads);
  for (size_t i = 0; i < keys.size(); i++)
    EXPECT_EQ(ht.lookup(keys[i]), payloads[i]);

  for (size_t i = 0; i < non_keys.size(); i++)
    EXPECT_EQ(ht.lookup(non_keys[i]), std::numeric_limits<Payload>::max());
}

TEST(RMIHashtable, Lookup8_64) {
  using Key = std::uint64_t;
  using Payload = std::uint64_t;
  const std::vector<Key> keys{0, 1, 5, 8, 9, 10};
  const std::vector<Key> non_keys{2, 3, 4, 6, 7};
  const std::vector<Payload> payloads{3, 7, 13, 42, 69, 1337};

  RMIHashtable<Key, Payload, 8, 100> ht(keys, payloads);
  for (size_t i = 0; i < keys.size(); i++)
    EXPECT_EQ(ht.lookup(keys[i]), payloads[i]);

  for (size_t i = 0; i < non_keys.size(); i++)
    EXPECT_EQ(ht.lookup(non_keys[i]), std::numeric_limits<Payload>::max());
}
