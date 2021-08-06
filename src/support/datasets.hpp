#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// TODO: where does this include come from lol?
#include "include/convenience/builtins.hpp"

class Dataset {
  /**
   * Deduplicates the dataset. Data will be sorted to make this work
   * @param dataset
   */
  static forceinline void deduplicate(std::vector<uint64_t>& dataset) {
    std::sort(dataset.begin(), dataset.end());
    dataset.erase(std::unique(dataset.begin(), dataset.end()), dataset.end());
    dataset.shrink_to_fit();
  }

  /**
   * Shuffles the given dataset
   * @param dataset
   * @param seed
   */
  static forceinline void shuffle(
      std::vector<uint64_t>& dataset,
      const uint64_t seed = std::random_device()()) {
    if (dataset.empty()) return;

    std::default_random_engine gen(seed);
    std::uniform_int_distribution<uint64_t> dist(0);

    // Fisher-Yates shuffle
    for (size_t i = dataset.size() - 1; i > 0; i--) {
      std::swap(dataset[i], dataset[dist(gen) % i]);
    }
  }

  /**
   * Helper to extract an 8 byte number encoded in little endian from a byte
   * vector
   */
  static forceinline uint64_t read_little_endian_8(
      const std::vector<unsigned char>& buffer, uint64_t offset) {
    return static_cast<uint64_t>(buffer[offset + 0]) |
           (static_cast<uint64_t>(buffer[offset + 1]) << 8) |
           (static_cast<uint64_t>(buffer[offset + 2]) << (2 * 8)) |
           (static_cast<uint64_t>(buffer[offset + 3]) << (3 * 8)) |
           (static_cast<uint64_t>(buffer[offset + 4]) << (4 * 8)) |
           (static_cast<uint64_t>(buffer[offset + 5]) << (5 * 8)) |
           (static_cast<uint64_t>(buffer[offset + 6]) << (6 * 8)) |
           (static_cast<uint64_t>(buffer[offset + 7]) << (7 * 8));
  }

  /**
   * Helper to extract a 4 byte number encoded in little endian from a byte
   * vector
   */
  static forceinline uint64_t read_little_endian_4(
      const std::vector<unsigned char>& buffer, uint64_t offset) {
    return buffer[offset + 0] | (buffer[offset + 1] << 8) |
           (buffer[offset + 2] << (2 * 8)) | (buffer[offset + 3] << (3 * 8));
  }

  /**
   * Loads the datasets values into memory
   * @return a sorted and deduplicated list of all members of the dataset
   */
  template <class Key>
  static std::vector<Key> load(std::string filepath) {
    // Read file into memory from disk. Directly map file for more performance
    std::ifstream input(filepath, std::ios::binary | std::ios::ate);
    std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (!input.is_open()) {
      std::cerr << "file '" + filepath + "' does not exist" << std::endl;
      return {};
    }

    const auto max_num_elements = (size - sizeof(std::uint64_t)) / sizeof(Key);
    std::vector<uint64_t> dataset(max_num_elements, 0);
    {
      std::vector<unsigned char> buffer(size);
      if (!input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Failed to read dataset '" + filepath + "'" << std::endl;
        return {};
      }

      // Parse file
      uint64_t num_elements = read_little_endian_8(buffer, 0);
      assert(num_elements <= max_num_elements);
      switch (sizeof(Key)) {
        case sizeof(std::uint64_t):
          for (uint64_t i = 0; i < num_elements; i++) {
            // 8 byte header, 8 bytes per entry
            uint64_t offset = i * 8 + 8;
            dataset[i] = read_little_endian_8(buffer, offset);
          }
          break;
        case sizeof(std::uint32_t):
          for (uint64_t i = 0; i < num_elements; i++) {
            // 8 byte header, 4 bytes per entry
            uint64_t offset = i * 4 + 8;
            dataset[i] = read_little_endian_4(buffer, offset);
          }
          break;
        default:
          std::cerr << "Unimplemented amount of bytes per value in dataset: " +
                           std::to_string(sizeof(Key))
                    << std::endl;
          return {};
      }
    }

    // remove duplicates from dataset and put it into random order
    deduplicate(dataset);
    shuffle(dataset);

    return dataset;
  }

 public:
  enum ID {
    GAPPED_10 = 1,
    SEQUENTIAL = 2,
    UNIFORM = 3,
    FB = 4,
    OSM = 5,
    WIKI = 6
  };

  static std::vector<std::uint64_t> load_cached(
      ID id, size_t dataset_size = 100000000) {
    static std::random_device rd;
    static std::default_random_engine rng(rd());

    static std::vector<std::uint64_t> ds_gapped_10;
    static std::vector<std::uint64_t> ds_sequential;
    static std::vector<std::uint64_t> ds_uniform;
    static auto ds_fb = load<std::uint64_t>("data/fb_200M_uint64");
    static auto ds_osm = load<std::uint64_t>("data/osm_cellids_200M_uint64");
    static auto ds_wiki = load<std::uint64_t>("data/wiki_ts_200M_uint64");

    switch (id) {
      case ID::GAPPED_10:
        if (ds_gapped_10.size() != dataset_size) {
          ds_gapped_10.resize(dataset_size);
          std::uniform_int_distribution<size_t> dist(0, 99999);
          for (size_t i = 0, num = 0; i < ds_gapped_10.size(); i++) {
            do num++;
            while (dist(rng) < 10000);
            ds_gapped_10[i] = num;
          }
          shuffle(ds_gapped_10);
        }
        return ds_gapped_10;
      case ID::SEQUENTIAL:
        if (ds_sequential.size() != dataset_size) {
          ds_sequential.resize(dataset_size);
          std::uint64_t k = 2000;
          for (size_t i = 0; i < ds_sequential.size(); i++, k++)
            ds_sequential[i] = k;
          shuffle(ds_sequential);
        }
        return ds_sequential;
      case ID::UNIFORM:
        if (ds_uniform.size() != dataset_size) {
          ds_uniform.resize(dataset_size);
          std::uniform_int_distribution<std::uint64_t> dist(
              0, std::numeric_limits<std::uint64_t>::max() - 1);
          for (size_t i = 0; i < ds_uniform.size(); i++)
            ds_uniform[i] = dist(rng);
          shuffle(ds_uniform);
        }
        return ds_uniform;
      case ID::FB:
        return ds_fb;
      case ID::OSM:
        return ds_osm;
      case ID::WIKI:
        return ds_wiki;
    }

    throw std::runtime_error("invalid datastet id " + std::to_string(id));
  }
};
