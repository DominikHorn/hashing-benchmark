#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/convenience/builtins.hpp"

namespace dataset {
/**
 * Deduplicates the dataset. Data will be sorted to make this work
 * @param dataset
 */
static forceinline void deduplicate_and_sort(std::vector<uint64_t>& dataset) {
  std::sort(dataset.begin(), dataset.end());
  dataset.erase(std::unique(dataset.begin(), dataset.end()), dataset.end());
  dataset.shrink_to_fit();
}

/**
 * Loads the datasets values into memory
 * @return a sorted and deduplicated list of all members of the dataset
 */
template <class Key>
std::vector<Key> load(std::string filepath) {
  std::cout << "loading dataset " << filepath << std::endl;

  // parsing helper functions
  auto read_little_endian_8 = [](const std::vector<unsigned char>& buffer,
                                 uint64_t offset) {
    return static_cast<uint64_t>(buffer[offset + 0]) |
           (static_cast<uint64_t>(buffer[offset + 1]) << 8) |
           (static_cast<uint64_t>(buffer[offset + 2]) << (2 * 8)) |
           (static_cast<uint64_t>(buffer[offset + 3]) << (3 * 8)) |
           (static_cast<uint64_t>(buffer[offset + 4]) << (4 * 8)) |
           (static_cast<uint64_t>(buffer[offset + 5]) << (5 * 8)) |
           (static_cast<uint64_t>(buffer[offset + 6]) << (6 * 8)) |
           (static_cast<uint64_t>(buffer[offset + 7]) << (7 * 8));
  };
  auto read_little_endian_4 = [](const std::vector<unsigned char>& buffer,
                                 uint64_t offset) {
    return buffer[offset + 0] | (buffer[offset + 1] << 8) |
           (buffer[offset + 2] << (2 * 8)) | (buffer[offset + 3] << (3 * 8));
  };

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
    if (!input.read(reinterpret_cast<char*>(buffer.data()), size))
      throw std::runtime_error("Failed to read dataset '" + filepath + "'");

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
        throw std::runtime_error(
            "unimplemented amount of bytes per value in dataset: " +
            std::to_string(sizeof(Key)));
    }
  }

  // remove duplicates from dataset and put it into random order
  deduplicate_and_sort(dataset);

  return dataset;
}

enum ID {
  SEQUENTIAL = 0,
  GAPPED_10 = 1,
  UNIFORM = 2,
  FB = 3,
  OSM = 4,
  WIKI = 5
};

static std::string name(ID id) {
  switch (id) {
    case ID::SEQUENTIAL:
      return "seq";
    case ID::GAPPED_10:
      return "gap_10";
    case ID::UNIFORM:
      return "uniform";
    case ID::FB:
      return "fb";
    case ID::OSM:
      return "osm";
    case ID::WIKI:
      return "wiki";
  }
};

static std::vector<std::uint64_t> load_cached(ID id, size_t dataset_size) {
  static std::random_device rd;
  static std::default_random_engine rng(rd());

  static std::vector<std::uint64_t> ds_gapped_10;
  static std::vector<std::uint64_t> ds_sequential;
  static std::vector<std::uint64_t> ds_uniform;
  static std::vector<std::uint64_t> ds_fb;
  static std::vector<std::uint64_t> ds_osm;
  static std::vector<std::uint64_t> ds_wiki;

  switch (id) {
    case ID::SEQUENTIAL:
      if (ds_sequential.size() != dataset_size) {
        ds_sequential.resize(dataset_size);
        std::uint64_t k = 20000;
        for (size_t i = 0; i < ds_sequential.size(); i++, k++)
          ds_sequential[i] = k;
        deduplicate_and_sort(ds_sequential);
      }
      return ds_sequential;
    case ID::GAPPED_10:
      if (ds_gapped_10.size() != dataset_size) {
        ds_gapped_10.resize(dataset_size);
        std::uniform_int_distribution<size_t> dist(0, 99999);
        for (size_t i = 0, num = 0; i < ds_gapped_10.size(); i++) {
          do num++;
          while (dist(rng) < 10000);
          ds_gapped_10[i] = num;
        }
        deduplicate_and_sort(ds_gapped_10);
      }
      return ds_gapped_10;
    case ID::UNIFORM:
      if (ds_uniform.size() != dataset_size) {
        ds_uniform.resize(dataset_size);
        std::uniform_int_distribution<std::uint64_t> dist(
            0, std::numeric_limits<std::uint64_t>::max() - 1);
        // TODO: ensure there are no duplicates
        for (size_t i = 0; i < ds_uniform.size(); i++)
          ds_uniform[i] = dist(rng);
        deduplicate_and_sort(ds_uniform);
      }
      return ds_uniform;
    case ID::FB:
      if (ds_fb.empty()) {
        ds_fb = load<std::uint64_t>("data/fb_200M_uint64");
        deduplicate_and_sort(ds_fb);
      }
      return ds_fb;
    case ID::OSM:
      if (ds_osm.empty()) {
        ds_osm = load<std::uint64_t>("data/osm_cellids_200M_uint64");
        deduplicate_and_sort(ds_osm);
      }
      return ds_osm;
    case ID::WIKI:
      if (ds_wiki.empty()) {
        ds_wiki = load<std::uint64_t>("data/wiki_ts_200M_uint64");
        deduplicate_and_sort(ds_wiki);
      }
      return ds_wiki;
  }

  throw std::runtime_error("invalid datastet id " + std::to_string(id));
}

static std::vector<std::uint64_t> load_cached_shuffled(ID id,
                                                       size_t dataset_size) {
  static std::random_device rd;
  static std::default_random_engine rng(rd());

  static std::vector<std::uint64_t> ds_sequential_shuffled;
  static std::vector<std::uint64_t> ds_gapped_10_shuffled;
  static std::vector<std::uint64_t> ds_uniform_shuffled;
  static std::vector<std::uint64_t> ds_fb_shuffled;
  static std::vector<std::uint64_t> ds_osm_shuffled;
  static std::vector<std::uint64_t> ds_wiki_shuffled;

  switch (id) {
    case ID::SEQUENTIAL:
      if (ds_sequential_shuffled.size() != dataset_size) {
        ds_sequential_shuffled = load_cached(ID::SEQUENTIAL, dataset_size);
        std::shuffle(ds_sequential_shuffled.begin(),
                     ds_sequential_shuffled.end(), rng);
      }
      return ds_sequential_shuffled;
    case ID::GAPPED_10:
      if (ds_gapped_10_shuffled.size() != dataset_size) {
        ds_gapped_10_shuffled = load_cached(ID::GAPPED_10, dataset_size);
        std::shuffle(ds_gapped_10_shuffled.begin(), ds_gapped_10_shuffled.end(),
                     rng);
      }
      return ds_gapped_10_shuffled;
    case ID::UNIFORM:
      if (ds_uniform_shuffled.size() != dataset_size) {
        ds_uniform_shuffled = load_cached(ID::UNIFORM, dataset_size);
        std::shuffle(ds_uniform_shuffled.begin(), ds_uniform_shuffled.end(),
                     rng);
      }
      return ds_uniform_shuffled;
    case ID::FB:
      if (ds_fb_shuffled.empty()) {
        ds_fb_shuffled = load_cached(ID::FB, dataset_size);
        std::shuffle(ds_fb_shuffled.begin(), ds_fb_shuffled.end(), rng);
      }
      return ds_fb_shuffled;
    case ID::OSM:
      if (ds_osm_shuffled.empty()) {
        ds_osm_shuffled = load_cached(ID::OSM, dataset_size);
        std::shuffle(ds_osm_shuffled.begin(), ds_osm_shuffled.end(), rng);
      }
      return ds_osm_shuffled;
    case ID::WIKI:
      if (ds_wiki_shuffled.empty()) {
        ds_wiki_shuffled = load_cached(ID::WIKI, dataset_size);
        std::shuffle(ds_wiki_shuffled.begin(), ds_wiki_shuffled.end(), rng);
      }
      return ds_wiki_shuffled;
  }

  throw std::runtime_error("invalid datastet id " + std::to_string(id));
}
};  // namespace dataset
