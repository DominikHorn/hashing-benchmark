#pragma once

#include <algorithm>
#include <random>

namespace dataset {
enum class ProbingDistribution {
  /// every key has the same probability to be queried
  UNIFORM,

  /// probing skewed according to exponential distribution, i.e.,
  /// some keys k are way more likely to be picked than others.
  /// rank(k) directly correlates to the k' probability of being picked
  EXPONENTIAL_SORTED,

  /// probing skewed according to exponential distribution, i.e.,
  /// some keys k are way more likely to be picked than others.
  /// rank(k) does not influence the k's probability of being picked
  EXPONENTIAL_RANDOM
};

inline std::string name(ProbingDistribution p_dist) {
  switch (p_dist) {
    case ProbingDistribution::UNIFORM:
      return "uniform";
    case ProbingDistribution::EXPONENTIAL_SORTED:
      return "exponential_sorted";
    case ProbingDistribution::EXPONENTIAL_RANDOM:
      return "exponential_random";
  }
  return "unnamed";
};

/**
 * generates a probing order for any dataset dataset, given a desired
 * distribution
 */
template <class T>
static std::vector<T> generate_probing_set(std::vector<T> dataset,
                                           ProbingDistribution distribution) {
  if (dataset.empty()) return {};

  std::random_device rd;
  std::default_random_engine rng(rd());

  size_t size = dataset.size();
  std::vector<T> probing_set(size, dataset[0]);

  switch (distribution) {
    case ProbingDistribution::UNIFORM: {
      std::uniform_int_distribution<> dist(0, dataset.size() - 1);
      for (size_t i = 0; i < size; i++) probing_set[i] = dataset[dist(rng)];
      break;
    }
    case ProbingDistribution::EXPONENTIAL_SORTED: {
      std::exponential_distribution<> dist(10);

      for (size_t i = 0; i < size; i++)
        probing_set[i] =
            dataset[(dataset.size() - 1) * std::min(1.0, dist(rng))];
      break;
    }
    case ProbingDistribution::EXPONENTIAL_RANDOM: {
      // shuffle to avoid skewed results for sorted data, i.e., when
      // dataset is sorted, this will always prefer lower keys. This
      // might make a difference for tries, e.g. when they are left deep
      // vs right deep!
      std::shuffle(dataset.begin(), dataset.end(), rng);

      std::exponential_distribution<> dist(10);

      for (size_t i = 0; i < size; i++)
        probing_set[i] =
            dataset[(dataset.size() - 1) * std::min(1.0, dist(rng))];
      break;
    }
  }

  return probing_set;
}

}  // namespace dataset
