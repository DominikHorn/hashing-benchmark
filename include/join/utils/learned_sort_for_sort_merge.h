/* Adapting the techniques in learned_sort.h for sort_merge */

#ifndef LEARNED_SORT_FOR_SORT_MERGE_H
#define LEARNED_SORT_FOR_SORT_MERGE_H

#include <immintrin.h> /* AVX intrinsics */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "utils/data_structures.h"
#include "utils/eth_data_structures.h"
#include "utils/barrier.h"

#include "configs/base_configs.h"

#ifdef USE_AVXSORT_AS_STD_SORT
#include "utils/eth_avx_sort/avxsort.h"
#ifdef USE_AVXMERGE_AS_STD_MERGE
#include "utils/eth_avx_merge/merge.h"
#endif
#endif

using std::vector;
using std::cerr;
using std::endl;

namespace learned_sort_for_sort_merge {

#ifdef LS_FOR_SORT_MERGE_IMV_AVX
typedef struct StateSIMDForLearnedSort StateSIMDForLearnedSort;
struct StateSIMDForLearnedSort {
  __m512i key;
  __m512i pred_model_idx;
  __mmask8 m_have_key;
  char stage;
};
#endif

template<class KeyType, class PayloadType>
struct training_point {
    Tuple<KeyType, PayloadType> x;
    double y;
};

template <class KeyType, class PayloadType>
class RMI {
 public:
  // Individual linear models
  struct linear_model {
    double slope = 0;
    double intercept = 0;
  };

  // CDF model hyperparameters
  struct Params {
    // Member fields
    unsigned int batch_sz;
    unsigned int fanout;
    float overallocation_ratio;
    float sampling_rate;
    unsigned int threshold;
    vector<unsigned int> arch;

    // Default hyperparameters
    static constexpr unsigned int DEFAULT_BATCH_SZ = LS_FOR_SORT_MERGE_DEFAULT_BATCH_SZ;
    static constexpr unsigned int DEFAULT_FANOUT = LS_FOR_SORT_MERGE_DEFAULT_FANOUT;
    static constexpr float DEFAULT_OVERALLOCATION_RATIO = LS_FOR_SORT_MERGE_DEFAULT_OVERALLOCATION_RATIO;
    static constexpr float DEFAULT_SAMPLING_RATE = LS_FOR_SORT_MERGE_DEFAULT_SAMPLING_RATE;
    static constexpr unsigned int DEFAULT_THRESHOLD = LS_FOR_SORT_MERGE_DEFAULT_THRESHOLD; //50, 100, 10000
    vector<unsigned int> DEFAULT_ARCH = {1, LS_FOR_SORT_MERGE_DEFAULT_ARCH_SECOND_LEVEL}; //1000, 10000, 100000
    static constexpr unsigned int MIN_SORTING_SIZE = LS_FOR_SORT_MERGE_MIN_SORTING_SIZE;

    // Default constructor
    Params();

    // Constructor with custom hyperparameter values
    Params(float sampling_rate, float overallocation, unsigned int fanout,
           unsigned int batch_size, unsigned int threshold,
           vector<unsigned int> model_arch);
  };

  // Member variables of the CDF model
  bool trained; 
  vector<vector<linear_model>> models;
  Tuple<KeyType, PayloadType> ** training_sample;
  Tuple<KeyType, PayloadType> * tmp_training_sample;
  Tuple<KeyType, PayloadType> * sorted_training_sample;
  uint32_t training_sample_size;
#ifdef BUILD_RMI_FROM_TWO_DATASETS
  Tuple<KeyType, PayloadType> ** training_sample_R;
  Tuple<KeyType, PayloadType> * tmp_training_sample_R;
  Tuple<KeyType, PayloadType> * sorted_training_sample_R;
  uint32_t training_sample_size_R;

  Tuple<KeyType, PayloadType> ** training_sample_S;
  Tuple<KeyType, PayloadType> * tmp_training_sample_S;
  Tuple<KeyType, PayloadType> * sorted_training_sample_S;
  uint32_t training_sample_size_S;
#endif

  Params hp;

  // CDF model constructor
  explicit RMI(Params p, Tuple<KeyType, PayloadType> * tmp_training_sample_in, Tuple<KeyType, PayloadType> * sorted_training_sample_in);

#ifdef BUILD_RMI_FROM_TWO_DATASETS
  explicit RMI(Params p, Tuple<KeyType, PayloadType> * tmp_training_sample_in, Tuple<KeyType, PayloadType> * sorted_training_sample_in, 
                         Tuple<KeyType, PayloadType> * tmp_training_sample_R_in, Tuple<KeyType, PayloadType> * sorted_training_sample_R_in,
                         Tuple<KeyType, PayloadType> * tmp_training_sample_S_in, Tuple<KeyType, PayloadType> * sorted_training_sample_S_in);
#endif
};

// Validate parameters
template<class KeyType, class PayloadType>
void validate_params(typename RMI<KeyType, PayloadType>::Params &, unsigned int);

// Training function based on one dataset
template<class KeyType, class PayloadType>
RMI<KeyType, PayloadType> train(
    Tuple<KeyType, PayloadType> *, unsigned int,
    typename RMI<KeyType, PayloadType>::Params &,
    Tuple<KeyType, PayloadType> *, 
    Tuple<KeyType, PayloadType> *, 
    int, int);

#ifdef BUILD_RMI_FROM_TWO_DATASETS
// Training function based on two datasets
template<class KeyType, class PayloadType>
RMI<KeyType, PayloadType> train(
    typename RMI<KeyType, PayloadType>::Params &,
    Tuple<KeyType, PayloadType> *, unsigned int,
    Tuple<KeyType, PayloadType> *, unsigned int,
    Tuple<KeyType, PayloadType> *, Tuple<KeyType, PayloadType> *,
    Tuple<KeyType, PayloadType> *, Tuple<KeyType, PayloadType> *,
    Tuple<KeyType, PayloadType> *, Tuple<KeyType, PayloadType> *, 
    int, int);
#endif

template<class KeyType, class PayloadType>
void histogram_and_get_max_capacity_for_major_buckets(int64_t *, unsigned int,
  RMI<KeyType, PayloadType> *,
  Tuple<KeyType, PayloadType> *, unsigned int,
  int, int);

template<class KeyType, class PayloadType>
void partition_major_buckets(int,
  RMI<KeyType, PayloadType> *, unsigned int,
  Tuple<KeyType, PayloadType> *, unsigned int, 
  Tuple<KeyType, PayloadType> *, int64_t *, int64_t *,
  Tuple<KeyType, PayloadType> *, int64_t *, int64_t *, int64_t *, int64_t *, 
  int, int);

template<class KeyType, class PayloadType>
void partition_major_buckets_threaded(int,
  RMI<KeyType, PayloadType> *, unsigned int,
  Tuple<KeyType, PayloadType> *, unsigned int, 
  Tuple<KeyType, PayloadType> *, int64_t *, int64_t *,
  Tuple<KeyType, PayloadType> *, int64_t *, int64_t *, int64_t *, int64_t *, 
  int, int);


template<class KeyType, class PayloadType>
void sort_avx(Tuple<KeyType, PayloadType> *, RMI<KeyType, PayloadType> *,
              unsigned int, unsigned int, unsigned int,
              unsigned int, Tuple<KeyType, PayloadType> *, uint64_t, 
              Tuple<KeyType, PayloadType> *, int64_t *,
              Tuple<KeyType, PayloadType> *, Tuple<KeyType, PayloadType> *,
              int64_t, Tuple<KeyType, PayloadType> *, 
              int64_t *, int64_t,
              int, int);

template<class KeyType, class PayloadType>
void sort_avx_from_seperate_partitions(Tuple<KeyType, PayloadType> *, RMI<KeyType, PayloadType> * rmi,
                                          unsigned int, unsigned int, unsigned int,
                                          unsigned int, Tuple<KeyType, PayloadType> **, uint64_t*, int64_t, Tuple<KeyType, PayloadType> *, 
                                          Tuple<KeyType, PayloadType> *, int64_t *,
                                          Tuple<KeyType, PayloadType> *, Tuple<KeyType, PayloadType> *, 
                                          int64_t*, int64_t, Tuple<KeyType, PayloadType> **, 
                                          int64_t **, int64_t,
                                          int, int);
}

using namespace learned_sort_for_sort_merge;

template <class KeyType, class PayloadType>
learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params::Params() {
  this->batch_sz = DEFAULT_BATCH_SZ;
  this->fanout = DEFAULT_FANOUT;
  this->overallocation_ratio = DEFAULT_OVERALLOCATION_RATIO;
  this->sampling_rate = DEFAULT_SAMPLING_RATE;
  this->threshold = DEFAULT_THRESHOLD;
  this->arch = DEFAULT_ARCH;
}

template <class KeyType, class PayloadType>
learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params::Params(
                       float sampling_rate, float overallocation,
                       unsigned int fanout, unsigned int batch_sz,
                       unsigned int threshold, vector<unsigned int> arch) {
  this->batch_sz = batch_sz;
  this->fanout = fanout;
  this->overallocation_ratio = overallocation;
  this->sampling_rate = sampling_rate;
  this->threshold = threshold;
  this->arch = std::move(arch);
}

template <class KeyType, class PayloadType>
learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::RMI(Params p, 
    Tuple<KeyType, PayloadType> * tmp_training_sample_in, 
    Tuple<KeyType, PayloadType> * sorted_training_sample_in) {
  this->trained = false;
  this->hp = p;
  this->models.resize(p.arch.size());
  this->tmp_training_sample = tmp_training_sample_in;
  this->sorted_training_sample = sorted_training_sample_in;

  for (size_t layer_idx = 0; layer_idx < p.arch.size(); ++layer_idx) {
    this->models[layer_idx].resize(p.arch[layer_idx]);
  }
}

#ifdef BUILD_RMI_FROM_TWO_DATASETS
template <class KeyType, class PayloadType>
learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::RMI(Params p, 
    Tuple<KeyType, PayloadType> * tmp_training_sample_in, Tuple<KeyType, PayloadType> * sorted_training_sample_in, 
    Tuple<KeyType, PayloadType> * tmp_training_sample_R_in, Tuple<KeyType, PayloadType> * sorted_training_sample_R_in, 
    Tuple<KeyType, PayloadType> * tmp_training_sample_S_in, Tuple<KeyType, PayloadType> * sorted_training_sample_S_in) {
  this->trained = false;
  this->hp = p;
  this->models.resize(p.arch.size());
  this->tmp_training_sample = tmp_training_sample_in;
  this->sorted_training_sample = sorted_training_sample_in;
  this->tmp_training_sample_R = tmp_training_sample_R_in;
  this->sorted_training_sample_R = sorted_training_sample_R_in;
  this->tmp_training_sample_S = tmp_training_sample_S_in;
  this->sorted_training_sample_S = sorted_training_sample_S_in;

  for (size_t layer_idx = 0; layer_idx < p.arch.size(); ++layer_idx) {
    this->models[layer_idx].resize(p.arch[layer_idx]);
  }
}
#endif

// Validate parameters
template<class KeyType, class PayloadType>
void learned_sort_for_sort_merge::validate_params(typename learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params &p, unsigned int INPUT_SZ)
{
  // Validate parameters
  if (p.batch_sz >= INPUT_SZ) {
    p.batch_sz = RMI<KeyType, PayloadType>::Params::DEFAULT_BATCH_SZ;
    cerr << "\33[93;1mWARNING\33[0m: Invalid batch size. Using default ("
         << RMI<KeyType, PayloadType>::Params::DEFAULT_BATCH_SZ << ")." << endl;
  }

  if (p.fanout >= INPUT_SZ) {
    p.fanout = RMI<KeyType, PayloadType>::Params::DEFAULT_FANOUT;
    cerr << "\33[93;1mWARNING\33[0m: Invalid fanout. Using default ("
         << RMI<KeyType, PayloadType>::Params::DEFAULT_FANOUT << ")." << endl;
  }

  if (p.overallocation_ratio <= 0) {
    p.overallocation_ratio = 1;
    cerr << "\33[93;1mWARNING\33[0m: Invalid overallocation ratio. Using "
            "default ("
         << RMI<KeyType, PayloadType>::Params::DEFAULT_OVERALLOCATION_RATIO << ")." << endl;
  }

  if (p.sampling_rate <= 0 or p.sampling_rate > 1) {
    p.sampling_rate = RMI<KeyType, PayloadType>::Params::DEFAULT_SAMPLING_RATE;
    cerr << "\33[93;1mWARNING\33[0m: Invalid sampling rate. Using default ("
         << RMI<KeyType, PayloadType>::Params::DEFAULT_SAMPLING_RATE << ")." << endl;
  }

  if (p.threshold <= 0 or p.threshold >= INPUT_SZ or
      p.threshold >= INPUT_SZ / p.fanout) {
    p.threshold = RMI<KeyType, PayloadType>::Params::DEFAULT_THRESHOLD;
    cerr << "\33[93;1mWARNING\33[0m: Invalid threshold. Using default ("
         << RMI<KeyType, PayloadType>::Params::DEFAULT_THRESHOLD << ")." << endl;
  }

  if (p.arch.size() > 2 or p.arch[0] != 1 or p.arch[1] <= 0) {
    p.arch = p.DEFAULT_ARCH;
    cerr << "\33[93;1mWARNING\33[0m: Invalid architecture. Using default {"
         << p.DEFAULT_ARCH[0] << ", " << p.DEFAULT_ARCH[1] << "}." << endl;
  }
}

template<class KeyType, class PayloadType>
learned_sort_for_sort_merge::RMI<KeyType, PayloadType> learned_sort_for_sort_merge::train(
    Tuple<KeyType, PayloadType> * begin, unsigned int INPUT_SZ,
    typename learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params &p, 
    Tuple<KeyType, PayloadType> * tmp_training_sample_in, 
    Tuple<KeyType, PayloadType> * sorted_training_sample_in,
    int thread_id, int partition_id) {
     
  // Initialize the CDF model
  RMI<KeyType, PayloadType> rmi(p, tmp_training_sample_in, sorted_training_sample_in);
  static const unsigned int NUM_LAYERS = p.arch.size();
  vector<vector<vector<training_point<KeyType, PayloadType>>>> training_data(NUM_LAYERS);
  for (unsigned int layer_idx = 0; layer_idx < NUM_LAYERS; ++layer_idx) {
    training_data[layer_idx].resize(p.arch[layer_idx]);
  }


  //----------------------------------------------------------//
  //                           SAMPLE                         //
  //----------------------------------------------------------//

  // Determine sample size
  const unsigned int SAMPLE_SZ = std::min<unsigned int>(
      INPUT_SZ, std::max<unsigned int>(p.sampling_rate * INPUT_SZ,
                                       RMI<KeyType, PayloadType>::Params::MIN_SORTING_SIZE));
    
  // Start sampling
  unsigned int offset = static_cast<unsigned int>(1. * INPUT_SZ / SAMPLE_SZ);
  unsigned int sample_count = 0; 
  Tuple<KeyType, PayloadType>* end = begin + INPUT_SZ;
  for (auto i = begin; i < end; i += offset) {
    // NOTE:  We don't directly assign SAMPLE_SZ to rmi.training_sample_sz
    //        to avoid issues with divisibility
    rmi.tmp_training_sample[sample_count] = *i;
    ++sample_count;
  }


#ifdef USE_AVXSORT_AS_STD_SORT          
/*  int64_t * inputptr = (int64_t *)(rmi.tmp_training_sample);
  int64_t * outputptr = (int64_t *)(rmi.sorted_training_sample);
  avxsort_int64(&inputptr, &outputptr, sample_count);
  rmi.training_sample = &(rmi.sorted_training_sample);
  rmi.training_sample_size = sample_count;
*/

  int64_t * inputptr =  (int64_t *)(rmi.tmp_training_sample);
  int64_t * outputptr = (int64_t *)(rmi.sorted_training_sample);
  avxsort_int64(&inputptr, &outputptr, sample_count);
  Tuple<KeyType, PayloadType>* tmp_outputptr = (Tuple<KeyType, PayloadType>*) outputptr;
  for(unsigned int k = 0; k < sample_count; k++){
    rmi.sorted_training_sample[k].key = tmp_outputptr[k].key;
    rmi.sorted_training_sample[k].payload = tmp_outputptr[k].payload;
  } 
  rmi.training_sample = &(rmi.sorted_training_sample);
  rmi.training_sample_size = sample_count;
#else

  std::sort((int64_t *)(rmi.tmp_training_sample), (int64_t *)(rmi.tmp_training_sample) + sample_count /*- 1*/);
  rmi.training_sample = &(rmi.tmp_training_sample);
  rmi.training_sample_size = sample_count;

  // Sort the sample using the provided comparison function
  //std::sort(rmi.tmp_training_sample.begin(), rmi.tmp_training_sample.end());
  //std::sort(rmi.training_sample.begin(), rmi.training_sample.end());
  //rmi.training_sample = &(rmi.tmp_training_sample);
#endif

  // Stop early if the array is identical
  if (((*rmi.training_sample)[0]).key == ((*rmi.training_sample)[sample_count - 1]).key) {
    return rmi;
  }

  //----------------------------------------------------------//
  //                     TRAIN THE MODELS                     //
  //----------------------------------------------------------//

  // Populate the training data for the root model
  for (unsigned int i = 0; i < SAMPLE_SZ; ++i) {
    training_data[0][0].push_back({(*rmi.training_sample)[i], 1. * i / SAMPLE_SZ});
  }

  // Train the root model using linear interpolation
  auto *current_training_data = &training_data[0][0];
  typename RMI<KeyType, PayloadType>::linear_model *current_model = &rmi.models[0][0];

  // Find the min and max values in the training set
  training_point<KeyType, PayloadType> min = current_training_data->front();
  training_point<KeyType, PayloadType> max = current_training_data->back();

  // Calculate the slope and intercept terms
  current_model->slope =
      1. / (max.x.key - min.x.key);  // Assuming min.y = 0 and max.y = 1
  current_model->intercept = -current_model->slope * min.x.key;

  // Extrapolate for the number of models in the next layer
  current_model->slope *= p.arch[1] - 1;
  current_model->intercept *= p.arch[1] - 1;

  // Populate the training data for the next layer
  for (const auto &d : *current_training_data) {
    // Predict the model index in next layer
    //unsigned int rank = current_model->slope * d.x.key + current_model->intercept;
    unsigned int rank = round(current_model->slope * d.x.key*1.00 + current_model->intercept);

    // Normalize the rank between 0 and the number of models in the next layer
    rank =
        std::max(static_cast<unsigned int>(0), std::min(p.arch[1] - 1, rank));

    // Place the data in the predicted training bucket
    training_data[1][rank].push_back(d);
  }

  // Train the leaf models
  for (unsigned int model_idx = 0; model_idx < p.arch[1]; ++model_idx) {
    // Update iterator variables
    current_training_data = &training_data[1][model_idx];
    current_model = &rmi.models[1][model_idx];

    // Interpolate the min points in the training buckets
    if (model_idx ==
        0) {  // The current model is the first model in the current layer

      if (current_training_data->size() <
          2) {  // Case 1: The first model in this layer is empty
        current_model->slope = 0;
        current_model->intercept = 0;

        // Insert a fictive training point to avoid propagating more than one
        // empty initial models.
        training_point<KeyType, PayloadType> tp;
        tp.x.key = 0;
        tp.x.payload = 0;
        tp.y = 0;
        current_training_data->push_back(tp);
      } else {  // Case 2: The first model in this layer is not empty

        min = current_training_data->front();
        max = current_training_data->back();

        current_model->slope =
            (max.y)*1.00 / (max.x.key - min.x.key);  // Hallucinating as if min.y = 0
        current_model->intercept = min.y - current_model->slope * min.x.key;
      }
    } else if (model_idx == p.arch[1] - 1) {
      if (current_training_data
              ->empty()) {  // Case 3: The final model in this layer is empty

        current_model->slope = 0;
        current_model->intercept = 1;
      } else {  // Case 4: The last model in this layer is not empty

        min = training_data[1][model_idx - 1].back();
        max = current_training_data->back();

        current_model->slope =
            (1 - min.y) * 1.00 / (max.x.key - min.x.key);  // Hallucinating as if max.y = 1
        current_model->intercept = min.y - current_model->slope * min.x.key;
      }
    } else {  // The current model is not the first model in the current layer

      if (current_training_data
              ->empty()) {  // Case 5: The intermediate model in
        // this layer is empty
        current_model->slope = 0;
        current_model->intercept =
            training_data[1][model_idx - 1].back().y;  // If the previous model
                                                       // was empty too, it will
                                                       // use the fictive
                                                       // training points

        // Insert a fictive training point to avoid propagating more than one
        // empty initial models.
        // NOTE: This will _NOT_ throw to DIV/0 due to identical x's and y's
        // because it is working backwards.
        training_point<KeyType, PayloadType> tp;
        tp.x = training_data[1][model_idx - 1].back().x;
        tp.y = training_data[1][model_idx - 1].back().y;
        current_training_data->push_back(tp);
      } else {  // Case 6: The intermediate leaf model is not empty

        min = training_data[1][model_idx - 1].back();
        max = current_training_data->back();

        current_model->slope = (max.y - min.y) * 1.00 / (max.x.key - min.x.key);
        current_model->intercept = min.y - current_model->slope * min.x.key;
      }
    }
  }

  // NOTE:
  // The last stage (layer) of this model contains weights that predict the CDF
  // of the keys (i.e. Range is [0-1])
  // When using this model to predict the position of the keys in the sorted
  // order, you MUST scale the
  // weights of the last layer to whatever range you are predicting for. The
  // inner layers of the model have
  // already been extrapolated to the length of the stage.git
  //
  // This is a design choice to help with the portability of the model.
  //
  rmi.trained = true;

  return rmi;
}  // end of training function

#ifdef BUILD_RMI_FROM_TWO_DATASETS
template<class KeyType, class PayloadType>
learned_sort_for_sort_merge::RMI<KeyType, PayloadType> learned_sort_for_sort_merge::train(
    typename learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params &p,
    Tuple<KeyType, PayloadType> * begin_R, unsigned int INPUT_SZ_R,
    Tuple<KeyType, PayloadType> * begin_S, unsigned int INPUT_SZ_S, 
    Tuple<KeyType, PayloadType> * tmp_training_sample_in, Tuple<KeyType, PayloadType> * sorted_training_sample_in,
    Tuple<KeyType, PayloadType> * tmp_training_sample_R_in, Tuple<KeyType, PayloadType> * sorted_training_sample_R_in,
    Tuple<KeyType, PayloadType> * tmp_training_sample_S_in, Tuple<KeyType, PayloadType> * sorted_training_sample_S_in,        
    int thread_id, int partition_id) {
     
  // Initialize the CDF model
  RMI<KeyType, PayloadType> rmi(p, tmp_training_sample_in, sorted_training_sample_in, 
                                   tmp_training_sample_R_in, sorted_training_sample_R_in,
                                   tmp_training_sample_S_in, sorted_training_sample_S_in);
  
  static const unsigned int NUM_LAYERS = p.arch.size();
  vector<vector<vector<training_point<KeyType, PayloadType>>>> training_data(NUM_LAYERS);
  for (unsigned int layer_idx = 0; layer_idx < NUM_LAYERS; ++layer_idx) {
    training_data[layer_idx].resize(p.arch[layer_idx]);
  }

  
  //----------------------------------------------------------//
  //                           SAMPLE                         //
  //----------------------------------------------------------//

  // Determine sample size
  const unsigned int SAMPLE_SZ_R = std::min<unsigned int>(
      INPUT_SZ_R, std::max<unsigned int>(p.sampling_rate * INPUT_SZ_R,
                                       RMI<KeyType, PayloadType>::Params::MIN_SORTING_SIZE));
  const unsigned int SAMPLE_SZ_S = std::min<unsigned int>(
      INPUT_SZ_S, std::max<unsigned int>(p.sampling_rate * INPUT_SZ_S,
                                       RMI<KeyType, PayloadType>::Params::MIN_SORTING_SIZE));
  const unsigned int SAMPLE_SZ = SAMPLE_SZ_R + SAMPLE_SZ_S;        
  
  // Start sampling
  unsigned int sample_count = 0; 

  unsigned int sample_count_R = 0;
  unsigned int offset_R = static_cast<unsigned int>(1. * INPUT_SZ_R / SAMPLE_SZ_R);
  Tuple<KeyType, PayloadType>* end_R = begin_R + INPUT_SZ_R;
  for (auto i = begin_R; i < end_R; i += offset_R) {
    // NOTE:  We don't directly assign SAMPLE_SZ to rmi.training_sample_sz
    //        to avoid issues with divisibility
    rmi.tmp_training_sample[sample_count] = *i;
    ++sample_count;

    rmi.tmp_training_sample_R[sample_count_R] = *i;
    ++sample_count_R;
  }

  unsigned int sample_count_S = 0;
  unsigned int offset_S = static_cast<unsigned int>(1. * INPUT_SZ_S / SAMPLE_SZ_S);
  Tuple<KeyType, PayloadType>* end_S = begin_S + INPUT_SZ_S;
  for (auto i = begin_S; i < end_S; i += offset_S) {
    // NOTE:  We don't directly assign SAMPLE_SZ to rmi.training_sample_sz
    //        to avoid issues with divisibility
    rmi.tmp_training_sample[sample_count] = *i;
    ++sample_count;

    rmi.tmp_training_sample_S[sample_count_S] = *i;
    ++sample_count_S;
  }

#ifdef USE_AVXSORT_AS_STD_SORT          

  int64_t * inputptr =  (int64_t *)(rmi.tmp_training_sample);
  int64_t * outputptr = (int64_t *)(rmi.sorted_training_sample);
  avxsort_int64(&inputptr, &outputptr, sample_count);
  Tuple<KeyType, PayloadType>* tmp_outputptr = (Tuple<KeyType, PayloadType>*) outputptr;
  for(unsigned int k = 0; k < sample_count; k++){
    rmi.sorted_training_sample[k].key = tmp_outputptr[k].key;
    rmi.sorted_training_sample[k].payload = tmp_outputptr[k].payload;
  } 
  rmi.training_sample = &(rmi.sorted_training_sample);
  rmi.training_sample_size = sample_count;

  int64_t * inputptr_R =  (int64_t *)(rmi.tmp_training_sample_R);
  int64_t * outputptr_R = (int64_t *)(rmi.sorted_training_sample_R);
  avxsort_int64(&inputptr_R, &outputptr_R, sample_count_R);
  Tuple<KeyType, PayloadType>* tmp_outputptr_R = (Tuple<KeyType, PayloadType>*) outputptr_R;
  for(unsigned int k = 0; k < sample_count_R; k++){
    rmi.sorted_training_sample_R[k].key = tmp_outputptr_R[k].key;
    rmi.sorted_training_sample_R[k].payload = tmp_outputptr_R[k].payload;
  } 
  rmi.training_sample_R = &(rmi.sorted_training_sample_R);
  rmi.training_sample_size_R = sample_count_R;

  int64_t * inputptr_S =  (int64_t *)(rmi.tmp_training_sample_S);
  int64_t * outputptr_S = (int64_t *)(rmi.sorted_training_sample_S);
  avxsort_int64(&inputptr_S, &outputptr_S, sample_count_S);
  Tuple<KeyType, PayloadType>* tmp_outputptr_S = (Tuple<KeyType, PayloadType>*) outputptr_S;
  for(unsigned int k = 0; k < sample_count_S; k++){
    rmi.sorted_training_sample_S[k].key = tmp_outputptr_S[k].key;
    rmi.sorted_training_sample_S[k].payload = tmp_outputptr_S[k].payload;
  } 
  rmi.training_sample_S = &(rmi.sorted_training_sample_S);
  rmi.training_sample_size_S = sample_count_S;

#else

  std::sort((int64_t *)(rmi.tmp_training_sample), (int64_t *)(rmi.tmp_training_sample) + sample_count /*- 1*/);
  rmi.training_sample = &(rmi.tmp_training_sample);
  rmi.training_sample_size = sample_count;

  std::sort((int64_t *)(rmi.tmp_training_sample_R), (int64_t *)(rmi.tmp_training_sample_R) + sample_count_R /*- 1*/);
  rmi.training_sample_R = &(rmi.tmp_training_sample_R);
  rmi.training_sample_size_R = sample_count_R;

  std::sort((int64_t *)(rmi.tmp_training_sample_S), (int64_t *)(rmi.tmp_training_sample_S) + sample_count_S /*- 1*/);
  rmi.training_sample_S = &(rmi.tmp_training_sample_S);
  rmi.training_sample_size_S = sample_count_S;
#endif


  // Stop early if the array is identical
  if (((*rmi.training_sample)[0]).key == ((*rmi.training_sample)[sample_count - 1]).key) {
    return rmi;
  }

  //----------------------------------------------------------//
  //                     TRAIN THE MODELS                     //
  //----------------------------------------------------------//

  // Populate the training data for the root model
  for (unsigned int i = 0; i < SAMPLE_SZ; ++i) {
    training_data[0][0].push_back({(*rmi.training_sample)[i], 1. * i / SAMPLE_SZ});
  }

  // Train the root model using linear interpolation
  auto *current_training_data = &training_data[0][0];
  typename RMI<KeyType, PayloadType>::linear_model *current_model = &rmi.models[0][0];

  // Find the min and max values in the training set
  training_point<KeyType, PayloadType> min = current_training_data->front();
  training_point<KeyType, PayloadType> max = current_training_data->back();

  // Calculate the slope and intercept terms
  current_model->slope =
      1. / (max.x.key - min.x.key);  // Assuming min.y = 0 and max.y = 1
  current_model->intercept = -current_model->slope * min.x.key;

  // Extrapolate for the number of models in the next layer
  current_model->slope *= p.arch[1] - 1;
  current_model->intercept *= p.arch[1] - 1;

  // Populate the training data for the next layer
  for (const auto &d : *current_training_data) {
    // Predict the model index in next layer
    //unsigned int rank = current_model->slope * d.x.key + current_model->intercept;
    unsigned int rank = round(current_model->slope * d.x.key*1.00 + current_model->intercept);

    // Normalize the rank between 0 and the number of models in the next layer
    rank =
        std::max(static_cast<unsigned int>(0), std::min(p.arch[1] - 1, rank));

    // Place the data in the predicted training bucket
    training_data[1][rank].push_back(d);
  }

  // Train the leaf models
  for (unsigned int model_idx = 0; model_idx < p.arch[1]; ++model_idx) {
    // Update iterator variables
    current_training_data = &training_data[1][model_idx];
    current_model = &rmi.models[1][model_idx];

    // Interpolate the min points in the training buckets
    if (model_idx ==
        0) {  // The current model is the first model in the current layer

      if (current_training_data->size() <
          2) {  // Case 1: The first model in this layer is empty
        current_model->slope = 0;
        current_model->intercept = 0;

        // Insert a fictive training point to avoid propagating more than one
        // empty initial models.
        training_point<KeyType, PayloadType> tp;
        tp.x.key = 0;
        tp.x.payload = 0;
        tp.y = 0;
        current_training_data->push_back(tp);
      } else {  // Case 2: The first model in this layer is not empty

        min = current_training_data->front();
        max = current_training_data->back();

        current_model->slope =
            (max.y)*1.00 / (max.x.key - min.x.key);  // Hallucinating as if min.y = 0
        current_model->intercept = min.y - current_model->slope * min.x.key;
      }
    } else if (model_idx == p.arch[1] - 1) {
      if (current_training_data
              ->empty()) {  // Case 3: The final model in this layer is empty

        current_model->slope = 0;
        current_model->intercept = 1;
      } else {  // Case 4: The last model in this layer is not empty

        min = training_data[1][model_idx - 1].back();
        max = current_training_data->back();

        current_model->slope =
            (1 - min.y) * 1.00 / (max.x.key - min.x.key);  // Hallucinating as if max.y = 1
        current_model->intercept = min.y - current_model->slope * min.x.key;
      }
    } else {  // The current model is not the first model in the current layer

      if (current_training_data
              ->empty()) {  // Case 5: The intermediate model in
        // this layer is empty
        current_model->slope = 0;
        current_model->intercept =
            training_data[1][model_idx - 1].back().y;  // If the previous model
                                                       // was empty too, it will
                                                       // use the fictive
                                                       // training points

        // Insert a fictive training point to avoid propagating more than one
        // empty initial models.
        // NOTE: This will _NOT_ throw to DIV/0 due to identical x's and y's
        // because it is working backwards.
        training_point<KeyType, PayloadType> tp;
        tp.x = training_data[1][model_idx - 1].back().x;
        tp.y = training_data[1][model_idx - 1].back().y;
        current_training_data->push_back(tp);
      } else {  // Case 6: The intermediate leaf model is not empty

        min = training_data[1][model_idx - 1].back();
        max = current_training_data->back();

        current_model->slope = (max.y - min.y) * 1.00 / (max.x.key - min.x.key);
        current_model->intercept = min.y - current_model->slope * min.x.key;
      }
    }
  }

  // NOTE:
  // The last stage (layer) of this model contains weights that predict the CDF
  // of the keys (i.e. Range is [0-1])
  // When using this model to predict the position of the keys in the sorted
  // order, you MUST scale the
  // weights of the last layer to whatever range you are predicting for. The
  // inner layers of the model have
  // already been extrapolated to the length of the stage.git
  //
  // This is a design choice to help with the portability of the model.
  //
  rmi.trained = true;

  return rmi;
}  // end of training function

#endif


template<class KeyType, class PayloadType>
void learned_sort_for_sort_merge::histogram_and_get_max_capacity_for_major_buckets(int64_t * out_hist, unsigned int out_hist_size, 
  learned_sort_for_sort_merge::RMI<KeyType, PayloadType>* rmi,
  Tuple<KeyType, PayloadType> * begin, unsigned int INPUT_SZ, 
  int thread_id, int partition_id)
{
  // NOTE: out_hist_size is supposted to be equal to the number of threads
  // Cache runtime parameters
  static const unsigned int FANOUT = out_hist_size;

  // Cache the model parameters
  auto root_slope = rmi->models[0][0].slope;
  auto root_intrcpt = rmi->models[0][0].intercept;
  unsigned int num_models = rmi->hp.arch[1];
  vector<double> slopes, intercepts;
  for (unsigned int i = 0; i < num_models; ++i) {
    slopes.push_back(rmi->models[1][i].slope);
    intercepts.push_back(rmi->models[1][i].intercept);
  }

  int pred_model_idx = 0;   double pred_cdf = 0.;
  Tuple<KeyType, PayloadType>* end = begin + INPUT_SZ;

  // Process each key in order
  for (auto cur_key = begin; cur_key < end; ++cur_key) {
    // Predict the model idx in the leaf layer
    pred_model_idx = static_cast<int>(std::max(
        0.,
        std::min(num_models - 1., root_slope * cur_key->key + root_intrcpt)));

    // Predict the CDF
    pred_cdf =
        slopes[pred_model_idx] * cur_key->key + intercepts[pred_model_idx];

    // Scale the CDF to the number of buckets
    pred_model_idx = static_cast<int>(
        std::max(0., std::min(FANOUT - 1., pred_cdf * FANOUT)));

    ++out_hist[pred_model_idx];    
  }
/*
  int64_t max_hist_val = 0;
  for(int i = 0; i < FANOUT; i++)
  {
    if(out_hist[i] > max_hist_val)
      max_hist_val = out_hist[i];
  }

  // find the suitable max_major_bucket_capacity for each partition (thread) based on the max value in the histogram
  int64_t a = (max_hist_val / rmi->hp.threshold) * rmi->hp.threshold;
  a = a + (int64_t) rmi->hp.threshold;
  for(int i = 0; i < FANOUT; i++)
    out_hist[i] = a;
*/
}

template<class KeyType, class PayloadType>
void learned_sort_for_sort_merge::partition_major_buckets(int is_R_relation, 
  learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi, unsigned int partitions_fanout,
  Tuple<KeyType, PayloadType> * begin, unsigned int INPUT_SZ, 
  Tuple<KeyType, PayloadType> * out_partitions, int64_t* out_partitions_offsets, int64_t * out_partitions_hist,  
  Tuple<KeyType, PayloadType> * out_repeated_keys, int64_t * out_repeated_keys_counts, int64_t* out_repeated_keys_offsets, int64_t * out_repeated_keys_hist, int64_t * out_total_repeated_keys_hist, 
  int thread_id, int partition_id)
{

  // NOTE: partitions_fanout is supposted to be equal to the number of threads
  // Cache runtime parameters
  static const unsigned int FANOUT = partitions_fanout;

  // Constants for repeated keys
  //const unsigned int EXCEPTION_VEC_INIT_CAPACITY = FANOUT;
  constexpr unsigned int EXC_CNT_THRESHOLD = 60;

#ifdef BUILD_RMI_FROM_TWO_DATASETS
  size_t TRAINING_SAMPLE_SZ;
  if(is_R_relation)
    TRAINING_SAMPLE_SZ = rmi->training_sample_size_R;
  else
    TRAINING_SAMPLE_SZ = rmi->training_sample_size_S;
#else
  const size_t TRAINING_SAMPLE_SZ = rmi->training_sample_size;
#endif

  // Initialize the exception lists for handling repeated keys
  vector<Tuple<KeyType, PayloadType>> repeated_keys;  // Stores the heavily repeated key values

  Tuple<KeyType, PayloadType>* curr_out_partitions[FANOUT];
  int64_t curr_out_partitions_hist[FANOUT];

  Tuple<KeyType, PayloadType>* curr_out_repeated_keys[FANOUT];
  int64_t * curr_out_repeated_keys_counts[FANOUT];
  int64_t curr_out_repeated_keys_hist[FANOUT];
  int64_t curr_out_total_repeated_keys_hist[FANOUT];

  for(unsigned int i = 0; i < FANOUT; i++)
  {
    curr_out_partitions[i] = out_partitions + out_partitions_offsets[i];
    curr_out_partitions_hist[i] = 0;
  
    curr_out_repeated_keys[i] = out_repeated_keys + out_repeated_keys_offsets[i];
    curr_out_repeated_keys_counts[i] = out_repeated_keys_counts + out_repeated_keys_offsets[i];
    curr_out_repeated_keys_hist[i] = 0;
    curr_out_total_repeated_keys_hist[i] = 0;
  }

  // Cache the model parameters
  auto root_slope = rmi->models[0][0].slope;
  auto root_intrcpt = rmi->models[0][0].intercept;
  unsigned int num_models = rmi->hp.arch[1];
  vector<double> slopes, intercepts;
  for (unsigned int i = 0; i < num_models; ++i) {
    slopes.push_back(rmi->models[1][i].slope);
    intercepts.push_back(rmi->models[1][i].intercept);
  }


  //----------------------------------------------------------//
  //       DETECT REPEATED KEYS IN THE TRAINING SAMPLE        //
  //----------------------------------------------------------//

  // Count the occurrences of equal keys
#ifdef BUILD_RMI_FROM_TWO_DATASETS
  unsigned int cnt_rep_keys = 1;
  if(is_R_relation)
  {
    for (size_t i = 1; i < TRAINING_SAMPLE_SZ; i++) {
      if (((*(rmi->training_sample_R))[i]).key == ((*(rmi->training_sample_R))[i - 1]).key) {
        ++cnt_rep_keys;
      } else {  // New values start here. Reset counter. Add value in the
        // exception_vals if above threshold
        if (cnt_rep_keys > EXC_CNT_THRESHOLD) {
          repeated_keys.push_back((*(rmi->training_sample_R))[i - 1]);
        }
        cnt_rep_keys = 1;
      }
    }

    if (cnt_rep_keys > EXC_CNT_THRESHOLD) {  // Last batch of repeated keys
      repeated_keys.push_back((*(rmi->training_sample_R))[TRAINING_SAMPLE_SZ - 1]);
    }  
  }
  else
  {
    for (size_t i = 1; i < TRAINING_SAMPLE_SZ; i++) {
      if (((*(rmi->training_sample_S))[i]).key == ((*(rmi->training_sample_S))[i - 1]).key) {
        ++cnt_rep_keys;
      } else {  // New values start here. Reset counter. Add value in the
        // exception_vals if above threshold
        if (cnt_rep_keys > EXC_CNT_THRESHOLD) {
          repeated_keys.push_back((*(rmi->training_sample_S))[i - 1]);
        }
        cnt_rep_keys = 1;
      }
    }

    if (cnt_rep_keys > EXC_CNT_THRESHOLD) {  // Last batch of repeated keys
      repeated_keys.push_back((*(rmi->training_sample_S))[TRAINING_SAMPLE_SZ - 1]);
    }  
  }
#else
  unsigned int cnt_rep_keys = 1;
  for (size_t i = 1; i < TRAINING_SAMPLE_SZ; i++) {
    if (((*(rmi->training_sample))[i]).key == ((*(rmi->training_sample))[i - 1]).key) {
      ++cnt_rep_keys;
    } else {  // New values start here. Reset counter. Add value in the
      // exception_vals if above threshold
      if (cnt_rep_keys > EXC_CNT_THRESHOLD) {
        repeated_keys.push_back((*(rmi->training_sample))[i - 1]);
      }
      cnt_rep_keys = 1;
    }
  }

  if (cnt_rep_keys > EXC_CNT_THRESHOLD) {  // Last batch of repeated keys
    repeated_keys.push_back((*(rmi->training_sample))[TRAINING_SAMPLE_SZ - 1]);
  }
#endif

  //----------------------------------------------------------//
  //             SHUFFLE THE KEYS INTO BUCKETS                //
  //----------------------------------------------------------//

  // For each spike value, predict the bucket.
  int pred_model_idx = 0;
  double pred_cdf = 0.;
  for (size_t i = 0; i < repeated_keys.size(); ++i) {
    pred_model_idx = static_cast<int>(
        std::max(0., std::min(num_models - 1.,
                              root_slope * repeated_keys[i].key + root_intrcpt)));
    pred_cdf =
        slopes[pred_model_idx] * repeated_keys[i].key + intercepts[pred_model_idx];
    pred_model_idx = static_cast<int>(
        std::max(0., std::min(FANOUT - 1., pred_cdf * FANOUT)));


    (curr_out_repeated_keys[pred_model_idx])[curr_out_repeated_keys_hist[pred_model_idx]] = repeated_keys[i];
    //(curr_out_repeated_keys_counts[pred_model_idx])[curr_out_repeated_keys_hist[pred_model_idx]] = 0;
    ++curr_out_repeated_keys_hist[pred_model_idx];
  }

  Tuple<KeyType, PayloadType>* end = begin + INPUT_SZ;
  if (repeated_keys.size() == 0)
  {// No significantly repeated keys in the sample
    pred_model_idx = 0;

    // Process each key in order
    for (auto cur_key = begin; cur_key < end; ++cur_key) {
      // Predict the model idx in the leaf layer
      pred_model_idx = static_cast<int>(std::max(
          0.,
          std::min(num_models - 1., root_slope * cur_key->key + root_intrcpt)));

      // Predict the CDF
      pred_cdf =
          slopes[pred_model_idx] * cur_key->key + intercepts[pred_model_idx];

      // Scale the CDF to the number of buckets
      pred_model_idx = static_cast<int>(
          std::max(0., std::min(FANOUT - 1., pred_cdf * FANOUT)));

      (curr_out_partitions[pred_model_idx])[curr_out_partitions_hist[pred_model_idx]] = *cur_key;
      ++curr_out_partitions_hist[pred_model_idx];
    }
  }
  else
  { // There are many repeated keys in the sample

    // Batch size for exceptions
    static constexpr unsigned int BATCH_SZ_EXP = 100;

    // Stores the predicted bucket for each input key in the current batch
    unsigned int pred_idx_in_batch_exc[BATCH_SZ_EXP] = {0};

    // Process elements in batches of size BATCH_SZ_EXP
    for (auto cur_key = begin; cur_key < end; cur_key += BATCH_SZ_EXP) {
      // Process each element in the batch and save their predicted indices
      for (unsigned int elm_idx = 0; elm_idx < BATCH_SZ_EXP; ++elm_idx) {
        // Predict the leaf model idx
        pred_idx_in_batch_exc[elm_idx] = static_cast<int>(std::max(
            0., std::min(num_models - 1.,
                         root_slope * cur_key[elm_idx].key + root_intrcpt)));

        // Predict the CDF
        pred_cdf = slopes[pred_idx_in_batch_exc[elm_idx]] * cur_key[elm_idx].key +
                   intercepts[pred_idx_in_batch_exc[elm_idx]];

        // Extrapolate the CDF to the number of buckets
        pred_idx_in_batch_exc[elm_idx] = static_cast<unsigned int>(
            std::max(0., std::min(FANOUT - 1., pred_cdf * FANOUT)));
      }

      // Go over the batch again and place the flagged keys in an exception
      // list
      bool exc_found = false;  // If exceptions in the batch, don't insert into
                               // buckets, but save in an exception list
      for (unsigned int elm_idx = 0; elm_idx < BATCH_SZ_EXP; ++elm_idx) {
        exc_found = false;
        // Iterate over the keys in the exception list corresponding to the
        // predicted rank for the current key in the batch and the rank of the
        // exception

        for (unsigned int j = 0;
             j < curr_out_repeated_keys_hist[pred_idx_in_batch_exc[elm_idx]];
             ++j) {
          // If key in exception list, then flag it and update the counts that
          // will be used later
          if(((curr_out_repeated_keys[pred_idx_in_batch_exc[elm_idx]])[j]).key == cur_key[elm_idx].key)
          {
            ++((curr_out_repeated_keys_counts[pred_idx_in_batch_exc[elm_idx]])[j]); // Increment count of exception value

            exc_found = true;
            ++curr_out_total_repeated_keys_hist[pred_idx_in_batch_exc[elm_idx]];
            break;
          }
        }

        if (!exc_found)  // If no exception value was found in the batch,
                         // then proceed to putting them in the predicted
                         // buckets
        {
          (curr_out_partitions[pred_idx_in_batch_exc[elm_idx]])[curr_out_partitions_hist[pred_idx_in_batch_exc[elm_idx]]] = cur_key[elm_idx];
          ++curr_out_partitions_hist[pred_idx_in_batch_exc[elm_idx]];    
        }
      }
    }
  }

  for(unsigned int i = 0; i < FANOUT; i++)
  {
    out_partitions_hist[i] = curr_out_partitions_hist[i];
    out_repeated_keys_hist[i] = curr_out_repeated_keys_hist[i];
    out_total_repeated_keys_hist[i] = curr_out_repeated_keys_hist[i];
  }
}

template<class KeyType, class PayloadType>
void learned_sort_for_sort_merge::partition_major_buckets_threaded(int is_R_relation, 
  learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi, unsigned int partitions_fanout,
  Tuple<KeyType, PayloadType> * begin, unsigned int INPUT_SZ, 
  Tuple<KeyType, PayloadType> ** out_partitions, int64_t* out_partitions_offsets, int64_t ** out_partitions_hist,  
  Tuple<KeyType, PayloadType> ** out_repeated_keys, int64_t ** out_repeated_keys_counts, int64_t* out_repeated_keys_offsets, int64_t ** out_repeated_keys_hist, int64_t ** out_total_repeated_keys_hist, 
  int thread_id, int partition_id)
{
  // NOTE: partitions_fanout is supposted to be equal to the number of threads
  // Cache runtime parameters
  static const unsigned int FANOUT = partitions_fanout;

  // Constants for repeated keys
  //const unsigned int EXCEPTION_VEC_INIT_CAPACITY = FANOUT;
  constexpr unsigned int EXC_CNT_THRESHOLD = 60;

#ifdef BUILD_RMI_FROM_TWO_DATASETS
  size_t TRAINING_SAMPLE_SZ;
  if(is_R_relation)
    TRAINING_SAMPLE_SZ = rmi->training_sample_size_R;
  else
    TRAINING_SAMPLE_SZ = rmi->training_sample_size_S;
#else
  const size_t TRAINING_SAMPLE_SZ = rmi->training_sample_size;
#endif

  // Initialize the exception lists for handling repeated keys
  vector<Tuple<KeyType, PayloadType>> repeated_keys;  // Stores the heavily repeated key values

  Tuple<KeyType, PayloadType>* curr_out_partitions[FANOUT];
  int64_t curr_out_partitions_hist[FANOUT];

  Tuple<KeyType, PayloadType>* curr_out_repeated_keys[FANOUT];
  int64_t * curr_out_repeated_keys_counts[FANOUT];
  int64_t curr_out_repeated_keys_hist[FANOUT];
  int64_t curr_out_total_repeated_keys_hist[FANOUT];

  for(unsigned int i = 0; i < FANOUT; i++)
  {
    curr_out_partitions[i] = out_partitions[thread_id] + out_partitions_offsets[i];
    curr_out_partitions_hist[i] = 0;
  
    curr_out_repeated_keys[i] = out_repeated_keys[thread_id] + out_repeated_keys_offsets[i];
    curr_out_repeated_keys_counts[i] = out_repeated_keys_counts[thread_id] + out_repeated_keys_offsets[i];
    curr_out_repeated_keys_hist[i] = 0;
    curr_out_total_repeated_keys_hist[i] = 0;
  }

  // Cache the model parameters
  auto root_slope = rmi->models[0][0].slope;
  auto root_intrcpt = rmi->models[0][0].intercept;
  unsigned int num_models = rmi->hp.arch[1];
  vector<double> slopes, intercepts;
  for (unsigned int i = 0; i < num_models; ++i) {
    slopes.push_back(rmi->models[1][i].slope);
    intercepts.push_back(rmi->models[1][i].intercept);
  }

  //----------------------------------------------------------//
  //       DETECT REPEATED KEYS IN THE TRAINING SAMPLE        //
  //----------------------------------------------------------//

  // Count the occurrences of equal keys
#ifdef BUILD_RMI_FROM_TWO_DATASETS
  unsigned int cnt_rep_keys = 1;
  if(is_R_relation)
  {
    for (size_t i = 1; i < TRAINING_SAMPLE_SZ; i++) {
      if (((*(rmi->training_sample_R))[i]).key == ((*(rmi->training_sample_R))[i - 1]).key) {
        ++cnt_rep_keys;
      } else {  // New values start here. Reset counter. Add value in the
        // exception_vals if above threshold
        if (cnt_rep_keys > EXC_CNT_THRESHOLD) {
          repeated_keys.push_back((*(rmi->training_sample_R))[i - 1]);
        }
        cnt_rep_keys = 1;
      }
    }

    if (cnt_rep_keys > EXC_CNT_THRESHOLD) {  // Last batch of repeated keys
      repeated_keys.push_back((*(rmi->training_sample_R))[TRAINING_SAMPLE_SZ - 1]);
    }  
  }
  else
  {
    for (size_t i = 1; i < TRAINING_SAMPLE_SZ; i++) {
      if (((*(rmi->training_sample_S))[i]).key == ((*(rmi->training_sample_S))[i - 1]).key) {
        ++cnt_rep_keys;
      } else {  // New values start here. Reset counter. Add value in the
        // exception_vals if above threshold
        if (cnt_rep_keys > EXC_CNT_THRESHOLD) {
          repeated_keys.push_back((*(rmi->training_sample_S))[i - 1]);
        }
        cnt_rep_keys = 1;
      }
    }

    if (cnt_rep_keys > EXC_CNT_THRESHOLD) {  // Last batch of repeated keys
      repeated_keys.push_back((*(rmi->training_sample_S))[TRAINING_SAMPLE_SZ - 1]);
    }  
  }

#else
  unsigned int cnt_rep_keys = 1;
  for (size_t i = 1; i < TRAINING_SAMPLE_SZ; i++) {
    if (((*(rmi->training_sample))[i]).key == ((*(rmi->training_sample))[i - 1]).key) {
      ++cnt_rep_keys;
    } else {  // New values start here. Reset counter. Add value in the
      // exception_vals if above threshold
      if (cnt_rep_keys > EXC_CNT_THRESHOLD) {
        repeated_keys.push_back((*(rmi->training_sample))[i - 1]);
      }
      cnt_rep_keys = 1;
    }
  }

  if (cnt_rep_keys > EXC_CNT_THRESHOLD) {  // Last batch of repeated keys
    repeated_keys.push_back((*(rmi->training_sample))[TRAINING_SAMPLE_SZ - 1]);
  }
#endif

  //----------------------------------------------------------//
  //             SHUFFLE THE KEYS INTO BUCKETS                //
  //----------------------------------------------------------//

  // For each spike value, predict the bucket.
  int pred_model_idx = 0;
  double pred_cdf = 0.;
  for (size_t i = 0; i < repeated_keys.size(); ++i) {
    pred_model_idx = static_cast<int>(
        std::max(0., std::min(num_models - 1.,
                              root_slope * repeated_keys[i].key + root_intrcpt)));
    pred_cdf =
        slopes[pred_model_idx] * repeated_keys[i].key + intercepts[pred_model_idx];
    pred_model_idx = static_cast<int>(
        std::max(0., std::min(FANOUT - 1., pred_cdf * FANOUT)));


    (curr_out_repeated_keys[pred_model_idx])[curr_out_repeated_keys_hist[pred_model_idx]] = repeated_keys[i];
    //(curr_out_repeated_keys_counts[pred_model_idx])[curr_out_repeated_keys_hist[pred_model_idx]] = 0;
    ++curr_out_repeated_keys_hist[pred_model_idx];
  }

  Tuple<KeyType, PayloadType>* end = begin + INPUT_SZ;
  if (repeated_keys.size() == 0)
  {// No significantly repeated keys in the sample
    pred_model_idx = 0;

    // Process each key in order
    for (auto cur_key = begin; cur_key < end; ++cur_key) {
      // Predict the model idx in the leaf layer
      pred_model_idx = static_cast<int>(std::max(
          0.,
          std::min(num_models - 1., root_slope * cur_key->key + root_intrcpt)));

      // Predict the CDF
      pred_cdf =
          slopes[pred_model_idx] * cur_key->key + intercepts[pred_model_idx];

      // Scale the CDF to the number of buckets
      pred_model_idx = static_cast<int>(
          std::max(0., std::min(FANOUT - 1., pred_cdf * FANOUT)));

      (curr_out_partitions[pred_model_idx])[curr_out_partitions_hist[pred_model_idx]] = *cur_key;
      ++curr_out_partitions_hist[pred_model_idx];
    }
  }
  else
  { // There are many repeated keys in the sample

    // Batch size for exceptions
    static constexpr unsigned int BATCH_SZ_EXP = 100;

    // Stores the predicted bucket for each input key in the current batch
    unsigned int pred_idx_in_batch_exc[BATCH_SZ_EXP] = {0};

    // Process elements in batches of size BATCH_SZ_EXP
    for (auto cur_key = begin; cur_key < end; cur_key += BATCH_SZ_EXP) {
      // Process each element in the batch and save their predicted indices
      for (unsigned int elm_idx = 0; elm_idx < BATCH_SZ_EXP; ++elm_idx) {
        // Predict the leaf model idx
        pred_idx_in_batch_exc[elm_idx] = static_cast<int>(std::max(
            0., std::min(num_models - 1.,
                         root_slope * cur_key[elm_idx].key + root_intrcpt)));

        // Predict the CDF
        pred_cdf = slopes[pred_idx_in_batch_exc[elm_idx]] * cur_key[elm_idx].key +
                   intercepts[pred_idx_in_batch_exc[elm_idx]];

        // Extrapolate the CDF to the number of buckets
        pred_idx_in_batch_exc[elm_idx] = static_cast<unsigned int>(
            std::max(0., std::min(FANOUT - 1., pred_cdf * FANOUT)));
      }

      // Go over the batch again and place the flagged keys in an exception
      // list
      bool exc_found = false;  // If exceptions in the batch, don't insert into
                               // buckets, but save in an exception list
      for (unsigned int elm_idx = 0; elm_idx < BATCH_SZ_EXP; ++elm_idx) {
        exc_found = false;
        // Iterate over the keys in the exception list corresponding to the
        // predicted rank for the current key in the batch and the rank of the
        // exception

        for (unsigned int j = 0;
             j < curr_out_repeated_keys_hist[pred_idx_in_batch_exc[elm_idx]];
             ++j) {
          // If key in exception list, then flag it and update the counts that
          // will be used later
          if(((curr_out_repeated_keys[pred_idx_in_batch_exc[elm_idx]])[j]).key == cur_key[elm_idx].key)
          {
            ++((curr_out_repeated_keys_counts[pred_idx_in_batch_exc[elm_idx]])[j]); // Increment count of exception value

            exc_found = true;
            ++curr_out_total_repeated_keys_hist[pred_idx_in_batch_exc[elm_idx]];
            break;
          }
        }

        if (!exc_found)  // If no exception value was found in the batch,
                         // then proceed to putting them in the predicted
                         // buckets
        {
          (curr_out_partitions[pred_idx_in_batch_exc[elm_idx]])[curr_out_partitions_hist[pred_idx_in_batch_exc[elm_idx]]] = cur_key[elm_idx];
          ++curr_out_partitions_hist[pred_idx_in_batch_exc[elm_idx]];    
        }
      }
    }
  }

  for(unsigned int i = 0; i < FANOUT; i++)
  {
    out_partitions_hist[thread_id][i] = curr_out_partitions_hist[i];
    out_repeated_keys_hist[thread_id][i] = curr_out_repeated_keys_hist[i];
    out_total_repeated_keys_hist[thread_id][i] = curr_out_repeated_keys_hist[i];
  }
 
}

template<class KeyType, class PayloadType>
void learned_sort_for_sort_merge::sort_avx(Tuple<KeyType, PayloadType> * sorted_output, learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi,
                                          unsigned int NUM_MINOR_BCKT_PER_MAJOR_BCKT, unsigned int MINOR_BCKTS_OFFSET, unsigned int TOT_NUM_MINOR_BCKTS,
                                          unsigned int INPUT_SZ, Tuple<KeyType, PayloadType> * major_bckt, uint64_t major_bckt_size, 
                                          Tuple<KeyType, PayloadType> * minor_bckts, int64_t * minor_bckt_sizes,
                                          Tuple<KeyType, PayloadType> * tmp_spill_bucket, Tuple<KeyType, PayloadType> * sorted_spill_bucket, 
                                          int64_t total_repeated_keys, Tuple<KeyType, PayloadType> * repeated_keys_predicted_ranks, 
                                          int64_t * repeated_key_counts, int64_t repeated_keys_predicted_ranks_count,
                                          int thread_id, int partition_id)
{
  // Cache runtime parameters
  static const unsigned int THRESHOLD = rmi->hp.threshold;
  static const unsigned int BATCH_SZ = rmi->hp.batch_sz;
 
  auto root_slope = rmi->models[0][0].slope;
  auto root_intrcpt = rmi->models[0][0].intercept;
  unsigned int num_models = rmi->hp.arch[1];
  vector<double> slopes, intercepts;
  for (unsigned int i = 0; i < num_models; ++i) {
    slopes.push_back(rmi->models[1][i].slope);
    intercepts.push_back(rmi->models[1][i].intercept);
  }

//if(thread_id==0 && partition_id==0)
//{
//  printf("thread_id %d NUM_MINOR_BCKT_PER_MAJOR_BCKT %ld MINOR_BCKTS_OFFSET %ld TOT_NUM_MINOR_BCKTS %ld INPUT_SZ %ld major_bckt_size %ld total_repeated_keys %ld repeated_keys_predicted_ranks_count %ld\n", thread_id, NUM_MINOR_BCKT_PER_MAJOR_BCKT, MINOR_BCKTS_OFFSET, TOT_NUM_MINOR_BCKTS, INPUT_SZ, major_bckt_size, total_repeated_keys, repeated_keys_predicted_ranks_count);
//}

  //vector<Tuple<KeyType, PayloadType>> minor_bckts(NUM_MINOR_BCKT_PER_MAJOR_BCKT * THRESHOLD);

  // Initialize the spill bucket
  Tuple<KeyType, PayloadType>* spill_bucket;
  unsigned int spill_bucket_size = 0;
  //vector<Tuple<KeyType, PayloadType>> tmp_spill_bucket;
  //spill_bucket = &tmp_spill_bucket;

  // Stores the index where the current bucket will start
  int bckt_start_offset = 0;

  // Stores the predicted CDF values for the elements in the current bucket
  #ifndef USE_AVXSORT_FOR_SORTING_MINOR_BCKTS      
  unsigned int pred_idx_cache[THRESHOLD];
  #endif

#ifndef LS_FOR_SORT_MERGE_IMV_AVX_MINOR_BCKTS
  // Caches the predicted bucket indices for each element in the batch
  vector<unsigned int> batch_cache(BATCH_SZ, 0);

    // Find out the number of batches for this bucket
  unsigned int num_batches = major_bckt_size / BATCH_SZ;
#endif

  // Array to keep track of sizes for the minor buckets in the current
  // bucket
  //vector<unsigned int> minor_bckt_sizes(NUM_MINOR_BCKT_PER_MAJOR_BCKT, 0);

  double pred_cdf = 0.;

  // Counts the nubmer of total elements that are in the buckets, hence
  // INPUT_SZ - spill_bucket.size() at the end of the recursive bucketization
  int64_t num_tot_elms_in_bckts = 0;

#ifdef LS_FOR_SORT_MERGE_IMV_AVX
  
  int32_t num, num_temp, k = 0, done = 0;
  void * curr_major_bckt_off;

  __m512i v_base_offset, v_base_offset_upper = _mm512_set1_epi64(major_bckt_size * sizeof(Tuple<KeyType, PayloadType>)), v_offset = _mm512_set1_epi64(0),
          v_slopes_addr = _mm512_set1_epi64((uint64_t) (&slopes[0])), v_intercepts_addr = _mm512_set1_epi64((uint64_t) (&intercepts[0])),
          general_reg_1, general_reg_2, v_pred_model_idx, v_all_ones = _mm512_set1_epi64(-1), v512_one = _mm512_set1_epi64(1), minor_bckt_sizes_512_avx, v_conflict;

  __m512d general_reg_1_double, general_reg_2_double, root_slope_avx = _mm512_set1_pd(root_slope), root_intrcpt_avx = _mm512_set1_pd(root_intrcpt),
          num_models_minus_one_avx = _mm512_set1_pd((double)num_models - 1.), v_zero512_double = _mm512_set1_pd(0.), 
          v_64bit_elem_size_double = _mm512_set1_pd(8.), minor_bckts_offset_avx = _mm512_set1_pd((double)MINOR_BCKTS_OFFSET),
          num_minor_bckt_per_major_bckt_minus_one_avx = _mm512_set1_pd((double)NUM_MINOR_BCKT_PER_MAJOR_BCKT - 1.),
          v_minor_bckt_sizes_addr_double = _mm512_cvtepi64_pd(_mm512_set1_epi64((uint64_t) minor_bckt_sizes)), threshold_avx = _mm512_set1_pd((double)THRESHOLD),
          v_minor_bckts_addr_double = _mm512_cvtepi64_pd(_mm512_set1_epi64((uint64_t) minor_bckts)), tot_num_minor_bckts_avx = _mm512_set1_pd((double)TOT_NUM_MINOR_BCKTS),
          intercepts_avx, slopes_avx;

  __attribute__((aligned(CACHE_LINE_SIZE))) uint64_t cur_offset = 0, *minor_bckt_sizes_pos, *minor_bckts_pos, *slopes_pos, *intercepts_pos, base_off[LS_FOR_SORT_MERGE_MAX_VECTOR_SCALE]; 
  __attribute__((aligned(CACHE_LINE_SIZE))) __mmask8 mask[LS_FOR_SORT_MERGE_VECTOR_SCALE + 1], m_minor_bckts_to_handle, m_no_conflict, m_spill_bckts_to_handle;
  __attribute__((aligned(CACHE_LINE_SIZE))) StateSIMDForLearnedSort state[LS_FOR_SORT_MERGE_SIMDStateSize + 1];

  for (int i = 0; i <= LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i) 
  {
      base_off[i] = i * sizeof(Tuple<KeyType, PayloadType>);
      mask[i] = (1 << i) - 1;
  }
  v_base_offset = _mm512_load_epi64(base_off);

#endif

#ifdef LS_FOR_SORT_MERGE_IMV_AVX_MINOR_BCKTS

  curr_major_bckt_off = (void *)major_bckt;

  // init # of the state
  for (int i = 0; i <= LS_FOR_SORT_MERGE_SIMDStateSize; ++i) {
      state[i].stage = 1;
      state[i].m_have_key = 0;
  }

  for (uint64_t cur = 0; 1;) 
  {
    k = (k >= LS_FOR_SORT_MERGE_SIMDStateSize) ? 0 : k;
    if (cur >= major_bckt_size) 
    {
        if (state[k].m_have_key == 0 && state[k].stage != 3) {
            ++done;
            state[k].stage = 3;
            ++k;
            continue;
        }
        if ((done >= LS_FOR_SORT_MERGE_SIMDStateSize)) {
            if (state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key > 0) {
                k = LS_FOR_SORT_MERGE_SIMDStateSize;
                state[LS_FOR_SORT_MERGE_SIMDStateSize].stage = 2;
            } else {
                break;
            }
        }
    }
    
    switch (state[k].stage) 
    {
      case 1: 
      {
      #ifdef LS_FOR_SORT_MERGE_PREFETCH_INPUT_FOR_MINOR_BCKTS
          _mm_prefetch((char *)(curr_major_bckt_off + cur_offset + LS_FOR_SORT_MERGE_PDIS), _MM_HINT_T0);
          _mm_prefetch((char *)(curr_major_bckt_off + cur_offset + LS_FOR_SORT_MERGE_PDIS + CACHE_LINE_SIZE), _MM_HINT_T0);
          _mm_prefetch((char *)(curr_major_bckt_off + cur_offset + LS_FOR_SORT_MERGE_PDIS + 2 * CACHE_LINE_SIZE), _MM_HINT_T0);
      #endif
          v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur_offset), v_base_offset);
          cur_offset = cur_offset + base_off[LS_FOR_SORT_MERGE_VECTOR_SCALE];
          state[k].m_have_key = _mm512_cmpgt_epi64_mask(v_base_offset_upper, v_offset);
          cur = cur + LS_FOR_SORT_MERGE_VECTOR_SCALE;
          if((int)(state[k].m_have_key) == 255)
          {
            state[k].key = _mm512_i64gather_epi64(v_offset, curr_major_bckt_off, 1);
            general_reg_2_double = _mm512_cvtepi64_pd(state[k].key);

            //general_reg_1_double = _mm512_fmadd_pd(general_reg_2_double, root_slope_avx, root_intrcpt_avx);
            //general_reg_1_double = _mm512_min_pd(general_reg_1_double, num_models_minus_one_avx);
            //general_reg_1_double = _mm512_max_pd(general_reg_1_double, v_zero512_double);
            //general_reg_1_double = _mm512_floor_pd(general_reg_1_double);
            //general_reg_1_double = _mm512_mul_pd(general_reg_1_double, v_64bit_elem_size_double);
            general_reg_1_double = _mm512_mul_pd(
                                        _mm512_floor_pd(
                                            _mm512_max_pd(
                                                _mm512_min_pd(
                                                    _mm512_fmadd_pd(general_reg_2_double, root_slope_avx, root_intrcpt_avx), num_models_minus_one_avx), v_zero512_double)), v_64bit_elem_size_double);            

            state[k].pred_model_idx = _mm512_cvtpd_epi64(general_reg_1_double);
          #ifdef LS_FOR_SORT_MERGE_PREFETCH_SLOPES_AND_INTERCEPTS_MINOR_BCKTS
            general_reg_1 = _mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr);
            general_reg_2 = _mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr);

            state[k].stage = 0;

            slopes_pos = (uint64_t *) &general_reg_1;
            intercepts_pos = (uint64_t *) &general_reg_2;     
            for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
            {   
                _mm_prefetch((char * )(slopes_pos[i]), _MM_HINT_T0);
                _mm_prefetch((char * )(intercepts_pos[i]), _MM_HINT_T0);
            }
          #else
            slopes_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr), 0, 1);
            intercepts_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr), 0, 1);
            
            general_reg_1_double = _mm512_fmadd_pd(general_reg_2_double, slopes_avx, intercepts_avx);

            //general_reg_1_double = _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, minor_bckts_offset_avx); 
            //general_reg_1_double = _mm512_min_pd(general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
            //general_reg_1_double = _mm512_max_pd(general_reg_1_double, v_zero512_double);
            //general_reg_1_double = _mm512_floor_pd(general_reg_1_double);

            general_reg_1_double = _mm512_floor_pd(
                                        _mm512_max_pd(
                                            _mm512_min_pd(
                                                _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, minor_bckts_offset_avx), num_minor_bckt_per_major_bckt_minus_one_avx), v_zero512_double));

            state[k].pred_model_idx = _mm512_cvtpd_epi64(general_reg_1_double);

            general_reg_2 = _mm512_cvtpd_epi64(
                                _mm512_fmadd_pd(general_reg_1_double, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

            //minor_bckt_sizes_512_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
            //general_reg_2_double = _mm512_cvtepi32_pd(minor_bckt_sizes_512_avx); 
            minor_bckt_sizes_512_avx = _mm512_i64gather_epi64(general_reg_2, 0, 1);
            general_reg_2_double = _mm512_cvtepi64_pd(minor_bckt_sizes_512_avx); 

            state[k].stage = 2;

            m_minor_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, threshold_avx);
            if(m_minor_bckts_to_handle)
            {
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
              general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

              //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

            #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF
                minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
            #endif
                minor_bckts_pos = (uint64_t *) &general_reg_1;
                for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
                { 
                    if (m_minor_bckts_to_handle & (1 << i)) 
                    {
            #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF            
                        _mm_prefetch((char * )(minor_bckt_sizes_pos[i]), _MM_HINT_T0);
            #endif
                        _mm_prefetch((char * )(minor_bckts_pos[i]), _MM_HINT_T0);
                    }
                }  
            }

          #endif
          }
          else
          {
            state[k].key = _mm512_mask_i64gather_epi64(state[k].key, state[k].m_have_key, v_offset, curr_major_bckt_off, 1);
            general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, state[k].m_have_key, state[k].key);

            general_reg_1_double = _mm512_mask_fmadd_pd(general_reg_2_double, state[k].m_have_key, root_slope_avx, root_intrcpt_avx);
            general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, num_models_minus_one_avx);
            general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
            general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);
            general_reg_1_double = _mm512_mask_mul_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_64bit_elem_size_double);
            
            state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);

          #ifdef LS_FOR_SORT_MERGE_PREFETCH_SLOPES_AND_INTERCEPTS_MINOR_BCKTS
            general_reg_1 = _mm512_mask_add_epi64(general_reg_1, state[k].m_have_key, state[k].pred_model_idx, v_slopes_addr);
            general_reg_2 = _mm512_mask_add_epi64(general_reg_2, state[k].m_have_key, state[k].pred_model_idx, v_intercepts_addr);
        
            state[k].stage = 0;

            slopes_pos = (uint64_t *) &general_reg_1;
            intercepts_pos = (uint64_t *) &general_reg_2;     
            for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
            {   
                if (state[k].m_have_key & (1 << i))
                {
                    _mm_prefetch((char * )(slopes_pos[i]), _MM_HINT_T0);
                    _mm_prefetch((char * )(intercepts_pos[i]), _MM_HINT_T0);
                }
            }
          #else
            slopes_avx = _mm512_mask_i64gather_pd(slopes_avx, state[k].m_have_key, 
                                      _mm512_mask_add_epi64(general_reg_1, state[k].m_have_key, state[k].pred_model_idx, v_slopes_addr), 0, 1);
            intercepts_avx = _mm512_mask_i64gather_pd(intercepts_avx, state[k].m_have_key, 
                                _mm512_mask_add_epi64(general_reg_2, state[k].m_have_key, state[k].pred_model_idx, v_intercepts_addr), 0, 1);

            general_reg_1_double = _mm512_mask_fmadd_pd(general_reg_2_double, state[k].m_have_key, slopes_avx, intercepts_avx);
            general_reg_1_double = _mm512_mask_fmsub_pd(general_reg_1_double, state[k].m_have_key, tot_num_minor_bckts_avx, minor_bckts_offset_avx); 
            general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
            general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
            general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);

            state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);

            general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, state[k].m_have_key, 
                                _mm512_mask_fmadd_pd(general_reg_1_double, state[k].m_have_key, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

            //minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi32(minor_bckt_sizes_512_avx, state[k].m_have_key, general_reg_2, 0, 1);
            //general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, state[k].m_have_key, minor_bckt_sizes_512_avx); 
            minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi64(minor_bckt_sizes_512_avx, state[k].m_have_key, general_reg_2, 0, 1);
            general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, state[k].m_have_key, minor_bckt_sizes_512_avx); 

            state[k].stage = 2;

            m_minor_bckts_to_handle = _mm512_mask_cmplt_pd_mask(state[k].m_have_key, general_reg_2_double, threshold_avx);
            if(m_minor_bckts_to_handle)
            {
                general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
                general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
                general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

                //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

            #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF
                minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
            #endif
                minor_bckts_pos = (uint64_t *) &general_reg_1;
                for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
                { 
                    if (m_minor_bckts_to_handle & (1 << i)) 
                    {
            #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF            
                        _mm_prefetch((char * )(minor_bckt_sizes_pos[i]), _MM_HINT_T0);
            #endif
                        _mm_prefetch((char * )(minor_bckts_pos[i]), _MM_HINT_T0);
                    }
                }  
            }
          #endif
          }
      }
      break;
    #ifdef LS_FOR_SORT_MERGE_PREFETCH_SLOPES_AND_INTERCEPTS_MINOR_BCKTS
      case 0: 
      {
        if((int)(state[k].m_have_key) == 255)
        {
          general_reg_2_double = _mm512_cvtepi64_pd(state[k].key);

          slopes_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr), 0, 1);
          intercepts_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr), 0, 1);

          general_reg_1_double = _mm512_fmadd_pd(general_reg_2_double, slopes_avx, intercepts_avx);
          
          //general_reg_1_double = _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, minor_bckts_offset_avx); 
          //general_reg_1_double = _mm512_min_pd(general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
          //general_reg_1_double = _mm512_max_pd(general_reg_1_double, v_zero512_double);
          //general_reg_1_double = _mm512_floor_pd(general_reg_1_double);

          general_reg_1_double = _mm512_floor_pd(
                                      _mm512_max_pd(
                                          _mm512_min_pd(
                                              _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, minor_bckts_offset_avx), num_minor_bckt_per_major_bckt_minus_one_avx), v_zero512_double));

          state[k].pred_model_idx = _mm512_cvtpd_epi64(general_reg_1_double);

          general_reg_2 = _mm512_cvtpd_epi64(
                              _mm512_fmadd_pd(general_reg_1_double, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

          //minor_bckt_sizes_512_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
          //general_reg_2_double = _mm512_cvtepi32_pd(minor_bckt_sizes_512_avx); 
          minor_bckt_sizes_512_avx = _mm512_i64gather_epi64(general_reg_2, 0, 1);
          general_reg_2_double = _mm512_cvtepi64_pd(minor_bckt_sizes_512_avx); 


          state[k].stage = 2;

          m_minor_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, threshold_avx);
          if(m_minor_bckts_to_handle)
          {
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
              general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

              //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

          #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF
              minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
          #endif
              minor_bckts_pos = (uint64_t *) &general_reg_1;
              for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
              { 
                  if (m_minor_bckts_to_handle & (1 << i)) 
                  {
          #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF            
                      _mm_prefetch((char * )(minor_bckt_sizes_pos[i]), _MM_HINT_T0);
          #endif
                      _mm_prefetch((char * )(minor_bckts_pos[i]), _MM_HINT_T0);
                  }
              }  
          }
        }
        else
        { 
          general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, state[k].m_have_key, state[k].key);

          slopes_avx = _mm512_mask_i64gather_pd(slopes_avx, state[k].m_have_key, 
                              _mm512_mask_add_epi64(general_reg_1, state[k].m_have_key, state[k].pred_model_idx, v_slopes_addr), 0, 1);
          intercepts_avx = _mm512_mask_i64gather_pd(intercepts_avx, state[k].m_have_key, 
                              _mm512_mask_add_epi64(general_reg_2, state[k].m_have_key, state[k].pred_model_idx, v_intercepts_addr), 0, 1);
                              
          general_reg_1_double = _mm512_mask_fmadd_pd(general_reg_2_double, state[k].m_have_key, slopes_avx, intercepts_avx);
          general_reg_1_double = _mm512_mask_fmsub_pd(general_reg_1_double, state[k].m_have_key, tot_num_minor_bckts_avx, minor_bckts_offset_avx); 
          general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
          general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
          general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);

          state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);

          general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, state[k].m_have_key, 
                              _mm512_mask_fmadd_pd(general_reg_1_double, state[k].m_have_key, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

          //minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi32(minor_bckt_sizes_512_avx, state[k].m_have_key, general_reg_2, 0, 1);
          //general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, state[k].m_have_key, minor_bckt_sizes_512_avx); 
          minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi64(minor_bckt_sizes_512_avx, state[k].m_have_key, general_reg_2, 0, 1);
          general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, state[k].m_have_key, minor_bckt_sizes_512_avx); 

          state[k].stage = 2;

          m_minor_bckts_to_handle = _mm512_mask_cmplt_pd_mask(state[k].m_have_key, general_reg_2_double, threshold_avx);
          if(m_minor_bckts_to_handle)
          {
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
              general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

              //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

          #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF
              minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
          #endif
              minor_bckts_pos = (uint64_t *) &general_reg_1;
              for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
              { 
                  if (m_minor_bckts_to_handle & (1 << i)) 
                  {
          #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF            
                      _mm_prefetch((char * )(minor_bckt_sizes_pos[i]), _MM_HINT_T0);
          #endif
                      _mm_prefetch((char * )(minor_bckts_pos[i]), _MM_HINT_T0);
                  }
              }  
          }
        }
      }
      break;
    #endif
      case 2:
      {
        v_pred_model_idx = _mm512_mask_blend_epi64(state[k].m_have_key, v_all_ones, state[k].pred_model_idx);
        v_conflict = _mm512_conflict_epi64(v_pred_model_idx);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, state[k].m_have_key);

        if (m_no_conflict) 
        {
          general_reg_1_double = _mm512_mask_cvtepi64_pd(general_reg_1_double, m_no_conflict, state[k].pred_model_idx);

          general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_no_conflict, 
                              _mm512_mask_fmadd_pd(general_reg_1_double, m_no_conflict, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

          //minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi32(minor_bckt_sizes_512_avx, m_no_conflict, general_reg_1, 0, 1);
          //general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, m_no_conflict, minor_bckt_sizes_512_avx); 
          minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi64(minor_bckt_sizes_512_avx, m_no_conflict, general_reg_1, 0, 1);
          general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, m_no_conflict, minor_bckt_sizes_512_avx); 

          m_minor_bckts_to_handle = _mm512_mask_cmplt_pd_mask(m_no_conflict, general_reg_2_double, threshold_avx);
          if(m_minor_bckts_to_handle)
          {
            general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
            general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
            general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, m_minor_bckts_to_handle, general_reg_2_double);

            _mm512_mask_i64scatter_epi64(0, m_minor_bckts_to_handle, general_reg_2, state[k].key, 1);

            //minor_bckt_sizes_512_avx = _mm256_mask_add_epi32(minor_bckt_sizes_512_avx, m_minor_bckts_to_handle, minor_bckt_sizes_512_avx, v512_one);
            //_mm512_mask_i64scatter_epi32(0, m_minor_bckts_to_handle, general_reg_1, minor_bckt_sizes_512_avx, 1);   
            minor_bckt_sizes_512_avx = _mm512_mask_add_epi64(minor_bckt_sizes_512_avx, m_minor_bckts_to_handle, minor_bckt_sizes_512_avx, v512_one);
            _mm512_mask_i64scatter_epi64(0, m_minor_bckts_to_handle, general_reg_1, minor_bckt_sizes_512_avx, 1);   
          }

          m_spill_bckts_to_handle = _mm512_kandn(m_minor_bckts_to_handle, m_no_conflict);
          if(m_spill_bckts_to_handle)
          {
              //auto curr_keys = (int64_t *) &state[k].key;
              auto curr_keys = (Tuple<KeyType, PayloadType> *) &state[k].key;          
              for(int j = 0; j < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++j)
              {
                  if (m_spill_bckts_to_handle & (1 << j)){ 
                      tmp_spill_bucket[spill_bucket_size] = curr_keys[j];
                      ++spill_bucket_size;
                  }
              }
          }
          state[k].m_have_key = _mm512_kandn(m_no_conflict, state[k].m_have_key);
        }
        num = _mm_popcnt_u32(state[k].m_have_key);

        if (num == LS_FOR_SORT_MERGE_VECTOR_SCALE || done >= LS_FOR_SORT_MERGE_SIMDStateSize)
        {
          //auto curr_keys = (int64_t *) &state[k].key;
          auto curr_keys = (Tuple<KeyType, PayloadType> *) &state[k].key;          
          auto pred_model_idx_list = (uint64_t *) &state[k].pred_model_idx;   

          for(int j = 0; j < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++j)
          {
              if (state[k].m_have_key & (1 << j)) 
              {

                  if (minor_bckt_sizes[pred_model_idx_list[j]] <
                      THRESHOLD) {  
                    minor_bckts[THRESHOLD * pred_model_idx_list[j] +
                                minor_bckt_sizes[pred_model_idx_list[j]]] = curr_keys[j];
                    ++minor_bckt_sizes[pred_model_idx_list[j]];
                  } else { 
                    tmp_spill_bucket[spill_bucket_size] = curr_keys[j];
                    ++spill_bucket_size;
                  }
              }                        
          }
          state[k].m_have_key = 0;
          state[k].stage = 1;
          --k;        
        } 
        else if (num == 0) 
        {
          state[k].stage = 1;
          --k;
        }
        else
        {
          if (done < LS_FOR_SORT_MERGE_SIMDStateSize)
          {
              num_temp = _mm_popcnt_u32(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key);
              if (num + num_temp < LS_FOR_SORT_MERGE_VECTOR_SCALE) {
                  // compress v
                  state[k].key = _mm512_maskz_compress_epi64(state[k].m_have_key, state[k].key);
                  state[k].pred_model_idx = _mm512_maskz_compress_epi64(state[k].m_have_key, state[k].pred_model_idx);
                  // expand v -> temp
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].key = _mm512_mask_expand_epi64(state[LS_FOR_SORT_MERGE_SIMDStateSize].key, _mm512_knot(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key), state[k].key);
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].pred_model_idx = _mm512_mask_expand_epi64(state[LS_FOR_SORT_MERGE_SIMDStateSize].pred_model_idx, _mm512_knot(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key), state[k].pred_model_idx);
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key = mask[num + num_temp];
                  state[k].m_have_key = 0;
                  state[k].stage = 1;
                  --k;
              } 
              else
              {
                  // expand temp -> v
                  state[k].key = _mm512_mask_expand_epi64(state[k].key, _mm512_knot(state[k].m_have_key), state[LS_FOR_SORT_MERGE_SIMDStateSize].key);
                  state[k].pred_model_idx = _mm512_mask_expand_epi64(state[k].pred_model_idx, _mm512_knot(state[k].m_have_key), state[LS_FOR_SORT_MERGE_SIMDStateSize].pred_model_idx);
                  // compress temp
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key = _mm512_kand(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key, _mm512_knot(mask[LS_FOR_SORT_MERGE_VECTOR_SCALE - num]));
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].key = _mm512_maskz_compress_epi64(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key, state[LS_FOR_SORT_MERGE_SIMDStateSize].key);
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].pred_model_idx = _mm512_maskz_compress_epi64(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key, state[LS_FOR_SORT_MERGE_SIMDStateSize].pred_model_idx);
                  state[k].m_have_key = mask[LS_FOR_SORT_MERGE_VECTOR_SCALE];
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key = (state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key >> (LS_FOR_SORT_MERGE_VECTOR_SCALE - num));

                  general_reg_1_double = _mm512_cvtepi64_pd(state[k].pred_model_idx);
          
                  general_reg_2 = _mm512_cvtpd_epi64(
                                  _mm512_fmadd_pd(general_reg_1_double, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

                  //minor_bckt_sizes_512_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
                  //general_reg_2_double = _mm512_cvtepi32_pd(minor_bckt_sizes_512_avx); 
                  minor_bckt_sizes_512_avx = _mm512_i64gather_epi64(general_reg_2, 0, 1);
                  general_reg_2_double = _mm512_cvtepi64_pd(minor_bckt_sizes_512_avx); 

                  state[k].stage = 2;

                  m_minor_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, threshold_avx);
                  if(m_minor_bckts_to_handle)
                  {
                      general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
                      general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
                      general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

                  #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF
                      minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
                  #endif
                      minor_bckts_pos = (uint64_t *) &general_reg_1;
                      for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
                      { 
                          if (m_minor_bckts_to_handle & (1 << i)) 
                          {
                  #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF            
                              _mm_prefetch((char * )(minor_bckt_sizes_pos[i]), _MM_HINT_T0);
                  #endif
                              _mm_prefetch((char * )(minor_bckts_pos[i]), _MM_HINT_T0);
                          }
                      }
                  }
              }
          }
        }
      }
      break;
    }

    ++k;
  }

#else
  // Iterate over the elements in the current bucket in batch-mode
  for (unsigned int batch_idx = 0; batch_idx < num_batches; ++batch_idx) 
  {
    // Iterate over the elements in the batch and store their predicted
    // ranks
    for (unsigned int elm_idx = 0; elm_idx < BATCH_SZ; ++elm_idx) 
    {
      // Find the current element
      auto cur_elm = major_bckt[bckt_start_offset + elm_idx];
      
      // Predict the leaf-layer model
      batch_cache[elm_idx] = static_cast<int>(std::max(
          0.,
          std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt)));

      // Predict the CDF
      pred_cdf = slopes[batch_cache[elm_idx]] * cur_elm.key +
                  intercepts[batch_cache[elm_idx]];

      // Scale the predicted CDF to the number of minor buckets and cache it
      batch_cache[elm_idx] = static_cast<int>(std::max(
          0., std::min(NUM_MINOR_BCKT_PER_MAJOR_BCKT - 1.,
                        pred_cdf * TOT_NUM_MINOR_BCKTS - MINOR_BCKTS_OFFSET)));

      //if(thread_id == 0 && partition_id == 0 && (bckt_start_offset + elm_idx) <= 25)
      //if(thread_id == 0 && partition_id == 0 /*&& (bckt_start_offset + elm_idx) <= 500000*/)
      //{
        /*printf("thread_id %d cur_elm %ld batch_cache[elm_idx] %d pred_cdf %lf MINOR_BCKTS_OFFSET %ld NUM_MINOR_BCKT_PER_MAJOR_BCKT %ld batch_cache[elm_idx] %ld \n", 
                thread_id, cur_elm.key, static_cast<int>(std::max(0.,std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt))),
                pred_cdf, MINOR_BCKTS_OFFSET, NUM_MINOR_BCKT_PER_MAJOR_BCKT, batch_cache[elm_idx]);*/

        /*printf("cur_elm %ld root_slope %lf root_intrcpt %lf batch_cache[elm_idx] %d slopes[batch_cache[elm_idx]] %lf intercepts[batch_cache[elm_idx]] %lf pred_cdf %lf pred_cdf * TOT_NUM_MINOR_BCKTS %lf MINOR_BCKTS_OFFSET %ld pred_cdf * TOT_NUM_MINOR_BCKTS - MINOR_BCKTS_OFFSET %lf NUM_MINOR_BCKT_PER_MAJOR_BCKT %ld batch_cache[elm_idx] %ld \n", 
                cur_elm.key, root_slope, root_intrcpt, static_cast<int>(std::max(0.,std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt))), 
                slopes[static_cast<int>(std::max(0., std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt)))], 
                intercepts[static_cast<int>(std::max(0., std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt)))], 
                pred_cdf, pred_cdf * TOT_NUM_MINOR_BCKTS, MINOR_BCKTS_OFFSET, pred_cdf * TOT_NUM_MINOR_BCKTS - MINOR_BCKTS_OFFSET, NUM_MINOR_BCKT_PER_MAJOR_BCKT, batch_cache[elm_idx]);*/
      //}


    }

    // Iterate over the elements in the batch again, and place them in the
    // sub-buckets, or spill bucket
    for (unsigned int elm_idx = 0; elm_idx < BATCH_SZ; ++elm_idx) {
      // Find the current element
      auto cur_elm = major_bckt[bckt_start_offset + elm_idx];

 //if(/*thread_id ==  0 &&*/  partition_id == 0 && (bckt_start_offset + elm_idx) <= 1000)
 //     {
      /*  printf("thread_id %d cur_elm %ld MINOR_BCKTS_OFFSET %ld NUM_MINOR_BCKT_PER_MAJOR_BCKT %ld batch_cache[elm_idx] %ld \n",
                thread_id, cur_elm.key, MINOR_BCKTS_OFFSET, NUM_MINOR_BCKT_PER_MAJOR_BCKT, batch_cache[elm_idx]);*/
// }


      // Check if the element will cause a bucket overflow
      if (minor_bckt_sizes[batch_cache[elm_idx]] <
          THRESHOLD) {  // The predicted bucket has not reached
        // full capacity, so place the element in
        // the bucket
        minor_bckts[THRESHOLD * batch_cache[elm_idx] +
                    minor_bckt_sizes[batch_cache[elm_idx]]] = cur_elm;
        ++minor_bckt_sizes[batch_cache[elm_idx]];
      } else {  // Place the item in the spill bucket
        //spill_bucket->push_back(cur_elm);
        tmp_spill_bucket[spill_bucket_size] = cur_elm;
        ++spill_bucket_size;
      }
    }

    // Update the start offset
    bckt_start_offset += BATCH_SZ;
  }

  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -//
  // Repeat the above for the rest of the elements in the     //
  // current bucket in case its size wasn't divisible by      //
  // the BATCH_SZ                                             //
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
  unsigned int num_remaining_elm =
      major_bckt_size - num_batches * BATCH_SZ;

  for (unsigned int elm_idx = 0; elm_idx < num_remaining_elm; ++elm_idx) {
    auto cur_elm = major_bckt[bckt_start_offset + elm_idx];
    batch_cache[elm_idx] = static_cast<int>(std::max(
        0., std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt)));
    pred_cdf = slopes[batch_cache[elm_idx]] * cur_elm.key +
                intercepts[batch_cache[elm_idx]];
    batch_cache[elm_idx] = static_cast<int>(std::max(
        0., std::min(NUM_MINOR_BCKT_PER_MAJOR_BCKT - 1.,
                      pred_cdf * TOT_NUM_MINOR_BCKTS - MINOR_BCKTS_OFFSET)));
  }

  for (unsigned elm_idx = 0; elm_idx < num_remaining_elm; ++elm_idx) {
    auto cur_elm = major_bckt[bckt_start_offset + elm_idx];
    if (minor_bckt_sizes[batch_cache[elm_idx]] < THRESHOLD) {
      minor_bckts[THRESHOLD * batch_cache[elm_idx] +
                  minor_bckt_sizes[batch_cache[elm_idx]]] = cur_elm;
      ++minor_bckt_sizes[batch_cache[elm_idx]];
    } else {
      //spill_bucket->push_back(cur_elm);
      tmp_spill_bucket[spill_bucket_size] = cur_elm;
      ++spill_bucket_size;
    }
  }

  //if(thread_id == 0 && partition_id == 0 /*&& major_bckt_idx <= 25*/)
  //{
    /*printf("minor bckt sizes \n");
    for(unsigned int i = 0; i < NUM_MINOR_BCKT_PER_MAJOR_BCKT; i++)
    {
      printf("minor bckt %d: %d ", i, minor_bckt_sizes[i]);
    }
    printf("\n");*/
  //}

#endif


  if(thread_id == 0 && partition_id == 0)
  {
    //printf("curr spill bucket is %d \n", spill_bucket->size());
    //printf("curr spill bucket is %d \n", spill_bucket_size);
  }

  //----------------------------------------------------------//
  //                MODEL-BASED COUNTING SORT                 //
  //----------------------------------------------------------//

  // Iterate over the minor buckets of the current bucket
  for (unsigned int bckt_idx = 0; bckt_idx < NUM_MINOR_BCKT_PER_MAJOR_BCKT; ++bckt_idx) 
  {
    if (minor_bckt_sizes[bckt_idx] > 0) 
    {
#ifdef USE_AVXSORT_FOR_SORTING_MINOR_BCKTS      
    Tuple<KeyType, PayloadType> * inputptr_min_bckt =  minor_bckts + bckt_idx * THRESHOLD;
    Tuple<KeyType, PayloadType> * outputptr_major_bckt = major_bckt + num_tot_elms_in_bckts;
    avxsort_tuples<KeyType, PayloadType>(&inputptr_min_bckt, &outputptr_major_bckt, minor_bckt_sizes[bckt_idx]);
    for(unsigned int k = 0; k < minor_bckt_sizes[bckt_idx]; k++){
      major_bckt[num_tot_elms_in_bckts + k].key = outputptr_major_bckt[k].key;
      major_bckt[num_tot_elms_in_bckts + k].payload = outputptr_major_bckt[k].payload;
    }

#else
      // Update the bucket start offset
      bckt_start_offset =
          1. * (MINOR_BCKTS_OFFSET + bckt_idx) *
          INPUT_SZ / TOT_NUM_MINOR_BCKTS;

      // Count array for the model-enhanced counting sort subroutine
      vector<unsigned int> cnt_hist(THRESHOLD, 0);

      /*
      * OPTIMIZATION
      * We check to see if the first and last element in the current bucket
      * used the same leaf model to obtain their CDF. If that is the case,
      * then we don't need to traverse the CDF model for every element in
      * this bucket, hence decreasing the inference complexity from
      * O(num_layer) to O(1).
      */

      int pred_model_first_elm = static_cast<int>(std::max(
          0., std::min(num_models - 1.,
                        root_slope * minor_bckts[bckt_idx * THRESHOLD].key +
                            root_intrcpt)));

      int pred_model_last_elm = static_cast<int>(std::max(
          0., std::min(num_models - 1.,
                    root_slope * minor_bckts[bckt_idx * THRESHOLD +
                                            minor_bckt_sizes[bckt_idx] - 1].key +
                        root_intrcpt)));

      if (pred_model_first_elm == pred_model_last_elm) 
      {  // Avoid CDF model traversal and predict the
         // CDF only using the leaf model
        // Iterate over the elements and place them into the minor buckets
        for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
              ++elm_idx) {
          // Find the current element
          auto cur_elm = minor_bckts[bckt_idx * THRESHOLD + elm_idx];

          // Predict the CDF
          pred_cdf = slopes[pred_model_first_elm] * cur_elm.key +
                      intercepts[pred_model_first_elm];

          // Scale the predicted CDF to the input size and cache it
          pred_idx_cache[elm_idx] = static_cast<int>(std::max(
              0., std::min(THRESHOLD - 1.,
                            (pred_cdf * INPUT_SZ) - bckt_start_offset)));

          // Update the counts
          ++cnt_hist[pred_idx_cache[elm_idx]];
        }
      } 
      else 
      {  // Fully traverse the CDF model again to predict the CDF of
         // the current element
        // Iterate over the elements and place them into the minor buckets
        for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
              ++elm_idx) {
          // Find the current element
          auto cur_elm = minor_bckts[bckt_idx * THRESHOLD + elm_idx];

          // Predict the model idx in the leaf layer
          auto model_idx_next_layer = static_cast<int>(
              std::max(0., std::min(num_models - 1.,
                                    root_slope * cur_elm.key + root_intrcpt)));
          // Predict the CDF
          pred_cdf = slopes[model_idx_next_layer] * cur_elm.key +
                      intercepts[model_idx_next_layer];

          // Scale the predicted CDF to the input size and cache it
          pred_idx_cache[elm_idx] = static_cast<unsigned int>(std::max(
              0., std::min(THRESHOLD - 1.,
                            (pred_cdf * INPUT_SZ) - bckt_start_offset)));

          // Update the counts
          ++cnt_hist[pred_idx_cache[elm_idx]];
        }
      }

      --cnt_hist[0];

      // Calculate the running totals
      for (unsigned int cnt_idx = 1; cnt_idx < THRESHOLD; ++cnt_idx) {
        cnt_hist[cnt_idx] += cnt_hist[cnt_idx - 1];
      }

      // Re-shuffle the elms based on the calculated cumulative counts
      for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
            ++elm_idx) {
        // Place the element in the predicted position in the array
        major_bckt[num_tot_elms_in_bckts + cnt_hist[pred_idx_cache[elm_idx]]] = 
            minor_bckts[bckt_idx * THRESHOLD + elm_idx];
        
        // Update counts
        --cnt_hist[pred_idx_cache[elm_idx]];
      }

      //----------------------------------------------------------//
      //                  TOUCH-UP & COMPACTION                   //
      //----------------------------------------------------------//

      // After the model-based bucketization process is done, switch to a
      // deterministic sort
      Tuple<KeyType, PayloadType> elm;
      int cmp_idx;

      // Perform Insertion Sort
      for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
            ++elm_idx) {
        cmp_idx = num_tot_elms_in_bckts + elm_idx - 1;
        elm = major_bckt[num_tot_elms_in_bckts + elm_idx];
        while (cmp_idx >= 0 && elm.key < major_bckt[cmp_idx].key) {
          major_bckt[cmp_idx + 1] = major_bckt[cmp_idx];
          --cmp_idx;
        }

        major_bckt[cmp_idx + 1] = elm;
      }
#endif


      num_tot_elms_in_bckts += minor_bckt_sizes[bckt_idx];

    } // end of iteration of each minor bucket
  }

  //std::cout << "learned_sort: spill_bucket size: "<< spill_bucket->size() << std::endl;
  //std::cout << "learned_sort: spill_bucket size: "<< spill_bucket_size << std::endl;

  //----------------------------------------------------------//
  //                 SORT THE SPILL BUCKET                    //
  //----------------------------------------------------------//

#ifdef USE_AVXSORT_AS_STD_SORT
  //uint32_t spill_bucket_size = spill_bucket->size();
  //int64_t sorted_spill_bucket_arr [spill_bucket_size];
  //int64_t * inputptr = (int64_t *)(&((*spill_bucket)[0])); 
  //int64_t * outputptr = (int64_t *)(sorted_spill_bucket_arr); 
  //avxsort_int64(&inputptr, &outputptr, spill_bucket_size);
  //vector<Tuple<KeyType, PayloadType>> sorted_spill_bucket((Tuple<KeyType, PayloadType>*)outputptr, ((Tuple<KeyType, PayloadType>*)outputptr) + spill_bucket_size);
  //spill_bucket = &sorted_spill_bucket;

//NOTE: Working code but with extra overhead for assigning tmp_outputptr back to sorted_spill_bucket
//  int64_t * inputptr =  (int64_t *)(tmp_spill_bucket);
//  int64_t * outputptr = (int64_t *)(sorted_spill_bucket);
//  avxsort_int64(&inputptr, &outputptr, spill_bucket_size);
//  Tuple<KeyType, PayloadType>* tmp_outputptr = (Tuple<KeyType, PayloadType>*) outputptr;
//  for(int k = 0; k < spill_bucket_size; k++){
//    sorted_spill_bucket[k].key = tmp_outputptr[k].key;
//    sorted_spill_bucket[k].payload = tmp_outputptr[k].payload;
//  } 
//  spill_bucket = sorted_spill_bucket;

  int64_t * inputptr =  (int64_t *)(tmp_spill_bucket);
  int64_t * outputptr = (int64_t *)(sorted_spill_bucket);
  avxsort_int64(&inputptr, &outputptr, spill_bucket_size);
  //Tuple<KeyType, PayloadType>* tmp_outputptr = (Tuple<KeyType, PayloadType>*) outputptr;
  spill_bucket = (Tuple<KeyType, PayloadType>*)outputptr;
  //sorted_spill_bucket = (Tuple<KeyType, PayloadType>*)outputptr;
  //spill_bucket = sorted_spill_bucket;

  //NOTE: Working code but with slower performance 
  //  avxsort_tuples<KeyType, PayloadType>(&tmp_spill_bucket, &sorted_spill_bucket, spill_bucket_size);
  //  spill_bucket = sorted_spill_bucket;
#else
  //std::sort((int64_t *)(&((*spill_bucket)[0])), (int64_t *)(&((*spill_bucket)[0])) + spill_bucket->size());
  ////std::sort(spill_bucket->begin(), spill_bucket->end());

  std::sort((int64_t *)(tmp_spill_bucket), (int64_t *)(tmp_spill_bucket) + spill_bucket_size /*- 1*/);
  spill_bucket = tmp_spill_bucket;
#endif

  //----------------------------------------------------------//
  //               PLACE BACK THE EXCEPTION VALUES            //
  //----------------------------------------------------------//

  vector<Tuple<KeyType, PayloadType>> linear_vals; vector<int64_t> linear_count;
  for (int64_t exc_elm_idx = 0; exc_elm_idx < repeated_keys_predicted_ranks_count; ++exc_elm_idx) {
    linear_vals.push_back(
        repeated_keys_predicted_ranks[exc_elm_idx]);
    linear_count.push_back(repeated_key_counts[exc_elm_idx]);
  }

  //----------------------------------------------------------//
  //               MERGE BACK INTO ORIGINAL ARRAY             //
  //----------------------------------------------------------//

  // Merge the spill bucket with the elements in the buckets
  #ifdef USE_AVXSORT_AS_STD_SORT 
    #ifdef USE_AVXMERGE_AS_STD_MERGE
      //avx_merge_int64((int64_t*) (major_bckt), (int64_t *)(&((*spill_bucket)[0])), ((int64_t *)sorted_output) + total_repeated_keys, num_tot_elms_in_bckts, spill_bucket_size);
      avx_merge_int64((int64_t*) (major_bckt), (int64_t *)(spill_bucket), ((int64_t *)sorted_output) + total_repeated_keys, num_tot_elms_in_bckts, spill_bucket_size);
    #else
        //std::merge((int64_t*)major_bckt, ((int64_t*)major_bckt) + num_tot_elms_in_bckts,
        //           (int64_t *)(&((*spill_bucket)[0])), (int64_t *)(&((*spill_bucket)[0])) + spill_bucket->size(),
        //           ((int64_t*)sorted_output) + total_repeated_keys);
        std::merge((int64_t*)major_bckt, ((int64_t*)major_bckt) + num_tot_elms_in_bckts,
                   (int64_t *)(spill_bucket), (int64_t *)(spill_bucket) + spill_bucket_size,
                   ((int64_t*)sorted_output) + total_repeated_keys);
    #endif
  #else
     std::merge((int64_t*)major_bckt, ((int64_t*)major_bckt) + num_tot_elms_in_bckts,
              (int64_t *)(spill_bucket), (int64_t *)(spill_bucket) + spill_bucket_size,
              ((int64_t*)sorted_output) + total_repeated_keys);
   //std::merge((int64_t*)major_bckt, ((int64_t*)major_bckt) + num_tot_elms_in_bckts,
   //           (int64_t *)(&((*spill_bucket)[0])), (int64_t *)(&((*spill_bucket)[0])) + spill_bucket->size(),
   //           ((int64_t*)sorted_output) + total_repeated_keys);
      //std::merge(major_bckt, major_bckt + num_tot_elms_in_bckts,
      //       spill_bucket->begin(), spill_bucket->end(),
      //       major_bckt + total_repeated_keys);
  #endif

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
  //            Start merging the exception values            //
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

  // The read index for the exceptions
  unsigned int exc_idx = 0;

  // The read index for the already-merged elements from the buckets and the
  // spill bucket
  unsigned int input_idx = total_repeated_keys;

  // The write index for the final merging of everything
  int ptr = 0;

  int64_t major_bckt_size_and_total_repeated_keys = major_bckt_size + total_repeated_keys;
  while (input_idx < major_bckt_size_and_total_repeated_keys && exc_idx < linear_vals.size()) {
    if (sorted_output[input_idx].key < linear_vals[exc_idx].key) {
      sorted_output[ptr] = sorted_output[input_idx];
      ptr++;
      input_idx++;
    } else {
      for (int i = 0; i < linear_count[exc_idx]; i++) {
        sorted_output[ptr + i] = linear_vals[exc_idx];
      }
      ptr += linear_count[exc_idx];
      exc_idx++;
    }
  }

  while (exc_idx < linear_vals.size()) {
    for (int i = 0; i < linear_count[exc_idx]; i++) {
      sorted_output[ptr + i] = linear_vals[exc_idx];
    }
    ptr += linear_count[exc_idx];
    exc_idx++;
  }

  while (input_idx < major_bckt_size_and_total_repeated_keys) {
    sorted_output[ptr] = sorted_output[input_idx];
    ptr++;
    input_idx++;
  }

  // The input array is now sorted  
}


template<class KeyType, class PayloadType>
void learned_sort_for_sort_merge::sort_avx_from_seperate_partitions(Tuple<KeyType, PayloadType> * sorted_output, learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi,
                                          unsigned int NUM_MINOR_BCKT_PER_MAJOR_BCKT, unsigned int MINOR_BCKTS_OFFSET, unsigned int TOT_NUM_MINOR_BCKTS,
                                          unsigned int INPUT_SZ, Tuple<KeyType, PayloadType> ** major_bckt, uint64_t* major_bckt_size, int64_t major_bckt_partition_offset, Tuple<KeyType, PayloadType> * tmp_major_bckt,
                                          Tuple<KeyType, PayloadType> * minor_bckts, int64_t * minor_bckt_sizes,
                                          Tuple<KeyType, PayloadType> * tmp_spill_bucket, Tuple<KeyType, PayloadType> * sorted_spill_bucket, 
                                          int64_t* total_repeated_keys, int64_t repeated_keys_offset, Tuple<KeyType, PayloadType> ** repeated_keys_predicted_ranks, 
                                          int64_t ** repeated_key_counts, int64_t repeated_keys_predicted_ranks_count,
                                          int thread_id, int partition_id)
{

 // Cache runtime parameters
  static const unsigned int THRESHOLD = rmi->hp.threshold;
  static const unsigned int BATCH_SZ = rmi->hp.batch_sz;
 
  auto root_slope = rmi->models[0][0].slope;
  auto root_intrcpt = rmi->models[0][0].intercept;
  unsigned int num_models = rmi->hp.arch[1];
  vector<double> slopes, intercepts;
  for (unsigned int i = 0; i < num_models; ++i) {
    slopes.push_back(rmi->models[1][i].slope);
    intercepts.push_back(rmi->models[1][i].intercept);
  }

//if(thread_id==0 && partition_id==0)
//{
//  printf("thread_id %d NUM_MINOR_BCKT_PER_MAJOR_BCKT %ld MINOR_BCKTS_OFFSET %ld TOT_NUM_MINOR_BCKTS %ld INPUT_SZ %ld major_bckt_size %ld total_repeated_keys %ld repeated_keys_predicted_ranks_count %ld\n", thread_id, NUM_MINOR_BCKT_PER_MAJOR_BCKT, MINOR_BCKTS_OFFSET, TOT_NUM_MINOR_BCKTS, INPUT_SZ, major_bckt_size, total_repeated_keys, repeated_keys_predicted_ranks_count);
//}

  //vector<Tuple<KeyType, PayloadType>> minor_bckts(NUM_MINOR_BCKT_PER_MAJOR_BCKT * THRESHOLD);

  // Initialize the spill bucket
  Tuple<KeyType, PayloadType>* spill_bucket;
  unsigned int spill_bucket_size = 0;
  //vector<Tuple<KeyType, PayloadType>> tmp_spill_bucket;
  //spill_bucket = &tmp_spill_bucket;

  // Stores the index where the current bucket will start
  int bckt_start_offset = 0;

  // Stores the predicted CDF values for the elements in the current bucket
  #ifndef USE_AVXSORT_FOR_SORT5ING_MINOR_BCKTS      
  unsigned int pred_idx_cache[THRESHOLD];
  #endif

#ifndef LS_FOR_SORT_MERGE_IMV_AVX_MINOR_BCKTS
  // Caches the predicted bucket indices for each element in the batch
  vector<unsigned int> batch_cache(BATCH_SZ, 0);

    // Find out the number of batches for this bucket
  unsigned int num_batches;
#endif

 // Array to keep track of sizes for the minor buckets in the current
  // bucket
  //vector<unsigned int> minor_bckt_sizes(NUM_MINOR_BCKT_PER_MAJOR_BCKT, 0);

  double pred_cdf = 0.;

  // Counts the nubmer of total elements that are in the buckets, hence
  // INPUT_SZ - spill_bucket.size() at the end of the recursive bucketization
  int64_t num_tot_elms_in_bckts = 0;

#ifdef LS_FOR_SORT_MERGE_IMV_AVX
  
  int32_t num, num_temp, k = 0, done = 0;
  void * curr_major_bckt_off;

  __m512i v_base_offset, v_base_offset_upper, v_offset,
          v_slopes_addr = _mm512_set1_epi64((uint64_t) (&slopes[0])), v_intercepts_addr = _mm512_set1_epi64((uint64_t) (&intercepts[0])),
          general_reg_1, general_reg_2, v_pred_model_idx, v_all_ones = _mm512_set1_epi64(-1), v512_one = _mm512_set1_epi64(1), minor_bckt_sizes_512_avx, v_conflict;

  __m512d general_reg_1_double, general_reg_2_double, root_slope_avx = _mm512_set1_pd(root_slope), root_intrcpt_avx = _mm512_set1_pd(root_intrcpt),
          num_models_minus_one_avx = _mm512_set1_pd((double)num_models - 1.), v_zero512_double = _mm512_set1_pd(0.), 
          v_64bit_elem_size_double = _mm512_set1_pd(8.), minor_bckts_offset_avx = _mm512_set1_pd((double)MINOR_BCKTS_OFFSET),
          num_minor_bckt_per_major_bckt_minus_one_avx = _mm512_set1_pd((double)NUM_MINOR_BCKT_PER_MAJOR_BCKT - 1.),
          v_minor_bckt_sizes_addr_double = _mm512_cvtepi64_pd(_mm512_set1_epi64((uint64_t) minor_bckt_sizes)), threshold_avx = _mm512_set1_pd((double)THRESHOLD),
          v_minor_bckts_addr_double = _mm512_cvtepi64_pd(_mm512_set1_epi64((uint64_t) minor_bckts)), tot_num_minor_bckts_avx = _mm512_set1_pd((double)TOT_NUM_MINOR_BCKTS),
          intercepts_avx, slopes_avx;

  __attribute__((aligned(CACHE_LINE_SIZE))) uint64_t cur_offset, *minor_bckt_sizes_pos, *minor_bckts_pos, *slopes_pos, *intercepts_pos, base_off[LS_FOR_SORT_MERGE_MAX_VECTOR_SCALE]; 
  __attribute__((aligned(CACHE_LINE_SIZE))) __mmask8 mask[LS_FOR_SORT_MERGE_VECTOR_SCALE + 1], m_minor_bckts_to_handle, m_no_conflict, m_spill_bckts_to_handle;
  __attribute__((aligned(CACHE_LINE_SIZE))) StateSIMDForLearnedSort state[LS_FOR_SORT_MERGE_SIMDStateSize + 1];


#endif

uint64_t all_major_bckt_size = 0;

for(int p = 0; p < NUM_THREADS_FOR_EVALUATION; p++)
{
  bckt_start_offset = 0;
  all_major_bckt_size += major_bckt_size[p];

#ifndef LS_FOR_SORT_MERGE_IMV_AVX_MINOR_BCKTS  
  num_batches = major_bckt_size[p] / BATCH_SZ;
#endif

#ifdef LS_FOR_SORT_MERGE_IMV_AVX
  k = 0; done = 0;

  v_base_offset_upper = _mm512_set1_epi64(major_bckt_size[p] * sizeof(Tuple<KeyType, PayloadType>)); v_offset = _mm512_set1_epi64(0);

  cur_offset = 0;

  for (int i = 0; i <= LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i) 
  {
      base_off[i] = i * sizeof(Tuple<KeyType, PayloadType>);
      mask[i] = (1 << i) - 1;
  }
  v_base_offset = _mm512_load_epi64(base_off);

#endif

#ifdef LS_FOR_SORT_MERGE_IMV_AVX_MINOR_BCKTS

  curr_major_bckt_off = (void *)(&(major_bckt[p][major_bckt_partition_offset]));

  // init # of the state
  for (int i = 0; i <= LS_FOR_SORT_MERGE_SIMDStateSize; ++i) {
      state[i].stage = 1;
      state[i].m_have_key = 0;
  }

  for (uint64_t cur = 0; 1;) 
  {
    k = (k >= LS_FOR_SORT_MERGE_SIMDStateSize) ? 0 : k;
    if (cur >= major_bckt_size[p]) 
    {
        if (state[k].m_have_key == 0 && state[k].stage != 3) {
            ++done;
            state[k].stage = 3;
            ++k;
            continue;
        }
        if ((done >= LS_FOR_SORT_MERGE_SIMDStateSize)) {
            if (state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key > 0) {
                k = LS_FOR_SORT_MERGE_SIMDStateSize;
                state[LS_FOR_SORT_MERGE_SIMDStateSize].stage = 2;
            } else {
                break;
            }
        }
    }
    
    switch (state[k].stage) 
    {
      case 1: 
      {
      #ifdef LS_FOR_SORT_MERGE_PREFETCH_INPUT_FOR_MINOR_BCKTS
          _mm_prefetch((char *)(curr_major_bckt_off + cur_offset + LS_FOR_SORT_MERGE_PDIS), _MM_HINT_T0);
          _mm_prefetch((char *)(curr_major_bckt_off + cur_offset + LS_FOR_SORT_MERGE_PDIS + CACHE_LINE_SIZE), _MM_HINT_T0);
          _mm_prefetch((char *)(curr_major_bckt_off + cur_offset + LS_FOR_SORT_MERGE_PDIS + 2 * CACHE_LINE_SIZE), _MM_HINT_T0);
      #endif
          v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur_offset), v_base_offset);
          cur_offset = cur_offset + base_off[LS_FOR_SORT_MERGE_VECTOR_SCALE];
          state[k].m_have_key = _mm512_cmpgt_epi64_mask(v_base_offset_upper, v_offset);
          cur = cur + LS_FOR_SORT_MERGE_VECTOR_SCALE;
          if((int)(state[k].m_have_key) == 255)
          {
            state[k].key = _mm512_i64gather_epi64(v_offset, curr_major_bckt_off, 1);
            general_reg_2_double = _mm512_cvtepi64_pd(state[k].key);

            //general_reg_1_double = _mm512_fmadd_pd(general_reg_2_double, root_slope_avx, root_intrcpt_avx);
            //general_reg_1_double = _mm512_min_pd(general_reg_1_double, num_models_minus_one_avx);
            //general_reg_1_double = _mm512_max_pd(general_reg_1_double, v_zero512_double);
            //general_reg_1_double = _mm512_floor_pd(general_reg_1_double);
            //general_reg_1_double = _mm512_mul_pd(general_reg_1_double, v_64bit_elem_size_double);
            general_reg_1_double = _mm512_mul_pd(
                                        _mm512_floor_pd(
                                            _mm512_max_pd(
                                                _mm512_min_pd(
                                                    _mm512_fmadd_pd(general_reg_2_double, root_slope_avx, root_intrcpt_avx), num_models_minus_one_avx), v_zero512_double)), v_64bit_elem_size_double);            

            state[k].pred_model_idx = _mm512_cvtpd_epi64(general_reg_1_double);
          #ifdef LS_FOR_SORT_MERGE_PREFETCH_SLOPES_AND_INTERCEPTS_MINOR_BCKTS
            general_reg_1 = _mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr);
            general_reg_2 = _mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr);

            state[k].stage = 0;

            slopes_pos = (uint64_t *) &general_reg_1;
            intercepts_pos = (uint64_t *) &general_reg_2;     
            for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
            {   
                _mm_prefetch((char * )(slopes_pos[i]), _MM_HINT_T0);
                _mm_prefetch((char * )(intercepts_pos[i]), _MM_HINT_T0);
            }
          #else
            slopes_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr), 0, 1);
            intercepts_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr), 0, 1);
            
            general_reg_1_double = _mm512_fmadd_pd(general_reg_2_double, slopes_avx, intercepts_avx);

            //general_reg_1_double = _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, minor_bckts_offset_avx); 
            //general_reg_1_double = _mm512_min_pd(general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
            //general_reg_1_double = _mm512_max_pd(general_reg_1_double, v_zero512_double);
            //general_reg_1_double = _mm512_floor_pd(general_reg_1_double);

            general_reg_1_double = _mm512_floor_pd(
                                        _mm512_max_pd(
                                            _mm512_min_pd(
                                                _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, minor_bckts_offset_avx), num_minor_bckt_per_major_bckt_minus_one_avx), v_zero512_double));

            state[k].pred_model_idx = _mm512_cvtpd_epi64(general_reg_1_double);

            general_reg_2 = _mm512_cvtpd_epi64(
                                _mm512_fmadd_pd(general_reg_1_double, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

            //minor_bckt_sizes_512_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
            //general_reg_2_double = _mm512_cvtepi32_pd(minor_bckt_sizes_512_avx); 
            minor_bckt_sizes_512_avx = _mm512_i64gather_epi64(general_reg_2, 0, 1);
            general_reg_2_double = _mm512_cvtepi64_pd(minor_bckt_sizes_512_avx); 

            state[k].stage = 2;

            m_minor_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, threshold_avx);
            if(m_minor_bckts_to_handle)
            {
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
              general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

              //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

            #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF
                minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
            #endif
                minor_bckts_pos = (uint64_t *) &general_reg_1;
                for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
                { 
                    if (m_minor_bckts_to_handle & (1 << i)) 
                    {
            #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF            
                        _mm_prefetch((char * )(minor_bckt_sizes_pos[i]), _MM_HINT_T0);
            #endif
                        _mm_prefetch((char * )(minor_bckts_pos[i]), _MM_HINT_T0);
                    }
                }  
            }

          #endif
          }
          else
          {
            state[k].key = _mm512_mask_i64gather_epi64(state[k].key, state[k].m_have_key, v_offset, curr_major_bckt_off, 1);
            general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, state[k].m_have_key, state[k].key);

            general_reg_1_double = _mm512_mask_fmadd_pd(general_reg_2_double, state[k].m_have_key, root_slope_avx, root_intrcpt_avx);
            general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, num_models_minus_one_avx);
            general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
            general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);
            general_reg_1_double = _mm512_mask_mul_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_64bit_elem_size_double);
            
            state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);

          #ifdef LS_FOR_SORT_MERGE_PREFETCH_SLOPES_AND_INTERCEPTS_MINOR_BCKTS
            general_reg_1 = _mm512_mask_add_epi64(general_reg_1, state[k].m_have_key, state[k].pred_model_idx, v_slopes_addr);
            general_reg_2 = _mm512_mask_add_epi64(general_reg_2, state[k].m_have_key, state[k].pred_model_idx, v_intercepts_addr);
        
            state[k].stage = 0;

            slopes_pos = (uint64_t *) &general_reg_1;
            intercepts_pos = (uint64_t *) &general_reg_2;     
            for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
            {   
                if (state[k].m_have_key & (1 << i))
                {
                    _mm_prefetch((char * )(slopes_pos[i]), _MM_HINT_T0);
                    _mm_prefetch((char * )(intercepts_pos[i]), _MM_HINT_T0);
                }
            }
          #else
            slopes_avx = _mm512_mask_i64gather_pd(slopes_avx, state[k].m_have_key, 
                                      _mm512_mask_add_epi64(general_reg_1, state[k].m_have_key, state[k].pred_model_idx, v_slopes_addr), 0, 1);
            intercepts_avx = _mm512_mask_i64gather_pd(intercepts_avx, state[k].m_have_key, 
                                _mm512_mask_add_epi64(general_reg_2, state[k].m_have_key, state[k].pred_model_idx, v_intercepts_addr), 0, 1);

            general_reg_1_double = _mm512_mask_fmadd_pd(general_reg_2_double, state[k].m_have_key, slopes_avx, intercepts_avx);
            general_reg_1_double = _mm512_mask_fmsub_pd(general_reg_1_double, state[k].m_have_key, tot_num_minor_bckts_avx, minor_bckts_offset_avx); 
            general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
            general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
            general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);

            state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);

            general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, state[k].m_have_key, 
                                _mm512_mask_fmadd_pd(general_reg_1_double, state[k].m_have_key, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

            //minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi32(minor_bckt_sizes_512_avx, state[k].m_have_key, general_reg_2, 0, 1);
            //general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, state[k].m_have_key, minor_bckt_sizes_512_avx); 
            minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi64(minor_bckt_sizes_512_avx, state[k].m_have_key, general_reg_2, 0, 1);
            general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, state[k].m_have_key, minor_bckt_sizes_512_avx); 

            state[k].stage = 2;

            m_minor_bckts_to_handle = _mm512_mask_cmplt_pd_mask(state[k].m_have_key, general_reg_2_double, threshold_avx);
            if(m_minor_bckts_to_handle)
            {
                general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
                general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
                general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

                //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

            #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF
                minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
            #endif
                minor_bckts_pos = (uint64_t *) &general_reg_1;
                for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
                { 
                    if (m_minor_bckts_to_handle & (1 << i)) 
                    {
            #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF            
                        _mm_prefetch((char * )(minor_bckt_sizes_pos[i]), _MM_HINT_T0);
            #endif
                        _mm_prefetch((char * )(minor_bckts_pos[i]), _MM_HINT_T0);
                    }
                }  
            }
          #endif
          }
      }
      break;
    #ifdef LS_FOR_SORT_MERGE_PREFETCH_SLOPES_AND_INTERCEPTS_MINOR_BCKTS
      case 0: 
      {
        if((int)(state[k].m_have_key) == 255)
        {
          general_reg_2_double = _mm512_cvtepi64_pd(state[k].key);

          slopes_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr), 0, 1);
          intercepts_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr), 0, 1);

          general_reg_1_double = _mm512_fmadd_pd(general_reg_2_double, slopes_avx, intercepts_avx);
          
          //general_reg_1_double = _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, minor_bckts_offset_avx); 
          //general_reg_1_double = _mm512_min_pd(general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
          //general_reg_1_double = _mm512_max_pd(general_reg_1_double, v_zero512_double);
          //general_reg_1_double = _mm512_floor_pd(general_reg_1_double);

          general_reg_1_double = _mm512_floor_pd(
                                      _mm512_max_pd(
                                          _mm512_min_pd(
                                              _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, minor_bckts_offset_avx), num_minor_bckt_per_major_bckt_minus_one_avx), v_zero512_double));

          state[k].pred_model_idx = _mm512_cvtpd_epi64(general_reg_1_double);

          general_reg_2 = _mm512_cvtpd_epi64(
                              _mm512_fmadd_pd(general_reg_1_double, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

          //minor_bckt_sizes_512_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
          //general_reg_2_double = _mm512_cvtepi32_pd(minor_bckt_sizes_512_avx); 
          minor_bckt_sizes_512_avx = _mm512_i64gather_epi64(general_reg_2, 0, 1);
          general_reg_2_double = _mm512_cvtepi64_pd(minor_bckt_sizes_512_avx); 


          state[k].stage = 2;

          m_minor_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, threshold_avx);
          if(m_minor_bckts_to_handle)
          {
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
              general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

              //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

          #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF
              minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
          #endif
              minor_bckts_pos = (uint64_t *) &general_reg_1;
              for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
              { 
                  if (m_minor_bckts_to_handle & (1 << i)) 
                  {
          #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF            
                      _mm_prefetch((char * )(minor_bckt_sizes_pos[i]), _MM_HINT_T0);
          #endif
                      _mm_prefetch((char * )(minor_bckts_pos[i]), _MM_HINT_T0);
                  }
              }  
          }
        }
        else
        { 
          general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, state[k].m_have_key, state[k].key);

          slopes_avx = _mm512_mask_i64gather_pd(slopes_avx, state[k].m_have_key, 
                              _mm512_mask_add_epi64(general_reg_1, state[k].m_have_key, state[k].pred_model_idx, v_slopes_addr), 0, 1);
          intercepts_avx = _mm512_mask_i64gather_pd(intercepts_avx, state[k].m_have_key, 
                              _mm512_mask_add_epi64(general_reg_2, state[k].m_have_key, state[k].pred_model_idx, v_intercepts_addr), 0, 1);
                              
          general_reg_1_double = _mm512_mask_fmadd_pd(general_reg_2_double, state[k].m_have_key, slopes_avx, intercepts_avx);
          general_reg_1_double = _mm512_mask_fmsub_pd(general_reg_1_double, state[k].m_have_key, tot_num_minor_bckts_avx, minor_bckts_offset_avx); 
          general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
          general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
          general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);

          state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);

          general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, state[k].m_have_key, 
                              _mm512_mask_fmadd_pd(general_reg_1_double, state[k].m_have_key, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

          //minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi32(minor_bckt_sizes_512_avx, state[k].m_have_key, general_reg_2, 0, 1);
          //general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, state[k].m_have_key, minor_bckt_sizes_512_avx); 
          minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi64(minor_bckt_sizes_512_avx, state[k].m_have_key, general_reg_2, 0, 1);
          general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, state[k].m_have_key, minor_bckt_sizes_512_avx); 

          state[k].stage = 2;

          m_minor_bckts_to_handle = _mm512_mask_cmplt_pd_mask(state[k].m_have_key, general_reg_2_double, threshold_avx);
          if(m_minor_bckts_to_handle)
          {
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
              general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
              general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

              //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

          #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF
              minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
          #endif
              minor_bckts_pos = (uint64_t *) &general_reg_1;
              for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
              { 
                  if (m_minor_bckts_to_handle & (1 << i)) 
                  {
          #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF            
                      _mm_prefetch((char * )(minor_bckt_sizes_pos[i]), _MM_HINT_T0);
          #endif
                      _mm_prefetch((char * )(minor_bckts_pos[i]), _MM_HINT_T0);
                  }
              }  
          }
        }
      }
      break;
    #endif
      case 2:
      {
        v_pred_model_idx = _mm512_mask_blend_epi64(state[k].m_have_key, v_all_ones, state[k].pred_model_idx);
        v_conflict = _mm512_conflict_epi64(v_pred_model_idx);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, state[k].m_have_key);

        if (m_no_conflict) 
        {
          general_reg_1_double = _mm512_mask_cvtepi64_pd(general_reg_1_double, m_no_conflict, state[k].pred_model_idx);

          general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_no_conflict, 
                              _mm512_mask_fmadd_pd(general_reg_1_double, m_no_conflict, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

          //minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi32(minor_bckt_sizes_512_avx, m_no_conflict, general_reg_1, 0, 1);
          //general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, m_no_conflict, minor_bckt_sizes_512_avx); 
          minor_bckt_sizes_512_avx = _mm512_mask_i64gather_epi64(minor_bckt_sizes_512_avx, m_no_conflict, general_reg_1, 0, 1);
          general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, m_no_conflict, minor_bckt_sizes_512_avx); 

          m_minor_bckts_to_handle = _mm512_mask_cmplt_pd_mask(m_no_conflict, general_reg_2_double, threshold_avx);
          if(m_minor_bckts_to_handle)
          {
            general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
            general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
            general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, m_minor_bckts_to_handle, general_reg_2_double);

            _mm512_mask_i64scatter_epi64(0, m_minor_bckts_to_handle, general_reg_2, state[k].key, 1);

            //minor_bckt_sizes_512_avx = _mm256_mask_add_epi32(minor_bckt_sizes_512_avx, m_minor_bckts_to_handle, minor_bckt_sizes_512_avx, v512_one);
            //_mm512_mask_i64scatter_epi32(0, m_minor_bckts_to_handle, general_reg_1, minor_bckt_sizes_512_avx, 1);   
            minor_bckt_sizes_512_avx = _mm512_mask_add_epi64(minor_bckt_sizes_512_avx, m_minor_bckts_to_handle, minor_bckt_sizes_512_avx, v512_one);
            _mm512_mask_i64scatter_epi64(0, m_minor_bckts_to_handle, general_reg_1, minor_bckt_sizes_512_avx, 1);   
          }

          m_spill_bckts_to_handle = _mm512_kandn(m_minor_bckts_to_handle, m_no_conflict);
          if(m_spill_bckts_to_handle)
          {
              //auto curr_keys = (int64_t *) &state[k].key;
              auto curr_keys = (Tuple<KeyType, PayloadType> *) &state[k].key;          
              for(int j = 0; j < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++j)
              {
                  if (m_spill_bckts_to_handle & (1 << j)){ 
                      tmp_spill_bucket[spill_bucket_size] = curr_keys[j];
                      ++spill_bucket_size;
                  }
              }
          }
          state[k].m_have_key = _mm512_kandn(m_no_conflict, state[k].m_have_key);
        }
        num = _mm_popcnt_u32(state[k].m_have_key);

        if (num == LS_FOR_SORT_MERGE_VECTOR_SCALE || done >= LS_FOR_SORT_MERGE_SIMDStateSize)
        {
          //auto curr_keys = (int64_t *) &state[k].key;
          auto curr_keys = (Tuple<KeyType, PayloadType> *) &state[k].key;          
          auto pred_model_idx_list = (uint64_t *) &state[k].pred_model_idx;   

          for(int j = 0; j < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++j)
          {
              if (state[k].m_have_key & (1 << j)) 
              {

                  if (minor_bckt_sizes[pred_model_idx_list[j]] <
                      THRESHOLD) {  
                    minor_bckts[THRESHOLD * pred_model_idx_list[j] +
                                minor_bckt_sizes[pred_model_idx_list[j]]] = curr_keys[j];
                    ++minor_bckt_sizes[pred_model_idx_list[j]];
                  } else { 
                    tmp_spill_bucket[spill_bucket_size] = curr_keys[j];
                    ++spill_bucket_size;
                  }
              }                        
          }
          state[k].m_have_key = 0;
          state[k].stage = 1;
          --k;        
        } 
        else if (num == 0) 
        {
          state[k].stage = 1;
          --k;
        }
        else
        {
          if (done < LS_FOR_SORT_MERGE_SIMDStateSize)
          {
              num_temp = _mm_popcnt_u32(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key);
              if (num + num_temp < LS_FOR_SORT_MERGE_VECTOR_SCALE) {
                  // compress v
                  state[k].key = _mm512_maskz_compress_epi64(state[k].m_have_key, state[k].key);
                  state[k].pred_model_idx = _mm512_maskz_compress_epi64(state[k].m_have_key, state[k].pred_model_idx);
                  // expand v -> temp
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].key = _mm512_mask_expand_epi64(state[LS_FOR_SORT_MERGE_SIMDStateSize].key, _mm512_knot(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key), state[k].key);
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].pred_model_idx = _mm512_mask_expand_epi64(state[LS_FOR_SORT_MERGE_SIMDStateSize].pred_model_idx, _mm512_knot(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key), state[k].pred_model_idx);
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key = mask[num + num_temp];
                  state[k].m_have_key = 0;
                  state[k].stage = 1;
                  --k;
              } 
              else
              {
                  // expand temp -> v
                  state[k].key = _mm512_mask_expand_epi64(state[k].key, _mm512_knot(state[k].m_have_key), state[LS_FOR_SORT_MERGE_SIMDStateSize].key);
                  state[k].pred_model_idx = _mm512_mask_expand_epi64(state[k].pred_model_idx, _mm512_knot(state[k].m_have_key), state[LS_FOR_SORT_MERGE_SIMDStateSize].pred_model_idx);
                  // compress temp
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key = _mm512_kand(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key, _mm512_knot(mask[LS_FOR_SORT_MERGE_VECTOR_SCALE - num]));
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].key = _mm512_maskz_compress_epi64(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key, state[LS_FOR_SORT_MERGE_SIMDStateSize].key);
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].pred_model_idx = _mm512_maskz_compress_epi64(state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key, state[LS_FOR_SORT_MERGE_SIMDStateSize].pred_model_idx);
                  state[k].m_have_key = mask[LS_FOR_SORT_MERGE_VECTOR_SCALE];
                  state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key = (state[LS_FOR_SORT_MERGE_SIMDStateSize].m_have_key >> (LS_FOR_SORT_MERGE_VECTOR_SCALE - num));

                  general_reg_1_double = _mm512_cvtepi64_pd(state[k].pred_model_idx);
          
                  general_reg_2 = _mm512_cvtpd_epi64(
                                  _mm512_fmadd_pd(general_reg_1_double, v_64bit_elem_size_double, v_minor_bckt_sizes_addr_double));

                  //minor_bckt_sizes_512_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
                  //general_reg_2_double = _mm512_cvtepi32_pd(minor_bckt_sizes_512_avx); 
                  minor_bckt_sizes_512_avx = _mm512_i64gather_epi64(general_reg_2, 0, 1);
                  general_reg_2_double = _mm512_cvtepi64_pd(minor_bckt_sizes_512_avx); 

                  state[k].stage = 2;

                  m_minor_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, threshold_avx);
                  if(m_minor_bckts_to_handle)
                  {
                      general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
                      general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
                      general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

                  #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF
                      minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
                  #endif
                      minor_bckts_pos = (uint64_t *) &general_reg_1;
                      for (int i = 0; i < LS_FOR_SORT_MERGE_VECTOR_SCALE; ++i)
                      { 
                          if (m_minor_bckts_to_handle & (1 << i)) 
                          {
                  #ifdef LS_FOR_SORT_MERGE_PREFETCH_MINOR_BCKT_SIZES_OFF            
                              _mm_prefetch((char * )(minor_bckt_sizes_pos[i]), _MM_HINT_T0);
                  #endif
                              _mm_prefetch((char * )(minor_bckts_pos[i]), _MM_HINT_T0);
                          }
                      }
                  }
              }
          }
        }
      }
      break;
    }

    ++k;
  }

#else

  // Iterate over the elements in the current bucket in batch-mode
  for (unsigned int batch_idx = 0; batch_idx < num_batches; ++batch_idx) 
  {
    // Iterate over the elements in the batch and store their predicted
    // ranks
    for (unsigned int elm_idx = 0; elm_idx < BATCH_SZ; ++elm_idx) 
    {
      // Find the current element
      auto cur_elm = major_bckt[p][major_bckt_partition_offset + bckt_start_offset + elm_idx];
      
      // Predict the leaf-layer model
      batch_cache[elm_idx] = static_cast<int>(std::max(
          0.,
          std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt)));

      // Predict the CDF
      pred_cdf = slopes[batch_cache[elm_idx]] * cur_elm.key +
                  intercepts[batch_cache[elm_idx]];

      // Scale the predicted CDF to the number of minor buckets and cache it
      batch_cache[elm_idx] = static_cast<int>(std::max(
          0., std::min(NUM_MINOR_BCKT_PER_MAJOR_BCKT - 1.,
                        pred_cdf * TOT_NUM_MINOR_BCKTS - MINOR_BCKTS_OFFSET)));

      //if(thread_id == 0 && partition_id == 0 && (bckt_start_offset + elm_idx) <= 25)
      //if(thread_id == 0 && partition_id == 0 /*&& (bckt_start_offset + elm_idx) <= 500000*/)
      //{
        /*printf("thread_id %d cur_elm %ld batch_cache[elm_idx] %d pred_cdf %lf MINOR_BCKTS_OFFSET %ld NUM_MINOR_BCKT_PER_MAJOR_BCKT %ld batch_cache[elm_idx] %ld \n", 
                thread_id, cur_elm.key, static_cast<int>(std::max(0.,std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt))),
                pred_cdf, MINOR_BCKTS_OFFSET, NUM_MINOR_BCKT_PER_MAJOR_BCKT, batch_cache[elm_idx]);*/

        /*printf("cur_elm %ld root_slope %lf root_intrcpt %lf batch_cache[elm_idx] %d slopes[batch_cache[elm_idx]] %lf intercepts[batch_cache[elm_idx]] %lf pred_cdf %lf pred_cdf * TOT_NUM_MINOR_BCKTS %lf MINOR_BCKTS_OFFSET %ld pred_cdf * TOT_NUM_MINOR_BCKTS - MINOR_BCKTS_OFFSET %lf NUM_MINOR_BCKT_PER_MAJOR_BCKT %ld batch_cache[elm_idx] %ld \n", 
                cur_elm.key, root_slope, root_intrcpt, static_cast<int>(std::max(0.,std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt))), 
                slopes[static_cast<int>(std::max(0., std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt)))], 
                intercepts[static_cast<int>(std::max(0., std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt)))], 
                pred_cdf, pred_cdf * TOT_NUM_MINOR_BCKTS, MINOR_BCKTS_OFFSET, pred_cdf * TOT_NUM_MINOR_BCKTS - MINOR_BCKTS_OFFSET, NUM_MINOR_BCKT_PER_MAJOR_BCKT, batch_cache[elm_idx]);*/
      //}


    }

    // Iterate over the elements in the batch again, and place them in the
    // sub-buckets, or spill bucket
    for (unsigned int elm_idx = 0; elm_idx < BATCH_SZ; ++elm_idx) {
      // Find the current element
      auto cur_elm = major_bckt[p][major_bckt_partition_offset + bckt_start_offset + elm_idx];

 //if(/*thread_id ==  0 &&*/  partition_id == 0 && (bckt_start_offset + elm_idx) <= 1000)
 //     {
      /*  printf("thread_id %d cur_elm %ld MINOR_BCKTS_OFFSET %ld NUM_MINOR_BCKT_PER_MAJOR_BCKT %ld batch_cache[elm_idx] %ld \n",
                thread_id, cur_elm.key, MINOR_BCKTS_OFFSET, NUM_MINOR_BCKT_PER_MAJOR_BCKT, batch_cache[elm_idx]);*/
// }


      // Check if the element will cause a bucket overflow
      if (minor_bckt_sizes[batch_cache[elm_idx]] <
          THRESHOLD) {  // The predicted bucket has not reached
        // full capacity, so place the element in
        // the bucket
        minor_bckts[THRESHOLD * batch_cache[elm_idx] +
                    minor_bckt_sizes[batch_cache[elm_idx]]] = cur_elm;
        ++minor_bckt_sizes[batch_cache[elm_idx]];
      } else {  // Place the item in the spill bucket
        //spill_bucket->push_back(cur_elm);
        tmp_spill_bucket[spill_bucket_size] = cur_elm;
        ++spill_bucket_size;
      }
    }

    // Update the start offset
    bckt_start_offset += BATCH_SZ;
  }

  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -//
  // Repeat the above for the rest of the elements in the     //
  // current bucket in case its size wasn't divisible by      //
  // the BATCH_SZ                                             //
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
  unsigned int num_remaining_elm =
      major_bckt_size[p] - num_batches * BATCH_SZ;

  for (unsigned int elm_idx = 0; elm_idx < num_remaining_elm; ++elm_idx) {
    auto cur_elm = major_bckt[p][major_bckt_partition_offset + bckt_start_offset + elm_idx];
    batch_cache[elm_idx] = static_cast<int>(std::max(
        0., std::min(num_models - 1., root_slope * cur_elm.key + root_intrcpt)));
    pred_cdf = slopes[batch_cache[elm_idx]] * cur_elm.key +
                intercepts[batch_cache[elm_idx]];
    batch_cache[elm_idx] = static_cast<int>(std::max(
        0., std::min(NUM_MINOR_BCKT_PER_MAJOR_BCKT - 1.,
                      pred_cdf * TOT_NUM_MINOR_BCKTS - MINOR_BCKTS_OFFSET)));
  }

  for (unsigned elm_idx = 0; elm_idx < num_remaining_elm; ++elm_idx) {
    auto cur_elm = major_bckt[p][major_bckt_partition_offset + bckt_start_offset + elm_idx];
    if (minor_bckt_sizes[batch_cache[elm_idx]] < THRESHOLD) {
      minor_bckts[THRESHOLD * batch_cache[elm_idx] +
                  minor_bckt_sizes[batch_cache[elm_idx]]] = cur_elm;
      ++minor_bckt_sizes[batch_cache[elm_idx]];
    } else {
      //spill_bucket->push_back(cur_elm);
      tmp_spill_bucket[spill_bucket_size] = cur_elm;
      ++spill_bucket_size;
    }
  }

  //if(thread_id == 0 && partition_id == 0 /*&& major_bckt_idx <= 25*/)
  //{
    /*printf("minor bckt sizes \n");
    for(unsigned int i = 0; i < NUM_MINOR_BCKT_PER_MAJOR_BCKT; i++)
    {
      printf("minor bckt %d: %d ", i, minor_bckt_sizes[i]);
    }
    printf("\n");*/
  //}

#endif
}

//  if(thread_id == 0 && partition_id == 0)
//  {
    //printf("curr spill bucket is %d \n", spill_bucket->size());
    //printf("curr spill bucket is %d \n", spill_bucket_size);
//  }



  //----------------------------------------------------------//
  //                MODEL-BASED COUNTING SORT                 //
  //----------------------------------------------------------//
  num_tot_elms_in_bckts = 0;
  
  // Iterate over the minor buckets of the current bucket
  for (unsigned int bckt_idx = 0; bckt_idx < NUM_MINOR_BCKT_PER_MAJOR_BCKT; ++bckt_idx) 
  {
    if (minor_bckt_sizes[bckt_idx] > 0) 
    {
#ifdef USE_AVXSORT_FOR_SORTING_MINOR_BCKTS      
    Tuple<KeyType, PayloadType> * inputptr_min_bckt =  minor_bckts + bckt_idx * THRESHOLD;
    Tuple<KeyType, PayloadType> * outputptr_major_bckt = tmp_major_bckt + num_tot_elms_in_bckts;
    avxsort_tuples<KeyType, PayloadType>(&inputptr_min_bckt, &outputptr_major_bckt, minor_bckt_sizes[bckt_idx]);
    for(unsigned int k = 0; k < minor_bckt_sizes[bckt_idx]; k++){
      tmp_major_bckt[num_tot_elms_in_bckts + k].key = outputptr_major_bckt[k].key;
      tmp_major_bckt[num_tot_elms_in_bckts + k].payload = outputptr_major_bckt[k].payload;
    }

#else
      // Update the bucket start offset
      bckt_start_offset =
          1. * (MINOR_BCKTS_OFFSET + bckt_idx) *
          INPUT_SZ / TOT_NUM_MINOR_BCKTS;

      // Count array for the model-enhanced counting sort subroutine
      vector<unsigned int> cnt_hist(THRESHOLD, 0);

      /*
      * OPTIMIZATION
      * We check to see if the first and last element in the current bucket
      * used the same leaf model to obtain their CDF. If that is the case,
      * then we don't need to traverse the CDF model for every element in
      * this bucket, hence decreasing the inference complexity from
      * O(num_layer) to O(1).
      */

      int pred_model_first_elm = static_cast<int>(std::max(
          0., std::min(num_models - 1.,
                        root_slope * minor_bckts[bckt_idx * THRESHOLD].key +
                            root_intrcpt)));

      int pred_model_last_elm = static_cast<int>(std::max(
          0., std::min(num_models - 1.,
                    root_slope * minor_bckts[bckt_idx * THRESHOLD +
                                            minor_bckt_sizes[bckt_idx] - 1].key +
                        root_intrcpt)));

      if (pred_model_first_elm == pred_model_last_elm) 
      {  // Avoid CDF model traversal and predict the
         // CDF only using the leaf model
        // Iterate over the elements and place them into the minor buckets
        for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
              ++elm_idx) {
          // Find the current element
          auto cur_elm = minor_bckts[bckt_idx * THRESHOLD + elm_idx];

          // Predict the CDF
          pred_cdf = slopes[pred_model_first_elm] * cur_elm.key +
                      intercepts[pred_model_first_elm];

          // Scale the predicted CDF to the input size and cache it
          pred_idx_cache[elm_idx] = static_cast<int>(std::max(
              0., std::min(THRESHOLD - 1.,
                            (pred_cdf * INPUT_SZ) - bckt_start_offset)));

          // Update the counts
          ++cnt_hist[pred_idx_cache[elm_idx]];
        }
      } 
      else 
      {  // Fully traverse the CDF model again to predict the CDF of
         // the current element
        // Iterate over the elements and place them into the minor buckets
        for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
              ++elm_idx) {
          // Find the current element
          auto cur_elm = minor_bckts[bckt_idx * THRESHOLD + elm_idx];

          // Predict the model idx in the leaf layer
          auto model_idx_next_layer = static_cast<int>(
              std::max(0., std::min(num_models - 1.,
                                    root_slope * cur_elm.key + root_intrcpt)));
          // Predict the CDF
          pred_cdf = slopes[model_idx_next_layer] * cur_elm.key +
                      intercepts[model_idx_next_layer];

          // Scale the predicted CDF to the input size and cache it
          pred_idx_cache[elm_idx] = static_cast<unsigned int>(std::max(
              0., std::min(THRESHOLD - 1.,
                            (pred_cdf * INPUT_SZ) - bckt_start_offset)));

          // Update the counts
          ++cnt_hist[pred_idx_cache[elm_idx]];
        }
      }

      --cnt_hist[0];

      // Calculate the running totals
      for (unsigned int cnt_idx = 1; cnt_idx < THRESHOLD; ++cnt_idx) {
        cnt_hist[cnt_idx] += cnt_hist[cnt_idx - 1];
      }

      // Re-shuffle the elms based on the calculated cumulative counts
      for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
            ++elm_idx) {
        // Place the element in the predicted position in the array
        tmp_major_bckt[num_tot_elms_in_bckts + cnt_hist[pred_idx_cache[elm_idx]]] = 
            minor_bckts[bckt_idx * THRESHOLD + elm_idx];
        
        // Update counts
        --cnt_hist[pred_idx_cache[elm_idx]];
      }

      //----------------------------------------------------------//
      //                  TOUCH-UP & COMPACTION                   //
      //----------------------------------------------------------//

      // After the model-based bucketization process is done, switch to a
      // deterministic sort
      Tuple<KeyType, PayloadType> elm;
      int cmp_idx;

      // Perform Insertion Sort
      for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
            ++elm_idx) {
        cmp_idx = num_tot_elms_in_bckts + elm_idx - 1;
        elm = tmp_major_bckt[num_tot_elms_in_bckts + elm_idx];
        while (cmp_idx >= 0 && elm.key < tmp_major_bckt[cmp_idx].key) {
          tmp_major_bckt[cmp_idx + 1] = tmp_major_bckt[cmp_idx];
          --cmp_idx;
        }

        tmp_major_bckt[cmp_idx + 1] = elm;
      }
#endif


      num_tot_elms_in_bckts += minor_bckt_sizes[bckt_idx];

    } // end of iteration of each minor bucket
  }

  //std::cout << "learned_sort: spill_bucket size: "<< spill_bucket->size() << std::endl;
  //std::cout << "learned_sort: spill_bucket size: "<< spill_bucket_size << std::endl;


  //----------------------------------------------------------//
  //                 SORT THE SPILL BUCKET                    //
  //----------------------------------------------------------//

#ifdef USE_AVXSORT_AS_STD_SORT
  //uint32_t spill_bucket_size = spill_bucket->size();
  //int64_t sorted_spill_bucket_arr [spill_bucket_size];
  //int64_t * inputptr = (int64_t *)(&((*spill_bucket)[0])); 
  //int64_t * outputptr = (int64_t *)(sorted_spill_bucket_arr); 
  //avxsort_int64(&inputptr, &outputptr, spill_bucket_size);
  //vector<Tuple<KeyType, PayloadType>> sorted_spill_bucket((Tuple<KeyType, PayloadType>*)outputptr, ((Tuple<KeyType, PayloadType>*)outputptr) + spill_bucket_size);
  //spill_bucket = &sorted_spill_bucket;

//NOTE: Working code but with extra overhead for assigning tmp_outputptr back to sorted_spill_bucket
//  int64_t * inputptr =  (int64_t *)(tmp_spill_bucket);
//  int64_t * outputptr = (int64_t *)(sorted_spill_bucket);
//  avxsort_int64(&inputptr, &outputptr, spill_bucket_size);
//  Tuple<KeyType, PayloadType>* tmp_outputptr = (Tuple<KeyType, PayloadType>*) outputptr;
//  for(int k = 0; k < spill_bucket_size; k++){
//    sorted_spill_bucket[k].key = tmp_outputptr[k].key;
//    sorted_spill_bucket[k].payload = tmp_outputptr[k].payload;
//  } 
//  spill_bucket = sorted_spill_bucket;

  int64_t * inputptr =  (int64_t *)(tmp_spill_bucket);
  int64_t * outputptr = (int64_t *)(sorted_spill_bucket);
  avxsort_int64(&inputptr, &outputptr, spill_bucket_size);
  //Tuple<KeyType, PayloadType>* tmp_outputptr = (Tuple<KeyType, PayloadType>*) outputptr;
  spill_bucket = (Tuple<KeyType, PayloadType>*)outputptr;
  //sorted_spill_bucket = (Tuple<KeyType, PayloadType>*)outputptr;
  //spill_bucket = sorted_spill_bucket;

  //NOTE: Working code but with slower performance 
  //  avxsort_tuples<KeyType, PayloadType>(&tmp_spill_bucket, &sorted_spill_bucket, spill_bucket_size);
  //  spill_bucket = sorted_spill_bucket;
#else
  //std::sort((int64_t *)(&((*spill_bucket)[0])), (int64_t *)(&((*spill_bucket)[0])) + spill_bucket->size());
  ////std::sort(spill_bucket->begin(), spill_bucket->end());

  std::sort((int64_t *)(tmp_spill_bucket), (int64_t *)(tmp_spill_bucket) + spill_bucket_size /*- 1*/);
  spill_bucket = tmp_spill_bucket;
#endif

  //----------------------------------------------------------//
  //               PLACE BACK THE EXCEPTION VALUES            //
  //----------------------------------------------------------//
  vector<Tuple<KeyType, PayloadType>> linear_vals; vector<int64_t> linear_count;
  int64_t all_repeated_keys = 0;
  for(int p = 0; p < NUM_THREADS_FOR_EVALUATION; p++)
  {
    for (int64_t exc_elm_idx = 0; exc_elm_idx < total_repeated_keys[p]; ++exc_elm_idx) {
      linear_vals.push_back(
          repeated_keys_predicted_ranks[p][repeated_keys_offset + exc_elm_idx]);
      linear_count.push_back(repeated_key_counts[p][repeated_keys_offset + exc_elm_idx]);
    }
    all_repeated_keys += total_repeated_keys[p];
  }

  //----------------------------------------------------------//
  //               MERGE BACK INTO ORIGINAL ARRAY             //
  //----------------------------------------------------------//

  // Merge the spill bucket with the elements in the buckets
  #ifdef USE_AVXSORT_AS_STD_SORT 
    #ifdef USE_AVXMERGE_AS_STD_MERGE
      //avx_merge_int64((int64_t*) (tmp_major_bckt), (int64_t *)(&((*spill_bucket)[0])), ((int64_t *)sorted_output) + all_repeated_keys, num_tot_elms_in_bckts, spill_bucket_size);
      avx_merge_int64((int64_t*) (tmp_major_bckt), (int64_t *)(spill_bucket), ((int64_t *)sorted_output) + all_repeated_keys, num_tot_elms_in_bckts, spill_bucket_size);
    #else
        //std::merge((int64_t*)tmp_major_bckt, ((int64_t*)tmp_major_bckt) + num_tot_elms_in_bckts,
        //           (int64_t *)(&((*spill_bucket)[0])), (int64_t *)(&((*spill_bucket)[0])) + spill_bucket->size(),
        //           ((int64_t*)sorted_output) + all_repeated_keys);
        std::merge((int64_t*)tmp_major_bckt, ((int64_t*)tmp_major_bckt) + num_tot_elms_in_bckts,
                   (int64_t *)(spill_bucket), (int64_t *)(spill_bucket) + spill_bucket_size,
                   ((int64_t*)sorted_output) + all_repeated_keys);
    #endif
  #else
     std::merge((int64_t*)tmp_major_bckt, ((int64_t*)tmp_major_bckt) + num_tot_elms_in_bckts,
              (int64_t *)(spill_bucket), (int64_t *)(spill_bucket) + spill_bucket_size,
              ((int64_t*)sorted_output) + all_repeated_keys);
   //std::merge((int64_t*)tmp_major_bckt, ((int64_t*)tmp_major_bckt) + num_tot_elms_in_bckts,
   //           (int64_t *)(&((*spill_bucket)[0])), (int64_t *)(&((*spill_bucket)[0])) + spill_bucket->size(),
   //           ((int64_t*)sorted_output) + all_repeated_keys);
      //std::merge(tmp_major_bckt, tmp_major_bckt + num_tot_elms_in_bckts,
      //       spill_bucket->begin(), spill_bucket->end(),
      //       tmp_major_bckt + all_repeated_keys);
  #endif


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
  //            Start merging the exception values            //
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

  // The read index for the exceptions
  unsigned int exc_idx = 0;

  // The read index for the already-merged elements from the buckets and the
  // spill bucket
  unsigned int input_idx = all_repeated_keys;

  // The write index for the final merging of everything
  int ptr = 0;

  int64_t major_bckt_size_and_total_repeated_keys = all_major_bckt_size + all_repeated_keys;
  while (input_idx < major_bckt_size_and_total_repeated_keys && exc_idx < linear_vals.size()) {
    if (sorted_output[input_idx].key < linear_vals[exc_idx].key) {
      sorted_output[ptr] = sorted_output[input_idx];
      ptr++;
      input_idx++;
    } else {
      for (int i = 0; i < linear_count[exc_idx]; i++) {
        sorted_output[ptr + i] = linear_vals[exc_idx];
      }
      ptr += linear_count[exc_idx];
      exc_idx++;
    }
  }

  while (exc_idx < linear_vals.size()) {
    for (int i = 0; i < linear_count[exc_idx]; i++) {
      sorted_output[ptr + i] = linear_vals[exc_idx];
    }
    ptr += linear_count[exc_idx];
    exc_idx++;
  }

  while (input_idx < major_bckt_size_and_total_repeated_keys) {
    sorted_output[ptr] = sorted_output[input_idx];
    ptr++;
    input_idx++;
  }

  // The input array is now sorted  

}

#endif  // LEARNED_SORT_FOR_SORT_MERGE_H
