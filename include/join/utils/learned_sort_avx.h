#ifndef LEARNED_SORT_AVX_H
#define LEARNED_SORT_AVX_H

/**
 * Vectorized version of learned_sort (Working only for 64-bit int keys (i.e., elements)) 
 */

#include <immintrin.h> /* AVX intrinsics */

#include "utils/learned_sort.h"

#define IMV_AVX

#define IMV_AVX_MAJOR_BCKTS_UNIQUE_KEYS
#define PREFETCH_INPUT_FOR_MAJOR_BCKTS_UNIQUE_KEYS
#define PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS
#define PREFETCH_SLOPES_AND_INTERCEPTS_MAJOR_BCKTS_UNIQUE_KEYS

#define IMV_AVX_MINOR_BCKTS
#define PREFETCH_INPUT_FOR_MINOR_BCKTS
#define PREFETCH_MINOR_BCKT_SIZES_OFF
#define PREFETCH_SLOPES_AND_INTERCEPTS_MINOR_BCKTS

#define WORDSIZE 8
#define VECTOR_SCALE 8
#define MAX_VECTOR_SCALE 16
#define SIMDStateSize 5
#define PDIS 320

namespace learned_sort {

#ifdef IMV_AVX
typedef struct StateSIMD StateSIMD;
struct StateSIMD {
  __m512i key;
  __m512i pred_model_idx;
  __mmask8 m_have_key;
  char stage;
};
#endif

// Training function
template <class RandomIt>
RMI<typename iterator_traits<RandomIt>::value_type> train_avx(
    RandomIt, RandomIt,
    typename RMI<typename iterator_traits<RandomIt>::value_type>::Params &);

// Default comparison function [std::less()] and default hyperparameters
// Drop-in replacement for std::sort()
template <class RandomIt>
void sort_avx(RandomIt, RandomIt);

// Default comparison function [std::less()] and custom hyperparameters
template <class RandomIt>
void sort_avx(
    RandomIt, RandomIt,
    typename RMI<typename iterator_traits<RandomIt>::value_type>::Params &);
}  // namespace learned_sort

using namespace learned_sort;

/**
 * Vectorized version of train() method
 */
template <class RandomIt>
learned_sort::RMI<typename iterator_traits<RandomIt>::value_type> learned_sort::train_avx(
    RandomIt begin, RandomIt end,
    typename RMI<typename iterator_traits<RandomIt>::value_type>::Params &p) {
  // Determine the data type
  typedef typename iterator_traits<RandomIt>::value_type T;

  // Determine input size
  const unsigned int INPUT_SZ = std::distance(begin, end);

  // Validate parameters
  if (p.batch_sz >= INPUT_SZ) {
    p.batch_sz = RMI<T>::Params::DEFAULT_BATCH_SZ;
    cerr << "\33[93;1mWARNING\33[0m: Invalid batch size. Using default ("
         << RMI<T>::Params::DEFAULT_BATCH_SZ << ")." << endl;
  }

  if (p.fanout >= INPUT_SZ) {
    p.fanout = RMI<T>::Params::DEFAULT_FANOUT;
    cerr << "\33[93;1mWARNING\33[0m: Invalid fanout. Using default ("
         << RMI<T>::Params::DEFAULT_FANOUT << ")." << endl;
  }

  if (p.overallocation_ratio <= 0) {
    p.overallocation_ratio = 1;
    cerr << "\33[93;1mWARNING\33[0m: Invalid overallocation ratio. Using "
            "default ("
         << RMI<T>::Params::DEFAULT_OVERALLOCATION_RATIO << ")." << endl;
  }

  if (p.sampling_rate <= 0 or p.sampling_rate > 1) {
    p.sampling_rate = RMI<T>::Params::DEFAULT_SAMPLING_RATE;
    cerr << "\33[93;1mWARNING\33[0m: Invalid sampling rate. Using default ("
         << RMI<T>::Params::DEFAULT_SAMPLING_RATE << ")." << endl;
  }

  if (p.threshold <= 0 or p.threshold >= INPUT_SZ or
      p.threshold >= INPUT_SZ / p.fanout) {
    p.threshold = RMI<T>::Params::DEFAULT_THRESHOLD;
    cerr << "\33[93;1mWARNING\33[0m: Invalid threshold. Using default ("
         << RMI<T>::Params::DEFAULT_THRESHOLD << ")." << endl;
  }

  if (p.arch.size() > 2 or p.arch[0] != 1 or p.arch[1] <= 0) {
    p.arch = p.DEFAULT_ARCH;
    cerr << "\33[93;1mWARNING\33[0m: Invalid architecture. Using default {"
         << p.DEFAULT_ARCH[0] << ", " << p.DEFAULT_ARCH[1] << "}." << endl;
  }

  // Initialize the CDF model
  RMI<T> rmi(p);
  static const unsigned int NUM_LAYERS = p.arch.size();
  vector<vector<vector<training_point<T>>>> training_data(NUM_LAYERS);
  for (unsigned int layer_idx = 0; layer_idx < NUM_LAYERS; ++layer_idx) {
    training_data[layer_idx].resize(p.arch[layer_idx]);
  }

  //----------------------------------------------------------//
  //                           SAMPLE                         //
  //----------------------------------------------------------//

  // Determine sample size
  const unsigned int SAMPLE_SZ = std::min<unsigned int>(
      INPUT_SZ, std::max<unsigned int>(p.sampling_rate * INPUT_SZ,
                                       RMI<T>::Params::MIN_SORTING_SIZE));

  // Create a sample array
  rmi.training_sample.reserve(SAMPLE_SZ);

  // Start sampling
  unsigned int offset = static_cast<unsigned int>(1. * INPUT_SZ / SAMPLE_SZ);
  for (auto i = begin; i < end; i += offset) {
    // NOTE:  We don't directly assign SAMPLE_SZ to rmi.training_sample_sz
    //        to avoid issues with divisibility
    rmi.training_sample.push_back(*i);
  }

  // Sort the sample using the provided comparison function
  std::sort(rmi.training_sample.begin(), rmi.training_sample.end());

  // Stop early if the array is identical
  if (rmi.training_sample.front() == rmi.training_sample.back()) {
    return rmi;
  }

  //----------------------------------------------------------//
  //                     TRAIN THE MODELS                     //
  //----------------------------------------------------------//

  // Populate the training data for the root model
  for (unsigned int i = 0; i < SAMPLE_SZ; ++i) {
    training_data[0][0].push_back({rmi.training_sample[i], 1. * i / SAMPLE_SZ});
  }

  // Train the root model using linear interpolation
  auto *current_training_data = &training_data[0][0];
  typename RMI<T>::linear_model *current_model = &rmi.models[0][0];

  // Find the min and max values in the training set
  training_point<T> min = current_training_data->front();
  training_point<T> max = current_training_data->back();

  // Calculate the slope and intercept terms
  current_model->slope =
      1. / (max.x - min.x);  // Assuming min.y = 0 and max.y = 1
  current_model->intercept = -current_model->slope * min.x;

  // Extrapolate for the number of models in the next layer
  current_model->slope *= p.arch[1] - 1;
  current_model->intercept *= p.arch[1] - 1;

  // Populate the training data for the next layer
  for (const auto &d : *current_training_data) {
    // Predict the model index in next layer
    //unsigned int rank = current_model->slope * d.x + current_model->intercept;
    unsigned int rank = round(current_model->slope * d.x*1.00 + current_model->intercept);

    // Normalize the rank between 0 and the number of models in the next layer
    rank =
        std::max(static_cast<unsigned int>(0), std::min(p.arch[1] - 1, rank));

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
        training_point<T> tp;
        tp.x = 0;
        tp.y = 0;
        current_training_data->push_back(tp);
      } else {  // Case 2: The first model in this layer is not empty

        min = current_training_data->front();
        max = current_training_data->back();

        current_model->slope =
            (max.y)*1.00 / (max.x - min.x);  // Hallucinating as if min.y = 0
        current_model->intercept = min.y - current_model->slope * min.x;
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
            (1 - min.y) * 1.00 / (max.x - min.x);  // Hallucinating as if max.y = 1
        current_model->intercept = min.y - current_model->slope * min.x;
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
        training_point<T> tp;
        tp.x = training_data[1][model_idx - 1].back().x;
        tp.y = training_data[1][model_idx - 1].back().y;
        current_training_data->push_back(tp);
      } else {  // Case 6: The intermediate leaf model is not empty

        min = training_data[1][model_idx - 1].back();
        max = current_training_data->back();

        current_model->slope = (max.y - min.y) * 1.00 / (max.x - min.x);
        current_model->intercept = min.y - current_model->slope * min.x;
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

template <class RandomIt>
void _sort_trained_avx(RandomIt begin, RandomIt end,
                   learned_sort::RMI<typename iterator_traits<RandomIt>::value_type> &rmi) {
  // Determine the data type
  typedef typename iterator_traits<RandomIt>::value_type T;

  // Cache runtime parameters
  static const unsigned int BATCH_SZ = rmi.hp.batch_sz;
  static const double OA_RATIO = rmi.hp.overallocation_ratio;
  static const unsigned int FANOUT = rmi.hp.fanout;
  static const unsigned int THRESHOLD = rmi.hp.threshold;

  // Determine the input size
  const unsigned int INPUT_SZ = std::distance(begin, end);

  //----------------------------------------------------------//
  //                          INIT                            //
  //----------------------------------------------------------//

  size_t ind;

  // Constants for buckets
  const unsigned int MAJOR_BCKT_CAPACITY = INPUT_SZ / FANOUT;

  // Constants for repeated keys
  const unsigned int EXCEPTION_VEC_INIT_CAPACITY = FANOUT;
  constexpr unsigned int EXC_CNT_THRESHOLD = 60;
  const size_t TRAINING_SAMPLE_SZ = rmi.training_sample.size();

  // Initialize the spill bucket
  vector<T> spill_bucket;

  // Initialize the buckets
  vector<T> major_bckts(INPUT_SZ + 1);

  // Array to keep track of the major bucket sizes
  vector<unsigned int> major_bckt_sizes(FANOUT, 0);

  // Initialize the exception lists for handling repeated keys
  vector<T> repeated_keys;  // Stores the heavily repeated key values
  vector<vector<T>> repeated_keys_predicted_ranks(
      EXCEPTION_VEC_INIT_CAPACITY);  // Stores the predicted ranks of each
                                     // repeated
                                     // key
  vector<vector<T>> repeated_key_counts(
      EXCEPTION_VEC_INIT_CAPACITY);  // Stores the count of repeated keys
  unsigned int total_repeated_keys = 0;

  // Counts the nubmer of total elements that are in the buckets, hence
  // INPUT_SZ - spill_bucket.size() at the end of the recursive bucketization
  unsigned int num_tot_elms_in_bckts = 0;

  // Cache the model parameters
  auto root_slope = rmi.models[0][0].slope;
  auto root_intrcpt = rmi.models[0][0].intercept;
  unsigned int num_models = rmi.hp.arch[1];
  vector<double> slopes, intercepts;
  for (unsigned int i = 0; i < num_models; ++i) {
    slopes.push_back(rmi.models[1][i].slope);
    intercepts.push_back(rmi.models[1][i].intercept);
  }

#ifdef IMV_AVX
  int32_t k = 0, done = 0, num, num_temp;
  __attribute__((aligned(CACHE_LINE_SIZE))) __mmask8 mask[VECTOR_SCALE + 1], m_no_conflict, m_major_bckts_to_handle, m_spill_bckts_to_handle;

  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(INPUT_SZ * sizeof(T)),  
  v_base_offset, v_slopes_addr = _mm512_set1_epi64((uint64_t) (&slopes[0])), v_intercepts_addr = _mm512_set1_epi64((uint64_t) (&intercepts[0])),  
  v_all_ones = _mm512_set1_epi64(-1), general_reg_1, general_reg_2, v_pred_model_idx, v_conflict;
  
  //__m512i  v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 = _mm512_set1_epi64(0), v_one = _mm512_set1_epi64(1);

  __m512d v_zero512_double = _mm512_set1_pd(0.), fanout_avx = _mm512_set1_pd((double)FANOUT), fanout_minus_one_avx = _mm512_set1_pd((double)FANOUT - 1.), 
  num_models_minus_one_avx = _mm512_set1_pd((double)num_models - 1.), root_slope_avx = _mm512_set1_pd(root_slope), 
  root_intrcpt_avx = _mm512_set1_pd(root_intrcpt), v_64bit_elem_size_double = _mm512_set1_pd(8.), v_32bit_elem_size_double = _mm512_set1_pd(4.), 
  v_major_bckt_sizes_addr_double = _mm512_cvtepi64_pd(_mm512_set1_epi64((uint64_t) (&major_bckt_sizes[0]))), 
  v_major_bckts_addr_double = _mm512_cvtepi64_pd(_mm512_set1_epi64((uint64_t) (&major_bckts[0]))), major_bckt_capacity_avx_double = _mm512_set1_pd((double)MAJOR_BCKT_CAPACITY),
  general_reg_1_double, general_reg_2_double, intercepts_avx, slopes_avx; 

  //__m512d v_one_double = _mm512_set1_pd(1.);

  __m256i major_bckt_sizes_256_avx, v256_one = _mm256_set1_epi32(1);
 
  __attribute__((aligned(CACHE_LINE_SIZE)))     uint64_t cur_offset = 0, base_off[MAX_VECTOR_SCALE], *major_bckts_pos, *major_bckt_sizes_pos, *slopes_pos, *intercepts_pos;
  __attribute__((aligned(CACHE_LINE_SIZE)))      StateSIMD state[SIMDStateSize + 1];

  for (int i = 0; i <= VECTOR_SCALE; ++i) 
  {
      base_off[i] = i * sizeof(T);
      mask[i] = (1 << i) - 1;
  }
  v_base_offset = _mm512_load_epi64(base_off);
#endif

  //----------------------------------------------------------//
  //       DETECT REPEATED KEYS IN THE TRAINING SAMPLE        //
  //----------------------------------------------------------//

  // Count the occurrences of equal keys
  unsigned int cnt_rep_keys = 1;
  for (size_t i = 1; i < TRAINING_SAMPLE_SZ; i++) {
    if (rmi.training_sample[i] == rmi.training_sample[i - 1]) {
      ++cnt_rep_keys;
    } else {  // New values start here. Reset counter. Add value in the
      // exception_vals if above threshold
      if (cnt_rep_keys > EXC_CNT_THRESHOLD) {
        repeated_keys.push_back(rmi.training_sample[i - 1]);
      }
      cnt_rep_keys = 1;
    }
  }

  if (cnt_rep_keys > EXC_CNT_THRESHOLD) {  // Last batch of repeated keys
    repeated_keys.push_back(rmi.training_sample[TRAINING_SAMPLE_SZ - 1]);
  }

  //----------------------------------------------------------//
  //             SHUFFLE THE KEYS INTO BUCKETS                //
  //----------------------------------------------------------//

  // For each spike value, predict the bucket.
  // In repeated_keys_predicted_ranks[predicted_bucket_idx] save the value, in
  // repeated_keys_predicted_ranks[predicted_bucket_idx] save the counts
  int pred_model_idx = 0;
  double pred_cdf = 0.;

  for (size_t i = 0; i < repeated_keys.size(); ++i) {
    pred_model_idx = static_cast<int>(
        std::max(0., std::min(num_models - 1.,
                              root_slope * repeated_keys[i] + root_intrcpt)));
    pred_cdf =
        slopes[pred_model_idx] * repeated_keys[i] + intercepts[pred_model_idx];
    pred_model_idx = static_cast<int>(
        std::max(0., std::min(FANOUT - 1., pred_cdf * FANOUT)));

    repeated_keys_predicted_ranks[pred_model_idx].push_back(repeated_keys[i]);
    repeated_key_counts[pred_model_idx].push_back(0);
  }

  if (repeated_keys.size() ==
      0) {  // No significantly repeated keys in the sample
    pred_model_idx = 0;

#ifdef IMV_AVX_MAJOR_BCKTS_UNIQUE_KEYS  
    // init # of the state
    for (int i = 0; i <= SIMDStateSize; ++i) {
        state[i].stage = 1;
        state[i].m_have_key = 0;
        state[i].pred_model_idx = _mm512_set1_epi64(0);
        state[i].key = _mm512_set1_epi64(0);
    }

    for (uint64_t cur = 0; 1;) 
    {
        k = (k >= SIMDStateSize) ? 0 : k;
        if (cur >= INPUT_SZ) 
        {
            if (state[k].m_have_key == 0 && state[k].stage != 3) {
                ++done;
                state[k].stage = 3;
                ++k;
                continue;
            }
            if ((done >= SIMDStateSize)) {
                if (state[SIMDStateSize].m_have_key > 0) {
                    k = SIMDStateSize;
                    state[SIMDStateSize].stage = 2;
                } else {
                    break;
                }
            }
        }

        switch (state[k].stage) 
        {
            case 1: 
            {
            #ifdef PREFETCH_INPUT_FOR_MAJOR_BCKTS_UNIQUE_KEYS    
                _mm_prefetch((char *)(((void * )(&begin[0])) + cur_offset + PDIS), _MM_HINT_T0);
                _mm_prefetch((char *)(((void * )(&begin[0])) + cur_offset + PDIS + CACHE_LINE_SIZE), _MM_HINT_T0);
                _mm_prefetch((char *)(((void * )(&begin[0])) + cur_offset + PDIS + 2 * CACHE_LINE_SIZE), _MM_HINT_T0);
            #endif
                v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur_offset), v_base_offset);
                cur_offset = cur_offset + base_off[VECTOR_SCALE];
                state[k].m_have_key = _mm512_cmpgt_epi64_mask(v_base_offset_upper, v_offset);
                cur = cur + VECTOR_SCALE;

                if((int)(state[k].m_have_key) == 255)
                {
                    state[k].key = _mm512_i64gather_epi64(v_offset, ((void * )(&begin[0])), 1);
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
                #ifdef PREFETCH_SLOPES_AND_INTERCEPTS_MAJOR_BCKTS_UNIQUE_KEYS
                    general_reg_1 = _mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr);
                    general_reg_2 = _mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr);
                
                    state[k].stage = 0;

                    slopes_pos = (uint64_t *) &general_reg_1;
                    intercepts_pos = (uint64_t *) &general_reg_2;     
                    for (int i = 0; i < VECTOR_SCALE; ++i)
                    {   
                        _mm_prefetch((char * )(slopes_pos[i]), _MM_HINT_T0);
                        _mm_prefetch((char * )(intercepts_pos[i]), _MM_HINT_T0);
                    }
                #else
                    slopes_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr), 0, 1);
                    intercepts_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr), 0, 1);

                    //general_reg_1_double = _mm512_fmadd_pd(general_reg_2_double, slopes_avx, intercepts_avx);
                    //general_reg_1_double = _mm512_mul_pd(general_reg_1_double, fanout_avx);
                    //general_reg_1_double = _mm512_min_pd(general_reg_1_double, fanout_minus_one_avx);
                    //general_reg_1_double = _mm512_max_pd(general_reg_1_double, v_zero512_double);
                    //general_reg_1_double = _mm512_floor_pd(general_reg_1_double);
                    general_reg_1_double = _mm512_floor_pd(
                                                _mm512_max_pd(
                                                    _mm512_min_pd(
                                                        _mm512_mul_pd(
                                                            _mm512_fmadd_pd(general_reg_2_double, slopes_avx, intercepts_avx), fanout_avx), fanout_minus_one_avx), v_zero512_double));

                    state[k].pred_model_idx = _mm512_cvtpd_epi64(general_reg_1_double);

                    general_reg_2 = _mm512_cvtpd_epi64(
                                        _mm512_fmadd_pd(general_reg_1_double, v_32bit_elem_size_double, v_major_bckt_sizes_addr_double));
                    major_bckt_sizes_256_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
                    general_reg_2_double = _mm512_cvtepi32_pd(major_bckt_sizes_256_avx); 

                    state[k].stage = 2;

                    m_major_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, major_bckt_capacity_avx_double);
                    if(m_major_bckts_to_handle)
                    {
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_major_bckts_to_handle, major_bckt_capacity_avx_double, general_reg_2_double);
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_major_bckts_to_handle, v_64bit_elem_size_double, v_major_bckts_addr_double);
                        general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_major_bckts_to_handle, general_reg_2_double);

                        //_mm512_mask_prefetch_i64scatter_pd(0, m_major_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

                    #ifdef PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS
                        major_bckt_sizes_pos = (uint64_t *) &general_reg_2;
                    #endif            
                        major_bckts_pos = (uint64_t *) &general_reg_1;     
                        for (int i = 0; i < VECTOR_SCALE; ++i)
                        { 
                            if (m_major_bckts_to_handle & (1 << i)) 
                            {
                    #ifdef PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS            
                                _mm_prefetch((char * )(major_bckt_sizes_pos[i]), _MM_HINT_T0);
                    #endif
                                _mm_prefetch((char * )(major_bckts_pos[i]), _MM_HINT_T0);
                            }
                        }
                    }
                #endif
                }
                else
                {
                    state[k].key = _mm512_mask_i64gather_epi64(state[k].key, state[k].m_have_key, v_offset, ((void * )(&begin[0])), 1);
                    general_reg_2_double = _mm512_mask_cvtepi64_pd(general_reg_2_double, state[k].m_have_key, state[k].key);
                    general_reg_1_double = _mm512_mask_fmadd_pd(general_reg_2_double, state[k].m_have_key, root_slope_avx, root_intrcpt_avx);
                    general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, num_models_minus_one_avx);
                    general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
                    general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);
                    general_reg_1_double = _mm512_mask_mul_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_64bit_elem_size_double);

                    state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);
              
                #ifdef PREFETCH_SLOPES_AND_INTERCEPTS_MAJOR_BCKTS_UNIQUE_KEYS
                    general_reg_1 = _mm512_mask_add_epi64(general_reg_1, state[k].m_have_key, state[k].pred_model_idx, v_slopes_addr);
                    general_reg_2 = _mm512_mask_add_epi64(general_reg_2, state[k].m_have_key, state[k].pred_model_idx, v_intercepts_addr);
                
                    state[k].stage = 0;

                    slopes_pos = (uint64_t *) &general_reg_1;
                    intercepts_pos = (uint64_t *) &general_reg_2;     
                    for (int i = 0; i < VECTOR_SCALE; ++i)
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
                    general_reg_1_double = _mm512_mask_mul_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, fanout_avx);
                    general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, fanout_minus_one_avx);
                    general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
                    general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);

                    state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);

                    general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, state[k].m_have_key, 
                                        _mm512_mask_fmadd_pd(general_reg_1_double, state[k].m_have_key, v_32bit_elem_size_double, v_major_bckt_sizes_addr_double));
                    major_bckt_sizes_256_avx = _mm512_mask_i64gather_epi32(major_bckt_sizes_256_avx, state[k].m_have_key, general_reg_2, 0, 1);
                    general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, state[k].m_have_key, major_bckt_sizes_256_avx); 
                    
                    state[k].stage = 2;

                    m_major_bckts_to_handle = _mm512_mask_cmplt_pd_mask(state[k].m_have_key, general_reg_2_double, major_bckt_capacity_avx_double);
                    if(m_major_bckts_to_handle)
                    {
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_major_bckts_to_handle, major_bckt_capacity_avx_double, general_reg_2_double);
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_major_bckts_to_handle, v_64bit_elem_size_double, v_major_bckts_addr_double);
                        general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_major_bckts_to_handle, general_reg_2_double);

                        //_mm512_mask_prefetch_i64scatter_pd(0, m_major_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

                    #ifdef PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS
                        major_bckt_sizes_pos = (uint64_t *) &general_reg_2;        
                    #endif    
                        major_bckts_pos = (uint64_t *) &general_reg_1;                
                        for (int i = 0; i < VECTOR_SCALE; ++i)
                        { 
                            if (m_major_bckts_to_handle & (1 << i)) 
                            {
                    #ifdef PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS
                                _mm_prefetch((char * )(major_bckt_sizes_pos[i]), _MM_HINT_T0);
                    #endif
                                _mm_prefetch((char * )(major_bckts_pos[i]), _MM_HINT_T0);
                            }
                        }
                    }
                #endif
                }
            }
            break;
        #ifdef PREFETCH_SLOPES_AND_INTERCEPTS_MAJOR_BCKTS_UNIQUE_KEYS
            case 0:
            {
                if((int)(state[k].m_have_key) == 255)
                {
                    general_reg_2_double = _mm512_cvtepi64_pd(state[k].key);

                    slopes_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr), 0, 1);
                    intercepts_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr), 0, 1);

                    //general_reg_1_double = _mm512_fmadd_pd(general_reg_2_double, slopes_avx, intercepts_avx);
                    //general_reg_1_double = _mm512_mul_pd(general_reg_1_double, fanout_avx);
                    //general_reg_1_double = _mm512_min_pd(general_reg_1_double, fanout_minus_one_avx);
                    //general_reg_1_double = _mm512_max_pd(general_reg_1_double, v_zero512_double);
                    //general_reg_1_double = _mm512_floor_pd(general_reg_1_double);
                    general_reg_1_double = _mm512_floor_pd(
                                                _mm512_max_pd(
                                                    _mm512_min_pd(
                                                        _mm512_mul_pd(
                                                            _mm512_fmadd_pd(general_reg_2_double, slopes_avx, intercepts_avx), fanout_avx), fanout_minus_one_avx), v_zero512_double));

                    state[k].pred_model_idx = _mm512_cvtpd_epi64(general_reg_1_double);

                    general_reg_2 = _mm512_cvtpd_epi64(
                                        _mm512_fmadd_pd(general_reg_1_double, v_32bit_elem_size_double, v_major_bckt_sizes_addr_double));
                    major_bckt_sizes_256_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
                    general_reg_2_double = _mm512_cvtepi32_pd(major_bckt_sizes_256_avx); 

                    state[k].stage = 2;

                    m_major_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, major_bckt_capacity_avx_double);
                    if(m_major_bckts_to_handle)
                    {
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_major_bckts_to_handle, major_bckt_capacity_avx_double, general_reg_2_double);
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_major_bckts_to_handle, v_64bit_elem_size_double, v_major_bckts_addr_double);
                        general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_major_bckts_to_handle, general_reg_2_double);

                        //_mm512_mask_prefetch_i64scatter_pd(0, m_major_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

                    #ifdef PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS
                        major_bckt_sizes_pos = (uint64_t *) &general_reg_2;
                    #endif            
                        major_bckts_pos = (uint64_t *) &general_reg_1;     
                        for (int i = 0; i < VECTOR_SCALE; ++i)
                        { 
                            if (m_major_bckts_to_handle & (1 << i)) 
                            {
                    #ifdef PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS            
                                _mm_prefetch((char * )(major_bckt_sizes_pos[i]), _MM_HINT_T0);
                    #endif
                                _mm_prefetch((char * )(major_bckts_pos[i]), _MM_HINT_T0);
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
                    general_reg_1_double = _mm512_mask_mul_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, fanout_avx);
                    general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, fanout_minus_one_avx);
                    general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
                    general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);

                    state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);

                    general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, state[k].m_have_key, 
                                        _mm512_mask_fmadd_pd(general_reg_1_double, state[k].m_have_key, v_32bit_elem_size_double, v_major_bckt_sizes_addr_double));
                    major_bckt_sizes_256_avx = _mm512_mask_i64gather_epi32(major_bckt_sizes_256_avx, state[k].m_have_key, general_reg_2, 0, 1);
                    general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, state[k].m_have_key, major_bckt_sizes_256_avx); 
                    
                    state[k].stage = 2;

                    m_major_bckts_to_handle = _mm512_mask_cmplt_pd_mask(state[k].m_have_key, general_reg_2_double, major_bckt_capacity_avx_double);
                    if(m_major_bckts_to_handle)
                    {
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_major_bckts_to_handle, major_bckt_capacity_avx_double, general_reg_2_double);
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_major_bckts_to_handle, v_64bit_elem_size_double, v_major_bckts_addr_double);
                        general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_major_bckts_to_handle, general_reg_2_double);

                        //_mm512_mask_prefetch_i64scatter_pd(0, m_major_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

                    #ifdef PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS
                        major_bckt_sizes_pos = (uint64_t *) &general_reg_2;        
                    #endif    
                        major_bckts_pos = (uint64_t *) &general_reg_1;                
                        for (int i = 0; i < VECTOR_SCALE; ++i)
                        { 
                            if (m_major_bckts_to_handle & (1 << i)) 
                            {
                    #ifdef PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS
                                _mm_prefetch((char * )(major_bckt_sizes_pos[i]), _MM_HINT_T0);
                    #endif
                                _mm_prefetch((char * )(major_bckts_pos[i]), _MM_HINT_T0);
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

                if (m_no_conflict) {
                    general_reg_1_double = _mm512_mask_cvtepi64_pd(general_reg_1_double, m_no_conflict, state[k].pred_model_idx);

                    general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_no_conflict, 
                                        _mm512_mask_fmadd_pd(general_reg_1_double, m_no_conflict, v_32bit_elem_size_double, v_major_bckt_sizes_addr_double));  
                    major_bckt_sizes_256_avx = _mm512_mask_i64gather_epi32(major_bckt_sizes_256_avx, m_no_conflict, general_reg_1, 0, 1);
                    general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, m_no_conflict, major_bckt_sizes_256_avx); 

                    m_major_bckts_to_handle = _mm512_mask_cmplt_pd_mask(m_no_conflict, general_reg_2_double, major_bckt_capacity_avx_double);
                    if(m_major_bckts_to_handle)
                    {
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_major_bckts_to_handle, major_bckt_capacity_avx_double, general_reg_2_double);
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_major_bckts_to_handle, v_64bit_elem_size_double, v_major_bckts_addr_double);
                        general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, m_major_bckts_to_handle, general_reg_2_double);

                        _mm512_mask_i64scatter_epi64(0, m_major_bckts_to_handle, general_reg_2, state[k].key, 1);

                        major_bckt_sizes_256_avx = _mm256_mask_add_epi32(major_bckt_sizes_256_avx, m_major_bckts_to_handle, major_bckt_sizes_256_avx, v256_one);
                        _mm512_mask_i64scatter_epi32(0, m_major_bckts_to_handle, general_reg_1, major_bckt_sizes_256_avx, 1);   
                    }
                    m_spill_bckts_to_handle = _mm512_kandn(m_major_bckts_to_handle, m_no_conflict);
                    if(m_spill_bckts_to_handle)
                    {
                        auto curr_keys = (int64_t *) &state[k].key;          
                        for(int j = 0; j < VECTOR_SCALE; ++j)
                        {
                            if (m_spill_bckts_to_handle & (1 << j)){ 
                                spill_bucket.push_back(curr_keys[j]);
                            }
                        }
                    }

                    state[k].m_have_key = _mm512_kandn(m_no_conflict, state[k].m_have_key);
                }
                num = _mm_popcnt_u32(state[k].m_have_key);

                if (num == VECTOR_SCALE || done >= SIMDStateSize) 
                {
                    auto curr_keys = (int64_t *) &state[k].key;   
                    auto pred_model_idx_list = (uint64_t *) &state[k].pred_model_idx;

                    for(int j = 0; j < VECTOR_SCALE; ++j)
                    {
                        if (state[k].m_have_key & (1 << j)) 
                        {
                            if (major_bckt_sizes[pred_model_idx_list[j]] <
                                MAJOR_BCKT_CAPACITY) {  // The predicted bucket is not full
                                major_bckts[MAJOR_BCKT_CAPACITY * pred_model_idx_list[j] +
                                            major_bckt_sizes[pred_model_idx_list[j]]] = curr_keys[j];

                                // Update the bucket size
                                ++major_bckt_sizes[pred_model_idx_list[j]];

                            } else {  // The predicted bucket is full, place in the spill bucket
                                spill_bucket.push_back(curr_keys[j]);
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
                    if (done < SIMDStateSize)
                    {
                        num_temp = _mm_popcnt_u32(state[SIMDStateSize].m_have_key);
                        if (num + num_temp < VECTOR_SCALE) {
                            // compress v
                            state[k].key = _mm512_maskz_compress_epi64(state[k].m_have_key, state[k].key);
                            state[k].pred_model_idx = _mm512_maskz_compress_epi64(state[k].m_have_key, state[k].pred_model_idx);
                            // expand v -> temp
                            state[SIMDStateSize].key = _mm512_mask_expand_epi64(state[SIMDStateSize].key, _mm512_knot(state[SIMDStateSize].m_have_key), state[k].key);
                            state[SIMDStateSize].pred_model_idx = _mm512_mask_expand_epi64(state[SIMDStateSize].pred_model_idx, _mm512_knot(state[SIMDStateSize].m_have_key), state[k].pred_model_idx);
                            state[SIMDStateSize].m_have_key = mask[num + num_temp];
                            state[k].m_have_key = 0;
                            state[k].stage = 1;
                            --k;
                        } else {
                            // expand temp -> v
                            state[k].key = _mm512_mask_expand_epi64(state[k].key, _mm512_knot(state[k].m_have_key), state[SIMDStateSize].key);
                            state[k].pred_model_idx = _mm512_mask_expand_epi64(state[k].pred_model_idx, _mm512_knot(state[k].m_have_key), state[SIMDStateSize].pred_model_idx);
                            // compress temp
                            state[SIMDStateSize].m_have_key = _mm512_kand(state[SIMDStateSize].m_have_key, _mm512_knot(mask[VECTOR_SCALE - num]));
                            state[SIMDStateSize].key = _mm512_maskz_compress_epi64(state[SIMDStateSize].m_have_key, state[SIMDStateSize].key);
                            state[SIMDStateSize].pred_model_idx = _mm512_maskz_compress_epi64(state[SIMDStateSize].m_have_key, state[SIMDStateSize].pred_model_idx);
                            state[k].m_have_key = mask[VECTOR_SCALE];
                            state[SIMDStateSize].m_have_key = (state[SIMDStateSize].m_have_key >> (VECTOR_SCALE - num));

                            general_reg_1_double = _mm512_cvtepi64_pd(state[k].pred_model_idx);
                            
                            general_reg_2 = _mm512_cvtpd_epi64(
                                                _mm512_fmadd_pd(general_reg_1_double, v_32bit_elem_size_double, v_major_bckt_sizes_addr_double));
                            major_bckt_sizes_256_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
                            general_reg_2_double = _mm512_cvtepi32_pd(major_bckt_sizes_256_avx); 

                            state[k].stage = 2;

                            m_major_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, major_bckt_capacity_avx_double);
                            if(m_major_bckts_to_handle)
                            {
                                general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_major_bckts_to_handle, major_bckt_capacity_avx_double, general_reg_2_double);
                                general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_major_bckts_to_handle, v_64bit_elem_size_double, v_major_bckts_addr_double);
                                general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_major_bckts_to_handle, general_reg_2_double);

                            #ifdef PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS
                                major_bckt_sizes_pos = (uint64_t *) &general_reg_2; 
                            #endif           
                                major_bckts_pos = (uint64_t *) &general_reg_1;                
                                for (int i = 0; i < VECTOR_SCALE; ++i)
                                { 
                                    if (m_major_bckts_to_handle & (1 << i)) 
                                    {
                            #ifdef PREFETCH_MAJOR_BCKT_SIZES_OFF_UNIQUE_KEYS
                                        _mm_prefetch((char * )(major_bckt_sizes_pos[i]), _MM_HINT_T0);
                            #endif
                                        _mm_prefetch((char * )(major_bckts_pos[i]), _MM_HINT_T0);
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
    //std::cout << "learned_sort_avx: spill_bucket size: "<< spill_bucket.size() << std::endl;
#else
    // Process each key in order
    for (auto cur_key = begin; cur_key < end; ++cur_key) {
      // Predict the model idx in the leaf layer
      pred_model_idx = static_cast<int>(std::max(
          0.,
          std::min(num_models - 1., root_slope * cur_key[0] + root_intrcpt)));

      // Predict the CDF
      pred_cdf =
          slopes[pred_model_idx] * cur_key[0] + intercepts[pred_model_idx];

      // Scale the CDF to the number of buckets
      pred_model_idx = static_cast<int>(
          std::max(0., std::min(FANOUT - 1., pred_cdf * FANOUT)));

      if (major_bckt_sizes[pred_model_idx] <
          MAJOR_BCKT_CAPACITY) {  // The predicted bucket is not full
        major_bckts[MAJOR_BCKT_CAPACITY * pred_model_idx +
                    major_bckt_sizes[pred_model_idx]] = cur_key[0];

        // Update the bucket size
        ++major_bckt_sizes[pred_model_idx];
      } else {  // The predicted bucket is full, place in the spill bucket
        spill_bucket.push_back(cur_key[0]);
      }
    } 
#endif
  } else {  // There are many repeated keys in the sample

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
                         root_slope * cur_key[elm_idx] + root_intrcpt)));

        // Predict the CDF
        pred_cdf = slopes[pred_idx_in_batch_exc[elm_idx]] * cur_key[elm_idx] +
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
             j < repeated_keys_predicted_ranks[pred_idx_in_batch_exc[elm_idx]]
                     .size();
             ++j) {
          // If key in exception list, then flag it and update the counts that
          // will be used later
          if (repeated_keys_predicted_ranks[pred_idx_in_batch_exc[elm_idx]]
                                           [j] == cur_key[elm_idx]) {
            ++repeated_key_counts[pred_idx_in_batch_exc[elm_idx]]
                                 [j];  // Increment count of exception value
            exc_found = true;
            ++total_repeated_keys;
            break;
          }
        }

        if (!exc_found)  // If no exception value was found in the batch,
                         // then proceed to putting them in the predicted
                         // buckets
        {
          // Check if the element will cause a bucket overflow
          if (major_bckt_sizes[pred_idx_in_batch_exc[elm_idx]] <
              MAJOR_BCKT_CAPACITY) {  // The predicted bucket has not reached
            // full capacity, so place the element
            // in the bucket
            major_bckts[MAJOR_BCKT_CAPACITY * pred_idx_in_batch_exc[elm_idx] +
                        major_bckt_sizes[pred_idx_in_batch_exc[elm_idx]]] =
                cur_key[elm_idx];
            ++major_bckt_sizes[pred_idx_in_batch_exc[elm_idx]];
          } else {  // Place the item in the spill bucket
            spill_bucket.push_back(cur_key[elm_idx]);
          }
        }
      }
    }
  }

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
  //               Second round of shuffling                  //
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

  unsigned int NUM_MINOR_BCKT_PER_MAJOR_BCKT = std::max(
      1u, static_cast<unsigned>(MAJOR_BCKT_CAPACITY * OA_RATIO / THRESHOLD));
  const unsigned int TOT_NUM_MINOR_BCKTS = NUM_MINOR_BCKT_PER_MAJOR_BCKT * FANOUT; 

  //const unsigned int ARR_SZ_MINOR_BCKTS =
  //    static_cast<int>(MAJOR_BCKT_CAPACITY * OA_RATIO) + 1;
  //const unsigned int TOT_NUM_MINOR_BCKTS =
  //    std::ceil(static_cast<int>(OA_RATIO * INPUT_SZ) / THRESHOLD);

  //vector<T> minor_bckts(ARR_SZ_MINOR_BCKTS);
  vector<T> minor_bckts(NUM_MINOR_BCKT_PER_MAJOR_BCKT * THRESHOLD);

  // Stores the index where the current bucket will start
  int bckt_start_offset = 0;

  // Stores the predicted CDF values for the elements in the current bucket
  unsigned int pred_idx_cache[THRESHOLD];

#ifndef IMV_AVX_MINOR_BCKTS
  // Caches the predicted bucket indices for each element in the batch
  vector<unsigned int> batch_cache(BATCH_SZ, 0);
#endif

  // Array to keep track of sizes for the minor buckets in the current
  // bucket
  vector<unsigned int> minor_bckt_sizes(NUM_MINOR_BCKT_PER_MAJOR_BCKT, 0);

  #ifdef IMV_AVX
    __m512d num_minor_bckt_per_major_bckt_avx = _mm512_set1_pd((double)NUM_MINOR_BCKT_PER_MAJOR_BCKT), num_minor_bckt_per_major_bckt_minus_one_avx = _mm512_set1_pd((double)NUM_MINOR_BCKT_PER_MAJOR_BCKT - 1.), 
    tot_num_minor_bckts_avx = _mm512_set1_pd((double)TOT_NUM_MINOR_BCKTS), threshold_avx = _mm512_set1_pd((double)THRESHOLD), major_bckt_idx_avx,
    v_minor_bckt_sizes_addr_double = _mm512_cvtepi64_pd(_mm512_set1_epi64((uint64_t) (&minor_bckt_sizes[0]))),
    v_minor_bckts_addr_double = _mm512_cvtepi64_pd(_mm512_set1_epi64((uint64_t) (&minor_bckts[0])));

    __m256i minor_bckt_sizes_256_avx;

    uint64_t curr_major_bckt_size, *minor_bckt_sizes_pos, *minor_bckts_pos;
    void * curr_major_bckt_off;
    __attribute__((aligned(CACHE_LINE_SIZE))) __mmask8 m_minor_bckts_to_handle;

  #endif

  // Iterate over each major bucket
  for (unsigned int major_bckt_idx = 0; major_bckt_idx < FANOUT;
       ++major_bckt_idx) {
    // Update the bucket start offset
    bckt_start_offset = major_bckt_idx * MAJOR_BCKT_CAPACITY;

    // Reset minor_bckt_sizes to all zeroes for the current major bucket
    fill(minor_bckt_sizes.begin(), minor_bckt_sizes.end(), 0);

    // Find out the number of batches for this bucket
    unsigned int num_batches = major_bckt_sizes[major_bckt_idx] / BATCH_SZ;

  #ifdef IMV_AVX_MINOR_BCKTS
    curr_major_bckt_size = major_bckt_sizes[major_bckt_idx]; 
    curr_major_bckt_off = ((void * )(&major_bckts[0])) + bckt_start_offset * sizeof(T);

    major_bckt_idx_avx = _mm512_set1_pd((double)major_bckt_idx);

    k = 0, done = 0, cur_offset = 0;
    v_base_offset_upper = _mm512_set1_epi64(curr_major_bckt_size * sizeof(T));

    // init # of the state
    for (int i = 0; i <= SIMDStateSize; ++i) {
        state[i].stage = 1;
        state[i].m_have_key = 0;
    }

    for (uint64_t cur = 0; 1;) 
    {
        k = (k >= SIMDStateSize) ? 0 : k;
        if (cur >= curr_major_bckt_size) 
        {
            if (state[k].m_have_key == 0 && state[k].stage != 3) {
                ++done;
                state[k].stage = 3;
                ++k;
                continue;
            }
            if ((done >= SIMDStateSize)) {
                if (state[SIMDStateSize].m_have_key > 0) {
                    k = SIMDStateSize;
                    state[SIMDStateSize].stage = 2;
                } else {
                    break;
                }
            }
        }

        switch (state[k].stage) 
        {
            case 1: 
            {
            #ifdef PREFETCH_INPUT_FOR_MINOR_BCKTS
                _mm_prefetch((char *)(curr_major_bckt_off + cur_offset + PDIS), _MM_HINT_T0);
                _mm_prefetch((char *)(curr_major_bckt_off + cur_offset + PDIS + CACHE_LINE_SIZE), _MM_HINT_T0);
                _mm_prefetch((char *)(curr_major_bckt_off + cur_offset + PDIS + 2 * CACHE_LINE_SIZE), _MM_HINT_T0);
            #endif
                v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur_offset), v_base_offset);
                cur_offset = cur_offset + base_off[VECTOR_SCALE];
                state[k].m_have_key = _mm512_cmpgt_epi64_mask(v_base_offset_upper, v_offset);
                cur = cur + VECTOR_SCALE;

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
                #ifdef PREFETCH_SLOPES_AND_INTERCEPTS_MINOR_BCKTS                    
                    general_reg_1 = _mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr);
                    general_reg_2 = _mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr);
                
                    state[k].stage = 0;

                    slopes_pos = (uint64_t *) &general_reg_1;
                    intercepts_pos = (uint64_t *) &general_reg_2;     
                    for (int i = 0; i < VECTOR_SCALE; ++i)
                    {   
                        _mm_prefetch((char * )(slopes_pos[i]), _MM_HINT_T0);
                        _mm_prefetch((char * )(intercepts_pos[i]), _MM_HINT_T0);
                    }
                #else
                    slopes_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr), 0, 1);
                    intercepts_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr), 0, 1);
  
                    general_reg_1_double = _mm512_fmadd_pd(general_reg_2_double, slopes_avx, intercepts_avx);
                    general_reg_2_double = _mm512_mul_pd(major_bckt_idx_avx, num_minor_bckt_per_major_bckt_avx);
                    //general_reg_1_double = _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, general_reg_2_double); 
                    //general_reg_1_double = _mm512_min_pd(general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
                    //general_reg_1_double = _mm512_max_pd(general_reg_1_double, v_zero512_double);
                    //general_reg_1_double = _mm512_floor_pd(general_reg_1_double);

                    general_reg_1_double = _mm512_floor_pd(
                                                _mm512_max_pd(
                                                    _mm512_min_pd(
                                                        _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, general_reg_2_double), num_minor_bckt_per_major_bckt_minus_one_avx), v_zero512_double));

                    state[k].pred_model_idx = _mm512_cvtpd_epi64(general_reg_1_double);

                    general_reg_2 = _mm512_cvtpd_epi64(
                                        _mm512_fmadd_pd(general_reg_1_double, v_32bit_elem_size_double, v_minor_bckt_sizes_addr_double));

                    minor_bckt_sizes_256_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
                    general_reg_2_double = _mm512_cvtepi32_pd(minor_bckt_sizes_256_avx); 

                    state[k].stage = 2;

                    m_minor_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, threshold_avx);
                    if(m_minor_bckts_to_handle)
                    {
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
                        general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

                        //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

                    #ifdef PREFETCH_MINOR_BCKT_SIZES_OFF
                        minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
                    #endif
                        minor_bckts_pos = (uint64_t *) &general_reg_1;
                        for (int i = 0; i < VECTOR_SCALE; ++i)
                        { 
                            if (m_minor_bckts_to_handle & (1 << i)) 
                            {
                    #ifdef PREFETCH_MINOR_BCKT_SIZES_OFF            
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
                #ifdef PREFETCH_SLOPES_AND_INTERCEPTS_MINOR_BCKTS                    
                    general_reg_1 = _mm512_mask_add_epi64(general_reg_1, state[k].m_have_key, state[k].pred_model_idx, v_slopes_addr);
                    general_reg_2 = _mm512_mask_add_epi64(general_reg_2, state[k].m_have_key, state[k].pred_model_idx, v_intercepts_addr);
                
                    state[k].stage = 0;

                    slopes_pos = (uint64_t *) &general_reg_1;
                    intercepts_pos = (uint64_t *) &general_reg_2;     
                    for (int i = 0; i < VECTOR_SCALE; ++i)
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
                    general_reg_2_double = _mm512_mask_mul_pd(general_reg_2_double, state[k].m_have_key, major_bckt_idx_avx, num_minor_bckt_per_major_bckt_avx);
                    general_reg_1_double = _mm512_mask_fmsub_pd(general_reg_1_double, state[k].m_have_key, tot_num_minor_bckts_avx, general_reg_2_double); 
                    general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
                    general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
                    general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);

                    state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);

                    general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, state[k].m_have_key, 
                                        _mm512_mask_fmadd_pd(general_reg_1_double, state[k].m_have_key, v_32bit_elem_size_double, v_minor_bckt_sizes_addr_double));

                    minor_bckt_sizes_256_avx = _mm512_mask_i64gather_epi32(minor_bckt_sizes_256_avx, state[k].m_have_key, general_reg_2, 0, 1);
                    general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, state[k].m_have_key, minor_bckt_sizes_256_avx); 

                    state[k].stage = 2;

                    m_minor_bckts_to_handle = _mm512_mask_cmplt_pd_mask(state[k].m_have_key, general_reg_2_double, threshold_avx);
                    if(m_minor_bckts_to_handle)
                    {
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
                        general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

                        //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

                    #ifdef PREFETCH_MINOR_BCKT_SIZES_OFF
                        minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
                    #endif
                        minor_bckts_pos = (uint64_t *) &general_reg_1;
                        for (int i = 0; i < VECTOR_SCALE; ++i)
                        { 
                            if (m_minor_bckts_to_handle & (1 << i)) 
                            {
                    #ifdef PREFETCH_MINOR_BCKT_SIZES_OFF            
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
        #ifdef PREFETCH_SLOPES_AND_INTERCEPTS_MINOR_BCKTS
            case 0: 
            {
                if((int)(state[k].m_have_key) == 255)
                {
                    general_reg_2_double = _mm512_cvtepi64_pd(state[k].key);

                    slopes_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_slopes_addr), 0, 1);
                    intercepts_avx = _mm512_i64gather_pd(_mm512_add_epi64(state[k].pred_model_idx, v_intercepts_addr), 0, 1);
  
                    general_reg_1_double = _mm512_fmadd_pd(general_reg_2_double, slopes_avx, intercepts_avx);
                    general_reg_2_double = _mm512_mul_pd(major_bckt_idx_avx, num_minor_bckt_per_major_bckt_avx);
                    //general_reg_1_double = _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, general_reg_2_double); 
                    //general_reg_1_double = _mm512_min_pd(general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
                    //general_reg_1_double = _mm512_max_pd(general_reg_1_double, v_zero512_double);
                    //general_reg_1_double = _mm512_floor_pd(general_reg_1_double);

                    general_reg_1_double = _mm512_floor_pd(
                                                _mm512_max_pd(
                                                    _mm512_min_pd(
                                                        _mm512_fmsub_pd(general_reg_1_double, tot_num_minor_bckts_avx, general_reg_2_double), num_minor_bckt_per_major_bckt_minus_one_avx), v_zero512_double));

                    state[k].pred_model_idx = _mm512_cvtpd_epi64(general_reg_1_double);

                    general_reg_2 = _mm512_cvtpd_epi64(
                                        _mm512_fmadd_pd(general_reg_1_double, v_32bit_elem_size_double, v_minor_bckt_sizes_addr_double));

                    minor_bckt_sizes_256_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
                    general_reg_2_double = _mm512_cvtepi32_pd(minor_bckt_sizes_256_avx); 

                    state[k].stage = 2;

                    m_minor_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, threshold_avx);
                    if(m_minor_bckts_to_handle)
                    {
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
                        general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

                        //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

                    #ifdef PREFETCH_MINOR_BCKT_SIZES_OFF
                        minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
                    #endif
                        minor_bckts_pos = (uint64_t *) &general_reg_1;
                        for (int i = 0; i < VECTOR_SCALE; ++i)
                        { 
                            if (m_minor_bckts_to_handle & (1 << i)) 
                            {
                    #ifdef PREFETCH_MINOR_BCKT_SIZES_OFF            
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
                    general_reg_2_double = _mm512_mask_mul_pd(general_reg_2_double, state[k].m_have_key, major_bckt_idx_avx, num_minor_bckt_per_major_bckt_avx);
                    general_reg_1_double = _mm512_mask_fmsub_pd(general_reg_1_double, state[k].m_have_key, tot_num_minor_bckts_avx, general_reg_2_double); 
                    general_reg_1_double = _mm512_mask_min_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, num_minor_bckt_per_major_bckt_minus_one_avx);
                    general_reg_1_double = _mm512_mask_max_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double, v_zero512_double);
                    general_reg_1_double = _mm512_mask_floor_pd(general_reg_1_double, state[k].m_have_key, general_reg_1_double);

                    state[k].pred_model_idx = _mm512_mask_cvtpd_epi64(state[k].pred_model_idx, state[k].m_have_key, general_reg_1_double);

                    general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, state[k].m_have_key, 
                                        _mm512_mask_fmadd_pd(general_reg_1_double, state[k].m_have_key, v_32bit_elem_size_double, v_minor_bckt_sizes_addr_double));

                    minor_bckt_sizes_256_avx = _mm512_mask_i64gather_epi32(minor_bckt_sizes_256_avx, state[k].m_have_key, general_reg_2, 0, 1);
                    general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, state[k].m_have_key, minor_bckt_sizes_256_avx); 

                    state[k].stage = 2;

                    m_minor_bckts_to_handle = _mm512_mask_cmplt_pd_mask(state[k].m_have_key, general_reg_2_double, threshold_avx);
                    if(m_minor_bckts_to_handle)
                    {
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
                        general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

                        //_mm512_mask_prefetch_i64scatter_pd(0, m_minor_bckts_to_handle, general_reg_1, 1, _MM_HINT_T0);

                    #ifdef PREFETCH_MINOR_BCKT_SIZES_OFF
                        minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
                    #endif
                        minor_bckts_pos = (uint64_t *) &general_reg_1;
                        for (int i = 0; i < VECTOR_SCALE; ++i)
                        { 
                            if (m_minor_bckts_to_handle & (1 << i)) 
                            {
                    #ifdef PREFETCH_MINOR_BCKT_SIZES_OFF            
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
                                        _mm512_mask_fmadd_pd(general_reg_1_double, m_no_conflict, v_32bit_elem_size_double, v_minor_bckt_sizes_addr_double));

                    minor_bckt_sizes_256_avx = _mm512_mask_i64gather_epi32(minor_bckt_sizes_256_avx, m_no_conflict, general_reg_1, 0, 1);
                    general_reg_2_double = _mm512_mask_cvtepi32_pd(general_reg_2_double, m_no_conflict, minor_bckt_sizes_256_avx); 

                    m_minor_bckts_to_handle = _mm512_mask_cmplt_pd_mask(m_no_conflict, general_reg_2_double, threshold_avx);
                    if(m_minor_bckts_to_handle)
                    {
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
                        general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
                        general_reg_2 = _mm512_mask_cvtpd_epi64(general_reg_2, m_minor_bckts_to_handle, general_reg_2_double);

                        _mm512_mask_i64scatter_epi64(0, m_minor_bckts_to_handle, general_reg_2, state[k].key, 1);

                        minor_bckt_sizes_256_avx = _mm256_mask_add_epi32(minor_bckt_sizes_256_avx, m_minor_bckts_to_handle, minor_bckt_sizes_256_avx, v256_one);
                        _mm512_mask_i64scatter_epi32(0, m_minor_bckts_to_handle, general_reg_1, minor_bckt_sizes_256_avx, 1);   
                    }

                    m_spill_bckts_to_handle = _mm512_kandn(m_minor_bckts_to_handle, m_no_conflict);
                    if(m_spill_bckts_to_handle)
                    {
                        auto curr_keys = (int64_t *) &state[k].key;          
                        for(int j = 0; j < VECTOR_SCALE; ++j)
                        {
                            if (m_spill_bckts_to_handle & (1 << j)){ 
                                spill_bucket.push_back(curr_keys[j]);
                            }
                        }
                    }
                    state[k].m_have_key = _mm512_kandn(m_no_conflict, state[k].m_have_key);
                }
                num = _mm_popcnt_u32(state[k].m_have_key);
                
                if (num == VECTOR_SCALE || done >= SIMDStateSize) 
                {
                    auto curr_keys = (int64_t *) &state[k].key;
                    auto pred_model_idx_list = (uint64_t *) &state[k].pred_model_idx;   

                    for(int j = 0; j < VECTOR_SCALE; ++j)
                    {
                        if (state[k].m_have_key & (1 << j)) 
                        {
                            if (minor_bckt_sizes[pred_model_idx_list[j]] <
                                THRESHOLD) {  // The predicted bucket has not reached
                                // full capacity, so place the element in
                                // the bucket
                                minor_bckts[THRESHOLD * pred_model_idx_list[j] +
                                            minor_bckt_sizes[pred_model_idx_list[j]]] = curr_keys[j];
                                ++minor_bckt_sizes[pred_model_idx_list[j]];
                            } else {  // Place the item in the spill bucket
                                spill_bucket.push_back(curr_keys[j]);
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
                    if (done < SIMDStateSize)
                    {
                        num_temp = _mm_popcnt_u32(state[SIMDStateSize].m_have_key);
                        if (num + num_temp < VECTOR_SCALE) {
                            // compress v
                            state[k].key = _mm512_maskz_compress_epi64(state[k].m_have_key, state[k].key);
                            state[k].pred_model_idx = _mm512_maskz_compress_epi64(state[k].m_have_key, state[k].pred_model_idx);
                            // expand v -> temp
                            state[SIMDStateSize].key = _mm512_mask_expand_epi64(state[SIMDStateSize].key, _mm512_knot(state[SIMDStateSize].m_have_key), state[k].key);
                            state[SIMDStateSize].pred_model_idx = _mm512_mask_expand_epi64(state[SIMDStateSize].pred_model_idx, _mm512_knot(state[SIMDStateSize].m_have_key), state[k].pred_model_idx);
                            state[SIMDStateSize].m_have_key = mask[num + num_temp];
                            state[k].m_have_key = 0;
                            state[k].stage = 1;
                            --k;
                        } 
                        else
                        {
                            // expand temp -> v
                            state[k].key = _mm512_mask_expand_epi64(state[k].key, _mm512_knot(state[k].m_have_key), state[SIMDStateSize].key);
                            state[k].pred_model_idx = _mm512_mask_expand_epi64(state[k].pred_model_idx, _mm512_knot(state[k].m_have_key), state[SIMDStateSize].pred_model_idx);
                            // compress temp
                            state[SIMDStateSize].m_have_key = _mm512_kand(state[SIMDStateSize].m_have_key, _mm512_knot(mask[VECTOR_SCALE - num]));
                            state[SIMDStateSize].key = _mm512_maskz_compress_epi64(state[SIMDStateSize].m_have_key, state[SIMDStateSize].key);
                            state[SIMDStateSize].pred_model_idx = _mm512_maskz_compress_epi64(state[SIMDStateSize].m_have_key, state[SIMDStateSize].pred_model_idx);
                            state[k].m_have_key = mask[VECTOR_SCALE];
                            state[SIMDStateSize].m_have_key = (state[SIMDStateSize].m_have_key >> (VECTOR_SCALE - num));

                            general_reg_1_double = _mm512_cvtepi64_pd(state[k].pred_model_idx);
                    
                            general_reg_2 = _mm512_cvtpd_epi64(
                                            _mm512_fmadd_pd(general_reg_1_double, v_32bit_elem_size_double, v_minor_bckt_sizes_addr_double));

                            minor_bckt_sizes_256_avx = _mm512_i64gather_epi32(general_reg_2, 0, 1);
                            general_reg_2_double = _mm512_cvtepi32_pd(minor_bckt_sizes_256_avx); 

                            state[k].stage = 2;

                            m_minor_bckts_to_handle = _mm512_cmplt_pd_mask(general_reg_2_double, threshold_avx);
                            if(m_minor_bckts_to_handle)
                            {
                                general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_1_double, m_minor_bckts_to_handle, threshold_avx, general_reg_2_double);
                                general_reg_2_double = _mm512_mask_fmadd_pd(general_reg_2_double, m_minor_bckts_to_handle, v_64bit_elem_size_double, v_minor_bckts_addr_double);
                                general_reg_1 = _mm512_mask_cvtpd_epi64(general_reg_1, m_minor_bckts_to_handle, general_reg_2_double);

                            #ifdef PREFETCH_MINOR_BCKT_SIZES_OFF
                                minor_bckt_sizes_pos = (uint64_t *) &general_reg_2;
                            #endif
                                minor_bckts_pos = (uint64_t *) &general_reg_1;
                                for (int i = 0; i < VECTOR_SCALE; ++i)
                                { 
                                    if (m_minor_bckts_to_handle & (1 << i)) 
                                    {
                            #ifdef PREFETCH_MINOR_BCKT_SIZES_OFF            
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
    for (unsigned int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      // Iterate over the elements in the batch and store their predicted
      // ranks
      for (unsigned int elm_idx = 0; elm_idx < BATCH_SZ; ++elm_idx) {
        // Find the current element
        auto cur_elm = major_bckts[bckt_start_offset + elm_idx];

        // Predict the leaf-layer model
        batch_cache[elm_idx] = static_cast<int>(std::max(
            0.,
            std::min(num_models - 1., root_slope * cur_elm + root_intrcpt)));

        // Predict the CDF
        pred_cdf = slopes[batch_cache[elm_idx]] * cur_elm +
                   intercepts[batch_cache[elm_idx]];

        // Scale the predicted CDF to the number of minor buckets and cache it
        batch_cache[elm_idx] = static_cast<int>(std::max(
            0., std::min(NUM_MINOR_BCKT_PER_MAJOR_BCKT - 1.,
                         pred_cdf * TOT_NUM_MINOR_BCKTS -
                             major_bckt_idx * NUM_MINOR_BCKT_PER_MAJOR_BCKT)));
      }

      // Iterate over the elements in the batch again, and place them in the
      // sub-buckets, or spill bucket
      for (unsigned int elm_idx = 0; elm_idx < BATCH_SZ; ++elm_idx) {
        // Find the current element
        auto cur_elm = major_bckts[bckt_start_offset + elm_idx];

        // Check if the element will cause a bucket overflow
        if (minor_bckt_sizes[batch_cache[elm_idx]] <
            THRESHOLD) {  // The predicted bucket has not reached
          // full capacity, so place the element in
          // the bucket
          minor_bckts[THRESHOLD * batch_cache[elm_idx] +
                      minor_bckt_sizes[batch_cache[elm_idx]]] = cur_elm;
          ++minor_bckt_sizes[batch_cache[elm_idx]];
        } else {  // Place the item in the spill bucket
          spill_bucket.push_back(cur_elm);
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
        major_bckt_sizes[major_bckt_idx] - num_batches * BATCH_SZ;

    for (unsigned int elm_idx = 0; elm_idx < num_remaining_elm; ++elm_idx) {
      auto cur_elm = major_bckts[bckt_start_offset + elm_idx];
      batch_cache[elm_idx] = static_cast<int>(std::max(
          0., std::min(num_models - 1., root_slope * cur_elm + root_intrcpt)));
      pred_cdf = slopes[batch_cache[elm_idx]] * cur_elm +
                 intercepts[batch_cache[elm_idx]];
      batch_cache[elm_idx] = static_cast<int>(std::max(
          0., std::min(NUM_MINOR_BCKT_PER_MAJOR_BCKT - 1.,
                       pred_cdf * TOT_NUM_MINOR_BCKTS -
                           major_bckt_idx * NUM_MINOR_BCKT_PER_MAJOR_BCKT)));
    }

    //for (unsigned elm_idx = 0;
    //     elm_idx < major_bckt_sizes[major_bckt_idx] - num_batches * BATCH_SZ;
    //     ++elm_idx) {
    for (unsigned elm_idx = 0; elm_idx < num_remaining_elm; ++elm_idx) {
      auto cur_elm = major_bckts[bckt_start_offset + elm_idx];
      if (minor_bckt_sizes[batch_cache[elm_idx]] < THRESHOLD) {
        minor_bckts[THRESHOLD * batch_cache[elm_idx] +
                    minor_bckt_sizes[batch_cache[elm_idx]]] = cur_elm;
        ++minor_bckt_sizes[batch_cache[elm_idx]];
      } else {
        spill_bucket.push_back(cur_elm);
      }
    }
  #endif

    //----------------------------------------------------------//
    //                MODEL-BASED COUNTING SORT                 //
    //----------------------------------------------------------//

    // Iterate over the minor buckets of the current bucket
    for (unsigned int bckt_idx = 0; bckt_idx < NUM_MINOR_BCKT_PER_MAJOR_BCKT;
         ++bckt_idx) {
      if (minor_bckt_sizes[bckt_idx] > 0) {
        // Update the bucket start offset
        bckt_start_offset =
            1. * (major_bckt_idx * NUM_MINOR_BCKT_PER_MAJOR_BCKT + bckt_idx) *
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
                         root_slope * minor_bckts[bckt_idx * THRESHOLD] +
                             root_intrcpt)));

        int pred_model_last_elm = static_cast<int>(std::max(
            0.,
            std::min(num_models - 1.,
                     root_slope * minor_bckts[bckt_idx * THRESHOLD +
                                              minor_bckt_sizes[bckt_idx] - 1] +
                         root_intrcpt)));

        if (pred_model_first_elm ==
            pred_model_last_elm) {  // Avoid CDF model traversal and predict the
          // CDF only using the leaf model

          // Iterate over the elements and place them into the minor buckets
          for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
               ++elm_idx) {
            // Find the current element
            auto cur_elm = minor_bckts[bckt_idx * THRESHOLD + elm_idx];

            // Predict the CDF
            pred_cdf = slopes[pred_model_first_elm] * cur_elm +
                       intercepts[pred_model_first_elm];

            // Scale the predicted CDF to the input size and cache it
            pred_idx_cache[elm_idx] = static_cast<int>(std::max(
                0., std::min(THRESHOLD - 1.,
                             (pred_cdf * INPUT_SZ) - bckt_start_offset)));

            // Update the counts
            ++cnt_hist[pred_idx_cache[elm_idx]];
          }
        } else {  // Fully traverse the CDF model again to predict the CDF of
                  // the
          // current element

          // Iterate over the elements and place them into the minor buckets
          for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
               ++elm_idx) {
            // Find the current element
            auto cur_elm = minor_bckts[bckt_idx * THRESHOLD + elm_idx];

            // Predict the model idx in the leaf layer
            auto model_idx_next_layer = static_cast<int>(
                std::max(0., std::min(num_models - 1.,
                                      root_slope * cur_elm + root_intrcpt)));
            // Predict the CDF
            pred_cdf = slopes[model_idx_next_layer] * cur_elm +
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
          major_bckts[num_tot_elms_in_bckts +
                      cnt_hist[pred_idx_cache[elm_idx]]] =
              minor_bckts[bckt_idx * THRESHOLD + elm_idx];
          // Update counts
          --cnt_hist[pred_idx_cache[elm_idx]];
        }

        //----------------------------------------------------------//
        //                  TOUCH-UP & COMPACTION                   //
        //----------------------------------------------------------//

        // After the model-based bucketization process is done, switch to a
        // deterministic sort
        T elm;
        int cmp_idx;

        // Perform Insertion Sort
        for (unsigned int elm_idx = 0; elm_idx < minor_bckt_sizes[bckt_idx];
             ++elm_idx) {
          cmp_idx = num_tot_elms_in_bckts + elm_idx - 1;
          elm = major_bckts[num_tot_elms_in_bckts + elm_idx];
          while (cmp_idx >= 0 && elm < major_bckts[cmp_idx]) {
            major_bckts[cmp_idx + 1] = major_bckts[cmp_idx];
            --cmp_idx;
          }

          major_bckts[cmp_idx + 1] = elm;
        }

        num_tot_elms_in_bckts += minor_bckt_sizes[bckt_idx];
      }  // end of iteration of each minor bucket
    }
  }  // end of iteration over each major bucket

   //std::cout << "learned_sort_avx: spill_bucket size: "<< spill_bucket.size() << std::endl;

  //----------------------------------------------------------//
  //                 SORT THE SPILL BUCKET                    //
  //----------------------------------------------------------//

  std::sort(spill_bucket.begin(), spill_bucket.end());

  //----------------------------------------------------------//
  //               PLACE BACK THE EXCEPTION VALUES            //
  //----------------------------------------------------------//

  vector<T> linear_vals, linear_count;

  for (auto val_idx = 0; val_idx < EXCEPTION_VEC_INIT_CAPACITY; ++val_idx) {
    for (size_t exc_elm_idx = 0;
         exc_elm_idx < repeated_keys_predicted_ranks[val_idx].size();
         ++exc_elm_idx) {
      linear_vals.push_back(
          repeated_keys_predicted_ranks[val_idx][exc_elm_idx]);
      linear_count.push_back(repeated_key_counts[val_idx][exc_elm_idx]);
    }
  }

  //----------------------------------------------------------//
  //               MERGE BACK INTO ORIGINAL ARRAY             //
  //----------------------------------------------------------//

  // Merge the spill bucket with the elements in the buckets
  std::merge(major_bckts.begin(), major_bckts.begin() + num_tot_elms_in_bckts,
             spill_bucket.begin(), spill_bucket.end(),
             begin + total_repeated_keys);

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

  while (input_idx < INPUT_SZ && exc_idx < linear_vals.size()) {
    if (begin[input_idx] < linear_vals[exc_idx]) {
      begin[ptr] = begin[input_idx];
      ptr++;
      input_idx++;
    } else {
      for (int i = 0; i < linear_count[exc_idx]; i++) {
        begin[ptr + i] = linear_vals[exc_idx];
      }
      ptr += linear_count[exc_idx];
      exc_idx++;
    }
  }

  while (exc_idx < linear_vals.size()) {
    for (int i = 0; i < linear_count[exc_idx]; i++) {
      begin[ptr + i] = linear_vals[exc_idx];
    }
    ptr += linear_count[exc_idx];
    exc_idx++;
  }

  while (input_idx < INPUT_SZ) {
    begin[ptr] = begin[input_idx];
    ptr++;
    input_idx++;
  }

  // The input array is now sorted
}

/**
 * Vectorized version of sort() method
 */
template <class RandomIt>
void learned_sort::sort_avx(
    RandomIt begin, RandomIt end,
    typename RMI<typename iterator_traits<RandomIt>::value_type>::Params
        &params) {
  // Use std::sort for very small arrays
  if (std::distance(begin, end) <=
      std::max(params.fanout * params.threshold, 5 * params.arch[1])) {
    std::sort(begin, end);
  } else {
    // Train
    RMI rmi = train_avx(begin, end, params);

    // Sort
    if (rmi.trained)
      _sort_trained_avx(begin, end, rmi);

    else  // Fall back in case the model could not be trained
      std::sort(begin, end);
  }
}

/**
 * Vectorized version of sort() method
 */
template <class RandomIt>
void learned_sort::sort_avx(RandomIt begin, RandomIt end) {
  typename RMI<typename iterator_traits<RandomIt>::value_type>::Params p;
  learned_sort::sort_avx(begin, end, p);
}

#endif  // LEARNED_SORT_AVX_H
