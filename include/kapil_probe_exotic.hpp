#pragma once

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <learned_hashing.hpp>
#include <limits>
#include <string>
#include <utility>

#include "include/convenience/builtins.hpp"
#include "include/support.hpp"

namespace masters_thesis {
template <class Key, class Payload, size_t BucketSize,
           class MMPHF,
          bool ManualPrefetch = false,
          Key Sentinel = std::numeric_limits<Key>::max()>
class KapilLinearExoticHashTable {
    
    public:
    MMPHF mmphf; 

  struct Bucket {
    std::array<Key, BucketSize> keys;
    std::array<Payload, BucketSize> payloads;
    Bucket* next = nullptr;

    Bucket() {
      // Sentinel value in each slot per default
      std::fill(keys.begin(), keys.end(), Sentinel);
    }


    bool insert(const Key& key, const Payload& payload,
                support::Tape<Bucket>& tape) {
      Bucket* current = this;

      for (size_t i = 0; i < BucketSize; i++) {
        if (current->keys[i] == Sentinel) {
          current->keys[i] = key;
          current->payloads[i] = payload;
          return true;
        }

        if (current->keys[i] == key) {
          current->keys[i] = key;
          current->payloads[i] = payload;
          return true;
        }
      }


      return false;



      // previous = current;
      

      // static var will be shared by all instances
      // previous->next = tape.alloc();
      // previous->next->insert(key, payload, tape);
    }

    size_t byte_size() const {
      return sizeof(Bucket) + (next != nullptr ? next->byte_size() : 0);
    }
  };

  /// directory of buckets
  std::vector<Bucket> buckets;

  /// model for predicting the correct index
//   Model model;

  /// allocator for buckets
  std::unique_ptr<support::Tape<Bucket>> tape;

  /**
   * Inserts a given (key,payload) tuple into the hashtable.
   *
   * Note that this function has to be private for now since
   * model retraining will be necessary if this is used as a
   * construction interface.
   */



  forceinline void insert(const Key& key, const Payload& payload) {
    auto index = mmphf(key)%(buckets.size());
    assert(index >= 0);
    assert(index < buckets.size());
    auto start=index;

    for(;(index-start<50000);)
    {
      // std::cout<<"index: "<<index<<std::endl;
      if(buckets[index%buckets.size()].insert(key, payload, *tape))
      {
        return ;
      }
      else
      {
        index++;
      }
    }

    throw std::runtime_error("Building " + this->name() + " failed: during probing, all buckets along the way are full");

    return ;

  }

 public:
  KapilLinearExoticHashTable() = default;

  /**
   * Constructs a KapilLinearExoticHashTable given a list of keys
   * together with their corresponding payloads
   */
  KapilLinearExoticHashTable(std::vector<std::pair<Key, Payload>> data)
      :  buckets((1 + data.size()*(1.00)) / BucketSize),
        tape(std::make_unique<support::Tape<Bucket>>()) {
            
    // ensure data is sorted
    std::sort(data.begin(), data.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // obtain list of keys -> necessary for model training
    std::vector<Key> keys;
    keys.reserve(data.size());
    std::transform(data.begin(), data.end(), std::back_inserter(keys),
                   [](const auto& p) { return p.first; });

    // train model on sorted data
    // model.train(keys.begin(), keys.end(), buckets.size());
    mmphf.construct(keys.begin(), keys.end());

    // insert all keys according to model prediction.
    // since we sorted above, this will permit further
    // optimizations during lookup etc & enable implementing
    // efficient iterators in the first place.
    for (const auto& d : data) insert(d.first, d.second);
  }

  class Iterator {
    size_t directory_ind, bucket_ind;
    Bucket const* current_bucket;
    const KapilLinearExoticHashTable& hashtable;

    Iterator(size_t directory_ind, size_t bucket_ind, Bucket const* bucket,
             const KapilLinearExoticHashTable& hashtable)
        : directory_ind(directory_ind),
          bucket_ind(bucket_ind),
          current_bucket(bucket),
          hashtable(hashtable) {}

   public:
    forceinline const Key& key() const {
      assert(current_bucket != nullptr);
      return current_bucket->keys[bucket_ind];
    }

    forceinline const Payload& payload() const {
      assert(current_bucket != nullptr);
      return current_bucket->payloads[bucket_ind];
    }

    forceinline Iterator& operator++() {
      assert(current_bucket != nullptr);

      // prefetch next bucket during iteration. If there is no
      // next bucket, then prefetch the next directory slot
      if (current_bucket->next != nullptr)
        prefetch(current_bucket->next, 0, 0);
      else if (directory_ind + 1 < hashtable.buckets.size())
        prefetch(&hashtable.buckets[directory_ind + 1], 0, 0);

      // since data was inserted in sorted order,
      // simply advancing to next slot does the trick
      bucket_ind++;

      // switch to next bucket
      if (bucket_ind >= BucketSize ||
          current_bucket->keys[bucket_ind] == Sentinel) {
        bucket_ind = 0;
        current_bucket = current_bucket->next;

        // switch to next bucket chain in directory
        if (current_bucket == nullptr &&
            ++directory_ind < hashtable.buckets.size())
          current_bucket = &hashtable.buckets[directory_ind];
      }

      return *this;
    }

    // // TODO(dominik): support postfix increment
    // forceinline Iterator operator++(int) {
    //   Iterator old = *this;
    //   ++this;
    //   return old;
    // }

    forceinline bool operator==(const Iterator& other) const {
      return directory_ind == other.directory_ind &&
             bucket_ind == other.bucket_ind &&
             current_bucket == other.current_bucket &&
             &hashtable == &other.hashtable;
    }

    friend class KapilLinearExoticHashTable;
  };

  /**
   * Past the end iterator, use like usual in stl
   */
  forceinline Iterator end() const {
    return {buckets.size(), 0, nullptr, *this};
  }

  forceinline int hash_val(const Key& key)
  {
    return mmphf(key);
  }

   void print_data_statistics()
  {
    std::vector<uint64_t> dist_from_ideal;
    std::vector<uint64_t> dist_to_empty;


    std::map<int,int> dist_from_ideal_map;
    std::map<int,int> dist_to_empty_map;

    for(uint64_t buck_ind=0;buck_ind<buckets.size();buck_ind++)
    {
      auto bucket = &buckets[buck_ind%buckets.size()];

      for (size_t i = 0; i < BucketSize; i++)
       {
        
          const auto& current_key = bucket->keys[i];
          if(current_key==Sentinel)
          {
            break;
          }

          size_t directory_ind = mmphf(current_key)%(buckets.size());
          // std::cout<<" pred val: "<<directory_ind<<" key val: "<<current_key<<" bucket val: "<<buck_ind<<std::endl;
          dist_from_ideal.push_back(directory_ind-buck_ind);
        }

    }  

    for(int buck_ind=0;buck_ind<buckets.size();buck_ind++)
    {
      auto directory_ind=buck_ind;
      auto start=directory_ind;
      for(;directory_ind<start+50000;)
      {
        auto bucket = &buckets[directory_ind%buckets.size()];

        // Generic non-SIMD algorithm. Note that a smart compiler might vectorize
        
        bool found_sentinel=false;
        for (size_t i = 0; i < BucketSize; i++)
        {
            const auto& current_key = bucket->keys[i];
            // std::cout<<current_key<<" match "<<key<<std::endl;
            if (current_key == Sentinel) {
              found_sentinel=true;
              break;
              // return end();
              }
        }

        if(found_sentinel)
        {
          break;
        }
        
        directory_ind++;        
      }  

      dist_to_empty.push_back(directory_ind-buck_ind);

    } 


    std::sort(dist_from_ideal.begin(),dist_from_ideal.end());
    std::sort(dist_to_empty.begin(),dist_to_empty.end());


    for(int i=0;i<dist_from_ideal.size();i++)
    {
      dist_from_ideal_map[dist_from_ideal[i]]=0;
    }

    for(int i=0;i<dist_from_ideal.size();i++)
    {
      dist_from_ideal_map[dist_from_ideal[i]]+=1;
    }

    for(int i=0;i<dist_to_empty.size();i++)
    {
      dist_to_empty_map[dist_to_empty[i]]=0;
    }

    for(int i=0;i<dist_to_empty.size();i++)
    {
      dist_to_empty_map[dist_to_empty[i]]+=1;
    }


    std::map<int, int>::iterator it;

    std::cout<<"Start Distance To Empty"<<std::endl;

    for (it = dist_to_empty_map.begin(); it != dist_to_empty_map.end(); it++)
    {
      std::cout<<"Distance To Empty: ";
      std::cout<<it->first<<" : "<<it->second<<std::endl;
        // std::cout << it->first    // string (key)
        //           << ':'
        //           << it->second   // string's value 
        //           << std::endl;
    }

    std::cout<<"Start Distance From Ideal"<<std::endl;

    for (it = dist_from_ideal_map.begin(); it != dist_from_ideal_map.end(); it++)
    {
      std::cout<<"Distance From Ideal: ";
      std::cout<<it->first<<" : "<<it->second<<std::endl;
        // std::cout << it->first    // string (key)
        //           << ':'
        //           << it->second   // string's value 
        //           << std::endl;
    }

    return;



  }

   int useless_func()
  {
    return 0;
  }



  /**
   * Returns an iterator pointing to the payload for a given key
   * or end() if no such key could be found
   *
   * @param key the key to search
   */
  forceinline int operator[](const Key& key) const {
    assert(key != Sentinel);

    // will become NOOP at compile time if ManualPrefetch == false
    // const auto prefetch_next = [](const auto& bucket) {
    //   if constexpr (ManualPrefetch)
    //     // manually prefetch next if != nullptr;
    //     if (bucket->next != nullptr) prefetch(bucket->next, 0, 0);
    // };

    // obtain directory bucket
    size_t directory_ind = mmphf(key)%(buckets.size());

    auto start=directory_ind;

    for(;directory_ind<start+50000;)
    {
       auto bucket = &buckets[directory_ind%buckets.size()];

      // Generic non-SIMD algorithm. Note that a smart compiler might vectorize
      // this nested loop construction anyways.
        // bool exit=false;
        for (size_t i = 0; i < BucketSize; i++)
       {
          const auto& current_key = bucket->keys[i];
          if (current_key == Sentinel) {
            return 0;
            // return end();
            }
          if (current_key == key) {
            return 1;
            // return {directory_ind, i, bucket, *this};
          }
        }
      // if(exit)
      // {
      //   break;
      // }

      directory_ind++;  

      //   prefetch_next(bucket);
      
    }

   
    return 0;
    // return end();
  }

  std::string name() {
    std::string prefix = (ManualPrefetch ? "Prefetched" : "");
    return prefix + "KapilLinearExoticHashTable<" + std::to_string(sizeof(Key)) + ", " +
           std::to_string(sizeof(Payload)) + ", " + std::to_string(BucketSize) +
           ", " + mmphf.name() + ">";
  }

  size_t directory_byte_size() const {
    size_t directory_bytesize = sizeof(decltype(buckets));
    for (const auto& bucket : buckets) directory_bytesize += bucket.byte_size();
    return directory_bytesize;
  }

  //kapil_change: assuming model size to be zero  
  size_t model_byte_size() const { return 0; }

  size_t byte_size() const { return model_byte_size() + directory_byte_size(); }
};
}  // namespace masters_thesis
