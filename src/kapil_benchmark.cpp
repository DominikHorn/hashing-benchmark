#pragma once

#include "include/mmphf_table.hpp"
#include "include/monotone_hashtable.hpp"
#include "include/kapil_chained.hpp"
#include "include/kapil_chained_exotic.hpp"
#include "include/kapil_chained_model.hpp"

#include "include/kapil_probe.hpp"
#include "include/kapil_probe_exotic.hpp"
#include "include/kapil_probe_model.hpp"


#include "include/kapil_cuckoo.hpp"
#include "include/kapil_cuckoo_model.hpp"
#include "include/kapil_cuckoo_exotic.hpp"

// Order is important
#include "include/convenience/undef.hpp"


// #include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <hashing.hpp>
#include <hashtable.hpp>
#include <iostream>
#include <iterator>
#include <learned_hashing.hpp>
#include <limits>
#include <masters_thesis.hpp>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <string>

#include <stdio.h>
#include <stdlib.h>

#include "../thirdparty/perfevent/PerfEvent.hpp"
#include "support/datasets.hpp"
#include "support/probing_set.hpp"
#include "../include/convenience/builtins.hpp"
#include "include/mmphf/rank_hash.hpp"
#include "include/rmi.hpp"

using Key = std::uint64_t;
using Payload = std::uint64_t;

const std::vector<std::int64_t> probe_distributions{
    // static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
    //     dataset::ProbingDistribution::EXPONENTIAL_SORTED),
    // static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
    //     dataset::ProbingDistribution::EXPONENTIAL_RANDOM),
    static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(
        dataset::ProbingDistribution::UNIFORM)};

const std::vector<std::int64_t> dataset_sizes{100000000};
const std::vector<std::int64_t> succ_probability{100,50,0};
// const std::vector<std::int64_t> datasets{
//     static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::SEQUENTIAL),
//     static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::GAPPED_10),
//     static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::UNIFORM),
//     static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::NORMAL),
//     static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::WIKI)
//     // static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::OSM),
//     // static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::FB)
//     };

const std::vector<dataset::ID> datasets{
    dataset::ID::SEQUENTIAL
    // dataset::ID::GAPPED_10,
    // dataset::ID::UNIFORM,
    // dataset::ID::NORMAL,
    // dataset::ID::WIKI
    // dataset::ID::OSM,
    // dataset::ID::FB
    };


template <class Table_class>
void print_data_statistics_generic(uint64_t dataset_size,dataset::ID did)
{
  
  std::cout << "performing setup... ";
  auto start = std::chrono::steady_clock::now();

  // Generate data (keys & payloads) & probing set
  std::vector<std::pair<Key, Payload>> data{};
  data.reserve(dataset_size);
  {
    auto keys = dataset::load_cached<Key>(did, dataset_size);

    std::transform(
        keys.begin(), keys.end(), std::back_inserter(data),
        [](const Key& key) { return std::make_pair(key, key - 5); });
    // int succ_probability=100;
    // probing_set = dataset::generate_probing_set(keys, probing_dist,succ_probability);
  }

  if (data.empty()) {
    // otherwise google benchmark produces an error ;(
    
    std::cout << "failed" << std::endl;
    return;
  }

  // build table
  // if (prev_table != nullptr) free_lambda();
  Table_class *prev_table = new Table_class(data);
  // free_lambda = []() { delete ((Table*)prev_table); };

  // measure time elapsed
  const auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "succeeded in " << std::setw(9) << diff.count() << " seconds"
            << std::endl;

 
  assert(prev_table != nullptr);
  Table_class* table = (Table_class*)prev_table;

//   std::cout << "Start Printing Data Statistics "<< std::endl;
//   auto start = high_resolution_clock::now(); 

  table->print_data_statistics();

//   auto stop = high_resolution_clock::now(); 
  // auto duration = duration_cast<milliseconds>(stop - start); 
//   auto duration = duration_cast<nanoseconds>(stop - start); 

  std::cout << "Done Printing Data Statistics "<< std::endl;

  delete ((Table_class*)prev_table);

  return;
}






// std::map<std::string, std::string> model_type_dict={
//     "Traditional":"Traditional",
//     "Exotic":"Exotic",
//     "Model":"Model"

// }

// std::map<std::string, std::string> scheme_dict={
//     "Cuckoo":"Cuckoo",
//     "Linear":"Linear",
//     "Chained":"Chained"
// }

// std::map<std::string, std::string> kickin_strat_dict={
//     "Balanced":"using KickingStrat = kapilmodelhashtable::KapilModelBalancedKicking;",
//     "Biased":"using KickingStrat = kapilmodelhashtable::KapilModelBiasedKicking<kickinit_strat_bias>;"
// }

template<class Hashfn>
void stats_linearprobe_model(uint64_t bucket_sz,uint64_t overalloc)
{   
    Hashfn Model;
    using KapilLinearModelHashTable= masters_thesis::KapilLinearModelHashTable<Key, Payload, bucket_sz,overalloc, Model>;

    for(int i=0;i<dataset_sizes.size();i++)
    {
        for(int j=0;j<datasets.size();j++)
        {
            std::cout<<" Dataset: "<<datasets[j]<<" Size: "<<dataset_sizes[i]<<std::endl;
            print_data_statistics_generic<masters_thesis::KapilLinearModelHashTable>(dataset_sizes[i], datasets[j]);
        }
    }
  
}

template<class Hashfn,uint64_t bucket_sz,uint64_t overalloc>
void stats_linearprobe_hash()
{   
    Hashfn Model;
    // using KapilLinearModelHashTable= masters_thesis::KapilLinearModelHashTable<Key, Payload, bucket_sz,overalloc, Model>;
    using KapilLinearHashTable = masters_thesis::KapilLinearHashTable<Key, Payload, bucket_sz,overalloc, Hashfn>;

    for(int i=0;i<dataset_sizes.size();i++)
    {
        for(int j=0;j<datasets.size();j++)
        {
            std::cout<<" Dataset: "<<datasets[j]<<" Size: "<<dataset_sizes[i]<<std::endl;
            print_data_statistics_generic<masters_thesis::KapilLinearHashTable>(dataset_sizes[i], datasets[j]);
        }
    }
  
}

void stats_choose_hashfn_scheme(std::string hashfn_identifier,std::string scheme_identifier,int bucket_sz,int overalloc,uint64_t max_error,uint64_t max_models)
{

  // std::map<std::string, std::string> hash_mapping_dict={
  //   "MURMUR":"using MURMUR = hashing::MurmurFinalizer<Key>;",
  //   "MultPrime64":"using MultPrime64 = hashing::MultPrime64;",
  //   "FibonacciPrime64":"using FibonacciPrime64 = hashing::FibonacciPrime64;",
  //   "AquaHash":"using AquaHash = hashing::AquaHash<Key>;",
  //   "XXHash3":"using XXHash3 = hashing::XXHash3<Key>;",
  //   "MWHC":"using MWHC = exotic_hashing::MWHC<Key>;",
  //   "FST":"using FST = exotic_hashing::FastSuccinctTrie<Data>;",
  //   "RadixSplineHash":"using RadixSplineHash = learned_hashing::RadixSplineHash<std::uint64_t,num_radix_bits,max_error,100000000>;",
  //   "RMIHash":"using RMIHash = learned_hashing::RMIHash<std::uint64_t,max_models>;"
  // }
  // if (scheme_identifier.compare("linear_probe"))
  // {

  //   if(hashfn_identifier.compare( "MURMUR")){
  //     stats_linearprobe_hash<hashing::MurmurFinalizer<Key>,bucket_sz,overalloc>();
  //     }
  //   if(hashfn_identifier.compare( "MultPrime64")){
  //     stats_linearprobe_hash<hashing::MultPrime64,bucket_sz,overalloc>();
  //     }
  //   if(hashfn_identifier.compare( "FibonacciPrime64")){
  //     stats_linearprobe_hash<hashing::FibonacciPrime64,bucket_sz,overalloc>();
  //     }
  //   if(hashfn_identifier.compare( "AquaHash")){
  //     stats_linearprobe_hash<hashing::AquaHash<Key>,bucket_sz,overalloc>();
  //     }
  //   if(hashfn_identifier.compare( "XXHash3")){
  //     stats_linearprobe_hash<hashing::XXHash3<Key>,bucket_sz,overalloc>();
  //     }
  //   // if(hashfn_identifier.compare( "MWHC")){
  //   //   stats_linearprobe_model<exotic_hashing::MWHC<Key>,bucket_sz,overalloc>();
  //   //   }
  //   // if(hashfn_identifier.compare( "FST")){
  //   //   stats_linearprobe_model<exotic_hashing::FastSuccinctTrie<Data>,bucket_sz,overalloc>();
  //   //   }
  //   // if(hashfn_identifier.compare( "RadixSplineHash")){
  //   //   stats_linearprobe_model<learned_hashing::RadixSplineHash<std::uint64_t,18,max_error,100000000>,bucket_sz,overalloc>();
  //   //   }  
  //   // if(hashfn_identifier.compare( "RMIHash")){
  //   //   stats_linearprobe_model<learned_hashing::RMIHash<std::uint64_t,max_models>,bucket_sz,overalloc>();
  //   //   }    
    

  // }


  return;

  
}



//printing data statistics
void expt_1()
{
  std::vector<uint64_t> bucket_sz_vec = {1,2,4,8};
  std::vector<uint64_t> overalloc_vec = {50,100,300};
  std::vector<std::string> trad_hash = {"MURMUR", "MultPrime64", "XXHash3"};
  std::vector<std::string> model_hash = {"RMIHash" ,"RadixSplineHash"};
  std::string scheme="linear_probe";

  for(int i=0;i<bucket_sz_vec.size();i++)
  {
    for(int j=0;j<overalloc_vec.size();j++)
    {
      for(int k=0;k<trad_hash.size();k++)
      {
        std::cout<<" Data Stats Print "<<bucket_sz_vec[i]<<" "<<overalloc_vec[j]<<" "<<trad_hash[k]<<" ";
        std::cout<<" Traditional Linear Balanced 0 0 0 "<<std::endl;
        stats_choose_hashfn_scheme(trad_hash[k],scheme,bucket_sz_vec[i],overalloc_vec[j],0,0);
      }
    }
  }

  for(int i=0;i<bucket_sz_vec.size();i++)
  {
    for(int j=0;j<overalloc_vec.size();j++)
    {
      for(int k=0;k<model_hash.size();k++)
      {
        std::cout<<" Data Stats Print "<<bucket_sz_vec[i]<<" "<<overalloc_vec[j]<<" "<<model_hash[k]<<" ";
        std::cout<<" Model Linear Balanced 0 1000 1024 "<<std::endl;
        stats_choose_hashfn_scheme(model_hash[k],scheme,bucket_sz_vec[i],overalloc_vec[j],1024,1000);
      }
    }
  }

  for(int i=0;i<bucket_sz_vec.size();i++)
  {
    for(int j=0;j<overalloc_vec.size();j++)
    {
      for(int k=0;k<model_hash.size();k++)
      {
        std::cout<<" Data Stats Print "<<bucket_sz_vec[i]<<" "<<overalloc_vec[j]<<" "<<model_hash[k]<<" ";
        std::cout<<" Model Linear Balanced 0 100000 32 "<<std::endl;
        stats_choose_hashfn_scheme(model_hash[k],scheme,bucket_sz_vec[i],overalloc_vec[j],32,100000);
      }
    }
  }

}
  

int main()
{

  // using RadixSplineHash = learned_hashing::RadixSplineHash<std::uint64_t,18,1024>;
  // using KapilLinearModelHashTable= masters_thesis::KapilLinearModelHashTable<Key, Payload, 2,50, RadixSplineHash>;

  // print_data_statistics_generic<KapilLinearModelHashTable>(dataset_sizes[0], datasets[0]);

  expt_1();

  return 0;
}
