
/* data generation functionalities for input relations */

#pragma once

#include <stdio.h>              /* perror */
#include <time.h>               /* time() */
#include <random>
#include <type_traits>

#include "configs/base_configs.h"
#include "utils/data_structures.h"
#include "utils/memory.h"
#include "utils/math.h"
#include "utils/barrier.h"
#include "utils/cpu_mapping.h"

#include <set>

#define ZERO_PAYLOAD

using namespace std;

enum synthetic_workload_distr_t
{
    NORMAL,
    UNIFORM,
    EXPONENTIAL,
    LOGNORMAL
};

struct DataDistnParams
{
    double normal_mean = 0;
    double normal_stddev = 1;
    double uniform_a = 0;
    double uniform_b = 1;
    double exponential_lambda = 2;
    double exponential_scale = 1e6;
    double lognormal_mean = 0;
    double lognormal_stddev = 0.25;
    double lognormal_scale = 1e6;
};

template<class KeyType, class PayloadType>
struct create_arg_t {
    Relation<KeyType, PayloadType>          rel;
    int64_t             firstkey;
    int64_t             maxid;
    Relation<KeyType, PayloadType> *        fullrel;
    volatile void *     locks;
    pthread_barrier_t * barrier;
};

template<class Type>
struct create_arg_generic_t {
    Type*          rel;
    int64_t     num_elems;
    int64_t             firstkey;
    int64_t             maxid;
    Type**        fullrel;
    volatile void *     locks;
    pthread_barrier_t * barrier;
};

//based on the ETH implementation 
//create an ETH workload relation of primary keys, where the keys are unique, have no-gaps, and randomly shuffled
template<class KeyType, class PayloadType>
int create_eth_workload_relation_pk(Relation<KeyType, PayloadType> * relation, int64_t num_tuples, int relation_padding);

//based on the ETH implementation 
//create an ETH workload relation of foreign keys, where the keys are non-unique, randomly shuffled, and each key between 0 and maxid exists at least once
template<class KeyType, class PayloadType>
int create_eth_workload_relation_fk(Relation<KeyType, PayloadType> *relation, int64_t num_tuples, const int64_t maxid, int relation_padding);

template<class KeyType, class PayloadType>
int create_input_workload_relation(KeyType* input_keys, Relation<KeyType, PayloadType> * relation, int64_t num_tuples, int relation_padding);

//create a synthetic workload relation of foreign keys, where the keys are following a certain data distribution
template<class KeyType, class PayloadType>
int create_synthetic_workload_relation_fk(Relation<KeyType, PayloadType> *relation, int64_t num_tuples, synthetic_workload_distr_t data_distn_type, DataDistnParams* data_distn_params, int relation_padding);

template<class KeyType, class PayloadType>
int numa_localize(Tuple<KeyType, PayloadType> * relation, int64_t num_tuples, uint32_t nthreads);

template<class KeyType, class PayloadType>
int numa_localize_varlen(Tuple<KeyType, PayloadType> * relation, int64_t* num_tuples_for_threads, uint32_t nthreads);

template<class Type>
int numa_localize_generic_varlen(Type * relation, int64_t* num_tuples_for_threads, uint32_t nthreads);

//based on the ETH implementation 
static int seeded = 0;
static unsigned int seedValue;

//based on the ETH implementation 
static void check_seed()
{
    if(!seeded) {
        seedValue = time(NULL);
        srand(seedValue);
        seeded = 1;
    }
}

//based on the ETH implementation 
template<class KeyType, class PayloadType>
void knuth_shuffle(Relation<KeyType, PayloadType> * relation)
{
    uint64_t i;
    for (i = relation->num_tuples - 1; i > 0; i--) {
        int64_t  j             = RAND_RANGE(i);
        KeyType tmp            = relation->tuples[i].key;
        relation->tuples[i].key = relation->tuples[j].key;
        relation->tuples[j].key = tmp;

        PayloadType tmp1            = relation->tuples[i].payload;
        relation->tuples[i].payload = relation->tuples[j].payload;
        relation->tuples[j].payload = tmp1;
    }
}

//based on the ETH implementation 
template<class KeyType, class PayloadType>
void random_real_data_uint_gen(Relation<KeyType, PayloadType> * rel,string filename_arg) 
{
    cout<<"reading file "<<filename_arg<<endl;
    string file_name;
    int c_check=0;
    if (filename_arg.compare("wiki")==0)
    {
      c_check=1;
      cout<<"wiki"<<endl;
      file_name= "/spinning/sabek/learned_join_datasets_sosd/wiki_ts_200M_uint64";  
    }

    if (filename_arg.compare("osm_cellids")==0)
    {
      c_check=1;
      cout<<"osm_cellids"<<endl;
       file_name="/spinning/sabek/learned_join_datasets_sosd/osm_cellids_800M_uint64";
    }

    if (filename_arg.compare("map_learned_index_paper")==0)
    {
      c_check=0;
      cout<<"map_learned_index_paper"<<endl;
      file_name="/spinning/sabek/learned_join_datasets_sosd/planet-170501.lon200M.bin";
      // file_name="data/planetbin";
    }


    if (filename_arg.compare("books64")==0)
    {
      c_check=1;
      cout<<"books64"<<endl;
      file_name="/spinning/sabek/learned_join_datasets_sosd/books_800M_uint64";
    }

    if (filename_arg.compare("fb")==0)
    {
      c_check=1;
      cout<<"fb"<<endl;
      file_name="/spinning/sabek/learned_join_datasets_sosd/fb_200M_uint64";
    }

    if(c_check==0)
    {
      if (filename_arg.compare("books32")==0)
      {
        cout<<"books32"<<endl;
        file_name="/spinning/sabek/learned_join_datasets_sosd/books_200M_uint32";
      }
    }
    std::ifstream input( file_name, std::ios::binary );  
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
    
    cout<<"done reading file"<<endl;

    uint64_t file_elements=0;

    file_elements = buffer[0] | (buffer[1] << 8) | (buffer[2] << 16) | (buffer[3] << 24) | buffer[4+0] << 32 | (buffer[4+1] << 40) | (buffer[4+2] << 48) | (buffer[4+3] << 56); 
    cout<<"value is: "<<file_elements<<" buffer size: "<<buffer.size()<<" c check: "<<c_check<<endl; 
    
    //file_elements=200000000-1;

    vector<uint64_t> v_int(file_elements,0.0);

    if(c_check==1)
    {
      for(uint64_t i=0;i<file_elements;i++)
      {
        uint64_t index=i*8+8;
        v_int[i]=buffer[index+0] | (buffer[index+1] << 8) | (buffer[index+2] << 16) | (buffer[index+3] << 24) | buffer[index+4+0] << 32 | (buffer[index+4+1] << 40) | (buffer[index+4+2] << 48) | (buffer[index+4+3] << 56);
      }
      cout<<"done building array "<<endl;
    }
    else
    {
      for(uint64_t i=0;i<file_elements;i++)
      {
        uint64_t index=i*4+8;
        v_int[i]=buffer[index+0] | (buffer[index+1] << 8) | (buffer[index+2] << 16) | (buffer[index+3] << 24);
      }

      cout<<"done building array "<<endl; 
    }

    cout<<v_int[0]<<" "<<v_int[1]<<" "<<v_int[2]<<" printing scores"<<endl;
    cout<<"last element: "<<v_int[v_int.size()-1]<<endl;
    std::sort(v_int.begin(), v_int.end());
    v_int.erase( unique( v_int.begin(), v_int.end() ), v_int.end() );

    file_elements=v_int.size();

    std::sort(v_int.begin(), v_int.end());
    uint64_t max_val = v_int[v_int.size()-1];
    cout<<"done sorting de duplicating array "<<v_int.size()<<" max value is: "<<v_int[v_int.size()-1]<<" max value is: "<<log2(v_int[v_int.size()-2])<<endl;
    cout<<"num tuples are: "<<rel->num_tuples<<endl;

    for (uint64_t i = 0; i < rel->num_tuples; i++) {
        /*if(c_check == 1)
        {
                //double ratio = (double)(v_int[i] * 1.) / (double)(1. * std::numeric_limits<uint64_t>::max());
                double ratio = static_cast<double>(v_int[i]) / static_cast<double>(max_val);    
                double val = ratio * static_cast<double>(std::numeric_limits<uint32_t>::max());
                rel->tuples[i].key = static_cast<KeyType>(val);//(KeyType)(val);
                if (((i > (rel->num_tuples/2 - 10)) && (i < (rel->num_tuples/2))) || (i > rel->num_tuples - 10)){
                printf("uint64 %lf v_int[i] %lu val %lf key %u ratio %lf \n", static_cast<double>(std::numeric_limits<uint64_t>::max()), v_int[i], val, rel->tuples[i].key, ratio);
                printf("uint64 %lf val %lf key %u \n", static_cast<double>(std::numeric_limits<uint64_t>::max()), val, rel->tuples[i].key);
                }
            #ifdef ZERO_PAYLOAD
                rel->tuples[i].payload = (PayloadType) 0; 
            #else
                rel->tuples[i].payload = (PayloadType)(val);
            #endif
            if (((i > (rel->num_tuples/2 - 10)) && (i < (rel->num_tuples/2))) || (i > rel->num_tuples - 10))
                //printf("uint64 %lf v_int[i] %lf val %lf key %u ratio %lf \n", static_cast<double>(std::numeric_limits<uint64_t>::max()), v_int[i], val, rel->tuples[i].key, ratio);
                printf("again key %u payload %u \n", rel->tuples[i].key, rel->tuples[i].payload);
        }
        else
        {*/
                rel->tuples[i].key = (KeyType)(v_int[i]);

        #ifdef ZERO_PAYLOAD
                rel->tuples[i].payload = (PayloadType) 0; 
        #else
                rel->tuples[i].payload = (PayloadType)(v_int[i]);
        #endif
        /*}*/
    }

    /* randomly shuffle elements */
    knuth_shuffle<KeyType, PayloadType>(rel);
}


//based on the ETH implementation 
template<class KeyType, class PayloadType>
void random_seq_holes_gen(Relation<KeyType, PayloadType> * rel) 
{
    uint64_t i;
    double hole_frac=0.1;

    set<uint32_t> set_uniq;
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<> distribution(0.0, 1.0);

    //cout<<"set started: "<<endl;

    for(uint64_t itr=0;itr<(hole_frac+1.0)*rel->num_tuples;itr++)
    {
      set_uniq.insert(itr);
    }

    //cout<<"set created: "<<endl;

    uint64_t count=set_uniq.size();

    while(count>rel->num_tuples)
    {
      uint32_t temp_val=distribution(generator)*(hole_frac+1.0)*rel->num_tuples;
      if(set_uniq.find(temp_val)!=set_uniq.end())
      {
        set_uniq.erase(temp_val);
        count--;
      }
    }
    //cout<<"set extracted: "<<endl;

    std::set<uint32_t>::iterator it=set_uniq.begin();

    for (i = 0; i < rel->num_tuples; i++) {
        rel->tuples[i].key = (KeyType)(*it+1);

#ifdef ZERO_PAYLOAD
        rel->tuples[i].payload = (PayloadType) 0; 
#else
        rel->tuples[i].payload = (PayloadType)(*it+1);
#endif
        it++;
    }

    /* randomly shuffle elements */
    knuth_shuffle<KeyType, PayloadType>(rel);
}

//based on the ETH implementation 
template<class KeyType, class PayloadType>
void random_uniq_unif_gen(Relation<KeyType, PayloadType> * rel) 
{
    uint64_t i;

    set<uint32_t> set_uniq;
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<> distribution(0.0, 1.0);

    uint64_t count=0;

    //cout<<"unif started"<<endl;

    while(count<rel->num_tuples)
    {
      uint32_t temp_val=distribution(generator)*UINT32_MAX;
      if(set_uniq.find(temp_val)==set_uniq.end())
      {
        set_uniq.insert(temp_val);
        count++;
      }
    }

    //cout<<"unif generation done"<<endl;

    std::set<uint32_t>::iterator it=set_uniq.begin();

    for (i = 0; i < rel->num_tuples; i++) {
        rel->tuples[i].key = (KeyType)(*it+1);

#ifdef ZERO_PAYLOAD
        rel->tuples[i].payload = (PayloadType) 0; 
#else
        rel->tuples[i].payload = (PayloadType)(*it+1);
#endif

        it++;
    }

    //cout<<"unif copying done"<<endl;

    /* randomly shuffle elements */
    knuth_shuffle<KeyType, PayloadType>(rel);
}


//based on the ETH implementation 
template<class KeyType, class PayloadType>
void random_unique_gen(Relation<KeyType, PayloadType> * rel) 
{
    uint64_t i;

    for (i = 0; i < rel->num_tuples; i++) {
        rel->tuples[i].key = (KeyType)(i+1);
#ifdef ZERO_PAYLOAD
        rel->tuples[i].payload = (PayloadType) 0; 
#else
        rel->tuples[i].payload = (PayloadType)(i+1);
#endif
    }

    /* randomly shuffle elements */
    knuth_shuffle<KeyType, PayloadType>(rel);
}

template<class KeyType, class PayloadType>
void input_tuples_gen(Relation<KeyType, PayloadType> * rel, KeyType* input_keys) 
{
    uint64_t i;
    for (i = 0; i < rel->num_tuples; i++) {
        rel->tuples[i].key = (KeyType)(input_keys[i]);

#ifdef ZERO_PAYLOAD
        rel->tuples[i].payload = (PayloadType) 0; 
#else
        rel->tuples[i].payload = (PayloadType)(input_keys[i]);
#endif
    }
}

template<class KeyType, class PayloadType>
int create_eth_workload_relation_pk(Relation<KeyType, PayloadType> *relation, int64_t num_tuples, int relation_padding) 
{
    check_seed();

    relation->num_tuples = num_tuples;
    relation->tuples = (Tuple<KeyType, PayloadType> *) alloc_aligned(num_tuples * sizeof(Tuple<KeyType, PayloadType>) + relation_padding);

    if (!relation->tuples) { 
        perror("out of memory");
        return -1; 
    }
  
    random_unique_gen<KeyType, PayloadType>(relation);
    //random_uniq_unif_gen<KeyType, PayloadType>(relation);
    //random_seq_holes_gen<KeyType, PayloadType>(relation);
    
    //random_real_data_uint_gen(relation, "books32");
    //random_real_data_uint_gen(relation, "books64");
    //random_real_data_uint_gen(relation, "fb");
    //random_real_data_uint_gen(relation, "osm_cellids");
    //random_real_data_uint_gen(relation, "wiki");

    return 0;
}

template<class KeyType, class PayloadType>
int create_input_workload_relation(KeyType* input_keys, Relation<KeyType, PayloadType> * relation, int64_t num_tuples, int relation_padding)
{
    relation->num_tuples = num_tuples;
    relation->tuples = (Tuple<KeyType, PayloadType> *) alloc_aligned(num_tuples * sizeof(Tuple<KeyType, PayloadType>) + relation_padding);

    if (!relation->tuples) { 
        perror("out of memory");
        return -1; 
    }
  
    input_tuples_gen(relation, input_keys);

    return 0;
}

template<class KeyType, class PayloadType>
int create_eth_workload_relation_fk(Relation<KeyType, PayloadType> *relation, int64_t num_tuples, const int64_t maxid, int relation_padding)
{
    int32_t i, iters;
    int64_t remainder;
    Relation<KeyType, PayloadType> tmp;

    check_seed();

    relation->num_tuples = num_tuples;
    relation->tuples = (Tuple<KeyType, PayloadType>*) alloc_aligned(relation->num_tuples * sizeof(Tuple<KeyType, PayloadType>) + relation_padding);
      
    if (!relation->tuples) { 
        perror("out of memory");
        return -1; 
    }
  
    /* alternative generation method */
    iters = num_tuples / maxid;
    for(i = 0; i < iters; i++){
        tmp.num_tuples = maxid;
        tmp.tuples = relation->tuples + maxid * i;
        random_unique_gen<KeyType, PayloadType>(&tmp);
    }

    /* if num_tuples is not an exact multiple of maxid */
    remainder = num_tuples % maxid;
    if(remainder > 0) {
        tmp.num_tuples = remainder;
        tmp.tuples = relation->tuples + maxid * iters;
        random_unique_gen<KeyType, PayloadType>(&tmp);
    }

    return 0;
}

template<class KeyType, class PayloadType>
void normal_distr_gen(Relation<KeyType, PayloadType> * relation, double mean = 0, double stddev = 1)
{
    random_device rd;
    mt19937 generator(rd());
    normal_distribution<> distribution(mean, stddev);

    uint64_t i; 

    for (i = 0; i < relation->num_tuples; i++)
    {
        relation->tuples[i].key = (KeyType)(distribution(generator) * relation->num_tuples);
#ifdef ZERO_PAYLOAD
        relation->tuples[i].payload = (PayloadType)0;
#else
        relation->tuples[i].payload = (PayloadType)(relation->tuples[i].key);
#endif
    }
}

template<class KeyType, class PayloadType>
void uniform_distr_gen(Relation<KeyType, PayloadType> * relation, double a = 0, double b = 1)
{
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<> distribution(a, b);

    uint64_t i; 

    for (i = 0; i < relation->num_tuples; i++)
    {
        relation->tuples[i].key = (KeyType)(distribution(generator) * relation->num_tuples);
#ifdef ZERO_PAYLOAD
        relation->tuples[i].payload = (PayloadType)0;
#else
        relation->tuples[i].payload = (PayloadType)(relation->tuples[i].key);
#endif
    }
}

template<class KeyType, class PayloadType>
void exponential_distr_gen(Relation<KeyType, PayloadType> * relation, double lambda = 2, double scale = 1e6)
{
    random_device rd;
    mt19937 generator(rd());
    exponential_distribution<> distribution(lambda);

    uint64_t i; 

    for (i = 0; i < relation->num_tuples; i++)
    {
        relation->tuples[i].key = (KeyType)(distribution(generator) * scale);
#ifdef ZERO_PAYLOAD
        relation->tuples[i].payload = (PayloadType)0;
#else
        relation->tuples[i].payload = (PayloadType)(relation->tuples[i].key);
#endif
    }
}

template<class KeyType, class PayloadType>
void lognormal_distr_gen(Relation<KeyType, PayloadType> * relation, double mean = 0, double stddev = 1, double scale = 1e6)
{
    random_device rd;
    mt19937 generator(rd());
    lognormal_distribution<> distribution(mean, stddev);

    uint64_t i; 

    for (i = 0; i < relation->num_tuples; i++)
    {
        relation->tuples[i].key = (KeyType)(distribution(generator) * scale);
#ifdef ZERO_PAYLOAD
        relation->tuples[i].payload = (PayloadType)0;
#else
        relation->tuples[i].payload = (PayloadType)(relation->tuples[i].key);
#endif
    }
}

template<class KeyType, class PayloadType>
int create_synthetic_workload_relation_fk(Relation<KeyType, PayloadType> *relation, int64_t num_tuples, synthetic_workload_distr_t data_distn_type, DataDistnParams* data_distn_params, int relation_padding) 
{
    relation->num_tuples = num_tuples;
    relation->tuples = (Tuple<KeyType, PayloadType> *) alloc_aligned(num_tuples * sizeof(Tuple<KeyType, PayloadType>) + relation_padding);

    if (!relation->tuples) { 
        perror("out of memory");
        return -1; 
    }

    switch (data_distn_type) {
      case NORMAL:
        normal_distr_gen<KeyType, PayloadType>(relation, data_distn_params->normal_mean, data_distn_params->normal_stddev);
        break;
      case UNIFORM:
        uniform_distr_gen<KeyType, PayloadType>(relation, data_distn_params->uniform_a, data_distn_params->uniform_b);
        break;
      case EXPONENTIAL:
        exponential_distr_gen<KeyType, PayloadType>(relation, data_distn_params->exponential_lambda, data_distn_params->exponential_scale);
        break;
      case LOGNORMAL:
        lognormal_distr_gen<KeyType, PayloadType>(relation, data_distn_params->lognormal_mean, data_distn_params->lognormal_stddev, data_distn_params->lognormal_scale);
        break;
      default:
        perror("undefined data distribution type");
        return -1;
    }

    return 0;
}

template<class KeyType, class PayloadType>
void * numa_localize_thread(void * args) 
{
    create_arg_t<KeyType, PayloadType> * arg = (create_arg_t<KeyType, PayloadType> *) args;
    Relation<KeyType, PayloadType> *   rel = & arg->rel;
    uint64_t i;
    
    for (i = 0; i < rel->num_tuples; i++) {
        rel->tuples[i].key = 0;
    }

    return 0;
}

template<class Type>
void * numa_localize_generic_thread(void * args) 
{
    create_arg_generic_t<Type> * arg = (create_arg_generic_t<Type> *) args;
    Type *   rel = arg->rel;
    uint64_t num_elems = arg->num_elems;
    uint64_t i;
    
    for (i = 0; i < num_elems; i++) {
        rel[i] = 0;
    }

    return 0;
}

template<class KeyType, class PayloadType>
int numa_localize(Tuple<KeyType, PayloadType> * relation, int64_t num_tuples, uint32_t nthreads)
{
    uint32_t i, rv;
    uint64_t offset = 0;

    /* we need aligned allocation of items */
    create_arg_t<KeyType, PayloadType> args[nthreads];
    pthread_t tid[nthreads];
    cpu_set_t set;
    pthread_attr_t attr;

    unsigned int pagesize;
    unsigned int npages;
    unsigned int npages_perthr;
    uint64_t ntuples_perthr;
    uint64_t ntuples_lastthr;

    pagesize        = getpagesize();
    npages          = (num_tuples * sizeof(Tuple<KeyType, PayloadType>)) / pagesize + 1;
    npages_perthr   = npages / nthreads;
    ntuples_perthr  = npages_perthr * (pagesize/sizeof(Tuple<KeyType, PayloadType>));
    ntuples_lastthr = num_tuples - ntuples_perthr * (nthreads-1);

    pthread_attr_init(&attr);

    for( i = 0; i < nthreads; i++ ) {
    #ifdef DEVELOPMENT_MODE
    int cpu_idx = get_cpu_id_develop(i);
    #else
    int cpu_idx = get_cpu_id_v2(i);
    #endif
       
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

        args[i].firstkey       = offset + 1;
        args[i].rel.tuples     = relation + offset;
        args[i].rel.num_tuples = (i == nthreads-1) ? ntuples_lastthr 
                                 : ntuples_perthr;
        offset += ntuples_perthr;

        rv = pthread_create(&tid[i], &attr, numa_localize_thread<KeyType, PayloadType>, (void*)&args[i]);
        if (rv){
            fprintf(stderr, "[ERROR] pthread_create() return code is %d\n", rv);
            exit(-1);
        }
    }

    for(i = 0; i < nthreads; i++){
        pthread_join(tid[i], NULL);
    }

    return 0;

}

template<class KeyType, class PayloadType>
int numa_localize_varlen(Tuple<KeyType, PayloadType> * relation, int64_t* num_tuples_for_threads, uint32_t nthreads)
{
    uint32_t i, rv;
    uint64_t offset = 0;

    /* we need aligned allocation of items */
    create_arg_t<KeyType, PayloadType> args[nthreads];
    pthread_t tid[nthreads];
    cpu_set_t set;
    pthread_attr_t attr;

/*
    unsigned int pagesize;
    unsigned int npages_perthr;
    uint64_t ntuples_perthr [nthreads];

    pagesize        = getpagesize();
    for(i = 0; i < nthreads; i++)
    {
        npages_perthr = (num_tuples_for_threads[i] * sizeof(Tuple<KeyType, PayloadType>)) / pagesize + 1;
        ntuples_perthr[i]  = npages_perthr * (pagesize/sizeof(Tuple<KeyType, PayloadType>));
    }
*/
    pthread_attr_init(&attr);

    for( i = 0; i < nthreads; i++ ) {
        #ifdef DEVELOPMENT_MODE
        int cpu_idx = get_cpu_id_develop(i);
        #else
        int cpu_idx = get_cpu_id_v2(i);
        #endif

        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

        args[i].firstkey       = offset + 1;
        args[i].rel.tuples     = relation + offset;
        args[i].rel.num_tuples = num_tuples_for_threads[i];

        offset += num_tuples_for_threads[i];

        rv = pthread_create(&tid[i], &attr, numa_localize_thread<KeyType, PayloadType>, (void*)&args[i]);
        if (rv){
            fprintf(stderr, "[ERROR] pthread_create() return code is %d\n", rv);
            exit(-1);
        }
    }

    for(i = 0; i < nthreads; i++){
        pthread_join(tid[i], NULL);
    }

    return 0;

}


template<class Type>
int numa_localize_generic_varlen(Type * relation, int64_t* num_tuples_for_threads, uint32_t nthreads)
{
    uint32_t i, rv;
    uint64_t offset = 0;

    /* we need aligned allocation of items */
    create_arg_generic_t<Type> args[nthreads];
    pthread_t tid[nthreads];
    cpu_set_t set;
    pthread_attr_t attr;

    pthread_attr_init(&attr);

    for( i = 0; i < nthreads; i++ ) {
        #ifdef DEVELOPMENT_MODE
        int cpu_idx = get_cpu_id_develop(i);
        #else
        int cpu_idx = get_cpu_id_v2(i);
        #endif

        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

        args[i].firstkey       = offset + 1;
        args[i].rel     = relation + offset;
        args[i].num_elems = num_tuples_for_threads[i];

        offset += num_tuples_for_threads[i];

        rv = pthread_create(&tid[i], &attr, numa_localize_generic_thread<Type>, (void*)&args[i]);
        if (rv){
            fprintf(stderr, "[ERROR] pthread_create() return code is %d\n", rv);
            exit(-1);
        }
    }

    for(i = 0; i < nthreads; i++){
        pthread_join(tid[i], NULL);
    }

    return 0;

}
