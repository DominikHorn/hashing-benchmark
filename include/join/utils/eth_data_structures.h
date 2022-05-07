#pragma once

/* More data structures used among all join algorithms */

#include <unordered_map> 

#include "configs/eth_configs.h"
#include "utils/math.h"
#include "utils/memory.h"
#include "utils/data_structures.h"
#include "utils/eth_generic_task_queue.h"
#include "utils/learned_sort_for_sort_merge.h"

 #ifdef INLJ_WITH_HASH_INDEX
        #ifdef HASH_SCHEME_AND_FUNCTION_MODE
            #ifdef CHAINTRADITIONAL
                #include "chained.hpp"
            #endif
            #ifdef CHAINLINEARMODEL
                #include "chained_model.hpp"
            #endif
            #ifdef CHAINEXOTIC
                #include "chained_exotic.hpp"
            #endif            
            #ifdef PROBETRADITIONAL
                #include "probe.hpp"
            #endif
            #ifdef PROBELINEARMODEL
                #include "probe_model.hpp"
            #endif
            #ifdef CUCKOOTRADITIONAL
                #include "cuckoo.hpp"
            #endif
            #ifdef CUCKOOLINEARMODEL
                #include "cuckoo_model.hpp"
            #endif            
                //TODO
        #endif
    #endif

#ifdef INLJ_WITH_CSS_TREE_INDEX
#include "utils/CC_CSSTree.h"
#endif

#ifdef INLJ_WITH_ART32_TREE_INDEX
#include "utils/art32_tree.h"
#endif

#ifdef INLJ_WITH_ART64_TREE_INDEX
#include "utils/art64_tree.h"
#endif


#ifdef INLJ_WITH_CUCKOO_HASH_INDEX
#include "utils/stanford_hash.h"
#endif

using namespace learned_sort_for_sort_merge;

/*********** Common data structures for ETH radix join ***********/
template<typename KeyType, typename PayloadType, typename TaskType>
struct ETHRadixJoinThread : JoinThreadBase<KeyType, PayloadType, TaskType> {
    int32_t ** histR;
    Tuple<KeyType, PayloadType> *  tmpR;
    int32_t ** histS;
    Tuple<KeyType, PayloadType> *  tmpS;
#ifdef SKEW_HANDLING
    TaskQueue<KeyType, PayloadType, TaskType> *      skew_queue;
    Task<KeyType, PayloadType> **        skewtask;
#endif
    /* stats about the thread */
    int32_t        parts_processed;

    /**** start stuff for learning RMI models ****/
    learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi;
    typename learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params p;
    Relation<KeyType, PayloadType> *     original_relR;
    Relation<KeyType, PayloadType> *     original_relS;
    Tuple<KeyType, PayloadType> * tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * sorted_training_sample_in;
    Tuple<KeyType, PayloadType> * r_tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * r_sorted_training_sample_in;
    Tuple<KeyType, PayloadType> * s_tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * s_sorted_training_sample_in;
    vector<vector<vector<training_point<KeyType, PayloadType>>>> * training_data;
    uint32_t tmp_training_sample_R_offset, tmp_training_sample_S_offset, tmp_training_sample_offset;
    uint32_t * sample_count, * sample_count_R, * sample_count_S;
    vector<double>* slopes; 
    vector<double>* intercepts;
    /**** end stuff for learning RMI models ****/

} __attribute__((aligned(CACHE_LINE_SIZE)));


template<typename KeyType, typename PayloadType, typename TaskType, typename JoinThreadType>
struct ETHPartition : PartitionBase<KeyType, PayloadType, TaskType, JoinThreadType> {
    Tuple<KeyType, PayloadType> *  tmp;
    int32_t ** hist;
    int64_t *  output;
    int32_t    R;
    uint32_t   D;
    uint32_t   padding;
} __attribute__((aligned(CACHE_LINE_SIZE)));

struct ETHBucketChainingBuild : BuildBase {
    uint32_t * next;
};

/**************** Common data structures for ETH non partition hash join ******************/

/*********** Hashtable and its buckets ******************************/
/********************************************************************/
#if PADDED_BUCKET==0
/** 
 * Normal hashtable buckets.
 *
 * if KEY_8B then key is 8B and sizeof(bucket_t) = 48B
 * else key is 16B and sizeof(bucket_t) = 32B
 */
template<typename KeyType, typename PayloadType>
struct Bucket {
    volatile char     latch;
    /* 3B hole */
    uint32_t          count;
    Bucket<KeyType, PayloadType> * next;
    Tuple<KeyType, PayloadType>  tuples[BUCKET_SIZE];
};
#else /* PADDED_BUCKET: bucket is padded to cache line size */
/** 
 * Cache-sized bucket where size of the bucket is padded
 * to cache line size (64B). 
 */
template<typename KeyType, typename PayloadType>
struct Bucket {
    volatile char     latch;
    /* 3B hole */
    uint32_t          count;
    Tuple<KeyType, PayloadType>  tuples[BUCKET_SIZE];
    Bucket<KeyType, PayloadType> * next;
} __attribute__ ((aligned(CACHE_LINE_SIZE)));
#endif /* PADDED_BUCKET */

/** Hashtable structure for NPO. */
template<typename KeyType, typename PayloadType>
struct Hashtable {
    Bucket<KeyType, PayloadType> * buckets;
    int32_t    num_buckets;
    uint32_t   hash_mask;
    uint32_t   skip_bits;
};

/** Pre-allocated bucket buffers are used for overflow-buckets. */
template<typename KeyType, typename PayloadType>
struct BucketBuffer {
    BucketBuffer<KeyType, PayloadType> * next;
    uint32_t count;
    Bucket<KeyType, PayloadType> buf[OVERFLOW_BUF_SIZE];
};

template<typename KeyType, typename PayloadType>
void 
init_bucket_buffer(BucketBuffer<KeyType, PayloadType> ** ppbuf)
{
    BucketBuffer<KeyType, PayloadType> * overflowbuf;
    overflowbuf = (BucketBuffer<KeyType, PayloadType>*) malloc(sizeof(BucketBuffer<KeyType, PayloadType>));
    overflowbuf->count = 0;
    overflowbuf->next  = NULL;

    *ppbuf = overflowbuf;
}

template<typename KeyType, typename PayloadType>
static inline void 
get_new_bucket(Bucket<KeyType, PayloadType> ** result, BucketBuffer<KeyType, PayloadType> ** buf)
{
    if((*buf)->count < OVERFLOW_BUF_SIZE) {
        *result = (*buf)->buf + (*buf)->count;
        (*buf)->count ++;
    }
    else {
        /* need to allocate new buffer */
        BucketBuffer<KeyType, PayloadType> * new_buf = (BucketBuffer<KeyType, PayloadType>*) 
                                                        malloc(sizeof(BucketBuffer<KeyType, PayloadType>));
        new_buf->count = 1;
        new_buf->next  = *buf;
        *buf    = new_buf;
        *result = new_buf->buf;
    }
}

template<typename KeyType, typename PayloadType>
void
free_bucket_buffer(BucketBuffer<KeyType, PayloadType> * buf)
{
    do {
        BucketBuffer<KeyType, PayloadType> * tmp = buf->next;
        free(buf);
        buf = tmp;
    } while(buf);
}

template<typename KeyType, typename PayloadType>
void 
allocate_hashtable(Hashtable<KeyType, PayloadType> ** ppht, uint32_t nbuckets)
{
    Hashtable<KeyType, PayloadType> * ht;

    ht              = (Hashtable<KeyType, PayloadType>*)malloc(sizeof(Hashtable<KeyType, PayloadType>));
    ht->num_buckets = nbuckets;
    NEXT_POW_2((ht->num_buckets));

    /* allocate hashtable buckets cache line aligned */
    if (posix_memalign((void**)&ht->buckets, CACHE_LINE_SIZE,
                       ht->num_buckets * sizeof(Bucket<KeyType, PayloadType>))){
        perror("Aligned allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    memset(ht->buckets, 0, ht->num_buckets * sizeof(Bucket<KeyType, PayloadType>));
    ht->skip_bits = 0; /* the default for modulo hash */
    ht->hash_mask = (ht->num_buckets - 1) << ht->skip_bits;
    *ppht = ht;
}

template<typename KeyType, typename PayloadType>
void 
destroy_hashtable(Hashtable<KeyType, PayloadType> * ht)
{
    free(ht->buckets);
    free(ht);
}

#define EVAL(Rhs, Tmp) { Rhs->key = Tmp->key;Rhs->payload = Tmp->payload;}

template<typename KeyType, typename PayloadType>
struct LinkedList
{
//attributes
	Bucket<KeyType, PayloadType>* header;
	Bucket<KeyType, PayloadType>* newBucket;
	int curEntry;
	int numBuckets;
//methods,

	void init()
	{
		newBucket=header=(Bucket<KeyType, PayloadType>*)malloc(sizeof(Bucket<KeyType, PayloadType>));
		header->next = NULL;		
		curEntry=0;
		numBuckets=0;
	}

	void destroy()
	{
		Bucket<KeyType, PayloadType> *b=header;
		Bucket<KeyType, PayloadType> *bnext=header;
		while(bnext!=NULL)
		{
			bnext=b->next;
			free(b);
			b=bnext;
		}
	}

	int size()
	{
		return (numBuckets*BUCKET_SIZE)+curEntry;
	}

	void fill(Tuple<KeyType, PayloadType> * r)
	{
		if(curEntry==BUCKET_SIZE)
		{
			if(newBucket->next==NULL)
			{
				newBucket->next=(Bucket<KeyType, PayloadType>*)malloc(sizeof(Bucket<KeyType, PayloadType>));
				newBucket->next->next = NULL;
			}
			newBucket=newBucket->next;
			EVAL((newBucket->tuples),r);
			curEntry=1;
			numBuckets++;
		}
		else
		{	
			EVAL((newBucket->tuples+curEntry),r);
			curEntry++;
		}
	}

	void copyToArray(Tuple<KeyType, PayloadType>* Rout)
	{
		/*Bucket<KeyType, PayloadType> *sBucket=header;
		int i=0;
		int cur=0;
		while(sBucket->next!=NULL)
		{
			for(i=0;i<BUCKET_SIZE;i++)
			{
				EVAL((Rout+cur),(sBucket->tuples+i));
				cur++;
			}
			sBucket=sBucket->next;
		}
		//the last bucket
		for(i=0;i<curEntry;i++)
		{
			EVAL((Rout+cur),(sBucket->tuples+i));
			cur++;
		}*/
		//using memory copy
		Bucket<KeyType, PayloadType> *sBucket=header;
		int i=0;
		int cur=0;
		int bucketSizeInBytes=BUCKET_SIZE*sizeof(Tuple<KeyType, PayloadType>);
		while(sBucket->next!=NULL)
		{
			memcpy(Rout+cur,sBucket->tuples,bucketSizeInBytes);
			cur=cur+BUCKET_SIZE;
			sBucket=sBucket->next;
		}
		//the last bucket
		memcpy(Rout+cur,sBucket->tuples,curEntry*sizeof(Tuple<KeyType, PayloadType>));
		//cur=cur+curEntry;
	}
	int print()
	{
		Bucket<KeyType, PayloadType> *sBucket=header;
		int i=0;
		int cur=0;
		while(sBucket->next!=NULL)
		{
			for(i=0;i<BUCKET_SIZE;i++)
			{
				printf("%d; ",sBucket->tuples[i].key);
				cur++;
			}
			sBucket=sBucket->next;
		}
		//the last bucket
		for(i=0;i<curEntry;i++)
		{
			printf("%d, ",(sBucket->tuples+i)->key);
			cur++;
		}
		return cur;
	}

	static void test()
	{		
		LinkedList<KeyType, PayloadType>* ll=(LinkedList<KeyType, PayloadType>*)malloc(sizeof(LinkedList<KeyType, PayloadType>));
		ll->init();		
		Tuple<KeyType, PayloadType> *r=(Tuple<KeyType, PayloadType> *)malloc(sizeof(Tuple<KeyType, PayloadType>));
		int i=0;		
		for(i=0;i<65;i++)
		{			
			r->key=i;
			ll->fill(r);
		}
	
		ll->print();
		ll->destroy();
	}
};

/***************************************************************/

template<typename KeyType, typename PayloadType, typename TaskType>
struct ETHNonPartitionJoinThread {
    int32_t             tid;
    Hashtable<KeyType, PayloadType> *  ht;
    Relation<KeyType, PayloadType>     relR;
    Relation<KeyType, PayloadType>     relS;
    pthread_barrier_t * barrier;
    int64_t             num_results = 0;

    /* results of the thread */
    ThreadResult * threadresult;
    
    /**** start stuff for learning RMI models ****/
    learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi;
    typename learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params p;
    Relation<KeyType, PayloadType> *     original_relR;
    Relation<KeyType, PayloadType> *     original_relS;
    Tuple<KeyType, PayloadType> * tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * sorted_training_sample_in;
    Tuple<KeyType, PayloadType> * r_tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * r_sorted_training_sample_in;
    Tuple<KeyType, PayloadType> * s_tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * s_sorted_training_sample_in;
    vector<vector<vector<training_point<KeyType, PayloadType>>>> * training_data;
    uint32_t tmp_training_sample_R_offset, tmp_training_sample_S_offset, tmp_training_sample_offset;
    uint32_t * sample_count, * sample_count_R, * sample_count_S;
    vector<double>* slopes; 
    vector<double>* intercepts;
    /**** end stuff for learning RMI models ****/

    /* stats about the thread */
    struct timeval start_time, partition_end_time, end_time;
#ifndef DEVELOPMENT_MODE
    //PerfEvent e_start_to_partition, e_partition_to_end;
#endif
#ifdef DEVELOPMENT_MODE
    unordered_map<uint64_t, uint64_t> * build_hash_bucket_visits;
    unordered_map<uint64_t, uint64_t> * probe_hash_bucket_visits;
    volatile char *    keys_hash_latch;
    vector<KeyType> * build_keys_list;
    vector<uint64_t> * build_keys_hash_list;
    vector<KeyType> * probe_keys_list;
    vector<uint64_t> * probe_keys_hash_list;    
#endif     
} __attribute__((aligned(CACHE_LINE_SIZE)));

template<typename KeyType, typename PayloadType, typename TaskType>
struct JoinThread 
{
    int32_t             tid;
    Relation<KeyType, PayloadType>     relR;
    Relation<KeyType, PayloadType>     relS;
    Relation<KeyType, PayloadType> *     original_relR;
    Relation<KeyType, PayloadType> *     original_relS;
    pthread_barrier_t * barrier;
    int64_t             num_results = 0;

    /* results of the thread */
    ThreadResult * threadresult;

    #ifdef INLJ_WITH_HASH_INDEX
        #ifdef HASH_SCHEME_AND_FUNCTION_MODE
            void* ht;
        #else
            Hashtable<KeyType, PayloadType> *  ht;
        #endif
    #endif

    KeyType * sorted_relation_r_keys_only;

    #ifdef INLJ_WITH_LEARNED_INDEX
    /**** start stuff for learning RMI models ****/
    /*learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi;
    typename learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params p;
    Tuple<KeyType, PayloadType> * tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * sorted_training_sample_in;
    Tuple<KeyType, PayloadType> * r_tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * r_sorted_training_sample_in;
    Tuple<KeyType, PayloadType> * s_tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * s_sorted_training_sample_in;
    vector<vector<vector<training_point<KeyType, PayloadType>>>> * training_data;
    uint32_t tmp_training_sample_R_offset, tmp_training_sample_S_offset, tmp_training_sample_offset;
    uint32_t * sample_count, * sample_count_R, * sample_count_S;
    vector<double>* slopes; 
    vector<double>* intercepts;*/
    /**** end stuff for learning RMI models ****/
    #ifdef INLJ_WITH_LEARNED_INDEX_MODEL_BASED_BUILD 
        KeyType * sorted_relation_r_gapped_keys_only;
        uint64_t sorted_relation_r_gapped_keys_only_size;
    #endif
    #endif

#ifdef INLJ_WITH_CSS_TREE_INDEX
    CC_CSSTree<KeyType, PayloadType> *tree;
#endif

#ifdef INLJ_WITH_ART32_TREE_INDEX
    ART32<PayloadType> *art32_tree;
#endif

#ifdef INLJ_WITH_ART64_TREE_INDEX
    ART<PayloadType> *art64_tree;
#endif

#ifdef INLJ_WITH_CUCKOO_HASH_INDEX
    CuckooHashMap<PayloadType> *cuckoo_hashmap;
#endif

    /* stats about the thread */
    struct timeval start_time, partition_end_time, end_time;
} __attribute__((aligned(CACHE_LINE_SIZE)));


template<typename KeyType, typename PayloadType>
struct ETHNonPartitionJoinBuild {
    Hashtable<KeyType, PayloadType> * ht;
    BucketBuffer<KeyType, PayloadType> ** overflowbuf;
    learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi;
    vector<double>* slopes;
    vector<double>* intercepts;
#ifdef DEVELOPMENT_MODE
    unordered_map<uint64_t, uint64_t> * build_hash_bucket_visits;
    unordered_map<uint64_t, uint64_t> * probe_hash_bucket_visits;
    volatile char *    keys_hash_latch;    
    vector<KeyType> * build_keys_list;
    vector<uint64_t> * build_keys_hash_list;
    vector<KeyType> * probe_keys_list;
    vector<uint64_t> * probe_keys_hash_list; 
#endif    
};

template<typename KeyType, typename PayloadType>
struct JoinBuild {
#ifdef INLJ_WITH_HASH_INDEX
    #ifdef HASH_SCHEME_AND_FUNCTION_MODE
        void* ht;
    #else
        Hashtable<KeyType, PayloadType> * ht;
        BucketBuffer<KeyType, PayloadType> ** overflowbuf;
    #endif
#endif
#ifdef INLJ_WITH_LEARNED_INDEX
    /*learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi;
    vector<double>* slopes;
    vector<double>* intercepts;*/
#endif
#ifdef INLJ_WITH_CSS_TREE_INDEX
    CC_CSSTree<KeyType, PayloadType> *tree;
#endif

#ifdef INLJ_WITH_ART32_TREE_INDEX
    ART32<PayloadType> *art32_tree;
#endif

#ifdef INLJ_WITH_ART64_TREE_INDEX
    ART<PayloadType> *art64_tree;
#endif

#ifdef INLJ_WITH_CUCKOO_HASH_INDEX
    CuckooHashMap<PayloadType> *cuckoo_hashmap;
#endif

    KeyType * sorted_relation_r_keys_only;
    #ifdef INLJ_WITH_LEARNED_INDEX_MODEL_BASED_BUILD 
    KeyType * sorted_relation_r_gapped_keys_only;
    uint64_t sorted_relation_r_gapped_keys_only_size;
    #endif    
    Relation<KeyType, PayloadType> *     original_relR;
    Relation<KeyType, PayloadType> *     original_relS;
};

/**************** Common data structures for ETH non partition hash join ******************/

/**
 * Various NUMA shuffling strategies for data shuffling phase of join
 * algorithms as also described by NUMA-aware data shuffling paper [CIDR'13].
 *
 * NUMA_SHUFFLE_RANDOM, NUMA_SHUFFLE_RING, NUMA_SHUFFLE_NEXT
 */
enum numa_strategy_t {RANDOM, RING, NEXT};

// Join configuration parameters.
struct joinconfig_t {
    int NTHREADS;
    int PARTFANOUT;
    int LEARNEDSORT;
    int SCALARMERGE;
    int MWAYMERGEBUFFERSIZE;
    enum numa_strategy_t NUMASTRATEGY;
};

template<typename KeyType, typename PayloadType>
struct ETHSortMergeMultiwayJoinThread {
    Tuple<KeyType, PayloadType> *  relR;
    Tuple<KeyType, PayloadType> *  relS;

    // temporary relations for partitioning output 
    Tuple<KeyType, PayloadType> *  tmp_partR;
    Tuple<KeyType, PayloadType> *  tmp_partS;

    // temporary relations for sorting output
    Tuple<KeyType, PayloadType> *  tmp_sortR;
    Tuple<KeyType, PayloadType> *  tmp_sortS;

    int32_t numR;
    int32_t numS;

    int32_t my_tid;
    int     nthreads;

     // join configuration parameters:
    joinconfig_t * joincfg;

    pthread_barrier_t * barrier;
    int64_t result;

    RelationPair<KeyType, PayloadType> ** threadrelchunks;

    // used for multi-way merging, shared by active threads in each NUMA.
    Tuple<KeyType, PayloadType> ** sharedmergebuffer;

    // arguments specific to mpsm-join:
    uint32_t ** histR;
    //Tuple<KeyType, PayloadType> * tmpRglobal;
    //uint64_t totalR;

    ThreadResult * threadresult;

#ifdef SKEW_HANDLING
    // skew handling task queues (1 per NUMA region).
    taskqueue_t ** numa_taskqueues;
    pthread_mutex_t* numa_taskqueues_locks;
    int* is_numa_taskqueues_created;
#endif

    struct timeval start_time, partition_end_time, sort_end_time, tmp_sort_end_time, multiwaymerge_end_time, mergejoin_end_time, tmp_mergejoin_end_time;
#ifndef DEVELOPMENT_MODE
    //PerfEvent e_start_to_partition, e_partition_to_sort, e_sort_to_multiwaymerge, e_multiwaymerge_to_mergejoin;
#endif     

}__attribute__((aligned(CACHE_LINE_SIZE)));

template<typename KeyType, typename PayloadType>
struct MergeNode {
    Tuple<KeyType, PayloadType> * buffer;
    volatile uint32_t count;
    volatile uint32_t head;
    volatile uint32_t tail;
} __attribute__((packed));

// This is a struct used for representing merge tasks when skew handling 
//    mechanism is enabled. Essentially, large merge tasks are decomposed into
//    smaller merge tasks and placed into a task queue.
template<typename KeyType, typename PayloadType>
struct MergeTask {
    Tuple<KeyType, PayloadType> * output;
    Relation<KeyType, PayloadType> ** runstomerge;
    int numruns;
    unsigned int totaltuples;
    // if heavy-hitter then not merged, directly copied
    int isheavyhitter; 
};

template<typename KeyType, typename PayloadType>
struct LearnedSortMergeMultiwayJoinThread : ETHSortMergeMultiwayJoinThread<KeyType, PayloadType> 
{
    int32_t numR_to_be_partitioned;
    int32_t numS_to_be_partitioned;

    Tuple<KeyType, PayloadType> * tmp_minor_bckts_r;
    Tuple<KeyType, PayloadType> * tmp_minor_bckts_s;
    int64_t * tmp_minor_bckt_sizes_r;
    int64_t * tmp_minor_bckt_sizes_s;
    Tuple<KeyType, PayloadType> * tmp_spill_bucket_r;
    Tuple<KeyType, PayloadType> * sorted_spill_bucket_r;
    Tuple<KeyType, PayloadType> * tmp_spill_bucket_s;
    Tuple<KeyType, PayloadType> * sorted_spill_bucket_s;
    Tuple<KeyType, PayloadType> ** tmp_repeatedKeysPredictedRanksR;
    Tuple<KeyType, PayloadType> ** tmp_repeatedKeysPredictedRanksS;
    int64_t ** tmp_repeatedKeysPredictedRanksCountsR;
    int64_t ** tmp_repeatedKeysPredictedRanksCountsS;
    int64_t tmp_repeatedKeysCountsR;
    int64_t tmp_repeatedKeysCountsS;
    int64_t tmp_total_repeatedKeysCountsR;
    int64_t tmp_total_repeatedKeysCountsS;
    unsigned int NUM_MINOR_BCKT_PER_MAJOR_BCKT_r;
    unsigned int NUM_MINOR_BCKT_PER_MAJOR_BCKT_s;
    unsigned int MINOR_BCKTS_OFFSET_r;
    unsigned int MINOR_BCKTS_OFFSET_s;
    unsigned int TOT_NUM_MINOR_BCKTS_r;
    unsigned int TOT_NUM_MINOR_BCKTS_s;
    unsigned int INPUT_SZ_r;
    unsigned int INPUT_SZ_s;

    int64_t tmp_r_partition_offset;
    int64_t tmp_s_partition_offset;
    int64_t tmp_r_repeated_keys_offset;
    int64_t tmp_s_repeated_keys_offset;    
    Tuple<KeyType, PayloadType> ** tmp_partR_arr;
    Tuple<KeyType, PayloadType> ** tmp_partS_arr;
    uint64_t* major_bckt_size_r_arr;
    uint64_t* major_bckt_size_s_arr;
    int64_t* tmp_total_repeatedKeysCountsR_arr;
    int64_t* tmp_total_repeatedKeysCountsS_arr;
    Tuple<KeyType, PayloadType> * tmp_major_bckts_r;
    Tuple<KeyType, PayloadType> * tmp_major_bckts_s;

    /**** start stuff for learning RMI models ****/
    learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi;
    learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi_r;
    learned_sort_for_sort_merge::RMI<KeyType, PayloadType> * rmi_s;
    typename learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params p;
    typename learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params r_p;
    typename learned_sort_for_sort_merge::RMI<KeyType, PayloadType>::Params s_p;
    Relation<KeyType, PayloadType> *     original_relR;
    Relation<KeyType, PayloadType> *     original_relS;
    Tuple<KeyType, PayloadType> * tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * sorted_training_sample_in;
    Tuple<KeyType, PayloadType> * r_tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * r_sorted_training_sample_in;
    Tuple<KeyType, PayloadType> * s_tmp_training_sample_in;
    Tuple<KeyType, PayloadType> * s_sorted_training_sample_in;
    vector<vector<vector<training_point<KeyType, PayloadType>>>> * training_data;
    vector<vector<vector<training_point<KeyType, PayloadType>>>> * r_training_data;
    vector<vector<vector<training_point<KeyType, PayloadType>>>> * s_training_data;
    uint32_t tmp_training_sample_R_offset, tmp_training_sample_S_offset, tmp_training_sample_offset;
    uint32_t * sample_count, * sample_count_R, * sample_count_S;
    vector<double>* slopes; vector<double>* intercepts;
    vector<double>* r_slopes; vector<double>* r_intercepts;
    vector<double>* s_slopes; vector<double>* s_intercepts;
    /**** end stuff for learning RMI models ****/


    struct timeval sample_end_time;
#ifndef DEVELOPMENT_MODE
    //PerfEvent e_start_to_sample, e_sample_to_partition, e_sort_to_mergejoin;
#endif
} __attribute__((aligned(CACHE_LINE_SIZE)));


template<typename KeyType, typename PayloadType>
void 
init_models_training_data_and_sample_counts(vector<vector<vector<training_point<KeyType, PayloadType>>>> * training_data, 
                    vector<unsigned int> arch, uint32_t * sample_count, uint32_t * sample_count_R, 
                    uint32_t * sample_count_S, int num_threads)
{
    training_data->clear();
    training_data->resize(arch.size());
    for (unsigned int layer_idx = 0; layer_idx < arch.size(); ++layer_idx) {
        (*training_data)[layer_idx].resize(arch[layer_idx]);
    }

    for(int i = 0; i < num_threads; i++)
    {
        sample_count[i] = 0;
#ifdef BUILD_RMI_FROM_TWO_DATASETS
        sample_count_R[i] = 0;
        sample_count_S[i] = 0;
#endif
    }
}

void 
free_models_sample_counts(uint32_t * sample_count, uint32_t * sample_count_R, uint32_t * sample_count_S)
{
    free(sample_count);
    free(sample_count_R);
    free(sample_count_S);
}
