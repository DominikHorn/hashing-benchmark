#pragma once

/* Another set of common configs among all join algorithms (old version) (NOT USED NOW) */


#include "configs/base_configs.h"

/************* Common configs for ETH radix join algorithms **************/
/** number of total radix bits used for partitioning. */
#ifndef NUM_RADIX_BITS
#define NUM_RADIX_BITS 10 //18 14
#endif

/** number of passes in multipass partitioning, currently fixed at 2. */
#ifndef NUM_PASSES
#define NUM_PASSES 1  //2
#endif

/** number of probe items for prefetching: must be a power of 2 */
#ifndef PROBE_BUFFER_SIZE
#define PROBE_BUFFER_SIZE 4
#endif

/** \internal some padding space is allocated for relations in order to
 *  avoid L1 conflict misses and PADDING_TUPLES is placed between 
 *  partitions in pass-1 of partitioning and SMALL_PADDING_TUPLES is placed
 *  between partitions in pass-2 of partitioning. 3 is a magic number. 
 */

#define PASS1RADIXBITS (NUM_RADIX_BITS/NUM_PASSES) 
#define PASS2RADIXBITS (NUM_RADIX_BITS-(NUM_RADIX_BITS/NUM_PASSES))

/* num-parts at pass-1 */
#define FANOUT_PASS1 (1 << (NUM_RADIX_BITS/NUM_PASSES)) //(1<<PASS1RADIXBITS)
/* num-parts at pass-1 */
#define FANOUT_PASS2 (1 << (NUM_RADIX_BITS-(NUM_RADIX_BITS/NUM_PASSES))) //(1<<PASS2RADIXBITS)


/************* Common configs for ETH non partition hash join algorithms **************/
/** Number of tuples that each bucket can hold */
#ifndef BUCKET_SIZE
#define BUCKET_SIZE 2
#endif

/** Should hashtable buckets be padded to cache line size */
#ifndef PADDED_BUCKET
#define PADDED_BUCKET 0 /* default case: not padded */
#endif

/** Pre-allocation size for overflow buffers */
#ifndef OVERFLOW_BUF_SIZE
#define OVERFLOW_BUF_SIZE 1024 
#endif

#define PREFETCH_NPJ

/** Prefetching hash buckets parameter */
#ifndef PREFETCH_DISTANCE
#define PREFETCH_DISTANCE 16  //10
#endif

//#define INPUT_HASH_TABLE_SIZE

#define USE_MURMUR3_HASH

/************* Common configs for ETH sortmerge join algorithms **************/
#define HAVE_AVX

/** The partitioning fan-out for the inital step of sort-merge joins */
#ifndef NRADIXBITS_DEFAULT
#define NRADIXBITS_DEFAULT 7
#endif

/** Default partitioning fan-out, can be adjusted from command line. */
#ifndef PARTFANOUT_DEFAULT
#define PARTFANOUT_DEFAULT (1<<NRADIXBITS_DEFAULT)
#endif

/**
 * Determines the size of the multi-way merge buffer.
 * Ideally, it should match the size of the L3 cache.
 * @note this buffer is shared by active nr. of threads in a NUMA-region.
 */
#ifndef MWAY_MERGE_BUFFER_SIZE_DEFAULT
#define MWAY_MERGE_BUFFER_SIZE_DEFAULT L3_CACHE_SIZE /* 20MB L3 cache as default value */
#endif

/**
 * Parallel AVX Merge Parameters:
 * #define MERGEBITONICWIDTH 4,8,16
 */
#ifndef MERGEBITONICWIDTH
#define MERGEBITONICWIDTH 16 /* 4,8,16 */
#endif