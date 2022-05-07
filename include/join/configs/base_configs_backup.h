
/* Common configs among all join algorithms (old version) (NOT USED NOW) */

#pragma once

/************* Processor Parameeters *************/
/* Define the underlying hardware type */
#define INTEL_E5 0
#define INTEL_XEON 1

/* Define to max. no. of expected CPU nodes. */
#define MAX_CPU_NODES 512

#ifndef CUSTOM_CPU_MAPPING
#define CUSTOM_CPU_MAPPING "configs/cpu-mapping.txt"
#endif

/** 
 * if the custom cpu mapping file exists, logical to physical mappings are
 * initialized from that file, next it will first try libNUMA if available,
 * and finally round-robin as last option.
 */
#ifndef CUSTOM_CPU_MAPPING_V2
#define CUSTOM_CPU_MAPPING_V2 "configs/cpu-mapping-v2.txt"
#endif

/************ Memory and Threading Parameters *****************/
/* Define NUMA if you support multiple NUMA nodes, otherwise it will consider everything on the first numa node. */
#define HAVE_LIBNUMA
#define MAX_THREADS MAX_CPU_NODES  //1024
#define REQUIRED_STACK_SIZE  (2*32*1024*1024) // (32*1024*1024)

/************* Cache Parameeters ****************/
/** L1 cache parameters. \note Change as needed for different machines */
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

/** L1 cache size */
#ifndef L1_CACHE_SIZE
#define L1_CACHE_SIZE  (1024*1024) //32768
#endif

/** L1 associativity */
#ifndef L1_ASSOCIATIVITY
#define L1_ASSOCIATIVITY 8
#endif

/** L2 Cache size of the system in bytes, used for determining block size */
#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE  (2*16*1024*1024) //(256*1024)
#endif

/** Number of tuples that can fit into L2 cache divided by 2 */
#ifndef BLOCKSIZE
#define BLOCKSIZE (L2_CACHE_SIZE / (2 * sizeof(int64_t)))
#endif

/** Logarithm base 2 of BLOCK_SIZE */
#ifndef LOG2_BLOCKSIZE
#define LOG2_BLOCKSIZE (log2(BLOCKSIZE))
#endif

/** L3 Cache size of the system in bytes. */
#ifndef L3_CACHE_SIZE
#define L3_CACHE_SIZE   (2*16*1024*1024)      // (20*1024*1024)
#endif

/****************** Skewness Parameters ***************/
#define SKEW_HANDLING 1
#define SKEW_DECOMPOSE_MARGIN (1.10) /* 10% margin */
#define SKEW_DECOMPOSE_SAMPLES 64 /* nr. of samples for range partitioning. */
#define SKEW_MAX_HEAVY_HITTERS 16 /* max nr. of heavy hitters to detect. */
#define SKEW_HEAVY_HITTER_THR 0.5 /* heavy hitter threshold freq. */


/*********** Input/Output Parameters **************/
//#define PERSIST_RELATIONS

/*********** Join Algorithm Parameters *******/
//#define DEVELOPMENT_MODE

/** 
 * Whether to use software write-combining optimized partitioning, 
 * see --enable-optimized-part config option 
 */
#define USE_SWWC_OPTIMIZED_PART 1

//#define TYPICAL_COPY_MODE

#define SMALL_PADDING_TUPLES_MULTIPLIER 3 


#define SKEWNESS_THRESHOLD_MULTIPLIER 64

#define USE_AVX_512

//#define USE_LEARNED_SORT_AVX

#define USE_AVXSORT_AS_STD_SORT

#ifdef USE_AVXSORT_AS_STD_SORT
#define USE_AVXMERGE_AS_STD_MERGE
#endif

#define USE_LEARNED_SORT_FOR_SORT_MERGE
#define USE_AVXSORT_FOR_SORTING_MINOR_BCKTS

#define BUILD_RMI_FROM_TWO_DATASETS
#define OVERALLOCATION_SIZE_RATIO 1
#define REPEATED_KEYS_SIZE_RATIO 0.5
#define SPILL_BUCKET_SIZE_RATIO 1
//#define USE_FIXED_PARTITION_SIZES

/******** Benchmarking Parameters ******/
//#define PERF_COUNTERS