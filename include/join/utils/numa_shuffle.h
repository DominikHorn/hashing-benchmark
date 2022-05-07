/**
 * @file   numa_shuffle.h
 * @author Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date   Wed May 23 16:43:30 2012
 *
 * @brief  Provides a machine specific implementation for NUMA-shuffling strategies.
 *
 * @note   The implementation needs to be customized for different hardware
 *         based on the NUMA topology.
 *
 * (c) 2014, ETH Zurich, Systems Group
 *
 */
#ifndef NUMA_SHUFFLE_H_
#define NUMA_SHUFFLE_H_

#include "configs/base_configs.h"
#include "utils/data_structures.h" /* enum numa_strategy_t */
#include "utils/eth_data_structures.h"
#include "utils/cpu_mapping.h" /* get_cpu_id() */
#include "utils/cpu_mapping_one_numa.h" /* get_cpu_id_develop() */
#include "utils/data_generation.h"   /* knuth_shuffle() */


/**
 * \ingroup numa
 *
 * Initialize the NUMA shuffling strategy with one of the following:
 *
 * RANDOM, RING, NEXT
 */
void
numa_shuffle_init(enum numa_strategy_t numastrategy, int nthreads);

/**
 * \ingroup numa
 *
 * Machine specific implementation of NUMA-shuffling strategies.
 *
 * @note The implementation needs to be customized for different hardware
 *       based on the NUMA topology.
 *
 * @param my_tid logical thread id of the calling thread.
 * @param i next thread index for shuffling (between 0 and nthreads)
 * @param nthreads number of total threads
 * @return the logical thread id of the destination thread for data shuffling
 */
int
get_numa_shuffle_strategy(int my_tid, int i, int nthreads);

/**
 * Various NUMA shuffling strategies as also described by NUMA-aware
 * data shuffling paper:
 * NUMA_SHUFFLE_RANDOM, NUMA_SHUFFLE_RING, NUMA_SHUFFLE_NEXT
 *
 * enum numa_strategy_t {RANDOM, RING, NEXT};
 */
static enum numa_strategy_t numastrategy_;

/** @note make sure that nthreads is always < 512 */
static Tuple<uint32_t, uint32_t> shuffleorder[512];


/**
 * Various NUMA shuffling strategies for data shuffling phase of join
 * algorithms as also described by NUMA-aware data shuffling paper [CIDR'13].
 *
 * NUMA_SHUFFLE_RANDOM, NUMA_SHUFFLE_RING, NUMA_SHUFFLE_NEXT
 */
void
numa_shuffle_init(enum numa_strategy_t numastrategy, int nthreads)
{
    numastrategy_ = numastrategy;
    if(numastrategy == RANDOM){
        /* if random, initialize a randomization once */
        Relation<uint32_t, uint32_t> ss;
        ss.tuples = (Tuple<uint32_t, uint32_t> *) shuffleorder;
        ss.num_tuples = nthreads;
        for(int s=0; s < nthreads; s++)
            ss.tuples[s].key = s;
        knuth_shuffle(&ss);
    }
}

/**
 * Machine specific implementation of NUMA-shuffling strategies.
 *
 * @note The implementation needs to be customized for different hardware
 *       based on the NUMA topology.
 *
 * @note RING-based shuffling is expected to work well only when using all 
 *       the threads. Better way to apply ring-based shuffling is to change the
 *       CPU-mappings in "cpu-mapping.txt" and use NEXT-based shuffling. 
 *
 * @param my_tid logical thread id of the calling thread.
 * @param nextidx next thread index for shuffling (between 0 and nthreads)
 * @param nthreads number of total threads
 * @return the logical thread id of the destination thread for data shuffling
 */
int
get_numa_shuffle_strategy(int my_tid, int nextidx, int nthreads)
{
    if(numastrategy_ == RANDOM){
        return shuffleorder[nextidx].key;
    }
    else if(numastrategy_ == RING){
        /* for Intel-E5-4640: */
        /*
        static int numa[64] = {
                0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
                2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
                3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 };
        nid = numa[my_tid];
        */
        #ifdef DEVELOPMENT_MODE
        return (my_tid + (nthreads/get_num_numa_regions_develop()) + nextidx) % nthreads;
        #else
        return (my_tid + (nthreads/get_num_numa_regions_v2()) + nextidx) % nthreads;
        #endif
    }
    else /* NEXT */ {
        return (my_tid + nextidx) % nthreads; // --> NUMA-SHUFF-NEXT-THR
    }
}

#endif /* NUMA_SHUFFLE_H_ */
