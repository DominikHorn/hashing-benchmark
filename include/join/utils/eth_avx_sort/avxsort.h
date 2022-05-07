/**
 * @file    avxsort.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Tue Dec 11 18:24:10 2012
 * @version $Id: avxsort.h 3408 2013-02-25 16:27:52Z bcagri $
 *
 * @brief   Implementation of SIMD sorting using AVX instructions.
 *
 *
 */

#ifndef AVXSORT_H
#define AVXSORT_H

#include <stdint.h>

#include "configs/base_configs.h"
#include "configs/eth_configs.h"

#include "utils/data_structures.h" 
#include "utils/eth_data_structures.h" 
#include "utils/memory.h"

#include "utils/eth_avx_sort/avxsort_core.h"

/**
 * @defgroup sorting Sorting routines
 * @{
 */


/**
 * Sorts given array of items using AVX instructions. If the input is aligned
 * to cache-line, then aligned version of the implementation is executed.
 *
 * @note output array must be pre-allocated before the call.
 *
 * @param inputptr
 * @param outputptr
 * @param nitems
 */
void
avxsort_int64(int64_t ** inputptr, int64_t ** outputptr, uint64_t nitems);

/**
 * \copydoc avxsort_int64
 * @note currently not implemented.
 */
void
avxsort_int32(int32_t ** inputptr, int32_t ** outputptr, uint64_t nitems);

/**
 * Sorts given array of tuples on "key" field using AVX instructions. If the
 * input is aligned to cache-line, then aligned version of the implementation
 * is executed.
 *
 * @note output array must be pre-allocated before the call.
 *
 * @param inputptr
 * @param outputptr
 * @param nitems
 */
template<class KeyType, class PayloadType>
void
avxsort_tuples(Tuple<KeyType, PayloadType> ** inputptr, Tuple<KeyType, PayloadType> ** outputptr, uint64_t nitems);

extern int keycmp(const void * k1, const void * k2);

/*******************************************************************************
 *                                                                             *
 *                               Implementations                               *
 *                                                                             *
 *******************************************************************************/
/**
 * Single threaded SIMD sorting method implemented with AVX 256-bit
 * vectors. 128-bit version is described in "Efficient Implementation of
 * Sorting on Multi-Core SIMD CPU Architecture" by Chhugani et al., PVLDB '08.
 * Sorts 64-bit items where each item can be either treated as 64-bit key
 * only or 32-bit (key,val) pairs.
 *
 * @param inputptr pointer to the input to be sorted
 * @param inputptr pointer to sorted output relation, memory pre-allocated.
 * @param nitems   number of input items
 *
 * @warning inputptr and outputptr might be swapped internally.
 */
void
avxsort_unaligned(int64_t ** inputptr, int64_t ** outputptr, uint64_t nitems)
{
    if(nitems <= 0)
        return;

    int64_t * input  = * inputptr;
    int64_t * output = * outputptr;

    uint64_t i;
    uint64_t nchunks = (nitems / BLOCKSIZE);
    int rem = (nitems % BLOCKSIZE);

    /* each chunk keeps track of its temporary memory offset */
    int64_t * ptrs[nchunks+1][2];/* [chunk-in, chunk-out-tmp] */
    uint32_t sizes[nchunks+1];

    for(i = 0; i <= nchunks; i++) {
        ptrs[i][0] = input + i *  BLOCKSIZE;
        ptrs[i][1] = output + i * BLOCKSIZE;
        sizes[i]   = BLOCKSIZE;
    }

    /** 1) Divide the input into chunks fitting into L2 cache. */
    /* one more chunk if not divisible */
    for(i = 0; i < nchunks; i++) {
        avxsort_block(&ptrs[i][0], &ptrs[i][1], BLOCKSIZE);
        swap(&ptrs[i][0], &ptrs[i][1]);
    }

    if(rem) {
        /* sort the last chunk which is less than BLOCKSIZE */
        avxsort_rem(&ptrs[i][0], &ptrs[i][1], rem);
        swap(&ptrs[i][0], &ptrs[i][1]);
        sizes[i] = rem;
    }


    /**
     * 2.a) for itr = [(logM) .. (logN -1)], merge sequences of length 2^itr to
     * obtain sorted sequences of length 2^{itr+1}.
     */
    nchunks += (rem > 0);
    /* printf("Merge chunks = %d\n", nchunks); */
    const uint64_t logN = ceil(log2(nitems));
    for(i = LOG2_BLOCKSIZE; i < logN; i++) {

        uint64_t k = 0;
        for(uint64_t j = 0; j < (nchunks-1); j += 2) {
            int64_t * inpA  = ptrs[j][0];
            int64_t * inpB  = ptrs[j+1][0];
            int64_t * out   = ptrs[j][1];
            uint32_t  sizeA = sizes[j];
            uint32_t  sizeB = sizes[j+1];

            merge16_varlen(inpA, inpB, out, sizeA, sizeB);

            /* setup new pointers */
            ptrs[k][0] = out;
            ptrs[k][1] = inpA;
            sizes[k]   = sizeA + sizeB;
            k++;
        }

        if((nchunks % 2)) {
            /* just move the pointers */
            ptrs[k][0] = ptrs[nchunks-1][0];
            ptrs[k][1] = ptrs[nchunks-1][1];
            sizes[k]   = sizes[nchunks-1];
            k++;
        }

        nchunks = k;
    }

    /* finally swap input/output pointers, where output holds the sorted list */
    * outputptr = ptrs[0][0];
    * inputptr  = ptrs[0][1];

}

/*******************************************************************************
 *                                                                             *
 *               Aligned Version of the Implementation                         *
 *                                                                             *
 *******************************************************************************/
void
avxsort_aligned(int64_t ** inputptr, int64_t ** outputptr, uint64_t nitems)
{
    if(nitems <= 0)
        return;

    int64_t * input  = * inputptr;
    int64_t * output = * outputptr;

    uint64_t i;
    uint64_t nchunks = (nitems / BLOCKSIZE);
    int rem = (nitems % BLOCKSIZE);
    /* printf("nchunks = %d, nitems = %d, rem = %d\n", nchunks, nitems, rem); */
    /* each chunk keeps track of its temporary memory offset */
    int64_t * ptrs[nchunks+1][2];/* [chunk-in, chunk-out-tmp] */
    uint32_t sizes[nchunks+1];

    for(i = 0; i <= nchunks; i++) {
        ptrs[i][0] = input + i *  BLOCKSIZE;
        ptrs[i][1] = output + i * BLOCKSIZE;
        sizes[i]   = BLOCKSIZE;
    }

    /** 1) Divide the input into chunks fitting into L2 cache. */
    /* one more chunk if not divisible */
    for(i = 0; i < nchunks; i++) {
        avxsort_block_aligned(&ptrs[i][0], &ptrs[i][1], BLOCKSIZE);
        swap(&ptrs[i][0], &ptrs[i][1]);
    }

    if(rem) {
        /* sort the last chunk which is less than BLOCKSIZE */
        avxsort_rem_aligned(&ptrs[i][0], &ptrs[i][1], rem);
        swap(&ptrs[i][0], &ptrs[i][1]);
        sizes[i] = rem;
    }


    /**
     * 2.a) for itr = [(logM) .. (logN -1)], merge sequences of length 2^itr to
     * obtain sorted sequences of length 2^{itr+1}.
     */
    nchunks += (rem > 0);
    /* printf("Merge chunks = %d\n", nchunks); */
    const uint64_t logN = ceil(log2(nitems));
    for(i = LOG2_BLOCKSIZE; i < logN; i++) {

        uint64_t k = 0;
        for(uint64_t j = 0; j < (nchunks-1); j += 2) {
            int64_t * inpA  = ptrs[j][0];
            int64_t * inpB  = ptrs[j+1][0];
            int64_t * out   = ptrs[j][1];
            uint32_t  sizeA = sizes[j];
            uint32_t  sizeB = sizes[j+1];

            merge8_varlen_aligned(inpA, inpB, out, sizeA, sizeB);

            /* setup new pointers */
            ptrs[k][0] = out;
            ptrs[k][1] = inpA;
            sizes[k]   = sizeA + sizeB;
            k++;
        }

        if((nchunks % 2)) {
            /* just move the pointers */
            ptrs[k][0] = ptrs[nchunks-1][0];
            ptrs[k][1] = ptrs[nchunks-1][1];
            sizes[k]   = sizes[nchunks-1];
            k++;
        }

        nchunks = k;
    }

    /* finally swap input/output pointers, where output holds the sorted list */
    * outputptr = ptrs[0][0];
    * inputptr  = ptrs[0][1];

}

template<class KeyType, class PayloadType>
void
avxsort_tuples(Tuple<KeyType, PayloadType> ** inputptr, Tuple<KeyType, PayloadType> ** outputptr, uint64_t nitems)
{
    int64_t * input  = (int64_t*)(*inputptr);
    int64_t * output = (int64_t*)(*outputptr);

    /* choose actual implementation depending on the input alignment */
    if(((uintptr_t)input % CACHE_LINE_SIZE) == 0
       && ((uintptr_t)output % CACHE_LINE_SIZE) == 0)
        avxsort_aligned(&input, &output, nitems);
    else
        avxsort_unaligned(&input, &output, nitems);

    *inputptr = (Tuple<KeyType, PayloadType> *)(input);
    *outputptr = (Tuple<KeyType, PayloadType> *)(output);
}

void
avxsort_int64(int64_t ** inputptr, int64_t ** outputptr, uint64_t nitems)
{
    /* \todo: implement */
    int64_t * input  = (int64_t*)(*inputptr);
    int64_t * output = (int64_t*)(*outputptr);

    /* choose actual implementation depending on the input alignment */
    if(((uintptr_t)input % CACHE_LINE_SIZE) == 0
       && ((uintptr_t)output % CACHE_LINE_SIZE) == 0)
        avxsort_aligned(&input, &output, nitems);
    else
        avxsort_unaligned(&input, &output, nitems);

    *inputptr = (int64_t *)(input);
    *outputptr = (int64_t *)(output);
}

void
avxsort_int32(int32_t ** inputptr, int32_t ** outputptr, uint64_t nitems)
{
    /* \todo: implement */
}


/** @} */

#endif /* AVXSORT_H */
