/**
 * @file    avxsort_multiwaymerge.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Tue Dec 11 18:24:10 2012
 * @version $Id $
 *
 * @brief   AVX sorting that uses multi-way merge for inputs > L3-size.
 *
 *
 */

#ifndef AVXSORT_MULTIWAYMERGE_H_
#define AVXSORT_MULTIWAYMERGE_H_

#include <stdint.h>
#include <math.h>   /* log2() */
#include <stdlib.h> /* posix_memalign(), size_t */
#include <stdio.h>  /* perror() */

#include "configs/base_configs.h"
#include "configs/eth_configs.h"

#include "utils/data_structures.h" 
#include "utils/eth_data_structures.h" 
#include "utils/memory.h"

#include "utils/eth_avx_sort/avxsort.h"               /* avxsort() */
#include "utils/eth_avx_merge/avx_multiwaymerge.h"     /* avx_multiway_merge() */
#include "utils/eth_avx_merge/merge.h"                 /* avx_merge_int64() */


/**
 * \ingroup sorting
 *
 * As an optimization, this algorithm does the
 * merging phase using a multi-way merge-tree and consumes less memory
 * bandwidth if the input size is greater than the L3 cache size.
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
avxsortmultiway_tuples(Tuple<KeyType, PayloadType> ** inputptr, Tuple<KeyType, PayloadType> ** outputptr, uint64_t nitems);

/**
 * \ingroup sorting
 *
 * As an optimization, this algorithm does the
 * merging phase using a multi-way merge-tree and consumes less memory
 * bandwidth if the input size is greater than the L3 cache size.
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
avxsortmultiway_int64(int64_t ** inputptr, int64_t ** outputptr, uint64_t nitems);

static inline void
swap(int64_t ** A, int64_t ** B)
{
    int64_t * tmp = *A;
    *A = *B;
    *B = tmp;
}

/** \todo Parameterize the L3-cache usage of avxsortmultiway_, currently taken from #L3_CACHE_SIZE. */
template<class KeyType, class PayloadType>
void
avxsortmultiway_int64(int64_t ** inputptr, int64_t ** outputptr, uint64_t nitems)
{
    int64_t * input  = (int64_t*)(*inputptr);
    int64_t * output = (int64_t*)(*outputptr);

    uint64_t i;
    uint32_t L3BLOCKSIZE =  L3_CACHE_SIZE / sizeof(Tuple<KeyType, PayloadType>);

    uint64_t nblocks = nitems / L3BLOCKSIZE;
    /* each chunk keeps track of its temporary memory offset */
    int64_t * ptrs[nblocks+1][2];/* [block-in, block-out-tmp] */
    uint32_t sizes[nblocks+1];

    uint64_t remsize = (nitems % L3BLOCKSIZE);
    for(i = 0; i < nblocks; i++){
        ptrs[i][0] = input + i * L3BLOCKSIZE;
        ptrs[i][1] = output + i * L3BLOCKSIZE;
        sizes[i] = L3BLOCKSIZE;
    }

    /** 1) Divide the input into chunks fitting into L3 cache. */
    for(i = 0; i < nblocks; i++) {
        avxsort_int64(&ptrs[i][0], &ptrs[i][1], L3BLOCKSIZE);
        swap(&ptrs[i][0], &ptrs[i][1]);
    }

    /* one more chunk if not divisible */
    if(remsize) {
        ptrs[i][0] = input + i * L3BLOCKSIZE;
        ptrs[i][1] = output + i * L3BLOCKSIZE;
        sizes[i] = remsize;
        /* sort the last chunk which is less than BLOCKSIZE */
        avxsort_int64(&ptrs[i][0], &ptrs[i][1], remsize);
        swap(&ptrs[i][0], &ptrs[i][1]);
        nblocks ++;
    }

    /** 2) Apply a multi-way merge if number of blocks are > 3 */
    if(nblocks > 3){
        /* multi-way merge */
        if((nblocks % 2) == 1){
            /* merge the remainder chunk into the last one */
            int nc1 = nblocks-2;
            int nc2 = nblocks-1;
            int64_t * inpA  = ptrs[nc1][0];
            int64_t * inpB  = ptrs[nc2][0];
            int64_t * out   = ptrs[nc1][1];
            uint32_t  sizeA = sizes[nc1];
            uint32_t  sizeB = sizes[nc2];

            /* merge16_varlen(inpA, inpB, out, sizeA, sizeB); */
            avx_merge_int64(inpA, inpB, out, sizeA, sizeB);

            /* setup new pointers */
            ptrs[nc1][0] = out;
            ptrs[nc1][1] = inpA;
            sizes[nc1]   = sizeA + sizeB;
            nblocks --;
        }

        /* now setup a multi-way merge. */
        /*
         * IMPORTANT NOTE: nblocks must be padded to pow2! If nblocks is not a power
         * of 2, then we introduce additional blocks with 0-tuples to achieve a pow2
         * multi-way merging.
         */
        uint64_t nblockspow2 = 1 << (int)(ceil(log2(nblocks)));
        /* printf("Merge nblocks = %"PRId64" -- nblocks(pow2) = %"PRId64"\n",
                nblocks, nblockspow2); */
        Relation<KeyType, PayloadType> rels[nblockspow2];
        Relation<KeyType, PayloadType>  * chunks[nblockspow2];

        for(i = 0; i < nblocks; i++) {
            rels[i].tuples     = (Tuple<KeyType, PayloadType> *) ptrs[i][0];
            rels[i].num_tuples = sizes[i];
            chunks[i]          = &rels[i];
        }
        for( ; i < nblockspow2; i++){
            rels[i].tuples = 0;
            rels[i].num_tuples = 0;
            chunks[i] = &rels[i];
        }

        uint32_t  bufntuples = (L3_CACHE_SIZE)/sizeof(Tuple<KeyType, PayloadType>);
        Tuple<KeyType, PayloadType> * outptr     = (Tuple<KeyType, PayloadType> *) ptrs[0][1];

        Tuple<KeyType, PayloadType> * fifobuffer = (Tuple<KeyType, PayloadType> *) alloc_aligned(L3_CACHE_SIZE);

        avx_multiway_merge(outptr, chunks, nblockspow2, fifobuffer, bufntuples);
        free(fifobuffer);
        /* finally swap input/output pointers, where output holds the sorted list */
        * outputptr = (int64_t *) outptr;
        * inputptr  = (int64_t *) ptrs[0][0];
    }
    else {
        /* apply 2-way merge */
        const uint64_t logN = ceil(log2(nitems));
        for(i = log2(L3BLOCKSIZE); i < logN; i++) {
            uint64_t k = 0;
            for(uint64_t j = 0; j < (nblocks-1); j += 2) {
                int64_t * inpA  = ptrs[j][0];
                int64_t * inpB  = ptrs[j+1][0];
                int64_t * out   = ptrs[j][1];
                uint32_t  sizeA = sizes[j];
                uint32_t  sizeB = sizes[j+1];

                avx_merge_int64(inpA, inpB, out, sizeA, sizeB);

                /* setup new pointers */
                ptrs[k][0] = out;
                ptrs[k][1] = inpA;
                sizes[k]   = sizeA + sizeB;
                k++;
            }

            if((nblocks % 2)) {
                /* just move the pointers */
                ptrs[k][0] = ptrs[nblocks-1][0];
                ptrs[k][1] = ptrs[nblocks-1][1];
                sizes[k]   = sizes[nblocks-1];
                k++;
            }

            nblocks = k;
        }

        /* finally swap input/output pointers, where output holds the sorted list */
        * outputptr = ptrs[0][0];
        * inputptr  = ptrs[0][1];
    }
}

template<class KeyType, class PayloadType>
void
avxsortmultiway_tuples(Tuple<KeyType, PayloadType> ** inputptr, Tuple<KeyType, PayloadType> ** outputptr, uint64_t nitems)
{
    int64_t * inp = (int64_t *) *inputptr;
    int64_t * out = (int64_t *) *outputptr;

    avxsortmultiway_int64<KeyType, PayloadType>(&inp, &out, nitems);
    *inputptr = (Tuple<KeyType, PayloadType> *)inp;
    *outputptr = (Tuple<KeyType, PayloadType> *)out;
}


#endif /* AVXSORT_MULTIWAYMERGE_H_ */
