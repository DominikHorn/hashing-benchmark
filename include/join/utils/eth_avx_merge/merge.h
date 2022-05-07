/**
 * @file    merge.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Tue Dec 11 18:24:10 2012
 * @version $Id $
 * 
 * @brief   Various merging methods, i.e. scalar, parallel with AVX, etc.
 * 
 * (c) 2014, ETH Zurich, Systems Group
 * 
 */
#ifndef MERGE_H
#define MERGE_H

#include <stdint.h>

#include "configs/base_configs.h"
#include "configs/eth_configs.h"

#include "utils/base_utils.h"
#include "utils/data_structures.h" /* tuple_t, relation_t */
#include "utils/eth_data_structures.h"
#include "utils/math.h"

#include "utils/eth_avx_sort/avxsort_core.h" /* Just because of inlines. TODO: fix this */

/**
 * Merges two sorted lists of length len into a new sorted list of length
 * outlen. The following configurations are possible:
 *
 * #define MERGEBRANCHING 0:NORMAL-IF, 1:CMOVE, 2:PREDICATION
 * #define MERGEBITONICWIDTH 4,8,16
 * MERGEALIGNED: depending on if inpA, inpB and out are cache-line aligned.
 *
 * @warning Assumes that inputs will have items multiple of 8.
 * @param inpA sorted list A
 * @param inpB sorted list B
 * @param out merged output list
 * @param len length of input
 *
 * @return output size
 */
template<class KeyType, class PayloadType>
uint64_t
avx_merge_tuples(Tuple<KeyType, PayloadType> * const inA,
          Tuple<KeyType, PayloadType> * const inB,
          Tuple<KeyType, PayloadType> * const outp,
          const uint64_t lenA,
          const uint64_t lenB);

/** 
 * Merges two sorted lists of length len into a new sorted list of length
 * outlen. The following configurations are possible:
 *
 * #define MERGEBRANCHING 0:NORMAL-IF, 1:CMOVE, 2:PREDICATION
 * #define MERGEBITONICWIDTH 4,8,16
 * MERGEALIGNED: depending on if inpA, inpB and out are cache-line aligned.
 *
 * @warning Assumes that inputs will have items multiple of 8. 
 * @param inpA sorted list A
 * @param inpB sorted list B
 * @param out merged output list
 * @param len length of input
 */
uint64_t
avx_merge_int64(int64_t * const inpA,
          int64_t * const inpB, 
          int64_t * const out, 
          const uint64_t lenA,
          const uint64_t lenB);

/**
 * Scalar method for merging 2 sorted lists into a 1 sorted list.
 *
 * @param inpA
 * @param inpB
 * @param out
 * @param lenA length of array A
 * @param lenB length of array B
 *
 * @return length of the new merged list
 */
uint64_t
scalar_merge_int64(int64_t * const inpA,
        int64_t * const inpB,
        int64_t * const out,
        const uint64_t lenA,
        const uint64_t lenB);

/**
 * Scalar merge routine for two sorted tuple arrays.
 *
 * @param inpA
 * @param inpB
 * @param out
 * @param lenA length of array A
 * @param lenB length of array B
 *
 * @return length of the new merged list
 */
template<class KeyType, class PayloadType>
uint64_t
scalar_merge_tuples(Tuple<KeyType, PayloadType> * const inpA,
        Tuple<KeyType, PayloadType> * const inpB,
        Tuple<KeyType, PayloadType> * const out,
        const uint64_t lenA,
        const uint64_t lenB);





/**
 * Merge branching, see avxcommon.h:
 * There are 2 ways to implement branches:
 *     1) With conditional move instr.s using inline assembly (IFELSEWITHCMOVE).
 *     2) With software predication (IFELSEWITHPREDICATION).
 *     3) With normal if-else
 */

uint64_t
scalar_merge_int64(int64_t * const inA,
                   int64_t * const inB,
                   int64_t * const outp,
                   const uint64_t lenA,
                   const uint64_t lenB)
{
    uint64_t i, j, k;

    i = 0; j = 0; k = 0;

    while ( i < lenA && j < lenB) {
        if(inA[i] < inB[j]) {
            outp[k] = inA[i];
            k++;
            i++;
        }
        else {
            outp[k] = inB[j];
            k++;
            j++;
        }
    }

    while ( i < lenA ) {
        outp[k] = inA[i];
        k++;
        i++;
    }

    while ( j < lenB ) {
        outp[k] = inB[j];
        k++;
        j++;
    }

    return k;
}

template<class KeyType, class PayloadType>
uint64_t
scalar_merge_tuples(Tuple<KeyType, PayloadType> * const inA,
             Tuple<KeyType, PayloadType> * const inB,
             Tuple<KeyType, PayloadType> * const outp,
             const uint64_t lenA,
             const uint64_t lenB)
{
    uint64_t i, j, k;

    i = 0; j = 0; k = 0;

    while ( i < lenA && j < lenB) {
        if(inA[i].key < inB[j].key) {
            outp[k] = inA[i];
            k++;
            i++;
        }
        else {
            outp[k] = inB[j];
            k++;
            j++;
        }
    }

    while ( i < lenA ) {
        outp[k] = inA[i];
        k++;
        i++;
    }

    while ( j < lenB ) {
        outp[k] = inB[j];
        k++;
        j++;
    }

    return k;
}

template<class KeyType, class PayloadType>
uint64_t
avx_merge_tuples(Tuple<KeyType, PayloadType> * const inA,
          Tuple<KeyType, PayloadType> * const inB,
          Tuple<KeyType, PayloadType> * const outp,
          const uint64_t lenA,
          const uint64_t lenB)
{
    int64_t * const inpA = (int64_t * const) inA;
    int64_t * const inpB = (int64_t * const) inB;
    int64_t * const out = (int64_t * const) outp;

    int isaligned = 0, iseqlen = 0;

    /* is-aligned ? */
    isaligned = (((uintptr_t)inpA % CACHE_LINE_SIZE) == 0) &&
                (((uintptr_t)inpB % CACHE_LINE_SIZE) == 0) &&
                (((uintptr_t)out  % CACHE_LINE_SIZE) == 0);

    /* is equal length? */
    /*There is a bug when size2 = size1 and eqlen enabled */
    /*iseqlen = (lenA == lenB);*/
    /* \todo FIXME There is a problem when using merge-eqlen variants, because the
    merge routine does not consider that other lists begin where one list ends
    and might be overwriting a few tuples. */
    if(iseqlen) {
        if(isaligned){
#if (MERGEBITONICWIDTH == 4)
            merge4_eqlen_aligned(inpA, inpB, out, lenA);
#elif (MERGEBITONICWIDTH == 8)
            merge8_eqlen_aligned(inpA, inpB, out, lenA);
#elif (MERGEBITONICWIDTH == 16)
            merge16_eqlen_aligned(inpA, inpB, out, lenA);
#endif
        }
        else {
#if (MERGEBITONICWIDTH == 4)
            merge4_eqlen(inpA, inpB, out, lenA);
#elif (MERGEBITONICWIDTH == 8)
            merge8_eqlen(inpA, inpB, out, lenA);
#elif (MERGEBITONICWIDTH == 16)
            merge16_eqlen(inpA, inpB, out, lenA);
#endif
        }
    }
    else {
        if(isaligned){
#if (MERGEBITONICWIDTH == 4)
            merge4_varlen_aligned(inpA, inpB, out, lenA, lenB);
#elif (MERGEBITONICWIDTH == 8)
            merge8_varlen_aligned(inpA, inpB, out, lenA, lenB);
#elif (MERGEBITONICWIDTH == 16)
            merge16_varlen_aligned(inpA, inpB, out, lenA, lenB);
#endif
        }
        else {
#if (MERGEBITONICWIDTH == 4)
            merge4_varlen(inpA, inpB, out, lenA, lenB);
#elif (MERGEBITONICWIDTH == 8)
            merge8_varlen(inpA, inpB, out, lenA, lenB);
#elif (MERGEBITONICWIDTH == 16)
            merge16_varlen(inpA, inpB, out, lenA, lenB);
#endif
        }
    }

    return (lenA + lenB);
}


uint64_t
avx_merge_int64(int64_t * const inpA,
          int64_t * const inpB,
          int64_t * const out,
          const uint64_t lenA,
          const uint64_t lenB)
{
    int isaligned = 0, iseqlen = 0;

    /* is-aligned ? */
    isaligned = (((uintptr_t)inpA % CACHE_LINE_SIZE) == 0) &&
                (((uintptr_t)inpB % CACHE_LINE_SIZE) == 0) &&
                (((uintptr_t)out  % CACHE_LINE_SIZE) == 0);

    /* is equal length? */
    /* iseqlen = (lenA == lenB); */
    /* TODO: There is a problem when using merge-eqlen variants, because the
    merge routine does not consider that other lists begin where one list ends
    and might be overwriting a few tuples. */
    if(iseqlen) {
        if(isaligned){
#if (MERGEBITONICWIDTH == 4)
            merge4_eqlen_aligned(inpA, inpB, out, lenA);
#elif (MERGEBITONICWIDTH == 8)
            merge8_eqlen_aligned(inpA, inpB, out, lenA);
#elif (MERGEBITONICWIDTH == 16)
            merge16_eqlen_aligned(inpA, inpB, out, lenA);
#endif
        }
        else {
#if (MERGEBITONICWIDTH == 4)
            merge4_eqlen(inpA, inpB, out, lenA);
#elif (MERGEBITONICWIDTH == 8)
            merge8_eqlen(inpA, inpB, out, lenA);
#elif (MERGEBITONICWIDTH == 16)
            merge16_eqlen(inpA, inpB, out, lenA);
#endif
        }
    }
    else {
        if(isaligned){
#if (MERGEBITONICWIDTH == 4)
            merge4_varlen_aligned(inpA, inpB, out, lenA, lenB);
#elif (MERGEBITONICWIDTH == 8)
            merge8_varlen_aligned(inpA, inpB, out, lenA, lenB);
#elif (MERGEBITONICWIDTH == 16)
            merge16_varlen_aligned(inpA, inpB, out, lenA, lenB);
#endif
        }
        else {
#if (MERGEBITONICWIDTH == 4)
            merge4_varlen(inpA, inpB, out, lenA, lenB);
#elif (MERGEBITONICWIDTH == 8)
            merge8_varlen(inpA, inpB, out, lenA, lenB);
#elif (MERGEBITONICWIDTH == 16)
            merge16_varlen(inpA, inpB, out, lenA, lenB);
#endif
        }
    }

    return (lenA + lenB);
}


#endif /* MERGE_H */
