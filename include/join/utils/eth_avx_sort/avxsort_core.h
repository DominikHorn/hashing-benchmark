/**
 * @file    avxsort_core.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @version $Id $
 *
 * @brief   AVX sorting core kernels, etc.
 *
 *
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>             /* qsort() */
#include <math.h>               /* log2()  */
#ifdef __cplusplus
#include <algorithm>            /* sort()  */
#endif

#include "configs/base_configs.h"
#include "configs/eth_configs.h"

#include "utils/math.h"

#include "utils/eth_avx_sort/avxcommon.h"

/* #include "iacaMarks.h" */

 /* just make the code compile without AVX support */
#ifndef HAVE_AVX
#include "utils/eth_avx_sort/avxintrin_emu.h"
#endif

/*******************************************************************************
 *                                                                             *
 *                               Declarations                                  *
 *                                                                             *
 *******************************************************************************/

/** 
 *  Fixed size single-block AVX sorting routine
 */
inline void
avxsort_block(int64_t ** inputptr, int64_t ** outputptr, const int BLOCK_SIZE);

/** 
 * In-register sorting of 4x4=16 <32-bit key,32-bit val> pairs pointed
 * by items-ptr. Uses 256-bit AVX registers.
 * 
 * @param items 16x64-bit values
 * @param output 4-sorted 16x64-bit output values
 */
inline void
inregister_sort_keyval32(int64_t * items, int64_t * output);

/** 
 * Merges two sorted lists of length len into a new sorted list of length
 * outlen. Uses 4-wide bitonic-merge network.
 *
 * @warning Assumes that inputs will have items multiple of 4. 
 * @param inpA sorted list A
 * @param inpB sorted list B
 * @param out merged output list
 * @param len length of input
 */
inline void
merge4_eqlen(int64_t * const inpA, int64_t * const inpB, 
             int64_t * const out, const uint32_t len);

/** 
 * Merges two sorted lists of length len into a new sorted list of length
 * outlen. Uses 16-wide bitonic-merge network.
 *
 * @warning Assumes that inputs will have items multiple of 16. 
 * @param inpA sorted list A
 * @param inpB sorted list B
 * @param out merged output list
 * @param len length of input
 */
inline void
merge16_eqlen(int64_t * const inpA, int64_t * const inpB, 
              int64_t * const out, const uint32_t len);


/*******************************************************************************
 *                                                                             *
 *                               Implementations                               *
 *                                                                             *
 *******************************************************************************/

/**************** Helper Methods************************************************/
inline __attribute__((__always_inline__)) uint32_t
mylog2(const uint32_t n) 
{
    register uint32_t res;
    __asm__ ( "\tbsr %1, %0\n" : "=r"(res) : "r"(n) );
    return res;
}
/*******************************************************************************/

inline void __attribute((always_inline))
merge4_eqlen(int64_t * const inpA, int64_t * const inpB,
             int64_t * const out, const uint32_t len)
{
    register block4 * inA  = (block4 *) inpA;
    register block4 * inB  = (block4 *) inpB;
    block4 * const    endA = (block4 *) (inpA + len);
    block4 * const    endB = (block4 *) (inpB + len);

    block4 * outp = (block4 *) out;

    register block4 * next = inB;

#ifndef USE_AVX_512
    register __m256d outreg1;
    register __m256d outreg2;

    register __m256d regA = _mm256_loadu_pd((double const *) inA);
    register __m256d regB = _mm256_loadu_pd((double const *) next);

    inA ++;
    inB ++;

    BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
    /* store outreg1 */
    _mm256_storeu_pd((double *) outp, outreg1);
#else
    register __m256i outreg1;
    register __m256i outreg2;

//#ifndef DEVELOPMENT_MODE
//    register __m256i regA = _mm256_loadu_epi64((int64_t const *) inA);
//    register __m256i regB = _mm256_loadu_epi64((int64_t const *) next);
//#else
    register __m256i regA = _mm256_mask_loadu_epi64(regA, 0xFF, (int64_t const *) inA);
    register __m256i regB = _mm256_mask_loadu_epi64(regB, 0xFF, (int64_t const *) next);
//#endif
    inA ++;
    inB ++;

    BITONIC_MERGE4(outreg1, outreg2, regA, regB);

    /* store outreg1 */
#ifndef DEVELOPMENT_MODE        
    _mm256_storeu_epi64((int64_t *) outp, outreg1);
#else
    _mm256_mask_storeu_epi64((int64_t *) outp, 0xFF, outreg1); 
#endif
#endif

    outp ++;

    while( inA < endA && inB < endB ) {

        /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
        IFELSECONDMOVE(next, inA, inB, 32);

#ifndef USE_AVX_512
        regA = outreg2;
        regB = _mm256_loadu_pd((double const *) next);
        
        BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
        _mm256_storeu_pd((double *) outp, outreg1);
#else
        regA = outreg2;
//#ifndef DEVELOPMENT_MODE
//        regB = _mm256_loadu_epi64((int64_t const *) next);
//#else
        regB = _mm256_mask_loadu_epi64(regB, 0xFF, (int64_t const *) next);
//#endif 
        BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
#ifndef DEVELOPMENT_MODE
        _mm256_storeu_epi64((int64_t *) outp, outreg1);
#else
        _mm256_mask_storeu_epi64((int64_t *) outp, 0xFF, outreg1); 
#endif

#endif
        outp ++;
    }

    /* handle remaining items */
    while( inA < endA ) {

#ifndef USE_AVX_512
        __m256d regA = _mm256_loadu_pd((double const *) inA);
        __m256d regB = outreg2;

        BITONIC_MERGE4(outreg1, outreg2, regA, regB);

        _mm256_storeu_pd((double *) outp, outreg1);
#else
//#ifndef DEVELOPMENT_MODE
//        __m256i regA = _mm256_loadu_epi64((int64_t const *) inA);
//#else
        __m256i regA = _mm256_mask_loadu_epi64(regA, 0xFF, (int64_t const *) inA);
//#endif
        __m256i regB = outreg2;

        BITONIC_MERGE4(outreg1, outreg2, regA, regB);
#ifndef DEVELOPMENT_MODE
        _mm256_storeu_epi64((int64_t *) outp, outreg1);
#else
        _mm256_mask_storeu_epi64((int64_t *) outp, 0xFF, outreg1);
#endif
#endif
        inA ++;
        outp ++;
    }

    while( inB < endB ) {

#ifndef USE_AVX_512
        __m256d regA = outreg2;
        __m256d regB = _mm256_loadu_pd((double const *) inB);

        BITONIC_MERGE4(outreg1, outreg2, regA, regB);

        _mm256_storeu_pd((double *) outp, outreg1);
#else
        __m256i regA = outreg2;
//#ifndef DEVELOPMENT_MODE
//        __m256i regB = _mm256_loadu_epi64((int64_t const *) inB);
//#else
        __m256i regB = _mm256_mask_loadu_epi64(regB, 0xFF, (int64_t const *) inB);
//#endif
        BITONIC_MERGE4(outreg1, outreg2, regA, regB);

#ifndef DEVELOPMENT_MODE
        _mm256_storeu_epi64((int64_t *) outp, outreg1);
#else
        _mm256_mask_storeu_epi64((int64_t *) outp, 0xFF, outreg1);
#endif
#endif
        inB ++;
        outp ++;
    }

    /* store the last remaining register values */
#ifndef USE_AVX_512
    _mm256_storeu_pd((double *) outp, outreg2);
#else

#ifndef DEVELOPMENT_MODE
    _mm256_storeu_epi64((int64_t *) outp, outreg2);
#else
    _mm256_mask_storeu_epi64((int64_t *) outp, 0xFF, outreg2);
#endif
#endif
}

/** 
 * Merges two sorted lists of length len into a new sorted list of length
 * outlen. Uses 8-wide bitonic-merge network.
 *
 * @warning Assumes that inputs will have items multiple of 8. 
 * @param inpA sorted list A
 * @param inpB sorted list B
 * @param out merged output list
 * @param len length of input
 */
inline void __attribute((always_inline))
merge8_eqlen(int64_t * const inpA, int64_t * const inpB,
             int64_t * const out, const uint32_t len)
{
    register block8 * inA  = (block8 *) inpA;
    register block8 * inB  = (block8 *) inpB;
    block8 * const    endA = (block8 *) (inpA + len);
    block8 * const    endB = (block8 *) (inpB + len);

    block8 * outp = (block8 *) out;

    register block8 * next = inB;

#ifndef USE_AVX_512        
    register __m256d outreg1l, outreg1h;
    register __m256d outreg2l, outreg2h;

    register __m256d regAl, regAh;
    register __m256d regBl, regBh;

    LOAD8U(regAl, regAh, inA);
    LOAD8U(regBl, regBh, next);

    inA ++;
    inB ++;

    BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                   regAl, regAh, regBl, regBh);

    /* store outreg1 */
    STORE8U(outp, outreg1l, outreg1h);
#else
    register __m512i outreg1;
    register __m512i outreg2;

    register __m512i regA;
    register __m512i regB;

    LOAD8U(regA, inA);
    LOAD8U(regB, next);

    inA ++;
    inB ++;

    BITONIC_MERGE8(outreg1, outreg2, regA, regB);

    /* store outreg1 */
    STORE8U(outp, outreg1);
#endif

    outp ++;

    while( inA < endA && inB < endB ) {

        /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
        IFELSECONDMOVE(next, inA, inB, 64);

#ifndef USE_AVX_512
        regAl = outreg2l;
        regAh = outreg2h;
        LOAD8U(regBl, regBh, next);
        
        BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                       regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8U(outp, outreg1l, outreg1h);
#else
        regA = outreg2;
        LOAD8U(regB, next);
        
        BITONIC_MERGE8(outreg1, outreg2, regA, regB);

        /* store outreg1 */
        STORE8U(outp, outreg1);
#endif

        outp ++;
    }

    /* handle remaining items */
    while( inA < endA ) {

#ifndef USE_AVX_512
        __m256d regAl, regAh;
        LOAD8U(regAl, regAh, inA);

        __m256d regBl = outreg2l;
        __m256d regBh = outreg2h;

        BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                       regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8U(outp, outreg1l, outreg1h);
#else
        __m512i regA;
        LOAD8U(regA, inA);

        __m512i regB = outreg2;

        BITONIC_MERGE8(outreg1, outreg2, regA, regB);

        /* store outreg1 */
        STORE8U(outp, outreg1);
#endif        
        outp ++;
        inA ++;
    }

    while( inB < endB ) {

#ifndef USE_AVX_512
        __m256d regAl = outreg2l;
        __m256d regAh = outreg2h;
        __m256d regBl, regBh;

        LOAD8U(regBl, regBh, inB);

        BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                       regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8U(outp, outreg1l, outreg1h);
#else
        __m512i regA = outreg2;
        __m512i regB;

        LOAD8U(regB, inB);

        BITONIC_MERGE8(outreg1, outreg2, regA, regB);

        /* store outreg1 */
        STORE8U(outp, outreg1);
#endif
        outp ++;
        inB ++;
    }

    /* store the last remaining register values */
#ifndef USE_AVX_512
    STORE8U(outp, outreg2l, outreg2h);
#else
    STORE8U(outp, outreg2);
#endif
}

inline void __attribute((always_inline))
merge16_eqlen(int64_t * const inpA, int64_t * const inpB,
              int64_t * const out, const uint32_t len)
{
    register block16 * inA  = (block16 *) inpA;
    register block16 * inB  = (block16 *) inpB;
    block16 * const    endA = (block16 *) (inpA + len);
    block16 * const    endB = (block16 *) (inpB + len);

    block16 * outp = (block16 *) out;

    register block16 * next = inB;

#ifndef USE_AVX_512
    __m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
    __m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;

    __m256d regAl1, regAl2, regAh1, regAh2;
    __m256d regBl1, regBl2, regBh1, regBh2;

    LOAD8U(regAl1, regAl2, inA);
    LOAD8U(regAh1, regAh2, ((block8 *)(inA) + 1));
    inA ++;

    LOAD8U(regBl1, regBl2, inB);
    LOAD8U(regBh1, regBh2, ((block8 *)(inB) + 1));
    inB ++;

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                    outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                    regAl1, regAl2, regAh1, regAh2,
                    regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8U(outp, outreg1l1, outreg1l2);
    STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
    __m512i outreg1l, outreg1h;
    __m512i outreg2l, outreg2h;

    __m512i regAl, regAh;
    __m512i regBl, regBh;

    LOAD8U(regAl, inA);
    LOAD8U(regAh, ((block8 *)(inA) + 1));
    inA ++;

    LOAD8U(regBl, inB);
    LOAD8U(regBh, ((block8 *)(inB) + 1));
    inB ++;

    BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                    regAl, regAh, regBl, regBh);

    /* store outreg1 */
    STORE8U(outp, outreg1l);
    STORE8U(((block8 *)outp + 1), outreg1h);
#endif
    outp ++;

    while( inA < endA && inB < endB ) {

        /** The inline assembly below does exactly the following code: */
        /* Option 3: with assembly */
        IFELSECONDMOVE(next, inA, inB, 128);
        
#ifndef USE_AVX_512
        regAl1 = outreg2l1;
        regAl2 = outreg2l2;
        regAh1 = outreg2h1;
        regAh2 = outreg2h2;

        LOAD8U(regBl1, regBl2, next);
        LOAD8U(regBh1, regBh2, ((block8 *)next + 1));
        
        BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                        outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                        regAl1, regAl2, regAh1, regAh2,
                        regBl1, regBl2, regBh1, regBh2);

        /* store outreg1 */
        STORE8U(outp, outreg1l1, outreg1l2);
        STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
        regAl = outreg2l;
        regAh = outreg2h;

        LOAD8U(regBl, next);
        LOAD8U(regBh, ((block8 *)next + 1));
        
        BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                        regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8U(outp, outreg1l);
        STORE8U(((block8 *)outp + 1), outreg1h);
#endif
        outp ++;
    }

    /* handle remaining items */
    while( inA < endA ) {
#ifndef USE_AVX_512
        __m256d regAl1, regAl2, regAh1, regAh2;
        __m256d regBl1 = outreg2l1;
        __m256d regBl2 = outreg2l2;
        __m256d regBh1 = outreg2h1;
        __m256d regBh2 = outreg2h2;

        LOAD8U(regAl1, regAl2, inA);
        LOAD8U(regAh1, regAh2, ((block8 *)(inA) + 1));
        inA ++;

        BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                        outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                        regAl1, regAl2, regAh1, regAh2,
                        regBl1, regBl2, regBh1, regBh2);

        /* store outreg1 */
        STORE8U(outp, outreg1l1, outreg1l2);
        STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
        __m512i regAl, regAh;
        __m512i regBl = outreg2l;
        __m512i regBh = outreg2h;

        LOAD8U(regAl, inA);
        LOAD8U(regAh, ((block8 *)(inA) + 1));
        inA ++;

        BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                        regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8U(outp, outreg1l);
        STORE8U(((block8 *)outp + 1), outreg1h);

#endif
        outp ++;
    }

    while( inB < endB ) {

#ifndef USE_AVX_512
        __m256d regBl1, regBl2, regBh1, regBh2;
        __m256d regAl1 = outreg2l1;
        __m256d regAl2 = outreg2l2;
        __m256d regAh1 = outreg2h1;
        __m256d regAh2 = outreg2h2;

        LOAD8U(regBl1, regBl2, inB);
        LOAD8U(regBh1, regBh2, ((block8 *)inB + 1));
        inB ++;

        BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                        outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                        regAl1, regAl2, regAh1, regAh2,
                        regBl1, regBl2, regBh1, regBh2);
        
        /* store outreg1 */
        STORE8U(outp, outreg1l1, outreg1l2);
        STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
        __m512i regBl, regBh;
        __m512i regAl = outreg2l;
        __m512i regAh = outreg2h;

        LOAD8U(regBl, inB);
        LOAD8U(regBh, ((block8 *)inB + 1));
        inB ++;

        BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                        regAl, regAh, regBl, regBh);
        
        /* store outreg1 */
        STORE8U(outp, outreg1l);
        STORE8U(((block8 *)outp + 1), outreg1h);
#endif
        outp ++;
    }

    /* store the last remaining register values */
#ifndef USE_AVX_512
    STORE8U(outp, outreg2l1, outreg2l2);
    STORE8U(((block8 *)outp + 1), outreg2h1, outreg2h2);
#else
    STORE8U(outp, outreg2l);
    STORE8U(((block8 *)outp + 1), outreg2h);
#endif
}

/** 
 * Merge two sorted arrays to a final output using 16-way AVX bitonic merge. 
 * 
 * @param inpA input array A
 * @param inpB input array B
 * @param Out  output array
 * @param lenA size of A
 * @param lenB size of B
 */
inline void __attribute__((always_inline))
merge16_varlen(int64_t * restrict inpA, 
               int64_t * restrict inpB,
               int64_t * restrict Out, 
               const uint32_t lenA, 
               const uint32_t lenB) 
{
    uint32_t lenA16 = lenA & ~0xF, lenB16 = lenB & ~0xF;
    uint32_t ai = 0, bi = 0;

    int64_t * out = Out;

    if(lenA16 > 16 && lenB16 > 16) {

        register block16 * inA  = (block16 *) inpA;
        register block16 * inB  = (block16 *) inpB;
        block16 * const    endA = (block16 *) (inpA + lenA) - 1;
        block16 * const    endB = (block16 *) (inpB + lenB) - 1;

        block16 * outp = (block16 *) out;

        register block16 * next = inB;

#ifndef USE_AVX_512
        __m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
        __m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;

        __m256d regAl1, regAl2, regAh1, regAh2;
        __m256d regBl1, regBl2, regBh1, regBh2;

        LOAD8U(regAl1, regAl2, inA);
        LOAD8U(regAh1, regAh2, ((block8 *)(inA) + 1));
        inA ++;

        LOAD8U(regBl1, regBl2, inB);
        LOAD8U(regBh1, regBh2, ((block8 *)(inB) + 1));
        inB ++;

        BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                        outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                        regAl1, regAl2, regAh1, regAh2,
                        regBl1, regBl2, regBh1, regBh2);

        /* store outreg1 */
        STORE8U(outp, outreg1l1, outreg1l2);
        STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
        __m512i outreg1l, outreg1h;
        __m512i outreg2l, outreg2h;

        __m512i regAl, regAh;
        __m512i regBl, regBh;

        LOAD8U(regAl, inA);
        LOAD8U(regAh, ((block8 *)(inA) + 1));
        inA ++;

        LOAD8U(regBl, inB);
        LOAD8U(regBh, ((block8 *)(inB) + 1));
        inB ++;

        BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                        regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8U(outp, outreg1l);
        STORE8U(((block8 *)outp + 1), outreg1h);
#endif
        outp ++;

        while( inA < endA && inB < endB ) {

            /** The inline assembly below does exactly the following code: */
            /* Option 3: with assembly */
            IFELSECONDMOVE(next, inA, inB, 128);

#ifndef USE_AVX_512        
            regAl1 = outreg2l1;
            regAl2 = outreg2l2;
            regAh1 = outreg2h1;
            regAh2 = outreg2h2;

            LOAD8U(regBl1, regBl2, next);
            LOAD8U(regBh1, regBh2, ((block8 *)next + 1));
        
            BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                            outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                            regAl1, regAl2, regAh1, regAh2,
                            regBl1, regBl2, regBh1, regBh2);

            /* store outreg1 */
            STORE8U(outp, outreg1l1, outreg1l2);
            STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
            regAl = outreg2l;
            regAh = outreg2h;

            LOAD8U(regBl, next);
            LOAD8U(regBh, ((block8 *)next + 1));
        
            BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                            regAl, regAh, regBl, regBh);

            /* store outreg1 */
            STORE8U(outp, outreg1l);
            STORE8U(((block8 *)outp + 1), outreg1h);
#endif
            outp ++;
        }

        /* flush the register to one of the lists */
        int64_t hireg[4] __attribute__((aligned(16)));

#ifndef USE_AVX_512       
        _mm256_store_pd ( (double *)hireg, outreg2h2);
#else
        __m256i outreg2h2 = _mm512_extracti64x4_epi64(outreg2h, 1);
        _mm256_store_epi64 ((int64_t *)hireg, outreg2h2);
#endif
        if(*((int64_t *)inA) >= *((int64_t*)(hireg+3))) {
            /* store the last remaining register values to A */
            inA --;
#ifndef USE_AVX_512
            STORE8U(inA, outreg2l1, outreg2l2);
            STORE8U(((block8 *)inA + 1), outreg2h1, outreg2h2);
#else
            STORE8U(inA, outreg2l);
            STORE8U(((block8 *)inA + 1), outreg2h);
#endif
        }
        else {
            /* store the last remaining register values to B */
            inB --;
#ifndef USE_AVX_512
            STORE8U(inB, outreg2l1, outreg2l2);
            STORE8U(((block8 *)inB + 1), outreg2h1, outreg2h2);
#else
            STORE8U(inB, outreg2l);
            STORE8U(((block8 *)inB + 1), outreg2h);
#endif
        }

        ai = ((int64_t *)inA - inpA);
        bi = ((int64_t *)inB - inpB);

        inpA = (int64_t *)inA;
        inpB = (int64_t *)inB;
        out  = (int64_t *)outp;
    }

    /* serial-merge */
    while(ai < lenA && bi < lenB){
        int64_t * in = inpB;
        uint32_t cmp = (*inpA < *inpB);
        uint32_t notcmp = !cmp;

        ai += cmp;
        bi += notcmp;
            
        if(cmp)
            in = inpA;
            
        *out = *in;
        out ++;
        inpA += cmp;
        inpB += notcmp;
    }

    if(ai < lenA) {
        /* if A has any more items to be output */

        if((lenA - ai) >= 8) {
            /* if A still has some times to be output with AVX */
            uint32_t lenA8 = ((lenA-ai) & ~0x7);
            register block8 * inA  = (block8 *) inpA;
            block8 * const    endA = (block8 *) (inpA + lenA8);
            block8 * outp = (block8 *) out;
            
            while(inA < endA) {
#ifndef USE_AVX_512
                __m256d regAl, regAh;
                LOAD8U(regAl, regAh, inA);
                STORE8U(outp, regAl, regAh);
#else
                __m512i regA;
                LOAD8U(regA, inA);
                STORE8U(outp, regA);
#endif
                outp ++;
                inA ++;
            }

            ai   += ((int64_t*)inA - inpA);
            inpA  = (int64_t *)inA;
            out   = (int64_t *)outp;
        }

        while(ai < lenA) {
            *out = *inpA;
            ai++;
            out++;
            inpA++;
        }
    } 
    else if(bi < lenB) {
        /* if B has any more items to be output */

        if((lenB - bi) >= 8) {
            /* if B still has some times to be output with AVX */
            uint32_t lenB8 = ((lenB-bi) & ~0x7);
            register block8 * inB  = (block8 *) inpB;
            block8 * const    endB = (block8 *) (inpB + lenB8);
            block8 * outp = (block8 *) out;
            
            while(inB < endB) {
#ifndef USE_AVX_512
                __m256d regBl, regBh;
                LOAD8U(regBl, regBh, inB);
                STORE8U(outp, regBl, regBh);
#else
                __m512i regB;
                LOAD8U(regB, inB);
                STORE8U(outp, regB);
#endif                
                outp ++;
                inB ++;
            }

            bi   += ((int64_t*)inB - inpB);
            inpB  = (int64_t *)inB;
            out   = (int64_t *)outp;
        }

        while(bi < lenB) {
            *out = *inpB;
            bi++;
            out++;
            inpB++;
        }
    }
}

/** 
 * Merge two sorted arrays to a final output using 8-way AVX bitonic merge. 
 * 
 * @param inpA input array A
 * @param inpB input array B
 * @param Out  output array
 * @param lenA size of A
 * @param lenB size of B
 */
inline void __attribute__((always_inline))
merge8_varlen(int64_t * restrict inpA, 
              int64_t * restrict inpB,
              int64_t * restrict Out, 
              const uint32_t lenA, 
              const uint32_t lenB) 
{
    uint32_t lenA8 = lenA & ~0x7, lenB8 = lenB & ~0x7;
    uint32_t ai = 0, bi = 0;

    int64_t * out = Out;

    if(lenA8 > 8 && lenB8 > 8) {

        register block8 * inA  = (block8 *) inpA;
        register block8 * inB  = (block8 *) inpB;
        block8 * const    endA = (block8 *) (inpA + lenA) - 1;
        block8 * const    endB = (block8 *) (inpB + lenB) - 1;

        block8 * outp = (block8 *) out;

        register block8 * next = inB;

#ifndef USE_AVX_512
        register __m256d outreg1l, outreg1h;
        register __m256d outreg2l, outreg2h;

        register __m256d regAl, regAh;
        register __m256d regBl, regBh;

        LOAD8U(regAl, regAh, inA);
        LOAD8U(regBl, regBh, next);

        inA ++;
        inB ++;

        BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                       regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8U(outp, outreg1l, outreg1h);
#else
        register __m512i outreg1;
        register __m512i outreg2;

        register __m512i regA;
        register __m512i regB;

        LOAD8U(regA, inA);
        LOAD8U(regB, next);

        inA ++;
        inB ++;

        BITONIC_MERGE8(outreg1, outreg2, regA, regB);

        /* store outreg1 */
        STORE8U(outp, outreg1);
#endif

        outp ++;

        while( inA < endA && inB < endB ) {

            /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
            IFELSECONDMOVE(next, inA, inB, 64);

#ifndef USE_AVX_512
            regAl = outreg2l;
            regAh = outreg2h;
            LOAD8U(regBl, regBh, next);
        
            BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                           regAl, regAh, regBl, regBh);
        
            /* store outreg1 */
            STORE8U(outp, outreg1l, outreg1h);
#else
            regA = outreg2;
            LOAD8U(regB, next);
        
            BITONIC_MERGE8(outreg1, outreg2, regA, regB);
        
            /* store outreg1 */
            STORE8U(outp, outreg1);
#endif
            outp ++;
        }

        /* flush the register to one of the lists */
        int64_t hireg[4] __attribute__((aligned(16)));

#ifndef USE_AVX_512
        _mm256_store_pd ( (double *)hireg, outreg2h);
#else
        __m256i outreg2h = _mm512_extracti64x4_epi64(outreg2, 1);
        _mm256_store_epi64 ((int64_t *)hireg, outreg2h);
#endif
        if(*((int64_t *)inA) >= *((int64_t*)(hireg+3))) {
            /* store the last remaining register values to A */
            inA --;
#ifndef USE_AVX_512            
            STORE8U(inA, outreg2l, outreg2h);
#else
            STORE8U(inA, outreg2);
#endif
        }
        else {
            /* store the last remaining register values to B */
            inB --;
#ifndef USE_AVX_512            
            STORE8U(inB, outreg2l, outreg2h);
#else
            STORE8U(inB, outreg2);
#endif
        }

        ai = ((int64_t *)inA - inpA);
        bi = ((int64_t *)inB - inpB);

        inpA = (int64_t *)inA;
        inpB = (int64_t *)inB;
        out  = (int64_t *)outp;
    }

    /* serial-merge */
    while(ai < lenA && bi < lenB){
        int64_t * in = inpB;
        uint32_t cmp = (*inpA < *inpB);
        uint32_t notcmp = !cmp;

        ai += cmp;
        bi += notcmp;
            
        if(cmp)
            in = inpA;
            
        *out = *in;
        out ++;
        inpA += cmp;
        inpB += notcmp;
    }

    if(ai < lenA) {
        /* if A has any more items to be output */

        if((lenA - ai) >= 8) {
            /* if A still has some times to be output with AVX */
            uint32_t lenA8_ = ((lenA-ai) & ~0x7);
            register block8 * inA  = (block8 *) inpA;
            block8 * const    endA = (block8 *) (inpA + lenA8_);
            block8 * outp = (block8 *) out;
            
            while(inA < endA) {
#ifndef USE_AVX_512                
                __m256d regAl, regAh;
                LOAD8U(regAl, regAh, inA);
                STORE8U(outp, regAl, regAh);
#else
                __m512i regA;
                LOAD8U(regA, inA);
                STORE8U(outp, regA);
#endif                
                outp ++;
                inA ++;
            }

            ai   += ((int64_t*)inA - inpA);
            inpA  = (int64_t *)inA;
            out   = (int64_t *)outp;
        }

        while(ai < lenA) {
            *out = *inpA;
            ai++;
            out++;
            inpA++;
        }
    } 
    else if(bi < lenB) {
        /* if B has any more items to be output */

        if((lenB - bi) >= 8) {
            /* if B still has some times to be output with AVX */
            uint32_t lenB8_ = ((lenB-bi) & ~0x7);
            register block8 * inB  = (block8 *) inpB;
            block8 * const    endB = (block8 *) (inpB + lenB8_);
            block8 * outp = (block8 *) out;
            
            while(inB < endB) {

#ifndef USE_AVX_512
                __m256d regBl, regBh;
                LOAD8U(regBl, regBh, inB);
                STORE8U(outp, regBl, regBh);
#else
                __m512i regB;
                LOAD8U(regB, inB);
                STORE8U(outp, regB);
#endif
                outp ++;
                inB ++;
            }

            bi   += ((int64_t*)inB - inpB);
            inpB  = (int64_t *)inB;
            out   = (int64_t *)outp;
        }

        while(bi < lenB) {
            *out = *inpB;
            bi++;
            out++;
            inpB++;
        }
    }
}

/** 
 * Merge two sorted arrays to a final output using 8-way AVX bitonic merge. 
 * 
 * @param inpA input array A
 * @param inpB input array B
 * @param Out  output array
 * @param lenA size of A
 * @param lenB size of B
 */
inline void __attribute__((always_inline))
merge8_varlen_aligned(int64_t * restrict inpA, 
                      int64_t * restrict inpB,
                      int64_t * restrict Out, 
                      const uint32_t lenA, 
                      const uint32_t lenB) 
{
    uint32_t lenA8 = lenA & ~0x7, lenB8 = lenB & ~0x7;
    uint32_t ai = 0, bi = 0;

    int64_t * out = Out;

    if(lenA8 > 8 && lenB8 > 8) {

        register block8 * inA  = (block8 *) inpA;
        register block8 * inB  = (block8 *) inpB;
        block8 * const    endA = (block8 *) (inpA + lenA) - 1;
        block8 * const    endB = (block8 *) (inpB + lenB) - 1;

        block8 * outp = (block8 *) out;

        register block8 * next = inB;

#ifndef USE_AVX_512
        register __m256d outreg1l, outreg1h;
        register __m256d outreg2l, outreg2h;

        register __m256d regAl, regAh;
        register __m256d regBl, regBh;

        LOAD8(regAl, regAh, inA);
        LOAD8(regBl, regBh, next);

        inA ++;
        inB ++;

        BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                       regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8(outp, outreg1l, outreg1h);
#else
        register __m512i outreg1;
        register __m512i outreg2;

        register __m512i regA;
        register __m512i regB;

        LOAD8(regA, inA);
        LOAD8(regB, next);

        inA ++;
        inB ++;

        BITONIC_MERGE8(outreg1, outreg2, regA, regB);

        /* store outreg1 */
        STORE8(outp, outreg1);
#endif
        outp ++;

        while( inA < endA && inB < endB ) {

            /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
            IFELSECONDMOVE(next, inA, inB, 64);

#ifndef USE_AVX_512
            regAl = outreg2l;
            regAh = outreg2h;
            LOAD8(regBl, regBh, next);
        
            BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                           regAl, regAh, regBl, regBh);
        
            /* store outreg1 */
            STORE8(outp, outreg1l, outreg1h);
#else
            regA = outreg2;
            LOAD8(regB, next);
        
            BITONIC_MERGE8(outreg1, outreg2, regA, regB);
        
            /* store outreg1 */
            STORE8(outp, outreg1);
#endif
            outp ++;
        }

        /* flush the register to one of the lists */
        int64_t hireg[4] __attribute__((aligned(32)));
#ifndef USE_AVX_512
        _mm256_store_pd ( (double *)hireg, outreg2h);
#else
        __m256i outreg2h = _mm512_extracti64x4_epi64(outreg2, 1);
        _mm256_store_epi64 ((int64_t *)hireg, outreg2h);
#endif
        if(*((int64_t *)inA) >= *((int64_t*)(hireg+3))) {
            /* store the last remaining register values to A */
            inA --;
#ifndef USE_AVX_512
            STORE8(inA, outreg2l, outreg2h);
#else
            STORE8(inA, outreg2);
#endif
        }
        else {
            /* store the last remaining register values to B */
            inB --;
#ifndef USE_AVX_512
            STORE8(inB, outreg2l, outreg2h);
#else
            STORE8(inB, outreg2);
#endif
        }

        ai = ((int64_t *)inA - inpA);
        bi = ((int64_t *)inB - inpB);

        inpA = (int64_t *)inA;
        inpB = (int64_t *)inB;
        out  = (int64_t *)outp;
    }

    /* serial-merge */
    while(ai < lenA && bi < lenB){
        int64_t * in = inpB;
        uint32_t cmp = (*inpA < *inpB);
        uint32_t notcmp = !cmp;

        ai += cmp;
        bi += notcmp;
            
        if(cmp)
            in = inpA;
            
        *out = *in;
        out ++;
        inpA += cmp;
        inpB += notcmp;
    }

    if(ai < lenA) {
        /* if A has any more items to be output */

        if((lenA - ai) >= 8) {
            /* if A still has some times to be output with AVX */
            uint32_t lenA8_ = ((lenA-ai) & ~0x7);
            register block8 * inA  = (block8 *) inpA;
            block8 * const    endA = (block8 *) (inpA + lenA8_);
            block8 * outp = (block8 *) out;
            
            while(inA < endA) {
#ifndef USE_AVX_512
                __m256d regAl, regAh;
                LOAD8U(regAl, regAh, inA);
                STORE8U(outp, regAl, regAh);
#else
                __m512i regA;
                LOAD8U(regA, inA);
                STORE8U(outp, regA);
#endif
                outp ++;
                inA ++;
            }

            ai   += ((int64_t*)inA - inpA);
            inpA  = (int64_t *)inA;
            out   = (int64_t *)outp;
        }

        while(ai < lenA) {
            *out = *inpA;
            ai++;
            out++;
            inpA++;
        }
    } 
    else if(bi < lenB) {
        /* if B has any more items to be output */

        if((lenB - bi) >= 8) {
            /* if B still has some times to be output with AVX */
            uint32_t lenB8_ = ((lenB-bi) & ~0x7);
            register block8 * inB  = (block8 *) inpB;
            block8 * const    endB = (block8 *) (inpB + lenB8_);
            block8 * outp = (block8 *) out;
            
            while(inB < endB) {
#ifndef USE_AVX_512
                __m256d regBl, regBh;
                LOAD8U(regBl, regBh, inB);
                STORE8U(outp, regBl, regBh);
#else
                __m512i regB;
                LOAD8U(regB, inB);
                STORE8U(outp, regB);
#endif  
                outp ++;
                inB ++;
            }

            bi   += ((int64_t*)inB - inpB);
            inpB  = (int64_t *)inB;
            out   = (int64_t *)outp;
        }

        while(bi < lenB) {
            *out = *inpB;
            bi++;
            out++;
            inpB++;
        }
    }
}

/** 
 * Merge two sorted arrays to a final output using 4-way AVX bitonic merge. 
 * 
 * @param inpA input array A
 * @param inpB input array B
 * @param Out  output array
 * @param lenA size of A
 * @param lenB size of B
 */
inline void __attribute__((always_inline))
merge4_varlen(int64_t * restrict inpA, 
              int64_t * restrict inpB,
              int64_t * restrict Out, 
              const uint32_t lenA, 
              const uint32_t lenB) 
{
    uint32_t lenA4 = lenA & ~0x3, lenB4 = lenB & ~0x3;
    uint32_t ai = 0, bi = 0;

    int64_t * out = Out;

    if(lenA4 > 4 && lenB4 > 4) {

        register block4 * inA  = (block4 *) inpA;
        register block4 * inB  = (block4 *) inpB;
        block4 * const    endA = (block4 *) (inpA + lenA) - 1;
        block4 * const    endB = (block4 *) (inpB + lenB) - 1;

        block4 * outp = (block4 *) out;

        register block4 * next = inB;
#ifndef USE_AVX_512
        register __m256d outreg1;
        register __m256d outreg2;

        register __m256d regA = _mm256_loadu_pd((double const *) inA);
        register __m256d regB = _mm256_loadu_pd((double const *) next);

        inA ++;
        inB ++;

        BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
        _mm256_storeu_pd((double *) outp, outreg1);
#else
        register __m256i outreg1;
        register __m256i outreg2;

//#ifndef DEVELOPMENT_MODE
//        register __m256i regA = _mm256_loadu_epi64((int64_t const *) inA);
//        register __m256i regB = _mm256_loadu_epi64((int64_t const *) next);
//#else
        register __m256i regA = _mm256_mask_loadu_epi64(regA, 0xFF, (int64_t const *) inA);
        register __m256i regB = _mm256_mask_loadu_epi64(regB, 0xFF, (int64_t const *) next);
//#endif
        inA ++;
        inB ++;

        BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
#ifndef DEVELOPMENT_MODE
        _mm256_storeu_epi64((int64_t *) outp, outreg1);
#else
        _mm256_mask_storeu_epi64((int64_t *) outp, 0xFF, outreg1);
#endif

#endif
        outp ++;

        while( inA < endA && inB < endB ) {
            /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
            IFELSECONDMOVE(next, inA, inB, 32);

#ifndef USE_AVX_512
            regA = outreg2;
            regB = _mm256_loadu_pd((double const *) next);
        
            BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
            /* store outreg1 */
            _mm256_storeu_pd((double *) outp, outreg1);
#else
            regA = outreg2;
//#ifndef DEVELOPMENT_MODE
//            regB = _mm256_loadu_epi64((int64_t const *) next);
//#else
            regB = _mm256_mask_loadu_epi64(regB, 0xFF, (int64_t const *) next);
//#endif
            BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
            /* store outreg1 */
#ifndef DEVELOPMENT_MODE
            _mm256_storeu_epi64((int64_t *) outp, outreg1);
#else
            _mm256_mask_storeu_epi64((int64_t *) outp, 0xFF, outreg1);
#endif
#endif
            outp ++;
        }

        /* flush the register to one of the lists */
        int64_t hireg[4] __attribute__((aligned(16)));
#ifndef USE_AVX_512        
        _mm256_store_pd ( (double *)hireg, outreg2);
#else
        _mm256_store_epi64 ((int64_t *)hireg, outreg2);
#endif
        if(*((int64_t *)inA) >= *((int64_t*)(hireg+3))) {
            /* store the last remaining register values to A */
            inA --;
#ifndef USE_AVX_512
            _mm256_storeu_pd((double *) inA, outreg2);
#else
#ifndef DEVELOPMENT_MODE
            _mm256_storeu_epi64 ((int64_t *) inA, outreg2);
#else
            _mm256_mask_storeu_epi64((int64_t *) inA, 0xF, outreg2); 
#endif
#endif
        }
        else {
            /* store the last remaining register values to B */
            inB --;
#ifndef USE_AVX_512
            _mm256_storeu_pd((double *) inB, outreg2);
#else
#ifndef DEVELOPMENT_MODE
            _mm256_storeu_epi64((int64_t *) inB, outreg2);
#else
            _mm256_mask_storeu_epi64((int64_t *) inB, 0xF, outreg2);
#endif
#endif
        }

        ai = ((int64_t *)inA - inpA);
        bi = ((int64_t *)inB - inpB);

        inpA = (int64_t *)inA;
        inpB = (int64_t *)inB;
        out  = (int64_t *)outp;
    }

    /* serial-merge */
    while(ai < lenA && bi < lenB){
        int64_t * in = inpB;
        uint32_t cmp = (*inpA < *inpB);
        uint32_t notcmp = !cmp;

        ai += cmp;
        bi += notcmp;
            
        if(cmp)
            in = inpA;
            
        *out = *in;
        out ++;
        inpA += cmp;
        inpB += notcmp;
    }

    if(ai < lenA) {
        /* if A has any more items to be output */

        if((lenA - ai) >= 8) {
            /* if A still has some times to be output with AVX */
            uint32_t lenA8 = ((lenA-ai) & ~0x7);
            register block8 * inA  = (block8 *) inpA;
            block8 * const    endA = (block8 *) (inpA + lenA8);
            block8 * outp = (block8 *) out;
            
            while(inA < endA) {
#ifndef USE_AVX_512                
                __m256d regAl, regAh;
                LOAD8U(regAl, regAh, inA);
                STORE8U(outp, regAl, regAh);
#else
                __m512i regA;
                LOAD8U(regA, inA);
                STORE8U(outp, regA);
#endif               
                outp ++;
                inA ++;
            }

            ai   += ((int64_t*)inA - inpA);
            inpA  = (int64_t *)inA;
            out   = (int64_t *)outp;
        }

        while(ai < lenA) {
            *out = *inpA;
            ai++;
            out++;
            inpA++;
        }
    } 
    else if(bi < lenB) {
        /* if B has any more items to be output */

        if((lenB - bi) >= 8) {
            /* if B still has some times to be output with AVX */
            uint32_t lenB8 = ((lenB-bi) & ~0x7);
            register block8 * inB  = (block8 *) inpB;
            block8 * const    endB = (block8 *) (inpB + lenB8);
            block8 * outp = (block8 *) out;
            
            while(inB < endB) {
#ifndef USE_AVX_512
                __m256d regBl, regBh;
                LOAD8U(regBl, regBh, inB);
                STORE8U(outp, regBl, regBh);
#else
                __m512i regB;
                LOAD8U(regB, inB);
                STORE8U(outp, regB);
#endif            
                outp ++;
                inB ++;
            }

            bi   += ((int64_t*)inB - inpB);
            inpB  = (int64_t *)inB;
            out   = (int64_t *)outp;
        }

        while(bi < lenB) {
            *out = *inpB;
            bi++;
            out++;
            inpB++;
        }
    }
}

/** aligned version */
inline void __attribute__((always_inline))
merge4_varlen_aligned(int64_t * restrict inpA, 
                      int64_t * restrict inpB,
                      int64_t * restrict Out, 
                      const uint32_t lenA, 
                      const uint32_t lenB) 
{
    uint32_t lenA4 = lenA & ~0x3, lenB4 = lenB & ~0x3;
    uint32_t ai = 0, bi = 0;

    int64_t * out = Out;

    if(lenA4 > 4 && lenB4 > 4) {

        register block4 * inA  = (block4 *) inpA;
        register block4 * inB  = (block4 *) inpB;
        block4 * const    endA = (block4 *) (inpA + lenA) - 1;
        block4 * const    endB = (block4 *) (inpB + lenB) - 1;

        block4 * outp = (block4 *) out;

        register block4 * next = inB;
#ifndef USE_AVX_512
        register __m256d outreg1;
        register __m256d outreg2;

        register __m256d regA = _mm256_load_pd((double const *) inA);
        register __m256d regB = _mm256_load_pd((double const *) next);

        inA ++;
        inB ++;

        BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
        _mm256_store_pd((double *) outp, outreg1);
#else
        register __m256i outreg1;
        register __m256i outreg2;

//#ifndef DEVELOPMENT_MODE
//        register __m256i regA = _mm256_load_epi64((int64_t const *) inA);
//        register __m256i regB = _mm256_load_epi64((int64_t const *) next);
//#else
        register __m256i regA = _mm256_mask_load_epi64(regA, 0xFF, (int64_t const *) inA);
        register __m256i regB = _mm256_mask_load_epi64(regB, 0xFF, (int64_t const *) next);
//#endif
        inA ++;
        inB ++;

        BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
#ifndef DEVELOPMENT_MODE
        _mm256_store_epi64((int64_t *) outp, outreg1);
#else
        _mm256_mask_store_epi64((int64_t *) outp, 0xFF, outreg1);
#endif

#endif
        outp ++;

        while( inA < endA && inB < endB ) {
            /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
            IFELSECONDMOVE(next, inA, inB, 32);

#ifndef USE_AVX_512
            regA = outreg2;
            regB = _mm256_load_pd((double const *) next);
        
            BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
            /* store outreg1 */
            _mm256_store_pd((double *) outp, outreg1);
#else
            regA = outreg2;
//#ifndef DEVELOPMENT_MODE
//            regB = _mm256_load_epi64((int64_t const *) next);
//#else
            regB = _mm256_mask_load_epi64(regB, 0xFF, (int64_t const *) next);
//#endif     
            BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
            /* store outreg1 */
#ifndef DEVELOPMENT_MODE
            _mm256_store_epi64((int64_t *) outp, outreg1);
#else
            _mm256_mask_store_epi64((int64_t *) outp, 0xFF, outreg1);
#endif

#endif
            outp ++;
        }

        /* flush the register to one of the lists */
        int64_t hireg[4] __attribute__((aligned(32)));
#ifndef USE_AVX_512        
        _mm256_store_pd ( (double *)hireg, outreg2);
#else
        _mm256_store_epi64 ( (int64_t *)hireg, outreg2);
#endif
        if(*((int64_t *)inA) >= *((int64_t*)(hireg+3))) {
            /* store the last remaining register values to A */
            inA --;
#ifndef USE_AVX_512
            _mm256_store_pd((double *) inA, outreg2);
#else
            _mm256_store_epi64((int64_t *) inA, outreg2);
#endif
        }
        else {
            /* store the last remaining register values to B */
            inB --;
#ifndef USE_AVX_512
            _mm256_store_pd((double *) inB, outreg2);
#else
            _mm256_store_epi64((int64_t *) inB, outreg2);
#endif
        }

        ai = ((int64_t *)inA - inpA);
        bi = ((int64_t *)inB - inpB);

        inpA = (int64_t *)inA;
        inpB = (int64_t *)inB;
        out  = (int64_t *)outp;
    }

    /* serial-merge */
    while(ai < lenA && bi < lenB){
        int64_t * in = inpB;
        uint32_t cmp = (*inpA < *inpB);
        uint32_t notcmp = !cmp;

        ai += cmp;
        bi += notcmp;
            
        if(cmp)
            in = inpA;
            
        *out = *in;
        out ++;
        inpA += cmp;
        inpB += notcmp;
    }

    if(ai < lenA) {
        /* if A has any more items to be output */

        if((lenA - ai) >= 8) {
            /* if A still has some times to be output with AVX */
            uint32_t lenA8 = ((lenA-ai) & ~0x7);
            register block8 * inA  = (block8 *) inpA;
            block8 * const    endA = (block8 *) (inpA + lenA8);
            block8 * outp = (block8 *) out;
            
            while(inA < endA) {
#ifndef USE_AVX_512                
                __m256d regAl, regAh;
                LOAD8U(regAl, regAh, inA);
                STORE8U(outp, regAl, regAh);
#else
                __m512i regA;
                LOAD8U(regA, inA);
                STORE8U(outp, regA);
#endif               
                outp ++;
                inA ++;
            }

            ai   += ((int64_t*)inA - inpA);
            inpA  = (int64_t *)inA;
            out   = (int64_t *)outp;
        }

        while(ai < lenA) {
            *out = *inpA;
            ai++;
            out++;
            inpA++;
        }
    } 
    else if(bi < lenB) {
        /* if B has any more items to be output */

        if((lenB - bi) >= 8) {
            /* if B still has some times to be output with AVX */
            uint32_t lenB8 = ((lenB-bi) & ~0x7);
            register block8 * inB  = (block8 *) inpB;
            block8 * const    endB = (block8 *) (inpB + lenB8);
            block8 * outp = (block8 *) out;
            
            while(inB < endB) {

#ifndef USE_AVX_512
                __m256d regBl, regBh;
                LOAD8U(regBl, regBh, inB);
                STORE8U(outp, regBl, regBh);
#else
                __m512i regB;
                LOAD8U(regB, inB);
                STORE8U(outp, regB);
#endif
                outp ++;
                inB ++;
            }

            bi   += ((int64_t*)inB - inpB);
            inpB  = (int64_t *)inB;
            out   = (int64_t *)outp;
        }

        while(bi < lenB) {
            *out = *inpB;
            bi++;
            out++;
            inpB++;
        }
    }
}

inline void __attribute((always_inline))
inregister_sort_keyval32(int64_t * items, int64_t * output)
{
 #ifdef USE_AVX_512
 #ifndef DEVELOPMENT_MODE
    __m512i ra = _mm512_loadu_epi64 ((int64_t const *)(items));
    __m512i rb = _mm512_loadu_epi64 ((int64_t const *)(items + 8));
    __m512i rc = _mm512_loadu_epi64 ((int64_t const *)(items + 16));
    __m512i rd = _mm512_loadu_epi64 ((int64_t const *)(items + 24));
#else
    __m512i ra = _mm512_mask_loadu_epi64 (ra, 0xFF, (int64_t const *)(items));
    __m512i rb = _mm512_mask_loadu_epi64 (rb, 0xFF, (int64_t const *)(items + 8));
    __m512i rc = _mm512_mask_loadu_epi64 (rc, 0xFF, (int64_t const *)(items + 16));
    __m512i rd = _mm512_mask_loadu_epi64 (rd, 0xFF, (int64_t const *)(items + 24)); 
#endif
    /* odd-even sorting network begins */
    /* 1st level of comparisons */
    __m512i ra1 = _mm512_min_epi64(ra, rb);
    __m512i rb1 = _mm512_max_epi64(ra, rb);

    __m512i rc1 = _mm512_min_epi64(rc, rd);
    __m512i rd1 = _mm512_max_epi64(rc, rd);

    /* 2nd level of comparisons */
    rb = _mm512_min_epi64(rb1, rd1);
    rd = _mm512_max_epi64(rb1, rd1);

    /* 3rd level of comparisons */
    __m512i ra2 = _mm512_min_epi64(ra1, rc1);
    __m512i rc2 = _mm512_max_epi64(ra1, rc1);

    /* 4th level of comparisons */
    __m512i rb3 = _mm512_min_epi64(rb, rc2);
    __m512i rc3 = _mm512_max_epi64(rb, rc2);

    /* shuffling using AVX 256 */
    //first batch
    __m256i ra2_1 = _mm512_extracti64x4_epi64(ra2, 0);
    __m256i rb3_1 = _mm512_extracti64x4_epi64(rb3, 0);
    __m256i rc3_1 = _mm512_extracti64x4_epi64(rc3, 0);
    __m256i rd_1 = _mm512_extracti64x4_epi64(rd, 0);

    __m256i ra3_1 = _mm256_unpacklo_epi64(ra2_1, rb3_1);
    __m256i rb4_1 = _mm256_unpackhi_epi64(ra2_1, rb3_1);

    __m256i rc4_1 = _mm256_unpacklo_epi64(rc3_1, rd_1);
    __m256i rd3_1 = _mm256_unpackhi_epi64(rc3_1, rd_1);

    __m256i ra4_1 = _mm256_permute2f128_si256(ra3_1, rc4_1, 0x20);
    __m256i rc5_1 = _mm256_permute2f128_si256(ra3_1, rc4_1, 0x31);

    __m256i rb5_1 = _mm256_permute2f128_si256(rb4_1, rd3_1, 0x20);
    __m256i rd4_1 = _mm256_permute2f128_si256(rb4_1, rd3_1, 0x31);

#ifndef DEVELOPMENT_MODE
    _mm256_storeu_epi64((int64_t *) output, ra4_1);
    _mm256_storeu_epi64((int64_t *) (output + 4), rb5_1);
    _mm256_storeu_epi64((int64_t *) (output + 8), rc5_1);
    _mm256_storeu_epi64((int64_t *) (output + 12), rd4_1);
#else
    _mm256_mask_storeu_epi64((int64_t *) output, 0xF, ra4_1);
    _mm256_mask_storeu_epi64((int64_t *) (output + 4), 0xF, rb5_1);
    _mm256_mask_storeu_epi64((int64_t *) (output + 8), 0xF, rc5_1);
    _mm256_mask_storeu_epi64((int64_t *) (output + 12), 0xF, rd4_1);
#endif
    //second batch
    __m256i ra2_2 = _mm512_extracti64x4_epi64(ra2, 1);
    __m256i rb3_2 = _mm512_extracti64x4_epi64(rb3, 1);
    __m256i rc3_2 = _mm512_extracti64x4_epi64(rc3, 1);
    __m256i rd_2 = _mm512_extracti64x4_epi64(rd, 1);

    __m256i ra3_2 = _mm256_unpacklo_epi64(ra2_2, rb3_2);
    __m256i rb4_2 = _mm256_unpackhi_epi64(ra2_2, rb3_2);

    __m256i rc4_2 = _mm256_unpacklo_epi64(rc3_2, rd_2);
    __m256i rd3_2 = _mm256_unpackhi_epi64(rc3_2, rd_2);

    __m256i ra4_2 = _mm256_permute2f128_si256(ra3_2, rc4_2, 0x20);
    __m256i rc5_2 = _mm256_permute2f128_si256(ra3_2, rc4_2, 0x31);

    __m256i rb5_2 = _mm256_permute2f128_si256(rb4_2, rd3_2, 0x20);
    __m256i rd4_2 = _mm256_permute2f128_si256(rb4_2, rd3_2, 0x31);

#ifndef DEVELOPMENT_MODE
    _mm256_storeu_epi64((int64_t *) (output + 16) , ra4_2);
    _mm256_storeu_epi64((int64_t *) (output + 20), rb5_2);
    _mm256_storeu_epi64((int64_t *) (output + 24), rc5_2);
    _mm256_storeu_epi64((int64_t *) (output + 28), rd4_2);
#else
    _mm256_mask_storeu_epi64((int64_t *) (output + 16), 0xF, ra4_2);
    _mm256_mask_storeu_epi64((int64_t *) (output + 20), 0xF, rb5_2);
    _mm256_mask_storeu_epi64((int64_t *) (output + 24), 0xF, rc5_2);
    _mm256_mask_storeu_epi64((int64_t *) (output + 28), 0xF, rd4_2);
#endif
#else
/* IACA_START */
    __m256d ra = _mm256_loadu_pd ((double const *)(items));
    __m256d rb = _mm256_loadu_pd ((double const *)(items + 4));
    __m256d rc = _mm256_loadu_pd ((double const *)(items + 8));
    __m256d rd = _mm256_loadu_pd ((double const *)(items + 12));

    /* odd-even sorting network begins */
    /* 1st level of comparisons */
    __m256d ra1 = _mm256_min_pd(ra, rb);
    __m256d rb1 = _mm256_max_pd(ra, rb);

    __m256d rc1 = _mm256_min_pd(rc, rd);
    __m256d rd1 = _mm256_max_pd(rc, rd);

    /* 2nd level of comparisons */
    rb = _mm256_min_pd(rb1, rd1);
    rd = _mm256_max_pd(rb1, rd1);

    /* 3rd level of comparisons */
    __m256d ra2 = _mm256_min_pd(ra1, rc1);
    __m256d rc2 = _mm256_max_pd(ra1, rc1);

    /* 4th level of comparisons */
    __m256d rb3 = _mm256_min_pd(rb, rc2);
    __m256d rc3 = _mm256_max_pd(rb, rc2);

    /* results are in ra2, rb3, rc3, rd */
    /**
     * Initial data and transposed data looks like following:
     *  a2={ x1  x2  x3  x4  }                      a4={ x1 x5 x9  x13 }
     *  b3={ x5  x6  x7  x8  }  === Transpose ===>  b5={ x2 x6 x10 x14 }
     *  c3={ x9  x10 x11 x12 }                      c5={ x3 x7 x11 x15 }
     *  d2={ x13 x14 x15 x16 }                      d4={ x4 x8 x12 x16 }
     */
    /* shuffle x2 and x5 - shuffle x4 and x7 */
    __m256d ra3 = _mm256_unpacklo_pd(ra2, rb3);
    __m256d rb4 = _mm256_unpackhi_pd(ra2, rb3);

    /* shuffle x10 and x13 - shuffle x12 and x15 */
    __m256d rc4 = _mm256_unpacklo_pd(rc3, rd);
    __m256d rd3 = _mm256_unpackhi_pd(rc3, rd);

    /* shuffle (x3,x7) and (x9,x13) pairs */
    __m256d ra4 = _mm256_permute2f128_pd(ra3, rc4, 0x20);
    __m256d rc5 = _mm256_permute2f128_pd(ra3, rc4, 0x31);

    /* shuffle (x4,x8) and (x10,x14) pairs */
    __m256d rb5 = _mm256_permute2f128_pd(rb4, rd3, 0x20);
    __m256d rd4 = _mm256_permute2f128_pd(rb4, rd3, 0x31);

    /* after this, results are in ra4, rb5, rc5, rd4 */
/* IACA_END */
    /* store */
    _mm256_storeu_pd((double *) output, ra4);
    _mm256_storeu_pd((double *) (output + 4), rb5);
    _mm256_storeu_pd((double *) (output + 8), rc5);
    _mm256_storeu_pd((double *) (output + 12), rd4);
#endif

}

inline void __attribute__((always_inline))
avxsort_block(int64_t ** inputptr, int64_t ** outputptr, int BLOCK_SIZE)
{                                                                   
    int64_t * ptrs[2];                                              
    const uint64_t logBSZ = log2(BLOCK_SIZE);                       
                                                                        
    ptrs[0] = *inputptr;                                            
    ptrs[1] = *outputptr;                                           
                                                                        
    /** 1.a) Perform in-register sort to get sorted seq of K(K=4)*/ 
#ifndef USE_AVX_512     
    block16 * inptr = (block16 *) ptrs[0];                          
    block16 * const end = (block16 *) (ptrs[0] + BLOCK_SIZE);       
#else
    block32 * inptr = (block32 *) ptrs[0];                          
    block32 * const end = (block32 *) (ptrs[0] + BLOCK_SIZE);
#endif

    while(inptr < end) {                                            
        inregister_sort_keyval32((int64_t*)inptr, (int64_t*)inptr); 
        inptr ++;                                                   
    }                                                               
                                                                        
                                                                        
    /**                                                             
     * 1.b) for itr <- [(logK) .. (logM - 3)]                       
     *  - Simultaneously merge 4 sequences (using a K by K          
     *  network) of length 2^itr to obtain sorted seq. of 2^{itr+1} 
     */                                                             
    uint64_t j;                                                     
    const uint64_t jend = logBSZ - 2;

    j = 2;
    {
        int ptridx = j & 1;                                         
        int64_t * inp = (int64_t *) ptrs[ptridx];                   
        int64_t * out = (int64_t *) ptrs[ptridx ^ 1];               
        int64_t * const end = (int64_t*) (inp + BLOCK_SIZE);        
                                                                        
        /**                                                         
         *  merge length 2^j lists beginnig at inp and output a     
         *  sorted list of length 2^(j+1) starting at out           
         */                                                         
        const uint64_t inlen  = (1 << j);                           
        const uint64_t outlen = (inlen << 1);                       
                                                                        
        while(inp < end) {                                          
                                                                        
            merge4_eqlen(inp, inp + inlen, out, inlen);             
            inp += outlen;                                          
            out += outlen;                                          
        }                                                           
    }                                                               
    j = 3;
    {
        int ptridx = j & 1;                                         
        int64_t * inp = (int64_t *) ptrs[ptridx];                   
        int64_t * out = (int64_t *) ptrs[ptridx ^ 1];               
        int64_t * const end = (int64_t*) (inp + BLOCK_SIZE);        
                                                                        
        /**                                                         
         *  merge length 2^j lists beginnig at inp and output a     
         *  sorted list of length 2^(j+1) starting at out           
         */                                                         
        const uint64_t inlen  = (1 << j);                           
        const uint64_t outlen = (inlen << 1);                       
                                                                        
        while(inp < end) {                                          
                                                                        
            merge8_eqlen(inp, inp + inlen, out, inlen);             
            inp += outlen;                                          
            out += outlen;                                          
        }                                                           
    }                                                               
    for(j = 4; j < jend; j++) {                                     
        int ptridx = j & 1;                                         
        int64_t * inp = (int64_t *) ptrs[ptridx];                   
        int64_t * out = (int64_t *) ptrs[ptridx ^ 1];               
        int64_t * const end = (int64_t*) (inp + BLOCK_SIZE);        
                                                                        
        /**                                                         
         *  merge length 2^j lists beginnig at inp and output a     
         *  sorted list of length 2^(j+1) starting at out           
         */                                                         
        const uint64_t inlen  = (1 << j);                           
        const uint64_t outlen = (inlen << 1);                       
                                                                        
        while(inp < end) {                                          
                                                                        
            merge16_eqlen(inp, inp + inlen, out, inlen);             
            inp += outlen;                                          
            out += outlen;                                          
                                                                        
            /* TODO: Try following. */                              
            /* simultaneous merge of 4 list pairs */                
            /* merge 4 seqs simultaneously (always >= 4) */         
            /* merge 2 seqs simultaneously (always >= 2) */         
        }                                                           
    }                                                               
                                                                        
    /**                                                             
     * 1.c) for itr = (logM - 2), simultaneously merge 2 sequences  
     *  (using a 2K by 2K network) of length M/4 to obtain sorted   
     *  sequences of M/2.                                           
     */                                                             
    uint64_t inlen  = (1 << j);                                     
    int64_t * inp;                                                  
    int64_t * out;                                                  
    int ptridx = j & 1;                                             
                                                                        
    inp = ptrs[ptridx];                                             
    out = ptrs[ptridx ^ 1];                                         
                                                                        
    merge16_eqlen(inp, inp + inlen, out, inlen);                     
    merge16_eqlen(inp+2*inlen, inp+3*inlen, out + 2*inlen, inlen);   
                                                                        
    /* TODO: simultaneous merge of 2 list pairs */                  
    /**                                                             
     * 1.d) for itr = (logM - 1), merge 2 final sequences (using a  
     * 4K by 4K network) of length M/2 to get sorted seq. of M.     
     */                                                             
    j++; /* j=(LOG2_BLOCK_SIZE-1); inputsize M/2 --> outputsize M*/ 
    inlen  = (1 << j);                                              
    /* now we know that input is out from the last pass */          
    merge16_eqlen(out, out + inlen, inp, inlen);                    
                                                                        
    /* finally swap input/output ptrs, output is the sorted list */ 
    * outputptr = inp;                                              
    * inputptr  = out;                                              
}

inline __attribute__((__always_inline__))
int keycmp(const void * k1, const void * k2)
{
    int64_t val = (*(int64_t *)k1 - *(int64_t *)k2);

    int ret = 0;
    if(val < 0)
        ret = -1;
    else if(val > 0)
        ret = 1;

    return ret;
}

inline __attribute__((__always_inline__)) void
swap(int64_t ** A, int64_t ** B)
{
    int64_t * tmp = *A;
    *A = *B;
    *B = tmp;
}


/** 
 * Sorts the last chunk of the input, which is less than BLOCKSIZE tuples.
 * @note This function assumes a hard-coded BLOCKSIZE of 16384 and nitems must
 * be less than 16384.
 *
 * @param inputptr 
 * @param outputptr 
 * @param nitems 
 */
inline void __attribute__((always_inline))
avxsort_rem(int64_t ** inputptr, int64_t ** outputptr, uint32_t nitems)
{
    int64_t * inp = *inputptr;
    int64_t * out = *outputptr;

#if 1 /* sort using AVX */
    /* each chunk keeps track of its temporary memory offset */
    //  int64_t * ptrs[8][2];/* [chunk-in, chunk-out-tmp] */
    int64_t * ptrs[22][2];/* [chunk-in, chunk-out-tmp] */

    uint32_t n = nitems, pos = 0, i = 0;

    //uint32_t nxtpow = 8192;/* TODO: infer from nitems, nearest pow2 to nitems */
    uint32_t nxtpow = n;
    NEXT_POW_2(nxtpow);
    
    //uint32_t sizes[6];
    uint32_t sizes[20];

    
    while(n < nxtpow) {
        nxtpow >>= 1;
    }

    while(nxtpow > 128) {
        ptrs[i][0] = inp + pos;
        ptrs[i][1] = out + pos;
        sizes[i]   = nxtpow;

        avxsort_block(&ptrs[i][0], &ptrs[i][1], nxtpow);
        pos += nxtpow;
        n   -= nxtpow;
        swap(&ptrs[i][0], &ptrs[i][1]);
        i++;

        while(n < nxtpow) {
            nxtpow >>= 1;
        }
    }

    if(n > 0) {
        /* sort last n < 128 items using scalar sort */
        ptrs[i][0] = inp + pos;
        ptrs[i][1] = out + pos;
        sizes[i]   = n;

#ifdef __cplusplus
        std::sort(ptrs[i][0], ptrs[i][0] + n);
#else
        qsort(ptrs[i][0], n, sizeof(int64_t), keycmp);
#endif
        /* no need to swap */
        i++;
    }

    uint32_t nchunks = i;

    /* merge sorted blocks */
    while(nchunks > 1) {
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

#else /* sort using scalar */

#ifdef __cplusplus
    std::sort(inp, inp + nitems);
#else
    qsort(inp, nitems, sizeof(int64_t), keycmp);
#endif

    *outputptr = inp;
    *inputptr  = out;

#endif    
}

/*******************************************************************************
 *                                                                             *
 *               Aligned Version of the Implementations                        *
 *                                                                             *
 *******************************************************************************/

inline void __attribute((always_inline))
inregister_sort_keyval32_aligned(int64_t * items, int64_t * output)
{
#ifdef USE_AVX_512
    __m512i ra = _mm512_load_epi64 ((int64_t const *)(items));
    __m512i rb = _mm512_load_epi64 ((int64_t const *)(items + 8));
    __m512i rc = _mm512_load_epi64 ((int64_t const *)(items + 16));
    __m512i rd = _mm512_load_epi64 ((int64_t const *)(items + 24));

    /* odd-even sorting network begins */
    /* 1st level of comparisons */
    __m512i ra1 = _mm512_min_epi64(ra, rb);
    __m512i rb1 = _mm512_max_epi64(ra, rb);

    __m512i rc1 = _mm512_min_epi64(rc, rd);
    __m512i rd1 = _mm512_max_epi64(rc, rd);

    /* 2nd level of comparisons */
    rb = _mm512_min_epi64(rb1, rd1);
    rd = _mm512_max_epi64(rb1, rd1);

    /* 3rd level of comparisons */
    __m512i ra2 = _mm512_min_epi64(ra1, rc1);
    __m512i rc2 = _mm512_max_epi64(ra1, rc1);

    /* 4th level of comparisons */
    __m512i rb3 = _mm512_min_epi64(rb, rc2);
    __m512i rc3 = _mm512_max_epi64(rb, rc2);

    /* shuffling using AVX 256 */
    //first batch
    __m256i ra2_1 = _mm512_extracti64x4_epi64(ra2, 0);
    __m256i rb3_1 = _mm512_extracti64x4_epi64(rb3, 0);
    __m256i rc3_1 = _mm512_extracti64x4_epi64(rc3, 0);
    __m256i rd_1 = _mm512_extracti64x4_epi64(rd, 0);

    __m256i ra3_1 = _mm256_unpacklo_epi64(ra2_1, rb3_1);
    __m256i rb4_1 = _mm256_unpackhi_epi64(ra2_1, rb3_1);

    __m256i rc4_1 = _mm256_unpacklo_epi64(rc3_1, rd_1);
    __m256i rd3_1 = _mm256_unpackhi_epi64(rc3_1, rd_1);

    __m256i ra4_1 = _mm256_permute2f128_si256(ra3_1, rc4_1, 0x20);
    __m256i rc5_1 = _mm256_permute2f128_si256(ra3_1, rc4_1, 0x31);

    __m256i rb5_1 = _mm256_permute2f128_si256(rb4_1, rd3_1, 0x20);
    __m256i rd4_1 = _mm256_permute2f128_si256(rb4_1, rd3_1, 0x31);

    _mm256_store_epi64((int64_t *) output, ra4_1);
    _mm256_store_epi64((int64_t *) (output + 4), rb5_1);
    _mm256_store_epi64((int64_t *) (output + 8), rc5_1);
    _mm256_store_epi64((int64_t *) (output + 12), rd4_1);

    //second batch
    __m256i ra2_2 = _mm512_extracti64x4_epi64(ra2, 1);
    __m256i rb3_2 = _mm512_extracti64x4_epi64(rb3, 1);
    __m256i rc3_2 = _mm512_extracti64x4_epi64(rc3, 1);
    __m256i rd_2 = _mm512_extracti64x4_epi64(rd, 1);

    __m256i ra3_2 = _mm256_unpacklo_epi64(ra2_2, rb3_2);
    __m256i rb4_2 = _mm256_unpackhi_epi64(ra2_2, rb3_2);

    __m256i rc4_2 = _mm256_unpacklo_epi64(rc3_2, rd_2);
    __m256i rd3_2 = _mm256_unpackhi_epi64(rc3_2, rd_2);

    __m256i ra4_2 = _mm256_permute2f128_si256(ra3_2, rc4_2, 0x20);
    __m256i rc5_2 = _mm256_permute2f128_si256(ra3_2, rc4_2, 0x31);

    __m256i rb5_2 = _mm256_permute2f128_si256(rb4_2, rd3_2, 0x20);
    __m256i rd4_2 = _mm256_permute2f128_si256(rb4_2, rd3_2, 0x31);

    _mm256_store_epi64((int64_t *) (output + 16) , ra4_2);
    _mm256_store_epi64((int64_t *) (output + 20), rb5_2);
    _mm256_store_epi64((int64_t *) (output + 24), rc5_2);
    _mm256_store_epi64((int64_t *) (output + 28), rd4_2);
#else
/* IACA_START */
    __m256d ra = _mm256_load_pd ((double const *)(items));
    __m256d rb = _mm256_load_pd ((double const *)(items + 4));
    __m256d rc = _mm256_load_pd ((double const *)(items + 8));
    __m256d rd = _mm256_load_pd ((double const *)(items + 12));

    /* odd-even sorting network begins */
    /* 1st level of comparisons */
    __m256d ra1 = _mm256_min_pd(ra, rb);
    __m256d rb1 = _mm256_max_pd(ra, rb);

    __m256d rc1 = _mm256_min_pd(rc, rd);
    __m256d rd1 = _mm256_max_pd(rc, rd);

    /* 2nd level of comparisons */
    rb = _mm256_min_pd(rb1, rd1);
    rd = _mm256_max_pd(rb1, rd1);

    /* 3rd level of comparisons */
    __m256d ra2 = _mm256_min_pd(ra1, rc1);
    __m256d rc2 = _mm256_max_pd(ra1, rc1);

    /* 4th level of comparisons */
    __m256d rb3 = _mm256_min_pd(rb, rc2);
    __m256d rc3 = _mm256_max_pd(rb, rc2);

    /* results are in ra2, rb3, rc3, rd */
    /**
     * Initial data and transposed data looks like following:
     *  a2={ x1  x2  x3  x4  }                      a4={ x1 x5 x9  x13 }
     *  b3={ x5  x6  x7  x8  }  === Transpose ===>  b5={ x2 x6 x10 x14 }
     *  c3={ x9  x10 x11 x12 }                      c5={ x3 x7 x11 x15 }
     *  d2={ x13 x14 x15 x16 }                      d4={ x4 x8 x12 x16 }
     */
    /* shuffle x2 and x5 - shuffle x4 and x7 */
    __m256d ra3 = _mm256_unpacklo_pd(ra2, rb3);
    __m256d rb4 = _mm256_unpackhi_pd(ra2, rb3);

    /* shuffle x10 and x13 - shuffle x12 and x15 */
    __m256d rc4 = _mm256_unpacklo_pd(rc3, rd);
    __m256d rd3 = _mm256_unpackhi_pd(rc3, rd);

    /* shuffle (x3,x7) and (x9,x13) pairs */
    __m256d ra4 = _mm256_permute2f128_pd(ra3, rc4, 0x20);
    __m256d rc5 = _mm256_permute2f128_pd(ra3, rc4, 0x31);

    /* shuffle (x4,x8) and (x10,x14) pairs */
    __m256d rb5 = _mm256_permute2f128_pd(rb4, rd3, 0x20);
    __m256d rd4 = _mm256_permute2f128_pd(rb4, rd3, 0x31);

    /* after this, results are in ra4, rb5, rc5, rd4 */
/* IACA_END */
    /* store */
    _mm256_store_pd((double *) output, ra4);
    _mm256_store_pd((double *) (output + 4), rb5);
    _mm256_store_pd((double *) (output + 8), rc5);
    _mm256_store_pd((double *) (output + 12), rd4);
#endif

}

inline void __attribute((always_inline))
merge4_eqlen_aligned(int64_t * const inpA, int64_t * const inpB,
                     int64_t * const out, const uint32_t len)
{
    register block4 * inA  = (block4 *) inpA;
    register block4 * inB  = (block4 *) inpB;
    block4 * const    endA = (block4 *) (inpA + len);
    block4 * const    endB = (block4 *) (inpB + len);

    block4 * outp = (block4 *) out;

    register block4 * next = inB;
        
#ifndef USE_AVX_512
    register __m256d outreg1;
    register __m256d outreg2;

    register __m256d regA = _mm256_load_pd((double const *) inA);
    register __m256d regB = _mm256_load_pd((double const *) next);

    inA ++;
    inB ++;

    BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
    /* store outreg1 */
    _mm256_store_pd((double *) outp, outreg1);
#else
    register __m256i outreg1;
    register __m256i outreg2;

//#ifndef DEVELOPMENT_MODE
//    register __m256i regA = _mm256_load_epi64((int64_t const *) inA);
//    register __m256i regB = _mm256_load_epi64((int64_t const *) next);
//#else
    register __m256i regA = _mm256_mask_load_epi64(regA, 0xFF, (int64_t const *) inA);
    register __m256i regB = _mm256_mask_load_epi64(regB, 0xFF, (int64_t const *) next);
//#endif
    inA ++;
    inB ++;

    BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
    /* store outreg1 */
#ifndef DEVELOPMENT_MODE
    _mm256_store_epi64((int64_t *) outp, outreg1);
#else
    _mm256_mask_store_epi64((int64_t *) outp, 0xFF, outreg1);
#endif
#endif
    outp ++;

    while( inA < endA && inB < endB ) {

        /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
        IFELSECONDMOVE(next, inA, inB, 32);

#ifndef USE_AVX_512
        regA = outreg2;
        regB = _mm256_load_pd((double const *) next);
        
        BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
        _mm256_store_pd((double *) outp, outreg1);
#else
        regA = outreg2;
//#ifndef DEVELOPMENT_MODE
//        regB = _mm256_load_epi64((int64_t const *) next);
//#else
        regB = _mm256_mask_load_epi64(regB, 0xFF, (int64_t const *) next);
//#endif
        
        BITONIC_MERGE4(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
        _mm256_store_epi64((int64_t *) outp, outreg1);
#endif
        outp ++;
    }

    /* handle remaining items */
    while( inA < endA ) {
#ifndef USE_AVX_512
        __m256d regA = _mm256_load_pd((double const *) inA);
        __m256d regB = outreg2;

        BITONIC_MERGE4(outreg1, outreg2, regA, regB);

        _mm256_store_pd((double *) outp, outreg1);
#else
//#ifndef DEVELOPMENT_MODE
//        __m256i regA = _mm256_load_epi64((int64_t const *) inA);
//#else
        __m256i regA = _mm256_mask_load_epi64(regA, 0xFF, (int64_t const *) inA);
//#endif
        __m256i regB = outreg2;

        BITONIC_MERGE4(outreg1, outreg2, regA, regB);

        _mm256_store_epi64((int64_t *) outp, outreg1);
#endif
        inA ++;
        outp ++;
    }

    while( inB < endB ) {

#ifndef USE_AVX_512
        __m256d regA = outreg2;
        __m256d regB = _mm256_load_pd((double const *) inB);

        BITONIC_MERGE4(outreg1, outreg2, regA, regB);

        _mm256_store_pd((double *) outp, outreg1);
#else
        __m256i regA = outreg2;
//#ifndef DEVELOPMENT_MODE
//        __m256i regB = _mm256_load_epi64((int64_t const *) inB);
//#else
        __m256i regB = _mm256_mask_load_epi64(regB, 0xFF, (int64_t const *) inB);
//#endif
        BITONIC_MERGE4(outreg1, outreg2, regA, regB);

        _mm256_store_epi64((int64_t *) outp, outreg1);
#endif
        inB ++;
        outp ++;
    }

    /* store the last remaining register values */
#ifndef USE_AVX_512
    _mm256_store_pd((double *) outp, outreg2);
#else
    _mm256_store_epi64((int64_t *) outp, outreg2);
#endif
}

inline void __attribute((always_inline))
merge8_eqlen_aligned(int64_t * const inpA, int64_t * const inpB,
                     int64_t * const out, const uint32_t len)
{
    register block8 * inA  = (block8 *) inpA;
    register block8 * inB  = (block8 *) inpB;
    block8 * const    endA = (block8 *) (inpA + len);
    block8 * const    endB = (block8 *) (inpB + len);

    block8 * outp = (block8 *) out;

    register block8 * next = inB;

#ifndef USE_AVX_512
    register __m256d outreg1l, outreg1h;
    register __m256d outreg2l, outreg2h;

    register __m256d regAl, regAh;
    register __m256d regBl, regBh;

    LOAD8(regAl, regAh, inA);
    LOAD8(regBl, regBh, next);

    inA ++;
    inB ++;

    BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                   regAl, regAh, regBl, regBh);

    /* store outreg1 */
    STORE8(outp, outreg1l, outreg1h);
#else
    register __m512i outreg1;
    register __m512i outreg2;

    register __m512i regA;
    register __m512i regB;

    LOAD8(regA, inA);
    LOAD8(regB, next);

    inA ++;
    inB ++;

    BITONIC_MERGE8(outreg1, outreg2, regA, regB);

    /* store outreg1 */
    STORE8(outp, outreg1);
#endif
    outp ++;

    while( inA < endA && inB < endB ) {

        /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
        IFELSECONDMOVE(next, inA, inB, 64);

#ifndef USE_AVX_512        
        regAl = outreg2l;
        regAh = outreg2h;
        LOAD8(regBl, regBh, next);
        
        BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                       regAl, regAh, regBl, regBh);
        
        /* store outreg1 */
        STORE8(outp, outreg1l, outreg1h);
#else
        regA = outreg2;
        LOAD8(regB, next);
        
        BITONIC_MERGE8(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
        STORE8(outp, outreg1);
#endif
        outp ++;
    }

    /* handle remaining items */
    while( inA < endA ) {
#ifndef USE_AVX_512
        __m256d regAl, regAh;
        LOAD8(regAl, regAh, inA);

        __m256d regBl = outreg2l;
        __m256d regBh = outreg2h;

        BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                       regAl, regAh, regBl, regBh);
        
        /* store outreg1 */
        STORE8(outp, outreg1l, outreg1h);
#else
        __m512i regA;
        LOAD8(regA, inA);

        __m512i regB = outreg2;

        BITONIC_MERGE8(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
        STORE8(outp, outreg1);
#endif        
        outp ++;
        inA ++;
    }

    while( inB < endB ) {

#ifndef USE_AVX_512
        __m256d regAl = outreg2l;
        __m256d regAh = outreg2h;
        __m256d regBl, regBh;

        LOAD8(regBl, regBh, inB);

        BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h,
                       regAl, regAh, regBl, regBh);
        
        /* store outreg1 */
        STORE8(outp, outreg1l, outreg1h);
#else
        __m512i regA = outreg2;
        __m512i regB;

        LOAD8(regB, inB);

        BITONIC_MERGE8(outreg1, outreg2, regA, regB);
        
        /* store outreg1 */
        STORE8(outp, outreg1);
#endif
        outp ++;
        inB ++;
    }

    /* store the last remaining register values */
#ifndef USE_AVX_512
    STORE8(outp, outreg2l, outreg2h);
#else
    STORE8(outp, outreg2);
#endif
}

inline void __attribute((always_inline))
merge16_eqlen_aligned(int64_t * const inpA, int64_t * const inpB,
                      int64_t * const out, const uint32_t len)
{
    register block16 * inA  = (block16 *) inpA;
    register block16 * inB  = (block16 *) inpB;
    block16 * const    endA = (block16 *) (inpA + len);
    block16 * const    endB = (block16 *) (inpB + len);

    block16 * outp = (block16 *) out;

    register block16 * next = inB;

#ifndef USE_AVX_512
    __m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
    __m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;

    __m256d regAl1, regAl2, regAh1, regAh2;
    __m256d regBl1, regBl2, regBh1, regBh2;

    LOAD8(regAl1, regAl2, inA);
    LOAD8(regAh1, regAh2, ((block8 *)(inA) + 1));
    inA ++;

    LOAD8(regBl1, regBl2, inB);
    LOAD8(regBh1, regBh2, ((block8 *)(inB) + 1));
    inB ++;

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                    outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                    regAl1, regAl2, regAh1, regAh2,
                    regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8(outp, outreg1l1, outreg1l2);
    STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
    __m512i outreg1l, outreg1h;
    __m512i outreg2l, outreg2h;

    __m512i regAl, regAh;
    __m512i regBl, regBh;

    LOAD8(regAl, inA);
    LOAD8(regAh, ((block8 *)(inA) + 1));
    inA ++;

    LOAD8(regBl, inB);
    LOAD8(regBh, ((block8 *)(inB) + 1));
    inB ++;

    BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                    regAl, regAh, regBl, regBh);

    /* store outreg1 */
    STORE8(outp, outreg1l);
    STORE8(((block8 *)outp + 1), outreg1h);
#endif
    outp ++;

    while( inA < endA && inB < endB ) {

        /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
        IFELSECONDMOVE(next, inA, inB, 128);

#ifndef USE_AVX_512
        regAl1 = outreg2l1;
        regAl2 = outreg2l2;
        regAh1 = outreg2h1;
        regAh2 = outreg2h2;

        LOAD8(regBl1, regBl2, next);
        LOAD8(regBh1, regBh2, ((block8 *)next + 1));
        
        BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                        outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                        regAl1, regAl2, regAh1, regAh2,
                        regBl1, regBl2, regBh1, regBh2);

        /* store outreg1 */
        STORE8(outp, outreg1l1, outreg1l2);
        STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
        regAl = outreg2l;
        regAh = outreg2h;

        LOAD8(regBl, next);
        LOAD8(regBh, ((block8 *)next + 1));
        
        BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                        regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8(outp, outreg1l);
        STORE8(((block8 *)outp + 1), outreg1h);
#endif
        outp ++;
    }

    /* handle remaining items */
    while( inA < endA ) {
#ifndef USE_AVX_512
        __m256d regAl1, regAl2, regAh1, regAh2;
        __m256d regBl1 = outreg2l1;
        __m256d regBl2 = outreg2l2;
        __m256d regBh1 = outreg2h1;
        __m256d regBh2 = outreg2h2;

        LOAD8(regAl1, regAl2, inA);
        LOAD8(regAh1, regAh2, ((block8 *)(inA) + 1));
        inA ++;

        BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                        outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                        regAl1, regAl2, regAh1, regAh2,
                        regBl1, regBl2, regBh1, regBh2);

        /* store outreg1 */
        STORE8(outp, outreg1l1, outreg1l2);
        STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
        __m512i regAl, regAh;
        __m512i regBl = outreg2l;
        __m512i regBh = outreg2h;

        LOAD8(regAl, inA);
        LOAD8(regAh, ((block8 *)(inA) + 1));
        inA ++;

        BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8(outp, outreg1l);
        STORE8(((block8 *)outp + 1), outreg1h);
#endif
        outp ++;
    }

    while( inB < endB ) {

#ifndef USE_AVX_512
        __m256d regBl1, regBl2, regBh1, regBh2;
        __m256d regAl1 = outreg2l1;
        __m256d regAl2 = outreg2l2;
        __m256d regAh1 = outreg2h1;
        __m256d regAh2 = outreg2h2;

        LOAD8(regBl1, regBl2, inB);
        LOAD8(regBh1, regBh2, ((block8 *)inB + 1));
        inB ++;

        BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                        outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                        regAl1, regAl2, regAh1, regAh2,
                        regBl1, regBl2, regBh1, regBh2);
        
        /* store outreg1 */
        STORE8(outp, outreg1l1, outreg1l2);
        STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
        __m512i regBl, regBh;
        __m512i regAl = outreg2l;
        __m512i regAh = outreg2h;

        LOAD8(regBl, inB);
        LOAD8(regBh, ((block8 *)inB + 1));
        inB ++;

        BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                        regAl, regAh, regBl, regBh);
        
        /* store outreg1 */
        STORE8(outp, outreg1l);
        STORE8(((block8 *)outp + 1), outreg1h);
#endif
        outp ++;
    }

    /* store the last remaining register values */
#ifndef USE_AVX_512
    STORE8(outp, outreg2l1, outreg2l2);
    STORE8(((block8 *)outp + 1), outreg2h1, outreg2h2);
#else
    STORE8(outp, outreg2l);
    STORE8(((block8 *)outp + 1), outreg2h);
#endif
}

/** 
 * Merge two sorted arrays to a final output using 16-way AVX bitonic merge. 
 * 
 * @param inpA input array A
 * @param inpB input array B
 * @param Out  output array
 * @param lenA size of A
 * @param lenB size of B
 */
inline void __attribute__((always_inline))
merge16_varlen_aligned(int64_t * restrict inpA, 
                       int64_t * restrict inpB,
                       int64_t * restrict Out, 
                       const uint32_t lenA, 
                       const uint32_t lenB) 
{
    uint32_t lenA16 = lenA & ~0xF, lenB16 = lenB & ~0xF;
    uint32_t ai = 0, bi = 0;

    int64_t * out = Out;

    if(lenA16 > 16 && lenB16 > 16) {

        register block16 * inA  = (block16 *) inpA;
        register block16 * inB  = (block16 *) inpB;
        block16 * const    endA = (block16 *) (inpA + lenA) - 1;
        block16 * const    endB = (block16 *) (inpB + lenB) - 1;

        block16 * outp = (block16 *) out;

        register block16 * next = inB;

#ifndef USE_AVX_512
        __m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
        __m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;

        __m256d regAl1, regAl2, regAh1, regAh2;
        __m256d regBl1, regBl2, regBh1, regBh2;

        LOAD8(regAl1, regAl2, inA);
        LOAD8(regAh1, regAh2, ((block8 *)(inA) + 1));
        inA ++;

        LOAD8(regBl1, regBl2, inB);
        LOAD8(regBh1, regBh2, ((block8 *)(inB) + 1));
        inB ++;

        BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                        outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                        regAl1, regAl2, regAh1, regAh2,
                        regBl1, regBl2, regBh1, regBh2);

        /* store outreg1 */
        STORE8(outp, outreg1l1, outreg1l2);
        STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
        __m512i outreg1l, outreg1h;
        __m512i outreg2l, outreg2h;

        __m512i regAl, regAh;
        __m512i regBl, regBh;

        LOAD8(regAl, inA);
        LOAD8(regAh, ((block8 *)(inA) + 1));
        inA ++;

        LOAD8(regBl, inB);
        LOAD8(regBh, ((block8 *)(inB) + 1));
        inB ++;

        BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                        regAl, regAh, regBl, regBh);

        /* store outreg1 */
        STORE8(outp, outreg1l);
        STORE8(((block8 *)outp + 1), outreg1h);
#endif
        outp ++;

        while( inA < endA && inB < endB ) {

            /** The inline assembly below does exactly the following code: */
            /* Option 3: with assembly */
            IFELSECONDMOVE(next, inA, inB, 128);

#ifndef USE_AVX_512
            regAl1 = outreg2l1;
            regAl2 = outreg2l2;
            regAh1 = outreg2h1;
            regAh2 = outreg2h2;

            LOAD8(regBl1, regBl2, next);
            LOAD8(regBh1, regBh2, ((block8 *)next + 1));
        
            BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2,
                            outreg2l1, outreg2l2, outreg2h1, outreg2h2,
                            regAl1, regAl2, regAh1, regAh2,
                            regBl1, regBl2, regBh1, regBh2);

            /* store outreg1 */
            STORE8(outp, outreg1l1, outreg1l2);
            STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
#else
            regAl = outreg2l;
            regAh = outreg2h;

            LOAD8(regBl, next);
            LOAD8(regBh, ((block8 *)next + 1));
        
            BITONIC_MERGE16(outreg1l, outreg1h, outreg2l, outreg2h,
                            regAl, regAh, regBl, regBh);

            /* store outreg1 */
            STORE8(outp, outreg1l);
            STORE8(((block8 *)outp + 1), outreg1h);
#endif
            outp ++;
        }

        /* flush the register to one of the lists */
        int64_t hireg[4] __attribute__((aligned(16)));

#ifndef USE_AVX_512
        _mm256_store_pd ( (double *)hireg, outreg2h2);
#else
        __m256i outreg2h2 = _mm512_extracti64x4_epi64(outreg2h, 1);
        _mm256_store_epi64 ( (int64_t *)hireg, outreg2h2);
#endif
        if(*((int64_t *)inA) >= *((int64_t*)(hireg+3))) {
            /* store the last remaining register values to A */
            inA --;
#ifndef USE_AVX_512            
            STORE8(inA, outreg2l1, outreg2l2);
            STORE8(((block8 *)inA + 1), outreg2h1, outreg2h2);
#else
            STORE8(inA, outreg2l);
            STORE8(((block8 *)inA + 1), outreg2h);
#endif
        }
        else {
            /* store the last remaining register values to B */
            inB --;
#ifndef USE_AVX_512
            STORE8(inB, outreg2l1, outreg2l2);
            STORE8(((block8 *)inB + 1), outreg2h1, outreg2h2);
#else
            STORE8(inB, outreg2l);
            STORE8(((block8 *)inB + 1), outreg2h);
#endif
        }

        ai = ((int64_t *)inA - inpA);
        bi = ((int64_t *)inB - inpB);

        inpA = (int64_t *)inA;
        inpB = (int64_t *)inB;
        out  = (int64_t *)outp;
    }

    /* serial-merge */
    while(ai < lenA && bi < lenB){
        int64_t * in = inpB;
        uint32_t cmp = (*inpA < *inpB);
        uint32_t notcmp = !cmp;

        ai += cmp;
        bi += notcmp;
            
        if(cmp)
            in = inpA;
            
        *out = *in;
        out ++;
        inpA += cmp;
        inpB += notcmp;
    }

    if(ai < lenA) {
        /* if A has any more items to be output */

        if((lenA - ai) >= 8) {
            /* if A still has some times to be output with AVX */
            uint32_t lenA8 = ((lenA-ai) & ~0x7);
            register block8 * inA  = (block8 *) inpA;
            block8 * const    endA = (block8 *) (inpA + lenA8);
            block8 * outp = (block8 *) out;
            
            while(inA < endA) {
#ifndef USE_AVX_512                
                __m256d regAl, regAh;
                LOAD8U(regAl, regAh, inA);
                STORE8U(outp, regAl, regAh);
#else
                __m512i regA;
                LOAD8U(regA, inA);
                STORE8U(outp, regA);
#endif
                outp ++;
                inA ++;
            }

            ai   += ((int64_t*)inA - inpA);
            inpA  = (int64_t *)inA;
            out   = (int64_t *)outp;
        }

        while(ai < lenA) {
            *out = *inpA;
            ai++;
            out++;
            inpA++;
        }
    } 
    else if(bi < lenB) {
        /* if B has any more items to be output */

        if((lenB - bi) >= 8) {
            /* if B still has some times to be output with AVX */
            uint32_t lenB8 = ((lenB-bi) & ~0x7);
            register block8 * inB  = (block8 *) inpB;
            block8 * const    endB = (block8 *) (inpB + lenB8);
            block8 * outp = (block8 *) out;
            
            while(inB < endB) {
#ifndef USE_AVX_512
                __m256d regBl, regBh;
                LOAD8U(regBl, regBh, inB);
                STORE8U(outp, regBl, regBh);
#else
                __m512i regB;
                LOAD8U(regB, inB);
                STORE8U(outp, regB);
#endif
                outp ++;
                inB ++;
            }

            bi   += ((int64_t*)inB - inpB);
            inpB  = (int64_t *)inB;
            out   = (int64_t *)outp;
        }

        while(bi < lenB) {
            *out = *inpB;
            bi++;
            out++;
            inpB++;
        }
    }
}

inline void __attribute__((always_inline))
avxsort_block_aligned(int64_t ** inputptr, int64_t ** outputptr, int BLOCK_SIZE)
{                                                                   
    int64_t * ptrs[2];                                              
    const uint64_t logBSZ = mylog2(BLOCK_SIZE);
                                                                        
    ptrs[0] = *inputptr;                                            
    ptrs[1] = *outputptr;                                           
                                                                        
    /** 1.a) Perform in-register sort to get sorted seq of K(K=4)*/
#ifndef USE_AVX_512     
    block16 * inptr = (block16 *) ptrs[0];                          
    block16 * const end = (block16 *) (ptrs[0] + BLOCK_SIZE);       
#else
    block32 * inptr = (block32 *) ptrs[0];                          
    block32 * const end = (block32 *) (ptrs[0] + BLOCK_SIZE);
#endif    
    while(inptr < end) {                                            
        inregister_sort_keyval32_aligned((int64_t*)inptr, (int64_t*)inptr); 
        inptr ++;                                                   
    }                                                               
                                                                        
                                                                        
    /**                                                             
     * 1.b) for itr <- [(logK) .. (logM - 3)]                       
     *  - Simultaneously merge 4 sequences (using a K by K          
     *  network) of length 2^itr to obtain sorted seq. of 2^{itr+1} 
     */                                                             
    uint64_t j;                                                     
    const uint64_t jend = logBSZ - 2;                               
                                                                        
    j = 2;
    {
        int ptridx = j & 1;                                         
        int64_t * inp = (int64_t *) ptrs[ptridx];                   
        int64_t * out = (int64_t *) ptrs[ptridx ^ 1];               
        int64_t * const end = (int64_t*) (inp + BLOCK_SIZE);        
                                                                        
        /**                                                         
         *  merge length 2^j lists beginnig at inp and output a     
         *  sorted list of length 2^(j+1) starting at out           
         */                                                         
        const uint64_t inlen  = (1 << j);                           
        const uint64_t outlen = (inlen << 1);                       
                                                                        
        while(inp < end) {                                          
                                                                        
            merge4_eqlen_aligned(inp, inp + inlen, out, inlen);
            inp += outlen;                                          
            out += outlen;                                          
        }                                                           
    }                                                               
    j = 3;
    {
        int ptridx = j & 1;                                         
        int64_t * inp = (int64_t *) ptrs[ptridx];                   
        int64_t * out = (int64_t *) ptrs[ptridx ^ 1];               
        int64_t * const end = (int64_t*) (inp + BLOCK_SIZE);        
                                                                        
        /**                                                         
         *  merge length 2^j lists beginnig at inp and output a     
         *  sorted list of length 2^(j+1) starting at out           
         */                                                         
        const uint64_t inlen  = (1 << j);                           
        const uint64_t outlen = (inlen << 1);                       
                                                                        
        while(inp < end) {                                          
                                                                        
            merge8_eqlen_aligned(inp, inp + inlen, out, inlen);             
            inp += outlen;                                          
            out += outlen;                                          
        }                                                           
    }                                                               
    for(j = 4; j < jend; j++) {                                     
        int ptridx = j & 1;                                         
        int64_t * inp = (int64_t *) ptrs[ptridx];                   
        int64_t * out = (int64_t *) ptrs[ptridx ^ 1];               
        int64_t * const end = (int64_t*) (inp + BLOCK_SIZE);        
                                                                        
        /**                                                         
         *  merge length 2^j lists beginnig at inp and output a     
         *  sorted list of length 2^(j+1) starting at out           
         */                                                         
        const uint64_t inlen  = (1 << j);                           
        const uint64_t outlen = (inlen << 1);                       
                                                                        
        while(inp < end) {                                          
                                                                        
            merge16_eqlen_aligned(inp, inp + inlen, out, inlen);             
            inp += outlen;                                          
            out += outlen;                                          
                                                                        
            /* TODO: Try following. */                              
            /* simultaneous merge of 4 list pairs */                
            /* merge 4 seqs simultaneously (always >= 4) */         
            /* merge 2 seqs simultaneously (always >= 2) */         
        }                                                           
    }
                                                                        
    /**                                                             
     * 1.c) for itr = (logM - 2), simultaneously merge 2 sequences  
     *  (using a 2K by 2K network) of length M/4 to obtain sorted   
     *  sequences of M/2.                                           
     */                                                             
    uint64_t inlen  = (1 << j);                                     
    int64_t * inp;                                                  
    int64_t * out;                                                  
    int ptridx = j & 1;                                             
                                                                        
    inp = ptrs[ptridx];                                             
    out = ptrs[ptridx ^ 1];                                         
                                                                        
    merge16_eqlen_aligned(inp, inp + inlen, out, inlen);                     
    merge16_eqlen_aligned(inp+2*inlen, inp+3*inlen, out + 2*inlen, inlen);   
                                                                        
    /* TODO: simultaneous merge of 2 list pairs */                  
    /**                                                             
     * 1.d) for itr = (logM - 1), merge 2 final sequences (using a  
     * 4K by 4K network) of length M/2 to get sorted seq. of M.     
     */                                                             
    j++; /* j=(LOG2_BLOCK_SIZE-1); inputsize M/2 --> outputsize M*/ 
    inlen  = (1 << j);                                              
    /* now we know that input is out from the last pass */          
    merge16_eqlen_aligned(out, out + inlen, inp, inlen);
                                                                        
    /* finally swap input/output ptrs, output is the sorted list */ 
    * outputptr = inp;                                              
    * inputptr  = out;                                              
}

/** 
 * Sorts the last chunk of the input, which is less than BLOCKSIZE tuples.
 * @note This function assumes a hard-coded BLOCKSIZE of 16384 and nitems must
 * be less than 16384.
 *
 * @param inputptr 
 * @param outputptr 
 * @param nitems 
 */
inline void __attribute__((always_inline))
avxsort_rem_aligned(int64_t ** inputptr, int64_t ** outputptr, uint32_t nitems)
{
    int64_t * inp = *inputptr;
    int64_t * out = *outputptr;

#if 1 /* sort using AVX */
    /* each chunk keeps track of its temporary memory offset */
//  int64_t * ptrs[8][2];/* [chunk-in, chunk-out-tmp] */
    int64_t * ptrs[22][2];/* [chunk-in, chunk-out-tmp] */

    uint32_t n = nitems, pos = 0, i = 0;

    //uint32_t nxtpow = 8192;/* TODO: infer from nitems, nearest pow2 to nitems */
    uint32_t nxtpow = n;
    NEXT_POW_2(nxtpow);
    
    //uint32_t sizes[6];
    uint32_t sizes[20];
    
    while(n < nxtpow) {
        nxtpow >>= 1;
    }

    while(nxtpow > 128) {
        ptrs[i][0] = inp + pos;
        ptrs[i][1] = out + pos;
        sizes[i]   = nxtpow;

        avxsort_block_aligned(&ptrs[i][0], &ptrs[i][1], nxtpow);
        pos += nxtpow;
        n   -= nxtpow;
        swap(&ptrs[i][0], &ptrs[i][1]);
        i++;

        while(n < nxtpow) {
            nxtpow >>= 1;
        }
    }

    if(n > 0) {
        /* sort last n < 128 items using scalar sort */
        ptrs[i][0] = inp + pos;
        ptrs[i][1] = out + pos;
        sizes[i]   = n;

#ifdef __cplusplus
        std::sort(ptrs[i][0], ptrs[i][0] + n);
#else
        qsort(ptrs[i][0], n, sizeof(int64_t), keycmp);
#endif
        /* no need to swap */
        i++;
    }

    uint32_t nchunks = i;

    /* merge sorted blocks */
    while(nchunks > 1) {
        uint64_t k = 0;
        for(uint64_t j = 0; j < (nchunks-1); j += 2) {
            int64_t * inpA  = ptrs[j][0];
            int64_t * inpB  = ptrs[j+1][0];
            int64_t * out   = ptrs[j][1];
            uint32_t  sizeA = sizes[j];
            uint32_t  sizeB = sizes[j+1];

            merge16_varlen_aligned(inpA, inpB, out, sizeA, sizeB);
            
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

#else /* sort using scalar */

#ifdef __cplusplus
    std::sort(inp, inp + nitems);
#else
    qsort(inp, nitems, sizeof(int64_t), keycmp);
#endif

    *outputptr = inp;
    *inputptr  = out;

#endif    
}
