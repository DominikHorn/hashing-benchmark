/**
 * @file    avxcommon.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Tue Dec 11 18:24:10 2012
 * @version $Id $
 * 
 * @brief   Common AVX code, kernels etc. used by implementations.
 * 
 * 
 */
#ifndef AVXCOMMON_H
#define AVXCOMMON_H

#include <immintrin.h> /* AVX intrinsics */

#include "configs/base_configs.h"
#include "utils/base_utils.h"

typedef struct block4  {int64_t val[4]; } block4;
typedef struct block8  {int64_t val[8]; } block8;
typedef struct block16 {int64_t val[16];} block16;
typedef struct block32 {int64_t val[32];} block32;

/** 
 * There are 2 ways to implement branches: 
 *     1) With conditional move instr.s using inline assembly (IFELSEWITHCMOVE).
 *     2) With software predication (IFELSEWITHPREDICATION).
 *     3) With normal if-else
 */
#define IFELSEWITHCMOVE       0
#define IFELSEWITHPREDICATION 1
#define IFELSEWITHNORMAL      0

/** Load 2 AVX 256-bit registers from the given address */
#ifndef USE_AVX_512
#define LOAD8(REGL, REGH, ADDR)                                         \
    do {                                                                \
        REGL = _mm256_load_pd((double const *) ADDR);                   \
        REGH = _mm256_load_pd((double const *)(((block4 *)ADDR) + 1));  \
    } while(0)

/** Load unaligned 2 AVX 256-bit registers from the given address */
#define LOAD8U(REGL, REGH, ADDR)                                        \
    do {                                                                \
        REGL = _mm256_loadu_pd((double const *) ADDR);                  \
        REGH = _mm256_loadu_pd((double const *)(((block4 *)ADDR) + 1)); \
    } while(0)

/** Store 2 AVX 256-bit registers to the given address */
#define STORE8(ADDR, REGL, REGH)                                    \
    do {                                                            \
        _mm256_store_pd((double *) ADDR, REGL);                     \
        _mm256_store_pd((double *)(((block4 *) ADDR) + 1), REGH);   \
    } while(0)

/** Store unaligned 2 AVX 256-bit registers to the given address */
#define STORE8U(ADDR, REGL, REGH)                                   \
    do {                                                            \
        _mm256_storeu_pd((double *) ADDR, REGL);                    \
        _mm256_storeu_pd((double *)(((block4 *) ADDR) + 1), REGH);  \
    } while(0)


/**
 * @note Reversing 64-bit values in an AVX register. It will be possible with
 * single _mm256_permute4x64_pd() instruction in AVX2.
 */
#define REVERSE(REG)                                    \
    do {                                                \
        /* first reverse each 128-bit lane */           \
        REG = _mm256_permute_pd(REG, 0x5);              \
        /* now shuffle 128-bit lanes */                 \
        REG = _mm256_permute2f128_pd(REG, REG, 0x1);    \
    } while(0)


/** Bitonic merge kernel for 2 x 4 elements after the reversing step. */
#define BITONIC4(O1, O2, A, B)                                          \
    do {                                                                \
        /* Level-1 comparisons */                                       \
        __m256d l1 = _mm256_min_pd(A, B);                               \
        __m256d h1 = _mm256_max_pd(A, B);                               \
                                                                        \
        /* Level-1 shuffles */                                          \
        __m256d l1p = _mm256_permute2f128_pd(l1, h1, 0x31);             \
        __m256d h1p = _mm256_permute2f128_pd(l1, h1, 0x20);             \
                                                                        \
        /* Level-2 comparisons */                                       \
        __m256d l2 = _mm256_min_pd(l1p, h1p);                           \
        __m256d h2 = _mm256_max_pd(l1p, h1p);                           \
                                                                        \
        /* Level-2 shuffles */                                          \
        __m256d l2p = _mm256_shuffle_pd(l2, h2, 0x0);                   \
        __m256d h2p = _mm256_shuffle_pd(l2, h2, 0xF);                   \
                                                                        \
        /* Level-3 comparisons */                                       \
        __m256d l3 = _mm256_min_pd(l2p, h2p);                           \
        __m256d h3 = _mm256_max_pd(l2p, h2p);                           \
                                                                        \
        /* Level-3 shuffles implemented with unpcklps unpckhps */       \
        /* AVX cannot shuffle both inputs from same 128-bit lane */     \
        /* so we need 2 more instructions for this operation. */        \
        __m256d l4 = _mm256_unpacklo_pd(l3, h3);                        \
        __m256d h4 = _mm256_unpackhi_pd(l3, h3);                        \
        O1 = _mm256_permute2f128_pd(l4, h4, 0x20);                      \
        O2 = _mm256_permute2f128_pd(l4, h4, 0x31);                      \
    } while(0)


/** Bitonic merge network for 2 x 8 elements without reversing B */
#define BITONIC8(O1, O2, O3, O4, A1, A2, B1, B2)                        \
    do {                                                                \
        /* Level-0 comparisons */                                       \
        __m256d l11 = _mm256_min_pd(A1, B1);                            \
        __m256d l12 = _mm256_min_pd(A2, B2);                            \
        __m256d h11 = _mm256_max_pd(A1, B1);                            \
        __m256d h12 = _mm256_max_pd(A2, B2);                            \
                                                                        \
        BITONIC4(O1, O2, l11, l12);                                     \
        BITONIC4(O3, O4, h11, h12);                                     \
    } while(0)


/** Bitonic merge kernel for 2 x 4 elements */
#define BITONIC_MERGE4(O1, O2, A, B)                                    \
    do {                                                                \
        /* reverse the order of input register B */                     \
        REVERSE(B);                                                     \
        BITONIC4(O1, O2, A, B);                                         \
    } while(0)


/** Bitonic merge kernel for 2 x 8 elements */
#define BITONIC_MERGE8(O1, O2, O3, O4, A1, A2, B1, B2)  \
        do {                                            \
            /* reverse the order of input B */          \
            REVERSE(B1);                                \
            REVERSE(B2);                                \
                                                        \
            /* Level-0 comparisons */                   \
            __m256d l11 = _mm256_min_pd(A1, B2);        \
            __m256d l12 = _mm256_min_pd(A2, B1);        \
            __m256d h11 = _mm256_max_pd(A1, B2);        \
            __m256d h12 = _mm256_max_pd(A2, B1);        \
                                                        \
            BITONIC4(O1, O2, l11, l12);                 \
            BITONIC4(O3, O4, h11, h12);                 \
        } while(0)

/** Bitonic merge kernel for 2 x 16 elements */
#define BITONIC_MERGE16(O1, O2, O3, O4, O5, O6, O7, O8,         \
                        A1, A2, A3, A4, B1, B2, B3, B4)         \
        do {                                                    \
            /** Bitonic merge kernel for 2 x 16 elemenets */    \
            /* reverse the order of input B */                  \
            REVERSE(B1);                                        \
            REVERSE(B2);                                        \
            REVERSE(B3);                                        \
            REVERSE(B4);                                        \
                                                                \
            /* Level-0 comparisons */                           \
            __m256d l01 = _mm256_min_pd(A1, B4);                \
            __m256d l02 = _mm256_min_pd(A2, B3);                \
            __m256d l03 = _mm256_min_pd(A3, B2);                \
            __m256d l04 = _mm256_min_pd(A4, B1);                \
            __m256d h01 = _mm256_max_pd(A1, B4);                \
            __m256d h02 = _mm256_max_pd(A2, B3);                \
            __m256d h03 = _mm256_max_pd(A3, B2);                \
            __m256d h04 = _mm256_max_pd(A4, B1);                \
                                                                \
            BITONIC8(O1, O2, O3, O4, l01, l02, l03, l04);       \
            BITONIC8(O5, O6, O7, O8, h01, h02, h03, h04);       \
        } while(0)
#else
/** Load aligned 1 AVX 512-bit registers from the given address */
#define LOAD8(REG, ADDR)                                                \
    do {                                                                \
        REG = _mm512_load_epi64((int64_t const *) ADDR);                \
    } while(0)

/** Load unaligned 1 AVX 512-bit registers from the given address */
#ifdef DEVELOPMENT_MODE
#define LOAD8U(REG, ADDR)                                                \
    do {                                                                 \
        REG = _mm512_mask_loadu_epi64(REG, 0xFF, (int64_t const *) ADDR);\
    } while(0)
#else
#define LOAD8U(REG, ADDR)                                                \
    do {                                                                 \
        REG = _mm512_loadu_epi64((int64_t const *) ADDR);                \
    } while(0)
#endif

/** Store 1 AVX 512-bit registers to the given address */
#define STORE8(ADDR, REG)                                           \
    do {                                                            \
        _mm512_store_epi64((int64_t *) ADDR, REG);                  \
    } while(0)

/** Store unaligned 1 AVX 512-bit registers to the given address */
#ifdef DEVELOPMENT_MODE
#define STORE8U(ADDR, REG)                                           \
    do {                                                             \
        _mm512_mask_storeu_epi64((int64_t *) ADDR, 0xFF, REG);       \
    } while(0)
#else
#define STORE8U(ADDR, REG)                                           \
    do {                                                             \
        _mm512_storeu_epi64((int64_t *) ADDR, REG);                  \
    } while(0)
#endif


/**
 * @note Reversing 64-bit values in an AVX register. It will be possible with
 * single _mm256_permute4x64_epi64() instruction in AVX2.
 */
#define REVERSE4x64(REG)                                 \
    do {                                                 \
        REG = _mm256_permute4x64_epi64(REG, 0b00011011); \
    } while(0)

/**
 * @note Reversing 64-bit values in an AVX-512 register. 
 */
#define REVERSE8x64(REG)                                             \
    do {                                                             \
        const __m512i shuffler = _mm512_set_epi64(0,1,2,3,4,5,6,7);  \
        REG = _mm512_permutexvar_epi64(shuffler, REG);               \
    } while(0)


/** Bitonic merge kernel for 2 x 4 elements after the reversing step. */
#define BITONIC4(O1, O2, A, B)                                          \
    do {                                                                \
        /* Level-1 comparisons */                                       \
        __m256i l1 = _mm256_min_epi64(A, B);                            \
        __m256i h1 = _mm256_max_epi64(A, B);                            \
                                                                        \
        /* Level-1 shuffles */                                          \
        __m256i l1p = _mm256_permute2f128_si256(l1, h1, 0x31);          \
        __m256i h1p = _mm256_permute2f128_si256(l1, h1, 0x20);          \
                                                                        \
        /* Level-2 comparisons */                                       \
        __m256i l2 = _mm256_min_epi64(l1p, h1p);                        \
        __m256i h2 = _mm256_max_epi64(l1p, h1p);                        \
                                                                        \
        /* Level-2 shuffles */                                          \
        __m256i l2p = _mm256_unpacklo_epi64(l2, h2);                    \
        __m256i h2p = _mm256_unpackhi_epi64(l2, h2);                    \
                                                                        \
        /* Level-3 comparisons */                                       \
        __m256i l3 = _mm256_min_epi64(l2p, h2p);                        \
        __m256i h3 = _mm256_max_epi64(l2p, h2p);                        \
                                                                        \
        /* Level-3 shuffles implemented with unpcklps unpckhps */       \
        /* AVX cannot shuffle both inputs from same 128-bit lane */     \
        /* so we need 2 more instructions for this operation. */        \
        __m256i l4 = _mm256_unpacklo_epi64(l3, h3);                     \
        __m256i h4 = _mm256_unpackhi_epi64(l3, h3);                     \
        O1 = _mm256_permute2f128_si256(l4, h4, 0x20);                   \
        O2 = _mm256_permute2f128_si256(l4, h4, 0x31);                   \
    } while(0)

/** Bitonic merge network for 2 x 8 elements without reversing B */
#define BITONIC8(O1, O2, O3, O4, A, B)                        \
    do {                                                              \
        /* Level-0 comparisons */                                     \
        __m512i l1 = _mm512_min_epi64(A, B);                          \
        __m512i h1 = _mm512_max_epi64(A, B);                          \
        __m256i l11 = _mm512_extracti64x4_epi64(l1, 0);               \
        __m256i l12 = _mm512_extracti64x4_epi64(l1, 1);               \
        __m256i h11 = _mm512_extracti64x4_epi64(h1, 0);               \
        __m256i h12 = _mm512_extracti64x4_epi64(h1, 1);               \
        BITONIC4(O1, O2, l11, l12);                                   \
        BITONIC4(O3, O4, h11, h12);                                   \
    } while(0)


/** Bitonic merge kernel for 2 x 4 elements */
#define BITONIC_MERGE4(O1, O2, A, B)                                    \
    do {                                                                \
        /* reverse the order of input register B */                     \
        REVERSE4x64(B);                                                 \
        BITONIC4(O1, O2, A, B);                                         \
    } while(0)


/** Bitonic merge kernel for 2 x 8 elements */
#define BITONIC_MERGE8(O1, O2, A, B)  \
        do {                                            \
            /* reverse the order of input B */          \
            REVERSE8x64(B);                             \
                                                        \
            /* Level-0 comparisons */                   \
            __m512i l1 = _mm512_min_epi64(A, B);        \
            __m512i h1 = _mm512_max_epi64(A, B);        \
            __m256i l11 = _mm512_extracti64x4_epi64(l1, 0);               \
            __m256i l12 = _mm512_extracti64x4_epi64(l1, 1);               \
            __m256i h11 = _mm512_extracti64x4_epi64(h1, 0);               \
            __m256i h12 = _mm512_extracti64x4_epi64(h1, 1);               \
                                                          \
            __m256i O11, O12;                             \
            BITONIC4(O11, O12, l11, l12);                 \
            int64_t* O11_ptr = (int64_t*) &O11;           \
            int64_t* O12_ptr = (int64_t*) &O12;           \
            O1 = _mm512_set_epi64(O12_ptr[3], O12_ptr[2], O12_ptr[1], O12_ptr[0], O11_ptr[3], O11_ptr[2], O11_ptr[1], O11_ptr[0]);              \
                                                          \
            __m256i O21, O22;                             \
            BITONIC4(O21, O22, h11, h12);                 \
            int64_t* O21_ptr = (int64_t*) &O21;           \
            int64_t* O22_ptr = (int64_t*) &O22;           \
            O2 = _mm512_set_epi64(O22_ptr[3], O22_ptr[2], O22_ptr[1], O22_ptr[0], O21_ptr[3], O21_ptr[2], O21_ptr[1], O21_ptr[0]);              \
        } while(0)

/** Bitonic merge kernel for 2 x 16 elements */
#define BITONIC_MERGE16(O1, O2, O3, O4,         \
                        A1, A2, B1, B2)                         \
        do {                                                    \
            /** Bitonic merge kernel for 2 x 16 elemenets */    \
            /* reverse the order of input B */                  \
            REVERSE8x64(B1);                                    \
            REVERSE8x64(B2);                                    \
                                                                \
            /* Level-0 comparisons */                           \
            __m512i l01 = _mm512_min_epi64(A1, B2);             \
            __m512i l02 = _mm512_min_epi64(A2, B1);             \
            __m512i h01 = _mm512_max_epi64(A1, B2);             \
            __m512i h02 = _mm512_max_epi64(A2, B1);             \
                                                                \
            __m256i O11, O12, O21, O22;                         \
            BITONIC8(O11, O12, O21, O22, l01, l02);             \
            int64_t* O11_ptr = (int64_t*) &O11;                 \
            int64_t* O12_ptr = (int64_t*) &O12;                 \
            int64_t* O21_ptr = (int64_t*) &O21;                 \
            int64_t* O22_ptr = (int64_t*) &O22;                 \
            O1 = _mm512_set_epi64(O12_ptr[3], O12_ptr[2], O12_ptr[1], O12_ptr[0], O11_ptr[3], O11_ptr[2], O11_ptr[1], O11_ptr[0]);              \
            O2 = _mm512_set_epi64(O22_ptr[3], O22_ptr[2], O22_ptr[1], O22_ptr[0], O21_ptr[3], O21_ptr[2], O21_ptr[1], O21_ptr[0]);              \
                                                                \
            __m256i O31, O32, O41, O42;                         \
            BITONIC8(O31, O32, O41, O42, h01, h02);             \
            int64_t* O31_ptr = (int64_t*) &O31;                 \
            int64_t* O32_ptr = (int64_t*) &O32;                 \
            int64_t* O41_ptr = (int64_t*) &O41;                 \
            int64_t* O42_ptr = (int64_t*) &O42;                 \
            O3 = _mm512_set_epi64(O32_ptr[3], O32_ptr[2], O32_ptr[1], O32_ptr[0], O31_ptr[3], O31_ptr[2], O31_ptr[1], O31_ptr[0]);              \
            O4 = _mm512_set_epi64(O42_ptr[3], O42_ptr[2], O42_ptr[1], O42_ptr[0], O41_ptr[3], O41_ptr[2], O41_ptr[1], O41_ptr[0]);              \
                                                                \
        } while(0)
#endif

/** 
 * There are 2 ways to implement branches: 
 *     1) With conditional move instr.s using inline assembly (IFELSEWITHCMOVE).
 *     2) With software predication (IFELSEWITHPREDICATION).
 *     3) With normal if-else
 */
#if IFELSEWITHCMOVE
#define IFELSECONDMOVE(NXT, INA, INB, INCR)                             \
    do {                                                                \
        register block4 * tmpA, * tmpB;                                 \
        register int64_t tmpKey;                                        \
                                                                        \
        __asm__ ( "mov %[A], %[tmpA]\n"         /* tmpA <-- inA      */ \
                  "add %[INC], %[A]\n"          /* inA += 4          */ \
                  "mov %[B], %[tmpB]\n"         /* tmpB <-- inB      */ \
                  "mov (%[tmpA]), %[tmpKey]\n"  /* tmpKey <-- *inA   */ \
                  "add %[INC], %[B]\n"          /* inB += 4          */ \
                  "mov %[tmpA], %[NEXT]\n"      /* next <-- A        */ \
                  "cmp (%[tmpB]), %[tmpKey]\n"  /* cmp(tmpKey,*inB ) */ \
                  "cmovnc %[tmpB], %[NEXT]\n"   /* if(A>=B) next<--B */ \
                  "cmovnc %[tmpA], %[A]\n"      /* if(A>=B) A<--oldA */ \
                  "cmovc %[tmpB], %[B]\n"       /* if(A<B)  B<--oldB */ \
                  : [A] "=r" (INA), [B] "=r" (INB), [NEXT] "=r" (NXT),  \
                    [tmpA] "=r" (tmpA), [tmpB] "=r" (tmpB),             \
                    [tmpKey] "=r" (tmpKey)                              \
                  : "0" (INA), "1" (INB), [INC] "i" (INCR)              \
                  :                                                     \
                  );                                                    \
    } while(0)

#elif IFELSEWITHPREDICATION
#define IFELSECONDMOVE(NXT, INA, INB, INCR)                 \
    do {                                                    \
        int8_t cmp = *((int64_t *)INA) < *((int64_t *)INB); \
        NXT  = cmp ? INA : INB;                             \
        INA += cmp;                                         \
        INB += !cmp;                                        \
    } while(0)

#elif IFELSEWITHNORMAL
#define IFELSECONDMOVE(NXT, INA, INB, INCR)                 \
            do {                                            \
                if(*((int64_t *)INA) < *((int64_t *)INB)) { \
                    NXT = INA;                              \
                    INA ++;                                 \
                }                                           \
                else {                                      \
                    NXT = INB;                              \
                    INB ++;                                 \
                }                                           \
            } while(0)                                      \

#endif

#endif /* AVXCOMMON_H */
