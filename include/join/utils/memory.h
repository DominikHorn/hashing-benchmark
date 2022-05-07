# pragma once

/* Common memory utilities for all join algorithms */

#include <stdio.h>              /* perror */
#include <stdlib.h>             /* posix_memalign */
#include <smmintrin.h>          /* simd only for 32-bit keys – SSE4.1 */
#include <immintrin.h>

#include "configs/base_configs.h"
#include "utils/data_structures.h"

#ifdef HAVE_LIBNUMA
#include <numa.h>               /* numa_alloc_local(), numa_free() */
#endif


/** checks malloc() result based on ETH implementation */
#ifndef MALLOC_CHECK
#define MALLOC_CHECK(M)                                                 \
    if(!M){                                                             \
        printf("[ERROR] MALLOC_CHECK: %s : %d\n", __FILE__, __LINE__);  \
        perror(": malloc() failed!\n");                                 \
        exit(EXIT_FAILURE);                                             \
    }
#endif

/** Align a pointer to given size */
#define ALIGNPTR(PTR, SZ) (((uintptr_t)PTR + (SZ-1)) & ~(SZ-1))

/** Align N to number of tuples that is a multiple of cache lines */
#define ALIGN_NUMTUPLES(TUPLESPERCACHELINE, N) ((N+TUPLESPERCACHELINE-1) & ~(TUPLESPERCACHELINE-1))

void * alloc_aligned(size_t size)
{
    void * ret;
    int rv;
    rv = posix_memalign((void**)&ret, CACHE_LINE_SIZE, size);

    if (rv) { 
        perror("alloc_aligned() failed: out of memory");
        return 0; 
    }
    
    return ret;
}

void *
alloc_aligned_threadlocal(size_t size)
{
#ifdef HAVE_LIBNUMA
    return numa_alloc_local(size);
#else
    return alloc_aligned(size);
#endif
}

void
free_threadlocal(void * ptr, size_t size)
{
#ifdef HAVE_LIBNUMA
    numa_free(ptr, size);
#else
    free(ptr);
#endif
}

/** software prefetching function based on ETH implementation */
//inline void prefetch(void * addr) __attribute__((always_inline));

//inline void prefetch(void * addr)
//{
//    /* #ifdef __x86_64__ */
//    __asm__ __volatile__ ("prefetcht0 %0" :: "m" (*(uint32_t*)addr));
//    /* _mm_prefetch(addr, _MM_HINT_T0); */
//    /* #endif */
//}

/** 
 * Makes a non-temporal write of 64 bytes from src to dst.
 * Uses vectorized non-temporal stores if available, falls
 * back to assignment copy (Based on the ETH implementation). 
 */
template<class KeyType, class PayloadType>
inline void store_nontemp_64B(void * dst, void * src);


template<class KeyType, class PayloadType>
inline void store_nontemp_64B(void * dst, void * src)
{

#ifndef TYPICAL_COPY_MODE
#ifdef __AVX__
    register __m256i * d1 = (__m256i*) dst;
    register __m256i s1 = *((__m256i*) src);
    register __m256i * d2 = d1+1;
    register __m256i s2 = *(((__m256i*) src)+1);

    _mm256_stream_si256(d1, s1);
    _mm256_stream_si256(d2, s2);

#elif defined(__SSE2__)

    register __m128i * d1 = (__m128i*) dst;
    register __m128i * d2 = d1+1;
    register __m128i * d3 = d1+2;
    register __m128i * d4 = d1+3;
    register __m128i s1 = *(__m128i*) src;
    register __m128i s2 = *((__m128i*)src + 1);
    register __m128i s3 = *((__m128i*)src + 2);
    register __m128i s4 = *((__m128i*)src + 3);

    _mm_stream_si128 (d1, s1);
    _mm_stream_si128 (d2, s2);
    _mm_stream_si128 (d3, s3);
    _mm_stream_si128 (d4, s4);

#else
    /* just copy with assignment */
    *(CacheLine<KeyType, PayloadType> *)dst = *(CacheLine<KeyType, PayloadType> *)src;

#endif
#else
    /* just copy with assignment */
    *(CacheLine<KeyType, PayloadType> *)dst = *(CacheLine<KeyType, PayloadType> *)src;
#endif

}
