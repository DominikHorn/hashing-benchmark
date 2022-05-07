#pragma once

/* Common math and hashing utils adapted from ETH (https://systems.ethz.ch/research/data-processing-on-modern-hardware/projects/parallel-and-distributed-joins.html) 
   and Stanford (http://graphics.stanford.edu/~seander/bithacks.html) */

#include <stdlib.h>             /* posix_memalign */
#include <math.h>               /* fmod, pow */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>

/* #define RADIX_HASH(V)  ((V>>7)^(V>>13)^(V>>21)^V) */
#define HASH_BIT_MODULO(K, MASK, NBITS) (((K) & MASK) >> NBITS)

#ifndef HASH
#define HASH(X, MASK, SKIP) (((X) & MASK) >> SKIP)
#endif


#ifndef NEXT_POW_2
/** 
 *  compute the next number, greater than or equal to 32-bit unsigned v.
 *  taken from "bit twiddling hacks":
 *  http://graphics.stanford.edu/~seander/bithacks.html
 */
#define NEXT_POW_2(V)                           \
    do {                                        \
        V--;                                    \
        V |= V >> 1;                            \
        V |= V >> 2;                            \
        V |= V >> 4;                            \
        V |= V >> 8;                            \
        V |= V >> 16;                           \
        V++;                                    \
    } while(0)
#endif

#ifndef MAX
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#endif

#ifndef MIN
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#endif

/* return a random number in range [0,N] */
#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))

#define divRoundDown(n,s)  ((n) / (s))
#define divRoundUp(n,s)    (((n) / (s)) + ((((n) % (s)) > 0) ? 1 : 0))

// Finalization step of Murmur3 hash (based on the stanford implementation for cuckoo hashing)
uint32_t murmur_hash_32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x85ebca6b;
  value ^= value >> 13;
  value *= 0xc2b2ae35;
  value ^= value >> 16;
  return value;
}

void avx_murmur_hash_32(uint64_t * items, uint64_t * results) 
{
  const __m512i c1 = _mm512_set1_epi64(0x85ebca6b);
  const __m512i c2 = _mm512_set1_epi64(0xc2b2ae35);

  __m512i k = _mm512_load_epi64((uint64_t *)(items));

  __m512i s1 = _mm512_srli_epi64(k, 16);
  __m512i x1 = _mm512_xor_epi64(k, s1);
  __m512i m1 = _mm512_mullo_epi64(x1, c1);

  __m512i s2 = _mm512_srli_epi64(m1, 13);
  __m512i x2 = _mm512_xor_epi64(m1, s2);
  __m512i m2 = _mm512_mullo_epi64(x2, c2);

  __m512i s3 = _mm512_srli_epi64(m2, 16);
  __m512i x3 = _mm512_xor_epi64(m2, s3);

  _mm512_store_epi64((uint64_t *) results, x3); 
}

// Fast alternative to modulo from Daniel Lemire  (based on the stanford implementation for cuckoo hashing)
uint32_t alt_mod(uint32_t x, uint32_t n) {
  return ((uint64_t) x * (uint64_t) n) >> 32 ;
}

inline double calc_linear(double x, double y, double z) {
    return std::fma(x, y, z);
}

inline size_t calc_bound(double inp, double bound) {
  if (inp < 0.0) return 0;
  return (inp > bound ? bound : (size_t)inp);
}
