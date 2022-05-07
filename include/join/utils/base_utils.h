#pragma once

/* Common printing and debugging utils for all join algorithms */

#include "emmintrin.h"
#include "immintrin.h"
#include "smmintrin.h"

#include <stdint.h>
#include <string.h>

#include "configs/base_configs.h"

/* just to enable compilation with g++ */
//#if defined(__cplusplus)
//#undef restrict
//#define restrict __restrict__
//#endif

typedef void * (*THREADFUNCPTR)(void *);

#ifdef DEVELOPMENT_MODE
#define DEBUGMSG(COND, MSG, ...)                                    \
    if(COND) { printf(MSG, ## __VA_ARGS__); }
#else
#define DEBUGMSG(COND, MSG, ...) 
#endif


void print256_num(__m256d var)
{
    double val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %lf %lf %lf %lf \n",
           val[0], val[1], val[2], val[3]);
}

void print256i_num(__m256i var)
{
    int64_t val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %ld %ld %ld %ld \n",
           val[0], val[1], val[2], val[3]);
}

void print256i32_num(__m256i var)
{
    int32_t val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %ld %ld %ld %ld %ld %ld %ld %ld \n",
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
}


void print512_num(__m512d var)
{
    double val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %lf %lf %lf %lf %lf %lf %lf %lf \n",
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
}

void print512dtoi_num(__m512d var)
{
    int64_t val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %ld %ld %ld %ld %ld %ld %ld %ld \n",
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
}

void print512i_num(__m512i var)
{
    int64_t val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %ld %ld %ld %ld %ld %ld %ld %ld \n",
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
}