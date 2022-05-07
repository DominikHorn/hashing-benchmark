/**
 * @file    scalar_multiwaymerge.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Tue Dec 11 18:24:10 2012
 * @version $Id $
 *
 * @brief   Scalar Multi-Way merging with cache-resident buffers.
 *
 * (c) 2014, ETH Zurich, Systems Group
 *
 */
#ifndef SCALARMULTIWAYMERGE_H
#define SCALARMULTIWAYMERGE_H

#include <assert.h> /* assert() */
#include <string.h> /* memcpy() */
#include <stdio.h>  /* printf() */
#include <math.h>   /* log2()   */

#include "configs/base_configs.h"
#include "configs/eth_configs.h"

#include "utils/base_utils.h"
#include "utils/data_structures.h" /* tuple_t, relation_t */
#include "utils/eth_data_structures.h"
#include "utils/math.h"


/**
 * Scalar Multi-Way Merging with cache-resident merge buffers.
 *
 * @note this implementation is the most optimized version, it eliminates
 * modulo operation for ring buffers and decomposes the merge of ring buffers.
 *
 * @param output resulting merged runs
 * @param parts input relations to merge
 * @param nparts number of input relations (fan-in)
 * @param bufntuples fifo buffer size in number of tuples
 * @param fifobuffer cache-resident fifo buffer
 *
 * @return total number of tuples
 */
template<class KeyType, class PayloadType>
uint64_t
scalar_multiway_merge(Tuple<KeyType, PayloadType> * output,
                      Relation<KeyType, PayloadType> ** parts,
                      uint32_t nparts,
                      Tuple<KeyType, PayloadType> * fifobuffer,
                      uint32_t bufntuples);

/**
 * Scalar Multi-Way Merging with cache-resident merge buffers.
 *
 * @note this implementation uses modulo operation for accessing ring buffers.
 *
 * @param output resulting merged runs
 * @param parts input relations to merge
 * @param nparts number of input relations (fan-in)
 * @param bufntuples fifo buffer size in number of tuples
 * @param fifobuffer cache-resident fifo buffer
 *
 * @return total number of tuples
 */
template<class KeyType, class PayloadType>
uint64_t
scalar_multiway_merge_modulo(Tuple<KeyType, PayloadType> * output,
                      Relation<KeyType, PayloadType> ** parts,
                      uint32_t nparts,
                      Tuple<KeyType, PayloadType> * fifobuffer,
                      uint32_t bufntuples);

/**
 * Scalar Multi-Way Merging with cache-resident merge buffers.
 *
 * @note this implementation uses bit-and'ing for accessing ring buffers.
 *       However, it requires ring buffer size to be power of 2, thus
 *       reducing effective cache usage.
 *
 * @param output resulting merged runs
 * @param parts input relations to merge
 * @param nparts number of input relations (fan-in)
 * @param bufntuples fifo buffer size in number of tuples
 * @param fifobuffer cache-resident fifo buffer
 *
 * @return total number of tuples
 */
template<class KeyType, class PayloadType>
uint64_t
scalar_multiway_merge_bitand(Tuple<KeyType, PayloadType> * output,
                      Relation<KeyType, PayloadType> ** parts,
                      uint32_t nparts,
                      Tuple<KeyType, PayloadType> * fifobuffer,
                      uint32_t bufntuples);


/*******************************************************************************
 *                                                                             *
 *                     Scalar Merge Declerations                               *
 *                                                                             *
 *******************************************************************************/
/********************* With Ring-Buffer Decomposing ****************************/
/** Scalar read & merge from 2 lists into the merge node ring buffer */
template<class KeyType, class PayloadType>
uint32_t
readmerge_scalar_decomposed(MergeNode<KeyType, PayloadType> * node,
                            Tuple<KeyType, PayloadType> ** inA,
                            Tuple<KeyType, PayloadType> ** inB,
                            uint32_t lenA,
                            uint32_t lenB,
                            uint32_t fifosize);

/** Read from 2 children nodes and merge by decomposing the ring-bufs */
template<class KeyType, class PayloadType>
void
merge_scalar_decomposed(MergeNode<KeyType, PayloadType> * node,
                        MergeNode<KeyType, PayloadType> * right,
                        MergeNode<KeyType, PayloadType> * left,
                        uint32_t fifosize, uint8_t rightdone, uint8_t leftdone);

/** Read from 2 children nodes and merge/store by decomposing the ring-bufs */
template<class KeyType, class PayloadType>
uint64_t
mergestore_scalar_decomposed(MergeNode<KeyType, PayloadType> * right,
                             MergeNode<KeyType, PayloadType> * left,
                             Tuple<KeyType, PayloadType> ** output,
                             uint32_t fifosize,
                             uint8_t rightdone,
                             uint8_t leftdone);

/************************ With Plain Modulo ************************************/
/** Scalar read & merge from 2 lists by just modulo iteration over ring buf. */
template<class KeyType, class PayloadType>
uint32_t
readmerge_scalar_modulo(MergeNode<KeyType, PayloadType> * node,
                        Tuple<KeyType, PayloadType> ** inA,
                        Tuple<KeyType, PayloadType> ** inB,
                        uint32_t lenA,
                        uint32_t lenB,
                        uint32_t fifosize);

/** Merge 2 children nodes by just modulo iteration over ring-bufs. */
template<class KeyType, class PayloadType>
void
merge_scalar_modulo(MergeNode<KeyType, PayloadType> * node,
                    MergeNode<KeyType, PayloadType> * right,
                    MergeNode<KeyType, PayloadType> * left,
                    uint32_t fifosize,
                    uint8_t rightdone, uint8_t leftdone);

/** Read from 2 nodes & merge/store by just modulo over ring-bufs. */
template<class KeyType, class PayloadType>
uint64_t
mergestore_scalar_modulo(MergeNode<KeyType, PayloadType> * right,
                         MergeNode<KeyType, PayloadType> * left,
                         Tuple<KeyType, PayloadType> ** output,
                         uint32_t fifosize,
                         uint8_t rightdone,
                         uint8_t leftdone);

/************************ With BitAND-Modulo ***********************************/
/** Scalar read & merge from 2 lists by bitand-modulo iteration over ring buf. */
template<class KeyType, class PayloadType>
uint32_t
readmerge_scalar_bitand(MergeNode<KeyType, PayloadType> * node,
                        Tuple<KeyType, PayloadType> ** inA,
                        Tuple<KeyType, PayloadType> ** inB,
                        uint32_t lenA,
                        uint32_t lenB,
                        uint32_t fifosize);

/** Merge 2 children nodes by bitand-modulo iteration over ring-bufs. */
template<class KeyType, class PayloadType>
void
merge_scalar_bitand(MergeNode<KeyType, PayloadType> * node,
                    MergeNode<KeyType, PayloadType> * right,
                    MergeNode<KeyType, PayloadType> * left,
                    uint32_t fifosize,
                    uint8_t rightdone, uint8_t leftdone);

/** Read from 2 nodes & merge/store by bitand-modulo over ring-bufs. */
template<class KeyType, class PayloadType>
uint64_t
mergestore_scalar_bitand(MergeNode<KeyType, PayloadType> * right,
                         MergeNode<KeyType, PayloadType> * left,
                         Tuple<KeyType, PayloadType> ** output,
                         uint32_t fifosize,
                         uint8_t rightdone,
                         uint8_t leftdone);

/*******************************************************************************
 *                                                                             *
 *                        Scalar Implementations                               *
 *                                                                             *
 *******************************************************************************/
/************* Scalar Multi-Way Merging using one of the above *****************/
/* Parameters: MWAYMERGE_DECOMPOSED, MWAYMERGE_MODULO, MWAYMERGE_BITAND        */

template<class KeyType, class PayloadType>
uint64_t
scalar_multiway_merge(Tuple<KeyType, PayloadType> * output,
                      Relation<KeyType, PayloadType> ** parts,
                      uint32_t nparts,
                      Tuple<KeyType, PayloadType> * fifobuffer,
                      uint32_t bufntuples)
{
    uint64_t totalmerged = 0;
    uint32_t nfifos        = nparts-2;
    uint32_t totalfifosize = bufntuples - nparts -
                             (nfifos * sizeof(MergeNode<KeyType, PayloadType>)
                              + nfifos * sizeof(uint8_t)
                              + nparts * sizeof(Relation<KeyType, PayloadType>)
                              + sizeof(Tuple<KeyType, PayloadType>) - 1) / sizeof(Tuple<KeyType, PayloadType>);

    uint32_t      fifosize   = totalfifosize / nfifos;
    MergeNode<KeyType, PayloadType> * nodes      = (MergeNode<KeyType, PayloadType> *)(fifobuffer + totalfifosize);
    uint8_t *     done       = (uint8_t *)(nodes + nfifos);

    /* printf("[INFO ] fifosize = %d, totalfifosize = %d tuples\n",  */
    /*        fifosize, totalfifosize); */

    for(uint32_t i = 0; i < nfifos; i++) {
        nodes[i].buffer = fifobuffer + fifosize * i;
        nodes[i].count  = 0;
        nodes[i].head   = 0;
        nodes[i].tail   = 0;
        done[i]         = 0;
    }

    uint32_t finished = 0;
    const uint32_t readthreshold = fifosize/2;

    while(!finished) {
        finished = 1;
        int m = nfifos - 1;

        /* first iterate through leafs and read as much data as possible */
        for(uint32_t c = 0; c < nparts; c += 2, m--) {
            if(!done[m] && (nodes[m].count < readthreshold)) {

                uint32_t A = c;
                uint32_t B = c + 1;
                Tuple<KeyType, PayloadType> * inA = parts[A]->tuples;
                Tuple<KeyType, PayloadType> * inB = parts[B]->tuples;

                uint32_t nread;

                /* if(!check_node_sorted(&nodes[m], fifosize)){ */
                /*     printf("before read:: Node not sorted\n"); */
                /*     exit(0); */
                /* } */
                nread = readmerge_scalar_decomposed(&nodes[m], &inA, &inB,
                                                    parts[A]->num_tuples,
                                                    parts[B]->num_tuples,
                                                    fifosize);


                /* if(!check_node_sorted(&nodes[m], fifosize)){ */
                /*     printf("after read:: Node not sorted\n"); */
                /*     exit(0); */
                /* } */

                parts[A]->num_tuples -= (inA - parts[A]->tuples);
                parts[B]->num_tuples -= (inB - parts[B]->tuples);

                parts[A]->tuples = inA;
                parts[B]->tuples = inB;

                done[m] = (nread == 0
                           || ((parts[A]->num_tuples == 0)
                               && (parts[B]->num_tuples == 0)));

                finished &= done[m];
            }
        }

        /* now iterate inner nodes and do merge for ready nodes */
        for(; m >= 0; m--) {
            if(!done[m]) {
                int r = 2*m+2;
                int l = r + 1;
                MergeNode<KeyType, PayloadType> * right = &nodes[r];
                MergeNode<KeyType, PayloadType> * left  = &nodes[l];

                uint8_t children_done = (done[r] | done[l]);

                if((children_done || nodes[m].count < readthreshold)
                   && nodes[m].count < fifosize) {
                    if(children_done || (right->count >= readthreshold
                                         && left->count >= readthreshold)) {

                        /* if(!check_node_sorted(right, fifosize)) */
                        /*     printf("Right Node not sorted\n"); */

                        /* if(!check_node_sorted(left, fifosize)) */
                        /*     printf("Left Node not sorted\n"); */

                        /* do a merge on right and left */
                        merge_scalar_decomposed(&nodes[m], right, left, fifosize,
                                done[r], done[l]//fullmerge//children_done /* full-merge? */
                                        );


                        /* if(!check_node_sorted(right, fifosize)) */
                        /*     printf("After merge- Right Node not sorted\n"); */

                        /* if(!check_node_sorted(left, fifosize)) */
                        /*     printf("After merge- Left Node not sorted\n"); */

                        /* if(!check_node_sorted(&nodes[m], fifosize)) */
                        /*     printf("After merge - Node not sorted\n"); */
                    }
                    done[m] = (done[r] & done[l]) &&
                              (right->count == 0 && left->count == 0);
                }

                finished &= done[m];
            }
        }

        totalmerged +=
        /* finally iterate for the root node and store data */
        mergestore_scalar_decomposed(&nodes[0], &nodes[1], &output, fifosize,
                                     done[0], done[1] /* full-merge? */
                                     );

    }

    return totalmerged;
}

template<class KeyType, class PayloadType>
uint64_t
scalar_multiway_merge_modulo(Tuple<KeyType, PayloadType> * output,
                      Relation<KeyType, PayloadType> ** parts,
                      uint32_t nparts,
                      Tuple<KeyType, PayloadType> * fifobuffer,
                      uint32_t bufntuples)
{
    uint64_t totalmerged = 0;
    uint32_t nfifos        = nparts-2;
    uint32_t totalfifosize = bufntuples - nparts -
                             (nfifos * sizeof(MergeNode<KeyType, PayloadType>)
                              + nfifos * sizeof(uint8_t)
                              + nparts * sizeof(Relation<KeyType, PayloadType>)
                              + sizeof(Tuple<KeyType, PayloadType>) - 1) / sizeof(Tuple<KeyType, PayloadType>);

    uint32_t      fifosize   = totalfifosize / nfifos;
    MergeNode<KeyType, PayloadType> * nodes      = (MergeNode<KeyType, PayloadType> *)(fifobuffer + totalfifosize);
    uint8_t *     done       = (uint8_t *)(nodes + nfifos);

    /* printf("[INFO ] fifosize = %d, totalfifosize = %d tuples\n",  */
    /*        fifosize, totalfifosize); */

    for(uint32_t i = 0; i < nfifos; i++) {
        nodes[i].buffer = fifobuffer + fifosize * i;
        nodes[i].count  = 0;
        nodes[i].head   = 0;
        nodes[i].tail   = 0;
        done[i]         = 0;
    }

    uint32_t finished = 0;
    const uint32_t readthreshold = fifosize/2;

    while(!finished) {
        finished = 1;
        int m = nfifos - 1;

        /* first iterate through leafs and read as much data as possible */
        for(uint32_t c = 0; c < nparts; c += 2, m--) {
            if(!done[m] && (nodes[m].count < readthreshold)) {

                uint32_t A = c;
                uint32_t B = c + 1;
                Tuple<KeyType, PayloadType> * inA = parts[A]->tuples;
                Tuple<KeyType, PayloadType> * inB = parts[B]->tuples;

                uint32_t nread;

                /* if(!check_node_sorted(&nodes[m], fifosize)){ */
                /*     printf("before read:: Node not sorted\n"); */
                /*     exit(0); */
                /* } */

                nread = readmerge_scalar_modulo(&nodes[m], &inA, &inB,
                                                parts[A]->num_tuples,
                                                parts[B]->num_tuples,
                                                fifosize);

                /* if(!check_node_sorted(&nodes[m], fifosize)){ */
                /*     printf("after read:: Node not sorted\n"); */
                /*     exit(0); */
                /* } */

                parts[A]->num_tuples -= (inA - parts[A]->tuples);
                parts[B]->num_tuples -= (inB - parts[B]->tuples);

                parts[A]->tuples = inA;
                parts[B]->tuples = inB;

                done[m] = (nread == 0
                           || ((parts[A]->num_tuples == 0)
                               && (parts[B]->num_tuples == 0)));

                finished &= done[m];
            }
        }

        /* now iterate inner nodes and do merge for ready nodes */
        for(; m >= 0; m--) {
            if(!done[m]) {
                int r = 2*m+2;
                int l = r + 1;
                MergeNode<KeyType, PayloadType> * right = &nodes[r];
                MergeNode<KeyType, PayloadType> * left  = &nodes[l];

                uint8_t children_done = (done[r] | done[l]);

                if((children_done || nodes[m].count < readthreshold)
                   && nodes[m].count < fifosize) {
                    if(children_done || (right->count >= readthreshold
                                         && left->count >= readthreshold)) {

                        /* if(!check_node_sorted(right, fifosize)) */
                        /*     printf("Right Node not sorted\n"); */

                        /* if(!check_node_sorted(left, fifosize)) */
                        /*     printf("Left Node not sorted\n"); */

                        /* do a merge on right and left */

                        merge_scalar_modulo(&nodes[m], right, left, fifosize,
                                done[r], done[l]//children_done /* full-merge? */
                                        );

                        /* if(!check_node_sorted(right, fifosize)) */
                        /*     printf("After merge- Right Node not sorted\n"); */

                        /* if(!check_node_sorted(left, fifosize)) */
                        /*     printf("After merge- Left Node not sorted\n"); */

                        /* if(!check_node_sorted(&nodes[m], fifosize)) */
                        /*     printf("After merge - Node not sorted\n"); */
                    }
                    done[m] = (done[r] & done[l]) &&
                              (right->count == 0 && left->count == 0);
                }

                finished &= done[m];
            }
        }

        totalmerged +=
        /* finally iterate for the root node and store data */
        mergestore_scalar_modulo(&nodes[0], &nodes[1], &output, fifosize,
                                     done[0], done[1] /* full-merge? */
                                     );

    }

    return totalmerged;
}

template<class KeyType, class PayloadType>
uint64_t
scalar_multiway_merge_bitand(Tuple<KeyType, PayloadType> * output,
                      Relation<KeyType, PayloadType> ** parts,
                      uint32_t nparts,
                      Tuple<KeyType, PayloadType> * fifobuffer,
                      uint32_t bufntuples)
{
    uint64_t totalmerged = 0;
    uint32_t nfifos        = nparts-2;
    uint32_t totalfifosize = bufntuples - nparts -
                             (nfifos * sizeof(MergeNode<KeyType, PayloadType>)
                              + nfifos * sizeof(uint8_t)
                              + nparts * sizeof(Relation<KeyType, PayloadType>)
                              + sizeof(Tuple<KeyType, PayloadType>) - 1) / sizeof(Tuple<KeyType, PayloadType>);

    /* we make sure fifosize is power of 2 */
    uint32_t      fifosize   = 1 << (int)(log2(totalfifosize / nfifos));
    MergeNode<KeyType, PayloadType> * nodes      = (MergeNode<KeyType, PayloadType> *)(fifobuffer + totalfifosize);
    uint8_t *     done       = (uint8_t *)(nodes + nfifos);

    /* printf("[INFO ] fifosize = %d, totalfifosize = %d tuples\n",  */
    /*        fifosize, totalfifosize); */

    for(uint32_t i = 0; i < nfifos; i++) {
        nodes[i].buffer = fifobuffer + fifosize * i;
        nodes[i].count  = 0;
        nodes[i].head   = 0;
        nodes[i].tail   = 0;
        done[i]         = 0;
    }

    uint32_t finished = 0;
    const uint32_t readthreshold = fifosize/2;

    while(!finished) {
        finished = 1;
        int m = nfifos - 1;

        /* first iterate through leafs and read as much data as possible */
        for(uint32_t c = 0; c < nparts; c += 2, m--) {
            if(!done[m] && (nodes[m].count < readthreshold)) {

                uint32_t A = c;
                uint32_t B = c + 1;
                Tuple<KeyType, PayloadType> * inA = parts[A]->tuples;
                Tuple<KeyType, PayloadType> * inB = parts[B]->tuples;

                uint32_t nread;

                /* if(!check_node_sorted(&nodes[m], fifosize)){ */
                /*     printf("before read:: Node not sorted\n"); */
                /*     exit(0); */
                /* } */

                nread = readmerge_scalar_bitand(&nodes[m], &inA, &inB,
                                                parts[A]->num_tuples,
                                                parts[B]->num_tuples,
                                                fifosize);

                /* if(!check_node_sorted(&nodes[m], fifosize)){ */
                /*     printf("after read:: Node not sorted\n"); */
                /*     exit(0); */
                /* } */

                parts[A]->num_tuples -= (inA - parts[A]->tuples);
                parts[B]->num_tuples -= (inB - parts[B]->tuples);

                parts[A]->tuples = inA;
                parts[B]->tuples = inB;

                done[m] = (nread == 0
                           || ((parts[A]->num_tuples == 0)
                               && (parts[B]->num_tuples == 0)));

                finished &= done[m];
            }
        }

        /* now iterate inner nodes and do merge for ready nodes */
        for(; m >= 0; m--) {
            if(!done[m]) {
                int r = 2*m+2;
                int l = r + 1;
                MergeNode<KeyType, PayloadType> * right = &nodes[r];
                MergeNode<KeyType, PayloadType> * left  = &nodes[l];

                uint8_t children_done = (done[r] | done[l]);

                if((children_done || nodes[m].count < readthreshold)
                   && nodes[m].count < fifosize) {
                    if(children_done || (right->count >= readthreshold
                                         && left->count >= readthreshold)) {

                        /* if(!check_node_sorted(right, fifosize)) */
                        /*     printf("Right Node not sorted\n"); */

                        /* if(!check_node_sorted(left, fifosize)) */
                        /*     printf("Left Node not sorted\n"); */

                        /* do a merge on right and left */
                        merge_scalar_modulo(&nodes[m], right, left, fifosize,
                                done[r], done[l]//children_done /* full-merge? */
                                        );

                        /* if(!check_node_sorted(right, fifosize)) */
                        /*     printf("After merge- Right Node not sorted\n"); */

                        /* if(!check_node_sorted(left, fifosize)) */
                        /*     printf("After merge- Left Node not sorted\n"); */

                        /* if(!check_node_sorted(&nodes[m], fifosize)) */
                        /*     printf("After merge - Node not sorted\n"); */
                    }
                    done[m] = (done[r] & done[l]) &&
                              (right->count == 0 && left->count == 0);
                }

                finished &= done[m];
            }
        }

        totalmerged +=
        /* finally iterate for the root node and store data */
        mergestore_scalar_bitand(&nodes[0], &nodes[1], &output, fifosize,
                                     done[0], done[1] /* full-merge? */
                                     );

    }

    return totalmerged;
}


/** This kernel takes two lists from ring buffers that can be linearly merged
    without a modulo operation on indices */
template<class KeyType, class PayloadType>    
inline void __attribute__((always_inline))
serialmergekernel(Tuple<KeyType, PayloadType> * restrict R, Tuple<KeyType, PayloadType> * restrict L, Tuple<KeyType, PayloadType> * restrict Out,
                  uint32_t * ri, uint32_t * li, uint32_t * oi, uint32_t * outnslots,
                  uint32_t rend, uint32_t lend)
{
    uint32_t rii = *ri, lii = *li, oii = *oi, nslots = *outnslots;

    while((nslots > 0 && rii < rend && lii < lend)){
        Tuple<KeyType, PayloadType> * in = L;
        uint32_t cmp = (R->key < L->key);
        uint32_t notcmp = !cmp;

        rii += cmp;
        lii += notcmp;

        if(cmp)
            in = R;

        nslots --;
        oii ++;
        *Out = *in;
        Out ++;
        R += cmp;
        L += notcmp;
    }

    *ri = rii;
    *li = lii;
    *oi = oii;
    *outnslots = nslots;
}


/**
 * Serially merge two merge node buffers to final output.
 *
 * @param R right merge node
 * @param L left merge node
 * @param[in,out] Out output array
 * @param[in,out] ri right node buffer index
 * @param[in,out] li left node buffer index
 * @param rend right node buffer end index
 * @param lend left node buffer end index
 */
template<class KeyType, class PayloadType>
inline void __attribute__((always_inline))
serialmergestorekernel(Tuple<KeyType, PayloadType> * restrict R, Tuple<KeyType, PayloadType> * restrict L,
                       Tuple<KeyType, PayloadType> ** Out,
                       uint32_t * ri, uint32_t * li,
                       uint32_t rend, uint32_t lend)
{
    uint32_t rii = *ri, lii = *li;
    Tuple<KeyType, PayloadType> * out = *Out;

    while(rii < rend && lii < lend){
        Tuple<KeyType, PayloadType> * in = L;
        uint32_t cmp = (R->key < L->key);
        uint32_t notcmp = !cmp;

        rii += cmp;
        lii += notcmp;

        if(cmp)
            in = R;

        *out = *in;
        out ++;
        R += cmp;
        L += notcmp;
    }

    *ri = rii;
    *li = lii;
    *Out = out;
}

/*******************************************************************************
 *         Scalar Multi-Way Merge with Ring-Buffer Decomposition               *
 *******************************************************************************/
template<class KeyType, class PayloadType>
uint32_t
readmerge_scalar_decomposed(MergeNode<KeyType, PayloadType> * node,
                            Tuple<KeyType, PayloadType> ** inA,
                            Tuple<KeyType, PayloadType> ** inB,
                            uint32_t lenA,
                            uint32_t lenB,
                            uint32_t fifosize)
{
    /* scalar optimized without modulo */
    uint32_t nodecount = node->count;
    uint32_t nodehead  = node->head;
    uint32_t nodetail  = node->tail;
    Tuple<KeyType, PayloadType> * Out = node->buffer;

    Tuple<KeyType, PayloadType> * A = *inA;
    Tuple<KeyType, PayloadType> * B = *inB;

    /* size related variables */
    uint32_t ri = 0, li = 0, outnslots;

    uint32_t oi = nodetail, oend;
    uint32_t oi2 = 0, oend2 = 0;

    if(nodehead > nodetail) {
        oend = nodehead;
    }
    else {
        oend = fifosize;
        oi2 = 0;
        oend2 = nodehead;
    }
    outnslots = oend - oi;

    Out += oi;

    /* fill first chunk of the node buffer */
    while( outnslots > 0 && ri < lenA && li < lenB ) {
        /* without branching, predication + cond movs. */
        Tuple<KeyType, PayloadType> * in = B;
        uint32_t cmp = (A->key < B->key);
        uint32_t notcmp = !cmp;
        ri += cmp;
        li += notcmp;

        if(cmp)
            in = A;

        outnslots --;
        oi ++;
        *Out = *in;
        Out ++;
        A += cmp;
        B += notcmp;
    }

    nodecount += (oi - nodetail);
    nodetail = ((oi == fifosize) ? 0 : oi);

    if(outnslots == 0 && oend2 != 0) {
        outnslots = oend2 - oi2;
        Out = node->buffer;

        /* fill second chunk of the node buffer */
        while( outnslots > 0 && ri < lenA && li < lenB ) {
            /* without branching, predication + cond movs. */
            Tuple<KeyType, PayloadType> * in = B;
            uint32_t cmp = (A->key < B->key);
            uint32_t notcmp = !cmp;
            ri += cmp;
            li += notcmp;

            if(cmp)
                in = A;

            outnslots --;
            oi2 ++;
            *Out = *in;
            Out ++;
            A += cmp;
            B += notcmp;

        }

        nodecount += oi2;
        nodetail = ((oi2 == fifosize) ? 0 : oi2);
    }

    if(nodecount < fifosize) {
        outnslots = fifosize - nodecount;
        oi = nodetail;
        oend = (nodetail + outnslots);
        if(oend > fifosize)
            oend = fifosize;
        outnslots = oend - oi;

        if(ri < lenA) {
            do{
                while( outnslots > 0 && ri < lenA ) {
                    outnslots --;
                    oi ++;
                    *Out = *A;
                    ri ++;
                    A++;
                    Out++;
                    nodecount ++;
                    nodetail ++;
                }

                if(oi == oend) {
                    oi = 0;
                    oend = nodehead;
                    if(nodetail >= fifosize){
                        nodetail = 0;
                        Out = node->buffer;
                    }
                    outnslots = oend - oi;
                }

            } while(nodecount < fifosize && ri < lenA);
        }
        else if(li < lenB) {
            do{
                while( outnslots > 0 && li < lenB ) {
                    outnslots --;
                    oi ++;
                    *Out = *B;
                    li ++;
                    B++;
                    Out++;
                    nodecount ++;
                    nodetail ++;
                }

                if(oi == oend) {
                    oi = 0;
                    oend = nodehead;
                    if(nodetail >= fifosize){
                        nodetail = 0;
                        Out = node->buffer;
                    }
                    outnslots = oend - oi;
                }
            } while(nodecount < fifosize && li < lenB);
        }
    }
    *inA = A;
    *inB = B;

    node->tail  = nodetail;
    node->count = nodecount;

    /* if(!check_node_sorted(node, fifosize)) */
    /*     printf("in merge_read() - Node not sorted\n"); */

    return (ri + li);
}

/**
 * Copy all tuples from src node to dest node.
 *
 * @param dest
 * @param src
 * @param fifosize
 */
template<class KeyType, class PayloadType>
void
direct_copy(MergeNode<KeyType, PayloadType> * dest, MergeNode<KeyType, PayloadType> * src, uint32_t fifosize)
{
    /* make sure dest has space and src has tuples */
    assert(dest->count < fifosize);
    assert(src->count > 0);

    /* Cases for the ring-buffer : 1) head < tail 2) head > tail */

    uint32_t dest_block_start[2];
    uint32_t dest_block_size[2];

    uint32_t src_block_start[2];
    uint32_t src_block_size[2];

    if(dest->head <= dest->tail){/* Case 1) */
        /* dest block-1 */
        dest_block_start[0] = dest->tail;
        dest_block_size[0] = fifosize - dest->tail;
        /* dest block-2 */
        dest_block_start[1] = 0;
        dest_block_size[1] = dest->head;
    }
    else {
        /* Case 2) dest-> head > dest->tail */
        /* dest block-1 */
        dest_block_start[0] = dest->tail;
        dest_block_size[0] = dest->head - dest->tail;
        /* no block-2 */
        dest_block_size[1] = 0;
    }

    if(src->head >= src->tail){/* Case 2) */
        /* src block-1 */
        src_block_start[0] = src->head;
        src_block_size[0] = fifosize - src->head;
        /* src block-2 */
        src_block_start[1] = 0;
        src_block_size[1] = src->tail;
    }
    else {
        /* Case 1) src-> head < src->tail */
        /* src block-1 */
        src_block_start[0] = src->head;
        src_block_size[0] = src->tail - src->head;
        /* no block-2 */
        src_block_size[1] = 0;
    }

    uint32_t copied = 0;
    for(int i = 0, j = 0; i < 2 && j < 2; ){
        uint32_t copysize = MIN(dest_block_size[i], src_block_size[j]);

        if(copysize > 0) {
            memcpy(dest->buffer + dest_block_start[i],
                    src->buffer + src_block_start[j],
                    copysize * sizeof(Tuple<KeyType, PayloadType>));

            dest_block_start[i] += copysize;
            src_block_start[j] += copysize;
            dest_block_size[i] -= copysize;
            src_block_size[j] -= copysize;
            copied += copysize;
        }

        if(dest_block_size[i] == 0)
            i++;

        if(src_block_size[j] == 0)
            j++;
    }

    dest->count += copied;
    dest->tail = (dest->tail + copied) % fifosize;
    src->count -= copied;
    src->head = (src->head + copied) % fifosize;

}

template<class KeyType, class PayloadType>
void
direct_copy_modulo(MergeNode<KeyType, PayloadType> * dest, MergeNode<KeyType, PayloadType> * src, uint32_t fifosize)
{
    while((dest->tail % fifosize) != (dest->head % fifosize)
          && (src->count > 0)){

        dest->buffer[(dest->tail % fifosize)] = src->buffer[(src->head % fifosize)];
        dest->tail = (dest->tail + 1) % fifosize;
        dest->count ++;
        src->head = (src->head + 1) % fifosize;
        src->count --;
    }
}

/**
 * Copy all tuples from src node to output array.
 *
 * @param dest pointer to tuple array for output
 * @param src
 * @param fifosize
 *
 * @return number of copied tuples
 */
template<class KeyType, class PayloadType>
uint32_t
direct_copy_to_output(Tuple<KeyType, PayloadType> * dest, MergeNode<KeyType, PayloadType> * src, uint32_t fifosize)
{
    /* make sure dest has space and src has tuples */
    //assert(src->count > 0);

    /* Cases for the ring-buffer : 1) head < tail 2) head > tail */
    uint32_t src_block_start[2];
    uint32_t src_block_size[2];

    if(src->head >= src->tail){/* Case 2) */
        /* src block-1 */
        src_block_start[0] = src->head;
        src_block_size[0] = fifosize - src->head;
        /* src block-2 */
        src_block_start[1] = 0;
        src_block_size[1] = src->tail;
    }
    else {
        /* Case 1) src-> head < src->tail */
        /* src block-1 */
        src_block_start[0] = src->head;
        src_block_size[0] = src->tail - src->head;
        /* no block-2 */
        src_block_size[1] = 0;
    }

    uint32_t copied = 0;
    for(int j = 0; j < 2; j++){
        uint32_t copysize = src_block_size[j];

        if(copysize > 0) {
            memcpy((void *)(dest + copied),
                    src->buffer + src_block_start[j],
                    copysize * sizeof(Tuple<KeyType, PayloadType>));

            copied += copysize;
        }
    }

    src->count -= copied;
    src->head = (src->head + copied) % fifosize;

    return copied;
}

template<class KeyType, class PayloadType>
void
merge_scalar_decomposed(MergeNode<KeyType, PayloadType> * node,
                        MergeNode<KeyType, PayloadType> * right,
                        MergeNode<KeyType, PayloadType> * left,
                        uint32_t fifosize, uint8_t rightdone, uint8_t leftdone)
{
    /* directly copy tuples from right or left if one of them done but not the other */
    if(rightdone && right->count == 0){
        if(!leftdone && left->count > 0){
            direct_copy(node, left, fifosize);
            return;
        }
    }
    else if(leftdone && left->count == 0){
        if(!rightdone && right->count > 0){
            direct_copy(node, right, fifosize);
            return;
        }
    }

    /* both done? */
    uint8_t done = rightdone & leftdone;

    uint32_t righttail  = right->tail;
    uint32_t rightcount = right->count;
    uint32_t righthead  = right->head;
    uint32_t lefttail   = left->tail;
    uint32_t leftcount  = left->count;
    uint32_t lefthead   = left->head;

    int rcases = 0, lcases = 0;
    uint32_t outnslots;

    uint32_t oi = node->tail, oend;
    if(node->head > node->tail) {
        oend = node->head;
    }
    else {
        oend = fifosize;
    }

    outnslots = oend - oi;

    uint32_t ri = righthead, rend;
    if(righthead >= righttail) {
        rend = fifosize;
        rcases = 1;
    }
    else {
        rend = righttail;
    }

    uint32_t li = lefthead, lend;
    if(lefthead >= lefttail) {
        lend = fifosize;
        lcases = 1;
    }
    else {
        lend = lefttail;
    }

    while(node->count < fifosize
          && (rightcount > 0 && leftcount > 0))
    {
        register Tuple<KeyType, PayloadType> * R = right->buffer + ri;
        register Tuple<KeyType, PayloadType> * L = left->buffer + li;
        register Tuple<KeyType, PayloadType> * Out = node->buffer + oi;

        serialmergekernel(R, L, Out, &ri, &li, &oi, &outnslots, rend, lend);

        node->count  += (oi - node->tail);
        node->tail = ((oi == fifosize) ? 0 : oi);
        rightcount -= (ri - righthead);
        righthead = ((ri == fifosize) ? 0 : ri);
        leftcount  -= (li - lefthead);
        lefthead = ((li == fifosize) ? 0 : li);


        if(oi == oend) {
            oi = 0;
            oend = node->head;
            outnslots = oend - oi;
        }

        if(rcases > 0 && ri == rend) {
            ri = 0;
            rend = righttail;
            rcases = 0;
        }

        if(lcases > 0 && li == lend) {
            li = 0;
            lend = lefttail;
            lcases = 0;
        }

    }

    /* not possible until we do not read new tuples anymore */
    if(done && node->count < fifosize) {
        Tuple<KeyType, PayloadType> * Out = node->buffer + node->tail;

        outnslots = fifosize - node->count;
        oi = node->tail;
        oend = (node->tail + outnslots);
        if(oend > fifosize)
            oend = fifosize;

        outnslots = oend - oi;

        if(rightcount > 0) {
            Tuple<KeyType, PayloadType> * R = right->buffer + righthead;

            ri = righthead;
            rend = righthead + rightcount;
            if(rend > fifosize)
                rend = fifosize;

            do{
                while( outnslots > 0 && ri < rend) {
                    outnslots --;
                    oi++;
                    ri ++;
                    *Out = *R;
                    Out ++;
                    R ++;
                    node->count ++;
                    rightcount --;
                    node->tail ++;
                    righthead ++;
                }

                /* node->count  += (oi - node->tail); */
                /* node->tail = ((oi == fifosize) ? 0 : oi); */
                /* rightcount -= (ri - righthead); */
                /* righthead = ((ri == fifosize) ? 0 : ri); */

                if(oi == oend) {
                    oi = 0;
                    oend = node->head;
                    if(node->tail >= fifosize){
                        node->tail = 0;
                        Out = node->buffer;
                    }
                }

                if(rcases > 0 && ri == rend) {
                    ri = 0;
                    rend = righttail;
                    rcases = 0;
                    if(righthead >= fifosize){
                        righthead = 0;
                        R = right->buffer;
                    }
                }
            } while(outnslots > 0 && rightcount > 0);

        }
        else if(leftcount > 0) {
            Tuple<KeyType, PayloadType> * L = left->buffer + lefthead;

            li = lefthead;
            lend = lefthead + leftcount;
            if(lend > fifosize)
                lend = fifosize;

            do {
                while( outnslots > 0 && li < lend) {
                    outnslots --;
                    oi++;
                    li ++;
                    *Out = *L;
                    Out ++;
                    L ++;
                    node->count ++;
                    leftcount --;
                    node->tail ++;
                    lefthead ++;
                }

                /* node->count  += (oi - node->tail); */
                /* node->tail = ((oi == fifosize) ? 0 : oi); */
                /* leftcount -= (li - lefthead); */
                /* lefthead = ((li == fifosize) ? 0 : li); */

                if(oi == oend) {
                    oi = 0;
                    oend = node->head;
                    if(node->tail >= fifosize) {
                        node->tail = 0;
                        Out = node->buffer;
                    }
                }


                if(lcases > 0 && li == lend) {
                    li = 0;
                    lend = lefttail;
                    lcases = 0;
                    if(lefthead >= fifosize) {
                        lefthead = 0;
                        L = left->buffer;
                    }
                }

            } while(outnslots > 0 && leftcount > 0);
        }
    }

    /* if(!check_node_sorted(node, fifosize)) */
    /*     printf("Node not sorted rsz=%d, lsz=%d\n", rsz, lsz); */


    right->count = rightcount;
    right->head  = righthead;
    left->count  = leftcount;
    left->head   = lefthead;

}

template<class KeyType, class PayloadType>
uint64_t
mergestore_scalar_decomposed(MergeNode<KeyType, PayloadType> * right,
                             MergeNode<KeyType, PayloadType> * left,
                             Tuple<KeyType, PayloadType> ** output,
                             uint32_t fifosize,
                             uint8_t rightdone, uint8_t leftdone)
{
    /* directly copy tuples from right or left if one of them done but not the other */
    if(rightdone && right->count == 0){
        if(!leftdone && left->count > 0){
            uint64_t numcopied = direct_copy_to_output(*output, left, fifosize);
            /*
            if(is_sorted_helper((int64_t*)(*output), numcopied) == 0){
                printf("[ERROR] 1.\n");
            }
            */
            *output += numcopied;
            return numcopied;
        }
    }
    else if(leftdone && left->count == 0){
        if(!rightdone && right->count > 0){
            uint64_t numcopied = direct_copy_to_output(*output, right, fifosize);
            /*
            if(is_sorted_helper((int64_t*)(*output), numcopied) == 0){
                printf("[ERROR] 2.\n");
            }
            */
            *output += numcopied;
            return numcopied;
        }
    }

    Tuple<KeyType, PayloadType> * Out = * output;
    int rcases = 0, lcases = 0;

    uint32_t ri = right->head, rend;
    if(right->head >= right->tail) {
        rend = fifosize;
        rcases = 1;
    }
    else {
        rend = right->tail;
    }

    uint32_t li = left->head, lend;
    if(left->head >= left->tail) {
        lend = fifosize;
        lcases = 1;
    }
    else {
        lend = left->tail;
    }

    while(right->count > 0 && left->count > 0) {

        register Tuple<KeyType, PayloadType> * R = right->buffer + ri;
        register Tuple<KeyType, PayloadType> * L = left->buffer + li;

        serialmergestorekernel(R, L, &Out, &ri, &li, rend, lend);

        right->count -= (ri - right->head);
        right->head = ((ri == fifosize) ? 0 : ri);
        left->count  -= (li - left->head);
        left->head = ((li == fifosize) ? 0 : li);

        if(rcases > 0 && ri == rend) {
            ri = 0;
            rend = right->tail;
            rcases = 0;
        }

        if(lcases > 0 && li == lend) {
            li = 0;
            lend = left->tail;
            lcases = 0;
        }
    }

    /* not possible until we do not read new tuples anymore */
    uint8_t done = rightdone & leftdone;
    if(done){
        if(right->count > 0) {
            Tuple<KeyType, PayloadType> * R = right->buffer + right->head;

            ri = right->head;
            rend = right->head + right->count;
            if(rend > fifosize)
                rend = fifosize;

            do{
                while( ri < rend ) {
                    ri ++;
                    *Out = *R;
                    Out ++;
                    R ++;
                    right->count --;
                    right->head ++;
                }

                if(rcases > 0 && ri == rend) {
                    ri = 0;
                    rend = right->tail;
                    rcases = 0;
                    if(right->head >= fifosize){
                        right->head = 0;
                        R = right->buffer;
                    }
                }
            } while(right->count > 0);

        }
        else if(left->count > 0) {
            Tuple<KeyType, PayloadType> * L = left->buffer + left->head;

            li = left->head;
            lend = left->head + left->count;
            if(lend > fifosize)
                lend = fifosize;

            do {
                while( li < lend ) {
                    li ++;
                    *Out = *L;
                    Out ++;
                    L ++;
                    left->count --;
                    left->head ++;
                }

                if(lcases > 0 && li == lend) {
                    li = 0;
                    lend = left->tail;
                    lcases = 0;
                    if(left->head >= fifosize) {
                        left->head = 0;
                        L = left->buffer;
                    }
                }

            } while(left->count > 0);
        }
    }

    uint64_t numstored = (Out - *output);
    *output = Out;

    return numstored;
}

/*******************************************************************************
 *         Scalar Multi-Way Merge with Ring-Buffer Modulo                      *
 *******************************************************************************/
template<class KeyType, class PayloadType>
uint32_t
readmerge_scalar_modulo(MergeNode<KeyType, PayloadType> * node,
                        Tuple<KeyType, PayloadType> ** inA,
                        Tuple<KeyType, PayloadType> ** inB,
                        uint32_t lenA,
                        uint32_t lenB,
                        uint32_t fifosize)
{
    /* scalar un-optimized with modulo op */
    Tuple<KeyType, PayloadType> * A = *inA;
    Tuple<KeyType, PayloadType> * B = *inB;
    Tuple<KeyType, PayloadType> * Out = node->buffer;

    /* size related variables */
    uint32_t ri = 0, li = 0, outnslots = (fifosize - node->count);

    while( outnslots > 0 && ri < lenA && li < lenB ) {

        if(A->key < B->key) {
            Out[node->tail] = *A;
            ri ++;
            A++;
        }
        else {
            Out[node->tail] = *B;
            li ++;
            B++;
        }

        outnslots --;
        node->tail = (node->tail + 1) % fifosize;
    }

    while( outnslots > 0 && ri < lenA ) {
        Out[node->tail] = *A;
        ri ++;
        A++;
        outnslots --;
        node->tail = (node->tail + 1) % fifosize;
    }

    while( outnslots > 0 && li < lenB ) {
        Out[node->tail] = *B;
        li ++;
        B++;
        outnslots --;
        node->tail = (node->tail + 1) % fifosize;
    }

    *inA = A;
    *inB = B;
    node->count  = fifosize - outnslots;

    return (ri + li);
}

template<class KeyType, class PayloadType>
void
merge_scalar_modulo(MergeNode<KeyType, PayloadType> * node,
                    MergeNode<KeyType, PayloadType> * right,
                    MergeNode<KeyType, PayloadType> * left,
                    uint32_t fifosize,
                    uint8_t rightdone, uint8_t leftdone)
{
    /* directly copy tuples from right or left if one of them done but not the other */
    if(rightdone && right->count == 0){
        if(!leftdone && left->count > 0){
            direct_copy(node, left, fifosize);
            return;
        }
    }
    else if(leftdone && left->count == 0){
        if(!rightdone && right->count > 0){
            direct_copy(node, right, fifosize);
            return;
        }
    }
    /* both done? */
    uint8_t done = rightdone & leftdone;

    /* first try with scalar merge */
    Tuple<KeyType, PayloadType> * R = right->buffer;
    Tuple<KeyType, PayloadType> * L = left->buffer;
    Tuple<KeyType, PayloadType> * Out = node->buffer;
    const uint32_t Rsz = right->count;
    const uint32_t Lsz = left->count;

    /* size related variables */
    uint32_t ri = 0, li = 0, outnslots = (fifosize - node->count);

    while( outnslots > 0 && ri < Rsz && li < Lsz ) {
        Tuple<KeyType, PayloadType> * A = R + right->head;
        Tuple<KeyType, PayloadType> * B = L + left->head;

        if(A->key < B->key) {
            Out[node->tail] = *A;
            ri ++;
            right->head = (right->head + 1) % fifosize;
        }
        else {
            Out[node->tail] = *B;
            li ++;
            left->head = (left->head + 1) % fifosize;
        }

        outnslots --;
        node->tail = (node->tail + 1) % fifosize;
    }

    if(done) {/* not possible until we do not read new tuples anymore */
        while( outnslots > 0 && ri < Rsz ) {
            Out[node->tail] = R[right->head];
            ri ++;
            right->head = (right->head + 1) % fifosize;
            outnslots --;
            node->tail = (node->tail + 1) % fifosize;
        }

        while( outnslots > 0 && li < Lsz ) {
            Out[node->tail] = L[left->head];
            li ++;
            left->head = (left->head + 1) % fifosize;
            outnslots --;
            node->tail = (node->tail + 1) % fifosize;
        }
    }

    /* assert(fifosize>=outnslots); */
    /* assert(Rsz>=ri); */
    /* assert(Lsz>=li); */

    node->count  = fifosize - outnslots;
    right->count = Rsz - ri;
    left->count  = Lsz - li;

}

template<class KeyType, class PayloadType>
uint64_t
mergestore_scalar_modulo(MergeNode<KeyType, PayloadType> * right,
                         MergeNode<KeyType, PayloadType> * left,
                         Tuple<KeyType, PayloadType> ** output,
                         uint32_t fifosize,
                         uint8_t rightdone, uint8_t leftdone)
{
    /* directly copy tuples from right or left if one of them done but not the other */
    if(rightdone && right->count == 0){
        if(!leftdone && left->count > 0){
            uint64_t numcopied = direct_copy_to_output(*output, left, fifosize);
            /*
            if(is_sorted_helper((int64_t*)(*output), numcopied) == 0){
                printf("[ERROR] 1.\n");
            }
            */
            *output += numcopied;
            return numcopied;
        }
    }
    else if(leftdone && left->count == 0){
        if(!rightdone && right->count > 0){
            uint64_t numcopied = direct_copy_to_output(*output, right, fifosize);
            /*
            if(is_sorted_helper((int64_t*)(*output), numcopied) == 0){
                printf("[ERROR] 2.\n");
            }
            */
            *output += numcopied;
            return numcopied;
        }
    }

    Tuple<KeyType, PayloadType> * R = right->buffer;
    Tuple<KeyType, PayloadType> * L = left->buffer;
    Tuple<KeyType, PayloadType> * Out = * output;
    const uint32_t Rsz = right->count;
    const uint32_t Lsz = left->count;

    /* size related variables */
    uint32_t ri = 0, li = 0;

    while( ri < Rsz && li < Lsz ) {
        Tuple<KeyType, PayloadType> * A = R + right->head;
        Tuple<KeyType, PayloadType> * B = L + left->head;

        if(A->key < B->key) {
            *Out = *A;
            ri ++;
            right->head = (right->head + 1) % fifosize;
        }
        else {
            *Out = *B;
            li ++;
            left->head = (left->head + 1) % fifosize;
        }

        Out ++;
    }

    uint8_t done = rightdone & leftdone;
    if(done){/* not possible until we do not read new tuples anymore */
        while( ri < Rsz ) {
            *Out = R[right->head];
            Out ++;
            ri ++;
            right->head = (right->head + 1) % fifosize;
        }

        while( li < Lsz ) {
            *Out = L[left->head];
            Out ++;
            li ++;
            left->head = (left->head + 1) % fifosize;
        }
    }

    /* assert(Rsz >= ri); */
    /* assert(Lsz >= li); */

    right->count = Rsz - ri;
    left->count  = Lsz - li;

#if 0
    if(is_sorted_helper((int64_t *) *output, ri+li)){
        static uint32_t written = 0;

        written += ri+li;
        printf("[INFO ] merge_store() successful TOTAL=%d, Wr=%d, R=%d, S=%d\n",
               written, ri+li, Rsz, Lsz);
    }
    else {
        printf("[ERROR] merge_store() failed!\n");
        exit(0);
    }
#endif

    uint64_t numstored = (Out - *output);
    *output = Out;

    return numstored;
}

/*******************************************************************************
 *         Scalar Multi-Way Merge with Ring-Buffer Bit-AND                     *
 *******************************************************************************/
template<class KeyType, class PayloadType>
uint32_t
readmerge_scalar_bitand(MergeNode<KeyType, PayloadType> * node,
                        Tuple<KeyType, PayloadType> ** inA,
                        Tuple<KeyType, PayloadType> ** inB,
                        uint32_t lenA,
                        uint32_t lenB,
                        uint32_t fifosize)
{
    /* first try with scalar merge */
    Tuple<KeyType, PayloadType> * restrict A = *inA;
    Tuple<KeyType, PayloadType> * restrict B = *inB;
    Tuple<KeyType, PayloadType> * restrict Out = node->buffer;

    /* size related variables */
    uint32_t ri = 0, li = 0, outnslots = (fifosize - node->count);

    uint32_t nt = node->tail;
    uint32_t fifosize_mask = fifosize-1;

    while( outnslots > 0 && ri < lenA && li < lenB ) {

        KeyType ak = A->key;
        KeyType bk = B->key;
        if(ak < bk) {
            Out[nt] = *A;
            ri ++;
            A++;
        }
        else {
            Out[nt] = *B;
            li ++;
            B++;
        }

        outnslots --;
        nt = (nt + 1) & fifosize_mask;
    }

    while( outnslots > 0 && ri < lenA ) {
        Out[nt] = *A;
        ri ++;
        A++;
        outnslots --;
        nt = (nt + 1) & fifosize_mask;
    }

    while( outnslots > 0 && li < lenB ) {
        Out[nt] = *B;
        li ++;
        B++;
        outnslots --;
        nt = (nt + 1) & fifosize_mask;
    }

    *inA = A;
    *inB = B;
    node->count  = fifosize - outnslots;
    node->tail = nt;

    return (ri + li);
}

template<class KeyType, class PayloadType>
void
merge_scalar_bitand(MergeNode<KeyType, PayloadType> * node,
                    MergeNode<KeyType, PayloadType> * right,
                    MergeNode<KeyType, PayloadType> * left,
                    uint32_t fifosize,
                    uint8_t rightdone, uint8_t leftdone)
{
    /* directly copy tuples from right or left if one of them done but not the other */
    if(rightdone && right->count == 0){
        if(!leftdone && left->count > 0){
            direct_copy(node, left, fifosize);
            return;
        }
    }
    else if(leftdone && left->count == 0){
        if(!rightdone && right->count > 0){
            direct_copy(node, right, fifosize);
            return;
        }
    }
    /* both done? */
    uint8_t done = rightdone & leftdone;

    /* first try with scalar merge */
    Tuple<KeyType, PayloadType> * R = right->buffer;
    Tuple<KeyType, PayloadType> * L = left->buffer;
    Tuple<KeyType, PayloadType> * Out = node->buffer;
    const uint32_t Rsz = right->count;
    const uint32_t Lsz = left->count;

    /* size related variables */
    uint32_t ri = 0, li = 0, outnslots = (fifosize - node->count);
    uint32_t fifosize_mask = fifosize-1;

    while( outnslots > 0 && ri < Rsz && li < Lsz ) {
        Tuple<KeyType, PayloadType> * A = R + right->head;
        Tuple<KeyType, PayloadType> * B = L + left->head;

        if(A->key < B->key) {
            Out[node->tail] = *A;
            ri ++;
            right->head = (right->head + 1) & fifosize_mask;
        }
        else {
            Out[node->tail] = *B;
            li ++;
            left->head = (left->head + 1) & fifosize_mask;
        }

        outnslots --;
        node->tail = (node->tail + 1) & fifosize_mask;
    }

    if(done) {/* not possible until we do not read new tuples anymore */
        while( outnslots > 0 && ri < Rsz ) {
            Out[node->tail] = R[right->head];
            ri ++;
            right->head = (right->head + 1) & fifosize_mask;
            outnslots --;
            node->tail = (node->tail + 1) & fifosize_mask;
        }

        while( outnslots > 0 && li < Lsz ) {
            Out[node->tail] = L[left->head];
            li ++;
            left->head = (left->head + 1) & fifosize_mask;
            outnslots --;
            node->tail = (node->tail + 1) & fifosize_mask;
        }
    }

    /* assert(fifosize>=outnslots); */
    /* assert(Rsz>=ri); */
    /* assert(Lsz>=li); */

    node->count  = fifosize - outnslots;
    right->count = Rsz - ri;
    left->count  = Lsz - li;

}

template<class KeyType, class PayloadType>
uint64_t
mergestore_scalar_bitand(MergeNode<KeyType, PayloadType> * right,
                         MergeNode<KeyType, PayloadType> * left,
                         Tuple<KeyType, PayloadType> ** output,
                         uint32_t fifosize,
                         uint8_t rightdone, uint8_t leftdone)
{
    /* directly copy tuples from right or left if one of them done but not the other */
    if(rightdone && right->count == 0){
        if(!leftdone && left->count > 0){
            uint64_t numcopied = direct_copy_to_output(*output, left, fifosize);
            /*
            if(is_sorted_helper((int64_t*)(*output), numcopied) == 0){
                printf("[ERROR] 1.\n");
            }
            */
            *output += numcopied;
            return numcopied;
        }
    }
    else if(leftdone && left->count == 0){
        if(!rightdone && right->count > 0){
            uint64_t numcopied = direct_copy_to_output(*output, right, fifosize);
            /*
            if(is_sorted_helper((int64_t*)(*output), numcopied) == 0){
                printf("[ERROR] 2.\n");
            }
            */
            *output += numcopied;
            return numcopied;
        }
    }

    Tuple<KeyType, PayloadType> * R = right->buffer;
    Tuple<KeyType, PayloadType> * L = left->buffer;
    Tuple<KeyType, PayloadType> * Out = * output;
    const uint32_t Rsz = right->count;
    const uint32_t Lsz = left->count;
    uint32_t fifosize_mask = fifosize-1;

    /* size related variables */
    uint32_t ri = 0, li = 0;

    while( ri < Rsz && li < Lsz ) {
        Tuple<KeyType, PayloadType> * A = R + right->head;
        Tuple<KeyType, PayloadType> * B = L + left->head;

        if(A->key < B->key) {
            *Out = *A;
            ri ++;
            right->head = (right->head + 1) & fifosize_mask;
        }
        else {
            *Out = *B;
            li ++;
            left->head = (left->head + 1) & fifosize_mask;
        }

        Out ++;
    }

    uint8_t done = rightdone & leftdone;
    if(done){/* not possible until we do not read new tuples anymore */
        while( ri < Rsz ) {
            *Out = R[right->head];
            Out ++;
            ri ++;
            right->head = (right->head + 1) & fifosize_mask;
        }

        while( li < Lsz ) {
            *Out = L[left->head];
            Out ++;
            li ++;
            left->head = (left->head + 1) & fifosize_mask;
        }
    }

    /* assert(Rsz >= ri); */
    /* assert(Lsz >= li); */

    right->count = Rsz - ri;
    left->count  = Lsz - li;

    uint64_t numstored = (Out - *output);
    *output = Out;

    return numstored;
}

#endif /* SCALARMULTIWAYMERGE_H */
