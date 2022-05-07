#pragma once

/* Basic data structures used among all joins */

#include <stdint.h>
#include <pthread.h>
#include <stdlib.h>

#include "configs/base_configs.h"
#include "utils/base_utils.h"
#include "utils/perf_event.h"

using namespace std;

/******* Input/Output Types **************/
/*****************************************/
template<class KeyType, class PayloadType>
struct Tuple {
    KeyType key;
    PayloadType  payload;
};

template<class KeyType, class PayloadType>
struct Relation {
  Tuple<KeyType, PayloadType> * tuples;
  uint64_t  num_tuples;
};

template<class KeyType, class PayloadType>
struct RelationPair {
    Relation<KeyType, PayloadType> R;
    Relation<KeyType, PayloadType> S;
};

struct ThreadResult {
    uint64_t  nresults;
    void *   results;
    uint32_t threadid;
};

struct Result {
    uint64_t         totalresults;
    ThreadResult *   resultlist;
    int              nthreads;
};
/**************************************/

/******* Statsitics **************/
/********************************/

struct TimingStats 
{
  double total_partitioning_time_usec = 0.0;
  double total_joining_time_usec = 0.0;
  double total_algorithm_time_usec = 0.0;
};

struct PerfEventStats 
{
  double total_partitioning_cycles = 0.0;
  double total_partitioning_instructions = 0.0;
  double total_partitioning_l1_misses = 0.0;
  double total_partitioning_llc_misses = 0.0;
  double total_partitioning_branch_misses = 0.0;
  double total_partitioning_task_clock = 0.0;
  double total_partitioning_instructions_per_cycle = 0.0;
  double total_partitioning_cpus = 0.0;
  double total_partitioning_ghz = 0.0;

  double total_joining_cycles = 0.0;
  double total_joining_instructions = 0.0;
  double total_joining_l1_misses = 0.0;
  double total_joining_llc_misses = 0.0;
  double total_joining_branch_misses = 0.0;
  double total_joining_task_clock = 0.0;
  double total_joining_instructions_per_cycle = 0.0;
  double total_joining_cpus = 0.0;
  double total_joining_ghz = 0.0;
};

struct SortMergeJoinTimingStats : TimingStats
{
  double total_sorting_time_usec = 0.0;
  double total_merging_time_usec = 0.0;
};

struct SortMergeJoinPerfEventStats : PerfEventStats
{
  double total_sorting_cycles = 0.0;
  double total_sorting_instructions = 0.0;
  double total_sorting_l1_misses = 0.0;
  double total_sorting_llc_misses = 0.0;
  double total_sorting_branch_misses = 0.0;
  double total_sorting_task_clock = 0.0;
  double total_sorting_instructions_per_cycle = 0.0;
  double total_sorting_cpus = 0.0;
  double total_sorting_ghz = 0.0;

  double total_merging_cycles = 0.0;
  double total_merging_instructions = 0.0;
  double total_merging_l1_misses = 0.0;
  double total_merging_llc_misses = 0.0;
  double total_merging_branch_misses = 0.0;
  double total_merging_task_clock = 0.0;
  double total_merging_instructions_per_cycle = 0.0;
  double total_merging_cpus = 0.0;
  double total_merging_ghz = 0.0;
};

struct LearnedSortMergeJoinTimingStats : SortMergeJoinTimingStats
{
  double total_sampling_time_usec = 0.0;
};

struct LearnedSortMergeJoinPerfEventStats : SortMergeJoinPerfEventStats
{
  double total_sampling_cycles = 0.0;
  double total_sampling_instructions = 0.0;
  double total_sampling_l1_misses = 0.0;
  double total_sampling_llc_misses = 0.0;
  double total_sampling_branch_misses = 0.0;
  double total_sampling_task_clock = 0.0;
  double total_sampling_instructions_per_cycle = 0.0;
  double total_sampling_cpus = 0.0;
  double total_sampling_ghz = 0.0;
};

/**************************************/

/******* TaskQueue **************/
/********************************/
/* Taken and adapted from ETH implementation */
template<class KeyType, class PayloadType>
struct Task {
    Relation<KeyType, PayloadType> relR;
    Relation<KeyType, PayloadType> tmpR;    
    Relation<KeyType, PayloadType> relS;
    Relation<KeyType, PayloadType> tmpS;
    Task<KeyType, PayloadType> *   next;
};

template<typename KeyType, typename PayloadType, typename TaskType>
struct TaskList {
    TaskType *      tasks;
    TaskList<KeyType, PayloadType, TaskType> *  next;
    int           curr;
};

template<typename KeyType, typename PayloadType, typename TaskType>
struct TaskQueue {
    pthread_mutex_t lock;
    pthread_mutex_t alloc_lock;
    TaskType *        head;
    TaskList<KeyType, PayloadType, TaskType> *   free_list;
    int32_t         count;
    int32_t         alloc_size;
};

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
TaskType * 
get_next_task(TaskQueue<KeyType, PayloadType, TaskType> * tq) __attribute__((always_inline));

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
void 
add_tasks(TaskQueue<KeyType, PayloadType, TaskType> * tq, TaskType * t) __attribute__((always_inline));

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
TaskType * 
get_next_task(TaskQueue<KeyType, PayloadType, TaskType> * tq) 
{
    pthread_mutex_lock(&tq->lock);
    TaskType * ret = 0;
    if(tq->count > 0){
        ret = tq->head;
        tq->head = ret->next;
        tq->count --;
    }
    pthread_mutex_unlock(&tq->lock);

    return ret;
}

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
void 
add_tasks(TaskQueue<KeyType, PayloadType, TaskType> * tq, TaskType * t) 
{
    pthread_mutex_lock(&tq->lock);
    t->next = tq->head;
    tq->head = t;
    tq->count ++;
    pthread_mutex_unlock(&tq->lock);
}

// atomically get the next available task
template<typename KeyType, typename PayloadType, typename TaskType>
inline 
TaskType * 
task_queue_get_atomic(TaskQueue<KeyType, PayloadType, TaskType> * tq) __attribute__((always_inline));

// atomically add a task 
template<typename KeyType, typename PayloadType, typename TaskType>
inline 
void 
task_queue_add_atomic(TaskQueue<KeyType, PayloadType, TaskType> * tq, TaskType * t) 
    __attribute__((always_inline));

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
void 
task_queue_add(TaskQueue<KeyType, PayloadType, TaskType> * tq, TaskType * t) __attribute__((always_inline));

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
void 
task_queue_copy_atomic(TaskQueue<KeyType, PayloadType, TaskType> * tq, TaskType * t)
    __attribute__((always_inline));

// get a free slot of task_t 
template<typename KeyType, typename PayloadType, typename TaskType>
inline 
TaskType * 
task_queue_get_slot_atomic(TaskQueue<KeyType, PayloadType, TaskType> * tq) __attribute__((always_inline));

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
TaskType * 
task_queue_get_slot(TaskQueue<KeyType, PayloadType, TaskType> * tq) __attribute__((always_inline));

// initialize a task queue with given allocation block size
template<typename KeyType, typename PayloadType, typename TaskType>
TaskQueue<KeyType, PayloadType, TaskType> * 
task_queue_init(int alloc_size);

template<typename KeyType, typename PayloadType, typename TaskType>
void 
task_queue_free(TaskQueue<KeyType, PayloadType, TaskType> * tq);

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
TaskType * 
task_queue_get_atomic(TaskQueue<KeyType, PayloadType, TaskType> * tq) 
{
    pthread_mutex_lock(&tq->lock);
    TaskType * ret = 0;
    if(tq->count > 0){
        ret      = tq->head;
        tq->head = ret->next;
        tq->count --;
    }
    pthread_mutex_unlock(&tq->lock);

    return ret;
}

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
void 
task_queue_add_atomic(TaskQueue<KeyType, PayloadType, TaskType> * tq, TaskType * t) 
{
    pthread_mutex_lock(&tq->lock);
    t->next  = tq->head;
    tq->head = t;
    tq->count ++;
    pthread_mutex_unlock(&tq->lock);

}

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
void 
task_queue_add(TaskQueue<KeyType, PayloadType, TaskType> * tq, TaskType * t) 
{
    t->next  = tq->head;
    tq->head = t;
    tq->count ++;
}

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
void 
task_queue_copy_atomic(TaskQueue<KeyType, PayloadType, TaskType> * tq, TaskType * t) 
{
    pthread_mutex_lock(&tq->lock);
    TaskType * slot = task_queue_get_slot(tq);
    *slot = *t; // copy 
    task_queue_add(tq, slot);
    pthread_mutex_unlock(&tq->lock);
}

template<typename KeyType, typename PayloadType, typename TaskType>
inline 
TaskType * 
task_queue_get_slot(TaskQueue<KeyType, PayloadType, TaskType> * tq)
{
    TaskList<KeyType, PayloadType, TaskType> * l = tq->free_list;
    TaskType * ret;

    if(l->curr < tq->alloc_size) {
        ret = &(l->tasks[l->curr]);
        l->curr++;
    }
    else {
        TaskList<KeyType, PayloadType, TaskType> * nl = (TaskList<KeyType, PayloadType, TaskType>*) malloc(sizeof(TaskList<KeyType, PayloadType, TaskType>));
        nl->tasks = (TaskType *) malloc(tq->alloc_size * sizeof(TaskType));
        nl->curr = 1;
        nl->next = tq->free_list;
        tq->free_list = nl;
        ret = &(nl->tasks[0]);
    }

    return ret;
}

// get a free slot of task_t 
template<typename KeyType, typename PayloadType, typename TaskType>
inline 
TaskType * 
task_queue_get_slot_atomic(TaskQueue<KeyType, PayloadType, TaskType> * tq)
{
    pthread_mutex_lock(&tq->alloc_lock);
    TaskType * ret = task_queue_get_slot(tq);
    pthread_mutex_unlock(&tq->alloc_lock);

    return ret;
}

// initialize a task queue with given allocation block size 
template<typename KeyType, typename PayloadType, typename TaskType>
TaskQueue<KeyType, PayloadType, TaskType> * 
task_queue_init(int alloc_size) 
{
    TaskQueue<KeyType, PayloadType, TaskType> * ret = (TaskQueue<KeyType, PayloadType, TaskType>*) malloc(sizeof(TaskQueue<KeyType, PayloadType, TaskType>));
    ret->free_list = (TaskList<KeyType, PayloadType, TaskType>*) malloc(sizeof(TaskList<KeyType, PayloadType, TaskType>));
    ret->free_list->tasks = (TaskType *) malloc(alloc_size * sizeof(TaskType));
    ret->free_list->curr = 0;
    ret->free_list->next = NULL;
    ret->count      = 0;
    ret->alloc_size = alloc_size;
    ret->head       = NULL;
    pthread_mutex_init(&ret->lock, NULL);
    pthread_mutex_init(&ret->alloc_lock, NULL);

    return ret;
}

template<typename KeyType, typename PayloadType, typename TaskType>
void 
task_queue_free(TaskQueue<KeyType, PayloadType, TaskType> * tq) 
{
    TaskList<KeyType, PayloadType, TaskType> * tmp = tq->free_list;
    while(tmp) {
        free(tmp->tasks);
        TaskList<KeyType, PayloadType, TaskType> * tmp2 = tmp->next;
        free(tmp);
        tmp = tmp2;
    }
    free(tq);
}

/********************************/

/******** Processing Types ************/
/**************************************/

template<typename KeyType, typename PayloadType, typename TaskType>
struct JoinThreadBase {
    Tuple<KeyType, PayloadType> *  relR;
    Tuple<KeyType, PayloadType> *  relS;
    TaskQueue<KeyType, PayloadType, TaskType> **  part_queue;
    TaskQueue<KeyType, PayloadType, TaskType> **  join_queue;
    pthread_barrier_t * barrier;
    ThreadResult * threadresult;

    struct timeval start_time, partition_end_time, end_time;
#ifndef DEVELOPMENT_MODE
    //PerfEvent e_start_to_partition, e_partition_to_end;
#endif     
    int64_t totalR;
    int64_t totalS;
    int64_t result;
    int32_t my_tid;
    int     nthreads;
    int32_t numR;
    int32_t numS;
};

template<typename KeyType, typename PayloadType, typename TaskType, typename JoinThreadType>
struct PartitionBase {
    Tuple<KeyType, PayloadType> *  rel;
    JoinThreadType *  thrargs;
    uint64_t   total_tuples;
    uint32_t   num_tuples;
    int        relidx;  /* 0: R, 1: S */
};

template<typename KeyType, typename PayloadType, typename TaskType, typename JoinThreadType>
struct EmptyParition {};

template<typename KeyType, typename PayloadType>
struct EmptyTask {};

struct BuildBase {
    uint32_t *  hist;
};

template<class KeyType, class PayloadType>
union CacheLine {
    struct {
        Tuple<KeyType, PayloadType> tuples[CACHE_LINE_SIZE/sizeof(Tuple<KeyType, PayloadType>)];
    } tuples;
    struct {
        Tuple<KeyType, PayloadType> tuples[CACHE_LINE_SIZE/sizeof(Tuple<KeyType, PayloadType>) - 1];
        int64_t slot;
    } data;
}; 

/*************************************/

