/**
 * @file    cpu_mapping.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Tue May 22 16:35:12 2012
 * @version $Id: cpu_mapping.h 4548 2013-12-07 16:05:16Z bcagri $
 * 
 * @brief  Provides cpu mapping utility function. 
 * 
 */

#ifndef CPU_MAPPING_H
#define CPU_MAPPING_H

#include <stdio.h>  /* FILE, fopen */
#include <stdlib.h> /* exit, perror */
#include <unistd.h> /* sysconf */
#include <assert.h> /* assert() */

#include <numaif.h> /* get_mempolicy() */
#ifdef HAVE_LIBNUMA
#include <numa.h>   /* for automatic NUMA-mappings */
#endif

#include "configs/base_configs.h"

// Returns SMT aware logical to physical CPU mapping for a given thread id. 
int get_cpu_id(int thread_id);
 
// Returns the NUMA id of the given thread id returned from get_cpu_id(int)
int get_numa_id(int mytid);

// Returns number of NUMA regions.
int get_num_numa_regions(void);

// Returns the NUMA-node id of a given memory address
int  get_numa_node_of_address(void * ptr);
 
// Initialize cpu mappings and NUMA topology (if libNUMA available).
// Try custom cpu mapping file first, if does not exist then round-robin
// initialization among available CPUs reported by the system. 
void cpu_mapping_init();

// De-initialize/free cpu mapping data structures.
void cpu_mapping_cleanup();

// Returns SMT aware logical to physical CPU mapping for a given logical thread id.
int get_cpu_id_v2(int thread_id);

// Returns whether given logical thread id is the first thread in its NUMA region.
int is_first_thread_in_numa_region(int logicaltid);
 
// Returns the index of the given logical thread within its NUMA-region.
int get_thread_index_in_numa(int logicaltid);

// Returns the NUMA-region id of the given logical thread id.
int get_numa_region_id(int logicaltid);
 
// Returns number of NUMA regions.
int get_num_numa_regions_v2(void);

// Set the given thread by physical thread-id (i.e. returned from get_cpu_id())
// as active in its NUMA-region.
void numa_thread_mark_active(int phytid);

// Return the active number of threads in the given NUMA-region.
int get_num_active_threads_in_numa(int numaregionid);

// Return the linearized index offset of the thread in the NUMA-topology mapping.
int get_numa_index_of_logical_thread(int logicaltid);

// Return the logical thread id from the linearized the NUMA-topology mapping.
int get_logical_thread_at_numa_index(int numaidx);


static int inited = 0;
static int max_cpus;
static int node_mapping[MAX_CPU_NODES];

static int inited_v2 = 0;
static int max_cpus_v2;
static int max_threads;
static int cpumapping[MAX_THREADS];

#ifdef HAVE_LIBNUMA
static int numamap1d[MAX_THREADS];
#endif

static int numthreads;
// if there is no info., default is assuming machine has only-1 NUMA region
static int numnodes = 1;
static int thrpernuma;
static int ** numa_v2;
static char ** numaactive;
static int * numaactivecount;
 
// Initializes the cpu mapping from the file defined by CUSTOM_CPU_MAPPING.
// The mapping used for our machine Intel L5520 is = "8 0 1 2 3 8 9 10 11".
static int init_mappings_from_file()
{
    FILE * cfg;
	int i;

    cfg = fopen(CUSTOM_CPU_MAPPING, "r");
    if (cfg!=NULL) {
        if(fscanf(cfg, "%d", &max_cpus) <= 0) {
            perror("Could not parse input!\n");
        }

        for(i = 0; i < max_cpus; i++){
            if(fscanf(cfg, "%d", &node_mapping[i]) <= 0) {
                perror("Could not parse input!\n");
            }
        }

        fclose(cfg);
        return 1;
    }


    // perror("Custom cpu mapping file not found!\n"); 
    return 0;
}

// Try custom cpu mapping file first, if does not exist then round-robin
// initialization among available CPUs reported by the system. 
static void init_mappings()
{
    if( init_mappings_from_file() == 0 ) {
        int i;
        
        max_cpus  = sysconf(_SC_NPROCESSORS_ONLN);
        for(i = 0; i < max_cpus; i++){
            node_mapping[i] = i;
        }
    }
}

// Returns SMT aware logical to physical CPU mapping for a given thread id.
int get_cpu_id(int thread_id) 
{
    if(!inited){
        init_mappings();
        inited = 1;
    }

    return node_mapping[thread_id % max_cpus];
}

// Topology of Intel E5-4640
// node 0 cpus: 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60
// node 1 cpus: 1 5 9 13 17 21 25 29 33 37 41 45 49 53 57 61
// node 2 cpus: 2 6 10 14 18 22 26 30 34 38 42 46 50 54 58 62
// node 3 cpus: 3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63

// Topology of Intel Xeon-6230
// node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59
// node 1 cpus: 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79

#if INTEL_E5
static int numa[][16] = {
    {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
    {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61},
    {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
    {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63} };
#endif

#if INTEL_XEON
static int numa[][40] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59},
    {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79} };
#endif

int get_numa_id(int mytid)
{
#if INTEL_E5
    int ret = 0;

    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 16; j++)
            if(numa[i][j] == mytid){
                ret = i;
                break;
            }
    
    return ret;
#elif INTEL_XEON
    int ret = 0;

    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 40; j++)
            if(numa[i][j] == mytid){
                ret = i;
                break;
            }
    
    return ret;    
#else
    return 0;
#endif
}

int get_num_numa_regions(void)
{
#if INTEL_E5
    return 4;
#elif INTEL_XEON
    return 2;    
#else
    return 1;
#endif
}

int get_numa_node_of_address(void * ptr)
{
#ifdef HAVE_LIBNUMA        
    int numa_node = -1;
    get_mempolicy(&numa_node, NULL, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR);
    return numa_node;
#else
    return 0;
#endif
}

// Initializes the cpu mapping from the file defined by CUSTOM_CPU_MAPPING.
// NUMBER-OF-THREADS(NTHR) and mapping of PHYSICAL-THR-ID for each LOGICAL-THR-ID from
// 0 to NTHR and optionally number of NUMA-nodes (overridden by libNUMA value if
// exists). 
// The mapping used for our machine Intel E5-4640 is = 
// "64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53  54 55 56 57 58 59 60 61 62 63 4".
static int init_mappings_from_file_v2()
{
    FILE * cfg;
	int i;

    cfg = fopen(CUSTOM_CPU_MAPPING_V2, "r");
    if (cfg!=NULL) {
        if(fscanf(cfg, "%d", &max_threads) <= 0) {
            perror("Could not parse input!\n");
        }

        for(i = 0; i < max_threads; i++){
            if(fscanf(cfg, "%d", &cpumapping[i]) <= 0) {
                perror("Could not parse input!\n");
            }
        }
        int numnodes_from_file = 1;
        if(fscanf(cfg, "%d", &numnodes_from_file) > 0) {
            // number of NUMA nodes configured by the user (if libNUMA not exists) 
            numnodes = numnodes_from_file;
        }

        fclose(cfg);
        fprintf(stdout, "\n[INFO ] CPU mappings are intialized from %s.\n"\
                        "[WARN ] Make sure to use at most %d threads with -n on the commandline.\n",
                        CUSTOM_CPU_MAPPING_V2, max_threads);
        return 1;
    }


    // perror("Custom cpu mapping file not found!\n");
    return 0;
}

// Initialize NUMA-topology with libnuma.
static void numa_default_init()
{
     // numnodes   = 1; 
        numthreads = max_cpus_v2;
        thrpernuma = max_cpus_v2;
        numa_v2 = (int **) malloc(sizeof(int *));
        numa_v2[0] = (int *) malloc(sizeof(int) * numthreads);
        numaactive = (char **) malloc(sizeof(char *));
        numaactive[0] = (char *) calloc(numthreads, sizeof(char));
        numaactivecount = (int *) calloc(numnodes, sizeof(int));
        for(int i = 0; i < max_cpus_v2; i++){
            if(max_cpus_v2 == max_threads)
                numa_v2[0][i] = cpumapping[i];
            else 
                numa_v2[0][i] = i;
        }
}

static void numa_init()
{
#ifdef HAVE_LIBNUMA
	int i, k, ncpus, j;
	struct bitmask *cpus;

    fprintf(stdout, "[INFO ] Getting the NUMA configuration automatically with libNUMA. \n");

	if (numa_available() < 0)  {
		//printf("no numa\n");
		numa_default_init();
		return;
	}

	numnodes   = numa_num_configured_nodes();
	cpus       = numa_allocate_cpumask();
	ncpus      = cpus->size;
	thrpernuma = sysconf(_SC_NPROCESSORS_ONLN) / numnodes; // max #threads per NUMA-region
	numa_v2 = (int **) malloc(sizeof(int *) * numnodes);
	numaactive = (char **) malloc(sizeof(char *) * numnodes);
	for (i = 0; i < numnodes ; i++) {
		numa_v2[i] = (int *) malloc(sizeof(int) * thrpernuma);
		numaactive[i] = (char *) calloc(thrpernuma, sizeof(char));
	}
	numaactivecount = (int *) calloc(numnodes, sizeof(int));

    //printf("\n");
	int nm = 0;
	for (i = 0; i < numnodes ; i++) {
		if (numa_node_to_cpus(i, cpus) < 0) {
			printf("node %d failed to convert\n",i); 
		}		
		//printf("Node-%d: ", i);
		j = 0;
		for (k = 0; k < ncpus; k++){
			if (numa_bitmask_isbitset(cpus, k)){
				//printf(" %s%d", k>0?", ":"", k);
                numa_v2[i][j] = k;
                j++;
                numamap1d[k] = nm++;
            }
		}
		//printf("\n");
	}
    numthreads = thrpernuma * numnodes;

    numa_free_cpumask(cpus);

#else
    fprintf(stdout, "[WARN ] NUMA is not available, using single NUMA-region as default.\n");
    numa_default_init();
#endif
}

void cpu_mapping_cleanup()
{
    for (int i = 0; i < numnodes ; i++) {
        free(numa_v2[i]);
        free(numaactive[i]);
    }
	free(numa_v2);
	free(numaactive);
	free(numaactivecount);
}
 
//  Try custom cpu mapping file first, if does not exist then round-robin
//  initialization among available CPUs reported by the system. 
void cpu_mapping_init()
{
    max_cpus_v2  = sysconf(_SC_NPROCESSORS_ONLN);
    if( init_mappings_from_file_v2() == 0 ) {
        int i;
    
        max_threads = max_cpus_v2;
        for(i = 0; i < max_cpus_v2; i++){
            cpumapping[i] = i;
        }
    }

    numa_init();
    inited_v2 = 1;
}

void numa_thread_mark_active(int phytid)
{
    int numaregionid = -1;
    for(int i = 0; i < numnodes; i++){
        for(int j = 0; j < thrpernuma; j++){
            if(numa_v2[i][j] == phytid){
                numaregionid = i;
                break;
            }
        }
        if(numaregionid != -1)
            break;
    }

    int thridx = -1;
    for(int i = 0; i < numnodes; i++){
        for(int j = 0; j < thrpernuma; j++){
            if(numa_v2[i][j] == phytid){
                thridx = j;
                break;
            }
        }
        if(thridx != -1)
            break;
    }

    if(numaactive[numaregionid][thridx] == 0){
        numaactive[numaregionid][thridx] = 1;
        numaactivecount[numaregionid] ++;
    }
}

// Returns SMT aware logical to physical CPU mapping for a given logical thr-id. 
int get_cpu_id_v2(int thread_id)
{
    if(!inited_v2){
        cpu_mapping_init();
        //inited_v2 = 1;
    }

    return cpumapping[thread_id % max_threads];
}

// Topology of Intel E5-4640 used in our experiments.
// node 0 cpus: 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60
// node 1 cpus: 1 5 9 13 17 21 25 29 33 37 41 45 49 53 57 61
// node 2 cpus: 2 6 10 14 18 22 26 30 34 38 42 46 50 54 58 62
// node 3 cpus: 3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63

// static int numa[][16] = {
//    {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60},
//    {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61},
//    {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62},
//    {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63} };
//

int is_first_thread_in_numa_region(int logicaltid)
{
    int phytid = get_cpu_id_v2(logicaltid);
	int ret = 0;
	for(int i = 0; i < numnodes; i++)
	{
        int j = 0;
        while(j < thrpernuma && !numaactive[i][j])
            j++;
        if(j < thrpernuma)
            ret = ret || (phytid == numa_v2[i][j]);
	}

    return ret;
}

int get_thread_index_in_numa(int logicaltid)
{
    int ret = -1;
    int phytid = get_cpu_id_v2(logicaltid);

    for(int i = 0; i < numnodes; i++){
        int active_idx = 0;
        for(int j = 0; j < thrpernuma; j++){
            if(numa_v2[i][j] == phytid){
                assert(numaactive[i][j]);
                ret = active_idx;
                break;
            }

            if(numaactive[i][j])
                active_idx ++;
        }
        if(ret != -1)
            break;
    }


    return ret;
}

int get_numa_region_id(int logicaltid)
{
    int ret = -1;
    int phytid = get_cpu_id_v2(logicaltid);

    for(int i = 0; i < numnodes; i++){
        for(int j = 0; j < thrpernuma; j++){
            if(numa_v2[i][j] == phytid){
                ret = i;
                break;
            }
        }
        if(ret != -1)
            break;
    }

    return ret;
}

int get_num_numa_regions_v2(void)
{
    return numnodes;
}

int get_num_active_threads_in_numa(int numaregionid)
{
    return numaactivecount[numaregionid];
}

int get_numa_index_of_logical_thread(int logicaltid)
{
#ifdef HAVE_LIBNUMA
    return numamap1d[logicaltid];
#else
    return cpumapping[logicaltid];
#endif
}

int get_logical_thread_at_numa_index(int numaidx)
{
#ifdef HAVE_LIBNUMA
    return numa_v2[numaidx/thrpernuma][numaidx%thrpernuma];
#else
    return cpumapping[numaidx];
#endif
}

#endif /* CPU_MAPPING_H */
