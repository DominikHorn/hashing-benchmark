# pragma once

/* Adapted version from cpu_mapping.h for one numa node */

#include <unistd.h> /* sysconf */

#include "configs/base_configs.h"

static int inited_develop = 0;
static int max_cpus_develop;
static int node_mapping_develop[MAX_CPU_NODES];

int get_cpu_id_develop(int thread_id);

int get_numa_id_develop(int mytid);

int get_num_numa_regions_develop(void);

int get_numa_node_of_address_develop(void * ptr);

int 
get_cpu_id_develop(int thread_id) 
{
    if(!inited_develop){
        int i;
        
        max_cpus_develop  = sysconf(_SC_NPROCESSORS_ONLN);
        for(i = 0; i < max_cpus_develop; i++){
            node_mapping_develop[i] = i;
        }

        inited_develop = 1;
    }

    return node_mapping_develop[thread_id % max_cpus_develop];
}

int
get_numa_id_develop(int mytid)
{
    return 0;
}

int
get_num_numa_regions_develop(void)
{
    return 1;
}

int 
get_numa_node_of_address_develop(void * ptr)
{
    return 0;
}
