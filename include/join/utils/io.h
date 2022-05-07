/* IO read/write functionalities for input relations adapted from ETH based on https://systems.ethz.ch/research/data-processing-on-modern-hardware/projects/parallel-and-distributed-joins.html */

#pragma once

#include <stdio.h>              /* perror */
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>

#include "utils/data_structures.h"
#include "utils/memory.h"
//#include "utils/eth_data_structures.h"
#include "utils/data_generation.h"
#include "utils/datasets.hpp"

#include "config.h"            /* autoconf header */
#include "configs/base_configs.h"
#include "configs/eth_configs.h"

template<class KeyType, class PayloadType>
struct write_read_arg_t {
    Relation<KeyType, PayloadType>          rel;
    int                 thread_id;
    const char * folder_path;    
    const char * filename;
    const char * file_extension;
    Relation<KeyType, PayloadType> *        fullrel;
};


//based on the ETH implementation 
// persisting a relation to the disk
template<class KeyType, class PayloadType>
void write_relation(Relation<KeyType, PayloadType>* rel, const char * filename);

template<class KeyType, class PayloadType>
void write_relation_threaded(Relation<KeyType, PayloadType>* rel, int nthreads, const char * folder_path, const char * filename, const char * file_extension);

//based on the ETH implementation 
// reading a persisted relation from the disk
template<class KeyType, class PayloadType>
int load_relation(Relation<KeyType, PayloadType>* relation, const char * filename, uint64_t num_tuples);

template<class KeyType, class PayloadType>
int load_relation_threaded(Relation<KeyType, PayloadType>* relation, int nthreads, const char * folder_path, const char * filename, const char * file_extension, uint64_t num_tuples, int is_s_relation = 0);

/**
 * Free memory allocated for only tuples.
 */
template<class KeyType, class PayloadType>
void delete_relation(Relation<KeyType, PayloadType> * rel);

template<typename T>
std::vector<T> load_binary_vector_data(std::string& filename);

template<class KeyType, class PayloadType>
void write_relation(Relation<KeyType, PayloadType>* rel, const char * filename)
{
    FILE * fp = fopen(filename, "w");
    uint64_t i;

    fprintf(fp, "#KEY, VAL\n");

    if (std::is_same<KeyType, int>::value)
    {
        if (std::is_same<PayloadType, int>::value)
            for (i = 0; i < rel->num_tuples; i++)
                fprintf(fp, "%d %d\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned int>::value)
            for (i = 0; i < rel->num_tuples; i++)                
                fprintf(fp, "%d %u\n", rel->tuples[i].key, rel->tuples[i].payload);            
        else if(std::is_same<PayloadType, long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%d %ld\n", rel->tuples[i].key, rel->tuples[i].payload);            
        else if(std::is_same<PayloadType, unsigned long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%d %lu\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, double>::value)
            for (i = 0; i < rel->num_tuples; i++)
                fprintf(fp, "%d %lf\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, long double>::value)
            for (i = 0; i < rel->num_tuples; i++)              
                fprintf(fp, "%d %Lf\n", rel->tuples[i].key, rel->tuples[i].payload);            
    }
    else if(std::is_same<KeyType, unsigned int>::value)
    {
        if (std::is_same<PayloadType, int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%u %d\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned int>::value)
            for (i = 0; i < rel->num_tuples; i++){            
                fprintf(fp, "%u %u\n", rel->tuples[i].key, rel->tuples[i].payload);
            }
        else if(std::is_same<PayloadType, long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%u %ld\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%u %lu\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, double>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%u %lf\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, long double>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%u %Lf\n", rel->tuples[i].key, rel->tuples[i].payload);            
    }
    else if(std::is_same<KeyType, long long int>::value)
    {
        if (std::is_same<PayloadType, int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%ld %d\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%ld %u\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%ld %ld\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%ld %lu\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, double>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%ld %lf\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, long double>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%ld %Lf\n", rel->tuples[i].key, rel->tuples[i].payload);
    }
    else if(std::is_same<KeyType, unsigned long long int>::value)
    {
        if (std::is_same<PayloadType, int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%lu %d\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%lu %u\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%lu %ld\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%lu %lu\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, double>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%lu %lf\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, long double>::value)  
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%lu %Lf\n", rel->tuples[i].key, rel->tuples[i].payload);                      
    }
    else if(std::is_same<KeyType, double>::value)
    {
        if (std::is_same<PayloadType, int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%lf %d\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned int>::value)
            for (i = 0; i < rel->num_tuples; i++)
                fprintf(fp, "%lf %u\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%lf %ld\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%lf %lu\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, double>::value)
            for (i = 0; i < rel->num_tuples; i++)
                fprintf(fp, "%lf %lf\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, long double>::value)            
            for (i = 0; i < rel->num_tuples; i++)
                fprintf(fp, "%lf %Lf\n", rel->tuples[i].key, rel->tuples[i].payload);
    }
    else if(std::is_same<KeyType, long double>::value)
    {
        if (std::is_same<PayloadType, int>::value)
            for (i = 0; i < rel->num_tuples; i++)
                fprintf(fp, "%Lf %d\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned int>::value)
            for (i = 0; i < rel->num_tuples; i++)
                fprintf(fp, "%Lf %u\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%Lf %ld\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, unsigned long long int>::value)
            for (i = 0; i < rel->num_tuples; i++)            
                fprintf(fp, "%Lf %lu\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, double>::value)
            for (i = 0; i < rel->num_tuples; i++)
                fprintf(fp, "%Lf %lf\n", rel->tuples[i].key, rel->tuples[i].payload);
        else if(std::is_same<PayloadType, long double>::value)           
            for (i = 0; i < rel->num_tuples; i++)
                fprintf(fp, "%Lf %Lf\n", rel->tuples[i].key, rel->tuples[i].payload);        
    }

    fclose(fp);
}

template<class KeyType, class PayloadType>
void * write_relation_thread(void * args) 
{
    write_read_arg_t<KeyType, PayloadType> * arg = (write_read_arg_t<KeyType, PayloadType> *) args;

    stringstream full_filename;
    
    full_filename << arg->folder_path;
    full_filename << arg->filename;
    full_filename << "_";
    full_filename << arg->thread_id;
    full_filename << arg->file_extension;

    write_relation(&(arg->rel), full_filename.str().c_str());

    return 0;
}

template<class KeyType, class PayloadType>
void write_relation_threaded(Relation<KeyType, PayloadType>* rel, int nthreads, const char * folder_path, const char * filename, const char * file_extension)
{    
    uint32_t i, rv;
    uint64_t offset = 0;

    write_read_arg_t<KeyType, PayloadType> args[nthreads];
    pthread_t tid[nthreads];
    cpu_set_t set;
    pthread_attr_t attr;

    uint64_t ntuples_perthr;
    uint64_t ntuples_lastthr;

    ntuples_perthr  = rel->num_tuples / nthreads;
    ntuples_lastthr = rel->num_tuples - ntuples_perthr * (nthreads-1);

    pthread_attr_init(&attr);

    for(i = 0; i < nthreads; i++ ) 
    {
        #ifdef DEVELOPMENT_MODE
        int cpu_idx = get_cpu_id_develop(i);
        #else
        int cpu_idx = get_cpu_id_v2(i);
        #endif
    
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

        args[i].folder_path = folder_path;
        args[i].filename = filename;
        args[i].file_extension = file_extension;
        args[i].thread_id = i;
        args[i].rel.tuples     = rel->tuples + offset;
        args[i].rel.num_tuples = (i == nthreads-1) ? ntuples_lastthr 
                                 : ntuples_perthr;
        offset += ntuples_perthr;

        rv = pthread_create(&tid[i], &attr, write_relation_thread<KeyType, PayloadType>, (void*)&args[i]);
        if (rv){
            fprintf(stderr, "[ERROR] pthread_create() return code is %d\n", rv);
            exit(-1);
        }
    }

    for(i = 0; i < nthreads; i++){
        pthread_join(tid[i], NULL);
    }
}


// based on the ETH implementation
template<class KeyType, class PayloadType>
void read_relation(Relation<KeyType, PayloadType> * rel, const char * filename){

    FILE * fp = fopen(filename, "r");

    // skip the header line 
    char c;
    do{
        c = fgetc(fp);
    } while (c != '\n');

    // search for a whitespace for "key payload" format 
    int fmtspace = 0;
    //int fmtcomma = 0;
    do{
        c = fgetc(fp);
        if(c == ' '){
            fmtspace = 1;
            break;
        }
    //    if(c == ','){
    //        fmtcomma = 1;
    //        break;
    //    }
    } while (c != '\n');

    // rewind back to the beginning and start parsing again 
    rewind(fp);
    // skip the header line
    do{
        c = fgetc(fp);
    } while (c != '\n');

    uint64_t ntuples = rel->num_tuples;
    KeyType key;
    PayloadType payload = 0;
    int warn = 1;
    if(fmtspace){
        if (std::is_same<KeyType, int>::value && std::is_same<PayloadType, int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%d %d\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, int>::value && std::is_same<PayloadType, unsigned int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%d %u\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, int>::value && std::is_same<PayloadType, long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%d %ld\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, int>::value && std::is_same<PayloadType, unsigned long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%d %lu\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, int>::value && std::is_same<PayloadType, double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%d %lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, int>::value && std::is_same<PayloadType, long double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%d %Lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, unsigned int>::value && std::is_same<PayloadType, int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%u %d\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, unsigned int>::value && std::is_same<PayloadType, unsigned int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%u %u\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, unsigned int>::value && std::is_same<PayloadType, long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%u %ld\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, unsigned int>::value && std::is_same<PayloadType, unsigned long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%u %lu\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, unsigned int>::value && std::is_same<PayloadType, double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%u %lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, unsigned int>::value && std::is_same<PayloadType, long double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%u %Lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, long long int>::value && std::is_same<PayloadType, int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%ld %d\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }        
        else if (std::is_same<KeyType, long long int>::value && std::is_same<PayloadType, unsigned int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%ld %u\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, long long int>::value && std::is_same<PayloadType, long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%ld %ld\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, long long int>::value && std::is_same<PayloadType, unsigned long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%ld %lu\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }        
        else if (std::is_same<KeyType, long long int>::value && std::is_same<PayloadType, double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%ld %lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        } 
        else if (std::is_same<KeyType, long long int>::value && std::is_same<PayloadType, long double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%ld %Lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }    
        else if (std::is_same<KeyType, unsigned long long int>::value && std::is_same<PayloadType, int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lu %d\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }       
        else if (std::is_same<KeyType, unsigned long long int>::value && std::is_same<PayloadType, unsigned int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lu %u\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, unsigned long long int>::value && std::is_same<PayloadType, long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lu %ld\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, unsigned long long int>::value && std::is_same<PayloadType, unsigned long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lu %lu\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }       
        else if (std::is_same<KeyType, unsigned long long int>::value && std::is_same<PayloadType, double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lu %lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }       
        else if (std::is_same<KeyType, unsigned long long int>::value && std::is_same<PayloadType, long double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lu %Lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else if (std::is_same<KeyType, double>::value && std::is_same<PayloadType, int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lf %d\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }       
        else if (std::is_same<KeyType, double>::value && std::is_same<PayloadType, unsigned int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lf %u\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }      
        else if (std::is_same<KeyType, double>::value && std::is_same<PayloadType, long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lf %ld\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }       
        else if (std::is_same<KeyType, double>::value && std::is_same<PayloadType, unsigned long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lf %lu\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }       
        else if (std::is_same<KeyType, double>::value && std::is_same<PayloadType, double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lf %lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }   
        else if (std::is_same<KeyType, double>::value && std::is_same<PayloadType, long double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%lf %Lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }    
        else if (std::is_same<KeyType, long double>::value && std::is_same<PayloadType, int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%Lf %d\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }      
        else if (std::is_same<KeyType, long double>::value && std::is_same<PayloadType, unsigned int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%Lf %u\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }       
        else if (std::is_same<KeyType, long double>::value && std::is_same<PayloadType, long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%Lf %ld\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }       
        else if (std::is_same<KeyType, long double>::value && std::is_same<PayloadType, unsigned long long int>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%Lf %lu\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }       
        else if (std::is_same<KeyType, long double>::value && std::is_same<PayloadType, double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%Lf %lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }       
        else if (std::is_same<KeyType, long double>::value && std::is_same<PayloadType, long double>::value)
        {
            for(uint64_t i = 0; i < ntuples; i++)
            {
                fscanf(fp, "%Lf %Lf\n", &key, &payload);
                rel->tuples[i].key = key;
                rel->tuples[i].payload = payload;
            }
        }
        else
        {
            perror("Not supported format");            
        }       
    }
    else
    {
        perror("Incorrect format");
    }

/*    
    for(uint64_t i = 0; i < ntuples; i++){
        if(fmtspace){
            if (std::is_same<KeyType, int>::value)
            {
                if (std::is_same<PayloadType, int>::value)
                        fscanf(fp, "%d %d\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned int>::value)
                        fscanf(fp, "%d %u\n", &key, &payload);            
                else if(std::is_same<PayloadType, long long int>::value)
                        fscanf(fp, "%d %ld\n", &key, &payload);            
                else if(std::is_same<PayloadType, unsigned long long int>::value)
                        fscanf(fp, "%d %lu\n", &key, &payload);
                else if(std::is_same<PayloadType, double>::value)
                        fscanf(fp, "%d %lf\n", &key, &payload);
                else if(std::is_same<PayloadType, long double>::value)
                        fscanf(fp, "%d %Lf\n", &key, &payload);            
            }
            else if(std::is_same<KeyType, unsigned int>::value)
            {
                if (std::is_same<PayloadType, int>::value)
                        fscanf(fp, "%u %d\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned int>::value)
                        fscanf(fp, "%u %u\n", &key, &payload);
                else if(std::is_same<PayloadType, long long int>::value)
                        fscanf(fp, "%u %ld\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned long long int>::value)
                        fscanf(fp, "%u %lu\n", &key, &payload);
                else if(std::is_same<PayloadType, double>::value)
                        fscanf(fp, "%u %lf\n", &key, &payload);
                else if(std::is_same<PayloadType, long double>::value)
                        fscanf(fp, "%u %Lf\n", &key, &payload);            
            }
            else if(std::is_same<KeyType, long long int>::value)
            {
                if (std::is_same<PayloadType, int>::value)
                        fscanf(fp, "%ld %d\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned int>::value)
                        fscanf(fp, "%ld %u\n", &key, &payload);
                else if(std::is_same<PayloadType, long long int>::value)
                        fscanf(fp, "%ld %ld\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned long long int>::value)
                        fscanf(fp, "%ld %lu\n", &key, &payload);
                else if(std::is_same<PayloadType, double>::value)
                        fscanf(fp, "%ld %lf\n", &key, &payload);
                else if(std::is_same<PayloadType, long double>::value)
                        fscanf(fp, "%ld %Lf\n", &key, &payload);
            }
            else if(std::is_same<KeyType, unsigned long long int>::value)
            {
                if (std::is_same<PayloadType, int>::value)
                        fscanf(fp, "%lu %d\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned int>::value)
                        fscanf(fp, "%lu %u\n", &key, &payload);
                else if(std::is_same<PayloadType, long long int>::value)
                        fscanf(fp, "%lu %ld\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned long long int>::value)
                        fscanf(fp, "%lu %lu\n", &key, &payload);
                else if(std::is_same<PayloadType, double>::value)
                        fscanf(fp, "%lu %lf\n", &key, &payload);
                else if(std::is_same<PayloadType, long double>::value)  
                        fscanf(fp, "%lu %Lf\n", &key, &payload);                      
            }
            else if(std::is_same<KeyType, double>::value)
            {
                if (std::is_same<PayloadType, int>::value)
                        fscanf(fp, "%lf %d\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned int>::value)
                        fscanf(fp, "%lf %u\n", &key, &payload);
                else if(std::is_same<PayloadType, long long int>::value)
                        fscanf(fp, "%lf %ld\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned long long int>::value)
                        fscanf(fp, "%lf %lu\n", &key, &payload);
                else if(std::is_same<PayloadType, double>::value)
                        fscanf(fp, "%lf %lf\n", &key, &payload);
                else if(std::is_same<PayloadType, long double>::value)            
                        fscanf(fp, "%lf %Lf\n", &key, &payload);
            }
            else if(std::is_same<KeyType, long double>::value)
            {
                if (std::is_same<PayloadType, int>::value)
                        fscanf(fp, "%Lf %d\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned int>::value)
                        fscanf(fp, "%Lf %u\n", &key, &payload);
                else if(std::is_same<PayloadType, long long int>::value)
                        fscanf(fp, "%Lf %ld\n", &key, &payload);
                else if(std::is_same<PayloadType, unsigned long long int>::value)
                        fscanf(fp, "%Lf %lu\n", &key, &payload);
                else if(std::is_same<PayloadType, double>::value)
                        fscanf(fp, "%Lf %lf\n", &key, &payload);
                else if(std::is_same<PayloadType, long double>::value)           
                        fscanf(fp, "%Lf %Lf\n", &key, &payload);        
            }
        }
        else
        {
            perror("Incorrect format");
        }

        if(warn && key < 0){
            warn = 0;
            printf("[WARN ] key=%d, payload=%d\n", key, payload);
        }
        rel->tuples[i].key = key;
        rel->tuples[i].payload = payload;
    }
*/
    fclose(fp);
}

template<class KeyType, class PayloadType>
int load_relation(Relation<KeyType, PayloadType>* relation, const char * filename, uint64_t num_tuples)
{
    relation->num_tuples = num_tuples;

    /* we need aligned allocation of items */
    relation->tuples = (Tuple<KeyType, PayloadType> *) alloc_aligned(num_tuples * sizeof(Tuple<KeyType, PayloadType>));

    if (!relation->tuples) { 
        perror("out of memory");
        return -1; 
    }

    /* load from the given input file */
    read_relation(relation, filename);

    return 0;    
}

template<class KeyType, class PayloadType>
void * read_relation_thread(void * args) 
{
    write_read_arg_t<KeyType, PayloadType> * arg = (write_read_arg_t<KeyType, PayloadType> *) args;

    stringstream full_filename;
    
    full_filename << arg->folder_path;
    full_filename << arg->filename;
    full_filename << "_";
    full_filename << arg->thread_id;
    full_filename << arg->file_extension;

    read_relation(&(arg->rel), full_filename.str().c_str());

    return 0;
}

template<class KeyType, class PayloadType>
void read_relation_threaded(Relation<KeyType, PayloadType>* rel, int nthreads, const char * folder_path, const char * filename, const char * file_extension)
{    
    uint32_t i, rv;
    uint64_t offset = 0;

    write_read_arg_t<KeyType, PayloadType> args[nthreads];
    pthread_t tid[nthreads];
    cpu_set_t set;
    pthread_attr_t attr;

    uint64_t ntuples_perthr;
    uint64_t ntuples_lastthr;

    ntuples_perthr  = rel->num_tuples / nthreads;
    ntuples_lastthr = rel->num_tuples - ntuples_perthr * (nthreads-1);

    pthread_attr_init(&attr);

    for(i = 0; i < nthreads; i++ ) 
    {
        #ifdef DEVELOPMENT_MODE
        int cpu_idx = get_cpu_id_develop(i);
        #else
        int cpu_idx = get_cpu_id_v2(i);
        #endif
    
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

        args[i].folder_path = folder_path;
        args[i].filename = filename;
        args[i].file_extension = file_extension;
        args[i].thread_id = i;
        args[i].rel.tuples     = rel->tuples + offset;
        args[i].rel.num_tuples = (i == nthreads-1) ? ntuples_lastthr 
                                 : ntuples_perthr;
        offset += ntuples_perthr;

        rv = pthread_create(&tid[i], &attr, read_relation_thread<KeyType, PayloadType>, (void*)&args[i]);
        if (rv){
            fprintf(stderr, "[ERROR] pthread_create() return code is %d\n", rv);
            exit(-1);
        }
    }

    for(i = 0; i < nthreads; i++){
        pthread_join(tid[i], NULL);
    }
}

template<class KeyType, class PayloadType>
int load_relation_threaded(Relation<KeyType, PayloadType>* relation, int nthreads, const char * folder_path, const char * filename, const char * file_extension, uint64_t num_tuples, int is_s_relation = 0)
{
    relation->num_tuples = num_tuples;

    /* we need aligned allocation of items */
    relation->tuples = (Tuple<KeyType, PayloadType> *) alloc_aligned(num_tuples * sizeof(Tuple<KeyType, PayloadType>));

    if (!relation->tuples) { 
        perror("out of memory");
        return -1; 
    }

    ///TODO: temporary checkup for hash benchmarking SOSD datasets
    #ifdef HASH_LEARNED_MODEL
    std::vector<std::pair<KeyType, PayloadType>> data{};
    data.reserve(num_tuples);
    {
      if (std::strcmp(filename,"fb_200M_uint64") == 0)
      {
        auto keys = dataset::load_cached<KeyType>(dataset::ID::FB, num_tuples, is_s_relation);

        std::transform(
            keys.begin(), keys.end(), std::back_inserter(data),
            [](const KeyType& key) { return std::make_pair(key, key - 5); });
      } 
      else if (std::strcmp(filename,"osm_cellids_800M_uint64") == 0)
      {
        auto keys = dataset::load_cached<KeyType>(dataset::ID::OSM, num_tuples, is_s_relation);

        std::transform(
            keys.begin(), keys.end(), std::back_inserter(data),
            [](const KeyType& key) { return std::make_pair(key, key - 5); });
      }   
      else if (std::strcmp(filename,"wiki_ts_200M_uint64") == 0)
      {
        auto keys = dataset::load_cached<KeyType>(dataset::ID::WIKI, num_tuples, is_s_relation);

        std::transform(
            keys.begin(), keys.end(), std::back_inserter(data),
            [](const KeyType& key) { return std::make_pair(key, key - 5); });
      }   
      else if (std::strcmp(filename,"r_UNIQUE_v5_uint32_uint32_640000000") == 0)
      {
        auto keys = dataset::load_cached<KeyType>(dataset::ID::GAPPED_10, num_tuples, is_s_relation);

        std::transform(
            keys.begin(), keys.end(), std::back_inserter(data),
            [](const KeyType& key) { return std::make_pair(key, key - 5); });
      }
      else if (std::strcmp(filename,"r_SEQ_HOLE_v5_uint32_uint32_640000000") == 0)
      {
        auto keys = dataset::load_cached<KeyType>(dataset::ID::SEQUENTIAL, num_tuples, is_s_relation);

        std::transform(
            keys.begin(), keys.end(), std::back_inserter(data),
            [](const KeyType& key) { return std::make_pair(key, key - 5); });
      }
      else if (std::strcmp(filename,"r_UNIFORM_v5_uint32_uint32_640000000") == 0)
      {
        auto keys = dataset::load_cached<KeyType>(dataset::ID::UNIFORM, num_tuples, is_s_relation);

        std::transform(
            keys.begin(), keys.end(), std::back_inserter(data),
            [](const KeyType& key) { return std::make_pair(key, key - 5); });
      }            
      else
      {
         perror("Unrecognized file!");
         return -1;
      }

    }

    for(uint64_t i = 0; i < num_tuples; i++)
    {
        relation->tuples[i].key = data[i].first;
        relation->tuples[i].payload = data[i].second;
    }
    #else
    read_relation_threaded(relation, nthreads, folder_path, filename, file_extension);
    #endif

    return 0;
}

// Materialize the generated results
template<class KeyType, class PayloadType>
void materialize_one_relation(KeyType* input_keys, uint64_t input_keys_len)
{
    Relation<KeyType, PayloadType> rel_r;
 
    int64_t result = 0;
    uint64_t curr_num_tuples_r = input_keys_len;

    string curr_rel_r_path = RELATION_R_PATH;
    string curr_rel_r_folder_path = RELATION_R_FOLDER_PATH;
    string curr_rel_r_file_name = RELATION_R_FILE_NAME;
    string curr_rel_r_file_extension = RELATION_R_FILE_EXTENSION;

    result = create_input_workload_relation<KeyType, PayloadType>(input_keys, &rel_r, curr_num_tuples_r, 0);
    //ASSERT_EQ(result, 0);
    #ifdef PERSIST_RELATIONS_FOR_EVALUATION
    write_relation_threaded<KeyType, PayloadType>(&rel_r, RELATION_R_FILE_NUM_PARTITIONS, curr_rel_r_folder_path.c_str(), curr_rel_r_file_name.c_str(), curr_rel_r_file_extension.c_str());
    write_relation<KeyType, PayloadType>(&rel_r, curr_rel_r_path.c_str());
    #endif

    KeyType * sorted_relation_r_keys_only = (KeyType *) alloc_aligned(rel_r.num_tuples  * sizeof(KeyType));
    Relation<KeyType, PayloadType> sorted_relation_r;

    for(int j = 0; j < rel_r.num_tuples; j++)
        sorted_relation_r_keys_only[j] = rel_r.tuples[j].key;

    std::sort((KeyType *)(sorted_relation_r_keys_only), (KeyType *)(sorted_relation_r_keys_only) + rel_r.num_tuples);

    sorted_relation_r.num_tuples = rel_r.num_tuples;
    sorted_relation_r.tuples = (Tuple<KeyType, PayloadType> *) alloc_aligned(rel_r.num_tuples * sizeof(Tuple<KeyType, PayloadType>));

    for(int j = 0; j < sorted_relation_r.num_tuples; j++)
        sorted_relation_r.tuples[j].key = sorted_relation_r_keys_only[j];

    string sorted_r_file_name = curr_rel_r_file_name + "_sorted";

    #ifdef PERSIST_RELATIONS_FOR_EVALUATION
    write_relation_threaded<KeyType, PayloadType>(&sorted_relation_r, RELATION_R_FILE_NUM_PARTITIONS, curr_rel_r_folder_path.c_str(), sorted_r_file_name.c_str(), curr_rel_r_file_extension.c_str());
    #endif
}

template<class KeyType, class PayloadType>
void delete_relation(Relation<KeyType, PayloadType> * rel)
{
    /* clean up */
    free(rel->tuples);
}

// Adapted implementation from the SOSD benchmark
// Loads values from binary file into vector.
template<typename T>
std::vector<T> load_binary_vector_data(std::string& filename) {
    std::vector<T> data;

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "unable to open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    // Read size.
    uint64_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
    data.resize(size);
    // Read values.
    in.read(reinterpret_cast<char*>(data.data()), size*sizeof(T));
    in.close();

    return data;
}