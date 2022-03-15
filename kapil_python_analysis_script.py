
import json,os, random, sys
import matplotlib
matplotlib.use('agg') #this is to run this code in azure server that does not support plt.show(), we can still save the plot
import matplotlib.pyplot as plt
from os import path
import numpy as np
from scipy.stats import trim_mean
from scipy.stats.mstats import gmean
from tqdm import tqdm
import math
import copy
import random

from matplotlib import rc, rcParams


# hashing scheme -> bucket size -> overalloc -> dataset -> hash function -> all latencies
# config: hashing scheme -> bucket size -> overalloc 
# instance: hashing scheme -> bucket size -> overalloc -> dataset -> hash function


# plt.plot(space_dict[rosetta_key], fpr_dict[rosetta_key],marker='o',markersize=10, lw=4, label="Rosetta", color=color_list[0])
# plt.plot(space_dict[surf_key], fpr_dict[surf_key],marker='o',markersize=10, lw=4, label="SuRF", color=color_list[1])
# plt.plot(space_dict[gcs_key], fpr_dict[gcs_key],marker='o',markersize=10, lw=4, label="SNARF with Golomb Coding", color=color_list[2])
# plt.plot(space_dict[ef_key], fpr_dict[ef_key],marker='o',markersize=10, lw=4, label="SNARF with Elias Fanno Coding", color=color_list[3])
# plt.plot(space_dict[cuckoo_key], fpr_dict[cuckoo_key],marker='o',markersize=10, lw=4, label="Cuckoo Filter", color=color_list[4])


hash_mapping_dict={
    "MURMUR":"Traditional",
    "MULTPrime64":"Traditional",
    "FibonacciPrime64":"Traditional",
    "AquaHash":"Traditional",
    "XXHash3":"Traditional",
    "MWHC":"Exotic",
    "FST":"Exotic",
    "RadixSplineHash":"Model",
    "RMIHash":"Model"
}

hash_color_dict={
    "MURMUR":"blue",
    "MultPrime64":"green",
    "FibonacciPrime64":"Traditional",
    "AquaHash":"Traditional",
    "XXHash3":"black",
    "MWHC":"purple",
    "FST":"Exotic",
    "RadixSplineHash":'red',
    "RMIHash":'orange'
}





def probe_latencies():
    config_dict={}
    instance_dict={}
    dataset_dict={}
    scheme_dict={}
    succ_query_dict={}

    # print("start")

    with open('kapil_results.json') as f:
    # with open('log_results.out') as f:
        lines = f.readlines()
        scheme_key=""
        bucket_size_key=""
        overalloc_key=""
        hash_func_key=""
        dataset_key=""
        model_error_key=""
        model_num_key=""
        cpu_time_key=""
        succ_query_key=""
        for line in lines:
            line=line.strip('\n')
            # print(line)
            if "Start Here" in line:
                line=line[line.find('Start Here'):]
                scheme_key=line.split(" ")[6]
                bucket_size_key=line.split(" ")[2]
                overalloc_key=line.split(" ")[3]
                # hash_func_key=hash_mapping_dict[line.split(" ")[4]]
                hash_func_key=line.split(" ")[4]
                model_error_key=line.split(" ")[-1]
                model_num_key=line.split(" ")[-2]
                config_key=scheme_key+"_"+bucket_size_key+"_"+overalloc_key

                scheme_dict[scheme_key]=1
                
                # print(line)
                # print(config_key)

                continue
            if "cpu_time" in line: 
                temp_str=line.split(" ")[-1]
                temp_str=temp_str.replace(",","")   
                cpu_time_key=float(temp_str)
                continue

            if "label" in line:   
                succ_query_key=line.split(":")[-1]
                succ_query_dict[succ_query_key]=1

                dataset_key=line.split(":")[-3]
                dataset_dict[dataset_key]=1

                config_key=scheme_key+"_"+bucket_size_key+"_"+overalloc_key
                config_dict[config_key]=1 
                instance_key=scheme_key+"_"+bucket_size_key+"_"+overalloc_key+"_"+dataset_key+"_"+hash_func_key+"_"+succ_query_key
                if instance_key in instance_dict.keys():
                    instance_dict[instance_key].append(cpu_time_key)
                else:
                    instance_dict[instance_key]=[]   
                    instance_dict[instance_key].append(cpu_time_key)             
                continue
    # y=1/0.0
    for config_key in config_dict.keys():
        print("\n\n\n\nConfiguration is:",config_key)
        for dataset_key in dataset_dict.keys():
            inst_key=config_key+"_"+dataset_key
            print("\n\nDataset:",dataset_key)
            for succ_key in succ_query_dict.keys():
                # inst_key=config_key+"_"+dataset_key
                print("\nSuccessful Percentage:",succ_key)    
                for temp_key in instance_dict.keys():
                    if inst_key in temp_key:
                        func_key=temp_key.split("_")[-2]
                        temp_succ_key=temp_key.split("_")[-1]
                        if temp_succ_key!=succ_key:
                            continue
                        instance_dict[temp_key].sort()
                        print("Hash Function: ",func_key," succ query ",succ_key,":",instance_dict[temp_key])

#For each (scheme, dataset), plot different (hash functions, scheme config) performance
def expt_1():

    config_dict={}
    instance_dict={}
    dataset_dict={}
    scheme_dict={}
    succ_query_dict={}

    # print("start")

    with open('kapil_results.json') as f:
    # with open('log_results.out') as f:
        lines = f.readlines()
        scheme_key=""
        bucket_size_key=""
        overalloc_key=""
        hash_func_key=""
        dataset_key=""
        model_error_key=""
        model_num_key=""
        cpu_time_key=""
        succ_query_key=""
        for line in lines:
            line=line.strip('\n')
            # print(line)
            if "Start Here" in line:
                line=line[line.find('Start Here'):]
                scheme_key=line.split(" ")[6]
                bucket_size_key=line.split(" ")[2]
                overalloc_key=line.split(" ")[3]
                # hash_func_key=hash_mapping_dict[line.split(" ")[4]]
                hash_func_key=line.split(" ")[4]
                model_error_key=line.split(" ")[-1]
                model_num_key=line.split(" ")[-2]
                config_key=scheme_key+";"+bucket_size_key+";"+overalloc_key

                scheme_dict[scheme_key]=1
                
                # print(line)
                # print(config_key)

                continue
            if "cpu_time" in line: 
                temp_str=line.split(" ")[-1]
                temp_str=temp_str.replace(",","")   
                cpu_time_key=float(temp_str)
                continue

            if "label" in line:   
                succ_query_key=line.split(":")[-1]
                succ_query_key=succ_query_key[:-1]
                succ_query_dict[succ_query_key]=1

                dataset_key=line.split(":")[-3]
                dataset_dict[dataset_key]=1

                config_key=scheme_key+";"+bucket_size_key+";"+overalloc_key
                config_dict[config_key]=1 
                instance_key=scheme_key+";"+bucket_size_key+";"+overalloc_key+";"+dataset_key+";"+hash_func_key+";"+succ_query_key
                if instance_key in instance_dict.keys():
                    instance_dict[instance_key].append(cpu_time_key)
                else:
                    instance_dict[instance_key]=[]   
                    instance_dict[instance_key].append(cpu_time_key)             
                continue

    

    for scheme_key in scheme_dict.keys():
        for dataset_key in dataset_dict.keys():

            done_dict={}           

            plt.figure(figsize=(8,6))

            font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 20}

            matplotlib.rc('font', **font)
            color_list=['blue','green','red',"orange","black"] 

           
            
            for temp_inst_keys in instance_dict.keys():

                while(temp_inst_keys[-1]!=";"):
                    temp_inst_keys=temp_inst_keys[:-1]

                if temp_inst_keys in done_dict.keys():
                    continue
                if ";4;100" not in temp_inst_keys:
                    continue    
                if scheme_key not in temp_inst_keys:
                    continue
                if dataset_key not in temp_inst_keys:
                    continue

                curr_inst_key=temp_inst_keys
                done_dict[curr_inst_key]=1

                latency_list=[]
                succ_key_list=list(succ_query_dict.keys())
                # print(succ_key_list)
                for i in range(0,len(succ_key_list)):
                    succ_key_list[i]=int(succ_key_list[i])
                succ_key_list.sort()

                for succ_key in succ_key_list:
                    
                    temp_key=curr_inst_key+str(succ_key)
                    if("MWHC" in curr_inst_key):
                        temp_key=curr_inst_key+"100"

                    latency_list.append(min(instance_dict[temp_key]))

                split_key=curr_inst_key.split(";")
                label_key=split_key[4]+";"+split_key[1]+";"+split_key[2]   

                print(split_key) 

                plt.plot(succ_key_list,latency_list,marker="o",markersize=10, lw=0.5, label=label_key, color=hash_color_dict[split_key[4]])   
                

           

             
            plt.xlabel('Percentage Successful Queries',fontsize=20,weight='bold')
            plt.legend(fontsize=20)
            #plt.xlim(0.9, 100000)
            # plt.xlabel( _metric + ' ' + subopt_type)
            # plt.yscale('log')
            plt.ylabel('Probe Latency(in ns)',fontsize=20,weight='bold')
            # plt.ylabel('95th Percentile Suboptimality')
            #plt.ylim(0.01, 100000)
            # plt.ylim(0.00005,1.5)
            plt.yticks(fontsize=18,weight='bold')
            plt.xticks(fontsize=18,weight='bold')
            # template_name=all_QTs[0].split('/')[-1]
            savefilename = "expt_1_"+scheme_key+"_"+dataset_key+'.png'
            # savepath = path.join(base_path, 'charts/' + savefilename)
            # plt.title("Geometric mean Suboptimality on a sequence of queries")
            # plt.title("Tail Suboptimality on a sequence of queries")
            plt.tight_layout()
            plt.savefig("figures/"+savefilename)
            plt.close()
            plt.clf()
 







# probe_latencies()
expt_1()