
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



def probe_latencies():
    config_dict={}
    instance_dict={}
    dataset_dict={}

    # print("start")

    with open('kapil_results_mar9.json') as f:
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
                
                # print(line)
                # print(config_key)

                continue
            if "cpu_time" in line: 
                temp_str=line.split(" ")[-1]
                temp_str=temp_str.replace(",","")   
                cpu_time_key=float(temp_str)
                continue

            if "label" in line:   
                dataset_key=line.split(":")[-3]
                dataset_dict[dataset_key]=1
                config_key=scheme_key+"_"+bucket_size_key+"_"+overalloc_key
                config_dict[config_key]=1 
                instance_key=scheme_key+"_"+bucket_size_key+"_"+overalloc_key+"_"+dataset_key+"_"+hash_func_key
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
            for temp_key in instance_dict.keys():
                if inst_key in temp_key:
                    func_key=temp_key.split("_")[-1]
                    instance_dict[temp_key].sort()
                    print("Hash Function: ",func_key,":",instance_dict[temp_key])



probe_latencies()