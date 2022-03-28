
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
    "RadixSplineHashSmall":'orange',
    "RMIHashSmall":'grey',
    "RadixSplineHashBig":'red',
    "RMIHashBig":'brown'
}


# file_str_latency="data_logs_combined/kapil_combined_mar20.json"
# file_str_stats="data_logs_combined/data_stats_combined_mar20.out"
file_str_latency="kapil_results.json"
file_str_stats="data_stats_mar14.out"
file_gap_stats="logs/gap_stats_21mar.out"
file_variance_stats="logs/variance_stats_mar22.out"
file_model_size_stats="logs/model_size_stats_mar22.out"
# file_variance_stats="logs/collision_stats.out"

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
    overalloc_dict={}

    # print("start")

    with open(file_str_latency) as f:
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
                if "Cuckoo" in scheme_key:
                    scheme_key=line.split(" ")[6]+"_"+line.split(" ")[7]
                bucket_size_key=line.split(" ")[2]
                overalloc_key=line.split(" ")[3]
                if int(overalloc_key)>10000:
                    overalloc_key=str(int(overalloc_key)-10000)
                else:
                    overalloc_key=str(int(overalloc_key)+100) 
                overalloc_dict[overalloc_key]=1       
                # hash_func_key=hash_mapping_dict[line.split(" ")[4]]
                hash_func_key=line.split(" ")[4]
                
                model_error_key=line.split(" ")[-1]
                model_num_key=line.split(" ")[-2]
                config_key=scheme_key+";"+bucket_size_key+";"+overalloc_key

                if "RMI" in hash_func_key :
                    if int(model_num_key)<10:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"

                if "Radix" in hash_func_key:
                    if int(model_error_key)<5000:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"        

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

            plt.figure(figsize=(14,12))

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
                if ";1;100" not in temp_inst_keys and "Cuckoo" not in temp_inst_keys and "MWHC" not in temp_inst_keys:
                    continue    

                if "Cuckoo"  in temp_inst_keys and ";4;15" not in temp_inst_keys:
                    continue    

                if "MWHC" not in temp_inst_keys:
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

                plt.plot(succ_key_list,latency_list,marker="o",markersize=10, lw=4.5, label=label_key, color=hash_color_dict[split_key[4]])   
                

           

             
            plt.xlabel('Percentage Successful Queries',fontsize=20,weight='bold')
            plt.legend(fontsize=15)
            #plt.xlim(0.9, 100000)
            # plt.xlabel( _metric + ' ' + subopt_type)
            # plt.yscale('log')
            plt.ylabel('Probe Latency(in ns)',fontsize=20,weight='bold')
            # plt.ylabel('95th Percentile Suboptimality')
            #plt.ylim(0.01, 100000)
            # plt.ylim(0.00005,1.5)
            plt.yticks(fontsize=36,weight='bold')
            plt.xticks(fontsize=36,weight='bold')
            # template_name=all_QTs[0].split('/')[-1]
            savefilename = "expt_1_"+scheme_key+"_"+dataset_key+'.png'
            # savepath = path.join(base_path, 'charts/' + savefilename)
            # plt.title("Geometric mean Suboptimality on a sequence of queries")
            # plt.title("Tail Suboptimality on a sequence of queries")
            plt.tight_layout()
            plt.savefig("figures/"+savefilename)
            plt.close()
            plt.clf()
 


#For each Linear Probing (dataset), plot different (hash functions, scheme config) distance plots
def expt_2_linear_dist_from_ideal():

    config_dict={}
    instance_dict={}
    dataset_dict={}
    scheme_dict={}
    succ_query_dict={}

    distance_from_ideal={}
    distance_to_empty={}

    temp_distance_from_ideal={}
    temp_distance_to_empty={}

    # print("start")

    with open(file_str_stats) as f:
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

                if "RMI" in hash_func_key :
                    if int(model_num_key)<10000:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"

                if "Radix" in hash_func_key:
                    if int(model_error_key)<100:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"        

                scheme_dict[scheme_key]=1

                
                # print(line)
                # print(config_key)
                continue

            # if scheme_key!="Linear":
            #     continue

            if "Dataset Size: " in line:
                dataset_key=line.split(" ")[-1]
                dataset_dict[dataset_key]=1    

            if "Distance To Empty:" in line:
                split_line=line.split(" ")
                if abs(int(split_line[3]))>5:
                    continue
                temp_distance_to_empty[abs(int(split_line[3]))]=int(split_line[5])

            if "Distance From Ideal:" in line:
                split_line=line.split(" ")
                if int(split_line[3])>0:
                    continue
                if abs(int(split_line[3]))>5:
                    continue
                temp_distance_from_ideal[abs(int(split_line[3]))]=int(split_line[5])    

            if "Distance From Ideal: 0" in line:
                instance_key=scheme_key+";"+bucket_size_key+";"+overalloc_key+";"+dataset_key+";"+hash_func_key
                distance_from_ideal[instance_key]=temp_distance_from_ideal
                distance_to_empty[instance_key]=temp_distance_to_empty
                temp_distance_from_ideal={}
                temp_distance_to_empty={}

    # print(distance_from_ideal)
    # print(distance_to_empty)
    key_to_plot="Linear;1;100;"
    for dataset_key in dataset_dict.keys():

        done_dict={}           

        plt.figure(figsize=(14,12))

        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 20}

        matplotlib.rc('font', **font)
        color_list=['blue','green','red',"orange","black"] 

        
        
        for temp_inst_keys in distance_from_ideal.keys():

            if temp_inst_keys in done_dict.keys():
                continue
            if key_to_plot not in temp_inst_keys:
                continue
            if dataset_key not in temp_inst_keys:
                continue

            curr_inst_key=temp_inst_keys
            done_dict[curr_inst_key]=1

            plot_dist_from_ideal=[]
            plot_dist_from_ideal_count=[]

            plot_dist_to_empty=[]
            plot_dist_to_empty_count=[]

            for temp_dist in distance_from_ideal[curr_inst_key].keys():
                plot_dist_from_ideal.append(temp_dist)
                plot_dist_from_ideal_count.append(distance_from_ideal[curr_inst_key][temp_dist])

            for temp_dist in distance_to_empty[curr_inst_key].keys():
                plot_dist_to_empty.append(temp_dist)
                plot_dist_to_empty_count.append(distance_to_empty[curr_inst_key][temp_dist])     


            split_key=curr_inst_key.split(";")
            label_key=split_key[4]+";"+split_key[1]+";"+split_key[2]   

            print(split_key) 

            plt.plot(plot_dist_from_ideal,plot_dist_from_ideal_count,marker="o",markersize=10, lw=4.5, label=label_key, color=hash_color_dict[split_key[4]])   
            

        

            
        plt.xlabel('Distance From Ideal',fontsize=20,weight='bold')
        plt.legend(fontsize=15)
        #plt.xlim(0.9, 100000)
        # plt.xlabel( _metric + ' ' + subopt_type)
        plt.yscale('log')
        plt.ylabel('Count',fontsize=20,weight='bold')
        # plt.ylabel('95th Percentile Suboptimality')
        #plt.ylim(0.01, 100000)
        # plt.ylim(0.00005,1.5)
        plt.yticks(fontsize=36,weight='bold')
        plt.xticks(fontsize=36,weight='bold')
        # template_name=all_QTs[0].split('/')[-1]
        savefilename = "expt_2_Linear"+"_"+dataset_key+'_dist_from_ideal.png'
        # savepath = path.join(base_path, 'charts/' + savefilename)
        # plt.title("Geometric mean Suboptimality on a sequence of queries")
        # plt.title("Tail Suboptimality on a sequence of queries")
        plt.tight_layout()
        plt.savefig("figures/"+savefilename)
        plt.close()
        plt.clf()

#For each Linear Probing (dataset), plot different (hash functions, scheme config) distance plots
def expt_2_linear_dist_to_empty():

    config_dict={}
    instance_dict={}
    dataset_dict={}
    scheme_dict={}
    succ_query_dict={}

    distance_from_ideal={}
    distance_to_empty={}

    temp_distance_from_ideal={}
    temp_distance_to_empty={}

    # print("start")

    with open(file_str_stats) as f:
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

                if "RMI" in hash_func_key :
                    if int(model_num_key)<10000:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"

                if "Radix" in hash_func_key:
                    if int(model_error_key)<100:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"        

                scheme_dict[scheme_key]=1

                
                # print(line)
                # print(config_key)
                continue

            # if scheme_key!="Linear":
            #     continue

            if "Dataset Size: " in line:
                dataset_key=line.split(" ")[-1]
                dataset_dict[dataset_key]=1    

            if "Distance To Empty:" in line:
                split_line=line.split(" ")
                if abs(int(split_line[3]))>5:
                    continue
                temp_distance_to_empty[abs(int(split_line[3]))]=int(split_line[5])

            if "Distance From Ideal:" in line:
                split_line=line.split(" ")
                if int(split_line[3])>0:
                    continue
                if abs(int(split_line[3]))>5:
                    continue
                temp_distance_from_ideal[abs(int(split_line[3]))]=int(split_line[5])    

            if "Distance From Ideal: 0" in line:
                instance_key=scheme_key+";"+bucket_size_key+";"+overalloc_key+";"+dataset_key+";"+hash_func_key
                distance_from_ideal[instance_key]=temp_distance_from_ideal
                distance_to_empty[instance_key]=temp_distance_to_empty
                temp_distance_from_ideal={}
                temp_distance_to_empty={}

    # print(distance_from_ideal)
    # print(distance_to_empty)
    key_to_plot="Linear;1;100;"
    for dataset_key in dataset_dict.keys():

        done_dict={}           

        plt.figure(figsize=(14,12))

        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 20}

        matplotlib.rc('font', **font)
        color_list=['blue','green','red',"orange","black"] 

        
        
        for temp_inst_keys in distance_from_ideal.keys():

            if temp_inst_keys in done_dict.keys():
                continue
            if key_to_plot not in temp_inst_keys:
                continue
            if dataset_key not in temp_inst_keys:
                continue

            curr_inst_key=temp_inst_keys
            done_dict[curr_inst_key]=1

            plot_dist_from_ideal=[]
            plot_dist_from_ideal_count=[]

            plot_dist_to_empty=[]
            plot_dist_to_empty_count=[]

            for temp_dist in distance_from_ideal[curr_inst_key].keys():
                plot_dist_from_ideal.append(temp_dist)
                plot_dist_from_ideal_count.append(distance_from_ideal[curr_inst_key][temp_dist])

            for temp_dist in distance_to_empty[curr_inst_key].keys():
                plot_dist_to_empty.append(temp_dist)
                plot_dist_to_empty_count.append(distance_to_empty[curr_inst_key][temp_dist])     


            split_key=curr_inst_key.split(";")
            label_key=split_key[4]+";"+split_key[1]+";"+split_key[2]   

            print(split_key) 

            plt.plot(plot_dist_to_empty,plot_dist_to_empty_count,marker="o",markersize=10, lw=4.5, label=label_key, color=hash_color_dict[split_key[4]])   
            

        

            
        plt.xlabel('Distance To Empty',fontsize=20,weight='bold')
        plt.legend(fontsize=15)
        #plt.xlim(0.9, 100000)
        # plt.xlabel( _metric + ' ' + subopt_type)
        plt.yscale('log')
        plt.ylabel('Count',fontsize=20,weight='bold')
        # plt.ylabel('95th Percentile Suboptimality')
        #plt.ylim(0.01, 100000)
        # plt.ylim(0.00005,1.5)
        plt.yticks(fontsize=36,weight='bold')
        plt.xticks(fontsize=36,weight='bold')
        # template_name=all_QTs[0].split('/')[-1]
        savefilename = "expt_2_Linear"+"_"+dataset_key+'_dist_to_empty.png'
        # savepath = path.join(base_path, 'charts/' + savefilename)
        # plt.title("Geometric mean Suboptimality on a sequence of queries")
        # plt.title("Tail Suboptimality on a sequence of queries")
        plt.tight_layout()
        plt.savefig("figures/"+savefilename)
        plt.close()
        plt.clf()




#For each Chained (dataset), plot different (hash functions, scheme config) num ele plots
def expt_2_chained():

    config_dict={}
    instance_dict={}
    dataset_dict={}
    scheme_dict={}
    succ_query_dict={}

    num_ele_per_loc={}
    num_buc_per_loc={}

    temp_num_ele_per_loc={}
    temp_num_buc_per_loc={}

    # print("start")

    with open(file_str_stats) as f:
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

                if "RMI" in hash_func_key :
                    if int(model_num_key)<10000:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"

                if "Radix" in hash_func_key:
                    if int(model_error_key)<100:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"        

                scheme_dict[scheme_key]=1

                
                # print(line)
                # print(config_key)
                continue

            # if scheme_key!="Linear":
            #     continue

            if "Dataset Size: " in line:
                dataset_key=line.split(" ")[-1]
                dataset_dict[dataset_key]=1    

            if "Num Elements: " in line:
                split_line=line.split(" ")
                if abs(int(split_line[4]))<100:
                    continue
                temp_num_ele_per_loc[abs(int(split_line[2]))]=int(split_line[4])
                temp_num_buc_per_loc[abs(int(split_line[2]))]=math.ceil(int(split_line[4])*1.00/int(bucket_size_key))

            # if "Distance From Ideal:" in line:
            #     split_line=line.split(" ")
            #     if int(split_line[3])>0:
            #         continue
            #     if abs(int(split_line[3]))>5:
            #         continue
            #     temp_distance_from_ideal[abs(int(split_line[3]))]=int(split_line[5])    

            if "PointProbe<" in line:
                instance_key=scheme_key+";"+bucket_size_key+";"+overalloc_key+";"+dataset_key+";"+hash_func_key
                num_ele_per_loc[instance_key]=temp_num_ele_per_loc
                num_buc_per_loc[instance_key]=temp_num_buc_per_loc
                temp_num_ele_per_loc={}
                temp_num_buc_per_loc={}

    # print(distance_from_ideal)
    # print(distance_to_empty)
    key_to_plot="Chained;1;100;"
    for dataset_key in dataset_dict.keys():

        done_dict={}           

        plt.figure(figsize=(14,12))

        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 20}

        matplotlib.rc('font', **font)
        color_list=['blue','green','red',"orange","black"] 

        
        
        for temp_inst_keys in num_buc_per_loc.keys():

            if temp_inst_keys in done_dict.keys():
                continue
            if key_to_plot not in temp_inst_keys:
                continue
            if dataset_key not in temp_inst_keys:
                continue

            curr_inst_key=temp_inst_keys
            done_dict[curr_inst_key]=1

            plot_num_ele_per_loc=[]
            plot_num_ele_per_loc_count=[]

            plot_num_buc_per_loc=[]
            plot_num_buc_per_loc_count=[]

            for temp_dist in num_ele_per_loc[curr_inst_key].keys():
                plot_num_ele_per_loc.append(temp_dist)
                plot_num_ele_per_loc_count.append(num_ele_per_loc[curr_inst_key][temp_dist])

            for temp_dist in num_buc_per_loc[curr_inst_key].keys():
                plot_num_buc_per_loc.append(temp_dist)
                plot_num_buc_per_loc_count.append(num_buc_per_loc[curr_inst_key][temp_dist])     


            split_key=curr_inst_key.split(";")
            label_key=split_key[4]+";"+split_key[1]+";"+split_key[2]   

            print(split_key) 

            plt.plot(plot_num_ele_per_loc,plot_num_ele_per_loc_count,marker="o",markersize=10, lw=4.5, label=label_key, color=hash_color_dict[split_key[4]])   
            

        

            
        plt.xlabel('Num Elements',fontsize=20,weight='bold')
        plt.legend(fontsize=15)
        #plt.xlim(0.9, 100000)
        # plt.xlabel( _metric + ' ' + subopt_type)
        plt.yscale('log')
        plt.ylabel('Count',fontsize=20,weight='bold')
        # plt.ylabel('95th Percentile Suboptimality')
        #plt.ylim(0.01, 100000)
        # plt.ylim(0.00005,1.5)
        plt.yticks(fontsize=36,weight='bold')
        plt.xticks(fontsize=36,weight='bold')
        # template_name=all_QTs[0].split('/')[-1]
        savefilename = "expt_2_Chained"+"_"+dataset_key+'_num_ele_per_loc.png'
        # savepath = path.join(base_path, 'charts/' + savefilename)
        # plt.title("Geometric mean Suboptimality on a sequence of queries")
        # plt.title("Tail Suboptimality on a sequence of queries")
        plt.tight_layout()
        plt.savefig("figures/"+savefilename)
        plt.close()
        plt.clf()






#For each Cuckoo (dataset), plot different (hash functions, scheme config) primary key ratio plots
def expt_2_cuckoo():

    config_dict={}
    instance_dict={}
    dataset_dict={}
    scheme_dict={}
    succ_query_dict={}

    primary_key_ratio={}

    
    # print("start")

    with open(file_str_stats) as f:
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
                if "Cuckoo" in scheme_key:
                    scheme_key=line.split(" ")[6]+"_"+line.split(" ")[7]
                bucket_size_key=line.split(" ")[2]
                overalloc_key=line.split(" ")[3]
                # hash_func_key=hash_mapping_dict[line.split(" ")[4]]
                hash_func_key=line.split(" ")[4]
                model_error_key=line.split(" ")[-1]
                model_num_key=line.split(" ")[-2]
                config_key=scheme_key+";"+bucket_size_key+";"+overalloc_key

                if "RMI" in hash_func_key :
                    if int(model_num_key)<10000:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"

                if "Radix" in hash_func_key:
                    if int(model_error_key)<100:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"        

                scheme_dict[scheme_key]=1

                
                # print(line)
                # print(config_key)
                continue

            # if scheme_key!="Linear":
            #     continue

            if "Dataset Size: " in line:
                dataset_key=line.split(" ")[-1]
                dataset_dict[dataset_key]=1    

            # if "Num Elements: " in line:
            #     split_line=line.split(" ")
            #     if abs(int(split_line[4]))<100:
            #         continue
            #     primary_key_ratio[]    

            # if "Distance From Ideal:" in line:
            #     split_line=line.split(" ")
            #     if int(split_line[3])>0:
            #         continue
            #     if abs(int(split_line[3]))>5:
            #         continue
            #     temp_distance_from_ideal[abs(int(split_line[3]))]=int(split_line[5])    

            if " Primary Key Ratio:" in line:
                instance_key=scheme_key+";"+bucket_size_key+";"+overalloc_key+";"+dataset_key+";"+hash_func_key
                split_line=line.split(" ")
                primary_key_ratio[instance_key]=float(split_line[-1])
                # num_buc_per_loc[instance_key]=temp_num_buc_per_loc
                # temp_num_ele_per_loc={}
                # temp_num_buc_per_loc={}

    # print(distance_from_ideal)
    # print(distance_to_empty)
    key_to_plot="Cuckoo_Biased;4;15;"
    for dataset_key in dataset_dict.keys():

        done_dict={}           

        plt.figure(figsize=(14,12))

        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 20}

        matplotlib.rc('font', **font)
        color_list=['blue','green','red',"orange","black"] 

        
        
        for temp_inst_keys in primary_key_ratio.keys():

            if temp_inst_keys in done_dict.keys():
                continue
            if key_to_plot not in temp_inst_keys:
                continue
            if dataset_key not in temp_inst_keys:
                continue

            curr_inst_key=temp_inst_keys
            done_dict[curr_inst_key]=1

            # plot_num_ele_per_loc=[]
            # plot_num_ele_per_loc_count=[]

            # plot_num_buc_per_loc=[]
            # plot_num_buc_per_loc_count=[]

            # for temp_dist in num_ele_per_loc[curr_inst_key].keys():
            #     plot_num_ele_per_loc.append(temp_dist)
            #     plot_num_ele_per_loc_count.append(num_ele_per_loc[curr_inst_key][temp_dist])

            # for temp_dist in num_buc_per_loc[curr_inst_key].keys():
            #     plot_num_buc_per_loc.append(temp_dist)
            #     plot_num_buc_per_loc_count.append(num_buc_per_loc[curr_inst_key][temp_dist])     


            split_key=curr_inst_key.split(";")
            label_key=split_key[4]+";"+split_key[1]+";"+split_key[2]   

            print(split_key) 

            plot_prim_key=[0]
            plot_prim_key_val=primary_key_ratio[curr_inst_key]

            plt.plot(plot_prim_key,plot_prim_key_val,marker="o",markersize=10, lw=4.5, label=label_key, color=hash_color_dict[split_key[4]])   
            

        

            
        plt.xlabel('None',fontsize=20,weight='bold')
        plt.legend(fontsize=15)
        #plt.xlim(0.9, 100000)
        # plt.xlabel( _metric + ' ' + subopt_type)
        # plt.yscale('log')
        plt.ylabel('Primary Key Ratio',fontsize=20,weight='bold')
        # plt.ylabel('95th Percentile Suboptimality')
        #plt.ylim(0.01, 100000)
        # plt.ylim(0.00005,1.5)
        plt.yticks(fontsize=36,weight='bold')
        plt.xticks(fontsize=36,weight='bold')
        # template_name=all_QTs[0].split('/')[-1]
        savefilename = "expt_2_Cuckoo"+"_"+dataset_key+'_primary_key_ratio.png'
        # savepath = path.join(base_path, 'charts/' + savefilename)
        # plt.title("Geometric mean Suboptimality on a sequence of queries")
        # plt.title("Tail Suboptimality on a sequence of queries")
        plt.tight_layout()
        plt.savefig("figures/"+savefilename)
        plt.close()
        plt.clf()





#For each Dataset, plot distribution of the gaps plots
def collision_expt_1_gaps():

    dataset_dict={}
    temp_stats_dict={}

    # print("start")

    with open(file_gap_stats) as f:
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

                if "RMI" in hash_func_key :
                    if int(model_num_key)<10000:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"

                if "Radix" in hash_func_key:
                    if int(model_error_key)<100:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"        

                continue

            if "Dataset Size: " in line:
                dataset_key=line.split(" ")[-1]
                dataset_dict[dataset_key]=1    

            if "Gap Stats:" in line:
                split_line=line.split(" ")
                if abs(int(split_line[2]))>200:
                    continue
                if abs(int(split_line[4]))<100000:
                    continue    
                temp_stats_dict[abs(int(split_line[2]))]=float(int(split_line[4]))


            if "End Gap Stats" in line:
                dataset_dict[dataset_key]=temp_stats_dict
                temp_stats_dict={}

    # print(dataset_dict.keys())

    plt.figure(figsize=(8,6))

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

    matplotlib.rc('font', **font)
    color_list=['blue','green','red',"orange","black","purple"] 

    color_count=0

    for dataset_key in dataset_dict.keys():

        plot_gap=[]
        plot_gap_count=[]

        for temp_gap in dataset_dict[dataset_key].keys():
            plot_gap.append(float(temp_gap)*0.01)
            plot_gap_count.append(dataset_dict[dataset_key][temp_gap])

        for item in range(0,len(plot_gap_count)):
            plot_gap_count[item]=plot_gap_count[item]*(math.pow(10,8)*1.00/sum(plot_gap_count))


        plt.plot(plot_gap,plot_gap_count,marker='o',markersize=5, lw=1.5, label=dataset_key, color=color_list[color_count])    
        color_count+=1

    plt.xlabel('Normalized Gap',fontsize=20,weight='bold')
    plt.legend(fontsize=20)
    #plt.xlim(0.9, 100000)
    # plt.xlabel( _metric + ' ' + subopt_type)
    plt.yscale('log')
    plt.ylabel('Count',fontsize=20,weight='bold')
    # plt.ylabel('95th Percentile Suboptimality')
    #plt.ylim(0.01, 100000)
    # plt.ylim(0.00005,1.5)
    plt.yticks(fontsize=15,weight='bold')
    plt.xticks(fontsize=15,weight='bold')
    # template_name=all_QTs[0].split('/')[-1]
    savefilename = "collision_expt_1_gaps.png"
    # savepath = path.join(base_path, 'charts/' + savefilename)
    # plt.title("Geometric mean Suboptimality on a sequence of queries")
    # plt.title("Tail Suboptimality on a sequence of queries")
    plt.tight_layout()
    plt.savefig("figures/"+savefilename)
    plt.close()
    plt.clf()



#For each dataset, plot colliding keys w.r.t increasinng number of slots
def collision_expt_2_variance():

    config_dict={}
    instance_dict={}
    dataset_dict={}
    scheme_dict={}
    succ_query_dict={}

    num_ele_per_loc={}
    num_buc_per_loc={}

    temp_num_ele_per_loc={}
    temp_num_buc_per_loc={}

    dataset_slots_with_collision_dict={}
    dataset_overalloc_dict={}

    # print("start")

    with open(file_variance_stats) as f:
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

                if "RMI" in hash_func_key :
                    if int(model_num_key)<10000:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"

                if "Radix" in hash_func_key:
                    if int(model_error_key)<100:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"        

                scheme_dict[scheme_key]=1

                
                # print(line)
                # print(config_key)
                continue

            # if scheme_key!="Linear":
            #     continue

            if "Dataset Size: " in line:
                dataset_key=line.split(" ")[-1]
                dataset_dict[dataset_key]=1    

            if "Num Elements: " in line:
                split_line=line.split(" ")
                # if abs(int(split_line[4]))<100:
                #     continue
                # if abs(int(split_line[2]))>5:
                #     continue    
                temp_num_ele_per_loc[abs(int(split_line[2]))]=int(split_line[4])
                temp_num_buc_per_loc[abs(int(split_line[2]))]=math.ceil(int(split_line[4])*1.00/int(bucket_size_key))

            # if "Distance From Ideal:" in line:
            #     split_line=line.split(" ")
            #     if int(split_line[3])>0:
            #         continue
            #     if abs(int(split_line[3]))>5:
            #         continue
            #     temp_distance_from_ideal[abs(int(split_line[3]))]=int(split_line[5])    

            if "CollisionStats<" in line:
                overalloc_key=int(overalloc_key)
                if overalloc_key<10000:
                    overalloc_key=1.00+(overalloc_key*1.00/100.00)
                else:
                    overalloc_key=(overalloc_key-10000)*1.00/100.0

                if dataset_key not in dataset_slots_with_collision_dict.keys():
                    dataset_slots_with_collision_dict[dataset_key]=[]

                if dataset_key not in dataset_overalloc_dict.keys():    
                    dataset_overalloc_dict[dataset_key]=[]
                
                temp_sum=0
                temp_num_ele=0
                for dist_key in temp_num_ele_per_loc.keys():
                    temp_sum+=temp_num_ele_per_loc[dist_key]
                    temp_num_ele+=(dist_key*temp_num_ele_per_loc[dist_key])

                # ans=(temp_sum-(temp_num_ele_per_loc[1]))*1.00/temp_sum
                # ans=((temp_num_ele_per_loc[0]))*1.00/temp_sum
                ans=(temp_num_ele-(temp_num_ele_per_loc[1]))*1.00/temp_num_ele
                print(dataset_key,overalloc_key,ans)
                print(temp_num_ele_per_loc)
                dataset_slots_with_collision_dict[dataset_key].append(ans)
                dataset_overalloc_dict[dataset_key].append(overalloc_key)    

                temp_num_ele_per_loc={}



                
    print(dataset_slots_with_collision_dict)
    print(dataset_overalloc_dict)
    key_to_plot="Chained;1;"
    plt.figure(figsize=(8,6))

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

    matplotlib.rc('font', **font)
    color_list=['blue','green','red',"orange","black"] 
    color_count=0    


    for dataset_key in dataset_overalloc_dict.keys():

        plt.plot(dataset_overalloc_dict[dataset_key],dataset_slots_with_collision_dict[dataset_key],marker='o',markersize=5 ,lw=2, label=dataset_key, color=color_list[color_count])
        color_count+=1    
            

        

            
    plt.xlabel('Num Slots/Num Keys',fontsize=20,weight='bold')
    plt.legend(fontsize=15)
    #plt.xlim(0.9, 100000)
    # plt.xlabel( _metric + ' ' + subopt_type)
    # plt.xscale('log')
    plt.ylabel('Proportion of Colliding Keys',fontsize=20,weight='bold')
    # plt.ylabel('95th Percentile Suboptimality')
    #plt.ylim(0.01, 100000)
    # plt.ylim(0.00005,1.5)
    plt.yticks(fontsize=18,weight='bold')
    plt.xticks(fontsize=18,weight='bold')
    # template_name=all_QTs[0].split('/')[-1]
    savefilename = "collision_expt_2_variance.png"
    # savepath = path.join(base_path, 'charts/' + savefilename)
    # plt.title("Geometric mean Suboptimality on a sequence of queries")
    # plt.title("Tail Suboptimality on a sequence of queries")
    plt.tight_layout()
    plt.savefig("figures/"+savefilename)
    plt.close()
    plt.clf()


#For each dataset, plot collisions with icreasing model size
def collision_expt_3_modelsize():

    config_dict={}
    instance_dict={}
    dataset_dict={}
    scheme_dict={}
    succ_query_dict={}

    num_ele_per_loc={}
    num_buc_per_loc={}

    temp_num_ele_per_loc={}
    temp_num_buc_per_loc={}

    main_slots_with_collision_dict={}
    main_model_size_dict={}

    # print("start")

    with open(file_model_size_stats) as f:
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

                if "RMI" in hash_func_key :
                    if int(model_num_key)<10000:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"

                if "Radix" in hash_func_key:
                    if int(model_error_key)<100:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"        

                scheme_dict[scheme_key]=1

                
                # print(line)
                # print(config_key)
                continue

            # if scheme_key!="Linear":
            #     continue

            if "Dataset Size: " in line:
                dataset_key=line.split(" ")[-1]
                dataset_dict[dataset_key]=1    

            if "Num Elements: " in line:
                split_line=line.split(" ")
                # if abs(int(split_line[4]))<100:
                #     continue
                # if abs(int(split_line[2]))>5:
                #     continue    
                temp_num_ele_per_loc[abs(int(split_line[2]))]=int(split_line[4])
                temp_num_buc_per_loc[abs(int(split_line[2]))]=math.ceil(int(split_line[4])*1.00/int(bucket_size_key))

            # if "Distance From Ideal:" in line:
            #     split_line=line.split(" ")
            #     if int(split_line[3])>0:
            #         continue
            #     if abs(int(split_line[3]))>5:
            #         continue
            #     temp_distance_from_ideal[abs(int(split_line[3]))]=int(split_line[5])    

            if "CollisionStats<" in line:
                overalloc_key=int(overalloc_key)
                if overalloc_key<10000:
                    overalloc_key=1.00+(overalloc_key*1.00/100.00)
                else:
                    overalloc_key=(overalloc_key-10000)*1.00/100.0

                temp_split_key=hash_func_key.split("_")
                temp_key=dataset_key+";"+temp_split_key[0]+temp_split_key[1]    

                if temp_key not in main_slots_with_collision_dict.keys():
                    main_slots_with_collision_dict[temp_key]=[]

                if temp_key not in main_model_size_dict.keys():    
                    main_model_size_dict[temp_key]=[]
                
                temp_sum=0
                temp_num_ele=0
                for dist_key in temp_num_ele_per_loc.keys():
                    temp_sum+=temp_num_ele_per_loc[dist_key]
                    temp_num_ele+=(dist_key*temp_num_ele_per_loc[dist_key])
    

                # ans=(temp_sum-(temp_num_ele_per_loc[1]))*1.00/temp_sum
                # ans=((temp_num_ele_per_loc[0]))*1.00/temp_sum
                ans=(temp_num_ele-(temp_num_ele_per_loc[1]))*1.00/temp_num_ele
                print(temp_key,overalloc_key,ans)
                print(temp_num_ele_per_loc)
                main_slots_with_collision_dict[temp_key].append(ans)
                main_model_size_dict[temp_key].append(int(model_num_key))    

                temp_num_ele_per_loc={}



                
    print(main_slots_with_collision_dict)
    print(main_model_size_dict)
    key_to_plot="Chained;1;"
      


    for dataset_key in dataset_dict.keys():

        plt.figure(figsize=(8,6))

        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 20}

        matplotlib.rc('font', **font)
        color_list=['blue','green','red',"orange","black"] 
        color_count=0  

        done_dict={}
        for temp_key in main_slots_with_collision_dict.keys():

            if dataset_key not in temp_key:
                continue 

            if temp_key in done_dict.keys():
                continue    

            label_key=temp_key.split(";")[-1]
    
            plt.plot(main_model_size_dict[temp_key],main_slots_with_collision_dict[temp_key],marker='o',markersize=5 ,lw=2, label=label_key, color=color_list[color_count])
            color_count+=1    
            

        plt.xlabel('Model Size',fontsize=20,weight='bold')
        plt.legend(fontsize=15)
        #plt.xlim(0.9, 100000)
        # plt.xlabel( _metric + ' ' + subopt_type)
        plt.xscale('log')
        plt.ylabel('Proportion of Colliding Keys',fontsize=20,weight='bold')
        # plt.ylabel('95th Percentile Suboptimality')
        plt.ylim(-0.10, 1.1)
        # plt.ylim(0.00005,1.5)
        plt.yticks(fontsize=18,weight='bold')
        plt.xticks(fontsize=18,weight='bold')
        # template_name=all_QTs[0].split('/')[-1]
        savefilename = "collision_expt_3_model_size_"+str(dataset_key)+".png"
        # savepath = path.join(base_path, 'charts/' + savefilename)
        # plt.title("Geometric mean Suboptimality on a sequence of queries")
        # plt.title("Tail Suboptimality on a sequence of queries")
        plt.tight_layout()
        plt.savefig("figures/"+savefilename)
        plt.close()
        plt.clf()



#For each (scheme, dataset), plot different (hash functions, scheme config) performance
def expt_3():

    config_dict={}
    instance_dict={}
    dataset_dict={}
    scheme_dict={}
    succ_query_dict={}
    overalloc_dict={}

    # print("start")

    with open(file_str_latency) as f:
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
                if "Cuckoo" in scheme_key:
                    scheme_key=line.split(" ")[6]+"_"+line.split(" ")[7]
                bucket_size_key=line.split(" ")[2]
                overalloc_key=line.split(" ")[3]
                if int(overalloc_key)>10000:
                    overalloc_key=str(int(overalloc_key)-10000)
                else:
                    overalloc_key=str(int(overalloc_key)+100) 
                overalloc_dict[overalloc_key]=1       
                # hash_func_key=hash_mapping_dict[line.split(" ")[4]]
                hash_func_key=line.split(" ")[4]
                
                model_error_key=line.split(" ")[-1]
                model_num_key=line.split(" ")[-2]
                config_key=scheme_key+";"+bucket_size_key+";"+overalloc_key

                if "RMI" in hash_func_key :
                    if int(model_num_key)<10:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"

                if "Radix" in hash_func_key:
                    if int(model_error_key)<5000:
                        hash_func_key=hash_func_key+"Small"
                    else:
                        hash_func_key=hash_func_key+"Big"        

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

            plt.figure(figsize=(14,12))

            font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 20}

            matplotlib.rc('font', **font)
            color_list=['blue','green','red',"orange","black"] 

           
            
            for temp_inst_keys in instance_dict.keys():

                # while(temp_inst_keys[-1]!=";"):
                #     temp_inst_keys=temp_inst_keys[:-1]

                if temp_inst_keys in done_dict.keys():
                    continue
                # if ";1;100" not in temp_inst_keys and "Cuckoo" not in temp_inst_keys and "MWHC" not in temp_inst_keys:
                #     continue    

                # if "Cuckoo"  in temp_inst_keys and ";4;15" not in temp_inst_keys:
                #     continue    

                # if "MWHC" not in temp_inst_keys:
                #     continue    

                if "Radix" in temp_inst_keys:
                    continue

                if scheme_key not in temp_inst_keys:
                    continue
                if dataset_key not in temp_inst_keys:
                    continue

                print("Here in the loop: ")    

                curr_inst_key=temp_inst_keys
                done_dict[curr_inst_key]=1

                latency_list=[]
                over_alloc_list=list(overalloc_dict.keys())
                if "MWHC" in curr_inst_key:
                    over_alloc_list=[110,120]
                else:    
                    over_alloc_list=[50,75,150,200,400]
                # print(succ_key_list)
                for i in range(0,len(over_alloc_list)):
                    
                    over_alloc_list[i]=int(over_alloc_list[i])
                over_alloc_list.sort()

                # for temp_iter in range(0,len(over_alloc_list)):
                #     if over_alloc_list[temp_iter]<=100:
                #         over_alloc_list[temp_iter]=150


                for overalloc_key in over_alloc_list:


                    split_key=curr_inst_key.split(";")
                    # label_key=split_key[4]+";"+split_key[1]+";"+split_key[2]   
                    
                    # if overalloc_key<100:
                    #     temp_key=split_key[0]+";"+split_key[1]+";"+str(200)
                    #     temp_key+=";"+split_key[3]+";"+split_key[4]+";"+split_key[5]
                    # else:
                    temp_key=split_key[0]+";"+split_key[1]+";"+str(overalloc_key)
                    temp_key+=";"+split_key[3]+";"+split_key[4]+";"+split_key[5]

                    # if("MWHC" in curr_inst_key):
                    #     temp_key=curr_inst_key+"100"

                    # print("compare",curr_inst_key,temp_key)

                    done_dict[temp_key]=1

                    # print(temp_key,instance_dict[temp_key])

                    latency_list.append(min(instance_dict[temp_key]))

                split_key=curr_inst_key.split(";")
                label_key=split_key[4] 

                print(split_key) 

                plt.plot(over_alloc_list,latency_list,marker="o",markersize=10, lw=4.5, label=label_key, color=hash_color_dict[split_key[4]])   
                

           

             
            plt.xlabel('OverAlloc',fontsize=20,weight='bold')
            plt.legend(fontsize=15)
            #plt.xlim(0.9, 100000)
            # plt.xlabel( _metric + ' ' + subopt_type)
            # plt.yscale('log')
            plt.ylabel('Probe Latency(in ns)',fontsize=20,weight='bold')
            # plt.ylabel('95th Percentile Suboptimality')
            #plt.ylim(0.01, 100000)
            # plt.ylim(0.00005,1.5)
            plt.yticks(fontsize=36,weight='bold')
            plt.xticks(fontsize=36,weight='bold')
            # template_name=all_QTs[0].split('/')[-1]
            savefilename = "expt_3_"+scheme_key+"_"+dataset_key+'.png'
            # savepath = path.join(base_path, 'charts/' + savefilename)
            # plt.title("Geometric mean Suboptimality on a sequence of queries")
            # plt.title("Tail Suboptimality on a sequence of queries")
            plt.tight_layout()
            plt.savefig("figures/"+savefilename)
            plt.close()
            plt.clf()
 



# probe_latencies()
# expt_1()
# expt_2_linear_dist_from_ideal()
# expt_2_linear_dist_to_empty()
# expt_2_chained()
# expt_2_cuckoo()
expt_3()

# collision_expt_1_gaps()
# collision_expt_2_variance()
# collision_expt_3_modelsize()