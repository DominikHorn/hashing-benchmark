# bucket_size=int(sys.argv[1])
# overalloc=int(sys.argv[2])
# model_name=str(sys.argv[3])
# model_type=str(sys.argv[4])
# hashing_scheme=str(sys.argv[5])
# kickinit_strat=str(sys.argv[6])
# kickinit_strat_bias=int(sys.argv[7])
# max_models=int(sys.argv[8])
# max_error=int(sys.argv[9])

# "MURMUR":"using MURMUR = hashing::MurmurFinalizer<Key>;",
#     "MultPrime64":"using MultPrime64 = hashing::MultPrime64;",
#     "FibonacciPrime64":"using FibonacciPrime64 = hashing::FibonacciPrime64;",
#     "AquaHash":"using AquaHash = hashing::AquaHash<Key>;",
#     "XXHash3":"using XXHash3 = hashing::XXHash3<Key>;",
#     "MWHC":"using MWHC = exotic_hashing::MWHC<Key>;",
#     "FST":"using FST = exotic_hashing::FastSuccinctTrie<Data>;",
#     "RadixSplineHash":"using RadixSplineHash = learned_hashing::RadixSplineHash<std::uint64_t,max_error,max_models>;",
#     "RMIHash":"using RMIHash = learned_hashing::RMIHash<std::uint64_t,max_models>;"
# python3 kapil_python_edit_script.py 1 20 "MWHC" "Exotic" "Chained" "Balanced" 0 0 0

#  python3 kapil_python_edit_script.py 1 100 "RMIHash" "Model" "Chained" "Balanced" 0 100 1024 

# python3 kapil_python_edit_script.py 4 15 "MURMUR" "Traditional" "Cuckoo" "Biased" 90 0 0

###########################CHAINED############################
###########################CHAINED############################
###########################CHAINED############################
#Traditional Chained Experiments 

for bucket_size in 1 
do
    for overalloc in 0 50 100 300 10050 10075
    do
        for model_name in "MURMUR" "MultPrime64" "XXHash3"
        do
            echo "Start Here" $bucket_size $overalloc $model_name "Traditional" "Chained" "Balanced" 0 0 0 >>kapil_results.json
            echo "Start Here" $bucket_size $overalloc $model_name "Traditional" "Chained" "Balanced" 0 0 0  >>data_stats_mar14.out
            python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Traditional" "Chained" "Balanced" 0 0 0
            bash run.sh >>data_stats_mar14.out
            cat benchmark_results.json >>kapil_results.json
        done
    done
done


#Model Chained Experiments 

for bucket_size in 1 
do
    for overalloc in 0 50 100 300 10050 10075
    do
        for model_name in  "RMIHash" "RadixSplineHash" 
        do
            echo "Start Here" $bucket_size $overalloc $model_name "Model" "Chained" "Balanced" 0 1 10024 >>kapil_results.json
            echo "Start Here" $bucket_size $overalloc $model_name "Model" "Chained" "Balanced" 0 1 10024 >>data_stats_mar14.out
            python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Model" "Chained" "Balanced" 0 1 10024
            bash run.sh >>data_stats_mar14.out
            cat benchmark_results.json >>kapil_results.json
        done
    done
done

for bucket_size in 1 
do
    for overalloc in 0 50 100 300 10050 10075
    do
        for model_name in "RMIHash" "RadixSplineHash" 
        do
            echo "Start Here" $bucket_size $overalloc $model_name "Model" "Chained" "Balanced" 0 100 1024 >>kapil_results.json
            echo "Start Here" $bucket_size $overalloc $model_name "Model" "Chained" "Balanced" 0 100 1024 >>data_stats_mar14.out
            python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Model" "Chained" "Balanced" 0 100 1024
            bash run.sh >>data_stats_mar14.out
            cat benchmark_results.json >>kapil_results.json
        done
    done
done


#Exotic Chained Experiments 

for bucket_size in 1
do
    for overalloc in 10 20
    do
        for model_name in "MWHC" 
        do
            echo "Start Here" $bucket_size $overalloc $model_name "Exotic" "Chained" "Balanced" 0 0 0 >>kapil_results.json
            echo "Start Here" $bucket_size $overalloc $model_name "Exotic" "Chained" "Balanced" 0 0 0 >>data_stats_mar14.out
            python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Exotic" "Chained" "Balanced" 0 0 0
            bash run.sh >>data_stats_mar14.out
            cat benchmark_results.json >>kapil_results.json
        done
    done
done




###########################LINEAR############################
###########################LINEAR############################
###########################LINEAR############################


#Model Linear Experiments 

# python3 kapil_python_edit_script.py 1 50 RMIHash Model Linear Balanced 0 1000 1024

for bucket_size in 1 
do
    for overalloc in 50 100 200 300 
    do
        for model_name in "RMIHash"  "RadixSplineHash" 
        do
            echo "Start Here" $bucket_size $overalloc $model_name "Model" "Linear" "Balanced" 0 1 10024 >>kapil_results.json
            echo "Start Here" $bucket_size $overalloc $model_name "Model" "Linear" "Balanced" 0 1 10024 >>data_stats_mar14.out
            python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Model" "Linear" "Balanced" 0 1 10024
            bash run.sh >>data_stats_mar14.out
            cat benchmark_results.json >>kapil_results.json
        done
    done
done

for bucket_size in 1
do
    for overalloc in 50 100 200 300 
    do
        for model_name in "RMIHash" "RadixSplineHash" 
        do
            echo "Start Here" $bucket_size $overalloc $model_name "Model" "Linear" "Balanced" 0 100 1024 >>kapil_results.json
            echo "Start Here" $bucket_size $overalloc $model_name "Model" "Linear" "Balanced" 0 100 1024 >>data_stats_mar14.out
            python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Model" "Linear" "Balanced" 0 100 1024
            bash run.sh >>data_stats_mar14.out
            cat benchmark_results.json >>kapil_results.json
        done
    done
done


#Traditional Linear Experiments 

for bucket_size in 1
do
    for overalloc in 50 100 200 300 
    do
        for model_name in "MURMUR" "MultPrime64" "XXHash3"
        do
            echo "Start Here" $bucket_size $overalloc $model_name "Traditional" "Linear" "Balanced" 0 0 0 >>kapil_results.json
            echo "Start Here" $bucket_size $overalloc $model_name "Traditional" "Linear" "Balanced" 0 0 0 >>data_stats_mar14.out
            python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Traditional" "Linear" "Balanced" 0 0 0
            bash run.sh >>data_stats_mar14.out
            cat benchmark_results.json >>kapil_results.json
        done
    done
done



###########################Cuckoo############################
###########################Cuckoo############################
###########################Cuckoo############################



#Model Cuckoo Experiments 

# for bucket_size in 4 8
# do
#     for overalloc in 15 30 
#     do
#         for model_name in "RMIHash"  "RadixSplineHash" 
#         do
#             echo "Start Here" $bucket_size $overalloc $model_name "Model" "Cuckoo" "Balanced" 0 1000 1024 >>kapil_results.json
#             echo "Start Here" $bucket_size $overalloc $model_name "Model" "Cuckoo" "Balanced" 0 1000 1024 >>data_stats_mar14.out
#             python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Model" "Cuckoo" "Balanced" 0 1000 1024
#             bash run.sh >>data_stats_mar14.out
#             cat benchmark_results.json >>kapil_results.json
#         done
#     done
# done

# for bucket_size in 4 8
# do
#     for overalloc in 15 30
#     do
#         for model_name in "RMIHash" "RadixSplineHash" 
#         do
#             echo "Start Here" $bucket_size $overalloc $model_name "Model" "Cuckoo" "Balanced" 0 100000 32 >>kapil_results.json
#             echo "Start Here" $bucket_size $overalloc $model_name "Model" "Cuckoo" "Balanced" 0 100000 32 >>data_stats_mar14.out
#             python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Model" "Cuckoo" "Balanced" 0 100000 32
#             bash run.sh >>data_stats_mar14.out
#             cat benchmark_results.json >>kapil_results.json
#         done
#     done
# done

#Traditional Cuckoo Experiments 


# for bucket_size in 4 8
# do
#     for overalloc in 15 30
#     do
#         for model_name in "MURMUR" "MultPrime64"  "XXHash3"
#         do
#             echo "Start Here" $bucket_size $overalloc $model_name "Traditional" "Cuckoo" "Balanced" 0 0 0 >>kapil_results.json
#             echo "Start Here" $bucket_size $overalloc $model_name "Traditional" "Cuckoo" "Balanced" 0 0 0 >>data_stats_mar14.out
#             python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Traditional" "Cuckoo" "Balanced" 0 0 0
#             bash run.sh >>data_stats_mar14.out
#             cat benchmark_results.json >>kapil_results.json
#         done
#     done
# done


#Model Cuckoo Experiments 

# for bucket_size in 4 8
# do
#     for overalloc in 15 30 
#     do
#         for model_name in "RMIHash"  "RadixSplineHash" 
#         do
#             echo "Start Here" $bucket_size $overalloc $model_name "Model" "Cuckoo" "Biased" 5 1000 1024 >>kapil_results.json
#             echo "Start Here" $bucket_size $overalloc $model_name "Model" "Cuckoo" "Biased" 5 1000 1024 >>data_stats_mar14.out
#             python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Model" "Cuckoo" "Biased" 5 1000 1024
#             bash run.sh >>data_stats_mar14.out
#             cat benchmark_results.json >>kapil_results.json
#         done
#     done
# done

# for bucket_size in 4 8
# do
#     for overalloc in 15 30
#     do
#         for model_name in "RMIHash" "RadixSplineHash" 
#         do
#             echo "Start Here" $bucket_size $overalloc $model_name "Model" "Cuckoo" "Biased" 5 100000 32 >>kapil_results.json
#             echo "Start Here" $bucket_size $overalloc $model_name "Model" "Cuckoo" "Biased" 5 100000 32 >>data_stats_mar14.out
#             python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Model" "Cuckoo" "Biased" 5 100000 32
#             bash run.sh >>data_stats_mar14.out
#             cat benchmark_results.json >>kapil_results.json
#         done
#     done
# done

#Traditional Cuckoo Experiments 


# for bucket_size in 4 8
# do
#     for overalloc in 15 30
#     do
#         for model_name in "MURMUR" "MultPrime64"  "XXHash3"
#         do
#             echo "Start Here" $bucket_size $overalloc $model_name "Traditional" "Cuckoo" "Biased" 5 0 0 >>kapil_results.json
#             echo "Start Here" $bucket_size $overalloc $model_name "Traditional" "Cuckoo" "Biased" 5 0 0 >>data_stats_mar14.out
#             python3 kapil_python_edit_script.py $bucket_size $overalloc $model_name "Traditional" "Cuckoo" "Biased" 5 0 0
#             bash run.sh >>data_stats_mar14.out
#             cat benchmark_results.json >>kapil_results.json
#         done
#     done
# done