# A script to run the experiments for all join variants

#!/bin/sh

process_join()
{
    threads=(1) #(2 4 8 16 32 64)
    rmi_models=(0 1 2 3 4 5 6 7 8 9)
    css_fanouts=(33) #(10 33 40)

    #dataset_folder_path=/spinning/sabek/learned_join_datasets_tpch/
    #dataset_folder_path=/spinning/sabek/learned_join_datasets_sosd/
    #dataset_folder_path=/spinning/sabek/learned_join_datasets/
    dataset_folder_path=/spinning/sabek/learned_hash_datasets/  #for learned hash projects only (SOSD datasets)

    #r_datasets=$1
    #r_datasets_sizes=$2
    #r_datasets_file_num_partitions=$3
    r_datasets_file_extension='"'.txt'"'
    #s_datasets=$4
    #s_datasets_sizes=$5
    #s_datasets_file_num_partitions=$6
    s_datasets_file_extension='"'.txt'"'

    output_folder_path=$7
    mkdir $output_folder_path

    run_nums=$8

    load_relations_for_evaluation=$9
    persist_relations_for_evaluation=${10}

    run_inlj_with_hash_index=${11}
    run_inlj_with_learned_index=${12}
    run_inlj_with_csstree_index=${13}
    run_inlj_with_art32tree_index=${14}
    run_inlj_with_art64tree_index=${15}
    run_inlj_with_cuckoohash_index=${16}

    #input_hash_table_size=${17}

    if [ ${run_inlj_with_hash_index} == 1 ]
    then
        echo "Running INLJ with hash index ..."
        
        for ds in ${!r_datasets[@]}
        do
            curr_r_dataset='"'$dataset_folder_path${r_datasets[$ds]}.txt'"'
            curr_r_dataset_size=${r_datasets_sizes[$ds]}
            curr_r_dataset_file_num_partitions=${r_datasets_file_num_partitions[$ds]}
            curr_s_dataset='"'$dataset_folder_path${s_datasets[$ds]}.txt'"'
            curr_s_dataset_size=${s_datasets_sizes[$ds]}
            curr_s_dataset_file_num_partitions=${s_datasets_file_num_partitions[$ds]}

            curr_input_hash_table_size=${input_hash_table_size[$ds]}
            curr_hash_scheme_and_function_mode=${hash_scheme_and_function_mode[$ds]}
            curr_hash_fun=${hash_fun[$ds]}
            curr_hash_overalloc=${hash_overalloc[$ds]}
            curr_hash_learned_model=${hash_learned_model[$ds]}

            echo 'Joining '$curr_r_dataset' '$curr_r_dataset_size' '$curr_s_dataset' '$curr_s_dataset_size'... \n'
            
            for th in ${!threads[@]}
            do
                curr_threads=${threads[$th]}

                curr_output_file=$output_folder_path'non_imv_inlj_with_hash_index_tuning_'$curr_r_dataset_size'_'$curr_s_dataset_size'_th_'$curr_threads'_hts_'$curr_input_hash_table_size'_hsf_'$curr_hash_scheme_and_function_mode'_hf_'$curr_hash_fun'_ho_'$curr_hash_overalloc'_hlm_'$curr_hash_learned_model'.csv'
                
                sh $(dirname "$0")/base_configs_maker.sh -INLJ_WITH_HASH_INDEX 1 \
                                                -INLJ_WITH_LEARNED_INDEX 0 \
                                                -INLJ_WITH_CSS_TREE_INDEX 0 \
                                                -INLJ_WITH_ART32_TREE_INDEX 0 \
                                                -INLJ_WITH_ART64_TREE_INDEX 0 \
                                                -INLJ_WITH_CUCKOO_HASH_INDEX 0 \
                                                -HASH_SCHEME_AND_FUNCTION_MODE $curr_hash_scheme_and_function_mode \
                                                -HASH_FUN $curr_hash_fun \
                                                -HASH_OVERALLOC $curr_hash_overalloc \
                                                -HASH_LEARNED_MODEL $curr_hash_learned_model \
                                                -PREFETCH_INLJ 0 \
                                                -RUN_LEARNED_TECHNIQUES_WITH_FIRST_LEVEL_ONLY 1 \
                                                -NUM_THREADS_FOR_EVALUATION $curr_threads \
                                                -RELATION_R_PATH $curr_r_dataset \
                                                -RELATION_R_FOLDER_PATH '"'$dataset_folder_path'"' \
                                                -RELATION_R_FILE_NAME '"'${r_datasets[$ds]}'"' \
                                                -RELATION_R_FILE_EXTENSION ${r_datasets_file_extension} \
                                                -RELATION_R_NUM_TUPLES $curr_r_dataset_size \
                                                -RELATION_R_FILE_NUM_PARTITIONS $curr_r_dataset_file_num_partitions \
                                                -RELATION_S_PATH $curr_s_dataset \
                                                -RELATION_KEY_TYPE uint64_t \
                                                -RELATION_PAYLOAD_TYPE uint64_t \
                                                -RELATION_S_FOLDER_PATH '"'$dataset_folder_path'"' \
                                                -RELATION_S_FILE_NAME '"'${s_datasets[$ds]}'"' \
                                                -RELATION_S_FILE_EXTENSION ${s_datasets_file_extension} \
                                                -RELATION_S_NUM_TUPLES ${curr_s_dataset_size} \
                                                -RELATION_S_FILE_NUM_PARTITIONS ${curr_s_dataset_file_num_partitions} \
                                                -BENCHMARK_RESULTS_PATH '"'${curr_output_file}'"' \
                                                -RUN_NUMS ${run_nums} -LOAD_RELATIONS_FOR_EVALUATION ${load_relations_for_evaluation} \
                                                -PERSIST_RELATIONS_FOR_EVALUATION ${persist_relations_for_evaluation} #\
                                                #-CUSTOM_CPU_MAPPING '"'../../include/configs/cpu-mapping_berners_lee.txt'"' \
                                                #-CUSTOM_CPU_MAPPING_V2 '"'../../include/configs/cpu-mapping-v2_berners_lee.txt'"'

                sh $(dirname "$0")/eth_configs_maker.sh   -BUCKET_SIZE 4 \
                                                -PREFETCH_DISTANCE 128 \
                                                -USE_MURMUR3_HASH 1 \
                                                -INPUT_HASH_TABLE_SIZE $curr_input_hash_table_size

                cmake -DCMAKE_BUILD_TYPE=Release -DVECTORWISE_BRANCHING=on $(dirname "$0")/../.. > /dev/null

                cd $(dirname "$0")/../../build/release

                make > /dev/null

                ./npj_join_runner

                cd ../../scripts/evaluation/
            done
        done

    fi

    if [ ${run_inlj_with_learned_index} == 1 ]
    then
        echo "Running INLJ with learned index ..."

        for ds in ${!r_datasets[@]}
        do
            curr_r_dataset='"'$dataset_folder_path${r_datasets[$ds]}.txt'"'
            curr_r_dataset_size=${r_datasets_sizes[$ds]}
            curr_r_dataset_file_num_partitions=${r_datasets_file_num_partitions[$ds]}
            curr_s_dataset='"'$dataset_folder_path${s_datasets[$ds]}.txt'"'
            curr_s_dataset_size=${s_datasets_sizes[$ds]}
            curr_s_dataset_file_num_partitions=${s_datasets_file_num_partitions[$ds]}

            echo 'Joining '$curr_r_dataset' '$curr_r_dataset_size' '$curr_s_dataset' '$curr_s_dataset_size'... \n'
            
            for th in ${!threads[@]}
            do
                curr_threads=${threads[$th]}

                for model in  ${!rmi_models[@]}
                do
                    #curr_rmi_model=${r_datasets[$ds]}'_key_uint32_'${rmi_models[$model]}
                    curr_rmi_model=${r_datasets[$ds]}'_'${rmi_models[$model]}
                    
                    curr_output_file=$output_folder_path'non_imv_inlj_with_learned_index_tuning_'$curr_r_dataset_size'_'$curr_s_dataset_size'_th_'$curr_threads'_rmi_'${rmi_models[$model]}'.csv'
                
                    sh $(dirname "$0")/base_configs_maker.sh -INLJ_WITH_HASH_INDEX 0 \
                                                    -INLJ_WITH_LEARNED_INDEX 1 \
                                                    -INLJ_WITH_LEARNED_INDEX_MODEL_BASED_BUILD 0 \
                                                    -INLJ_WITH_LEARNED_GAPS_FACTOR 4 \
                                                    -INLJ_WITH_CSS_TREE_INDEX 0 \
                                                    -INLJ_WITH_ART32_TREE_INDEX 0 \
                                                    -INLJ_WITH_ART64_TREE_INDEX 0 \
                                                    -INLJ_WITH_CUCKOO_HASH_INDEX 0 \
                                                    -RUN_LEARNED_TECHNIQUES_WITH_FIRST_LEVEL_ONLY 1 \
                                                    -INLJ_RMI_DATA_PATH '"'/spinning/sabek/rmi_data'"' \
                                                    -INLJ_RMI_NAMESPACE ${curr_rmi_model} \
                                                    -NUM_THREADS_FOR_EVALUATION $curr_threads \
                                                    -RELATION_KEY_TYPE uint64_t \
                                                    -RELATION_PAYLOAD_TYPE uint64_t \
                                                    -RELATION_R_PATH $curr_r_dataset \
                                                    -RELATION_R_FOLDER_PATH '"'$dataset_folder_path'"' \
                                                    -RELATION_R_FILE_NAME '"'${r_datasets[$ds]}'"' \
                                                    -RELATION_R_FILE_EXTENSION ${r_datasets_file_extension} \
                                                    -RELATION_R_NUM_TUPLES $curr_r_dataset_size \
                                                    -RELATION_R_FILE_NUM_PARTITIONS $curr_r_dataset_file_num_partitions \
                                                    -RELATION_S_PATH $curr_s_dataset \
                                                    -RELATION_S_FOLDER_PATH '"'$dataset_folder_path'"' \
                                                    -RELATION_S_FILE_NAME '"'${s_datasets[$ds]}'"' \
                                                    -RELATION_S_FILE_EXTENSION ${s_datasets_file_extension} \
                                                    -RELATION_S_NUM_TUPLES ${curr_s_dataset_size} \
                                                    -RELATION_S_FILE_NUM_PARTITIONS ${curr_s_dataset_file_num_partitions} \
                                                    -BENCHMARK_RESULTS_PATH '"'${curr_output_file}'"' \
                                                    -RUN_NUMS ${run_nums} -LOAD_RELATIONS_FOR_EVALUATION ${load_relations_for_evaluation} \
                                                    -PERSIST_RELATIONS_FOR_EVALUATION ${persist_relations_for_evaluation} #\
                                                    #-CUSTOM_CPU_MAPPING '"'../../include/configs/cpu-mapping_berners_lee.txt'"' \
                                                    #-CUSTOM_CPU_MAPPING_V2 '"'../../include/configs/cpu-mapping-v2_berners_lee.txt'"'

                    cmake -DCMAKE_BUILD_TYPE=Release -DVECTORWISE_BRANCHING=on $(dirname "$0")/../.. > /dev/null

                    cd $(dirname "$0")/../../build/release

                    make > /dev/null

                    ./npj_join_runner

                    cd ../../scripts/evaluation/
                done
            done
        done

    fi

    if [ ${run_inlj_with_csstree_index} == 1 ]
    then
        echo "Running INLJ with css tree index ..."

        for ds in ${!r_datasets[@]}
        do
            curr_r_dataset='"'$dataset_folder_path${r_datasets[$ds]}.txt'"'
            curr_r_dataset_size=${r_datasets_sizes[$ds]}
            curr_r_dataset_file_num_partitions=${r_datasets_file_num_partitions[$ds]}
            curr_s_dataset='"'$dataset_folder_path${s_datasets[$ds]}.txt'"'
            curr_s_dataset_size=${s_datasets_sizes[$ds]}
            curr_s_dataset_file_num_partitions=${s_datasets_file_num_partitions[$ds]}

            echo 'Joining '$curr_r_dataset' '$curr_r_dataset_size' '$curr_s_dataset' '$curr_s_dataset_size'... \n'
            
            for th in ${!threads[@]}
            do
                curr_threads=${threads[$th]}

                for f in ${!css_fanouts[@]}
                do
                    curr_fanout=${css_fanouts[$f]}

                    curr_output_file=$output_folder_path'non_imv_inlj_with_csstree_index_tuning_'$curr_r_dataset_size'_'$curr_s_dataset_size'_th_'$curr_threads'_f_'$curr_fanout'.csv'

                    sh $(dirname "$0")/base_configs_maker.sh -INLJ_WITH_HASH_INDEX 0 \
                                                -INLJ_WITH_LEARNED_INDEX 0 \
                                                -INLJ_WITH_CSS_TREE_INDEX 1 \
                                                -INLJ_WITH_ART32_TREE_INDEX 0 \
                                                -INLJ_WITH_ART64_TREE_INDEX 0 \
                                                -INLJ_WITH_CUCKOO_HASH_INDEX 0 \
                                                -INLJ_CSS_TREE_FANOUT $curr_fanout \
                                                -NUM_THREADS_FOR_EVALUATION $curr_threads \
                                                -RELATION_R_PATH $curr_r_dataset \
                                                -RELATION_R_FOLDER_PATH '"'$dataset_folder_path'"' \
                                                -RELATION_R_FILE_NAME '"'${r_datasets[$ds]}'"' \
                                                -RELATION_R_FILE_EXTENSION ${r_datasets_file_extension} \
                                                -RELATION_R_NUM_TUPLES $curr_r_dataset_size \
                                                -RELATION_R_FILE_NUM_PARTITIONS $curr_r_dataset_file_num_partitions \
                                                -RELATION_S_PATH $curr_s_dataset \
                                                -RELATION_S_FOLDER_PATH '"'$dataset_folder_path'"' \
                                                -RELATION_S_FILE_NAME '"'${s_datasets[$ds]}'"' \
                                                -RELATION_S_FILE_EXTENSION ${s_datasets_file_extension} \
                                                -RELATION_S_NUM_TUPLES ${curr_s_dataset_size} \
                                                -RELATION_S_FILE_NUM_PARTITIONS ${curr_s_dataset_file_num_partitions} \
                                                -BENCHMARK_RESULTS_PATH '"'${curr_output_file}'"' \
                                                -RUN_NUMS ${run_nums} -LOAD_RELATIONS_FOR_EVALUATION ${load_relations_for_evaluation} \
                                                -PERSIST_RELATIONS_FOR_EVALUATION ${persist_relations_for_evaluation} #\
                                                #-CUSTOM_CPU_MAPPING '"'../../include/configs/cpu-mapping_berners_lee.txt'"' \
                                                #-CUSTOM_CPU_MAPPING_V2 '"'../../include/configs/cpu-mapping-v2_berners_lee.txt'"'

                    cmake -DCMAKE_BUILD_TYPE=Release -DVECTORWISE_BRANCHING=on $(dirname "$0")/../.. > /dev/null

                    cd $(dirname "$0")/../../build/release

                    make > /dev/null

                    ./npj_join_runner

                    cd ../../scripts/evaluation/

                done
                
            done
        done

    fi

    if [ ${run_inlj_with_art32tree_index} == 1 ]
    then
        echo "Running INLJ with ART32 tree index ..."

        for ds in ${!r_datasets[@]}
        do
            curr_r_dataset='"'$dataset_folder_path${r_datasets[$ds]}.txt'"'
            curr_r_dataset_size=${r_datasets_sizes[$ds]}
            curr_r_dataset_file_num_partitions=${r_datasets_file_num_partitions[$ds]}
            curr_s_dataset='"'$dataset_folder_path${s_datasets[$ds]}.txt'"'
            curr_s_dataset_size=${s_datasets_sizes[$ds]}
            curr_s_dataset_file_num_partitions=${s_datasets_file_num_partitions[$ds]}

            echo 'Joining '$curr_r_dataset' '$curr_r_dataset_size' '$curr_s_dataset' '$curr_s_dataset_size'... \n'
            
            for th in ${!threads[@]}
            do
                curr_threads=${threads[$th]}

                curr_output_file=$output_folder_path'non_imv_inlj_with_art32tree_index_tuning_'$curr_r_dataset_size'_'$curr_s_dataset_size'_th_'$curr_threads'.csv'

                sh $(dirname "$0")/base_configs_maker.sh -INLJ_WITH_HASH_INDEX 0 \
                                            -INLJ_WITH_LEARNED_INDEX 0 \
                                            -INLJ_WITH_CSS_TREE_INDEX 0 \
                                            -INLJ_WITH_ART32_TREE_INDEX 1 \
                                            -INLJ_WITH_ART64_TREE_INDEX 0 \
                                            -INLJ_WITH_CUCKOO_HASH_INDEX 0 \
                                            -NUM_THREADS_FOR_EVALUATION $curr_threads \
                                            -RELATION_R_PATH $curr_r_dataset \
                                            -RELATION_R_FOLDER_PATH '"'$dataset_folder_path'"' \
                                            -RELATION_R_FILE_NAME '"'${r_datasets[$ds]}'"' \
                                            -RELATION_R_FILE_EXTENSION ${r_datasets_file_extension} \
                                            -RELATION_R_NUM_TUPLES $curr_r_dataset_size \
                                            -RELATION_R_FILE_NUM_PARTITIONS $curr_r_dataset_file_num_partitions \
                                            -RELATION_S_PATH $curr_s_dataset \
                                            -RELATION_S_FOLDER_PATH '"'$dataset_folder_path'"' \
                                            -RELATION_S_FILE_NAME '"'${s_datasets[$ds]}'"' \
                                            -RELATION_S_FILE_EXTENSION ${s_datasets_file_extension} \
                                            -RELATION_S_NUM_TUPLES ${curr_s_dataset_size} \
                                            -RELATION_S_FILE_NUM_PARTITIONS ${curr_s_dataset_file_num_partitions} \
                                            -BENCHMARK_RESULTS_PATH '"'${curr_output_file}'"' \
                                            -RUN_NUMS ${run_nums} -LOAD_RELATIONS_FOR_EVALUATION ${load_relations_for_evaluation} \
                                            -PERSIST_RELATIONS_FOR_EVALUATION ${persist_relations_for_evaluation} \
                                            -CUSTOM_CPU_MAPPING '"'../../include/configs/cpu-mapping_berners_lee.txt'"' \
                                            -CUSTOM_CPU_MAPPING_V2 '"'../../include/configs/cpu-mapping-v2_berners_lee.txt'"'

                cmake -DCMAKE_BUILD_TYPE=Release -DVECTORWISE_BRANCHING=on $(dirname "$0")/../.. > /dev/null

                cd $(dirname "$0")/../../build/release

                make > /dev/null

                ./npj_join_runner

                cd ../../scripts/evaluation/
            done
        done

    fi

    if [ ${run_inlj_with_art64tree_index} == 1 ]
    then
        echo "Running INLJ with ART64 tree index ..."

        for ds in ${!r_datasets[@]}
        do
            curr_r_dataset='"'$dataset_folder_path${r_datasets[$ds]}.txt'"'
            curr_r_dataset_size=${r_datasets_sizes[$ds]}
            curr_r_dataset_file_num_partitions=${r_datasets_file_num_partitions[$ds]}
            curr_s_dataset='"'$dataset_folder_path${s_datasets[$ds]}.txt'"'
            curr_s_dataset_size=${s_datasets_sizes[$ds]}
            curr_s_dataset_file_num_partitions=${s_datasets_file_num_partitions[$ds]}

            echo 'Joining '$curr_r_dataset' '$curr_r_dataset_size' '$curr_s_dataset' '$curr_s_dataset_size'... \n'
            
            for th in ${!threads[@]}
            do
                curr_threads=${threads[$th]}

                curr_output_file=$output_folder_path'non_imv_inlj_with_art64tree_index_tuning_'$curr_r_dataset_size'_'$curr_s_dataset_size'_th_'$curr_threads'.csv'

                sh $(dirname "$0")/base_configs_maker.sh -INLJ_WITH_HASH_INDEX 0 \
                                            -INLJ_WITH_LEARNED_INDEX 0 \
                                            -INLJ_WITH_CSS_TREE_INDEX 0 \
                                            -INLJ_WITH_ART32_TREE_INDEX 0 \
                                            -INLJ_WITH_ART64_TREE_INDEX 1 \
                                            -INLJ_WITH_CUCKOO_HASH_INDEX 0 \
                                            -NUM_THREADS_FOR_EVALUATION $curr_threads \
                                            -RELATION_R_PATH $curr_r_dataset \
                                            -RELATION_R_FOLDER_PATH '"'$dataset_folder_path'"' \
                                            -RELATION_R_FILE_NAME '"'${r_datasets[$ds]}'"' \
                                            -RELATION_R_FILE_EXTENSION ${r_datasets_file_extension} \
                                            -RELATION_R_NUM_TUPLES $curr_r_dataset_size \
                                            -RELATION_R_FILE_NUM_PARTITIONS $curr_r_dataset_file_num_partitions \
                                            -RELATION_S_PATH $curr_s_dataset \
                                            -RELATION_S_FOLDER_PATH '"'$dataset_folder_path'"' \
                                            -RELATION_S_FILE_NAME '"'${s_datasets[$ds]}'"' \
                                            -RELATION_S_FILE_EXTENSION ${s_datasets_file_extension} \
                                            -RELATION_S_NUM_TUPLES ${curr_s_dataset_size} \
                                            -RELATION_S_FILE_NUM_PARTITIONS ${curr_s_dataset_file_num_partitions} \
                                            -BENCHMARK_RESULTS_PATH '"'${curr_output_file}'"' \
                                            -RUN_NUMS ${run_nums} -LOAD_RELATIONS_FOR_EVALUATION ${load_relations_for_evaluation} \
                                            -PERSIST_RELATIONS_FOR_EVALUATION ${persist_relations_for_evaluation} \
                                            -CUSTOM_CPU_MAPPING '"'../../include/configs/cpu-mapping_berners_lee.txt'"' \
                                            -CUSTOM_CPU_MAPPING_V2 '"'../../include/configs/cpu-mapping-v2_berners_lee.txt'"'

                cmake -DCMAKE_BUILD_TYPE=Release -DVECTORWISE_BRANCHING=on $(dirname "$0")/../.. > /dev/null

                cd $(dirname "$0")/../../build/release

                make > /dev/null

                ./npj_join_runner

                cd ../../scripts/evaluation/
            done
        done

    fi

    if [ ${run_inlj_with_cuckoohash_index} == 1 ]
    then
        echo "Running INLJ with cuckoo hash index ..."

        for ds in ${!r_datasets[@]}
        do
            curr_r_dataset='"'$dataset_folder_path${r_datasets[$ds]}.txt'"'
            curr_r_dataset_size=${r_datasets_sizes[$ds]}
            curr_r_dataset_file_num_partitions=${r_datasets_file_num_partitions[$ds]}
            curr_s_dataset='"'$dataset_folder_path${s_datasets[$ds]}.txt'"'
            curr_s_dataset_size=${s_datasets_sizes[$ds]}
            curr_s_dataset_file_num_partitions=${s_datasets_file_num_partitions[$ds]}

            echo 'Joining '$curr_r_dataset' '$curr_r_dataset_size' '$curr_s_dataset' '$curr_s_dataset_size'... \n'
            
            for th in ${!threads[@]}
            do
                curr_threads=${threads[$th]}

                curr_output_file=$output_folder_path'non_imv_inlj_with_cuckoohash_index_tuning_'$curr_r_dataset_size'_'$curr_s_dataset_size'_th_'$curr_threads'.csv'

                sh $(dirname "$0")/base_configs_maker.sh -INLJ_WITH_HASH_INDEX 0 \
                                            -INLJ_WITH_LEARNED_INDEX 0 \
                                            -INLJ_WITH_CSS_TREE_INDEX 0 \
                                            -INLJ_WITH_ART32_TREE_INDEX 0 \
                                            -INLJ_WITH_ART64_TREE_INDEX 0 \
                                            -INLJ_WITH_CUCKOO_HASH_INDEX 1 \
                                            -NUM_THREADS_FOR_EVALUATION $curr_threads \
                                            -RELATION_R_PATH $curr_r_dataset \
                                            -RELATION_R_FOLDER_PATH '"'$dataset_folder_path'"' \
                                            -RELATION_R_FILE_NAME '"'${r_datasets[$ds]}'"' \
                                            -RELATION_R_FILE_EXTENSION ${r_datasets_file_extension} \
                                            -RELATION_R_NUM_TUPLES $curr_r_dataset_size \
                                            -RELATION_R_FILE_NUM_PARTITIONS $curr_r_dataset_file_num_partitions \
                                            -RELATION_S_PATH $curr_s_dataset \
                                            -RELATION_S_FOLDER_PATH '"'$dataset_folder_path'"' \
                                            -RELATION_S_FILE_NAME '"'${s_datasets[$ds]}'"' \
                                            -RELATION_S_FILE_EXTENSION ${s_datasets_file_extension} \
                                            -RELATION_S_NUM_TUPLES ${curr_s_dataset_size} \
                                            -RELATION_S_FILE_NUM_PARTITIONS ${curr_s_dataset_file_num_partitions} \
                                            -BENCHMARK_RESULTS_PATH '"'${curr_output_file}'"' \
                                            -RUN_NUMS ${run_nums} -LOAD_RELATIONS_FOR_EVALUATION ${load_relations_for_evaluation} \
                                            -PERSIST_RELATIONS_FOR_EVALUATION ${persist_relations_for_evaluation} \
                                            -CUSTOM_CPU_MAPPING '"'../../include/configs/cpu-mapping_berners_lee.txt'"' \
                                            -CUSTOM_CPU_MAPPING_V2 '"'../../include/configs/cpu-mapping-v2_berners_lee.txt'"'

                cmake -DCMAKE_BUILD_TYPE=Release -DVECTORWISE_BRANCHING=on $(dirname "$0")/../.. > /dev/null

                cd $(dirname "$0")/../../build/release

                make > /dev/null

                ./npj_join_runner

                cd ../../scripts/evaluation/
            done
        done

    fi
}

run_nums=3 #1 5 10
load_relations_for_evaluation=1 #0
persist_relations_for_evaluation=0 #1

#unique datasets
################

r_datasets=(r_UNIQUE_v1_uint32_uint32_16000000 r_UNIQUE_v1_uint32_uint32_16000000 r_UNIQUE_v1_uint32_uint32_16000000 r_UNIQUE_v2_uint32_uint32_32000000 r_UNIQUE_v2_uint32_uint32_32000000 r_UNIQUE_v2_uint32_uint32_32000000 r_UNIQUE_v3_uint32_uint32_128000000 r_UNIQUE_v3_uint32_uint32_128000000 r_UNIQUE_v3_uint32_uint32_128000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000) #(r_UNIQUE_v1_uint32_uint32_16000000 r_UNIQUE_v2_uint32_uint32_32000000 r_UNIQUE_v3_uint32_uint32_128000000 r_UNIQUE_v4_uint32_uint32_384000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v6_uint32_uint32_896000000 r_UNIQUE_v7_uint32_uint32_1152000000 r_UNIQUE_v8_uint32_uint32_1664000000 r_UNIQUE_v9_uint32_uint32_1920000000)
s_datasets=(s_UNIQUE_v2_uint32_uint32_32000000 s_UNIQUE_v3_uint32_uint32_128000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v1_uint32_uint32_16000000 s_UNIQUE_v3_uint32_uint32_128000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v1_uint32_uint32_16000000 s_UNIQUE_v2_uint32_uint32_32000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v1_uint32_uint32_16000000 s_UNIQUE_v2_uint32_uint32_32000000 s_UNIQUE_v3_uint32_uint32_128000000) #(s_UNIQUE_v1_uint32_uint32_16000000 s_UNIQUE_v2_uint32_uint32_32000000 s_UNIQUE_v3_uint32_uint32_128000000 s_UNIQUE_v4_uint32_uint32_384000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v6_uint32_uint32_896000000 s_UNIQUE_v7_uint32_uint32_1152000000 s_UNIQUE_v8_uint32_uint32_1664000000 s_UNIQUE_v9_uint32_uint32_1920000000)
r_datasets_sizes=(16E6 16E6 16E6 32E6 32E6 32E6 128E6 128E6 128E6 640E6 640E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(32E6 128E6 640E6 16E6 128E6 640E6 16E6 32E6 640E6 16E6 32E6 128E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(16777216 16777216 16777216 33554432 33554432 33554432 134217728 134217728 134217728 536870912 536870912 536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0 0 0 0 0 0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_unique/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_unique/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_unique/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_unique/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size

r_datasets=(r_UNIQUE_v1_uint32_uint32_16000000 r_UNIQUE_v2_uint32_uint32_32000000 r_UNIQUE_v3_uint32_uint32_128000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v8_uint32_uint32_1664000000) #(r_UNIQUE_v1_uint32_uint32_16000000 r_UNIQUE_v2_uint32_uint32_32000000 r_UNIQUE_v3_uint32_uint32_128000000 r_UNIQUE_v4_uint32_uint32_384000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v6_uint32_uint32_896000000 r_UNIQUE_v7_uint32_uint32_1152000000 r_UNIQUE_v8_uint32_uint32_1664000000 r_UNIQUE_v9_uint32_uint32_1920000000)
s_datasets=(s_UNIQUE_v1_uint32_uint32_16000000 s_UNIQUE_v2_uint32_uint32_32000000 s_UNIQUE_v3_uint32_uint32_128000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v8_uint32_uint32_1664000000) #(s_UNIQUE_v1_uint32_uint32_16000000 s_UNIQUE_v2_uint32_uint32_32000000 s_UNIQUE_v3_uint32_uint32_128000000 s_UNIQUE_v4_uint32_uint32_384000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v6_uint32_uint32_896000000 s_UNIQUE_v7_uint32_uint32_1152000000 s_UNIQUE_v8_uint32_uint32_1664000000 s_UNIQUE_v9_uint32_uint32_1920000000)
r_datasets_sizes=(16E6 32E6 128E6 640E6 1664E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(16E6 32E6 128E6 640E6 1664E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(16777216 33554432 134217728 536870912 1073741824) #(33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#r_datasets=(r_UNIQUE_v2_uint32_uint32_32000000) #(r_UNIQUE_v1_uint32_uint32_16000000 r_UNIQUE_v2_uint32_uint32_32000000 r_UNIQUE_v3_uint32_uint32_128000000 r_UNIQUE_v4_uint32_uint32_384000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v6_uint32_uint32_896000000 r_UNIQUE_v7_uint32_uint32_1152000000 r_UNIQUE_v8_uint32_uint32_1664000000 r_UNIQUE_v9_uint32_uint32_1920000000)
#s_datasets=(s_UNIQUE_v2_uint32_uint32_32000000) #(s_UNIQUE_v1_uint32_uint32_16000000 s_UNIQUE_v2_uint32_uint32_32000000 s_UNIQUE_v3_uint32_uint32_128000000 s_UNIQUE_v4_uint32_uint32_384000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v6_uint32_uint32_896000000 s_UNIQUE_v7_uint32_uint32_1152000000 s_UNIQUE_v8_uint32_uint32_1664000000 s_UNIQUE_v9_uint32_uint32_1920000000)
#r_datasets_sizes=(32E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
#s_datasets_sizes=(32E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
#r_datasets_file_num_partitions=(32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
#s_datasets_file_num_partitions=(32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
#input_hash_table_size=(16777216 33554432 134217728 536870912 1073741824) #(33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
#hash_scheme_and_function_mode=(0)
#hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
#hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
#hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

r_datasets=(r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000) #(r_UNIQUE_v1_uint32_uint32_16000000 r_UNIQUE_v2_uint32_uint32_32000000 r_UNIQUE_v3_uint32_uint32_128000000 r_UNIQUE_v4_uint32_uint32_384000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v6_uint32_uint32_896000000 r_UNIQUE_v7_uint32_uint32_1152000000 r_UNIQUE_v8_uint32_uint32_1664000000 r_UNIQUE_v9_uint32_uint32_1920000000)
s_datasets=(s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v5_uint32_uint32_640000000) #(s_UNIQUE_v1_uint32_uint32_16000000 s_UNIQUE_v2_uint32_uint32_32000000 s_UNIQUE_v3_uint32_uint32_128000000 s_UNIQUE_v4_uint32_uint32_384000000 s_UNIQUE_v5_uint32_uint32_640000000 s_UNIQUE_v6_uint32_uint32_896000000 s_UNIQUE_v7_uint32_uint32_1152000000 s_UNIQUE_v8_uint32_uint32_1664000000 s_UNIQUE_v9_uint32_uint32_1920000000)
r_datasets_sizes=(640E6 640E6 640E6 640E6 640E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(640E6 640E6 640E6 640E6 640E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(213333333 213333333 213333333 213333333 213333333 213333333) #(536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL CHAINLINEARMODEL CHAINLINEARMODEL) #(0)
hash_fun=(XXHASH3 MURMUR AQUA MULTPRIME XXHASH3 XXHASH3 MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RMIHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

r_datasets=(r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64) 
s_datasets=(r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000 r_UNIQUE_v5_uint32_uint32_640000000) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64)
r_datasets_sizes=(10E6 50E6 150E6 300E6 600E6) #(200E6 800E6 200E6 800E6 200E6) (10E6 50E6 150E6 300E6 600E6) (640E6 640E6 640E6 640E6 640E6)
s_datasets_sizes=(150E6 150E6 150E6 150E6 150E6) #(200E6 800E6 200E6 800E6 200E6) (150E6 150E6 150E6 150E6 150E6) (640E6 640E6 640E6 640E6 640E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32) #(32 32 32 32 32)
s_datasets_file_num_partitions=(32 32 32 32 32 32) #(32 32 32 32 32)
input_hash_table_size=(10000000 50000000 150000000 300000000 600000000) #(3000000 15000000 50000000 100000000 200000000) 66666666 (536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL) #(CHAINEXOTIC CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL CHAINLINEARMODEL CHAINLINEARMODEL CHAINLINEARMODEL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL) #(0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR) #(XXHASH3 MURMUR AQUA MULTPRIME)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RMIHash RMIHash RMIHash RMIHash RMIHash) #RMIHash RadixSplineHash PGMHash

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_unique/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_unique_with_chasing_counter/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_unique_hashbench/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_unique/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_model_based_build_index_unique/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_unique_without_bs/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_unique/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_unique/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_unique_with_index_size/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art64tree_index_unique_with_index_size/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_cuckoohash_index_unique_with_index_size/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 0 0 1 $input_hash_table_size


#lognormal datasets
###################

r_datasets=(r_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 r_LOGNORMAL_v2_uint32_uint32_segma_1_32000000  r_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 r_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000) #(r_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 r_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 r_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 r_LOGNORMAL_v4_uint32_uint32_segma_1_384000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v6_uint32_uint32_segma_1_896000000 r_LOGNORMAL_v7_uint32_uint32_segma_1_1152000000 r_LOGNORMAL_v8_uint32_uint32_segma_1_1664000000 r_LOGNORMAL_v9_uint32_uint32_segma_1_1920000000)
s_datasets=(s_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 s_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 s_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 s_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 s_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 s_LOGNORMAL_v3_uint32_uint32_segma_1_128000000) #(s_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 s_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 s_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 s_LOGNORMAL_v4_uint32_uint32_segma_1_384000000 s_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 s_LOGNORMAL_v6_uint32_uint32_segma_1_896000000 s_LOGNORMAL_v7_uint32_uint32_segma_1_1152000000 s_LOGNORMAL_v8_uint32_uint32_segma_1_1664000000 s_LOGNORMAL_v9_uint32_uint32_segma_1_1920000000)
r_datasets_sizes=(32E6 32E6 128E6 128E6 640E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(128E6 640E6 32E6 640E6 32E6 128E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(33554432 33554432 134217728 134217728 536870912 536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

r_datasets=(r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000) #(r_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 r_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 r_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 r_LOGNORMAL_v4_uint32_uint32_segma_1_384000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v6_uint32_uint32_segma_1_896000000 r_LOGNORMAL_v7_uint32_uint32_segma_1_1152000000 r_LOGNORMAL_v8_uint32_uint32_segma_1_1664000000 r_LOGNORMAL_v9_uint32_uint32_segma_1_1920000000)
s_datasets=(r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000) #(s_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 s_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 s_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 s_LOGNORMAL_v4_uint32_uint32_segma_1_384000000 s_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 s_LOGNORMAL_v6_uint32_uint32_segma_1_896000000 s_LOGNORMAL_v7_uint32_uint32_segma_1_1152000000 s_LOGNORMAL_v8_uint32_uint32_segma_1_1664000000 s_LOGNORMAL_v9_uint32_uint32_segma_1_1920000000)
r_datasets_sizes=(10E6 50E6 150E6 300E6 600E6) #(200E6 800E6 200E6 800E6 200E6) (10E6 50E6 150E6 300E6 600E6) (640E6 640E6 640E6 640E6 640E6)
s_datasets_sizes=(150E6 150E6 150E6 150E6 150E6) #(200E6 800E6 200E6 800E6 200E6) (150E6 150E6 150E6 150E6 150E6) (640E6 640E6 640E6 640E6 640E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32) #(32 32 32 32 32)
s_datasets_file_num_partitions=(32 32 32 32 32 32) #(32 32 32 32 32)
input_hash_table_size=(10000000 50000000 150000000 300000000 600000000) #66666666 (536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL) #(CHAINEXOTIC CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL CHAINLINEARMODEL CHAINLINEARMODEL CHAINLINEARMODEL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL) #(0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR) #(XXHASH3 MURMUR AQUA MULTPRIME)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RMIHash RMIHash RMIHash RMIHash RMIHash) #RMIHash RadixSplineHash PGMHash

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_lognormal/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_lognormal_hashbench/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_lognormal/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_lognormal/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_lognormal/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size


r_datasets=(r_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 r_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 r_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000) #(r_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 r_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 r_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 r_LOGNORMAL_v4_uint32_uint32_segma_1_384000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v6_uint32_uint32_segma_1_896000000 r_LOGNORMAL_v7_uint32_uint32_segma_1_1152000000 r_LOGNORMAL_v8_uint32_uint32_segma_1_1664000000 r_LOGNORMAL_v9_uint32_uint32_segma_1_1920000000)
s_datasets=(s_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 s_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 s_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 s_LOGNORMAL_v5_uint32_uint32_segma_1_640000000) #(s_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 s_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 s_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 s_LOGNORMAL_v4_uint32_uint32_segma_1_384000000 s_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 s_LOGNORMAL_v6_uint32_uint32_segma_1_896000000 s_LOGNORMAL_v7_uint32_uint32_segma_1_1152000000 s_LOGNORMAL_v8_uint32_uint32_segma_1_1664000000 s_LOGNORMAL_v9_uint32_uint32_segma_1_1920000000)
r_datasets_sizes=(16E6 32E6 128E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(16E6 32E6 128E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(16777216 33554432 134217728 536870912) #(33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

r_datasets=(r_LOGNORMAL_v3_uint32_uint32_segma_1_128000000) #(r_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 r_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 r_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 r_LOGNORMAL_v4_uint32_uint32_segma_1_384000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v6_uint32_uint32_segma_1_896000000 r_LOGNORMAL_v7_uint32_uint32_segma_1_1152000000 r_LOGNORMAL_v8_uint32_uint32_segma_1_1664000000 r_LOGNORMAL_v9_uint32_uint32_segma_1_1920000000)
s_datasets=(s_LOGNORMAL_v5_uint32_uint32_segma_1_640000000) #(s_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 s_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 s_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 s_LOGNORMAL_v4_uint32_uint32_segma_1_384000000 s_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 s_LOGNORMAL_v6_uint32_uint32_segma_1_896000000 s_LOGNORMAL_v7_uint32_uint32_segma_1_1152000000 s_LOGNORMAL_v8_uint32_uint32_segma_1_1664000000 s_LOGNORMAL_v9_uint32_uint32_segma_1_1920000000)
r_datasets_sizes=(128E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(16777216 33554432 134217728 536870912) #(33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#r_datasets=(r_LOGNORMAL_v2_uint32_uint32_segma_1_32000000) #(r_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 r_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 r_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 r_LOGNORMAL_v4_uint32_uint32_segma_1_384000000 r_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 r_LOGNORMAL_v6_uint32_uint32_segma_1_896000000 r_LOGNORMAL_v7_uint32_uint32_segma_1_1152000000 r_LOGNORMAL_v8_uint32_uint32_segma_1_1664000000 r_LOGNORMAL_v9_uint32_uint32_segma_1_1920000000)
#s_datasets=(s_LOGNORMAL_v2_uint32_uint32_segma_1_32000000) #(s_LOGNORMAL_v1_uint32_uint32_segma_1_16000000 s_LOGNORMAL_v2_uint32_uint32_segma_1_32000000 s_LOGNORMAL_v3_uint32_uint32_segma_1_128000000 s_LOGNORMAL_v4_uint32_uint32_segma_1_384000000 s_LOGNORMAL_v5_uint32_uint32_segma_1_640000000 s_LOGNORMAL_v6_uint32_uint32_segma_1_896000000 s_LOGNORMAL_v7_uint32_uint32_segma_1_1152000000 s_LOGNORMAL_v8_uint32_uint32_segma_1_1664000000 s_LOGNORMAL_v9_uint32_uint32_segma_1_1920000000)
#r_datasets_sizes=(32E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
#s_datasets_sizes=(32E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
#r_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
#s_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
#input_hash_table_size=(16777216 33554432 134217728 536870912) #(33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
#hash_scheme_and_function_mode=(0 0 0 0)
#hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
#hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
#hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_lognormal/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_lognormal/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_model_based_build_index_lognormal/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_lognormal_without_bs/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_lognormal/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_lognormal/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size

#seq_hole datasets
################

r_datasets=(r_SEQ_HOLE_v1_uint32_uint32_16000000 r_SEQ_HOLE_v1_uint32_uint32_16000000 r_SEQ_HOLE_v1_uint32_uint32_16000000 r_SEQ_HOLE_v2_uint32_uint32_32000000 r_SEQ_HOLE_v2_uint32_uint32_32000000 r_SEQ_HOLE_v2_uint32_uint32_32000000 r_SEQ_HOLE_v3_uint32_uint32_128000000 r_SEQ_HOLE_v3_uint32_uint32_128000000 r_SEQ_HOLE_v3_uint32_uint32_128000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000) #(r_SEQ_HOLE_v1_uint32_uint32_16000000 r_SEQ_HOLE_v2_uint32_uint32_32000000 r_SEQ_HOLE_v3_uint32_uint32_128000000 r_SEQ_HOLE_v4_uint32_uint32_384000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v6_uint32_uint32_896000000 r_SEQ_HOLE_v7_uint32_uint32_1152000000 r_SEQ_HOLE_v8_uint32_uint32_1664000000 r_SEQ_HOLE_v9_uint32_uint32_1920000000)
s_datasets=(s_SEQ_HOLE_v2_uint32_uint32_32000000 s_SEQ_HOLE_v3_uint32_uint32_128000000 s_SEQ_HOLE_v5_uint32_uint32_640000000 s_SEQ_HOLE_v1_uint32_uint32_16000000 s_SEQ_HOLE_v3_uint32_uint32_128000000 s_SEQ_HOLE_v5_uint32_uint32_640000000 s_SEQ_HOLE_v1_uint32_uint32_16000000 s_SEQ_HOLE_v2_uint32_uint32_32000000 s_SEQ_HOLE_v5_uint32_uint32_640000000 s_SEQ_HOLE_v1_uint32_uint32_16000000 s_SEQ_HOLE_v2_uint32_uint32_32000000 s_SEQ_HOLE_v3_uint32_uint32_128000000) #(s_SEQ_HOLE_v1_uint32_uint32_16000000 s_SEQ_HOLE_v2_uint32_uint32_32000000 s_SEQ_HOLE_v3_uint32_uint32_128000000 s_SEQ_HOLE_v4_uint32_uint32_384000000 s_SEQ_HOLE_v5_uint32_uint32_640000000 s_SEQ_HOLE_v6_uint32_uint32_896000000 s_SEQ_HOLE_v7_uint32_uint32_1152000000 s_SEQ_HOLE_v8_uint32_uint32_1664000000 s_SEQ_HOLE_v9_uint32_uint32_1920000000)
r_datasets_sizes=(16E6 16E6 16E6 32E6 32E6 32E6 128E6 128E6 128E6 640E6 640E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(32E6 128E6 640E6 16E6 128E6 640E6 16E6 32E6 640E6 16E6 32E6 128E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(16777216 16777216 16777216 33554432 33554432 33554432 134217728 134217728 134217728 536870912 536870912 536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0 0 0 0 0 0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#r_datasets=(r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64) 
#s_datasets=(r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64)
#r_datasets_sizes=(640E6 640E6 640E6 640E6 640E6) #(200E6 800E6 200E6 800E6 200E6) (10E6 50E6 150E6 300E6 600E6)
#s_datasets_sizes=(640E6 640E6 640E6 640E6 640E6) #(200E6 800E6 200E6 800E6 200E6) (150E6 150E6 150E6 150E6 150E6)
#r_datasets_file_num_partitions=(32 32 32 32 32 32) #(32 32 32 32 32)
#s_datasets_file_num_partitions=(32 32 32 32 32 32) #(32 32 32 32 32)
#input_hash_table_size=(640000000 640000000 640000000 640000000 640000000) #66666666 (536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
#hash_scheme_and_function_mode=(CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL) #(CHAINEXOTIC CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL CHAINLINEARMODEL CHAINLINEARMODEL CHAINLINEARMODEL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL) #(0)
#hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR) #(XXHASH3 MURMUR AQUA MULTPRIME)
#hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
#hash_learned_model=(RMIHash RMIHash RMIHash RMIHash RMIHash) #RMIHash RadixSplineHash PGMHash

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_seq_hole/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_seq_hole_hashbench/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_seq_hole/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_seq_hole/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size


#r_datasets=(r_SEQ_HOLE_v1_uint32_uint32_16000000 r_SEQ_HOLE_v2_uint32_uint32_32000000 r_SEQ_HOLE_v3_uint32_uint32_128000000 r_SEQ_HOLE_v5_uint32_uint32_640000000) #(r_SEQ_HOLE_v1_uint32_uint32_16000000 r_SEQ_HOLE_v2_uint32_uint32_32000000 r_SEQ_HOLE_v3_uint32_uint32_128000000 r_SEQ_HOLE_v4_uint32_uint32_384000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v6_uint32_uint32_896000000 r_SEQ_HOLE_v7_uint32_uint32_1152000000 r_SEQ_HOLE_v8_uint32_uint32_1664000000 r_SEQ_HOLE_v9_uint32_uint32_1920000000)
#s_datasets=(s_SEQ_HOLE_v1_uint32_uint32_16000000 s_SEQ_HOLE_v2_uint32_uint32_32000000 s_SEQ_HOLE_v3_uint32_uint32_128000000 s_SEQ_HOLE_v5_uint32_uint32_640000000) #(s_SEQ_HOLE_v1_uint32_uint32_16000000 s_SEQ_HOLE_v2_uint32_uint32_32000000 s_SEQ_HOLE_v3_uint32_uint32_128000000 s_SEQ_HOLE_v4_uint32_uint32_384000000 s_SEQ_HOLE_v5_uint32_uint32_640000000 s_SEQ_HOLE_v6_uint32_uint32_896000000 s_SEQ_HOLE_v7_uint32_uint32_1152000000 s_SEQ_HOLE_v8_uint32_uint32_1664000000 s_SEQ_HOLE_v9_uint32_uint32_1920000000)
#r_datasets_sizes=(16E6 32E6 128E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
#s_datasets_sizes=(16E6 32E6 128E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
#r_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
#s_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
#input_hash_table_size=(16777216 33554432 134217728 536870912) #(33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
#hash_scheme_and_function_mode=(0 0 0 0)
#hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
#hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
#hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

r_datasets=(r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000) #(r_SEQ_HOLE_v1_uint32_uint32_16000000 r_SEQ_HOLE_v2_uint32_uint32_32000000 r_SEQ_HOLE_v3_uint32_uint32_128000000 r_SEQ_HOLE_v4_uint32_uint32_384000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v6_uint32_uint32_896000000 r_SEQ_HOLE_v7_uint32_uint32_1152000000 r_SEQ_HOLE_v8_uint32_uint32_1664000000 r_SEQ_HOLE_v9_uint32_uint32_1920000000)
s_datasets=(r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000 r_SEQ_HOLE_v5_uint32_uint32_640000000) #(s_SEQ_HOLE_v1_uint32_uint32_16000000 s_SEQ_HOLE_v2_uint32_uint32_32000000 s_SEQ_HOLE_v3_uint32_uint32_128000000 s_SEQ_HOLE_v4_uint32_uint32_384000000 s_SEQ_HOLE_v5_uint32_uint32_640000000 s_SEQ_HOLE_v6_uint32_uint32_896000000 s_SEQ_HOLE_v7_uint32_uint32_1152000000 s_SEQ_HOLE_v8_uint32_uint32_1664000000 s_SEQ_HOLE_v9_uint32_uint32_1920000000)
r_datasets_sizes=(10E6 50E6 150E6 300E6 600E6) #(200E6 800E6 200E6 800E6 200E6) (10E6 50E6 150E6 300E6 600E6) (640E6 640E6 640E6 640E6 640E6)
s_datasets_sizes=(150E6 150E6 150E6 150E6 150E6) #(200E6 800E6 200E6 800E6 200E6) (150E6 150E6 150E6 150E6 150E6) (640E6 640E6 640E6 640E6 640E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32) #(32 32 32 32 32)
s_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32) #(32 32 32 32 32)
input_hash_table_size=(10000000 50000000 150000000 300000000 600000000 10000000 50000000 150000000 300000000 600000000 10000000 50000000 150000000 300000000 600000000 10000000 50000000 150000000 300000000 600000000) #(3000000 15000000 50000000 100000000 200000000) 66666666 (536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(CHAINEXOTIC CHAINEXOTIC CHAINEXOTIC CHAINEXOTIC CHAINEXOTIC CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL) #(CHAINEXOTIC CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL CHAINLINEARMODEL CHAINLINEARMODEL CHAINLINEARMODEL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL) #(0)
hash_fun=(MWHC MWHC MWHC MWHC MWHC MULTPRIME MULTPRIME MULTPRIME MULTPRIME MULTPRIME MURMUR MURMUR MURMUR MURMUR MURMUR MULTPRIME MULTPRIME MULTPRIME MULTPRIME MULTPRIME) #(XXHASH3 MURMUR AQUA MULTPRIME)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RMIHash RMIHash RMIHash RMIHash RMIHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RMIHash RMIHash RMIHash RMIHash RMIHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash) #RMIHash RadixSplineHash PGMHash

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_seq_hole/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_seq_hole_hashbench/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_seq_hole/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_model_based_build_index_seq_hole/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_seq_hole/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_seq_hole/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size


#uniform datasets
################

r_datasets=(r_UNIFORM_v2_uint32_uint32_32000000 r_UNIFORM_v2_uint32_uint32_32000000  r_UNIFORM_v3_uint32_uint32_128000000 r_UNIFORM_v3_uint32_uint32_128000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v5_uint32_uint32_640000000) #(r_UNIFORM_v1_uint32_uint32_16000000 r_UNIFORM_v2_uint32_uint32_32000000 r_UNIFORM_v3_uint32_uint32_128000000 r_UNIFORM_v4_uint32_uint32_384000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v6_uint32_uint32_896000000 r_UNIFORM_v7_uint32_uint32_1152000000 r_UNIFORM_v8_uint32_uint32_1664000000 r_UNIFORM_v9_uint32_uint32_1920000000)
s_datasets=(s_UNIFORM_v3_uint32_uint32_128000000 s_UNIFORM_v5_uint32_uint32_640000000 s_UNIFORM_v2_uint32_uint32_32000000 s_UNIFORM_v5_uint32_uint32_640000000 s_UNIFORM_v2_uint32_uint32_32000000 s_UNIFORM_v3_uint32_uint32_128000000) #(s_UNIFORM_v1_uint32_uint32_16000000 s_UNIFORM_v2_uint32_uint32_32000000 s_UNIFORM_v3_uint32_uint32_128000000 s_UNIFORM_v4_uint32_uint32_384000000 s_UNIFORM_v5_uint32_uint32_640000000 s_UNIFORM_v6_uint32_uint32_896000000 s_UNIFORM_v7_uint32_uint32_1152000000 s_UNIFORM_v8_uint32_uint32_1664000000 s_UNIFORM_v9_uint32_uint32_1920000000)
r_datasets_sizes=(32E6 32E6 128E6 128E6 640E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(128E6 640E6 32E6 640E6 32E6 128E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(33554432 33554432 134217728 134217728 536870912 536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_uniform/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_uniform/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_uniform/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_uniform/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size


r_datasets=(r_UNIFORM_v1_uint32_uint32_16000000 r_UNIFORM_v2_uint32_uint32_32000000 r_UNIFORM_v3_uint32_uint32_128000000 r_UNIFORM_v5_uint32_uint32_640000000) #(r_UNIFORM_v1_uint32_uint32_16000000 r_UNIFORM_v2_uint32_uint32_32000000 r_UNIFORM_v3_uint32_uint32_128000000 r_UNIFORM_v4_uint32_uint32_384000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v6_uint32_uint32_896000000 r_UNIFORM_v7_uint32_uint32_1152000000 r_UNIFORM_v8_uint32_uint32_1664000000 r_UNIFORM_v9_uint32_uint32_1920000000)
s_datasets=(s_UNIFORM_v1_uint32_uint32_16000000 s_UNIFORM_v2_uint32_uint32_32000000 s_UNIFORM_v3_uint32_uint32_128000000 s_UNIFORM_v5_uint32_uint32_640000000) #(s_UNIFORM_v1_uint32_uint32_16000000 s_UNIFORM_v2_uint32_uint32_32000000 s_UNIFORM_v3_uint32_uint32_128000000 s_UNIFORM_v4_uint32_uint32_384000000 s_UNIFORM_v5_uint32_uint32_640000000 s_UNIFORM_v6_uint32_uint32_896000000 s_UNIFORM_v7_uint32_uint32_1152000000 s_UNIFORM_v8_uint32_uint32_1664000000 s_UNIFORM_v9_uint32_uint32_1920000000)
r_datasets_sizes=(16E6 32E6 128E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(16E6 32E6 128E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(16777216 33554432 134217728 536870912) #(33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#r_datasets=(r_UNIFORM_v2_uint32_uint32_32000000) #(r_UNIFORM_v1_uint32_uint32_16000000 r_UNIFORM_v2_uint32_uint32_32000000 r_UNIFORM_v3_uint32_uint32_128000000 r_UNIFORM_v4_uint32_uint32_384000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v6_uint32_uint32_896000000 r_UNIFORM_v7_uint32_uint32_1152000000 r_UNIFORM_v8_uint32_uint32_1664000000 r_UNIFORM_v9_uint32_uint32_1920000000)
#s_datasets=(s_UNIFORM_v2_uint32_uint32_32000000) #(s_UNIFORM_v1_uint32_uint32_16000000 s_UNIFORM_v2_uint32_uint32_32000000 s_UNIFORM_v3_uint32_uint32_128000000 s_UNIFORM_v4_uint32_uint32_384000000 s_UNIFORM_v5_uint32_uint32_640000000 s_UNIFORM_v6_uint32_uint32_896000000 s_UNIFORM_v7_uint32_uint32_1152000000 s_UNIFORM_v8_uint32_uint32_1664000000 s_UNIFORM_v9_uint32_uint32_1920000000)
#r_datasets_sizes=(32E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
#s_datasets_sizes=(32E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
#r_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
#s_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
#input_hash_table_size=(16777216 33554432 134217728 536870912) #(33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
#hash_scheme_and_function_mode=(0 0 0 0)
#hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
#hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
#hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

r_datasets=(r_UNIFORM_v2_uint32_uint32_32000000 r_UNIFORM_v2_uint32_uint32_32000000 r_UNIFORM_v3_uint32_uint32_128000000 r_UNIFORM_v3_uint32_uint32_128000000) #(r_UNIFORM_v1_uint32_uint32_16000000 r_UNIFORM_v2_uint32_uint32_32000000 r_UNIFORM_v3_uint32_uint32_128000000 r_UNIFORM_v4_uint32_uint32_384000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v6_uint32_uint32_896000000 r_UNIFORM_v7_uint32_uint32_1152000000 r_UNIFORM_v8_uint32_uint32_1664000000 r_UNIFORM_v9_uint32_uint32_1920000000)
s_datasets=(s_UNIFORM_v3_uint32_uint32_128000000 s_UNIFORM_v5_uint32_uint32_640000000 s_UNIFORM_v2_uint32_uint32_32000000 s_UNIFORM_v5_uint32_uint32_640000000) #(s_UNIFORM_v1_uint32_uint32_16000000 s_UNIFORM_v2_uint32_uint32_32000000 s_UNIFORM_v3_uint32_uint32_128000000 s_UNIFORM_v4_uint32_uint32_384000000 s_UNIFORM_v5_uint32_uint32_640000000 s_UNIFORM_v6_uint32_uint32_896000000 s_UNIFORM_v7_uint32_uint32_1152000000 s_UNIFORM_v8_uint32_uint32_1664000000 s_UNIFORM_v9_uint32_uint32_1920000000)
r_datasets_sizes=(32E6 32E6 128E6 128E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(128E6 640E6 32E6 640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(16777216 33554432 134217728 536870912) #(33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

r_datasets=(r_UNIFORM_v5_uint32_uint32_640000000) #(r_UNIFORM_v1_uint32_uint32_16000000 r_UNIFORM_v2_uint32_uint32_32000000 r_UNIFORM_v3_uint32_uint32_128000000 r_UNIFORM_v4_uint32_uint32_384000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v6_uint32_uint32_896000000 r_UNIFORM_v7_uint32_uint32_1152000000 r_UNIFORM_v8_uint32_uint32_1664000000 r_UNIFORM_v9_uint32_uint32_1920000000)
s_datasets=(s_UNIFORM_v5_uint32_uint32_640000000) #(s_UNIFORM_v1_uint32_uint32_16000000 s_UNIFORM_v2_uint32_uint32_32000000 s_UNIFORM_v3_uint32_uint32_128000000 s_UNIFORM_v4_uint32_uint32_384000000 s_UNIFORM_v5_uint32_uint32_640000000 s_UNIFORM_v6_uint32_uint32_896000000 s_UNIFORM_v7_uint32_uint32_1152000000 s_UNIFORM_v8_uint32_uint32_1664000000 s_UNIFORM_v9_uint32_uint32_1920000000)
r_datasets_sizes=(640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
s_datasets_sizes=(640E6) #(16E6 32E6 128E6 384E6 640E6 896E6 1152E6 1664E6 1920E6)
r_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
s_datasets_file_num_partitions=(32 32 32 32) #(64 64 64 64 64 64 64 64 64)
input_hash_table_size=(536870912) #(33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)


r_datasets=(r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v5_uint32_uint32_640000000) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64) 
s_datasets=(r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v5_uint32_uint32_640000000 r_UNIFORM_v5_uint32_uint32_640000000) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64)
r_datasets_sizes=(10E6 50E6 150E6 300E6 600E6) #(200E6 800E6 200E6 800E6 200E6) (10E6 50E6 150E6 300E6 600E6) (640E6 640E6 640E6 640E6 640E6)
s_datasets_sizes=(150E6 150E6 150E6 150E6 150E6) #(200E6 800E6 200E6 800E6 200E6) (150E6 150E6 150E6 150E6 150E6) (640E6 640E6 640E6 640E6 640E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32) #(32 32 32 32 32)
s_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32) #(32 32 32 32 32)
input_hash_table_size=(10000000 50000000 150000000 300000000 600000000 10000000 50000000 150000000 300000000 600000000 10000000 50000000 150000000 300000000 600000000 10000000 50000000 150000000 300000000 600000000) #(3000000 15000000 50000000 100000000 200000000) 66666666 (536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(CHAINEXOTIC CHAINEXOTIC CHAINEXOTIC CHAINEXOTIC CHAINEXOTIC CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL) #(CHAINEXOTIC CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL CHAINLINEARMODEL CHAINLINEARMODEL CHAINLINEARMODEL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL) #(0)
hash_fun=(MWHC MWHC MWHC MWHC MWHC MULTPRIME MULTPRIME MULTPRIME MULTPRIME MULTPRIME MURMUR MURMUR MURMUR MURMUR MURMUR MULTPRIME MULTPRIME MULTPRIME MULTPRIME MULTPRIME) #(XXHASH3 MURMUR AQUA MULTPRIME)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RMIHash RMIHash RMIHash RMIHash RMIHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RMIHash RMIHash RMIHash RMIHash RMIHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash) #RMIHash RadixSplineHash PGMHash

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_uniform/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_uniform_with_chasing_counter/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_uniform_hashbench/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_uniform/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_uniform_without_bs/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_model_based_build_index_uniform/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_uniform/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_uniform/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size

#sosd datasets
################

r_datasets=(books_200M_uint32) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64) 
s_datasets=(books_200M_uint32) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64)
r_datasets_sizes=(200E6) #(200E6 800E6 200E6 800E6 200E6)
s_datasets_sizes=(200E6) #(200E6 800E6 200E6 800E6 200E6)
r_datasets_file_num_partitions=(32) #(32 32 32 32 32)
s_datasets_file_num_partitions=(32) #(32 32 32 32 32)
input_hash_table_size=(536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_books_200M_uint32/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_sosd_books_200M_uint32/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_sosd_books_200M_uint32_without_bs/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_sosd_books_200M_uint32/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_sosd_books_200M_uint32/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size

r_datasets=(books_800M_uint64) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64) 
s_datasets=(books_800M_uint64) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64)
r_datasets_sizes=(800E6) #(200E6 800E6 200E6 800E6 200E6)
s_datasets_sizes=(800E6) #(200E6 800E6 200E6 800E6 200E6)
r_datasets_file_num_partitions=(32) #(32 32 32 32 32)
s_datasets_file_num_partitions=(32) #(32 32 32 32 32)
input_hash_table_size=(1073741824) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_books_800M_uint64/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_sosd_books_800M_uint64/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_sosd_books_800M_uint64_without_bs/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_sosd_books_800M_uint64/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_sosd_books_800M_uint64/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size


r_datasets=(fb_200M_uint64 fb_200M_uint64 fb_200M_uint64 fb_200M_uint64 fb_200M_uint64 fb_200M_uint64) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64) 
s_datasets=(fb_200M_uint64 fb_200M_uint64 fb_200M_uint64 fb_200M_uint64 fb_200M_uint64 fb_200M_uint64) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64)
r_datasets_sizes=(100E6 100E6 100E6 100E6 100E6 100E6) #(200E6 800E6 200E6 800E6 200E6) (10E6 25E6 50E6 100E6 200E6) (10E6 25E6 50E6 100E6 10E6 25E6 50E6 100E6)
s_datasets_sizes=(25E6 25E6 25E6 25E6 25E6 25E6) #(200E6 800E6 200E6 800E6 200E6) (50E6 50E6 50E6 50E6 50E6) (25E6 25E6 25E6 25E6 25E6 25E6 25E6 25E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32) #(32 32 32 32 32)
s_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32) #(32 32 32 32 32)
input_hash_table_size=(100000000 100000000 100000000 100000000 100000000 100000000 25000000 50000000 10000000 25000000 50000000 10000000 25000000 50000000 10000000 25000000 50000000) #66666666 (536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(PROBETRADITIONAL PROBETRADITIONAL PROBELINEARMODEL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOLINEARMODEL) #(CHAINEXOTIC CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL CHAINLINEARMODEL CHAINLINEARMODEL CHAINLINEARMODEL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL) #(0)
hash_fun=(MURMUR MULTPRIME MURMUR MURMUR MULTPRIME MULTPRIME MURMUR MURMUR MURMUR) #(XXHASH3 MURMUR AQUA MULTPRIME MWHC)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RMIHash RMIHash RMIHash RadixSplineHash RadixSplineHash RMIHash RMIHash RMIHash RMIHash) #RMIHash RadixSplineHash PGMHash

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_fb_200M_uint64/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_fb_200M_uint64_with_chasing_counter/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_fb_200M_uint64_hashbench/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size 
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_sosd_fb_200M_uint64/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_sosd_fb_200M_uint64_without_bs/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_sosd_fb_200M_uint64/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_sosd_fb_200M_uint64/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size


r_datasets=(osm_cellids_800M_uint64 osm_cellids_800M_uint64 osm_cellids_800M_uint64 osm_cellids_800M_uint64 osm_cellids_800M_uint64) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64) 
s_datasets=(osm_cellids_800M_uint64 osm_cellids_800M_uint64 osm_cellids_800M_uint64 osm_cellids_800M_uint64 osm_cellids_800M_uint64) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64)
r_datasets_sizes=(400E6 400E6 400E6 400E6 400E6) #(200E6 800E6 200E6 800E6 200E6) (50E6 100E6 200E6 400E6 800E6
s_datasets_sizes=(100E6 100E6 100E6 100E6 100E6) #(200E6 800E6 200E6 800E6 200E6) (200E6 200E6 200E6 200E6 200E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32) #(32 32 32 32 32)
s_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32) #(32 32 32 32 32)
input_hash_table_size=(400000000 400000000 400000000 400000000 400000000 50000000 100000000 200000000 50000000 100000000 200000000 50000000 100000000 200000000 50000000 100000000 200000000) #(1073741824) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(CHAINLINEARMODEL CHAINLINEARMODEL PROBETRADITIONAL PROBETRADITIONAL PROBELINEARMODEL CUCKOOLINEARMODEL CUCKOOLINEARMODEL) #(CHAINEXOTIC CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL CHAINLINEARMODEL CHAINLINEARMODEL CHAINLINEARMODEL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL) #(0)
hash_fun=(MWHC MWHC MURMUR MULTPRIME MURMUR MURMUR MULTPRIME MULTPRIME MULTPRIME MURMUR MURMUR MURMUR) #(XXHASH3 MURMUR AQUA MULTPRIME)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RMIHash RadixSplineHash RMIHash RadixSplineHash RMIHash RadixSplineHash RMIHash RMIHash RMIHash RadixSplineHash RadixSplineHash RadixSplineHash) #RMIHash RadixSplineHash PGMHash

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_osm_cellids_800M_uint64/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_osm_cellids_800M_uint64_with_chasing_counter/
output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_osm_cellids_800M_uint64_hashbench/
process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_sosd_osm_cellids_800M_uint64/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_sosd_osm_cellids_800M_uint64_without_bs/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_sosd_osm_cellids_800M_uint64/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_sosd_osm_cellids_800M_uint64/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size

r_datasets=(wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64) 
s_datasets=(wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64 wiki_ts_200M_uint64) #(books_200M_uint32 books_800M_uint64 fb_200M_uint64 osm_cellids_800M_uint64 wiki_ts_200M_uint64)
r_datasets_sizes=(100E6 100E6 100E6 100E6 100E6 100E6 100E6 100E6) #(200E6 800E6 200E6 800E6 200E6) (10E6 25E6 50E6 100E6 200E6) (10E6 25E6 50E6 100E6 10E6 25E6 50E6 100E6)
s_datasets_sizes=(25E6 25E6 25E6 25E6 25E6 25E6 25E6 25E6) #(200E6 800E6 200E6 800E6 200E6) (50E6 50E6 50E6 50E6 50E6) (25E6 25E6 25E6 25E6 25E6 25E6 25E6 25E6)
r_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32) #(32 32 32 32 32)
s_datasets_file_num_partitions=(32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32) #(32 32 32 32 32)
input_hash_table_size=(100000000 100000000 100000000 100000000 100000000 100000000 100000000 100000000 25000000 50000000 10000000 25000000 50000000 10000000 25000000 50000000 10000000 25000000 50000000) #66666666 (536870912) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(PROBETRADITIONAL PROBETRADITIONAL PROBELINEARMODEL PROBELINEARMODEL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOLINEARMODEL CUCKOOLINEARMODEL) #(CHAINEXOTIC CUCKOOLINEARMODEL CUCKOOLINEARMODEL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL CUCKOOTRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL PROBETRADITIONAL CHAINLINEARMODEL CHAINLINEARMODEL CHAINLINEARMODEL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL CHAINTRADITIONAL) #(0)
hash_fun=(MURMUR MULTPRIME MULTPRIME MULTPRIME MURMUR MULTPRIME MULTPRIME MULTPRIME MURMUR MULTPRIME MULTPRIME MULTPRIME MURMUR MURMUR MURMUR) #(XXHASH3 MURMUR AQUA MULTPRIME MWHC)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RMIHash RadixSplineHash RadixSplineHash RMIHash RMIHash RadixSplineHash RMIHash RMIHash RadixSplineHash RadixSplineHash RadixSplineHash RMIHash RMIHash RMIHash) #RMIHash RadixSplineHash PGMHash

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_wiki_ts_200M_uint64/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_wiki_ts_200M_uint64_with_chasing_counter/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_sosd_wiki_ts_200M_uint64_hashbench/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_sosd_wiki_ts_200M_uint64/
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_sosd_wiki_ts_200M_uint64_without_bs/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_sosd_wiki_ts_200M_uint64/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_sosd_wiki_ts_200M_uint64/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size


#tpch workloads
################

r_datasets=(r_q3_v1_uint32_uint32_178 r_q3_v2_uint32_uint32_87)
s_datasets=(s_q3_v1_uint32_uint32_5787 s_q3_v2_uint32_uint32_233105) 
r_datasets_sizes=(178 87) 
s_datasets_sizes=(5787 233105)
r_datasets_file_num_partitions=(32 32) 
s_datasets_file_num_partitions=(32 32)
input_hash_table_size=(16777216 16777216) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_tpch_q3/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_tpch_q3/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_tpch_q3/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_tpch_q3/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size

r_datasets=(r_q5_v1_uint32_uint32_1 r_q5_v2_uint32_uint32_5 r_q5_v3_uint32_uint32_191 r_q5_v4_uint32_uint32_88 r_q5_v5_uint32_uint32_38)
s_datasets=(s_q5_v1_uint32_uint32_20 s_q5_v2_uint32_uint32_885 s_q5_v3_uint32_uint32_131 s_q5_v4_uint32_uint32_865032 s_q5_v5_uint32_uint32_690) 
r_datasets_sizes=(1 5 191 88 38) 
s_datasets_sizes=(20 885 131 865032 690)
r_datasets_file_num_partitions=(32 32 32 32 32) 
s_datasets_file_num_partitions=(32 32 32 32 32)
input_hash_table_size=(16777216 16777216 16777216 16777216 16777216) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_tpch_q5/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_tpch_q5/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_tpch_q5/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_tpch_q5/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size


r_datasets=(r_q9_v1_uint32_uint32_22 r_q9_v2_uint32_uint32_47 r_q9_v3_uint32_uint32_705 r_q9_v4_uint32_uint32_73)
s_datasets=(s_q9_v1_uint32_uint32_690 s_q9_v2_uint32_uint32_11078 s_q9_v3_uint32_uint32_38 s_q9_v4_uint32_uint32_742875) 
r_datasets_sizes=(22 47 705 73) 
s_datasets_sizes=(690 11078 38 742875)
r_datasets_file_num_partitions=(32 32 32 32) 
s_datasets_file_num_partitions=(32 32 32 32)
input_hash_table_size=(16777216 16777216 16777216 16777216) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_tpch_q9/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_tpch_q9/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_tpch_q9/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_tpch_q9/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size


r_datasets=(r_q18_v1_uint32_uint32_1 r_q18_v2_uint32_uint32_922 r_q18_v3_uint32_uint32_1)
s_datasets=(s_q18_v1_uint32_uint32_41501 s_q18_v2_uint32_uint32_1 s_q18_v3_uint32_uint32_752845) 
r_datasets_sizes=(1 922 1) 
s_datasets_sizes=(41501 1 752845)
r_datasets_file_num_partitions=(32 32 32) 
s_datasets_file_num_partitions=(32 32 32)
input_hash_table_size=(16777216 16777216 16777216) #(16777216(for_16E6) 33554432(for_32E6) 134217728(for_128E6) 536870912(for_640E6) 1073741824(for_1664E6) 2147483648(for_1920E6))
hash_scheme_and_function_mode=(0 0 0 0)
hash_fun=(MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR MURMUR)
hash_overalloc=(10 10 10 10 10 10 10 10 10 10 10 10)
hash_learned_model=(RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash RadixSplineHash)

#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_hash_index_tpch_q18/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 1 0 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_learned_index_tpch_q18/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 1 0 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_csstree_index_tpch_q18/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 1 0 $input_hash_table_size
#output_folder_path=/spinning/sabek/learned_join_results/non_imv_inlj_with_art32tree_index_tpch_q18/
#process_join $r_datasets $r_datasets_sizes $r_datasets_file_num_partitions $s_datasets $s_datasets_sizes $s_datasets_file_num_partitions $output_folder_path $run_nums $load_relations_for_evaluation $persist_relations_for_evaluation 0 0 0 1 $input_hash_table_size



