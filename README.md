
# Hashing Benchmarking

This repository has the source code for the implementation of various hash functions and schemes used in our "Can Learned Models Replace Hash Functions?" VLDB submission. 

# Files

- `include/`: 
  - `convenience/`: commonly used macros and related functionality 
    - `builtins.hpp`: helper cpp macros like `forceinline` 
    - `undef.hpp`: undef for macros to make sure they don't leak should this code be included somewhere else 
  - `chained.hpp`: chained hash table using traditional hash functions
  - `chained_model.hpp`: chained hash table using learned hash functions
  - `chained_exotic.hpp`: chained hash table using perfect hash functions
  - `probe.hpp`: linear probing hash table using traditional hash functions
  - `probe_model.hpp`: linear probing hash table using learned hash functions
  - `probe_exotic.hpp`: linear probing hash table using perfect hash functions
  - `cuckoo.hpp`: Cuckoo hash table using traditional hash functions
  - `cuckoo_model.hpp`: Cuckoo hash table using learned hash functions
  - `cuckoo_exotic.hpp`: Cuckoo hash table using perfect hash functions
  - `mmphf_table.hpp`: hashtable exploiting additional guarantees of minimal monotone perfect hash functions
  - `monotone_hashtable.hpp`: work in progress implementation of a hashtable exploiting monotone hash functions to offer lower bound lookups & scanning elements sequentially
  - `support.hpp`: simple tape storage implementation to eliminate small allocs in hashtables
- `src/`: test and benchmark driver code 
  - `benchmarks/`:
    - `passive_stats.hpp`: Benchmark Code for collecting passive stats of hash tables
    - `template_tables.hpp`: Benchmark Code for collecting insert and probe stats of hash tables
    - `tables.hpp`: some hashtable benchmark experiments 
    - `template_tables_range.hpp`: Benchmark Code for collecting range query stats of hash tables
  - `support/`: code shared by different benchmarks and tests for loading datasets and generating probe distributions
  <!-- - `tests/`: testcase code to ensure everything works correctly. Seems to have never been updated  -->
  - `benchmarks.cpp`: original entry point for benchmarks target, idk if still used - **TODO(kapil)**
  - `tests.cpp`: original entry point for tests target, idk if still relevant as also stated above - **TODO(kapil)**
  - remaining files idk - **TODO(kapil)**
- `thirdparty/`: cmake dependency declarations
- `build-debug.sh`: make debug build - 
- `build.sh`: make production build - 
- `cleanup.py`: deduplicate and sort measurements json file - 
- `CMakeLists.txt`: cmake target definitions - 
- `export.py`: original plotting script - 
- `edit_benchmark.py`: script to copy relevant code for benchmarking
- `benchmark.sh`: script to run probe and insert relevant code for benchmarking
- `masters_thesis.hpp`: header file exposing everything from include/ as a library (to be used by benchmarks and tests)
- `*results*.json`: benchmark results from internal measurements 
- `only_new.py`: helper script for `run.sh`. Extracts all datapoints we already measures from results.json and ensures that we only run new datapoints
- `perf.sh`: like `run.sh` but with perf instrumentation
- `README.md` this file
- `requirements.txt`: python requirements 
- `run.sh`: original script to build and execute benchmark target - 
- `setup.sh`: original script to setup repo (submodule checkout, cmake configure etc) - 
- `test.sh`: orignal script to build and execute tests

Full scripts for re-producing our experiments for the VLDB submission are under construction.

