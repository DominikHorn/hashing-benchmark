
# Hashing Benchmarking

This repository has the source code for the implementation of various hash functions and schemes used in our "Can Learned Models Replace Hash Functions?" VLDB submission. 

### Files

- Hash table implementation using different combinations of hashing schemes and functions:
  - `include/chained.hpp`: chained hash table using traditional hash functions
  - `include/chained_model.hpp`: chained hash table using learned hash functions
  - `include/chained_exotic.hpp`: chained hash table using perfect hash functions
  - `include/probe.hpp`: linear probing hash table using traditional hash functions
  - `include/probe_model.hpp`: linear probing hash table using learned hash functions
  - `include/probe_exotic.hpp`: linear probing hash table using perfect hash functions
  - `inclulde/cuckoo.hpp`: cuckoo hash table using traditional hash functions
  - `include/cuckoo_model.hpp`: cuckoo hash table using learned hash functions
  - `include/cuckoo_exotic.hpp`: cuckoo hash table using perfect hash functions 
  <!-- - `include/mmphf_table.hpp`: hashtable exploiting additional guarantees of minimal monotone perfect hash functions -->
  <!-- - `include/monotone_hashtable.hpp`: work in progress implementation of a hashtable exploiting monotone hash functions to offer lower bound lookups & scanning elements sequentially -->
  
- Optimization stuff
  - `include/convenience/`: commonly used cpp macros (e.g., `forceinline`) and related functionality 
    <!-- - `builtins.hpp`: helper cpp macros like `forceinline` -->
    <!-- - `undef.hpp`: undef for macros to make sure they don't leak should this code be included somewhere else -->
  - `include/support.hpp`: simple tape storage implementation to eliminate small allocs in hashtables

- Testing and benchmarking driver code
  - `src/benchmarks/`:
    - `passive_stats.hpp`: benchmark code for collecting passive stats of hash tables
    - `template_tables.hpp`: benchmark code for collecting insert and probe stats of hash tables
    - `tables.hpp`: some hashtable benchmark experiments 
    - `template_tables_range.hpp`: benchmark code for collecting range query stats of hash tables
  - `src/support/`: code shared by different benchmarks and tests for loading datasets and generating probe distributions
  <!-- - `src/tests/`: testcase code to ensure everything works correctly. Seems to have never been updated  -->
  - `src/benchmarks.cpp`: original entry point for benchmarks target 
  - `src/tests.cpp`: original entry point for tests target 
  - `cleanup.py`: deduplicate and sort measurements json file 

- Building and running scripts
  - `setup.sh`: original script to setup repo (submodule checkout, cmake configure etc) 
  - `requirements.txt`: python requirements 
  - `CMakeLists.txt`: cmake target definitions 
  - `thirdparty/`: cmake dependency declarations
  - `build-debug.sh`: make debug build 
  - `build.sh`: make production build 
  - `run.sh`: original script to build and execute benchmark target 
  - `perf.sh`: like `run.sh` but with perf instrumentation
  - `only_new.py`: helper script for `run.sh`, which extracts all datapoints we already measured from results.json and ensures that we only run new datapoints
  - `test.sh`: orignal script to build and execute tests
  - `benchmark.sh`: script to run probe and insert relevant code for benchmarking

- `*results*.json`: benchmark results from internal measurements 

- `README.md` this file

<!-- - `export.py`: original plotting script -->
<!-- - `edit_benchmark.py`: script to copy relevant code for benchmarking -->
<!-- - `masters_thesis.hpp`: header file exposing everything from include/ as a library (to be used by benchmarks and tests) -->

### Running Steps

Full steps to re-produce our experimental results for the VLDB submission are under construction.

