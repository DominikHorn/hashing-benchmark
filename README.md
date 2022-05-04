"Can Learned Models Replace Hash Functions?" VLDB submission.
This repository has code for various hash functions and schemes.
Full Scripts for Re-evaluation are under work

# Files

- `include/`: 
  - `convenience/`: commonly used macros and related functionality **TODO(kapil)** is this even used anymore?
    - `builtins.hpp`: helper cpp macros like `forceinline` **TODO(kapil)** is this even used anymore?
    - `undef.hpp`: undef for macros to make sure they don't leak should this code be included somewhere else **TODO(kapil)** is this even used anymore?
  - `chained.hpp`: chained hash table using traditional hash functions
  - `chained_model.hpp`: chained hash table using learned hash functions
  - `chained_exotic.hpp`: chained hash table using perfect hash functions
  - `probe.hpp`: chained hash table using traditional hash functions
  - `probe_model.hpp`: chained hash table using learned hash functions
  - `probe_exotic.hpp`: chained hash table using perfect hash functions
  - `mmphf_table.hpp`: hashtable exploiting additional guarantees of minimal monotone perfect hash functions
  - `monotone_hashtable.hpp`: work in progress implementation of a hashtable exploiting monotone hash functions to offer lower bound lookups & scanning elements sequentially
  - `support.hpp`: simple tape storage implementation to eliminate small allocs in hashtables
- `src/`: test and benchmark driver code + **TODO(kapil)**
  - `benchmarks/`: **TODO(kapil)**
    - `*`: **TODO(kapil)**
    - `tables.hpp`: some hashtable benchmark experiments **TODO(kapil)** is this even used anymore?
    - `*`: **TODO(kapil)**
  - `support/`: code shared by different benchmarks and tests for loading datasets and generating probe distributions **TODO(kapil)** is this even used anymore?
  - `tests/`: testcase code to ensure everything works correctly. Seems to have never been updated **TODO(kapil)** (?)
  - `benchmarks.cpp`: original entry point for benchmarks target, idk if still used - **TODO(kapil)**
  - `tests.cpp`: original entry point for tests target, idk if still relevant as also stated above - **TODO(kapil)**
  - remaining files idk - **TODO(kapil)**
- `thirdparty/`: cmake dependency declarations
- `build-debug.sh`: make debug build - **TODO(kapil)** is this relevant anymore?
- `build.sh`: make production build - **TODO(kapil)** is this relevant anymore?
- `cleanup.py`: deduplicate and sort measurements json file - **TODO(kapil)** is this relevant anymore?
- `CMakeLists.txt`: cmake target definitions - **TODO(kapil)** is this relevant anymore?
- `export.py`: original plotting script - **TODO(kapil)** is this relevant anymore?
- `kapil_*`: **TODO(kapil)**
- `masters_thesis.hpp`: header file exposing everything from include/ as a library (to be used by benchmarks and tests)
- `*results*.json`: benchmark results from internal measurements - **TODO(kapil)** is this relevant anymore?
- `only_new.py`: helper script for `run.sh`. Extracts all datapoints we already measures from results.json and ensures that we only run new datapoints
- `perf.sh`: like `run.sh` but with perf instrumentation
- `README.md` this file
- `requirements.txt`: outdated python requirements - **TODO(kapil)** update
- `run.sh`: original script to build and execute benchmark target - **TODO(kapil)** is this relevant anymore?
- `setup.sh`: original script to setup repo (submodule checkout, cmake configure etc) - **TODO(kapil)** please update with your necessary setup if any
- `test.sh`: orignal script to build and execute tests- **TODO(kapil)** is this relevant anymore?
