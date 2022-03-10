#!/bin/bash

# Setup script
source .env
set -e
cd "$(dirname "$0")"

# Parse arguments
TARGET=${1:-"benchmarks"}
BUILD_TYPE=${2:-"RELEASE"}
BUILD_DIR="cmake-build-$(echo "${BUILD_TYPE}" | awk '{print tolower($0)}')"

# Build the target
./build.sh ${TARGET} ${BUILD_TYPE}

# Find out which benchmarks are new
# NEW_REGEX=$(python only_new.py)

# Execute the target
# _GLIBCXX_REGEX_STATE_LIMIT=3000 ${BUILD_DIR}/src/${TARGET} --benchmark_out=benchmark_results.json --benchmark_out_format=json --benchmark_filter="${NEW_REGEX}"

_GLIBCXX_REGEX_STATE_LIMIT=3000 ${BUILD_DIR}/src/${TARGET} --benchmark_out=benchmark_results.json --benchmark_out_format=json 