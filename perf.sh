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

# Execute the target
#perf stat -e LLC-load-misses ${BUILD_DIR}/${TARGET}
perf record -e LLC-load-misses -c 100 ${BUILD_DIR}/${TARGET}

