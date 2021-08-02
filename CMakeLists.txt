cmake_minimum_required(VERSION 3.19)
project("masters-thesis" VERSION 1.0
  DESCRIPTION "TBD"
  HOMEPAGE_URL "https://github.com/DominikHorn/masters-thesis"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fsanitize=address,leak,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# ==== Non compatible compile flags ====
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-static-libsan HAS_LIBSAN)
if (HAS_LIBSAN)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -static-libsan")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -static-libasan")
endif()

# ==== Dependencies ====
include(${PROJECT_SOURCE_DIR}/thirdparty/googlebenchmark.cmake)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/hashing)

# ==== Benchmark executable ====
add_executable(benchmarks src/benchmarks.cpp)
target_link_libraries(benchmarks PRIVATE hashing ${GOOGLEBENCHMARK_LIBARY})
