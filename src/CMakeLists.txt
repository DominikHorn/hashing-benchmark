cmake_minimum_required(VERSION 3.19)

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

# ==== General Dependencies ====
include(${PROJECT_SOURCE_DIR}/thirdparty/hashing.cmake)
include(${PROJECT_SOURCE_DIR}/thirdparty/learned_hashing.cmake)
include(${PROJECT_SOURCE_DIR}/thirdparty/exotic_hashing.cmake)
include(${PROJECT_SOURCE_DIR}/thirdparty/hashtable.cmake)

# ==== Test target ====
add_executable(mt_tests tests.cpp)

include(${PROJECT_SOURCE_DIR}/thirdparty/googletest.cmake)
target_link_libraries(mt_tests PRIVATE ${PROJECT_NAME} ${HASHING_LIBRARY} ${LEARNED_HASHING_LIBRARY} ${EXOTIC_HASHING_LIBRARY} ${HASHTABLE_LIBRARY} ${GOOGLETEST_LIBRARY})

# enable ctest support (i.e., test discovery)
include(GoogleTest)
gtest_discover_tests(mt_tests)

# ==== Benchmark executable ====
add_executable(benchmarks benchmarks.cpp)

include(${PROJECT_SOURCE_DIR}/thirdparty/googlebenchmark.cmake)
target_link_libraries(benchmarks PRIVATE ${PROJECT_NAME} ${HASHING_LIBRARY} ${LEARNED_HASHING_LIBRARY} ${EXOTIC_HASHING_LIBRARY} ${HASHTABLE_LIBRARY} ${GOOGLEBENCHMARK_LIBRARY})
