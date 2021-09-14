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
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# ==== Non compatible compile flags ====
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-static-libsan HAS_LIBSAN)
if (HAS_LIBSAN)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -static-libsan")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -static-libasan")
endif()

# ==== Dependencies ====
include(thirdparty/googlebenchmark.cmake)
include(FetchContent)

FetchContent_Declare(
  hashing
  GIT_REPOSITORY git@github.com:DominikHorn/hashing.git
  GIT_TAG bd92466
  )
FetchContent_Declare(
  learned-hashing
  GIT_REPOSITORY git@github.com:DominikHorn/learned-hashing.git
  GIT_TAG f717473
  )
FetchContent_Declare(
  exotic-hashing
  GIT_REPOSITORY git@github.com:DominikHorn/exotic-hashing.git
  GIT_TAG 21f59bc
  )
FetchContent_Declare(
  hashtable
  GIT_REPOSITORY git@github.com:DominikHorn/hashtable.git
  GIT_TAG 65f74b4
  )

FetchContent_MakeAvailable(hashing learned-hashing exotic-hashing hashtable)

# ==== Test target ====
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/96f4ce02a3a78d63981c67acbd368945d11d7d70.zip
  )
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_executable(mt_tests src/tests.cpp)
target_link_libraries(mt_tests learned-hashing gtest_main)
# enable ctest support (i.e., test discovery)
include(GoogleTest)
gtest_discover_tests(mt_tests)

# ==== Benchmark executable ====
add_executable(benchmarks src/benchmarks.cpp)
target_link_libraries(benchmarks PRIVATE hashing learned-hashing exotic-hashing hashtable ${GOOGLEBENCHMARK_LIBRARY})
