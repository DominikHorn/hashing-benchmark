cmake_minimum_required(VERSION 3.19)
project("masters-thesis" VERSION 1.0
  DESCRIPTION "TBD"
  HOMEPAGE_URL "https://github.com/DominikHorn/masters-thesis"
  LANGUAGES CXX)

# Benchmark and test code
get_directory_property(hasParent PARENT_DIRECTORY)
if (NOT hasParent)
  add_subdirectory(src)
endif()
