include(FetchContent)

set(EXOTIC_HASHING_LIBRARY exotic-hashing)
FetchContent_Declare(
  ${EXOTIC_HASHING_LIBRARY}
  GIT_REPOSITORY https://github.com/DominikHorn/exotic-hashing.git
  GIT_TAG 77ae8a6 
  )
FetchContent_MakeAvailable(${EXOTIC_HASHING_LIBRARY})
