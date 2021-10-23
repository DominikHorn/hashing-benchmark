include(FetchContent)

set(EXOTIC_HASHING_LIBRARY exotic-hashing)
FetchContent_Declare(
  ${EXOTIC_HASHING_LIBRARY}
  GIT_REPOSITORY git@github.com:DominikHorn/exotic-hashing.git
  GIT_TAG 2bd1f6f
  )
FetchContent_MakeAvailable(${EXOTIC_HASHING_LIBRARY})
