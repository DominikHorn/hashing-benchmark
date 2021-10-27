include(FetchContent)

set(EXOTIC_HASHING_LIBRARY exotic-hashing)
FetchContent_Declare(
  ${EXOTIC_HASHING_LIBRARY}
  GIT_REPOSITORY git@github.com:DominikHorn/exotic-hashing.git
  GIT_TAG 96b8139
  )
FetchContent_MakeAvailable(${EXOTIC_HASHING_LIBRARY})
