include(FetchContent)

set(HASHING_LIBRARY hashing)
FetchContent_Declare(
  ${HASHING_LIBRARY}
  GIT_REPOSITORY git@github.com:DominikHorn/hashing.git
  GIT_TAG 943dac2
  )
FetchContent_MakeAvailable(${HASHING_LIBRARY})
