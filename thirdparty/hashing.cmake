include(FetchContent)

set(HASHING_LIBRARY hashing)
FetchContent_Declare(
  ${HASHING_LIBRARY}
  GIT_REPOSITORY https://github.com/DominikHorn/hashing.git 
  GIT_TAG d6a907a 
  )
FetchContent_MakeAvailable(${HASHING_LIBRARY})
