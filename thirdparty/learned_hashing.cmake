include(FetchContent)

set(LEARNED_HASHING_LIBRARY learned-hashing)
FetchContent_Declare(
  ${LEARNED_HASHING_LIBRARY}
  GIT_REPOSITORY git@github.com:DominikHorn/learned-hashing.git
  GIT_TAG d9d04f0
  )
FetchContent_MakeAvailable(${LEARNED_HASHING_LIBRARY})
