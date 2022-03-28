include(FetchContent)

set(LEARNED_HASHING_LIBRARY learned-hashing)
FetchContent_Declare(
  ${LEARNED_HASHING_LIBRARY}
  GIT_REPOSITORY https://github.com/DominikHorn/learned-hashing.git
  GIT_TAG 35c8490
  )
FetchContent_MakeAvailable(${LEARNED_HASHING_LIBRARY})
