include(FetchContent)

set(HASHTABLE_LIBRARY hashtable)
FetchContent_Declare(
  ${HASHTABLE_LIBRARY}
  GIT_REPOSITORY git@github.com:DominikHorn/hashtable.git
  GIT_TAG 93b508d
  )
FetchContent_MakeAvailable(${HASHTABLE_LIBRARY})
