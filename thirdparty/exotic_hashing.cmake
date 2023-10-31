include(FetchContent)

set(EXOTIC_HASHING_LIBRARY exotic-hashing)
FetchContent_Declare(
  ${EXOTIC_HASHING_LIBRARY}
  GIT_REPOSITORY https://github.com/DominikHorn/exotic-hashing.git
  GIT_TAG bdadf098db34bbe309fdc395978b22b39b6823b2 
  )
FetchContent_MakeAvailable(${EXOTIC_HASHING_LIBRARY})
