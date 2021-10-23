include(FetchContent)

set(GOOGLETEST_CONTENT googletest)
set(GOOGLETEST_LIBRARY gtest_main)
FetchContent_Declare(
  ${GOOGLETEST_CONTENT}
  URL https://github.com/google/googletest/archive/96f4ce02a3a78d63981c67acbd368945d11d7d70.zip
  )

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(${GOOGLETEST_CONTENT})
