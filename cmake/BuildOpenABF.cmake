# Declare the project
FetchContent_Declare(
    OpenABF
    GIT_REPOSITORY https://github.com/educelab/OpenABF.git
    GIT_TAG 3c1b52a
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(OpenABF)

find_package(Eigen3 REQUIRED NO_MODULE)
if(CMAKE_GENERATOR MATCHES "Ninja|.*Makefiles.*" AND "${CMAKE_BUILD_TYPE}" MATCHES "^$|Debug")
  message(AUTHOR_WARNING "Configuring a Debug build. Eigen performance will be degraded. If you need debug symbols, \
    consider setting CMAKE_BUILD_TYPE to RelWithDebInfo. Otherwise, set to Release to maximize performance.")
endif()
