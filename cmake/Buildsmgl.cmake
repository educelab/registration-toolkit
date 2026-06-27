FetchContent_Declare(
    smgl
    GIT_REPOSITORY https://gitlab.com/educelab/smgl.git
    GIT_TAG d74d76d121a18afab9b12dd9dc3d643f4e620ff1
    EXCLUDE_FROM_ALL
)
set(SMGL_BUILD_JSON ${RT_BUILD_JSON} CACHE INTERNAL "")
set(SMGL_USE_BOOSTFS OFF CACHE INTERNAL "")
set(SMGL_BUILD_TESTS OFF CACHE INTERNAL "")
set(SMGL_BUILD_DOCS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(smgl)
