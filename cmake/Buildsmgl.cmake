find_package(nlohmann_json 3.9.1)
if(nlohmann_json_FOUND)
    option(RT_BUILD_JSON "Build in-source JSON library" off)
else()
    option(RT_BUILD_JSON "Build in-source JSON library" on)
endif()
FetchContent_Declare(
    smgl
    GIT_REPOSITORY https://gitlab.com/educelab/smgl.git
    GIT_TAG v0.10.1
    EXCLUDE_FROM_ALL
)
set(SMGL_BUILD_JSON ${RT_BUILD_JSON} CACHE INTERNAL "")
set(SMGL_USE_BOOSTFS ${RT_USE_BOOSTFS} CACHE INTERNAL "")
set(SMGL_BUILD_TESTS OFF CACHE INTERNAL "")
set(SMGL_BUILD_DOCS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(smgl)
