########
# Core #
########

### OpenCV ###
find_package(OpenCV 4 QUIET REQUIRED)

### ITK ###
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND  CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 15)
    find_package(ITK 5.4 REQUIRED)
else()
    find_package(ITK REQUIRED)
endif()
set(ITKIOTransformLibs
    ITKIOTransformBase
    ITKIOTransformHDF5
    ITKIOTransformInsightLegacy
    ITKIOTransformMatlab
)

### VTK ###
find_package(VTK REQUIRED)
if(VTK_VERSION_MAJOR VERSION_LESS 8.9)
  include(${VTK_USE_FILE})
endif()

### libtiff ###
find_package(TIFF REQUIRED)

### smgl ###
find_package(nlohmann_json 3.9.1 QUIET)
if(nlohmann_json_FOUND)
    option(RT_BUILD_JSON "Build JSON library from source" off)
else()
    option(RT_BUILD_JSON "Build JSON library from source" on)
endif()
include(Buildsmgl)

### bvh ###
include(Buildbvh)

### libcore ###
# libcore must be installed on the system so that downstream projects linking
# rt::core can resolve the PUBLIC educelab::core dependency (a FetchContent
# build is not installed). See cmake/Config.cmake.in.
find_package(EduceLabCore 0.3.0 CONFIG REQUIRED)

### OpenABF ###
include(BuildOpenABF)

### spdlog ###
find_package(spdlog 1.9.0 CONFIG REQUIRED)

############
# Optional #
############
option(RT_USE_VOLCART "Use the Volume Cartographer library" off)
if(RT_USE_VOLCART)
    find_package(VC 2.13 CONFIG)
endif()
