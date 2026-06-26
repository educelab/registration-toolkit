## Google Test ##
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        EXCLUDE_FROM_ALL
)
set(INSTALL_GTEST OFF CACHE BOOL OFF FORCE)
FetchContent_MakeAvailable(googletest)

## Build the tests ##
set(tests
    src/TestITKOCVBridge.cpp
    src/TestUVMapIO.cpp
    src/TestLandmarkIO.cpp
    src/TestReorderUnorganizedTexture.cpp
)

foreach(src ${tests})
    get_filename_component(filename ${src} NAME_WE)
    set(testname rt_${filename})
    add_executable(${testname} ${src})
    target_link_libraries(${testname}
        rt::core
        opencv_imgproc
        gtest_main
        gmock_main
    )
    add_test(
        NAME ${testname}
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
        COMMAND ${testname}
    )
endforeach()

## Tests that exercise render graph nodes ##
set(graph_tests
    src/TestImageOps.cpp
)

foreach(src ${graph_tests})
    get_filename_component(filename ${src} NAME_WE)
    set(testname rt_${filename})
    add_executable(${testname} ${src})
    target_link_libraries(${testname}
        rt::core
        rt::graph
        gtest_main
        gmock_main
    )
    add_test(
        NAME ${testname}
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
        COMMAND ${testname}
    )
endforeach()