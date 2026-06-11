# Only apps *require* Boost
find_package(Boost 1.70 CONFIG REQUIRED COMPONENTS program_options)

add_executable(rt_apply_transform src/ApplyTransform.cpp)
target_link_libraries(rt_apply_transform
    rt::core
    Boost::program_options
)

add_executable(rt_register src/Register.cpp)
target_link_libraries(rt_register
    rt::core
    rt::graph
    Boost::program_options
    smgl::smgl
)
target_include_directories(rt_register
    PUBLIC
        $<BUILD_INTERFACE:${libcore_SOURCE_DIR}/include>
)

add_executable(rt_reorder_texture src/ReorderTexture.cpp)
target_link_libraries(rt_reorder_texture
    rt::core
    rt::graph
    Boost::program_options
    smgl::smgl
)
target_include_directories(rt_reorder_texture
    PUBLIC
        $<BUILD_INTERFACE:${libcore_SOURCE_DIR}/include>
)

add_executable(rt_raw_tiff_converter src/RawTIFFConverter.cpp)
target_link_libraries(rt_raw_tiff_converter
    rt::core
)

add_executable(rt_generate_landmarks src/GenerateLandmarks.cpp)
target_link_libraries(rt_generate_landmarks
    rt::core
)

add_executable(rt_plot_landmarks src/PlotLandmarks.cpp)
target_link_libraries(rt_plot_landmarks
    rt::core
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    Boost::program_options
)

add_executable(rt_swap_landmarks src/SwapLandmarks.cpp)
target_link_libraries(rt_swap_landmarks
    rt::core
    Boost::program_options
)

add_executable(rt_segment_disegni src/SegmentDisegni.cpp)
target_link_libraries(rt_segment_disegni
    rt::core
    Boost::program_options
)

add_executable(rt_retexture_mesh src/RetextureMesh.cpp)
target_link_libraries(rt_retexture_mesh
    rt::core
    Boost::program_options
)

add_executable(rt_version src/Version.cpp)
target_link_libraries(rt_version rt::core)

add_executable(rt_seam_flattening src/SeamFlattening.cpp)
target_include_directories(rt_seam_flattening
    PRIVATE
        $<BUILD_INTERFACE:${OpenABF_SOURCE_DIR}/single_include>
)
target_link_libraries(rt_seam_flattening
    rt::core
    Boost::program_options
    Eigen3::Eigen
    opencv_core
)


if(VC_FOUND)
    add_executable(rt_dewarp src/TextureDewarp.cpp)
    target_link_libraries(rt_dewarp
        rt::core
        VC::texturing
    )
endif()

install(
    TARGETS
        rt_apply_transform
        rt_register
        rt_reorder_texture
        rt_raw_tiff_converter
        rt_generate_landmarks
        rt_plot_landmarks
        rt_swap_landmarks
        rt_segment_disegni
        rt_retexture_mesh
        rt_version
    RUNTIME DESTINATION bin
    COMPONENT Programs
)
