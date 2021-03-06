cmake_minimum_required( VERSION 3.6 )
project( imageseq2rgbd )
set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )

# libcgt
find_package( libcgt REQUIRED )

# gflags
add_subdirectory( "../../third_party/gflags"
    "${PROJECT_SOURCE_DIR}/third_party/gflags"
)

# pystring
set( HEADERS "../../third_party/pystring/pystring.h" )
set( SOURCES "../../third_party/pystring/pystring.cpp" )

# main
list( APPEND SOURCES src/main.cpp )

add_executable( imageseq2rgbd ${HEADERS} ${SOURCES} )
set_property( TARGET imageseq2rgbd PROPERTY CXX_STANDARD 11 )
target_include_directories( imageseq2rgbd
    PRIVATE ../../
)
target_link_libraries( imageseq2rgbd
    gflags cgt_core cgt_camera_wrappers
)
