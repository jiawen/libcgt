cmake_minimum_required( VERSION 3.4 )
project( rgbd2png )

# libcgt
set( LIBCGT_DIR ../.. )
include_directories( ${LIBCGT_DIR}/build/include )
include_directories( ${LIBCGT_DIR}/build/include/core )
link_directories( ${LIBCGT_DIR}/build/lib )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp )

add_executable( rgbd2png ${HEADERS} ${SOURCES} )
set_property( TARGET rgbd2png PROPERTY CXX_STANDARD 11 )

target_link_libraries( rgbd2png
    debug cgt_cored
    debug cgt_camera_wrappersd
    optimized cgt_core
    optimized cgt_camera_wrappers
)
