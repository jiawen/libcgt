cmake_minimum_required( VERSION 3.6 )
project( imageseq2rgbd )
# Broken??
# vs_startup_project( imageseq2rgbd )

# gflags
find_package(gflags REQUIRED)

# libcgt
set( LIBCGT_DIR ../.. )
include_directories( ${LIBCGT_DIR}/build/lib/include )
include_directories( ${LIBCGT_DIR}/build/lib/include/core )
link_directories( ${LIBCGT_DIR}/build/lib/lib )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp )

add_executable( imageseq2rgbd ${HEADERS} ${SOURCES} )
set_property( TARGET imageseq2rgbd PROPERTY CXX_STANDARD 11 )

# TODO: once libcgt is a proper target, then it's easy:
# set_target_properties( cgt_core PROPERTIES DEBUG_POSTFIX "d" )
# target_link_libraries( imageseq2rgbd cgt_core )

if( WIN32 )
    target_link_libraries( imageseq2rgbd
		gflags_shared
        debug cgt_cored
        debug cgt_camera_wrappersd
        optimized cgt_core
        optimized cgt_camera_wrappers )
else()
    target_link_libraries( imageseq2rgbd
        gflags_shared cgt_core cgt_camera_wrappers )
endif()
