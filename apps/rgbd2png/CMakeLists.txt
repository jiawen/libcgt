cmake_minimum_required( VERSION 3.6 )
project( rgbd2png CXX )
set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )

# libcgt
set( LIBCGT_DIR ${PROJECT_SOURCE_DIR}/../.. )
include_directories( ${LIBCGT_DIR}/build/lib/include )
include_directories( ${LIBCGT_DIR}/build/lib/include/core )
link_directories( ${LIBCGT_DIR}/build/lib/lib )

# gflags
if( EXISTS "${LIBCGT_DIR}/third_party/gflags/CMakeLists.txt" )
    add_subdirectory( "${LIBCGT_DIR}/third_party/gflags"
        "${PROJECT_SOURCE_DIR}/third_party/gflags" )
else()
    find_package(gflags REQUIRED)
endif()

# pystring
include_directories( ${LIBCGT_DIR} )
set( HEADERS "${LIBCGT_DIR}/third_party/pystring/pystring.h" )
set( SOURCES "${LIBCGT_DIR}/third_party/pystring/pystring.cpp" )

# main
list( APPEND SOURCES src/main.cpp )

add_executable( rgbd2png ${HEADERS} ${SOURCES} )
set_property( TARGET rgbd2png PROPERTY CXX_STANDARD 11 )

# TODO: once libcgt is a proper target, then it's easy:
# set_target_properties(cgt_core PROPERTIES DEBUG_POSTFIX "d")
# target_link_libraries( rgb2png cgt_core )

if( WIN32 )
    target_link_libraries( rgbd2png
        debug cgt_cored
        debug cgt_camera_wrappersd
        optimized cgt_core
        optimized cgt_camera_wrappers )
else()
    target_link_libraries( rgbd2png
        cgt_core cgt_camera_wrappers )
endif()
