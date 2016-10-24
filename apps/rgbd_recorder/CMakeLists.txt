cmake_minimum_required( VERSION 3.6 )
project( rgbd_recorder CXX )
set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )

# libcgt
set( LIBCGT_DIR ${PROJECT_SOURCE_DIR}/../.. )
include_directories( ${LIBCGT_DIR}/build/lib/include )
include_directories( ${LIBCGT_DIR}/build/lib/include/core )
include_directories( ${LIBCGT_DIR}/build/lib/include/camera_wrappers )
include_directories( ${LIBCGT_DIR}/build/lib/include/opencv_interop )
link_directories( ${LIBCGT_DIR}/build/lib/lib )

# gflags
if( EXISTS "${LIBCGT_DIR}/third_party/gflags/CMakeLists.txt" )
    add_subdirectory( "${LIBCGT_DIR}/third_party/gflags"
        "${PROJECT_SOURCE_DIR}/third_party/gflags" )
else()
    find_package(gflags REQUIRED)
endif()

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core REQUIRED )
find_package( Qt5Widgets REQUIRED )

# pystring
include_directories( ${LIBCGT_DIR} )
set( HEADERS "${LIBCGT_DIR}/third_party/pystring/pystring.h" )
set( SOURCES "${LIBCGT_DIR}/third_party/pystring/pystring.cpp" )

# main
list( APPEND HEADERS src/Viewfinder.h )
list( APPEND SOURCES src/main.cpp src/Viewfinder.cpp )

add_executable( rgbd_recorder ${HEADERS} ${SOURCES} )
set_property( TARGET rgbd_recorder PROPERTY CXX_STANDARD 11 )

target_link_libraries( rgbd_recorder
	gflags
    Qt5::Core Qt5::Widgets
    debug cgt_cored
    debug cgt_camera_wrappersd
    debug cgt_qt_interopd
    optimized cgt_core
    optimized cgt_camera_wrappers
    optimized cgt_qt_interop
)
