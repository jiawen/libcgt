cmake_minimum_required( VERSION 3.4 )
project( picture_taker )

# libcgt
set( LIBCGT_DIR ${PROJECT_SOURCE_DIR}/../../.. )
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

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core REQUIRED )
find_package( Qt5Widgets REQUIRED )

set( PICTURE_TAKER_SRCS
	src/PictureTaker.cpp
	src/PictureTakerViewfinder.h
	src/PictureTakerViewfinder.cpp
)

set( DEPTH_AVERAGER_SRCS
	src/DepthAverager.cpp
	src/DepthAveragerViewfinder.h
	src/DepthAveragerViewfinder.cpp
)

add_executable( picture_taker ${PICTURE_TAKER_SRCS} )
set_property( TARGET picture_taker PROPERTY CXX_STANDARD 11 )

add_executable( depth_averager ${DEPTH_AVERAGER_SRCS} )
set_property( TARGET depth_averager PROPERTY CXX_STANDARD 11 )

target_link_libraries( picture_taker
    gflags
    Qt5::Core Qt5::Widgets
    debug cgt_cored
    debug cgt_camera_wrappersd
    debug cgt_qt_interopd
    optimized cgt_core
    optimized cgt_camera_wrappers
    optimized cgt_qt_interop
)

target_link_libraries( depth_averager
    gflags
    Qt5::Core Qt5::Widgets
    debug cgt_cored
    debug cgt_camera_wrappersd
    debug cgt_qt_interopd
    optimized cgt_core
    optimized cgt_camera_wrappers
    optimized cgt_qt_interop
)