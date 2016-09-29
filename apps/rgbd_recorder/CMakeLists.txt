cmake_minimum_required( VERSION 3.4 )
project( rgbd_recorder )

# gflags
find_package(gflags REQUIRED)

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core REQUIRED )
find_package( Qt5Widgets REQUIRED )

# libcgt
set( LIBCGT_DIR ../.. )
include_directories( ${LIBCGT_DIR}/build/include )
include_directories( ${LIBCGT_DIR}/build/include/core )
link_directories( ${LIBCGT_DIR}/build/lib )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
file( GLOB_RECURSE SOURCES_CPP src/*.cpp )

add_executable( rgbd_recorder ${HEADERS} ${SOURCES_CPP} )
set_property( TARGET rgbd_recorder PROPERTY CXX_STANDARD 11 )

target_link_libraries( rgbd_recorder
	gflags_shared
    Qt5::Core Qt5::Widgets
    debug cgt_cored
    debug cgt_camera_wrappersd
    debug cgt_qt_interopd
    optimized cgt_core
    optimized cgt_camera_wrappers
    optimized cgt_qt_interop
)
