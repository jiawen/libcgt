cmake_minimum_required( VERSION 3.4 )
project( calibrator )

# gflags
find_package(gflags REQUIRED)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core )
find_package( Qt5Widgets )

# libcgt
set( LIBCGT_DIR ../../.. )
include_directories( ${LIBCGT_DIR}/build/include )
include_directories( ${LIBCGT_DIR}/build/include/core )
include_directories( ${LIBCGT_DIR}/build/include/camera_wrappers )
include_directories( ${LIBCGT_DIR}/build/include/opencv_interop )
include_directories( ${LIBCGT_DIR}/build/include/qt_interop )
link_directories( ${LIBCGT_DIR}/build/lib )

#file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
#file( GLOB_RECURSE SOURCES_CPP src/*.cpp )

add_executable( intrinsics_calibrator
	src/common.h src/common.cpp src/intrinsics_calibrator.cpp )
set_property( TARGET intrinsics_calibrator PROPERTY CXX_STANDARD 11 )

add_executable( stereo_calibrator
	src/common.h src/common.cpp src/stereo_calibrator.cpp )
set_property( TARGET stereo_calibrator PROPERTY CXX_STANDARD 11 )

target_link_libraries( intrinsics_calibrator
	gflags_shared
	${OpenCV_LIBS}
	debug cgt_cored
	debug cgt_camera_wrappersd
	debug cgt_opencv_interopd
	optimized cgt_core
	optimized cgt_camera_wrappers
	optimized cgt_opencv_interop
)

# TODO: why is the name gflags_shared and not gflags?
target_link_libraries( stereo_calibrator
	gflags_shared
	${OpenCV_LIBS}
	debug cgt_cored
	debug cgt_camera_wrappersd
	debug cgt_opencv_interopd
	optimized cgt_core
	optimized cgt_camera_wrappers
	optimized cgt_opencv_interop
)