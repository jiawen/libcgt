cmake_minimum_required( VERSION 3.4 )
project( calibrator CXX )
set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT stereo_calibrator )

# libcgt
set( LIBCGT_DIR ${PROJECT_SOURCE_DIR}/../../.. )
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

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( intrinsics_calibrator
	src/common.h src/common.cpp src/intrinsics_calibrator.cpp )
set_property( TARGET intrinsics_calibrator PROPERTY CXX_STANDARD 11 )

add_executable( stereo_calibrator
	src/common.h src/common.cpp src/stereo_calibrator.cpp )
set_property( TARGET stereo_calibrator PROPERTY CXX_STANDARD 11 )

target_link_libraries( intrinsics_calibrator
	gflags
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
	gflags
	${OpenCV_LIBS}
	debug cgt_cored
	debug cgt_camera_wrappersd
	debug cgt_opencv_interopd
	optimized cgt_core
	optimized cgt_camera_wrappers
	optimized cgt_opencv_interop
)