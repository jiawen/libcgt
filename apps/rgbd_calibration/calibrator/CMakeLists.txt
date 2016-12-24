cmake_minimum_required( VERSION 3.6 )
project( calibrator CXX )
set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT stereo_calibrator )

# libcgt
find_package( libcgt REQUIRED )

# gflags
if( EXISTS "../../../third_party/gflags/CMakeLists.txt" )
    add_subdirectory( "../../../third_party/gflags"
        "${PROJECT_SOURCE_DIR}/third_party/gflags" )
else()
    find_package( gflags REQUIRED )
endif()

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# pystring
set( PYSTRING_HEADERS "../../../third_party/pystring/pystring.h" )
set( PYSTRING_SOURCES "../../../third_party/pystring/pystring.cpp" )

add_executable( intrinsics_calibrator
    ${PYSTRING_HEADERS} ${PYSTRING_SOURCES}
    src/common.h src/common.cpp src/intrinsics_calibrator.cpp )
set_property( TARGET intrinsics_calibrator PROPERTY CXX_STANDARD 11 )
target_include_directories( intrinsics_calibrator
    PRIVATE ../../..
)
target_link_libraries( intrinsics_calibrator
    gflags
    ${OpenCV_LIBS}
    cgt_core cgt_camera_wrappers cgt_opencv_interop
)

add_executable( stereo_calibrator
    ${PYSTRING_HEADERS} ${PYSTRING_SOURCES}
    src/common.h src/common.cpp src/stereo_calibrator.cpp )
set_property( TARGET stereo_calibrator PROPERTY CXX_STANDARD 11 )
target_include_directories( stereo_calibrator
    PRIVATE ../../..
)
target_link_libraries( stereo_calibrator
    gflags
    ${OpenCV_LIBS}
    cgt_core cgt_camera_wrappers cgt_opencv_interop
)