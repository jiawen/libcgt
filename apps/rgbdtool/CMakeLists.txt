cmake_minimum_required( VERSION 3.6 )
project( rgbdtool CXX )
set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )

# gflags
add_subdirectory( "../../third_party/gflags"
    "${PROJECT_SOURCE_DIR}/third_party/gflags"
)

# libcgt
find_package( libcgt REQUIRED )

# OpenCV
find_package( OpenCV REQUIRED )

# pystring
set( HEADERS "../../third_party/pystring/pystring.h" )
set( SOURCES "../../third_party/pystring/pystring.cpp" )

# main
list( APPEND SOURCES src/main.cpp )

add_executable( rgbdtool ${HEADERS} ${SOURCES} )
set_property( TARGET rgbdtool PROPERTY CXX_STANDARD 11 )
target_include_directories( rgbdtool
    PRIVATE ../../
)
target_link_libraries( rgbdtool
    gflags
	${OpenCV_LIBS}
	cgt_core
	cgt_camera_wrappers
	cgt_opencv_interop
)
