cmake_minimum_required( VERSION 3.4 )
project( picture_taker )

# libcgt
find_package( libcgt REQUIRED )

# gflags
if( EXISTS "../../../third_party/gflags/CMakeLists.txt" )
    add_subdirectory( "../../../third_party/gflags"
        "${PROJECT_SOURCE_DIR}/third_party/gflags"
    )
else()
    find_package( gflags REQUIRED )
endif()

# pystring
set( PYSTRING_HEADERS "../../../third_party/pystring/pystring.h" )
set( PYSTRING_SOURCES "../../../third_party/pystring/pystring.cpp" )

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core REQUIRED )
find_package( Qt5Widgets REQUIRED )

set( PICTURE_TAKER_SRCS
	${PYSTRING_HEADERS} ${PYSTRING_SOURCES}
	src/PictureTaker.cpp
	src/PictureTakerViewfinder.h
	src/PictureTakerViewfinder.cpp
)

set( DEPTH_AVERAGER_SRCS
	${PYSTRING_HEADERS} ${PYSTRING_SOURCES}
	src/DepthAverager.cpp
	src/DepthAveragerViewfinder.h
	src/DepthAveragerViewfinder.cpp
)

add_executable( picture_taker ${PICTURE_TAKER_SRCS} )
set_property( TARGET picture_taker PROPERTY CXX_STANDARD 11 )
target_include_directories( picture_taker
    PRIVATE ../../..
)
target_link_libraries( picture_taker
    gflags
    Qt5::Core Qt5::Widgets
    gflags cgt_core cgt_camera_wrappers cgt_qt_interop
)

add_executable( depth_averager ${DEPTH_AVERAGER_SRCS} )
set_property( TARGET depth_averager PROPERTY CXX_STANDARD 11 )
target_include_directories( depth_averager
    PRIVATE ../../..
)
target_link_libraries( depth_averager
    gflags
    Qt5::Core Qt5::Widgets
    gflags cgt_core cgt_camera_wrappers cgt_qt_interop
)