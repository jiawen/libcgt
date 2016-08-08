cmake_minimum_required( VERSION 3.4 )
project( picture_taker )

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core )
find_package( Qt5Widgets )

# libcgt
set( LIBCGT_DIR ../../.. )
include_directories( ${LIBCGT_DIR}/build/include )
include_directories( ${LIBCGT_DIR}/build/include/core )
link_directories( ${LIBCGT_DIR}/build/lib )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
file( GLOB_RECURSE SOURCES_CPP src/*.cpp )

add_executable( picture_taker ${HEADERS} ${SOURCES_CPP} )
set_property( TARGET picture_taker PROPERTY CXX_STANDARD 11 )

target_link_libraries( picture_taker
    Qt5::Core Qt5::Widgets
    ${OpenCV_LIBS}
    debug cgt_cored
    debug cgt_camera_wrappersd
    debug cgt_opencv_interopd
    debug cgt_qt_interopd
    optimized cgt_core
    optimized cgt_camera_wrappers
    optimized cgt_opencv_interop
    optimized cgt_qt_interop
)
