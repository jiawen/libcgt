cmake_minimum_required( VERSION 3.6 )
project( rgbd_recorder CXX )
set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )

# libcgt
find_package( libcgt REQUIRED )

# gflags
if( EXISTS "../../third_party/gflags/CMakeLists.txt" )
    add_subdirectory( "../../third_party/gflags"
        "${PROJECT_SOURCE_DIR}/third_party/gflags"
    )
else()
    find_package( gflags REQUIRED )
endif()

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core REQUIRED )
find_package( Qt5Widgets REQUIRED )

# pystring
set( HEADERS "../../third_party/pystring/pystring.h" )
set( SOURCES "../../third_party/pystring/pystring.cpp" )

# main
list( APPEND HEADERS src/Viewfinder.h )
list( APPEND SOURCES src/main.cpp src/Viewfinder.cpp )

add_executable( rgbd_recorder ${HEADERS} ${SOURCES} )
set_property( TARGET rgbd_recorder PROPERTY CXX_STANDARD 11 )
target_include_directories( rgbd_recorder
    PRIVATE ../../
)
target_link_libraries( rgbd_recorder
	gflags
    Qt5::Core Qt5::Widgets
	cgt_core
    cgt_camera_wrappers
    cgt_qt_interop
)
