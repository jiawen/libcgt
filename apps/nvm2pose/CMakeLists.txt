cmake_minimum_required( VERSION 3.6 )
project( nvm2pose )
set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )

# libcgt
find_package( libcgt REQUIRED )

# gflags
if( EXISTS "../../third_party/gflags/CMakeLists.txt" )
    add_subdirectory( "../../third_party/gflags"
	    "${PROJECT_SOURCE_DIR}/third_party/gflags" )
else()
	find_package( gflags REQUIRED )
endif()

# pystring
set( HEADERS "../../third_party/pystring/pystring.h" )
set( SOURCES "../../third_party/pystring/pystring.cpp" )

# main
list( APPEND SOURCES src/main.cpp )

add_executable( nvm2pose ${HEADERS} ${SOURCES} )
set_property( TARGET nvm2pose PROPERTY CXX_STANDARD 11 )
target_include_directories( nvm2pose
    PRIVATE ../../
)
target_link_libraries( nvm2pose
    gflags cgt_core cgt_camera_wrappers
)
