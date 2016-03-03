cmake_minimum_required( VERSION 3.1 )
project( cgt_kinect1x )

# Kinect v1.x SDK.
set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} )
find_package( Kinect1x REQUIRED )

# Qt 5.
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core )
find_package( Qt5Widgets )

# Add include to the set of compiler include paths
include_directories( ../core/src )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp  )

# TODO: how do we add these into the "additional dependencies" line for the
# static library project.
# Apparently, amstrmid.lib and sapi.lib are not used.
# set( MS_SPEECH_LIBRARIES strmiids.lib dmoguids.lib msdmo.lib avrt.lib )

add_library( cgt_kinect1x ${HEADERS} ${SOURCES} )
set_property( TARGET cgt_kinect1x PROPERTY CXX_STANDARD 11 )
target_link_libraries( cgt_kinect1x cgt_core Kinect1x::Kinect1x Qt5::Core Qt5::Widgets )

install( TARGETS cgt_kinect1x DESTINATION lib )
install( DIRECTORY src/ DESTINATION include/kinect1x
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
