cmake_minimum_required( VERSION 3.1 )
project( cgt_kinect1 )

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core )
find_package( Qt5Widgets )

# Add include to the set of compiler include paths
include_directories( ../core/include )
include_directories( include )
include_directories( $ENV{KINECTSDK10_DIR}inc )

file( GLOB_RECURSE HEADERS include/*.h include/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp  )

# TODO: how do we add these into the "additional dependencies" line for the
# static library project.
# Apparently, amstrmid.lib and sapi.lib are not used.
# set( MS_SPEECH_LIBRARIES strmiids.lib dmoguids.lib msdmo.lib avrt.lib )

add_library( cgt_kinect1 ${HEADERS} ${SOURCES} )
#target_link_libraries( cgt_kinect1 cgt_core Qt5::Core Qt5::Widgets )
target_link_libraries( cgt_kinect1 strmiids.lib Qt5::Core Qt5::Widgets )
