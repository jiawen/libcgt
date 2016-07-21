cmake_minimum_required( VERSION 3.4 )
project( cgt_kinect1x )

# Kinect v1.x SDK.
set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} )
find_package( Kinect1x REQUIRED )

# Add include to the set of compiler include paths
include_directories( ../core/src )

#file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
#file( GLOB_RECURSE SOURCES src/*.cpp  )

set( HEADERS src/KinectCamera.h src/KinectUtils.h src/RGBDStream.h )
set( SOURCES src/KinectCamera.cpp src/KinectUtils.cpp src/RGBDStream.cpp )

# TODO: how do we add these into the "additional dependencies" line for the
# static library project.
# Apparently, amstrmid.lib and sapi.lib are not used.
# set( MS_SPEECH_LIBRARIES strmiids.lib dmoguids.lib msdmo.lib avrt.lib )

add_library( cgt_kinect1x SHARED ${HEADERS} ${SOURCES} )
target_link_libraries( cgt_kinect1x cgt_core Kinect1x::Kinect1x )

install( TARGETS cgt_kinect1x DESTINATION lib )
install( DIRECTORY src/ DESTINATION include/kinect1x
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
