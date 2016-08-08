cmake_minimum_required( VERSION 3.4 )
project( cgt_camera_wrappers )

# Add libcgt/core to the set of compiler include paths
include_directories( ../core/src )
include_directories( .. )

set( CAMERA_WRAPPER_HEADERS RGBDStream.h PixelFormat.h PoseStream.h StreamConfig.h )
set( CAMERA_WRAPPER_SOURCES RGBDStream.cpp PixelFormat.cpp PoseStream.cpp )

# Kinect v1.x SDK.
set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Kinect1x )
find_package( Kinect1x )
if( KINECT1X_FOUND )
    set( KINECT1X_HEADERS
		Kinect1x/KinectCamera.h
		Kinect1x/KinectCameraImpl.h
		Kinect1x/KinectUtils.h
	)
    set( KINECT1X_SOURCES
		Kinect1x/KinectCamera.cpp
		Kinect1x/KinectCameraImpl.cpp
		Kinect1x/KinectUtils.cpp
	)

    # TODO: how do we add these into the "additional dependencies" line for the
    # static library project.
    # Apparently, amstrmid.lib and sapi.lib are not used.
    # set( MS_SPEECH_LIBRARIES strmiids.lib dmoguids.lib msdmo.lib avrt.lib )
endif()

# OpenNI2 SDK.
set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/OpenNI2 )
find_package( OpenNI2 )
if( OpenNI2_FOUND )
    include_directories( ${OPENNI2_INCLUDE_DIRS} )

    file( GLOB_RECURSE OPENNI2_HEADERS OpenNI2/*.h OpenNI2*.inl )
    file( GLOB_RECURSE OPENNI2_SOURCES OpenNI2/*.cpp )
endif()

# TODO: append list correctly
add_library( cgt_camera_wrappers SHARED
    ${CAMERA_WRAPPER_HEADERS} ${KINECT1X_HEADERS} ${OPENNI2_HEADERS}
    ${CAMERA_WRAPPER_SOURCES} ${KINECT1X_SOURCES} ${OPENNI2_SOURCES}
)
target_link_libraries( cgt_camera_wrappers cgt_core Kinect1x::Kinect1x ${OPENNI2_LIBRARY} )

install( TARGETS cgt_camera_wrappers DESTINATION lib )
install( DIRECTORY ./ DESTINATION include/camera_wrappers
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )

if( Kinect1x_FOUND )
	# TODO: don't export KinectCameraImpl.h.
    install( DIRECTORY Kinect1x/ DESTINATION include/camera_wrappers/Kinect1x
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
endif()

if( OpenNI2_FOUND )
	# TODO: don't export OpenNI2CameraImpl.h.
    install( DIRECTORY OpenNI2/ DESTINATION include/camera_wrappers/OpenNI2
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
endif()
