cmake_minimum_required( VERSION 3.4 )
project( cgt_camera_wrappers )

# TODO: make BUILD_KINECT and BUILD_OPENNI2 options.

set( CAMERA_WRAPPER_HEADERS
    RGBDStream.h PixelFormat.h PoseStream.h StreamConfig.h )
set( CAMERA_WRAPPER_SOURCES
    RGBDStream.cpp PixelFormat.cpp PoseStream.cpp )
set( LIBRARY_DEPENDENCIES cgt_core )

# Kinect v1.x SDK.
if( WIN32 )
    set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Kinect1x )
    find_package( Kinect1x )
    if( Kinect1x_FOUND )
        option( WITH_KINECT1X "Build support for Kinect SDK 1.x." ON )
    endif()

    if( WITH_KINECT1X )
	    set( Kinect1x_DEPENDENCY
            "find_dependency( Kinect1x )" PARENT_SCOPE
        )
        set( Kinect1x_HEADERS
            Kinect1x/KinectCamera.h
            Kinect1x/KinectCameraImpl.h
            Kinect1x/KinectUtils.h )
        set( Kinect1x_SOURCES
            Kinect1x/KinectCamera.cpp
            Kinect1x/KinectCameraImpl.cpp
            Kinect1x/KinectUtils.cpp )

        set( CAMERA_WRAPPER_HEADERS
            ${CAMERA_WRAPPER_HEADERS} ${Kinectx_HEADERS} )
        set( CAMERA_WRAPPER_SOURCES
            ${CAMERA_WRAPPER_SOURCES} ${Kinect1x_SOURCES} )
        set( LIBRARY_DEPENDENCIES
            ${LIBRARY_DEPENDENCIES} Kinect1x::Kinect1x )
        # TODO: how do we add these into the "additional dependencies" line for the
        # static library project.
        # Apparently, amstrmid.lib and sapi.lib are not used.
        # set( MS_SPEECH_LIBRARIES strmiids.lib dmoguids.lib msdmo.lib avrt.lib )
    else()
        message( "Building without Kinect 1.x support." )
    endif()
endif()

## OpenNI2 SDK.

# So we can find FindOpenNI2.cmake.
set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/OpenNI2 )
find_package( OpenNI2 )
if( OpenNI2_FOUND )
    option( WITH_OPENNI2 "Build support for OpenNI2 SDK." ON )
endif()
if( WITH_OPENNI2 )
    set( OpenNI2_DEPENDENCY
        "find_dependency( OpenNI2 )" PARENT_SCOPE
    )
    file( GLOB_RECURSE OPENNI2_HEADERS OpenNI2/*.h OpenNI2*.inl )
    file( GLOB_RECURSE OPENNI2_SOURCES OpenNI2/*.cpp )
	    
	set( CAMERA_WRAPPER_HEADERS
        ${CAMERA_WRAPPER_HEADERS} ${OPENNI2_HEADERS} )
	set( CAMERA_WRAPPER_SOURCES
        ${CAMERA_WRAPPER_SOURCES} ${OPENNI2_SOURCES} )
	set( LIBRARY_DEPENDENCIES
        ${LIBRARY_DEPENDENCIES} ${OPENNI2_LIBRARY} )
else()
    message( "Building without OpenNI2 libraries." )
endif()

add_library( cgt_camera_wrappers SHARED
    ${CAMERA_WRAPPER_HEADERS} ${CAMERA_WRAPPER_SOURCES} )
target_include_directories( cgt_camera_wrappers
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
		$<INSTALL_INTERFACE:include>
)
target_link_libraries( cgt_camera_wrappers ${LIBRARY_DEPENDENCIES} )

install( TARGETS cgt_camera_wrappers
	EXPORT libcgt-targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)

install( DIRECTORY ./ DESTINATION include/libcgt/camera_wrappers
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
if( Kinect1x_FOUND )
	# TODO: don't export KinectCameraImpl.h.
    install( DIRECTORY Kinect1x/
        DESTINATION include/libcgt/camera_wrappers/Kinect1x
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
	
	# TODO: investigate if this is the right thing to do to export our custom
	# FindKinect1x.cmake file.
	#install( FILES Kinect1x/FindKinect1x.cmake
    #    DESTINATION lib/cmake
    #)
endif()
if( OpenNI2_FOUND )
    # TODO: don't export OpenNI2CameraImpl.h.
    install( DIRECTORY OpenNI2/
        DESTINATION include/libcgt/camera_wrappers/OpenNI2
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
    )
endif()
