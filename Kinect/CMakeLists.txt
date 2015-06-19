cmake_minimum_required( VERSION 2.8 )
project( libcgt_kinect )

# Debug mode
set( CMAKE_DEBUG_POSTFIX d )

# Release mode
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_SECURE_SCL=0" )

# Qt 4
find_package( Qt4 REQUIRED )

# Add include to the set of compiler include paths
include_directories( ../core/include )
include_directories( include )
include_directories( $ENV{KINECTSDK10_DIR}inc )

file( GLOB_RECURSE HEADERS include/*.h include/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp  )

QT4_WRAP_CPP( HEADERS_MOC include/QKinect.h include/QKinectSpeechThread.h include/QKinectThread.h )

include( ${QT_USE_FILE} )
add_definitions( ${QT_DEFINITIONS} )
add_definitions( -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DUNICODE -D_UNICODE )

if( WIN32 )
    option( WIN32_USE_MP "Set to ON to use multithreaded builds." ON )
    if( WIN32_USE_MP )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
    endif( WIN32_USE_MP )
endif( WIN32 )


add_library(libcgt_kinect ${HEADERS} ${SOURCES} ${HEADERS_MOC})
target_link_libraries(libcgt_kinect ${QT_LIBRARIES})
