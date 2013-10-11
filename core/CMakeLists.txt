cmake_minimum_required( VERSION 2.8 )
project( libcgt_core )

# Debug mode
set( CMAKE_DEBUG_POSTFIX d )

if( UNIX )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
endif( UNIX )

# Qt 4
find_package( Qt4 REQUIRED )

# Add include to the set of compiler include paths
include_directories( include )

file( GLOB_RECURSE HEADERS include/*.h include/*.inl external/*.h )
file( GLOB_RECURSE SOURCES src/*.cpp external/*.cpp )

QT4_WRAP_CPP( HEADERS_MOC include/time/QGameLoop.h )

include( ${QT_USE_FILE} )
add_definitions( ${QT_DEFINITIONS} )

if( WIN32 )
    # Release mode
    set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_SECURE_SCL=0" )
    add_definitions( -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DUNICODE -D_UNICODE )

    option( WIN32_USE_MP "Set to ON to use multithreaded builds." ON )
    if( WIN32_USE_MP )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
    endif( WIN32_USE_MP )
endif( WIN32 )

add_library( libcgt_core ${HEADERS} ${SOURCES} ${HEADERS_MOC} )
target_link_libraries( libcgt_core ${QT_LIBRARIES} )
