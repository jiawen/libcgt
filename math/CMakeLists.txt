cmake_minimum_required( VERSION 3.4 )
project( libcgt_math )

# Debug mode
set( CMAKE_DEBUG_POSTFIX d )

# Release mode
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_SECURE_SCL=0" )

# Qt 4
find_package( Qt4 REQUIRED )

# Add include to the set of compiler include paths
include_directories( ../core/include )
include_directories( include )
include_directories( $ENV{ICPP_COMPILER12}mkl/include )

include_directories( $ENV{SUITESPARSED}/UFConfig $ENV{SUITESPARSED}/CHOLMOD/Include $ENV{SUITESPARSED}/SPQR/Include )

file( GLOB_RECURSE HEADERS include/*.h include/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp  )

include( ${QT_USE_FILE} )
add_definitions( ${QT_DEFINITIONS} )
add_definitions( -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DUNICODE -D_UNICODE )

if( WIN32 )
    option( WIN32_USE_MP "Set to ON to use multithreaded builds." ON )
    if( WIN32_USE_MP )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
    endif( WIN32_USE_MP )
endif( WIN32 )

add_library( libcgt_math ${HEADERS} ${SOURCES} ${HEADERS_MOC} )
target_link_libraries( libcgt_math ${QT_LIBRARIES} )
