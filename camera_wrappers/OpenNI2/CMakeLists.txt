cmake_minimum_required( VERSION 3.4 )
project( cgt_openni2 )

# OpenNI2 SDK.
set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} )
find_package( OpenNI2 REQUIRED )

include_directories( ${OPENNI2_INCLUDE_DIRS} )

# Add libcgt/core to the set of compiler include paths
include_directories( ../core/src )

file( GLOB_RECURSE HEADERS *.h *.inl )
file( GLOB_RECURSE SOURCES *.cpp  )

add_library( cgt_openni2 SHARED ${HEADERS} ${SOURCES} )
target_link_libraries( cgt_openni2 cgt_core ${OPENNI2_LIBRARY} )

install( TARGETS cgt_openni2 DESTINATION lib )
install( DIRECTORY ./ DESTINATION include/OpenNI2
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
