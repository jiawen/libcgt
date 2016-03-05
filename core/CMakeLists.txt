cmake_minimum_required( VERSION 3.1 )
project( cgt_core )

# Add include to the set of compiler include paths
include_directories( src )
include_directories( third_party )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl third_party/*.h )
file( GLOB_RECURSE SOURCES src/*.cpp third_party/*.cpp )

add_library( cgt_core SHARED ${HEADERS} ${SOURCES} )
set_property( TARGET cgt_core PROPERTY CXX_STANDARD 11 )

install( TARGETS cgt_core DESTINATION lib )
install( DIRECTORY src/ DESTINATION include/core
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
install( DIRECTORY third_party/ DESTINATION include/third_party
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
