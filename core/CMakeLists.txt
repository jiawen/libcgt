cmake_minimum_required( VERSION 3.4 )
project( cgt_core )

# Add include to the set of compiler include paths
include_directories( .. )
include_directories( src )
include_directories( third_party )

file( GLOB_RECURSE HEADERS
	src/*.h src/*.inl
	../third_party/lodepng/lodepng.h
	../third_party/pystring/pystring.h )
file( GLOB_RECURSE SOURCES
	src/*.cpp
	../third_party/lodepng/lodepng.cpp
	../third_party/pystring/pystring.cpp )

add_library( cgt_core SHARED ${HEADERS} ${SOURCES} )

install( TARGETS cgt_core DESTINATION lib EXPORT cgt_core-targets )
install( EXPORT cgt_core-targets DESTINATION lib/cmake )
install( DIRECTORY src/ DESTINATION include/core
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
install( DIRECTORY third_party/ DESTINATION include/third_party
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
